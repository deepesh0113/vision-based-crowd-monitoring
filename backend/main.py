from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # for local dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def convert_timestamp_ns_to_seconds(ts_str):
    """
    Supports:
    1) 'MM:SS:MS'  -> '00:00:400'
    2) raw integer nanoseconds -> 400000000 (0.4 sec)
    """
    try:
        s = str(ts_str).strip()

        # Case 1: "MM:SS:MS"
        if ":" in s:
            parts = s.split(":")
            # "MM:SS:MS"
            if len(parts) == 3:
                minutes = int(parts[0])
                seconds = int(parts[1])
                millis = int(parts[2])
                total_seconds = minutes * 60 + seconds + millis / 1000.0
                return total_seconds
            # "MM:SS"
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
            # anything else: fail
            return None

        # Case 2: plain number -> treat as nanoseconds
        # e.g. 400000000 -> 0.4 sec
        val = float(s)
        # if it's very small (< 1e6) maybe it's already seconds or ms,
        # but since column is timestamp_ns, we assume nanoseconds:
        return val / 1e9

    except Exception as e:
        print("Timestamp parse error for:", ts_str, "->", e)
        return None


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    CSV must contain: date, timestamp_ns, frame_index, count

    Returns JSON:
      - records      -> [{ date, timestamp, count }]
      - time_series  -> [{ date, time_sec, count }]
      - per_second   -> [{ date, second, avg_count }]
      - frame_series -> [{ date, frame_index, count }]
      - summary      -> stats
    """
    try:
        raw = await file.read()

        # --- Read CSV into DataFrame ---
        try:
            df = pd.read_csv(BytesIO(raw))
        except Exception as e:
            print("Pandas read_csv error:", e)
            return {
                "status": "error",
                "message": f"Failed to read CSV: {e}",
                "records": [],
            }

        print("CSV columns:", list(df.columns))

        required_cols = {"date", "timestamp_ns", "frame_index", "count"}
        if not required_cols.issubset(df.columns):
            return {
                "status": "error",
                "message": (
                    "CSV must contain columns: "
                    "date, timestamp_ns, frame_index, count"
                ),
                "records": [],
            }

        # Normalize types
        df["date"] = df["date"].astype(str)

        # Convert timestamp_ns -> seconds using robust parser
        df["time_sec"] = df["timestamp_ns"].apply(convert_timestamp_ns_to_seconds)

        # Check if *all* conversions failed
        if df["time_sec"].isna().all():
            return {
                "status": "error",
                "message": (
                    "No valid rows after timestamp conversion. "
                    "Ensure 'timestamp_ns' is either 'MM:SS:MS' (e.g. 00:00:400) "
                    "or a numeric nanoseconds value (e.g. 400000000)."
                ),
                "records": [],
            }

        # Drop only rows that failed; keep others
        df = df.dropna(subset=["time_sec"])

        # ---------- 1) time series: (date, time_sec, count) ----------
        time_series_df = df[["date", "time_sec", "count"]].copy()
        time_series = time_series_df.to_dict(orient="records")

        # ---------- 2) per-second average, grouped by (date, sec_bucket) ----------
        df["sec_bucket"] = df["time_sec"].astype(int)
        per_second_df = (
            df.groupby(["date", "sec_bucket"])["count"]
            .mean()
            .reset_index()
            .rename(columns={"sec_bucket": "second", "count": "avg_count"})
        )
        per_second = per_second_df.to_dict(orient="records")

        # ---------- 3) frame_index vs count, with date ----------
        frame_series_df = df[["date", "frame_index", "count"]].copy()
        frame_series = frame_series_df.to_dict(orient="records")

        # ---------- Summary stats over all rows ----------
        summary = {
            "min_count": float(df["count"].min()),
            "max_count": float(df["count"].max()),
            "mean_count": float(df["count"].mean()),
            "num_points": int(len(df)),
        }

        # ---------- records: what Dashboard.js uses for main time-series ----------
        # Each record: { date, timestamp, count }
        records = [
            {
                "date": row["date"],
                "timestamp": float(row["time_sec"]),
                "count": float(row["count"]),
            }
            for _, row in time_series_df.iterrows()
        ]

        response = {
            "status": "success",
            "records": records,
            "time_series": time_series,
            "per_second": per_second,
            "frame_series": frame_series,
            "summary": summary,
        }

        print("Processed rows:", len(records))
        return response

    except Exception as e:
        # This catches any unexpected errors and still returns JSON
        print("Unexpected server error:", e)
        return {
            "status": "error",
            "message": f"Failed to process CSV: {e}",
            "records": [],
        }
