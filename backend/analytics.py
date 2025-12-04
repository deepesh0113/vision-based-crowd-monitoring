# main.py
import os
import time
import threading
import csv
from datetime import datetime
from io import StringIO
import tempfile




import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# ---------------- SETTINGS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crowd_counting.pth")

USE_GPU = True
USE_FP16_IF_CUDA = True
FRAME_RESIZE = (512, 384)
SKIP_FRAMES = 10
SMOOTH_WINDOW = 3
FLUSH_INTERVAL = 2.0
PRINT_EVERY = 50
UPSAMPLE_DMAP = False
SCALE_BY_AREA = True
# ------------------------------------------

device = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")
print("Device:", device)

# ---------- Camera hinderance helpers ----------
_hinder_counters = {
    "frozen": 0,
    "covered": 0,
    "low_contrast": 0,
    "obstructed": 0,
}
FROZEN_DIFF_THRESH = 2.0
FROZEN_FRAMES_THRESH = 5
COVERED_MEAN_THRESH = 15
COVERED_FRAMES_THRESH = 3
LOW_CONTRAST_STD_THRESH = 10.0
LOW_CONTRAST_FRAMES_THRESH = 5
OBSTRUCT_EDGE_DENSITY_THRESH = 0.005
OBSTRUCT_FRAMES_THRESH = 5
DARK_PCT_THRESH = 0.50

_prev_gray_for_hinder = None


def check_camera_hinder(frame_bgr):
    global _prev_gray_for_hinder, _hinder_counters
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())

    if _prev_gray_for_hinder is None:
        diff_mean = 255.0
    else:
        diff = cv2.absdiff(gray, _prev_gray_for_hinder)
        diff_mean = float(diff.mean())

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))
    dark_pct = float(np.mean(gray < 30))

    if diff_mean < FROZEN_DIFF_THRESH:
        _hinder_counters["frozen"] += 1
    else:
        _hinder_counters["frozen"] = 0

    if mean < COVERED_MEAN_THRESH and std < LOW_CONTRAST_STD_THRESH:
        _hinder_counters["covered"] += 1
    else:
        _hinder_counters["covered"] = 0

    if std < LOW_CONTRAST_STD_THRESH:
        _hinder_counters["low_contrast"] += 1
    else:
        _hinder_counters["low_contrast"] = 0

    if edge_density < OBSTRUCT_EDGE_DENSITY_THRESH or dark_pct >= DARK_PCT_THRESH:
        _hinder_counters["obstructed"] += 1
    else:
        _hinder_counters["obstructed"] = 0

    reason = ""
    if _hinder_counters["frozen"] >= FROZEN_FRAMES_THRESH:
        reason = "camera_frozen"
    elif _hinder_counters["covered"] >= COVERED_FRAMES_THRESH:
        reason = "lens_covered_or_extremely_dark"
    elif _hinder_counters["obstructed"] >= OBSTRUCT_FRAMES_THRESH:
        reason = "obstructed_or_low_edge_density"
    elif _hinder_counters["low_contrast"] >= LOW_CONTRAST_FRAMES_THRESH:
        reason = "low_contrast"

    _prev_gray_for_hinder = gray.copy()

    return bool(reason), reason


def enhance_frame(frame_bgr,
                  gamma=1.6,
                  use_clahe=True,
                  clahe_clip=2.0,
                  clahe_tile=(8, 8),
                  denoise=False):
    img = frame_bgr.copy()
    if gamma != 1.0:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(256)]).astype("uint8")
        img = cv2.LUT(img, table)

    if use_clahe:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
        v_clahe = clahe.apply(v)
        hsv_clahe = cv2.merge((h, s, v_clahe))
        img = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ---------- MCNN model ----------
class MC_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.column1 = nn.Sequential(
            nn.Conv2d(3, 8, 9, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 7, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 7, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 16, 7, padding="same"),
            nn.ReLU(),
            nn.Conv2d(16, 8, 7, padding="same"),
            nn.ReLU(),
        )

        self.column2 = nn.Sequential(
            nn.Conv2d(3, 10, 7, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding="same"),
            nn.ReLU(),
            nn.Conv2d(40, 20, 5, padding="same"),
            nn.ReLU(),
            nn.Conv2d(20, 10, 5, padding="same"),
            nn.ReLU(),
        )

        self.column3 = nn.Sequential(
            nn.Conv2d(3, 12, 5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(48, 24, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(24, 12, 3, padding="same"),
            nn.ReLU(),
        )

        self.fusion_layer = nn.Conv2d(30, 1, 1, padding=0)

    def forward(self, img_tensor):
        x1 = self.column1(img_tensor)
        x2 = self.column2(img_tensor)
        x3 = self.column3(img_tensor)
        x = torch.cat((x1, x2, x3), 1)
        x = self.fusion_layer(x)
        return x


def load_model(path, device):
    m = MC_CNN().to(device).eval()
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    elif isinstance(ckpt, nn.Module):
        ckpt.to(device).eval()
        return ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format")

    new_state = {}
    for k, v in state.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        new_state[nk] = v

    m.load_state_dict(new_state, strict=False)
    return m


model = load_model(MODEL_PATH, device)
if device.type == "cuda" and USE_FP16_IF_CUDA:
    model.half()


def frame_to_tensor(frame_bgr):
    if FRAME_RESIZE is not None:
        frame_bgr = cv2.resize(frame_bgr, FRAME_RESIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0)
    if device.type == "cuda" and USE_FP16_IF_CUDA:
        tensor = tensor.half().to(device, non_blocking=True)
    else:
        tensor = tensor.to(device, non_blocking=True)
    return tensor


def ms_to_time_str(t_ms):
    ms = int(t_ms % 1000)
    total_seconds = int(t_ms // 1000)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}:{ms:03d}"


# # ---------- per-run tracking (in-memory CSV, no file on disk) ----------
# RUNS = {}  # run_id -> {"csv": str, "done": bool}
# RUNS_LOCK = threading.Lock()


# def process_video_to_csv(video_path: str, run_id: str):
#     """
#     Process video, compute crowd counts, and build a CSV string in memory.
#     No CSV file is saved to disk.
#     """
#     global _hinder_counters, _prev_gray_for_hinder
#     _hinder_counters = {k: 0 for k in _hinder_counters}
#     _prev_gray_for_hinder = None

#     # in-memory CSV buffer
#     buf = StringIO()
#     writer = csv.writer(buf)

#     # today's date in yyyy-mm-dd format
#     today_str = datetime.now().strftime("%Y-%m-%d")

#     # header
#     writer.writerow(["date", "timestamp_ms", "frame_index", "count", "alert"])

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Cannot open video:", video_path)
#         # still store at least header
#         with RUNS_LOCK:
#             RUNS[run_id]["csv"] = buf.getvalue()
#             RUNS[run_id]["done"] = True
#         return

#     smoother = []
#     processed = 0
#     last_flush = time.time()
#     start_wall = time.time()

#     with torch.no_grad():
#         while True:
#             ok, frame = cap.read()
#             if not ok:
#                 print("End of video reached.")
#                 break

#             if processed % SKIP_FRAMES != 0:
#                 processed += 1
#                 continue

#             enhanced = enhance_frame(frame, gamma=1.6, use_clahe=True, denoise=False)
#             inp = frame_to_tensor(enhanced)

#             out = model(inp)
#             out_f = out.float() if (device.type == "cuda" and USE_FP16_IF_CUDA) else out
#             out_f = torch.relu(out_f)
#             dmap = out_f.squeeze(0).squeeze(0)

#             orig_h, orig_w = frame.shape[:2]
#             if FRAME_RESIZE is not None:
#                 feed_w, feed_h = FRAME_RESIZE
#             else:
#                 feed_h, feed_w = inp.shape[2], inp.shape[3]

#             if UPSAMPLE_DMAP:
#                 dmap_unsq = dmap.unsqueeze(0).unsqueeze(0)
#                 dmap_up = F.interpolate(
#                     dmap_unsq,
#                     size=(orig_h, orig_w),
#                     mode="bilinear",
#                     align_corners=False,
#                 )
#                 dmap_up = dmap_up.squeeze(0).squeeze(0)
#                 count_val = float(dmap_up.sum().item())
#             elif SCALE_BY_AREA:
#                 raw_count = float(dmap.sum().item())
#                 scale = (orig_h * orig_w) / (feed_h * feed_w)
#                 count_val = raw_count * scale
#             else:
#                 count_val = float(dmap.sum().item())

#             if SMOOTH_WINDOW and SMOOTH_WINDOW > 1:
#                 smoother.append(count_val)
#                 if len(smoother) > SMOOTH_WINDOW:
#                     smoother.pop(0)
#                 display_count = sum(smoother) / len(smoother)
#             else:
#                 display_count = count_val

#             t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
#             time_str = ms_to_time_str(t_ms)

#             is_hindered, reason = check_camera_hinder(frame)
#             alert_field = reason if is_hindered else ""
#             if is_hindered:
#                 display_count = 0.0

#             writer.writerow([
#                 today_str,              # date (yyyy-mm-dd)
#                 time_str,               # timestamp_ms (mm:ss:ms string)
#                 processed,              # frame_index
#                 f"{display_count:.3f}", # count
#                 alert_field             # alert
#             ])

#             processed += 1

#             if processed % PRINT_EVERY == 0:
#                 elapsed = time.time() - start_wall
#                 fps_eff = processed / elapsed if elapsed > 0 else 0
#                 print(
#                     f"Processed {processed} frames (effective FPS: {fps_eff:.2f}), "
#                     f"last count {display_count:.2f}"
#                 )

#     cap.release()
#     csv_text = buf.getvalue()
#     buf.close()

#     with RUNS_LOCK:
#         RUNS[run_id]["csv"] = csv_text
#         RUNS[run_id]["done"] = True

#     try:
#         os.remove(video_path)
#     except OSError:
#         pass

#     print("Finished. Stored CSV in memory for run_id:", run_id)


# def run_processing(video_path: str, run_id: str):
#     with RUNS_LOCK:
#         RUNS[run_id]["done"] = False

#     process_video_to_csv(video_path, run_id)


# def run_processing(video_path: str, run_id: str):
#     csv_path = None
#     with RUNS_LOCK:
#         csv_path = RUNS[run_id]["csv_path"]
#         RUNS[run_id]["done"] = False

#     process_video_to_csv(video_path, csv_path)

#     with RUNS_LOCK:
#         RUNS[run_id]["done"] = True

#     try:
#         os.remove(video_path)
#     except OSError:
#         pass


# ---------- per-run tracking (in-memory CSV, no file on disk) ----------
RUNS = {}  # run_id -> {"csv": str, "done": bool}
RUNS_LOCK = threading.Lock()


def process_video_to_csv(video_path: str, run_id: str):
    """
    Process video, compute crowd counts, and build a CSV string in memory.
    No CSV file is saved to disk.
    """
    global _hinder_counters, _prev_gray_for_hinder
    _hinder_counters = {k: 0 for k in _hinder_counters}
    _prev_gray_for_hinder = None

    # in-memory CSV buffer
    buf = StringIO()
    writer = csv.writer(buf)

    # today's date in yyyy-mm-dd format
    today_str = datetime.now().strftime("%Y-%m-%d")

    # header
    writer.writerow(["date", "timestamp_ms", "frame_index", "count", "alert"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video:", video_path)
        # still store at least header
        with RUNS_LOCK:
            RUNS[run_id]["csv"] = buf.getvalue()
            RUNS[run_id]["done"] = True
        return

    smoother = []
    processed = 0
    start_wall = time.time()

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                print("End of video reached.")
                break

            if processed % SKIP_FRAMES != 0:
                processed += 1
                continue

            enhanced = enhance_frame(frame, gamma=1.6, use_clahe=True, denoise=False)
            inp = frame_to_tensor(enhanced)

            out = model(inp)
            out_f = out.float() if (device.type == "cuda" and USE_FP16_IF_CUDA) else out
            out_f = torch.relu(out_f)
            dmap = out_f.squeeze(0).squeeze(0)

            orig_h, orig_w = frame.shape[:2]
            if FRAME_RESIZE is not None:
                feed_w, feed_h = FRAME_RESIZE
            else:
                feed_h, feed_w = inp.shape[2], inp.shape[3]

            if UPSAMPLE_DMAP:
                dmap_unsq = dmap.unsqueeze(0).unsqueeze(0)
                dmap_up = F.interpolate(
                    dmap_unsq,
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                )
                dmap_up = dmap_up.squeeze(0).squeeze(0)
                count_val = float(dmap_up.sum().item())
            elif SCALE_BY_AREA:
                raw_count = float(dmap.sum().item())
                scale = (orig_h * orig_w) / (feed_h * feed_w)
                count_val = raw_count * scale
            else:
                count_val = float(dmap.sum().item())

            if SMOOTH_WINDOW and SMOOTH_WINDOW > 1:
                smoother.append(count_val)
                if len(smoother) > SMOOTH_WINDOW:
                    smoother.pop(0)
                display_count = sum(smoother) / len(smoother)
            else:
                display_count = count_val

            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            time_str = ms_to_time_str(t_ms)

            is_hindered, reason = check_camera_hinder(frame)
            alert_field = reason if is_hindered else ""
            if is_hindered:
                display_count = 0.0

            writer.writerow([
                today_str,              # date (yyyy-mm-dd)
                time_str,               # timestamp_ms (mm:ss:ms string)
                processed,              # frame_index
                f"{display_count:.3f}", # count
                alert_field             # alert
            ])

            processed += 1

            if processed % PRINT_EVERY == 0:
                elapsed = time.time() - start_wall
                fps_eff = processed / elapsed if elapsed > 0 else 0
                print(
                    f"Processed {processed} frames (effective FPS: {fps_eff:.2f}), "
                    f"last count {display_count:.2f}"
                )

    cap.release()
    csv_text = buf.getvalue()
    buf.close()

    with RUNS_LOCK:
        RUNS[run_id]["csv"] = csv_text
        RUNS[run_id]["done"] = True

    print("Finished. Stored CSV in memory for run_id:", run_id)


def run_processing(video_path: str, run_id: str):
    """
    Wrapper used by BackgroundTasks.
    """
    with RUNS_LOCK:
        RUNS[run_id]["done"] = False

    process_video_to_csv(video_path, run_id)

    # Remove temp video file if it still exists
    try:
        os.remove(video_path)
    except OSError:
        pass


# ---------- FastAPI ----------
# app = FastAPI()
# app = FastAPI(title="Video Analytics API")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# @app.post("/process_video/")
# async def process_video(
#     background_tasks: BackgroundTasks,              # <-- no default, comes first
#     video: UploadFile = File(...),                 # <-- name "video" to match frontend
# ):
#     # unique id per upload
#     run_id = str(int(time.time() * 1000))
#     video_filename = f"uploaded_{run_id}.mp4"
#     video_path = os.path.join(BASE_DIR, video_filename)
#     csv_path = os.path.join(BASE_DIR, f"output_with_hinderance_{run_id}.csv")

#     # save video
#     with open(video_path, "wb") as f:
#         while True:
#             chunk = await video.read(1024 * 1024)
#             if not chunk:
#                 break
#             f.write(chunk)

#     with RUNS_LOCK:
#         RUNS[run_id] = {"csv_path": csv_path, "done": False}

#     # run processing in background
#     background_tasks.add_task(run_processing, video_path, run_id)

#     return {"status": "processing_started", "run_id": run_id}

# ---------- FastAPI ----------
app = FastAPI(title="Video Analytics API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process_video/")
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    # unique id per upload
    run_id = str(int(time.time() * 1000))

    print(f"[process_video] run_id={run_id}")

    # create a temporary file (NOT in your backend folder)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        temp_video_path = tmp.name
        print(f"[process_video] writing temp video to: {temp_video_path}")

        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)

    # initialize run entry in memory
    with RUNS_LOCK:
        RUNS[run_id] = {"csv": "", "done": False}

    # run processing in background
    background_tasks.add_task(run_processing, temp_video_path, run_id)

    return {"status": "processing_started", "run_id": run_id}

# @app.get("/crowd_txt/")
# async def crowd_txt(run_id: str = Query(...)):
#     with RUNS_LOCK:
#         run_info = RUNS.get(run_id)

#     if not run_info:
#         return JSONResponse({"csv": "", "done": True})

#     csv_path = run_info["csv_path"]
#     done = run_info["done"]

#     if not os.path.exists(csv_path):
#         return JSONResponse({"csv": "", "done": done})

#     with open(csv_path, "r", encoding="utf-8") as f:
#         csv_text = f.read()

#     return JSONResponse({"csv": csv_text, "done": done})


@app.get("/crowd_txt/{run_id}")
async def crowd_txt(run_id: str):
    """
    Frontend calls: GET /analytics/crowd_txt/{run_id}
    We return the CSV text (from memory) + done flag.
    """
    with RUNS_LOCK:
        run_info = RUNS.get(run_id)

    if not run_info:
        # unknown run_id
        return JSONResponse({"csv": "", "done": True})

    csv_text = run_info.get("csv", "")
    done = run_info.get("done", False)

    return JSONResponse({"csv": csv_text, "done": done})


