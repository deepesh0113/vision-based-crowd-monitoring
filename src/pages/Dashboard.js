import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  CartesianGrid,
  Legend,
  Cell,
  ReferenceLine,
} from "recharts";

function useWindowWidth() {
  const [width, setWidth] = useState(window.innerWidth);
  useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);
  return width;
}

function Dashboard() {
  const navigate = useNavigate();

  const [csvFile, setCsvFile] = useState(null);

  // filtered data used by day-wise graphs
  const [graphData, setGraphData] = useState([]); // time-series records
  const [perSecondData, setPerSecondData] = useState([]);
  const [frameSeries, setFrameSeries] = useState([]);
  const [summary, setSummary] = useState(null);

  // raw (unfiltered) data for filters
  const [rawGraphData, setRawGraphData] = useState([]);
  const [rawPerSecondData, setRawPerSecondData] = useState([]);
  const [rawFrameSeries, setRawFrameSeries] = useState([]);

  // day-wise filter
  const [availableDates, setAvailableDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState("");

  // month-wise filter
  const [availableMonths, setAvailableMonths] = useState([]);
  const [selectedMonth, setSelectedMonth] = useState("");
  const [monthDailyData, setMonthDailyData] = useState([]); // aggregated per day for month view

  // view mode: "day" or "month"
  const [viewMode, setViewMode] = useState("day");

  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]); // left-side history
  const [selectedHistory, setSelectedHistory] = useState(null);

  // thresholds
  const [minThreshold, setMinThreshold] = useState("");
  const [maxThreshold, setMaxThreshold] = useState("");

  // details toggles
  const [showTimeDetails, setShowTimeDetails] = useState(false);
  const [showPerSecondDetails, setShowPerSecondDetails] = useState(false);
  const [showFrameDetails, setShowFrameDetails] = useState(false);


  // year-wise filter
  const [availableYears, setAvailableYears] = useState([]);
  const [selectedYear, setSelectedYear] = useState("");
  const [yearMonthlyData, setYearMonthlyData] = useState([]); // aggregated per month for year view

  const [showMonthDetails, setShowMonthDetails] = useState(false);
  const [showYearDetails, setShowYearDetails] = useState(false);
  const [showMonthDetails2, setShowMonthDetails2] = useState(false);
  const [showMonthDetails3, setShowMonthDetails3] = useState(false);
  const [showYearDetails2, setShowYearDetails2] = useState(false);
  const [showYearDetails3, setShowYearDetails3] = useState(false);



  const width = useWindowWidth();
  const isMobile = width < 700;

  // parsed thresholds
  const parsedMin = parseFloat(minThreshold);
  const parsedMax = parseFloat(maxThreshold);
  const thresholdsActive =
    !Number.isNaN(parsedMin) && !Number.isNaN(parsedMax);

  // Theme colors
  const BG = "#020617";
  const BLUE = "#3b82f6";
  const CYAN = "#0ea5e9";
  const GREEN = "#22c55e";
  const RED = "#ef4444";
  const AMBER = "#facc15";
  const LENS_ALERT = "#a855f7";   // violet for lens_covered_or_extremely_dark
  const FREEZE_ALERT = "#f97316"; // orange for camera_frozen


  const styles = {
    page: {
      minHeight: "100vh",
      backgroundColor: BG,
      fontFamily:
        "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      display: "flex",
      justifyContent: "center",
      padding: isMobile ? "18px 4vw 32px" : "28px 28px 40px",
      boxSizing: "border-box",
    },
    content: {
      width: "100%",
      maxWidth: 1120,
      display: "flex",
      flexDirection: "column",
      gap: isMobile ? 16 : 20,
      animation: "fadeIn 0.5s ease-in",
    },
    topBar: {
      display: "flex",
      alignItems: "center",
      justifyContent: "flex-start",
      gap: 16,
      flexWrap: "wrap",
    },
    backBtn: {
      borderRadius: 999,
      border: "1px solid #1e293b",
      backgroundColor: "transparent",
      color: "#9ca3af",
      padding: "7px 16px",
      fontSize: 13,
      cursor: "pointer",
    },
    brandBlock: {
      display: "flex",
      flexDirection: "column",
      gap: 2,
      textAlign: "left",
    },
    brand: {
      fontSize: 11,
      textTransform: "uppercase",
      letterSpacing: "0.23em",
      color: "#6b7280",
    },
    header: {
      fontWeight: 600,
      fontSize: isMobile ? "1.25rem" : "1.6rem",
      color: "#e5e7eb",
    },
    subtitle: {
      fontSize: 12,
      color: "#9ca3af",
    },

    theoryCard: {
      background: "radial-gradient(circle at top left, #020617, #020617 60%)",
      borderRadius: 18,
      padding: isMobile ? "14px 14px" : "18px 18px",
      boxShadow: "0 24px 52px rgba(15,23,42,0.9)",
      border: "1px solid #111827",
    },
    theoryTitle: {
      fontSize: 14,
      textTransform: "uppercase",
      letterSpacing: "0.18em",
      color: "#9ca3af",
      marginBottom: 8,
    },
    theoryHeading: {
      fontSize: 15,
      fontWeight: 600,
      color: "#e5e7eb",
      marginBottom: 8,
    },
    theoryText: {
      fontSize: 13,
      color: "#9ca3af",
      lineHeight: 1.65,
    },
    theoryList: {
      marginTop: 10,
      paddingLeft: 18,
      fontSize: 13,
      color: "#9ca3af",
      lineHeight: 1.6,
    },

    mainRow: {
      display: "flex",
      flexDirection: isMobile ? "column" : "row",
      gap: 18,
      alignItems: "flex-start",
    },
    leftCol: {
      flex: isMobile ? "unset" : "0 0 280px",
      width: isMobile ? "100%" : 280,
      display: "flex",
      flexDirection: "column",
      gap: 14,
    },
    rightCol: {
      flex: 1,
      display: "flex",
      flexDirection: "column",
      gap: 14,
    },

    historyCard: {
      background: "radial-gradient(circle at top, #020617, #020617 70%)",
      borderRadius: 18,
      padding: isMobile ? "14px 14px" : "16px 16px",
      boxShadow: "0 22px 50px rgba(15,23,42,0.9)",
      border: "1px solid #111827",
    },
    historyTitle: {
      fontSize: 14,
      color: "#93c5fd",
      fontWeight: 600,
      marginBottom: 6,
    },
    historySub: {
      fontSize: 12,
      color: "#6b7280",
      marginBottom: 10,
    },
    historyList: {
      margin: 0,
      padding: 0,
      listStyle: "none",
      maxHeight: 260,
      overflowY: "auto",
    },
    historyItem: {
      padding: "7px 0",
      borderBottom: "1px solid rgba(15,23,42,0.9)",
    },
    historyName: {
      fontSize: 13,
      color: "#e5e7eb",
      marginBottom: 2,
      wordBreak: "break-all",
    },
    historyMeta: {
      fontSize: 11,
      color: "#9ca3af",
    },
    historyEmpty: {
      fontSize: 12,
      color: "#6b7280",
      marginTop: 6,
    },

    graphBox: {
      background: "radial-gradient(circle at top right, #020617, #020617 70%)",
      borderRadius: 18,
      border: "1px solid #111827",
      boxShadow: "0 24px 52px rgba(15,23,42,0.95)",
      padding: isMobile ? "16px 12px 20px" : "18px 18px 24px",
      display: "flex",
      flexDirection: "column",
      gap: 10,
    },
    graphTitle: {
      fontSize: 14,
      color: BLUE,
      fontWeight: 600,
    },
    graphSubtitle: {
      fontSize: 12,
      color: "#6b7280",
    },
    fileInput: {
      marginTop: 8,
      marginBottom: 10,
      padding: 7,
      borderRadius: 10,
      background: BG,
      color: "#e5e7eb",
      border: "1px solid #111827",
      fontSize: 13,
    },
    resetBtn: {
      backgroundColor: "#10b981",
      color: "#fff",
      padding: "8px 20px",
      borderRadius: 999,
      cursor: "pointer",
      border: "none",
      fontWeight: 600,
    },
    actionBtn: {
      backgroundColor: CYAN,
      color: "#0b1120",
      cursor: "pointer",
      border: "none",
      fontWeight: 600,
      padding: "7px 16px",
      borderRadius: 999,
      fontSize: 13,
    },
    selectedFile: {
      marginTop: 8,
      fontSize: 12,
      color: "#93c5fd",
      wordBreak: "break-all",
    },

    graphArea: {
      width: "100%",
      height: 320,
      marginTop: 4,
    },
    graphPlaceholder: {
      flex: 1,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontSize: 13,
      color: CYAN,
      borderRadius: 12,
      border: "1px dashed #1e293b",
    },
    summaryRow: {
      display: "flex",
      flexWrap: "wrap",
      gap: 10,
      fontSize: 12,
      color: "#9ca3af",
      marginTop: 4,
    },
    summaryChip: {
      padding: "4px 10px",
      borderRadius: 999,
      border: "1px solid #1e293b",
    },
    thresholdRow: {
      display: "flex",
      flexWrap: "wrap",
      gap: 8,
      marginTop: 10,
      alignItems: "center",
      fontSize: 12,
      color: "#9ca3af",
    },
    thresholdInput: {
      width: 90,
      padding: "4px 8px",
      borderRadius: 999,
      border: "1px solid #1f2937",
      backgroundColor: "#020617",
      color: "#e5e7eb",
      fontSize: 12,
      outline: "none",
    },
    detailBtn: {
      alignSelf: "flex-end",
      marginTop: 4,
      padding: "4px 10px",
      fontSize: 11,
      borderRadius: 999,
      border: "1px solid #1f2937",
      backgroundColor: "#020617",
      color: "#9ca3af",
      cursor: "pointer",
    },
    detailPanel: {
      marginTop: 8,
      padding: "8px 10px",
      borderRadius: 12,
      border: "1px solid #1f2937",
      backgroundColor: "#020617",
      fontSize: 11,
      maxHeight: 160,
      overflowY: "auto",
      lineHeight: 1.5,
    },
    detailSectionTitle: {
      fontWeight: 600,
      marginTop: 4,
      marginBottom: 2,
      fontSize: 11,
    },
    dateSelect: {
      padding: "4px 8px",
      borderRadius: 999,
      border: "1px solid #1f2937",
      backgroundColor: "#020617",
      color: "#e5e7eb",
      fontSize: 12,
      outline: "none",
    },
    filterBtn: {
      backgroundColor: BLUE,
      color: "#f9fafb",
      border: "none",
      borderRadius: 999,
      padding: "6px 14px",
      fontSize: 12,
      cursor: "pointer",
      fontWeight: 500,
    },
    viewToggleRow: {
      display: "flex",
      flexWrap: "wrap",
      gap: 8,
      marginTop: 10,
      alignItems: "center",
      fontSize: 12,
      color: "#9ca3af",
    },
    viewToggleBtn: {
      padding: "5px 12px",
      borderRadius: 999,
      border: "1px solid #1f2937",
      backgroundColor: "#020617",
      color: "#9ca3af",
      fontSize: 12,
      cursor: "pointer",
    },
    viewToggleBtnActive: {
      backgroundColor: BLUE,
      color: "#f9fafb",
      borderColor: BLUE,
    },
  };

  // custom dot renderer for line charts based on thresholds
  // const renderColoredDot = (props) => {
  //   const { cx, cy, payload } = props;
  //   const value = payload.count;
  //   let fill = "#60a5fa";

  //   if (thresholdsActive) {
  //     if (value > parsedMax) fill = RED; // alert
  //     else if (value >= parsedMin && value <= parsedMax) fill = GREEN; // safe
  //     else fill = AMBER; // below min
  //   }

  //   return (
  //     <circle cx={cx} cy={cy} r={3} fill={fill} stroke={BG} strokeWidth={1} />
  //   );
  // };
  // custom dot renderer for line charts based on thresholds + alert type
  const renderColoredDot = (props) => {
    const { cx, cy, payload } = props;
    const value = payload.count;

    // alert can be: "lens_covered_or_extremely_dark", "camera_frozen", or empty/undefined
    const alertType = payload.alert ? String(payload.alert).trim() : "";

    let fill = "#60a5fa"; // default blue-ish
    let radius = 3;

    if (alertType === "lens_covered_or_extremely_dark") {
      // camera covered / extremely dark -> violet dot
      fill = LENS_ALERT;
      radius = 5;
    } else if (alertType === "camera_frozen") {
      // camera frozen -> orange dot
      fill = FREEZE_ALERT;
      radius = 5;
    } else if (thresholdsActive) {
      // fallback to threshold-based coloring if no explicit alert
      if (value > parsedMax) fill = RED; // alert
      else if (value >= parsedMin && value <= parsedMax) fill = GREEN; // safe
      else fill = AMBER; // below min
    }

    return (
      <circle
        cx={cx}
        cy={cy}
        r={radius}
        fill={fill}
        stroke={BG}
        strokeWidth={1}
      />
    );
  };


  // ---------- Filter helpers ----------

  // Day-wise filter: single date
  const applyDateFilter = (date, recordsSrc, perSecondSrc, frameSrc) => {
    if (!date) {
      setGraphData(recordsSrc);
      setPerSecondData(perSecondSrc);
      setFrameSeries(frameSrc);
      return;
    }

    const filteredRecords = recordsSrc.filter((r) => r.date === date);
    const filteredPerSecond = perSecondSrc.filter((r) => r.date === date);
    const filteredFrame = frameSrc.filter((r) => r.date === date);

    setGraphData(filteredRecords);
    setPerSecondData(filteredPerSecond);
    setFrameSeries(filteredFrame);
  };

  // Month-wise filter: aggregate per date inside selected month
  const applyMonthFilter = (monthKey, recordsSrc) => {
    if (!monthKey) {
      setMonthDailyData([]);
      return;
    }

    // recordsSrc has { date, timestamp, count }
    const monthRecords = recordsSrc.filter(
      (r) => r.date && r.date.startsWith(monthKey)
    );

    const dayMap = {};

    monthRecords.forEach((r) => {
      const d = r.date; // "YYYY-MM-DD"
      if (!dayMap[d]) {
        dayMap[d] = {
          date: d,
          dayLabel: d.slice(8, 10),
          total: 0,
          max: -Infinity,
          n: 0,
        };
      }
      dayMap[d].total += r.count;
      dayMap[d].max = Math.max(dayMap[d].max, r.count);
      dayMap[d].n += 1;
    });

    const dailyArr = Object.values(dayMap)
      .map((d) => ({
        date: d.date,
        day: d.dayLabel,
        avg_count: d.total / d.n,
        max_count: d.max,
        total_count: d.total,
      }))
      .sort((a, b) => a.date.localeCompare(b.date));

    setMonthDailyData(dailyArr);
  };


  // Year-wise filter: aggregate per month inside selected year
  const applyYearFilter = (yearKey, recordsSrc) => {
    if (!yearKey) {
      setYearMonthlyData([]);
      return;
    }

    // recordsSrc has { date, timestamp, count }
    const yearRecords = recordsSrc.filter(
      (r) => r.date && r.date.startsWith(yearKey)
    );

    const monthMap = {};
    yearRecords.forEach((r) => {
      // monthKey = "YYYY-MM"
      const monthKey = r.date.slice(0, 7);
      const shortMonth = monthKey.slice(5, 7); // "01", "02", ...

      if (!monthMap[monthKey]) {
        monthMap[monthKey] = {
          month: shortMonth, // for x-axis
          monthLabel: monthKey, // full "YYYY-MM"
          total: 0,
          max: -Infinity,
          n: 0,
        };
      }
      monthMap[monthKey].total += r.count;
      monthMap[monthKey].max = Math.max(monthMap[monthKey].max, r.count);
      monthMap[monthKey].n += 1;
    });

    const monthlyArr = Object.values(monthMap)
      .map((m) => ({
        month: m.month,
        monthLabel: m.monthLabel,
        avg_count: m.total / m.n,
        max_count: m.max,
        total_count: m.total,
      }))
      .sort((a, b) => a.monthLabel.localeCompare(b.monthLabel));

    setYearMonthlyData(monthlyArr);
  };


  // ---------- Upload handler ----------

  const handleCSVUpload = async () => {
    if (!csvFile) {
      alert("Please select a CSV file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", csvFile);

    setLoading(true);

    try {
      // const res = await fetch("http://127.0.0.1:8000/upload-csv", {
      //   method: "POST",
      //   body: formData,
      // });
      const res = await fetch("http://127.0.0.1:8000/dashboard/upload-csv", {
        method: "POST",
        body: formData,
      });


      const data = await res.json();

      if (data.status === "error") {
        alert(data.message || "Error processing CSV");
        setLoading(false);
        return;
      }

      const records = data.records || [];
      const perSecondAll = data.per_second || [];
      const frameAll = data.frame_series || [];
      const summaryObj = data.summary || null;

      // store raw (unfiltered)
      setRawGraphData(records);
      setRawPerSecondData(perSecondAll);
      setRawFrameSeries(frameAll);

      // collect unique dates from records
      const uniqueDates = Array.from(
        new Set(records.map((r) => r.date).filter(Boolean))
      );
      setAvailableDates(uniqueDates);
      const initialDate = uniqueDates[0] || "";
      setSelectedDate(initialDate);

      // collect unique months from records (YYYY-MM)
      const uniqueMonths = Array.from(
        new Set(
          records
            .map((r) => (r.date ? r.date.slice(0, 7) : null))
            .filter(Boolean)
        )
      );
      const uniqueYears = Array.from(
        new Set(
          records
            .map((r) => (r.date ? r.date.slice(0, 4) : null))
            .filter(Boolean)
        )
      );
      uniqueYears.sort();
      setAvailableYears(uniqueYears);

      const initialYear = uniqueYears[0] || "";
      setSelectedYear(initialYear);
      uniqueMonths.sort();
      setAvailableMonths(uniqueMonths);
      const initialMonth = uniqueMonths[0] || "";
      setSelectedMonth(initialMonth);




      // default to day view
      setViewMode("day");

      // apply initial day filter
      applyDateFilter(initialDate, records, perSecondAll, frameAll);

      // compute month aggregation for initial month (for when user switches view)
      applyMonthFilter(initialMonth, records);

      applyYearFilter(initialYear, records);

      setSummary(summaryObj);

      // ---- History entry ----
      const minVal =
        summaryObj && typeof summaryObj.min_count !== "undefined"
          ? summaryObj.min_count
          : "—";

      const maxVal =
        summaryObj && typeof summaryObj.max_count !== "undefined"
          ? summaryObj.max_count
          : "—";

      const newHistoryItem = {
        name: csvFile.name,
        uploadedAt: new Date().toLocaleString(),
        points: records.length,
        minCount: minVal,
        maxCount: maxVal,
      };

      setHistory((prev) => [newHistoryItem, ...prev]);
      setSelectedHistory(newHistoryItem);
    } catch (err) {
      console.error("Error:", err);
      alert("Failed to upload CSV");
    }

    setLoading(false);
  };


  const resetView = () => {
    setCsvFile(null);
    setGraphData([]);
    setPerSecondData([]);
    setFrameSeries([]);
    setSummary(null);
    setMinThreshold("");
    setMaxThreshold("");
    setShowTimeDetails(false);
    setShowPerSecondDetails(false);
    setShowFrameDetails(false);

    setRawGraphData([]);
    setRawPerSecondData([]);
    setRawFrameSeries([]);
    setAvailableDates([]);
    setSelectedDate("");
    setAvailableMonths([]);
    setSelectedMonth("");
    setMonthDailyData([]);
    setViewMode("day");
    setTimeout(() => {
      window.location.reload();
    }, 150);
  };

  // ---------- helpers for detail panels ----------

  const getTimeAlerts = () =>
    thresholdsActive ? graphData.filter((p) => p.count > parsedMax) : [];

  const getTimeSafe = () =>
    thresholdsActive
      ? graphData.filter(
        (p) => p.count >= parsedMin && p.count <= parsedMax
      )
      : [];

  const getPerSecondAlerts = () =>
    thresholdsActive
      ? perSecondData.filter((p) => p.avg_count > parsedMax)
      : [];

  const getPerSecondSafe = () =>
    thresholdsActive
      ? perSecondData.filter(
        (p) => p.avg_count >= parsedMin && p.avg_count <= parsedMax
      )
      : [];

  const getFrameAlerts = () =>
    thresholdsActive ? frameSeries.filter((p) => p.count > parsedMax) : [];

  const getFrameSafe = () =>
    thresholdsActive
      ? frameSeries.filter(
        (p) => p.count >= parsedMin && p.count <= parsedMax
      )
      : [];


  // ---------- Month-wise detail helpers (use monthDailyData) ----------
  const getMonthAlertDays = () =>
    thresholdsActive
      ? monthDailyData.filter((d) => d.max_count > parsedMax)
      : [];

  const getMonthSafeDays = () =>
    thresholdsActive
      ? monthDailyData.filter(
        (d) => d.avg_count >= parsedMin && d.avg_count <= parsedMax
      )
      : [];

  // ---------- Year-wise detail helpers (use yearMonthlyData) ----------
  const getYearAlertMonths = () =>
    thresholdsActive
      ? yearMonthlyData.filter((m) => m.max_count > parsedMax)
      : [];

  const getYearSafeMonths = () =>
    thresholdsActive
      ? yearMonthlyData.filter(
        (m) => m.avg_count >= parsedMin && m.avg_count <= parsedMax
      )
      : [];




  return (
    <>
      <div style={styles.page}>
        <div style={styles.content}>
          {/* TOP BAR */}
          <div style={styles.topBar}>
            <button style={styles.backBtn} onClick={() => navigate(-1)}>
              ← Back
            </button>

            <div style={styles.brandBlock}>
              <div style={styles.brand}>VigilNet Dashboard</div>
              <div style={styles.header}>Historical Crowd Analytics</div>
              <div style={styles.subtitle}>
                Upload model outputs as CSV and explore time-based crowd trends
                for research, debugging, and reporting.
              </div>
            </div>
          </div>

          {/* THEORY CARD */}
          <div style={styles.theoryCard}>
            <div style={styles.theoryTitle}>Why this dashboard matters</div>
            <div style={styles.theoryHeading}>
              From raw CSV logs to decisions you can defend
            </div>
            <p style={styles.theoryText}>
              VigilNet’s historical dashboard turns date + timestamp + count
              logs into a visual narrative. You can zoom into a single day or
              step back to see how whole months behave, spotting spikes, quiet
              days, and recurring patterns.
            </p>
            <ul style={styles.theoryList}>
              <li>
                <strong>Day-wise:</strong> inspect how counts evolve within a
                day (timestamp-wise, per-second, per-frame).
              </li>
              <li>
                <strong>Month-wise:</strong> summarise how each day in a month
                behaves using average, peak, and total crowd counts.
              </li>
            </ul>
          </div>

          {/* MAIN ROW */}
          <div style={styles.mainRow}>
            {/* LEFT COLUMN: HISTORY */}
            <div style={styles.leftCol}>
              <div style={styles.historyCard}>
                <div style={styles.historyTitle}>Upload history</div>
                <div style={styles.historySub}>
                  Recent CSV files processed on this dashboard.
                </div>

                {history.length === 0 ? (
                  <div style={styles.historyEmpty}>
                    No files analyzed yet. Upload a CSV to start building your
                    history.
                  </div>
                ) : (
                  <>
                    <ul style={styles.historyList}>
                      {history.map((item, idx) => (
                        <li
                          key={idx}
                          style={{
                            ...styles.historyItem,
                            cursor: "pointer",
                            backgroundColor:
                              selectedHistory === item
                                ? "rgba(15,23,42,0.8)"
                                : "transparent",
                          }}
                          onClick={() => setSelectedHistory(item)}
                        >
                          <div style={styles.historyName}>{item.name}</div>
                          <div style={styles.historyMeta}>
                            {item.points} points · {item.uploadedAt}
                          </div>
                        </li>
                      ))}
                    </ul>

                    {selectedHistory && (
                      <div
                        style={{
                          marginTop: 10,
                          paddingTop: 8,
                          borderTop: "1px solid rgba(15,23,42,0.9)",
                          fontSize: 12,
                          color: "#9ca3af",
                        }}
                      >
                        <div>
                          Min count:{" "}
                          {selectedHistory.minCount?.toFixed
                            ? selectedHistory.minCount.toFixed(2)
                            : selectedHistory.minCount}
                        </div>
                        <div>
                          Max count:{" "}
                          {selectedHistory.maxCount?.toFixed
                            ? selectedHistory.maxCount.toFixed(2)
                            : selectedHistory.maxCount}
                        </div>
                        <div>Total points: {selectedHistory.points}</div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>

            {/* RIGHT COLUMN: UPLOAD + GRAPHS */}
            <div style={styles.rightCol}>
              {/* CSV Upload + Thresholds + View & Filters */}
              <div style={styles.graphBox}>
                <div style={styles.graphTitle}>Upload CSV file</div>
                <div style={styles.graphSubtitle}>
                  Expected format:&nbsp;
                  <code>date, timestamp_ns, frame_index, count</code>
                  <br />
                  Or<br />
                  <code>date, timestamp_ns, frame_index, count, alert</code>
                  <br /><br />

                  {/* Legend additions */}
                  <span style={{ color: "#a855f7", fontWeight: "bold" }}>●</span>
                  &nbsp;violet dot = lens_covered_or_extremely_dark
                  <br />

                  <span style={{ color: "#f97316", fontWeight: "bold" }}>●</span>
                  &nbsp;orange dot = camera_frozen
                </div>




                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => setCsvFile(e.target.files[0])}
                  style={styles.fileInput}
                />

                <div style={{ display: "flex", gap: 10, marginTop: 5 }}>
                  <button onClick={handleCSVUpload} style={styles.actionBtn}>
                    {loading ? "Processing..." : "Upload & Process"}
                  </button>

                  <button onClick={resetView} style={styles.resetBtn}>
                    Clear / Refresh
                  </button>
                </div>

                {csvFile && (
                  <div style={styles.selectedFile}>
                    Selected file: {csvFile.name}
                  </div>
                )}

                {/* Threshold controls */}
                <div style={styles.thresholdRow}>
                  <span>Thresholds:</span>
                  <span>Min (safe start)</span>
                  <input
                    type="number"
                    step="0.01"
                    value={minThreshold}
                    onChange={(e) => setMinThreshold(e.target.value)}
                    style={styles.thresholdInput}
                    placeholder="e.g. 100"
                  />
                  <span>Max (alert)</span>
                  <input
                    type="number"
                    step="0.01"
                    value={maxThreshold}
                    onChange={(e) => setMaxThreshold(e.target.value)}
                    style={styles.thresholdInput}
                    placeholder="e.g. 150"
                  />
                </div>

                {/* View mode toggle */}
                <div style={styles.viewToggleRow}>
                  <span>View mode:</span>
                  <button
                    style={{
                      ...styles.viewToggleBtn,
                      ...(viewMode === "day" ? styles.viewToggleBtnActive : {}),
                    }}
                    onClick={() => {
                      setViewMode("day");
                      applyDateFilter(
                        selectedDate,
                        rawGraphData,
                        rawPerSecondData,
                        rawFrameSeries
                      );
                    }}
                  >
                    Day-wise
                  </button>

                  <button
                    style={{
                      ...styles.viewToggleBtn,
                      ...(viewMode === "month" ? styles.viewToggleBtnActive : {}),
                    }}
                    onClick={() => {
                      setViewMode("month");
                      applyMonthFilter(selectedMonth, rawGraphData);
                    }}
                  >
                    Month-wise
                  </button>

                  <button
                    style={{
                      ...styles.viewToggleBtn,
                      ...(viewMode === "year" ? styles.viewToggleBtnActive : {}),
                    }}
                    onClick={() => {
                      setViewMode("year");
                      applyYearFilter(selectedYear, rawGraphData);
                    }}
                  >
                    Year-wise
                  </button>
                </div>


                {/* Day-wise vs Month-wise filters */}
                {viewMode === "day" && availableDates.length > 0 && (
                  <div style={styles.thresholdRow}>
                    <span>Filter by date:</span>
                    <select
                      value={selectedDate}
                      onChange={(e) => {
                        const value = e.target.value;
                        setSelectedDate(value);
                        applyDateFilter(
                          value,
                          rawGraphData,
                          rawPerSecondData,
                          rawFrameSeries
                        );
                      }}
                      style={styles.dateSelect}
                    >
                      <option value="">All dates</option>
                      {availableDates.map((d) => (
                        <option key={d} value={d}>
                          {d}
                        </option>
                      ))}
                    </select>
                    <button
                      style={styles.filterBtn}
                      onClick={() =>
                        applyDateFilter(
                          selectedDate,
                          rawGraphData,
                          rawPerSecondData,
                          rawFrameSeries
                        )
                      }
                    >
                      FILTER
                    </button>
                  </div>
                )}

                {viewMode === "month" && availableMonths.length > 0 && (
                  <div style={styles.thresholdRow}>
                    <span>Filter by month:</span>
                    <select
                      value={selectedMonth}
                      onChange={(e) => {
                        const value = e.target.value;
                        setSelectedMonth(value);
                        applyMonthFilter(value, rawGraphData);
                      }}
                      style={styles.dateSelect}
                    >
                      <option value="">All months</option>
                      {availableMonths.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                    <button
                      style={styles.filterBtn}
                      onClick={() =>
                        applyMonthFilter(selectedMonth, rawGraphData)
                      }
                    >
                      FILTER
                    </button>
                  </div>
                )}

                {viewMode === "year" && availableYears.length > 0 && (
                  <div style={styles.thresholdRow}>
                    <span>Filter by year:</span>
                    <select
                      value={selectedYear}
                      onChange={(e) => {
                        const value = e.target.value;
                        setSelectedYear(value);
                        applyYearFilter(value, rawGraphData);
                      }}
                      style={styles.dateSelect}
                    >
                      <option value="">All years</option>
                      {availableYears.map((y) => (
                        <option key={y} value={y}>
                          {y}
                        </option>
                      ))}
                    </select>
                    <button
                      style={styles.filterBtn}
                      onClick={() => applyYearFilter(selectedYear, rawGraphData)}
                    >
                      FILTER
                    </button>
                  </div>
                )}


                {summary && (
                  <div style={styles.summaryRow}>
                    <span style={styles.summaryChip}>
                      Min count: {summary.min_count.toFixed(2)}
                    </span>
                    <span style={styles.summaryChip}>
                      Max count: {summary.max_count.toFixed(2)}
                    </span>
                    <span style={styles.summaryChip}>
                      Mean count: {summary.mean_count.toFixed(2)}
                    </span>
                    <span style={styles.summaryChip}>
                      Points: {summary.num_points}
                    </span>
                  </div>
                )}
              </div>

              {/* ==== GRAPHS SECTION ==== */}

              {viewMode === "day" ? (
                <>
                  {/* DAY-WISE: GRAPH 1 */}
                  <div style={styles.graphBox}>
                    <div style={styles.graphTitle}>
                      Crowd trends (time-series, per day)
                    </div>
                    <div style={styles.graphSubtitle}>
                      time_sec vs count – green = within safe range, red = above
                      alert threshold.
                    </div>

                    <button
                      style={styles.detailBtn}
                      onClick={() => setShowTimeDetails((prev) => !prev)}
                    >
                      {showTimeDetails ? "Hide details" : "Show details"}
                    </button>

                    <div style={styles.graphArea}>
                      {graphData.length === 0 ? (
                        <div style={styles.graphPlaceholder}>
                          Upload a CSV file and pick a date to render the
                          time–series chart.
                        </div>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={graphData}>
                            <XAxis
                              dataKey="timestamp"
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Time (sec)",
                                position: "insideBottom",
                                offset: -4,
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <YAxis
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Count",
                                angle: -90,
                                position: "insideLeft",
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "#020617",
                                border: "1px solid #1e293b",
                                borderRadius: 8,
                                fontSize: 11,
                              }}
                              labelStyle={{ color: "#e5e7eb" }}
                            />
                            {thresholdsActive && (
                              <>
                                <ReferenceLine
                                  y={parsedMin}
                                  stroke={GREEN}
                                  strokeDasharray="3 3"
                                  label={{
                                    value: "Min safe",
                                    fill: GREEN,
                                    fontSize: 10,
                                  }}
                                />
                                <ReferenceLine
                                  y={parsedMax}
                                  stroke={RED}
                                  strokeDasharray="3 3"
                                  label={{
                                    value: "Max alert",
                                    fill: RED,
                                    fontSize: 10,
                                  }}
                                />
                              </>
                            )}
                            <Line
                              type="monotone"
                              dataKey="count"
                              stroke={BLUE}
                              strokeWidth={2}
                              dot={(props) => renderColoredDot(props)}
                              activeDot={{ r: 5 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      )}
                    </div>

                    {showTimeDetails && thresholdsActive && (
                      <div style={styles.detailPanel}>
                        <div style={styles.detailSectionTitle}>
                          Alert points (count &gt; max)
                        </div>
                        {getTimeAlerts().length === 0 ? (
                          <div>No alert points.</div>
                        ) : (
                          getTimeAlerts().map((p, idx) => (
                            <div key={`ta-${idx}`}>
                              date = {p.date}, t = {p.timestamp}, count ={" "}
                              {p.count.toFixed(3)}
                            </div>
                          ))
                        )}
                        <div style={styles.detailSectionTitle}>
                          Safe points (between min &amp; max)
                        </div>
                        {getTimeSafe().length === 0 ? (
                          <div>No safe points based on current thresholds.</div>
                        ) : (
                          getTimeSafe().map((p, idx) => (
                            <div key={`ts-${idx}`}>
                              date = {p.date}, t = {p.timestamp}, count ={" "}
                              {p.count.toFixed(3)}
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </div>

                  {/* DAY-WISE: GRAPH 2 */}
                  <div style={styles.graphBox}>
                    <div style={styles.graphTitle}>
                      Per-second average crowd level (selected day)
                    </div>
                    <div style={styles.graphSubtitle}>
                      Bars show average count per whole second – green safe, red
                      alert.
                    </div>

                    <button
                      style={styles.detailBtn}
                      onClick={() =>
                        setShowPerSecondDetails((prev) => !prev)
                      }
                    >
                      {showPerSecondDetails ? "Hide details" : "Show details"}
                    </button>

                    <div style={styles.graphArea}>
                      {perSecondData.length === 0 ? (
                        <div style={styles.graphPlaceholder}>
                          Upload a CSV and select a date to see per-second
                          averages.
                        </div>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={perSecondData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                              dataKey="second"
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Second",
                                position: "insideBottom",
                                offset: -4,
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <YAxis
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Avg Count",
                                angle: -90,
                                position: "insideLeft",
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "#020617",
                                border: "1px solid #1e293b",
                                borderRadius: 8,
                                fontSize: 11,
                              }}
                              labelStyle={{ color: "#e5e7eb" }}
                            />
                            <Legend />
                            <Bar dataKey="avg_count" name="Avg count">
                              {perSecondData.map((entry, index) => {
                                const v = entry.avg_count;
                                let fillColor = "#60a5fa";
                                if (thresholdsActive) {
                                  if (v > parsedMax) fillColor = RED;
                                  else if (
                                    v >= parsedMin &&
                                    v <= parsedMax
                                  )
                                    fillColor = GREEN;
                                  else fillColor = AMBER;
                                }
                                return (
                                  <Cell
                                    key={`cell-${index}`}
                                    fill={fillColor}
                                  />
                                );
                              })}
                            </Bar>
                            {thresholdsActive && (
                              <>
                                <ReferenceLine
                                  y={parsedMin}
                                  stroke={GREEN}
                                  strokeDasharray="3 3"
                                />
                                <ReferenceLine
                                  y={parsedMax}
                                  stroke={RED}
                                  strokeDasharray="3 3"
                                />
                              </>
                            )}
                          </BarChart>
                        </ResponsiveContainer>
                      )}
                    </div>

                    {showPerSecondDetails && thresholdsActive && (
                      <div style={styles.detailPanel}>
                        <div style={styles.detailSectionTitle}>
                          Alert seconds (avg &gt; max)
                        </div>
                        {getPerSecondAlerts().length === 0 ? (
                          <div>No alert seconds.</div>
                        ) : (
                          getPerSecondAlerts().map((p, idx) => (
                            <div key={`pa-${idx}`}>
                              date = {p.date}, second = {p.second}, avg ={" "}
                              {p.avg_count.toFixed(3)}
                            </div>
                          ))
                        )}
                        <div style={styles.detailSectionTitle}>
                          Safe seconds (between min &amp; max)
                        </div>
                        {getPerSecondSafe().length === 0 ? (
                          <div>
                            No safe seconds for current thresholds.
                          </div>
                        ) : (
                          getPerSecondSafe().map((p, idx) => (
                            <div key={`ps-${idx}`}>
                              date = {p.date}, second = {p.second}, avg ={" "}
                              {p.avg_count.toFixed(3)}
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </div>

                  {/* DAY-WISE: GRAPH 3 */}
                  <div style={styles.graphBox}>
                    <div style={styles.graphTitle}>
                      Frame index vs crowd count (selected day)
                    </div>
                    <div style={styles.graphSubtitle}>
                      Helps you inspect how the model behaves frame by frame for
                      the chosen day.
                    </div>

                    <button
                      style={styles.detailBtn}
                      onClick={() => setShowFrameDetails((prev) => !prev)}
                    >
                      {showFrameDetails ? "Hide details" : "Show details"}
                    </button>

                    <div style={styles.graphArea}>
                      {frameSeries.length === 0 ? (
                        <div style={styles.graphPlaceholder}>
                          Upload a CSV and select a date to see frame-based
                          trend.
                        </div>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={frameSeries}>
                            <XAxis
                              dataKey="frame_index"
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Frame index",
                                position: "insideBottom",
                                offset: -4,
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <YAxis
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Count",
                                angle: -90,
                                position: "insideLeft",
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "#020617",
                                border: "1px solid #1e293b",
                                borderRadius: 8,
                                fontSize: 11,
                              }}
                              labelStyle={{ color: "#e5e7eb" }}
                            />
                            {thresholdsActive && (
                              <>
                                <ReferenceLine
                                  y={parsedMin}
                                  stroke={GREEN}
                                  strokeDasharray="3 3"
                                />
                                <ReferenceLine
                                  y={parsedMax}
                                  stroke={RED}
                                  strokeDasharray="3 3"
                                />
                              </>
                            )}
                            <Line
                              type="monotone"
                              dataKey="count"
                              stroke={CYAN}
                              strokeWidth={2}
                              dot={(props) => renderColoredDot(props)}
                              activeDot={{ r: 5 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      )}
                    </div>

                    {showFrameDetails && thresholdsActive && (
                      <div style={styles.detailPanel}>
                        <div style={styles.detailSectionTitle}>
                          Alert frames (count &gt; max)
                        </div>
                        {getFrameAlerts().length === 0 ? (
                          <div>No alert frames.</div>
                        ) : (
                          getFrameAlerts().map((p, idx) => (
                            <div key={`fa-${idx}`}>
                              date = {p.date}, frame = {p.frame_index}, count ={" "}
                              {p.count.toFixed(3)}
                            </div>
                          ))
                        )}
                        <div style={styles.detailSectionTitle}>
                          Safe frames (between min &amp; max)
                        </div>
                        {getFrameSafe().length === 0 ? (
                          <div>No safe frames for current thresholds.</div>
                        ) : (
                          getFrameSafe().map((p, idx) => (
                            <div key={`fs-${idx}`}>
                              date = {p.date}, frame = {p.frame_index}, count ={" "}
                              {p.count.toFixed(3)}
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </div>
                </>
              ) : viewMode === "month" ? (
                <>
                  {/* MONTH-WISE: GRAPH 1 - Day vs avg count */}
                  {/* MONTH-WISE: GRAPH 1 - Day vs avg_count */}
                  <div style={styles.graphBox}>
                    <div style={styles.graphTitle}>
                      Month overview: average crowd per day
                    </div>
                    <div style={styles.graphSubtitle}>
                      Each point = one day. Useful for spotting consistently
                      busy or calm days in the selected month.
                    </div>

                    {/* NEW: details toggle button */}
                    <button
                      style={styles.detailBtn}
                      onClick={() => setShowMonthDetails((prev) => !prev)}
                    >
                      {showMonthDetails ? "Hide details" : "Show details"}
                    </button>

                    <div style={styles.graphArea}>
                      {monthDailyData.length === 0 ? (
                        <div style={styles.graphPlaceholder}>
                          Upload a CSV and pick a month to see day-wise
                          averages.
                        </div>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={monthDailyData}>
                            <XAxis
                              dataKey="day"
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Day of month",
                                position: "insideBottom",
                                offset: -4,
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <YAxis
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Average count",
                                angle: -90,
                                position: "insideLeft",
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "#020617",
                                border: "1px solid #1e293b",
                                borderRadius: 8,
                                fontSize: 11,
                              }}
                              labelStyle={{ color: "#e5e7eb" }}
                            />
                            <Line
                              type="monotone"
                              dataKey="avg_count"
                              stroke={BLUE}
                              strokeWidth={2}
                              dot={{ r: 3 }}
                              activeDot={{ r: 4 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      )}
                    </div>

                    {/* NEW: month details panel */}
                    {showMonthDetails && thresholdsActive && (
                      <div style={styles.detailPanel}>
                        <div style={styles.detailSectionTitle}>
                          Alert days (max &gt; max threshold)
                        </div>
                        {getMonthAlertDays().length === 0 ? (
                          <div>No alert days.</div>
                        ) : (
                          getMonthAlertDays().map((d, idx) => (
                            <div key={`md-alert-${idx}`}>
                              day = {d.day}, avg = {d.avg_count.toFixed(2)}, max ={" "}
                              {d.max_count.toFixed(2)}, total = {d.total_count.toFixed(2)}
                            </div>
                          ))
                        )}

                        <div style={styles.detailSectionTitle}>
                          Safe days (avg between min &amp; max)
                        </div>
                        {getMonthSafeDays().length === 0 ? (
                          <div>No safe days for current thresholds.</div>
                        ) : (
                          getMonthSafeDays().map((d, idx) => (
                            <div key={`md-safe-${idx}`}>
                              day = {d.day}, avg = {d.avg_count.toFixed(2)}, max ={" "}
                              {d.max_count.toFixed(2)}, total = {d.total_count.toFixed(2)}
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </div>


                  {/* MONTH-WISE: GRAPH 2 - Day vs max count */}
                  <div style={styles.graphBox}>
                    <div style={styles.graphTitle}>
                      Month overview: peak crowd per day
                    </div>
                    <div style={styles.graphSubtitle}>
                      Shows the highest count observed on each day of the
                      selected month.
                    </div>

                    {/* NEW: details toggle */}
                    <button
                      style={styles.detailBtn}
                      onClick={() => setShowMonthDetails2((prev) => !prev)}
                    >
                      {showMonthDetails2 ? "Hide details" : "Show details"}
                    </button>

                    <div style={styles.graphArea}>
                      {monthDailyData.length === 0 ? (
                        <div style={styles.graphPlaceholder}>
                          Upload a CSV and pick a month to see max counts per
                          day.
                        </div>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={monthDailyData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                              dataKey="day"
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Day of month",
                                position: "insideBottom",
                                offset: -4,
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <YAxis
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Max count",
                                angle: -90,
                                position: "insideLeft",
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "#020617",
                                border: "1px solid #1e293b",
                                borderRadius: 8,
                                fontSize: 11,
                              }}
                              labelStyle={{ color: "#e5e7eb" }}
                            />
                            <Legend />
                            <Bar dataKey="max_count" name="Max count">
                              {monthDailyData.map((entry, index) => {
                                const v = entry.max_count;
                                let fillColor = "#60a5fa";
                                if (thresholdsActive) {
                                  if (v > parsedMax) fillColor = RED;
                                  else if (
                                    v >= parsedMin &&
                                    v <= parsedMax
                                  )
                                    fillColor = GREEN;
                                  else fillColor = AMBER;
                                }
                                return (
                                  <Cell
                                    key={`mmax-${index}`}
                                    fill={fillColor}
                                  />
                                );
                              })}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      )}
                    </div>
                    {/* NEW: details panel */}
                    {showMonthDetails2 && thresholdsActive && (
                      <div style={styles.detailPanel}>
                        <div style={styles.detailSectionTitle}>
                          Alert days (max &gt; max threshold)
                        </div>
                        {getMonthAlertDays().length === 0 ? (
                          <div>No alert days.</div>
                        ) : (
                          getMonthAlertDays().map((d, idx) => (
                            <div key={`md2-alert-${idx}`}>
                              day = {d.day}, max = {d.max_count.toFixed(2)}, avg ={" "}
                              {d.avg_count.toFixed(2)}, total ={" "}
                              {d.total_count.toFixed(2)}
                            </div>
                          ))
                        )}

                        <div style={styles.detailSectionTitle}>
                          Safe days (avg between min &amp; max)
                        </div>
                        {getMonthSafeDays().length === 0 ? (
                          <div>No safe days for current thresholds.</div>
                        ) : (
                          getMonthSafeDays().map((d, idx) => (
                            <div key={`md2-safe-${idx}`}>
                              day = {d.day}, max = {d.max_count.toFixed(2)}, avg ={" "}
                              {d.avg_count.toFixed(2)}, total ={" "}
                              {d.total_count.toFixed(2)}
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </div>

                  {/* MONTH-WISE: GRAPH 3 - Day vs total count */}
                  <div style={styles.graphBox}>
                    <div style={styles.graphTitle}>
                      Month overview: total crowd signal per day
                    </div>
                    <div style={styles.graphSubtitle}>
                      Sum of all counts for each day – useful for load planning
                      and total crowd exposure.
                    </div>

                    {/* NEW: details toggle */}
                    <button
                      style={styles.detailBtn}
                      onClick={() => setShowMonthDetails3((prev) => !prev)}
                    >
                      {showMonthDetails3 ? "Hide details" : "Show details"}
                    </button>

                    <div style={styles.graphArea}>
                      {monthDailyData.length === 0 ? (
                        <div style={styles.graphPlaceholder}>
                          Upload a CSV and pick a month to see total signals per
                          day.
                        </div>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={monthDailyData}>
                            <XAxis
                              dataKey="day"
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Day of month",
                                position: "insideBottom",
                                offset: -4,
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <YAxis
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Total count (sum)",
                                angle: -90,
                                position: "insideLeft",
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "#020617",
                                border: "1px solid #1e293b",
                                borderRadius: 8,
                                fontSize: 11,
                              }}
                              labelStyle={{ color: "#e5e7eb" }}
                            />
                            <Line
                              type="monotone"
                              dataKey="total_count"
                              stroke={CYAN}
                              strokeWidth={2}
                              dot={{ r: 3 }}
                              activeDot={{ r: 4 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      )}
                    </div>

                    {/* NEW: details panel */}
                    {showMonthDetails3 && thresholdsActive && (
                      <div style={styles.detailPanel}>
                        <div style={styles.detailSectionTitle}>
                          Alert days (max &gt; max threshold)
                        </div>
                        {getMonthAlertDays().length === 0 ? (
                          <div>No alert days.</div>
                        ) : (
                          getMonthAlertDays().map((d, idx) => (
                            <div key={`md3-alert-${idx}`}>
                              day = {d.day}, total = {d.total_count.toFixed(2)}, max ={" "}
                              {d.max_count.toFixed(2)}, avg = {d.avg_count.toFixed(2)}
                            </div>
                          ))
                        )}

                        <div style={styles.detailSectionTitle}>
                          Safe days (avg between min &amp; max)
                        </div>
                        {getMonthSafeDays().length === 0 ? (
                          <div>No safe days for current thresholds.</div>
                        ) : (
                          getMonthSafeDays().map((d, idx) => (
                            <div key={`md3-safe-${idx}`}>
                              day = {d.day}, total = {d.total_count.toFixed(2)}, max ={" "}
                              {d.max_count.toFixed(2)}, avg = {d.avg_count.toFixed(2)}
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <>
                  {/* YEAR-WISE: GRAPH 1 - Month vs avg_count */}
                  <div style={styles.graphBox}>
                    <div style={styles.graphTitle}>
                      Year overview: average crowd per month
                    </div>
                    <div style={styles.graphSubtitle}>
                      Each point = one month. Helps you compare how busy months are within the selected year.
                    </div>

                    {/* NEW: details toggle button */}
                    <button
                      style={styles.detailBtn}
                      onClick={() => setShowYearDetails((prev) => !prev)}
                    >
                      {showYearDetails ? "Hide details" : "Show details"}
                    </button>

                    <div style={styles.graphArea}>
                      {yearMonthlyData.length === 0 ? (
                        <div style={styles.graphPlaceholder}>
                          Upload a CSV and pick a year to see month-wise averages.
                        </div>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={yearMonthlyData}>
                            <XAxis
                              dataKey="month"
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Month (1–12)",
                                position: "insideBottom",
                                offset: -4,
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <YAxis
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Average count",
                                angle: -90,
                                position: "insideLeft",
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "#020617",
                                border: "1px solid #1e293b",
                                borderRadius: 8,
                                fontSize: 11,
                              }}
                              labelStyle={{ color: "#e5e7eb" }}
                            />
                            <Line
                              type="monotone"
                              dataKey="avg_count"
                              stroke={BLUE}
                              strokeWidth={2}
                              dot={{ r: 3 }}
                              activeDot={{ r: 4 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      )}
                    </div>

                    {/* NEW: year details panel */}
                    {showYearDetails && thresholdsActive && (
                      <div style={styles.detailPanel}>
                        <div style={styles.detailSectionTitle}>
                          Alert months (max &gt; max threshold)
                        </div>
                        {getYearAlertMonths().length === 0 ? (
                          <div>No alert months.</div>
                        ) : (
                          getYearAlertMonths().map((m, idx) => (
                            <div key={`ym-alert-${idx}`}>
                              month = {m.monthLabel}, avg = {m.avg_count.toFixed(2)}, max ={" "}
                              {m.max_count.toFixed(2)}, total = {m.total_count.toFixed(2)}
                            </div>
                          ))
                        )}

                        <div style={styles.detailSectionTitle}>
                          Safe months (avg between min &amp; max)
                        </div>
                        {getYearSafeMonths().length === 0 ? (
                          <div>No safe months for current thresholds.</div>
                        ) : (
                          getYearSafeMonths().map((m, idx) => (
                            <div key={`ym-safe-${idx}`}>
                              month = {m.monthLabel}, avg = {m.avg_count.toFixed(2)}, max ={" "}
                              {m.max_count.toFixed(2)}, total = {m.total_count.toFixed(2)}
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </div>


                  {/* YEAR-WISE: GRAPH 2 - Month vs max_count */}
                  <div style={styles.graphBox}>
                    <div style={styles.graphTitle}>
                      Year overview: peak crowd per month
                    </div>
                    <div style={styles.graphSubtitle}>
                      Shows the highest count observed in each month of the selected year.
                    </div>

                    {/* NEW: details toggle */}
                    <button
                      style={styles.detailBtn}
                      onClick={() => setShowYearDetails2((prev) => !prev)}
                    >
                      {showYearDetails2 ? "Hide details" : "Show details"}
                    </button>

                    <div style={styles.graphArea}>
                      {yearMonthlyData.length === 0 ? (
                        <div style={styles.graphPlaceholder}>
                          Upload a CSV and pick a year to see monthly max crowd.
                        </div>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={yearMonthlyData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                              dataKey="month"
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Month (1–12)",
                                position: "insideBottom",
                                offset: -4,
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <YAxis
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Max count",
                                angle: -90,
                                position: "insideLeft",
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "#020617",
                                border: "1px solid #1e293b",
                                borderRadius: 8,
                                fontSize: 11,
                              }}
                              labelStyle={{ color: "#e5e7eb" }}
                            />
                            <Legend />
                            <Bar dataKey="max_count" name="Max count">
                              {yearMonthlyData.map((entry, index) => {
                                const v = entry.max_count;
                                let fillColor = "#60a5fa";
                                if (thresholdsActive) {
                                  if (v > parsedMax) fillColor = RED;
                                  else if (v >= parsedMin && v <= parsedMax)
                                    fillColor = GREEN;
                                  else fillColor = AMBER;
                                }
                                return <Cell key={`ymax-${index}`} fill={fillColor} />;
                              })}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      )}
                    </div>
                    {/* NEW: details panel */}
                    {showYearDetails2 && thresholdsActive && (
                      <div style={styles.detailPanel}>
                        <div style={styles.detailSectionTitle}>
                          Alert months (max &gt; max threshold)
                        </div>
                        {getYearAlertMonths().length === 0 ? (
                          <div>No alert months.</div>
                        ) : (
                          getYearAlertMonths().map((m, idx) => (
                            <div key={`ym2-alert-${idx}`}>
                              month = {m.monthLabel}, max = {m.max_count.toFixed(2)}, avg ={" "}
                              {m.avg_count.toFixed(2)}, total ={" "}
                              {m.total_count.toFixed(2)}
                            </div>
                          ))
                        )}

                        <div style={styles.detailSectionTitle}>
                          Safe months (avg between min &amp; max)
                        </div>
                        {getYearSafeMonths().length === 0 ? (
                          <div>No safe months for current thresholds.</div>
                        ) : (
                          getYearSafeMonths().map((m, idx) => (
                            <div key={`ym2-safe-${idx}`}>
                              month = {m.monthLabel}, max = {m.max_count.toFixed(2)}, avg ={" "}
                              {m.avg_count.toFixed(2)}, total ={" "}
                              {m.total_count.toFixed(2)}
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </div>

                  {/* YEAR-WISE: GRAPH 3 - Month vs total_count */}
                  <div style={styles.graphBox}>
                    <div style={styles.graphTitle}>
                      Year overview: total crowd signal per month
                    </div>
                    <div style={styles.graphSubtitle}>
                      Sum of all counts for each month – useful for annual planning and capacity checks.
                    </div>

                    {/* NEW: details toggle */}
                    <button
                      style={styles.detailBtn}
                      onClick={() => setShowYearDetails3((prev) => !prev)}
                    >
                      {showYearDetails3 ? "Hide details" : "Show details"}
                    </button>

                    <div style={styles.graphArea}>
                      {yearMonthlyData.length === 0 ? (
                        <div style={styles.graphPlaceholder}>
                          Upload a CSV and pick a year to see total signal per month.
                        </div>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={yearMonthlyData}>
                            <XAxis
                              dataKey="month"
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Month (1–12)",
                                position: "insideBottom",
                                offset: -4,
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <YAxis
                              tick={{ fill: "#93c5fd", fontSize: 11 }}
                              label={{
                                value: "Total count (sum)",
                                angle: -90,
                                position: "insideLeft",
                                fill: "#6b7280",
                                fontSize: 11,
                              }}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "#020617",
                                border: "1px solid #1e293b",
                                borderRadius: 8,
                                fontSize: 11,
                              }}
                              labelStyle={{ color: "#e5e7eb" }}
                            />
                            <Line
                              type="monotone"
                              dataKey="total_count"
                              stroke={CYAN}
                              strokeWidth={2}
                              dot={{ r: 3 }}
                              activeDot={{ r: 4 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      )}
                    </div>
                    {/* NEW: details panel */}
                    {showYearDetails3 && thresholdsActive && (
                      <div style={styles.detailPanel}>
                        <div style={styles.detailSectionTitle}>
                          Alert months (max &gt; max threshold)
                        </div>
                        {getYearAlertMonths().length === 0 ? (
                          <div>No alert months.</div>
                        ) : (
                          getYearAlertMonths().map((m, idx) => (
                            <div key={`ym3-alert-${idx}`}>
                              month = {m.monthLabel}, total ={" "}
                              {m.total_count.toFixed(2)}, max ={" "}
                              {m.max_count.toFixed(2)}, avg ={" "}
                              {m.avg_count.toFixed(2)}
                            </div>
                          ))
                        )}

                        <div style={styles.detailSectionTitle}>
                          Safe months (avg between min &amp; max)
                        </div>
                        {getYearSafeMonths().length === 0 ? (
                          <div>No safe months for current thresholds.</div>
                        ) : (
                          getYearSafeMonths().map((m, idx) => (
                            <div key={`ym3-safe-${idx}`}>
                              month = {m.monthLabel}, total ={" "}
                              {m.total_count.toFixed(2)}, max ={" "}
                              {m.max_count.toFixed(2)}, avg ={" "}
                              {m.avg_count.toFixed(2)}
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div >
    </>
  );
}

export default Dashboard;
