import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import CsvViewer from "../components/CsvViewer";

const API_BASE = "http://localhost:8000"; // backend URL

function useWindowWidth() {
  const [width, setWidth] = useState(window.innerWidth);
  useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);
  return width;
}

function MainPage({ user }) {
  const navigate = useNavigate();
  const [threshold, setThreshold] = useState("");
  const [liveCount, setLiveCount] = useState(null);
  const [analysisStarted, setAnalysisStarted] = useState(false);

  const [systemSettingsOpen, setSystemSettingsOpen] = useState(false);
  const systemSettingsRef = useRef(null);

  const [selectedSystemCamera, setSelectedSystemCamera] = useState("");

  const [crowdTxtMap, setCrowdTxtMap] = useState({});
  const [pollTimers, setPollTimers] = useState({});

  const width = useWindowWidth();
  const isMobile = width < 700;

  useEffect(() => {
    return () => {
      Object.values(pollTimers).forEach((id) => clearInterval(id));
    };
  }, [pollTimers]);

  // const styles = {
  //   page: {
  //     backgroundColor: "#020617",
  //     minHeight: "100vh",
  //     padding: isMobile ? "18px 4vw" : "32px 56px",
  //     fontFamily:
  //       "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  //     color: "#e5e7eb",
  //     display: "flex",
  //     flexDirection: "column",
  //     gap: isMobile ? 18 : 28,
  //     position: "relative",
  //     animation: "fadeInDown 0.6s",
  //     boxSizing: "border-box",
  //   },
  //   welcomeRow: {
  //     display: "flex",
  //     flexDirection: isMobile ? "column" : "row",
  //     justifyContent: "space-between",
  //     alignItems: isMobile ? "flex-start" : "center",
  //     marginBottom: isMobile ? 12 : 20,
  //     width: "100%",
  //   },
  //   brandTitle: {
  //     fontSize: isMobile ? 14 : 16,
  //     textTransform: "uppercase",
  //     letterSpacing: "0.22em",
  //     color: "#9ca3af",
  //     marginBottom: 4,
  //   },
  //   welcomeText: {
  //     fontSize: isMobile ? 20 : 26,
  //     fontWeight: 600,
  //     color: "#e5e7eb",
  //   },
  //   welcomeSubtext: {
  //     marginTop: 4,
  //     fontSize: 13,
  //     color: "#6b7280",
  //   },
  //   cameraPairsContainer: {
  //     display: "flex",
  //     flexDirection: "column",
  //     gap: isMobile ? 12 : 24,
  //     flexWrap: "nowrap",
  //     animation: "fadeIn 0.7s",
  //     width: "100%",
  //   },
  //   cameraPair: {
  //     display: isMobile ? "block" : "flex",
  //     gap: isMobile ? 12 : 20,
  //     alignItems: isMobile ? "stretch" : "flex-start",
  //     width: "100%",
  //     marginBottom: isMobile ? 10 : 0,
  //   },
  //   cameraCard: {
  //     backgroundColor: "#020617",
  //     borderRadius: 16,
  //     boxShadow: "0 16px 40px rgba(15,23,42,0.9)",
  //     overflow: "hidden",
  //     paddingBottom: 16,
  //     animation: "fadeInUp 0.7s",
  //     flex: isMobile ? "unset" : "1 1 50%",
  //     minWidth: isMobile ? "100%" : 360,
  //     position: "relative",
  //     border: "1px solid #1f2937",
  //     transition: "transform 0.2s ease, box-shadow 0.2s ease",
  //   },
  //   cameraTitle: {
  //     padding: isMobile ? "10px 12px 4px" : "14px 18px 4px",
  //     borderBottom: "1px solid #111827",
  //     backgroundColor: "#020617",
  //   },
  //   cameraNameInput: {
  //     width: "100%",
  //     padding: isMobile ? 7 : 9,
  //     fontSize: isMobile ? 13 : 15,
  //     borderRadius: 10,
  //     border: "1px solid #1f2937",
  //     backgroundColor: "#020617",
  //     color: "#e5e7eb",
  //     fontWeight: 500,
  //     outline: "none",
  //     boxSizing: "border-box",
  //   },
  //   cameraFrame: {
  //     width: "100%",
  //     height: isMobile ? 180 : 360,
  //     backgroundColor: "#020617",
  //     borderRadius: 14,
  //     display: "flex",
  //     justifyContent: "center",
  //     alignItems: "center",
  //     overflow: "hidden",
  //     position: "relative",
  //     marginTop: 8,
  //     marginBottom: 10,
  //     padding: 8,
  //     boxSizing: "border-box",
  //   },
  //   cameraVideo: {
  //     width: "100%",
  //     height: "100%",
  //     objectFit: "contain",
  //     borderRadius: "14px",
  //     backgroundColor: "#000",
  //   },
  //   csvBox: {
  //     width: "100%",
  //     height: "100%",
  //     borderRadius: 12,
  //     border: "1px solid #1f2937",
  //     backgroundColor: "#020617",
  //     padding: 10,
  //     boxSizing: "border-box",
  //     display: "flex",
  //     flexDirection: "column",
  //   },
  //   csvTitle: {
  //     fontSize: 13,
  //     fontWeight: 600,
  //     color: "#93c5fd",
  //     marginBottom: 6,
  //   },
  //   csvContent: {
  //     flex: 1,
  //     fontSize: 11,
  //     lineHeight: 1.4,
  //     color: "#e5e7eb",
  //     whiteSpace: "pre-wrap",
  //     overflowY: "auto",
  //     paddingRight: 4,
  //   },
  //   cameraSettings: {
  //     marginTop: 8,
  //     padding: isMobile ? "0 10px" : "0 18px",
  //     fontSize: 13,
  //   },
  //   toggleSwitch: {
  //     marginTop: 10,
  //     padding: isMobile ? "0 10px" : "0 18px",
  //     display: "flex",
  //     alignItems: "center",
  //     justifyContent: "space-between",
  //   },
  //   toggleLabel: {
  //     fontWeight: 500,
  //     fontSize: isMobile ? 13 : 14,
  //     color: "#e5e7eb",
  //   },
  //   deleteCameraBtn: {
  //     position: "absolute",
  //     top: 20,
  //     right: 20,
  //     backgroundColor: "#b91c1c",
  //     border: "1px solid #fecaca",
  //     borderRadius: 999,
  //     color: "#fee2e2",
  //     cursor: "pointer",
  //     padding: "6px 12px",
  //     fontWeight: 600,
  //     fontSize: 11,
  //     zIndex: 10,
  //   },
  //   buttonRow: {
  //     marginTop: isMobile ? 20 : 28,
  //     display: "flex",
  //     justifyContent: "center",
  //     gap: isMobile ? 10 : 16,
  //     flexWrap: "wrap",
  //     animation: "fadeInUp 0.7s",
  //   },
  //   actionButton: {
  //     backgroundColor: "#1d4ed8",
  //     color: "#e5e7eb",
  //     borderRadius: 999,
  //     padding: isMobile ? "11px 14px" : "12px 20px",
  //     fontSize: isMobile ? 14 : 15,
  //     fontWeight: 600,
  //     cursor: "pointer",
  //     border: "none",
  //     boxShadow: "0 12px 30px rgba(37,99,235,0.35)",
  //     letterSpacing: "0.03em",
  //     minWidth: isMobile ? "100%" : 210,
  //   },
  //   addCameraBtn: {
  //     marginTop: 16,
  //     backgroundColor: "#111827",
  //     color: "#e5e7eb",
  //     borderRadius: 999,
  //     padding: isMobile ? "10px 14px" : "11px 20px",
  //     fontSize: 14,
  //     fontWeight: 500,
  //     cursor: "pointer",
  //     border: "1px solid #374151",
  //     minWidth: isMobile ? "100%" : 260,
  //     alignSelf: "center",
  //   },
  //   sidePanel: {
  //     position: isMobile ? "static" : "fixed",
  //     top: isMobile ? undefined : 96,
  //     right: isMobile ? undefined : 28,
  //     width: isMobile ? "100%" : 340,
  //     backgroundColor: "#020617",
  //     boxShadow: "0 16px 45px rgba(15,23,42,0.95)",
  //     borderRadius: 20,
  //     padding: isMobile ? 14 : 18,
  //     zIndex: 90,
  //     animation: "fadeInUp 0.5s",
  //     color: "#e5e7eb",
  //     border: "1px solid #1f2937",
  //     boxSizing: "border-box",
  //     marginTop: isMobile ? 16 : 0,
  //   },
  //   sidePanelTitle: {
  //     fontWeight: 600,
  //     fontSize: isMobile ? 15 : 17,
  //     marginBottom: 10,
  //     color: "#93c5fd",
  //   },
  //   closeBtn: {
  //     marginTop: 10,
  //     width: "100%",
  //     backgroundColor: "#111827",
  //     color: "#e5e7eb",
  //     border: "1px solid #374151",
  //     borderRadius: 999,
  //     padding: "7px 10px",
  //     fontSize: 13,
  //     cursor: "pointer",
  //   },
  //   systemSelect: {
  //     width: "100%",
  //     padding: 8,
  //     borderRadius: 10,
  //     border: "1px solid #374151",
  //     backgroundColor: "#020617",
  //     color: "#e5e7eb",
  //     marginBottom: 12,
  //     fontSize: 13,
  //   },
  //   uploadInput: {
  //     marginTop: 4,
  //     marginBottom: 10,
  //     fontSize: 13,
  //     backgroundColor: "#111827",
  //   },
  // };


  const styles = {
  page: {
    background:
      "radial-gradient(circle at 0% 0%, rgba(56,189,248,0.08), transparent 55%), radial-gradient(circle at 100% 100%, rgba(37,99,235,0.16), transparent 55%), #020617",
    minHeight: "100vh",
    padding: isMobile ? "18px 4vw" : "32px 56px",
    fontFamily:
      "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    color: "#e5e7eb",
    display: "flex",
    flexDirection: "column",
    gap: isMobile ? 18 : 28,
    position: "relative",
    animation: "fadeInDown 0.6s",
    boxSizing: "border-box",
  },

  welcomeRow: {
    display: "flex",
    flexDirection: isMobile ? "column" : "row",
    justifyContent: "space-between",
    alignItems: isMobile ? "flex-start" : "center",
    marginBottom: isMobile ? 12 : 20,
    width: "100%",
  },
  brandTitle: {
    fontSize: isMobile ? 13 : 14,
    textTransform: "uppercase",
    letterSpacing: "0.24em",
    color: "#64748b",
    marginBottom: 4,
  },
  welcomeText: {
    fontSize: isMobile ? 20 : 26,
    fontWeight: 600,
    color: "#e5e7eb",
  },
  welcomeSubtext: {
    marginTop: 4,
    fontSize: 13,
    color: "#6b7280",
  },

  cameraPairsContainer: {
    display: "flex",
    flexDirection: "column",
    gap: isMobile ? 12 : 24,
    flexWrap: "nowrap",
    animation: "fadeIn 0.7s",
    width: "100%",
  },

  cameraPair: {
    display: isMobile ? "block" : "flex",
    gap: isMobile ? 12 : 20,
    alignItems: isMobile ? "stretch" : "flex-start",
    width: "100%",
    marginBottom: isMobile ? 10 : 0,
  },

  cameraCard: {
    background:
      "linear-gradient(135deg, rgba(15,23,42,0.98), rgba(15,23,42,1))",
    borderRadius: 18,
    boxShadow: "0 20px 52px rgba(15,23,42,0.98)",
    overflow: "hidden",
    paddingBottom: 16,
    animation: "fadeInUp 0.7s",
    flex: isMobile ? "unset" : "1 1 50%",
    minWidth: isMobile ? "100%" : 360,
    position: "relative",
    border: "1px solid rgba(148,163,184,0.30)",
    transition: "transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease",
  },

  cameraTitle: {
    padding: isMobile ? "10px 12px 4px" : "14px 18px 4px",
    borderBottom: "1px solid #111827",
    backgroundColor: "#020617",
  },

  cameraNameInput: {
    width: "100%",
    padding: isMobile ? 7 : 9,
    fontSize: isMobile ? 13 : 15,
    borderRadius: 10,
    border: "1px solid #1f2937",
    backgroundColor: "#020617",
    color: "#e5e7eb",
    fontWeight: 500,
    outline: "none",
    boxSizing: "border-box",
  },

  cameraFrame: {
    width: "100%",
    height: isMobile ? 190 : 360,
    backgroundColor: "#020617",
    borderRadius: 16,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    overflow: "hidden",
    position: "relative",
    marginTop: 8,
    marginBottom: 10,
    padding: 8,
    boxSizing: "border-box",
    border: "1px solid rgba(30,64,175,0.7)",
  },

  cameraVideo: {
    width: "100%",
    height: "100%",
    objectFit: "contain",
    borderRadius: "14px",
    backgroundColor: "#000",
  },

  csvBox: {
    width: "100%",
    height: "100%",
    borderRadius: 14,
    border: "1px solid rgba(51,65,85,0.9)",
    backgroundColor: "#020617",
    padding: 10,
    boxSizing: "border-box",
    display: "flex",
    flexDirection: "column",
  },

  csvTitle: {
    fontSize: 12,
    fontWeight: 600,
    color: "#93c5fd",
    marginBottom: 6,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
  },

  csvContent: {
    flex: 1,
    fontSize: 11,
    lineHeight: 1.5,
    color: "#e5e7eb",
    whiteSpace: "pre-wrap",
    overflowY: "auto",
    paddingRight: 4,
    fontFamily:
      "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
  },

  cameraSettings: {
    marginTop: 8,
    padding: isMobile ? "0 10px" : "0 18px",
    fontSize: 13,
    color: "#9ca3af",
  },

  toggleSwitch: {
    marginTop: 10,
    padding: isMobile ? "0 10px" : "0 18px",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },

  toggleLabel: {
    fontWeight: 500,
    fontSize: isMobile ? 13 : 14,
    color: "#e5e7eb",
  },

  deleteCameraBtn: {
    position: "absolute",
    top: 19,
    right: 21,
    backgroundColor: "#b91c1c",
    border: "1px solid #fecaca",
    borderRadius: 999,
    color: "#fee2e2",
    cursor: "pointer",
    padding: "5px 11px",
    fontWeight: 600,
    fontSize: 11,
    zIndex: 10,
  },

  buttonRow: {
    marginTop: isMobile ? 20 : 28,
    display: "flex",
    justifyContent: "center",
    gap: isMobile ? 10 : 16,
    flexWrap: "wrap",
    animation: "fadeInUp 0.7s",
  },

  actionButton: {
    background:
      "radial-gradient(circle at 0% 0%, #22d3ee, #2563eb)",
    color: "#e5e7eb",
    borderRadius: 999,
    padding: isMobile ? "11px 14px" : "12px 20px",
    fontSize: isMobile ? 14 : 15,
    fontWeight: 600,
    cursor: "pointer",
    border: "1px solid rgba(148,163,184,0.35)",
    boxShadow: "0 16px 38px rgba(15,23,42,0.95)",
    letterSpacing: "0.06em",
    minWidth: isMobile ? "100%" : 210,
  },

  addCameraBtn: {
    marginTop: 16,
    backgroundColor: "#020617",
    color: "#e5e7eb",
    borderRadius: 999,
    padding: isMobile ? "10px 14px" : "11px 20px",
    fontSize: 14,
    fontWeight: 500,
    cursor: "pointer",
    border: "1px solid #374151",
    minWidth: isMobile ? "100%" : 260,
    alignSelf: "center",
  },

  sidePanel: {
    position: isMobile ? "static" : "fixed",
    top: isMobile ? undefined : 96,
    right: isMobile ? undefined : 28,
    width: isMobile ? "100%" : 340,
    background:
      "linear-gradient(145deg, rgba(15,23,42,0.98), rgba(15,23,42,1))",
    boxShadow: "0 18px 50px rgba(15,23,42,0.98)",
    borderRadius: 20,
    padding: isMobile ? 14 : 18,
    zIndex: 90,
    animation: "fadeInUp 0.5s",
    color: "#e5e7eb",
    border: "1px solid rgba(148,163,184,0.30)",
    boxSizing: "border-box",
    marginTop: isMobile ? 16 : 0,
  },

  sidePanelTitle: {
    fontWeight: 600,
    fontSize: isMobile ? 15 : 17,
    marginBottom: 10,
    color: "#93c5fd",
    letterSpacing: "0.04em",
  },

  closeBtn: {
    marginTop: 10,
    width: "100%",
    backgroundColor: "#020617",
    color: "#e5e7eb",
    border: "1px solid #374151",
    borderRadius: 999,
    padding: "7px 10px",
    fontSize: 13,
    cursor: "pointer",
  },

  systemSelect: {
    width: "100%",
    padding: 8,
    borderRadius: 10,
    border: "1px solid #374151",
    backgroundColor: "#020617",
    color: "#e5e7eb",
    marginBottom: 12,
    fontSize: 13,
  },

  uploadInput: {
    marginTop: 4,
    marginBottom: 10,
    fontSize: 13,
    backgroundColor: "#111827",
  },
};

  const [cameraPairs, setCameraPairs] = useState([]);

  const [availableCameras] = useState([
    "Integrated Webcam",
    "USB Camera 1",
    "USB Camera 2",
    "Virtual Camera",
  ]);

  useEffect(() => {
    function handleClickOutside(event) {
      if (
        systemSettingsRef.current &&
        !systemSettingsRef.current.contains(event.target) &&
        systemSettingsOpen
      ) {
        setSystemSettingsOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [systemSettingsOpen]);

  const toggleCamera = (pairId, cameraId) => {
    setCameraPairs((pairs) =>
      pairs.map((pair) =>
        pair.pairId === pairId
          ? {
            ...pair,
            cameras: pair.cameras.map((cam) =>
              cam.id === cameraId ? { ...cam, on: !cam.on } : cam
            ),
          }
          : pair
      )
    );
  };

  const updateCameraName = (pairId, cameraId, newName) => {
    setCameraPairs((pairs) =>
      pairs.map((pair) =>
        pair.pairId === pairId
          ? {
            ...pair,
            cameras: pair.cameras.map((cam) =>
              cam.id === cameraId ? { ...cam, name: newName } : cam
            ),
          }
          : pair
      )
    );
  };

  const uploadVideoFile = async (pairId, cameraId, file) => {
    if (!file) return;

    setCameraPairs((pairs) =>
      pairs.map((pair) =>
        pair.pairId === pairId
          ? {
            ...pair,
            cameras: pair.cameras.map((cam) =>
              cam.id === cameraId
                ? { ...cam, src: URL.createObjectURL(file), uploadedFile: file }
                : cam
            ),
          }
          : pair
      )
    );
  };

  const startAnalysis = async (pairId, cameraId) => {
    const targetPair = cameraPairs.find((p) => p.pairId === pairId);
    const cam = targetPair.cameras.find((c) => c.id === cameraId);

    if (!cam.uploadedFile) {
      alert("Upload a video before starting analysis.");
      return;
    }
    if (!threshold) {
      alert("Please enter threshold before analysis.");
      return;
    }

    setAnalysisStarted(true);

    const formData = new FormData();
    formData.append("video", cam.uploadedFile);
    formData.append("threshold", threshold);   

    try {
      const res = await fetch(`${API_BASE}/analytics/process_video/`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!data.run_id) {
        console.error("No run_id in response", data);
        setAnalysisStarted(false);
        return;
      }

      const runId = data.run_id;
      pollResults(pairId, runId);
    } catch (err) {
      console.error("Error during analysis:", err);
    }

    setAnalysisStarted(false);
  };


  const pollResults = (pairId, runId) => {
    if (pollTimers[pairId]) clearInterval(pollTimers[pairId]);

    let intervalId = setInterval(() => {
      fetch(`${API_BASE}/analytics/crowd_txt/${runId}`)
        .then((r) => r.json())
        .then((data) => {
          if (!data || !data.csv) return;

          setCrowdTxtMap((prev) => ({
            ...prev,
            [pairId]: data.csv,
          }));

          const lines = data.csv.trim().split("\n");
          if (lines.length >= 2) {
            const lastRow = lines[lines.length - 1].split(",");
            const countVal = parseFloat(lastRow[2]);
            if (!isNaN(countVal)) setLiveCount(countVal);
          }

          if (data.done) clearInterval(intervalId);
        })
        .catch((err) => console.error("POLL ERROR:", err));
    }, 800);

    setPollTimers((prev) => ({ ...prev, [pairId]: intervalId }));
  };

  const addCameraPair = () => {
    const newPairId = cameraPairs.length
      ? cameraPairs[cameraPairs.length - 1].pairId + 1
      : 1;
    const baseCamId = cameraPairs.reduce(
      (max, pair) => Math.max(max, ...pair.cameras.map((c) => c.id)),
      0
    );
    const leftId = baseCamId + 1;
    const rightId = baseCamId + 2;

    setCameraPairs((prev) => [
      ...prev,
      {
        pairId: newPairId,
        cameras: [
          {
            id: leftId,
            name: `Camera ${newPairId}`,
            src: "",
            on: true,
            uploadedFile: null,
            role: "real",
          },
          {
            id: rightId,
            name: `Crowd Log ${newPairId}`,
            src: "",
            on: true,
            uploadedFile: null,
            role: "csv",
          },
        ],
      },
    ]);
  };

  const deleteCameraPair = (pairId) => {
    if (pollTimers[pairId]) {
      clearInterval(pollTimers[pairId]);
      setPollTimers((prev) => {
        const copy = { ...prev };
        delete copy[pairId];
        return copy;
      });
    }

    setCameraPairs((pairs) => pairs.filter((pair) => pair.pairId !== pairId));
    setCrowdTxtMap((prev) => {
      const copy = { ...prev };
      delete copy[pairId];
      return copy;
    });
  };

  const addSystemCamera = () => {
    if (!selectedSystemCamera) return;
    const newPairId = cameraPairs.length
      ? cameraPairs[cameraPairs.length - 1].pairId + 1
      : 1;
    const baseCamId = cameraPairs.reduce(
      (max, pair) => Math.max(max, ...pair.cameras.map((c) => c.id)),
      0
    );
    const leftId = baseCamId + 1;
    const rightId = baseCamId + 2;

    setCameraPairs((prev) => [
      ...prev,
      {
        pairId: newPairId,
        cameras: [
          {
            id: leftId,
            name: `${selectedSystemCamera} - Real`,
            src: "",
            on: true,
            uploadedFile: null,
            role: "real",
          },
          {
            id: rightId,
            name: `${selectedSystemCamera} - Crowd Log`,
            src: "",
            on: true,
            uploadedFile: null,
            role: "csv",
          },
        ],
      },
    ]);
    setSelectedSystemCamera("");
    setSystemSettingsOpen(false);
  };

  const downloadCsvForPair = (pairId, cameraName) => {
    const csvText = crowdTxtMap[pairId];
    if (!csvText) {
      alert("No CSV data available yet for this camera.");
      return;
    }

    const safeName =
      (cameraName || "crowd_output")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_") || "crowd_output";

    const blob = new Blob([csvText], {
      type: "text/csv;charset=utf-8;",
    });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `${safeName}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div style={styles.page}>
      <div style={styles.welcomeRow}>
        <div>
          <div style={styles.brandTitle}>VigilNet</div>
          <div style={styles.welcomeText}>Welcome to VigilNet</div>
          <div style={styles.welcomeSubtext}>
            Intelligent CCTV and crowd monitoring console
          </div>
        </div>
      </div>

      <div style={styles.cameraPairsContainer}>
        {cameraPairs.map((pair) => (
          <div key={pair.pairId} style={styles.cameraPair}>
            {pair.cameras.map((cam) => {
              const isCsvCard = cam.role === "csv";

              const isVideo =
                (cam.uploadedFile &&
                  cam.uploadedFile.type &&
                  cam.uploadedFile.type.startsWith("video")) ||
                (cam.src &&
                  (cam.src.endsWith(".mp4") ||
                    cam.src.endsWith(".webm") ||
                    cam.src.startsWith("blob:")));

              return (
                <div key={cam.id} style={styles.cameraCard}>
                  <div style={styles.cameraTitle}>
                    <input
                      type="text"
                      value={cam.name}
                      onChange={(e) =>
                        updateCameraName(pair.pairId, cam.id, e.target.value)
                      }
                      style={styles.cameraNameInput}
                    />
                  </div>

                  <div style={styles.cameraFrame}>
                    {cam.on ? (
                      isCsvCard ? (
                        <div style={styles.csvBox}>
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "space-between",
                              marginBottom: 6,
                            }}
                          >
                            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                              <div style={styles.csvTitle}>Crowd CSV (live)</div>

                              {/* <button
                                onClick={() => downloadCsvForPair(pair.pairId, cam.name)}
                                style={{
                                  backgroundColor: "#0ea5e9",
                                  color: "#0b1120",
                                  border: "none",
                                  borderRadius: "8px",
                                  padding: "4px 10px",
                                  fontSize: "12px",
                                  fontWeight: "600",
                                  marginLeft: "400px",
                                  cursor: "pointer",
                                }}
                              > 
                                Download CSV
                              </button> */}
                            </div>

                          </div>

                          <CsvViewer
                            csvText={crowdTxtMap[pair.pairId]}
                            fallbackMessage="Upload a video on the left to stream CSV from backend."
                            styleOverride={styles.csvContent}
                            threshold={threshold}
                          />
                        </div>
                      ) : (
                        <>
                          {!cam.src ? (
                            <div
                              style={{
                                color: "#6b7280",
                                fontSize: 13,
                                fontStyle: "italic",
                              }}
                            >
                              Upload a video to begin
                            </div>
                          ) : isVideo ? (
                            <video
                              key={cam.src}
                              autoPlay
                              muted
                              loop
                              playsInline
                              disablePictureInPicture
                              controls={false}
                              style={styles.cameraVideo}
                              src={cam.src}
                              onContextMenu={(e) => e.preventDefault()}
                              onPause={(e) => e.target.play()}
                            />
                          ) : (
                            <img
                              style={styles.cameraVideo}
                              alt={`${cam.name} Feed`}
                              src={cam.src}
                            />
                          )}
                        </>
                      )
                    ) : (
                      <div
                        style={{
                          color: "#6b7280",
                          fontStyle: "italic",
                          fontSize: 13,
                        }}
                      >
                        Camera Off
                      </div>
                    )}
                  </div>

                  <div style={styles.toggleSwitch}>
                    <label style={styles.toggleLabel}>{cam.name} On/Off</label>
                    <input
                      type="checkbox"
                      checked={cam.on}
                      onChange={() => toggleCamera(pair.pairId, cam.id)}
                    />
                  </div>

                  {cam.role === "real" && (
                    <div style={styles.cameraSettings}>
                      <label style={{ fontSize: 13 }}>
                        Upload Video:
                        <input
                          type="file"
                          accept="video/*"
                          onChange={(e) => {
                            if (e.target.files && e.target.files[0]) {
                              uploadVideoFile(
                                pair.pairId,
                                cam.id,
                                e.target.files[0]
                              );
                            }
                          }}
                          style={styles.uploadInput}
                        />

                        <div style={{ marginTop: "10px" }}>
                          <label
                            style={{
                              marginRight: "8px",
                              fontSize: "14px",
                              color: "#93c5fd",
                            }}
                          >
                            Threshold:
                          </label>

                          <input
                            type="number"
                            value={threshold}
                            onChange={(e) => setThreshold(e.target.value)}
                            placeholder="Enter threshold"
                            style={{
                              padding: "6px 10px",
                              border: "1px solid #374151",
                              borderRadius: "8px",
                              backgroundColor: "#0f172a",
                              color: "white",
                              fontSize: "13px",
                              width: "150px",
                            }}
                          />

                          <button
                            onClick={() => startAnalysis(pair.pairId, cam.id)}
                            style={{
                              marginLeft: "12px",
                              padding: "6px 16px",
                              backgroundColor: "#0ea5e9",
                              color: "#0b1120",
                              fontWeight: "600",
                              borderRadius: "10px",
                              cursor: "pointer",
                              border: "none",
                            }}
                          >
                            {analysisStarted ? "Processing..." : "Start Analysis"}
                          </button>
                        </div>
                      </label>
                    </div>
                  )}

                  {cam.role === "real" && (
                    <button
                      onClick={() => deleteCameraPair(pair.pairId)}
                      style={styles.deleteCameraBtn}
                      title="Delete this camera pair"
                    >
                      Delete
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      <button style={styles.addCameraBtn} onClick={addCameraPair}>
        + Add New Feed
      </button>

      <div style={styles.buttonRow}>
        <button
          style={styles.actionButton}
          onClick={() => setSystemSettingsOpen((v) => !v)}
        >
          System Settings
        </button>

        <button
          style={styles.actionButton}
          onClick={() => navigate("/dashboard")}
        >
          Analytics Dashboard
        </button>

        <button
          style={styles.actionButton}
          onClick={() => navigate("/crowd_count_photo")}
        >
          Crowd Counting from Photo
        </button>
      </div>

      {systemSettingsOpen && (
        <div style={styles.sidePanel} ref={systemSettingsRef}>
          <div style={styles.sidePanelTitle}>System Settings</div>

          <label style={{ fontWeight: 600, fontSize: 13 }}>
            Select Camera:
          </label>
          <select
            value={selectedSystemCamera}
            onChange={(e) => setSelectedSystemCamera(e.target.value)}
            style={styles.systemSelect}
          >
            <option value="">-- Choose a camera --</option>
            {availableCameras.map((cam, idx) => (
              <option key={idx} value={cam}>
                {cam}
              </option>
            ))}
          </select>

          <button
            disabled={!selectedSystemCamera}
            onClick={addSystemCamera}
            style={{
              ...styles.actionButton,
              width: "100%",
              opacity: selectedSystemCamera ? 1 : 0.5,
              marginBottom: 10,
            }}
          >
            Add Selected Camera
          </button>

          <button
            onClick={() => setSystemSettingsOpen(false)}
            style={styles.closeBtn}
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
}

export default MainPage;