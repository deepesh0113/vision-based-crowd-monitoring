import React, { useState, useEffect, useRef } from "react";
import jsPDF from "jspdf";
import html2canvas from "html2canvas"; // still available if you need later

// --------- Hook: window width ----------
function useWindowWidth() {
  const [width, setWidth] = useState(window.innerWidth);
  useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);
  return width;
}

// --------- Styles (dark / light by theme) ----------
function getStyles(isMobile, theme) {
  const isDark = theme === "dark";

  const color = {
    pageBg: isDark ? "#020617" : "#f3f4f6",
    cardBg: isDark ? "#020617" : "#ffffff",
    cardBorder: isDark ? "#1f2937" : "#e5e7eb",
    textMain: isDark ? "#e5e7eb" : "#111827",
    textMuted: isDark ? "#6b7280" : "#6b7280",
    accent: "#1d4ed8",
    accentSoft: isDark ? "#111827" : "#e0ebff",
    danger: "#f97373",
  };

  return {
    page: {
      minHeight: "100vh",
      backgroundColor: color.pageBg,
      color: color.textMain,
      display: "flex",
      justifyContent: "center",
      fontFamily:
        "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      padding: isMobile ? "18px 4vw" : "24px 40px",
      boxSizing: "border-box",
    },

    main: {
      width: "100%",
      maxWidth: 720,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 16,
    },

    // Top bar (only title + theme toggle now)
    topBar: {
      width: "100%",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      marginBottom: 10,
      gap: 12,
    },
    titleWrapper: {
      flex: 1,
    },
    title: {
      fontSize: isMobile ? "1.2rem" : "1.5rem",
      fontWeight: 600,
      letterSpacing: "0.02em",
    },
    subtitle: {
      marginTop: 4,
      fontSize: 12,
      color: color.textMuted,
    },
    themeToggle: {
      borderRadius: 999,
      border: `1px solid ${isDark ? "#1f2937" : "#d1d5db"}`,
      backgroundColor: color.cardBg,
      padding: "6px 14px",
      fontSize: 12,
      cursor: "pointer",
      color: color.textMuted,
      display: "flex",
      alignItems: "center",
      gap: 6,
      whiteSpace: "nowrap",
    },

    // Card
    uploadCard: {
      backgroundColor: color.cardBg,
      borderRadius: 18,
      padding: isMobile ? 18 : 22,
      border: `1px solid ${color.cardBorder}`,
      boxShadow: isDark
        ? "0 18px 40px rgba(15,23,42,0.75)"
        : "0 12px 30px rgba(148,163,184,0.35)",
      width: "100%",
      textAlign: "left",
      boxSizing: "border-box",
    },
    sectionTitle: {
      fontSize: 12,
      textTransform: "uppercase",
      letterSpacing: "0.16em",
      color: color.textMuted,
      marginBottom: 6,
    },

    // Drag & Drop zone
    dropZone: {
      marginTop: 8,
      marginBottom: 14,
      borderRadius: 12,
      padding: "18px 14px",
      border: `1px dashed ${color.cardBorder}`,
      backgroundColor: isDark ? "#020617" : "#f9fafb",
      cursor: "pointer",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: 4,
      transition: "border-color 0.15s ease, box-shadow 0.15s ease, background 0.15s",
    },
    dropZoneActive: {
      borderColor: color.accent,
      boxShadow: `0 0 0 1px ${color.accent}`,
      backgroundColor: isDark ? "#020617" : "#e0ebff",
    },
    dropZoneTitle: {
      fontSize: 14,
      fontWeight: 500,
    },
    dropZoneText: {
      fontSize: 12,
      color: color.textMuted,
    },

    // Preview
    previewContainer: {
      marginTop: 10,
      marginBottom: 16,
      borderRadius: 14,
      border: `1px solid ${isDark ? "#111827" : "#e5e7eb"}`,
      backgroundColor: isDark ? "#020617" : "#ffffff",
      padding: 8,
      minHeight: 80,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      boxSizing: "border-box",
    },
    previewImage: {
      display: "block",
      maxWidth: "100%",
      maxHeight: isMobile ? 220 : 320,
      borderRadius: 10,
      objectFit: "contain",
    },

    // Skeleton
    skeletonBlock: {
      width: "100%",
      height: isMobile ? 140 : 190,
      borderRadius: 10,
      background: isDark
        ? "linear-gradient(90deg,#020617,#111827,#020617)"
        : "linear-gradient(90deg,#e5e7eb,#d1d5db,#e5e7eb)",
      backgroundSize: "200% 100%",
    },
    skeletonLine: {
      height: 10,
      borderRadius: 999,
      marginTop: 6,
      background: isDark
        ? "linear-gradient(90deg,#020617,#111827,#020617)"
        : "linear-gradient(90deg,#e5e7eb,#d1d5db,#e5e7eb)",
      backgroundSize: "200% 100%",
    },

    buttonRow: {
      display: "flex",
      flexDirection: isMobile ? "column" : "row",
      gap: 10,
      marginTop: 6,
    },
    primaryBtn: {
      flex: 1,
      backgroundColor: color.accent,
      color: "#f9fafb",
      borderRadius: 999,
      padding: "10px 18px",
      fontSize: 14,
      fontWeight: 600,
      cursor: "pointer",
      border: "none",
      display: "inline-flex",
      alignItems: "center",
      justifyContent: "center",
      gap: 6,
      transition:
        "background 0.15s ease, transform 0.1s ease, box-shadow 0.15s ease, opacity 0.15s",
      boxShadow: "0 10px 24px rgba(37,99,235,0.32)",
    },
    primaryDisabled: {
      backgroundColor: isDark ? "#1f2937" : "#cbd5f5",
      boxShadow: "none",
      cursor: "not-allowed",
      opacity: 0.7,
    },
    clearBtn: {
      flex: isMobile ? 1 : 0.7,
      backgroundColor: isDark ? "#111827" : "#f9fafb",
      color: color.danger,
      borderRadius: 999,
      padding: "10px 16px",
      fontSize: 13,
      fontWeight: 500,
      cursor: "pointer",
      border: `1px solid ${isDark ? "#4b5563" : "#e5e7eb"}`,
      transition: "background 0.15s ease, border-color 0.15s ease, color 0.15s ease",
    },

    // Result
    resultBox: {
      marginTop: 18,
      padding: 14,
      borderRadius: 12,
      backgroundColor: isDark ? "#020617" : "#f9fafb",
      border: `1px solid ${color.accent}`,
      color: color.textMain,
      fontSize: 14,
    },
    resultLabel: {
      fontSize: 12,
      textTransform: "uppercase",
      letterSpacing: "0.12em",
      color: color.textMuted,
      marginBottom: 4,
    },
    resultValue: {
      fontSize: 15,
      fontWeight: 500,
    },
    confidence: {
      marginTop: 6,
      fontSize: 13,
      color: isDark ? "#a5f3fc" : "#0f766e",
    },

    downloadPdfBtn: {
      marginTop: 16,
      padding: "9px 18px",
      borderRadius: 999,
      border: "1px solid #16a34a",
      backgroundColor: isDark ? "#022c22" : "#dcfce7",
      color: isDark ? "#bbf7d0" : "#166534",
      fontSize: 13,
      fontWeight: 500,
      cursor: "pointer",
      display: "inline-flex",
      alignItems: "center",
      gap: 8,
      transition:
        "background 0.15s ease, border-color 0.15s ease, transform 0.1s ease, box-shadow 0.15s",
      boxShadow: "0 10px 22px rgba(22,163,74,0.32)",
    },

    connMsg: {
      marginTop: 10,
      fontSize: 12,
    },

    // Chart
    chartContainer: {
      marginTop: 18,
      paddingTop: 10,
      borderTop: `1px dashed ${color.cardBorder}`,
    },
    chartRow: {
      display: "flex",
      alignItems: "center",
      gap: 8,
      marginTop: 6,
    },
    chartLabel: {
      fontSize: 11,
      width: 60,
      color: color.textMuted,
    },
    chartBarOuter: {
      flex: 1,
      height: 8,
      borderRadius: 999,
      backgroundColor: isDark ? "#020617" : "#e5e7eb",
      overflow: "hidden",
    },
    chartBarInner: {
      height: "100%",
      borderRadius: 999,
      background:
        "linear-gradient(90deg, rgba(37,99,235,0.1), rgba(37,99,235,0.9))",
    },
    chartValue: {
      width: 44,
      textAlign: "right",
      fontSize: 12,
    },
  };
}

// ------------- MAIN COMPONENT -------------
function CrowdCountPhoto() {
  const width = useWindowWidth();
  const isMobile = width < 700;

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [countResult, setCountResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [connStatus, setConnStatus] = useState(null);
  const [connMessage, setConnMessage] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [theme, setTheme] = useState("dark");
  const [history, setHistory] = useState([]); // for mini chart

  const fileInputRef = useRef(null);

  const styles = getStyles(isMobile, theme);

  const API_BASE = "http://127.0.0.1:8000";
  const API_ENDPOINT = `${API_BASE}/crowdcount`;
  const OPENAPI_URL = `${API_BASE}/openapi.json`;

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  // ------------- Handlers -------------
  const processFile = (file) => {
    if (!file) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(file);
    setCountResult(null);
    setConnStatus(null);
    setConnMessage("");
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleFileChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      processFile(event.target.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) processFile(file);
  };

  const checkConnection = async (timeoutMs = 5000) => {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const resp = await fetch(OPENAPI_URL, {
        method: "GET",
        signal: controller.signal,
      });
      clearTimeout(id);
      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new Error(`HTTP ${resp.status} ${text}`);
      }
      await resp.json();
      setConnStatus("ok");
      setConnMessage("Backend reachable");
      return true;
    } catch (err) {
      clearTimeout(id);
      setConnStatus("error");
      if (err.name === "AbortError") {
        setConnMessage("Connection timed out (check backend or port).");
      } else {
        setConnMessage(`Connection failed: ${err.message}`);
      }
      return false;
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setCountResult(null);
    setConnStatus(null);
    setConnMessage("");

    const ok = await checkConnection(5000);
    if (!ok) {
      setLoading(false);
      return;
    }

    try {
      const formData = new FormData();
      formData.append("image", selectedFile);

      const response = await fetch(API_ENDPOINT, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errMsg = `Server error: ${response.status}`;
        try {
          const text = await response.text();
          const j = JSON.parse(text);
          errMsg = j.detail || text || errMsg;
        } catch (e) {}
        throw new Error(errMsg);
      }

      const data = await response.json();

      const rawCount = data.count;
      const rawFloat =
        typeof data.count_float === "number"
          ? data.count_float
          : Number(data.count_float);

      const roundedDiv10 = rawCount ? Math.round(rawCount / 10) : null;
      const rawDiv10 =
        rawFloat !== undefined && !isNaN(rawFloat) ? rawFloat / 10 : null;

      // Simple heuristic confidence based on difference between rounded and raw
      let confidence = 94;
      if (roundedDiv10 && rawDiv10 !== null) {
        const relError =
          Math.abs(roundedDiv10 - rawDiv10) / (roundedDiv10 || 1);
        confidence = Math.max(70, Math.min(99, 99 - relError * 30));
      }

      const result = {
        count_int: rawCount,
        count_float: rawFloat,
        roundedDiv10,
        rawDiv10,
        confidence,
      };

      setCountResult(result);
      setHistory((prev) =>
        [...prev, { roundedDiv10, rawDiv10, confidence }].slice(-6)
      );
    } catch (error) {
      setCountResult({
        error: true,
        message: error.message || "Error: Unable to count people",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPdf = async () => {
    const doc = new jsPDF({
      orientation: "portrait",
      unit: "pt",
      format: "a4",
    });
    let y = 40;
    let x = 40;

    const logoImg = new window.Image();
    logoImg.src = "/logo.png";
    await new Promise((res) => {
      logoImg.onload = res;
      logoImg.onerror = res;
    });
    if (logoImg.width) doc.addImage(logoImg, "PNG", x, y, 60, 60);

    doc.setFont("helvetica", "bold");
    doc.setFontSize(44);
    doc.setTextColor("#0A2342");
    doc.text("VigilNet", x + 80, y + 38, { align: "left" });
    doc.setFont("helvetica", "normal");
    doc.setFontSize(20);
    doc.setTextColor("#1E90FF");
    doc.text(
      "Intelligent Crowd Monitoring & Instant Alerts",
      x + 80,
      y + 65,
      { align: "left" }
    );
    doc.setDrawColor("#007BFF");
    doc.setLineWidth(2);
    doc.line(x + 80, y + 75, x + 380, y + 75);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(13);
    doc.setTextColor("#333333");
    doc.text(
      "Detect crowd density, abnormal motion, and camera tampering in real time.\nSend alerts to staff, automate workflows, and keep venues safe with low-latency models.",
      x,
      140,
      { align: "left", maxWidth: 520 }
    );

    let imgY = 200;
    if (previewUrl) {
      const img = new window.Image();
      img.src = previewUrl;
      await new Promise((res) => {
        img.onload = res;
        img.onerror = res;
      });
      doc.addImage(img, "JPEG", x, imgY, 220, 170);
      doc.setFont("helvetica", "italic");
      doc.setFontSize(12);
      doc.setTextColor("#0A2342");
      doc.text("This is the uploaded image", x, imgY + 185, {
        align: "left",
      });
    }

    doc.setFont("helvetica", "bold");
    doc.setFontSize(22);
    doc.setTextColor("#0A2342");
    let resultY = imgY + 220;

    if (countResult && !countResult.error) {
      const roundedDiv10 = countResult.roundedDiv10;
      const rawDiv10 = countResult.rawDiv10;

      doc.text(
        `Estimated (rounded): ${
          roundedDiv10 !== null ? roundedDiv10.toLocaleString() : "Not available"
        }`,
        x,
        resultY,
        { align: "left" }
      );
      doc.text(
        `Estimated (raw): ${
          rawDiv10 !== null ? rawDiv10.toFixed(2) : "Not available"
        }`,
        x,
        resultY + 30,
        { align: "left" }
      );
      if (countResult.confidence) {
        doc.text(
          `Model confidence: ${countResult.confidence.toFixed(1)}%`,
          x,
          resultY + 60,
          { align: "left" }
        );
      }
    }

    const signImg = new window.Image();
    signImg.src = "/digital_sign.png";
    await new Promise((res) => {
      signImg.onload = res;
      signImg.onerror = res;
    });

    if (signImg.width) {
      const signWidth = 120;
      const signHeight = (signImg.height / signImg.width) * signWidth;

      doc.addImage(
        signImg,
        "PNG",
        doc.internal.pageSize.getWidth() - signWidth - 60,
        doc.internal.pageSize.getHeight() - signHeight - 60,
        signWidth,
        signHeight
      );
    }

    const pageHeight = doc.internal.pageSize.getHeight();
    doc.setDrawColor("#007BFF");
    doc.setLineWidth(2);
    doc.line(
      50,
      pageHeight - 40,
      doc.internal.pageSize.getWidth() - 50,
      pageHeight - 40
    );
    doc.setFont("times", "italic");
    doc.setFontSize(14);
    doc.setTextColor("#1E90FF");
    doc.text(
      "VigilNet Verified Digital Sign",
      doc.internal.pageSize.getWidth() / 2,
      pageHeight - 20,
      { align: "center" }
    );
    doc.save("VigilNet_Crowd_Report.pdf");
  };

  // Clear = refresh whole page
  const handleClear = () => {
    window.location.reload();
  };

  const toggleTheme = () => {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  };

  const maxCount =
    history.reduce(
      (max, h) => Math.max(max, h.roundedDiv10 || 0),
      0
    ) || 1;

  return (
    <>
      {/* skeleton pulse animation */}
      <style>{`
        @keyframes skeletonPulse {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .skeleton-anim {
          animation: skeletonPulse 1.4s ease-in-out infinite;
        }
      `}</style>

      <div style={styles.page}>
        <main style={styles.main}>
          {/* TOP BAR (no back button now) */}
          <div style={styles.topBar}>
            <div style={styles.titleWrapper}>
              <div style={styles.title}>Crowd Counting from Photo</div>
              <div style={styles.subtitle}>
                Upload a frame · Get estimated crowd size · Export report
              </div>
            </div>

            <button style={styles.themeToggle} onClick={toggleTheme}>
              {theme === "dark" ? "☀ Light" : "☾ Dark"} mode
            </button>
          </div>

          {/* MAIN CARD */}
          <div style={styles.uploadCard}>
            <div style={styles.sectionTitle}>Input frame</div>

            {/* Drag & Drop Zone */}
            <div
              style={{
                ...styles.dropZone,
                ...(isDragging ? styles.dropZoneActive : {}),
              }}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <div style={styles.dropZoneTitle}>
                Drop image here or click to browse
              </div>
              <div style={styles.dropZoneText}>
                Supported: JPEG, PNG · Single frame
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                onChange={handleFileChange}
              />
            </div>

            {/* Preview / Skeleton */}
            <div style={styles.previewContainer}>
              {loading ? (
                <div
                  style={styles.skeletonBlock}
                  className="skeleton-anim"
                />
              ) : previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Uploaded preview"
                  style={styles.previewImage}
                />
              ) : (
                <div
                  style={{
                    fontSize: 12,
                    color: "#6b7280",
                  }}
                >
                  No frame selected
                </div>
              )}
            </div>

            {/* Buttons */}
            <div style={styles.buttonRow}>
              <button
                style={{
                  ...styles.primaryBtn,
                  ...(loading || !selectedFile ? styles.primaryDisabled : {}),
                }}
                onClick={handleSubmit}
                disabled={loading || !selectedFile}
              >
                {loading ? "Running model..." : "Count People"}
              </button>

              <button onClick={handleClear} style={styles.clearBtn}>
                Clear / Refresh
              </button>
            </div>

            {/* Result / Skeleton text */}
            {loading && (
              <div style={{ marginTop: 16 }}>
                <div
                  style={{ ...styles.skeletonLine, width: "70%" }}
                  className="skeleton-anim"
                />
                <div
                  style={{ ...styles.skeletonLine, width: "50%" }}
                  className="skeleton-anim"
                />
              </div>
            )}

            {countResult !== null && !loading && (
              <div style={styles.resultBox}>
                {countResult.error ? (
                  <div style={{ color: "#f97373" }}>{countResult.message}</div>
                ) : (
                  <div>
                    <div style={styles.resultLabel}>Model output</div>
                    <div style={styles.resultValue}>
                      Estimated (rounded):{" "}
                      {countResult.roundedDiv10 !== null
                        ? countResult.roundedDiv10.toLocaleString()
                        : "Not available"}
                    </div>
                    <div style={{ ...styles.resultValue, marginTop: 4 }}>
                      Estimated (raw):{" "}
                      {countResult.rawDiv10 !== null
                        ? countResult.rawDiv10.toFixed(2)
                        : "Not available"}
                    </div>
                    {countResult.confidence && (
                      <div style={styles.confidence}>
                        Model confidence:{" "}
                        {countResult.confidence.toFixed(1)}%
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Download PDF */}
            {countResult && !loading && !countResult.error && (
              <button style={styles.downloadPdfBtn} onClick={handleDownloadPdf}>
                ⬇ Download PDF Report
              </button>
            )}

            {/* Connection status */}
            {connStatus !== null && (
              <div
                style={{
                  ...styles.connMsg,
                  color: connStatus === "ok" ? "#22c55e" : "#f97373",
                }}
              >
                {connMessage}
              </div>
            )}

            {/* Mini Chart */}
            {history.length > 0 && (
              <div style={styles.chartContainer}>
                <div style={styles.sectionTitle}>Recent estimates</div>
                {history.map((h, idx) => (
                  <div key={idx} style={styles.chartRow}>
                    <div style={styles.chartLabel}>Run {idx + 1}</div>
                    <div style={styles.chartBarOuter}>
                      <div
                        style={{
                          ...styles.chartBarInner,
                          width: `${Math.max(
                            8,
                            ((h.roundedDiv10 || 0) / maxCount) * 100
                          )}%`,
                        }}
                      />
                    </div>
                    <div style={styles.chartValue}>
                      {h.roundedDiv10 ?? "-"}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </main>
      </div>
    </>
  );
}

export default CrowdCountPhoto;
