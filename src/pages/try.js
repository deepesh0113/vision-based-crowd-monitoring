// import React, { useState, useEffect } from "react";
// import { useNavigate } from "react-router-dom";
// import {
//   LineChart,
//   Line,
//   XAxis,
//   YAxis,
//   Tooltip,
//   ResponsiveContainer,
//   BarChart,
//   Bar,
//   CartesianGrid,
//   Legend,
//   Cell,
//   ReferenceLine,
// } from "recharts";

// function useWindowWidth() {
//   const [width, setWidth] = useState(window.innerWidth);
//   useEffect(() => {
//     const handleResize = () => setWidth(window.innerWidth);
//     window.addEventListener("resize", handleResize);
//     return () => window.removeEventListener("resize", handleResize);
//   }, []);
//   return width;
// }

// function Dashboard() {
//   const navigate = useNavigate();

//   const [csvFile, setCsvFile] = useState(null);

//   // main time series used by first graph
//   const [graphData, setGraphData] = useState([]);

//   // extra datasets from backend
//   const [timeSeries, setTimeSeries] = useState([]);      // time_sec vs count
//   const [perSecondData, setPerSecondData] = useState([]); // second vs avg_count
//   const [frameSeries, setFrameSeries] = useState([]);    // frame_index vs count
//   const [summary, setSummary] = useState(null);

//   const [loading, setLoading] = useState(false);
//   const [history, setHistory] = useState([]); // left-side history

//   // thresholds
//   const [minThreshold, setMinThreshold] = useState("");
//   const [maxThreshold, setMaxThreshold] = useState("");

//   // details toggles
//   const [showTimeDetails, setShowTimeDetails] = useState(false);
//   const [showPerSecondDetails, setShowPerSecondDetails] = useState(false);
//   const [showFrameDetails, setShowFrameDetails] = useState(false);

//   const width = useWindowWidth();
//   const isMobile = width < 700;

//   // parsed thresholds
//   const parsedMin = parseFloat(minThreshold);
//   const parsedMax = parseFloat(maxThreshold);
//   const thresholdsActive =
//     !Number.isNaN(parsedMin) && !Number.isNaN(parsedMax);

//   // Theme colors
//   const BG = "#020617";
//   const BLUE = "#3b82f6";
//   const CYAN = "#0ea5e9";
//   const GREEN = "#22c55e";
//   const RED = "#ef4444";
//   const AMBER = "#facc15";

//   const styles = {
//     page: {
//       minHeight: "100vh",
//       backgroundColor: BG,
//       fontFamily:
//         "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
//       display: "flex",
//       justifyContent: "center",
//       padding: isMobile ? "18px 4vw 32px" : "28px 28px 40px",
//       boxSizing: "border-box",
//     },
//     content: {
//       width: "100%",
//       maxWidth: 1120,
//       display: "flex",
//       flexDirection: "column",
//       gap: isMobile ? 16 : 20,
//       animation: "fadeIn 0.5s ease-in",
//     },
//     topBar: {
//       display: "flex",
//       alignItems: "center",
//       justifyContent: "flex-start",
//       gap: 16,
//       flexWrap: "wrap",
//     },
//     backBtn: {
//       borderRadius: 999,
//       border: "1px solid #1e293b",
//       backgroundColor: "transparent",
//       color: "#9ca3af",
//       padding: "7px 16px",
//       fontSize: 13,
//       cursor: "pointer",
//     },
//     brandBlock: {
//       display: "flex",
//       flexDirection: "column",
//       gap: 2,
//       textAlign: "left",
//     },
//     brand: {
//       fontSize: 11,
//       textTransform: "uppercase",
//       letterSpacing: "0.23em",
//       color: "#6b7280",
//     },
//     header: {
//       fontWeight: 600,
//       fontSize: isMobile ? "1.25rem" : "1.6rem",
//       color: "#e5e7eb",
//     },
//     subtitle: {
//       fontSize: 12,
//       color: "#9ca3af",
//     },

//     theoryCard: {
//       background: "radial-gradient(circle at top left, #020617, #020617 60%)",
//       borderRadius: 18,
//       padding: isMobile ? "14px 14px" : "18px 18px",
//       boxShadow: "0 24px 52px rgba(15,23,42,0.9)",
//       border: "1px solid #111827",
//     },
//     theoryTitle: {
//       fontSize: 14,
//       textTransform: "uppercase",
//       letterSpacing: "0.18em",
//       color: "#9ca3af",
//       marginBottom: 8,
//     },
//     theoryHeading: {
//       fontSize: 15,
//       fontWeight: 600,
//       color: "#e5e7eb",
//       marginBottom: 8,
//     },
//     theoryText: {
//       fontSize: 13,
//       color: "#9ca3af",
//       lineHeight: 1.65,
//     },
//     theoryList: {
//       marginTop: 10,
//       paddingLeft: 18,
//       fontSize: 13,
//       color: "#9ca3af",
//       lineHeight: 1.6,
//     },

//     mainRow: {
//       display: "flex",
//       flexDirection: isMobile ? "column" : "row",
//       gap: 18,
//       alignItems: "flex-start",
//     },
//     leftCol: {
//       flex: isMobile ? "unset" : "0 0 280px",
//       width: isMobile ? "100%" : 280,
//       display: "flex",
//       flexDirection: "column",
//       gap: 14,
//     },
//     rightCol: {
//       flex: 1,
//       display: "flex",
//       flexDirection: "column",
//       gap: 14,
//     },

//     historyCard: {
//       background: "radial-gradient(circle at top, #020617, #020617 70%)",
//       borderRadius: 18,
//       padding: isMobile ? "14px 14px" : "16px 16px",
//       boxShadow: "0 22px 50px rgba(15,23,42,0.9)",
//       border: "1px solid #111827",
//     },
//     historyTitle: {
//       fontSize: 14,
//       color: "#93c5fd",
//       fontWeight: 600,
//       marginBottom: 6,
//     },
//     historySub: {
//       fontSize: 12,
//       color: "#6b7280",
//       marginBottom: 10,
//     },
//     historyList: {
//       margin: 0,
//       padding: 0,
//       listStyle: "none",
//       maxHeight: 260,
//       overflowY: "auto",
//     },
//     historyItem: {
//       padding: "7px 0",
//       borderBottom: "1px solid rgba(15,23,42,0.9)",
//     },
//     historyName: {
//       fontSize: 13,
//       color: "#e5e7eb",
//       marginBottom: 2,
//       wordBreak: "break-all",
//     },
//     historyMeta: {
//       fontSize: 11,
//       color: "#9ca3af",
//     },
//     historyEmpty: {
//       fontSize: 12,
//       color: "#6b7280",
//       marginTop: 6,
//     },

//     graphBox: {
//       background: "radial-gradient(circle at top right, #020617, #020617 70%)",
//       borderRadius: 18,
//       border: "1px solid #111827",
//       boxShadow: "0 24px 52px rgba(15,23,42,0.95)",
//       padding: isMobile ? "16px 12px 20px" : "18px 18px 24px",
//       display: "flex",
//       flexDirection: "column",
//       gap: 10,
//     },
//     graphTitle: {
//       fontSize: 14,
//       color: BLUE,
//       fontWeight: 600,
//     },
//     graphSubtitle: {
//       fontSize: 12,
//       color: "#6b7280",
//     },
//     fileInput: {
//       marginTop: 8,
//       marginBottom: 10,
//       padding: 7,
//       borderRadius: 10,
//       background: BG,
//       color: "#e5e7eb",
//       border: "1px solid #111827",
//       fontSize: 13,
//     },
//     resetBtn: {
//       backgroundColor: "#10b981",
//       color: "#fff",
//       padding: "8px 20px",
//       borderRadius: 999,
//       cursor: "pointer",
//       border: "none",
//       fontWeight: 600,
//     },
//     actionBtn: {
//       backgroundColor: CYAN,
//       color: "#0b1120",
//       cursor: "pointer",
//       border: "none",
//       fontWeight: 600,
//       padding: "7px 16px",
//       borderRadius: 999,
//       fontSize: 13,
//     },
//     selectedFile: {
//       marginTop: 8,
//       fontSize: 12,
//       color: "#93c5fd",
//       wordBreak: "break-all",
//     },

//     graphArea: {
//       width: "100%",
//       height: 320,
//       marginTop: 4,
//     },
//     graphPlaceholder: {
//       flex: 1,
//       display: "flex",
//       alignItems: "center",
//       justifyContent: "center",
//       fontSize: 13,
//       color: CYAN,
//       borderRadius: 12,
//       border: "1px dashed #1e293b",
//     },
//     summaryRow: {
//       display: "flex",
//       flexWrap: "wrap",
//       gap: 10,
//       fontSize: 12,
//       color: "#9ca3af",
//       marginTop: 4,
//     },
//     summaryChip: {
//       padding: "4px 10px",
//       borderRadius: 999,
//       border: "1px solid #1e293b",
//     },
//     thresholdRow: {
//       display: "flex",
//       flexWrap: "wrap",
//       gap: 8,
//       marginTop: 10,
//       alignItems: "center",
//       fontSize: 12,
//       color: "#9ca3af",
//     },
//     thresholdInput: {
//       width: 90,
//       padding: "4px 8px",
//       borderRadius: 999,
//       border: "1px solid #1f2937",
//       backgroundColor: "#020617",
//       color: "#e5e7eb",
//       fontSize: 12,
//       outline: "none",
//     },
//     detailBtn: {
//       alignSelf: "flex-end",
//       marginTop: 4,
//       padding: "4px 10px",
//       fontSize: 11,
//       borderRadius: 999,
//       border: "1px solid #1f2937",
//       backgroundColor: "#020617",
//       color: "#9ca3af",
//       cursor: "pointer",
//     },
//     detailPanel: {
//       marginTop: 8,
//       padding: "8px 10px",
//       borderRadius: 12,
//       border: "1px solid #1f2937",
//       backgroundColor: "#020617",
//       fontSize: 11,
//       maxHeight: 160,
//       overflowY: "auto",
//       lineHeight: 1.5,
//     },
//     detailSectionTitle: {
//       fontWeight: 600,
//       marginTop: 4,
//       marginBottom: 2,
//       fontSize: 11,
//     },
//   };

//   // custom dot renderer for line charts based on thresholds
//   const renderColoredDot = (props) => {
//     const { cx, cy, payload } = props;
//     const value = payload.count;
//     let fill = "#60a5fa";

//     if (thresholdsActive) {
//       if (value > parsedMax) fill = RED; // alert
//       else if (value >= parsedMin && value <= parsedMax) fill = GREEN; // safe
//       else fill = AMBER; // below min
//     }

//     return (
//       <circle cx={cx} cy={cy} r={3} fill={fill} stroke={BG} strokeWidth={1} />
//     );
//   };

//   const handleCSVUpload = async () => {
//     if (!csvFile) {
//       alert("Please select a CSV file first!");
//       return;
//     }

//     const formData = new FormData();
//     formData.append("file", csvFile);

//     setLoading(true);

//     try {
//       const res = await fetch("http://127.0.0.1:8000/upload-csv", {
//         method: "POST",
//         body: formData,
//       });

//       const data = await res.json();

//       if (data.status === "error") {
//         alert(data.message || "Error processing CSV");
//         setLoading(false);
//         return;
//       }

//       setGraphData(data.records || []);
//       setTimeSeries(data.time_series || []);
//       setPerSecondData(data.per_second || []);
//       setFrameSeries(data.frame_series || []);
//       setSummary(data.summary || null);

//       setHistory((prev) => [
//         {
//           name: csvFile.name,
//           uploadedAt: new Date().toLocaleString(),
//           points: (data.records || []).length,
//         },
//         ...prev,
//       ]);
//     } catch (err) {
//       console.error("Error:", err);
//       alert("Failed to upload CSV");
//     }

//     setLoading(false);
//   };

//   const resetView = () => {
//     setCsvFile(null);
//     setGraphData([]);
//     setTimeSeries([]);
//     setPerSecondData([]);
//     setFrameSeries([]);
//     setSummary(null);
//     setMinThreshold("");
//     setMaxThreshold("");
//     setShowTimeDetails(false);
//     setShowPerSecondDetails(false);
//     setShowFrameDetails(false);
//     setTimeout(() => {
//       window.location.reload();
//     }, 150);
//   };

//   // helpers for detail panels
//   const getTimeAlerts = () =>
//     thresholdsActive
//       ? graphData.filter((p) => p.count > parsedMax)
//       : [];

//   const getTimeSafe = () =>
//     thresholdsActive
//       ? graphData.filter(
//         (p) => p.count >= parsedMin && p.count <= parsedMax
//       )
//       : [];

//   const getPerSecondAlerts = () =>
//     thresholdsActive
//       ? perSecondData.filter((p) => p.avg_count > parsedMax)
//       : [];

//   const getPerSecondSafe = () =>
//     thresholdsActive
//       ? perSecondData.filter(
//         (p) => p.avg_count >= parsedMin && p.avg_count <= parsedMax
//       )
//       : [];

//   const getFrameAlerts = () =>
//     thresholdsActive
//       ? frameSeries.filter((p) => p.count > parsedMax)
//       : [];

//   const getFrameSafe = () =>
//     thresholdsActive
//       ? frameSeries.filter(
//         (p) => p.count >= parsedMin && p.count <= parsedMax
//       )
//       : [];

//   return (
//     <>
//       <div style={styles.page}>
//         <div style={styles.content}>
//           {/* TOP BAR */}
//           <div style={styles.topBar}>
//             <button style={styles.backBtn} onClick={() => navigate(-1)}>
//               ← Back
//             </button>

//             <div style={styles.brandBlock}>
//               <div style={styles.brand}>VigilNet Dashboard</div>
//               <div style={styles.header}>Historical Crowd Analytics</div>
//               <div style={styles.subtitle}>
//                 Upload model outputs as CSV and explore time-based crowd trends
//                 for research, debugging, and reporting.
//               </div>
//             </div>
//           </div>

//           {/* THEORY CARD */}
//           <div style={styles.theoryCard}>
//             <div style={styles.theoryTitle}>Why this dashboard matters</div>
//             <div style={styles.theoryHeading}>
//               From raw CSV logs to decisions you can defend
//             </div>
//             <p style={styles.theoryText}>
//               VigilNet’s historical dashboard turns timestamp–count logs into a
//               visual narrative. Instead of scanning thousands of rows in Excel
//               or terminal prints, you get an immediate sense of where crowd
//               density spikes, how long risk levels persist, and whether your
//               models behave consistently across events and days.
//             </p>
//             <ul style={styles.theoryList}>
//               <li>
//                 <strong>Model evaluation:</strong> overlay time and crowd count
//                 to spot drift, undercounting, or saturation.
//               </li>
//               <li>
//                 <strong>Operational insight:</strong> identify peak windows,
//                 entry bottlenecks, and exit clearing times.
//               </li>
//               <li>
//                 <strong>Professional reporting:</strong> graphs exported from
//                 here can go directly into BTP reports, papers, or presentations.
//               </li>
//             </ul>
//           </div>

//           {/* MAIN ROW */}
//           <div style={styles.mainRow}>
//             {/* LEFT COLUMN: HISTORY */}
//             <div style={styles.leftCol}>
//               <div style={styles.historyCard}>
//                 <div style={styles.historyTitle}>Upload history</div>
//                 <div style={styles.historySub}>
//                   Recent CSV files processed on this dashboard.
//                 </div>

//                 {history.length === 0 ? (
//                   <div style={styles.historyEmpty}>
//                     No files analyzed yet. Upload a CSV to start building your
//                     history.
//                   </div>
//                 ) : (
//                   <ul style={styles.historyList}>
//                     {history.map((item, idx) => (
//                       <li key={idx} style={styles.historyItem}>
//                         <div style={styles.historyName}>{item.name}</div>
//                         <div style={styles.historyMeta}>
//                           {item.points} points · {item.uploadedAt}
//                         </div>
//                       </li>
//                     ))}
//                   </ul>
//                 )}
//               </div>
//             </div>

//             {/* RIGHT COLUMN: UPLOAD + GRAPHS */}
//             <div style={styles.rightCol}>
//               {/* CSV Upload + Thresholds */}
//               <div style={styles.graphBox}>
//                 <div style={styles.graphTitle}>Upload CSV file</div>
//                 <div style={styles.graphSubtitle}>
//                   Expected format:&nbsp;
//                   <code>timestamp_ns, frame_index, count</code>
//                 </div>

//                 <input
//                   type="file"
//                   accept=".csv"
//                   onChange={(e) => setCsvFile(e.target.files[0])}
//                   style={styles.fileInput}
//                 />

//                 <div style={{ display: "flex", gap: 10, marginTop: 5 }}>
//                   <button onClick={handleCSVUpload} style={styles.actionBtn}>
//                     {loading ? "Processing..." : "Upload & Process"}
//                   </button>


//                   <button onClick={resetView} style={styles.resetBtn}>
//                     Clear / Refresh
//                   </button>
//                 </div>


//                 {csvFile && (
//                   <div style={styles.selectedFile}>
//                     Selected file: {csvFile.name}
//                   </div>
//                 )}

//                 {/* Threshold controls */}
//                 <div style={styles.thresholdRow}>
//                   <span>Thresholds:</span>
//                   <span>Min (safe start)</span>
//                   <input
//                     type="number"
//                     step="0.01"
//                     value={minThreshold}
//                     onChange={(e) => setMinThreshold(e.target.value)}
//                     style={styles.thresholdInput}
//                     placeholder="e.g. 100"
//                   />
//                   <span>Max (alert)</span>
//                   <input
//                     type="number"
//                     step="0.01"
//                     value={maxThreshold}
//                     onChange={(e) => setMaxThreshold(e.target.value)}
//                     style={styles.thresholdInput}
//                     placeholder="e.g. 150"
//                   />
//                 </div>

//                 {summary && (
//                   <div style={styles.summaryRow}>
//                     <span style={styles.summaryChip}>
//                       Min count: {summary.min_count.toFixed(2)}
//                     </span>
//                     <span style={styles.summaryChip}>
//                       Max count: {summary.max_count.toFixed(2)}
//                     </span>
//                     <span style={styles.summaryChip}>
//                       Mean count: {summary.mean_count.toFixed(2)}
//                     </span>
//                     <span style={styles.summaryChip}>
//                       Points: {summary.num_points}
//                     </span>
//                   </div>
//                 )}
//               </div>

//               {/* GRAPH 1: Time-series trend */}
//               <div style={styles.graphBox}>
//                 <div style={styles.graphTitle}>
//                   Crowd trends (time-series)
//                 </div>
//                 <div style={styles.graphSubtitle}>
//                   time_sec vs count – green = within safe range, red = above
//                   alert threshold.
//                 </div>

//                 <button
//                   style={styles.detailBtn}
//                   onClick={() => setShowTimeDetails((prev) => !prev)}
//                 >
//                   {showTimeDetails ? "Hide details" : "Show details"}
//                 </button>

//                 <div style={styles.graphArea}>
//                   {graphData.length === 0 ? (
//                     <div style={styles.graphPlaceholder}>
//                       Upload a CSV file to render the time–series chart.
//                     </div>
//                   ) : (
//                     <ResponsiveContainer width="100%" height="100%">
//                       <LineChart data={graphData}>
//                         <XAxis
//                           dataKey="timestamp"
//                           tick={{ fill: "#93c5fd", fontSize: 11 }}
//                           label={{
//                             value: "Time (sec)",
//                             position: "insideBottom",
//                             offset: -4,
//                             fill: "#6b7280",
//                             fontSize: 11,
//                           }}
//                         />
//                         <YAxis
//                           tick={{ fill: "#93c5fd", fontSize: 11 }}
//                           label={{
//                             value: "Count",
//                             angle: -90,
//                             position: "insideLeft",
//                             fill: "#6b7280",
//                             fontSize: 11,
//                           }}
//                         />
//                         <Tooltip
//                           contentStyle={{
//                             backgroundColor: "#020617",
//                             border: "1px solid #1e293b",
//                             borderRadius: 8,
//                             fontSize: 11,
//                           }}
//                           labelStyle={{ color: "#e5e7eb" }}
//                         />
//                         {thresholdsActive && (
//                           <>
//                             <ReferenceLine
//                               y={parsedMin}
//                               stroke={GREEN}
//                               strokeDasharray="3 3"
//                               label={{
//                                 value: "Min safe",
//                                 fill: GREEN,
//                                 fontSize: 10,
//                               }}
//                             />
//                             <ReferenceLine
//                               y={parsedMax}
//                               stroke={RED}
//                               strokeDasharray="3 3"
//                               label={{
//                                 value: "Max alert",
//                                 fill: RED,
//                                 fontSize: 10,
//                               }}
//                             />
//                           </>
//                         )}
//                         <Line
//                           type="monotone"
//                           dataKey="count"
//                           stroke={BLUE}
//                           strokeWidth={2}
//                           dot={(props) => renderColoredDot(props)}
//                           activeDot={{ r: 5 }}
//                         />

//                       </LineChart>
//                     </ResponsiveContainer>
//                   )}
//                 </div>

//                 {showTimeDetails && thresholdsActive && (
//                   <div style={styles.detailPanel}>
//                     <div style={styles.detailSectionTitle}>
//                       Alert points (count &gt; max)
//                     </div>
//                     {getTimeAlerts().length === 0 ? (
//                       <div>No alert points.</div>
//                     ) : (
//                       getTimeAlerts().map((p, idx) => (
//                         <div key={`ta-${idx}`}>
//                           t = {p.timestamp}, count = {p.count.toFixed(3)}
//                         </div>
//                       ))
//                     )}
//                     <div style={styles.detailSectionTitle}>
//                       Safe points (between min &amp; max)
//                     </div>
//                     {getTimeSafe().length === 0 ? (
//                       <div>No safe points based on current thresholds.</div>
//                     ) : (
//                       getTimeSafe().map((p, idx) => (
//                         <div key={`ts-${idx}`}>
//                           t = {p.timestamp}, count = {p.count.toFixed(3)}
//                         </div>
//                       ))
//                     )}
//                   </div>
//                 )}
//               </div>

//               {/* GRAPH 2: Per-second average (bar chart) */}
//               <div style={styles.graphBox}>
//                 <div style={styles.graphTitle}>
//                   Per-second average crowd level
//                 </div>
//                 <div style={styles.graphSubtitle}>
//                   Bars show average count per whole second – green safe, red
//                   alert.
//                 </div>

//                 <button
//                   style={styles.detailBtn}
//                   onClick={() =>
//                     setShowPerSecondDetails((prev) => !prev)
//                   }
//                 >
//                   {showPerSecondDetails ? "Hide details" : "Show details"}
//                 </button>

//                 <div style={styles.graphArea}>
//                   {perSecondData.length === 0 ? (
//                     <div style={styles.graphPlaceholder}>
//                       Upload a CSV to see per-second averages.
//                     </div>
//                   ) : (
//                     <ResponsiveContainer width="100%" height="100%">
//                       <BarChart data={perSecondData}>
//                         <CartesianGrid strokeDasharray="3 3" />
//                         <XAxis
//                           dataKey="second"
//                           tick={{ fill: "#93c5fd", fontSize: 11 }}
//                           label={{
//                             value: "Second",
//                             position: "insideBottom",
//                             offset: -4,
//                             fill: "#6b7280",
//                             fontSize: 11,
//                           }}
//                         />
//                         <YAxis
//                           tick={{ fill: "#93c5fd", fontSize: 11 }}
//                           label={{
//                             value: "Avg Count",
//                             angle: -90,
//                             position: "insideLeft",
//                             fill: "#6b7280",
//                             fontSize: 11,
//                           }}
//                         />
//                         <Tooltip
//                           contentStyle={{
//                             backgroundColor: "#020617",
//                             border: "1px solid #1e293b",
//                             borderRadius: 8,
//                             fontSize: 11,
//                           }}
//                           labelStyle={{ color: "#e5e7eb" }}
//                         />
//                         <Legend />
//                         <Bar dataKey="avg_count" name="Avg count">
//                           {perSecondData.map((entry, index) => {
//                             const v = entry.avg_count;
//                             let fillColor = "#60a5fa";
//                             if (thresholdsActive) {
//                               if (v > parsedMax) fillColor = RED;
//                               else if (
//                                 v >= parsedMin &&
//                                 v <= parsedMax
//                               )
//                                 fillColor = GREEN;
//                               else fillColor = AMBER;
//                             }
//                             return (
//                               <Cell key={`cell-${index}`} fill={fillColor} />
//                             );
//                           })}
//                         </Bar>
//                         {thresholdsActive && (
//                           <>
//                             <ReferenceLine
//                               y={parsedMin}
//                               stroke={GREEN}
//                               strokeDasharray="3 3"
//                             />
//                             <ReferenceLine
//                               y={parsedMax}
//                               stroke={RED}
//                               strokeDasharray="3 3"
//                             />
//                           </>
//                         )}
//                       </BarChart>
//                     </ResponsiveContainer>
//                   )}
//                 </div>

//                 {showPerSecondDetails && thresholdsActive && (
//                   <div style={styles.detailPanel}>
//                     <div style={styles.detailSectionTitle}>
//                       Alert seconds (avg &gt; max)
//                     </div>
//                     {getPerSecondAlerts().length === 0 ? (
//                       <div>No alert seconds.</div>
//                     ) : (
//                       getPerSecondAlerts().map((p, idx) => (
//                         <div key={`pa-${idx}`}>
//                           second = {p.second}, avg ={" "}
//                           {p.avg_count.toFixed(3)}
//                         </div>
//                       ))
//                     )}
//                     <div style={styles.detailSectionTitle}>
//                       Safe seconds (between min &amp; max)
//                     </div>
//                     {getPerSecondSafe().length === 0 ? (
//                       <div>No safe seconds for current thresholds.</div>
//                     ) : (
//                       getPerSecondSafe().map((p, idx) => (
//                         <div key={`ps-${idx}`}>
//                           second = {p.second}, avg ={" "}
//                           {p.avg_count.toFixed(3)}
//                         </div>
//                       ))
//                     )}
//                   </div>
//                 )}
//               </div>

//               {/* GRAPH 3: Frame index vs count */}
//               <div style={styles.graphBox}>
//                 <div style={styles.graphTitle}>
//                   Frame index vs crowd count
//                 </div>
//                 <div style={styles.graphSubtitle}>
//                   Helps you see how model behaves as frames progress.
//                 </div>

//                 <button
//                   style={styles.detailBtn}
//                   onClick={() => setShowFrameDetails((prev) => !prev)}
//                 >
//                   {showFrameDetails ? "Hide details" : "Show details"}
//                 </button>

//                 <div style={styles.graphArea}>
//                   {frameSeries.length === 0 ? (
//                     <div style={styles.graphPlaceholder}>
//                       Upload a CSV to see frame-based trend.
//                     </div>
//                   ) : (
//                     <ResponsiveContainer width="100%" height="100%">
//                       <LineChart data={frameSeries}>
//                         <XAxis
//                           dataKey="frame_index"
//                           tick={{ fill: "#93c5fd", fontSize: 11 }}
//                           label={{
//                             value: "Frame index",
//                             position: "insideBottom",
//                             offset: -4,
//                             fill: "#6b7280",
//                             fontSize: 11,
//                           }}
//                         />
//                         <YAxis
//                           tick={{ fill: "#93c5fd", fontSize: 11 }}
//                           label={{
//                             value: "Count",
//                             angle: -90,
//                             position: "insideLeft",
//                             fill: "#6b7280",
//                             fontSize: 11,
//                           }}
//                         />
//                         <Tooltip
//                           contentStyle={{
//                             backgroundColor: "#020617",
//                             border: "1px solid #1e293b",
//                             borderRadius: 8,
//                             fontSize: 11,
//                           }}
//                           labelStyle={{ color: "#e5e7eb" }}
//                         />
//                         {thresholdsActive && (
//                           <>
//                             <ReferenceLine
//                               y={parsedMin}
//                               stroke={GREEN}
//                               strokeDasharray="3 3"
//                             />
//                             <ReferenceLine
//                               y={parsedMax}
//                               stroke={RED}
//                               strokeDasharray="3 3"
//                             />
//                           </>
//                         )}
//                         <Line
//                           type="monotone"
//                           dataKey="count"
//                           stroke={CYAN}
//                           strokeWidth={2}
//                           dot={(props) => renderColoredDot(props)}
//                           activeDot={{ r: 5 }}
//                         />

//                       </LineChart>
//                     </ResponsiveContainer>
//                   )}
//                 </div>

//                 {showFrameDetails && thresholdsActive && (
//                   <div style={styles.detailPanel}>
//                     <div style={styles.detailSectionTitle}>
//                       Alert frames (count &gt; max)
//                     </div>
//                     {getFrameAlerts().length === 0 ? (
//                       <div>No alert frames.</div>
//                     ) : (
//                       getFrameAlerts().map((p, idx) => (
//                         <div key={`fa-${idx}`}>
//                           frame = {p.frame_index}, count ={" "}
//                           {p.count.toFixed(3)}
//                         </div>
//                       ))
//                     )}
//                     <div style={styles.detailSectionTitle}>
//                       Safe frames (between min &amp; max)
//                     </div>
//                     {getFrameSafe().length === 0 ? (
//                       <div>No safe frames for current thresholds.</div>
//                     ) : (
//                       getFrameSafe().map((p, idx) => (
//                         <div key={`fs-${idx}`}>
//                           frame = {p.frame_index}, count ={" "}
//                           {p.count.toFixed(3)}
//                         </div>
//                       ))
//                     )}
//                   </div>
//                 )}
//               </div>
//             </div>
//           </div>
//         </div>
//       </div>
//     </>
//   );
// }

// export default Dashboard;


# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# import pandas as pd
# from io import BytesIO

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# def convert_timestamp_ns_to_seconds(ts_str: str):
#     """
#     Convert 'MM:SS:MS' like '00:00:400' -> seconds as float (0.4)
#     """
#     try:
#         parts = ts_str.split(":")
#         if len(parts) != 3:
#             return None
#         minutes = int(parts[0])
#         seconds = int(parts[1])
#         millis = int(parts[2])
#         total_seconds = minutes * 60 + seconds + millis / 1000.0
#         return total_seconds
#     except Exception:
#         return None


# @app.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...)):
#     """
#     CSV must contain: timestamp_ns, frame_index, count
#     Returns:
#       - records      -> used by your existing frontend graph
#       - time_series  -> time_sec vs count
#       - per_second   -> second vs avg_count
#       - frame_series -> frame_index vs count
#       - summary      -> stats
#     """
#     try:
#         raw = await file.read()
#         df = pd.read_csv(BytesIO(raw))

#         required_cols = {"timestamp_ns", "frame_index", "count"}
#         if not required_cols.issubset(df.columns):
#             return {
#                 "status": "error",
#                 "message": "CSV must contain columns: timestamp_ns, frame_index, count",
#                 "records": [],
#             }

#         # Convert timestamp_ns -> seconds
#         df["time_sec"] = df["timestamp_ns"].astype(str).apply(convert_timestamp_ns_to_seconds)
#         df = df.dropna(subset=["time_sec"])

#         # 1) time series: time_sec vs count
#         time_series_df = df[["time_sec", "count"]].copy()
#         time_series = time_series_df.to_dict(orient="records")

#         # 2) per-second avg (bucket)
#         df["sec_bucket"] = df["time_sec"].astype(int)
#         per_second_df = (
#             df.groupby("sec_bucket")["count"]
#             .mean()
#             .reset_index()
#             .rename(columns={"sec_bucket": "second", "count": "avg_count"})
#         )
#         per_second = per_second_df.to_dict(orient="records")

#         # 3) frame_index vs count
#         frame_series_df = df[["frame_index", "count"]].copy()
#         frame_series = frame_series_df.to_dict(orient="records")

#         # Summary
#         summary = {
#             "min_count": float(df["count"].min()),
#             "max_count": float(df["count"].max()),
#             "mean_count": float(df["count"].mean()),
#             "num_points": int(len(df)),
#         }

#         # ✅ Compatibility for existing frontend:
#         # make a `records` array with `timestamp` and `count`
#         records = [
#             {"timestamp": row["time_sec"], "count": row["count"]}
#             for row in time_series
#         ]

#         return {
#             "status": "success",
#             "records": records,          # <-- your React uses this
#             "time_series": time_series,
#             "per_second": per_second,
#             "frame_series": frame_series,
#             "summary": summary,
#         }

#     except Exception as e:
#         return {
#             "status": "error",
#             "message": f"Failed to process CSV: {str(e)}",
#             "records": [],
#         }




// after more
// import React, { useState, useEffect } from "react";
// import { useNavigate } from "react-router-dom";
// import {
//   LineChart,
//   Line,
//   XAxis,
//   YAxis,
//   Tooltip,
//   ResponsiveContainer,
//   BarChart,
//   Bar,
//   CartesianGrid,
//   Legend,
//   Cell,
//   ReferenceLine,
// } from "recharts";

// function useWindowWidth() {
//   const [width, setWidth] = useState(window.innerWidth);
//   useEffect(() => {
//     const handleResize = () => setWidth(window.innerWidth);
//     window.addEventListener("resize", handleResize);
//     return () => window.removeEventListener("resize", handleResize);
//   }, []);
//   return width;
// }

// function Dashboard() {
//   const navigate = useNavigate();

//   const [csvFile, setCsvFile] = useState(null);

//   // filtered data used by graphs
//   const [graphData, setGraphData] = useState([]); // time-series records
//   const [perSecondData, setPerSecondData] = useState([]);
//   const [frameSeries, setFrameSeries] = useState([]);
//   const [summary, setSummary] = useState(null);

//   // raw (unfiltered) data for date filtering
//   const [rawGraphData, setRawGraphData] = useState([]);
//   const [rawPerSecondData, setRawPerSecondData] = useState([]);
//   const [rawFrameSeries, setRawFrameSeries] = useState([]);

//   const [availableDates, setAvailableDates] = useState([]);
//   const [selectedDate, setSelectedDate] = useState("");

//   const [loading, setLoading] = useState(false);
//   const [history, setHistory] = useState([]); // left-side history
//   const [selectedHistory, setSelectedHistory] = useState(null);


//   // thresholds
//   const [minThreshold, setMinThreshold] = useState("");
//   const [maxThreshold, setMaxThreshold] = useState("");

//   // details toggles
//   const [showTimeDetails, setShowTimeDetails] = useState(false);
//   const [showPerSecondDetails, setShowPerSecondDetails] = useState(false);
//   const [showFrameDetails, setShowFrameDetails] = useState(false);

//   const width = useWindowWidth();
//   const isMobile = width < 700;

//   // parsed thresholds
//   const parsedMin = parseFloat(minThreshold);
//   const parsedMax = parseFloat(maxThreshold);
//   const thresholdsActive =
//     !Number.isNaN(parsedMin) && !Number.isNaN(parsedMax);

//   // Theme colors
//   const BG = "#020617";
//   const BLUE = "#3b82f6";
//   const CYAN = "#0ea5e9";
//   const GREEN = "#22c55e";
//   const RED = "#ef4444";
//   const AMBER = "#facc15";

//   const styles = {
//     page: {
//       minHeight: "100vh",
//       backgroundColor: BG,
//       fontFamily:
//         "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
//       display: "flex",
//       justifyContent: "center",
//       padding: isMobile ? "18px 4vw 32px" : "28px 28px 40px",
//       boxSizing: "border-box",
//     },
//     content: {
//       width: "100%",
//       maxWidth: 1120,
//       display: "flex",
//       flexDirection: "column",
//       gap: isMobile ? 16 : 20,
//       animation: "fadeIn 0.5s ease-in",
//     },
//     topBar: {
//       display: "flex",
//       alignItems: "center",
//       justifyContent: "flex-start",
//       gap: 16,
//       flexWrap: "wrap",
//     },
//     backBtn: {
//       borderRadius: 999,
//       border: "1px solid #1e293b",
//       backgroundColor: "transparent",
//       color: "#9ca3af",
//       padding: "7px 16px",
//       fontSize: 13,
//       cursor: "pointer",
//     },
//     brandBlock: {
//       display: "flex",
//       flexDirection: "column",
//       gap: 2,
//       textAlign: "left",
//     },
//     brand: {
//       fontSize: 11,
//       textTransform: "uppercase",
//       letterSpacing: "0.23em",
//       color: "#6b7280",
//     },
//     header: {
//       fontWeight: 600,
//       fontSize: isMobile ? "1.25rem" : "1.6rem",
//       color: "#e5e7eb",
//     },
//     subtitle: {
//       fontSize: 12,
//       color: "#9ca3af",
//     },

//     theoryCard: {
//       background: "radial-gradient(circle at top left, #020617, #020617 60%)",
//       borderRadius: 18,
//       padding: isMobile ? "14px 14px" : "18px 18px",
//       boxShadow: "0 24px 52px rgba(15,23,42,0.9)",
//       border: "1px solid #111827",
//     },
//     theoryTitle: {
//       fontSize: 14,
//       textTransform: "uppercase",
//       letterSpacing: "0.18em",
//       color: "#9ca3af",
//       marginBottom: 8,
//     },
//     theoryHeading: {
//       fontSize: 15,
//       fontWeight: 600,
//       color: "#e5e7eb",
//       marginBottom: 8,
//     },
//     theoryText: {
//       fontSize: 13,
//       color: "#9ca3af",
//       lineHeight: 1.65,
//     },
//     theoryList: {
//       marginTop: 10,
//       paddingLeft: 18,
//       fontSize: 13,
//       color: "#9ca3af",
//       lineHeight: 1.6,
//     },

//     mainRow: {
//       display: "flex",
//       flexDirection: isMobile ? "column" : "row",
//       gap: 18,
//       alignItems: "flex-start",
//     },
//     leftCol: {
//       flex: isMobile ? "unset" : "0 0 280px",
//       width: isMobile ? "100%" : 280,
//       display: "flex",
//       flexDirection: "column",
//       gap: 14,
//     },
//     rightCol: {
//       flex: 1,
//       display: "flex",
//       flexDirection: "column",
//       gap: 14,
//     },

//     historyCard: {
//       background: "radial-gradient(circle at top, #020617, #020617 70%)",
//       borderRadius: 18,
//       padding: isMobile ? "14px 14px" : "16px 16px",
//       boxShadow: "0 22px 50px rgba(15,23,42,0.9)",
//       border: "1px solid #111827",
//     },
//     historyTitle: {
//       fontSize: 14,
//       color: "#93c5fd",
//       fontWeight: 600,
//       marginBottom: 6,
//     },
//     historySub: {
//       fontSize: 12,
//       color: "#6b7280",
//       marginBottom: 10,
//     },
//     historyList: {
//       margin: 0,
//       padding: 0,
//       listStyle: "none",
//       maxHeight: 260,
//       overflowY: "auto",
//     },
//     historyItem: {
//       padding: "7px 0",
//       borderBottom: "1px solid rgba(15,23,42,0.9)",
//     },
//     historyName: {
//       fontSize: 13,
//       color: "#e5e7eb",
//       marginBottom: 2,
//       wordBreak: "break-all",
//     },
//     historyMeta: {
//       fontSize: 11,
//       color: "#9ca3af",
//     },
//     historyEmpty: {
//       fontSize: 12,
//       color: "#6b7280",
//       marginTop: 6,
//     },

//     graphBox: {
//       background: "radial-gradient(circle at top right, #020617, #020617 70%)",
//       borderRadius: 18,
//       border: "1px solid #111827",
//       boxShadow: "0 24px 52px rgba(15,23,42,0.95)",
//       padding: isMobile ? "16px 12px 20px" : "18px 18px 24px",
//       display: "flex",
//       flexDirection: "column",
//       gap: 10,
//     },
//     graphTitle: {
//       fontSize: 14,
//       color: BLUE,
//       fontWeight: 600,
//     },
//     graphSubtitle: {
//       fontSize: 12,
//       color: "#6b7280",
//     },
//     fileInput: {
//       marginTop: 8,
//       marginBottom: 10,
//       padding: 7,
//       borderRadius: 10,
//       background: BG,
//       color: "#e5e7eb",
//       border: "1px solid #111827",
//       fontSize: 13,
//     },
//     resetBtn: {
//       backgroundColor: "#10b981",
//       color: "#fff",
//       padding: "8px 20px",
//       borderRadius: 999,
//       cursor: "pointer",
//       border: "none",
//       fontWeight: 600,
//     },
//     actionBtn: {
//       backgroundColor: CYAN,
//       color: "#0b1120",
//       cursor: "pointer",
//       border: "none",
//       fontWeight: 600,
//       padding: "7px 16px",
//       borderRadius: 999,
//       fontSize: 13,
//     },
//     selectedFile: {
//       marginTop: 8,
//       fontSize: 12,
//       color: "#93c5fd",
//       wordBreak: "break-all",
//     },

//     graphArea: {
//       width: "100%",
//       height: 320,
//       marginTop: 4,
//     },
//     graphPlaceholder: {
//       flex: 1,
//       display: "flex",
//       alignItems: "center",
//       justifyContent: "center",
//       fontSize: 13,
//       color: CYAN,
//       borderRadius: 12,
//       border: "1px dashed #1e293b",
//     },
//     summaryRow: {
//       display: "flex",
//       flexWrap: "wrap",
//       gap: 10,
//       fontSize: 12,
//       color: "#9ca3af",
//       marginTop: 4,
//     },
//     summaryChip: {
//       padding: "4px 10px",
//       borderRadius: 999,
//       border: "1px solid #1e293b",
//     },
//     thresholdRow: {
//       display: "flex",
//       flexWrap: "wrap",
//       gap: 8,
//       marginTop: 10,
//       alignItems: "center",
//       fontSize: 12,
//       color: "#9ca3af",
//     },
//     thresholdInput: {
//       width: 90,
//       padding: "4px 8px",
//       borderRadius: 999,
//       border: "1px solid #1f2937",
//       backgroundColor: "#020617",
//       color: "#e5e7eb",
//       fontSize: 12,
//       outline: "none",
//     },
//     detailBtn: {
//       alignSelf: "flex-end",
//       marginTop: 4,
//       padding: "4px 10px",
//       fontSize: 11,
//       borderRadius: 999,
//       border: "1px solid #1f2937",
//       backgroundColor: "#020617",
//       color: "#9ca3af",
//       cursor: "pointer",
//     },
//     detailPanel: {
//       marginTop: 8,
//       padding: "8px 10px",
//       borderRadius: 12,
//       border: "1px solid #1f2937",
//       backgroundColor: "#020617",
//       fontSize: 11,
//       maxHeight: 160,
//       overflowY: "auto",
//       lineHeight: 1.5,
//     },
//     detailSectionTitle: {
//       fontWeight: 600,
//       marginTop: 4,
//       marginBottom: 2,
//       fontSize: 11,
//     },
//     dateSelect: {
//       padding: "4px 8px",
//       borderRadius: 999,
//       border: "1px solid #1f2937",
//       backgroundColor: "#020617",
//       color: "#e5e7eb",
//       fontSize: 12,
//       outline: "none",
//     },
//     filterBtn: {
//       backgroundColor: BLUE,
//       color: "#f9fafb",
//       border: "none",
//       borderRadius: 999,
//       padding: "6px 14px",
//       fontSize: 12,
//       cursor: "pointer",
//       fontWeight: 500,
//     },
//   };

//   // custom dot renderer for line charts based on thresholds
//   const renderColoredDot = (props) => {
//     const { cx, cy, payload } = props;
//     const value = payload.count;
//     let fill = "#60a5fa";

//     if (thresholdsActive) {
//       if (value > parsedMax) fill = RED; // alert
//       else if (value >= parsedMin && value <= parsedMax) fill = GREEN; // safe
//       else fill = AMBER; // below min
//     }

//     return (
//       <circle cx={cx} cy={cy} r={3} fill={fill} stroke={BG} strokeWidth={1} />
//     );
//   };

//   // apply date filter to all datasets
//   const applyDateFilter = (date, recordsSrc, perSecondSrc, frameSrc) => {
//     if (!date) {
//       setGraphData(recordsSrc);
//       setPerSecondData(perSecondSrc);
//       setFrameSeries(frameSrc);
//       return;
//     }

//     const filteredRecords = recordsSrc.filter((r) => r.date === date);
//     const filteredPerSecond = perSecondSrc.filter((r) => r.date === date);
//     const filteredFrame = frameSrc.filter((r) => r.date === date);

//     setGraphData(filteredRecords);
//     setPerSecondData(filteredPerSecond);
//     setFrameSeries(filteredFrame);
//   };

//   const handleCSVUpload = async () => {
//     if (!csvFile) {
//       alert("Please select a CSV file first!");
//       return;
//     }

//     const formData = new FormData();
//     formData.append("file", csvFile);

//     setLoading(true);

//     try {
//       const res = await fetch("http://127.0.0.1:8000/upload-csv", {
//         method: "POST",
//         body: formData,
//       });

//       const data = await res.json();

//       if (data.status === "error") {
//         alert(data.message || "Error processing CSV");
//         setLoading(false);
//         return;
//       }

//       const records = data.records || [];
//       const perSecondAll = data.per_second || [];
//       const frameAll = data.frame_series || [];

//       // store raw (unfiltered)
//       setRawGraphData(records);
//       setRawPerSecondData(perSecondAll);
//       setRawFrameSeries(frameAll);

//       // collect unique dates from records
//       const uniqueDates = Array.from(
//         new Set(records.map((r) => r.date).filter(Boolean))
//       );

//       setAvailableDates(uniqueDates);
//       const initialDate = uniqueDates[0] || "";
//       setSelectedDate(initialDate);

//       // apply initial filter (first date or all)
//       applyDateFilter(initialDate, records, perSecondAll, frameAll);

//       setSummary(data.summary || null);

//       // setHistory((prev) => [
//       //   {
//       //     name: csvFile.name,
//       //     uploadedAt: new Date().toLocaleString(),
//       //     points: records.length,
//       //   },
//       //   ...prev,
//       // ]);
//       // Make sure summary exists before saving to history
//       const minVal =
//         data.summary && typeof data.summary.min_count !== "undefined"
//           ? data.summary.min_count
//           : "—";

//       const maxVal =
//         data.summary && typeof data.summary.max_count !== "undefined"
//           ? data.summary.max_count
//           : "—";

//       // Build history item
//       const newHistoryItem = {
//         name: csvFile.name,
//         uploadedAt: new Date().toLocaleString(),
//         points: records.length,
//         minCount: minVal,
//         maxCount: maxVal,
//       };

//       // Save into history
//       setHistory((prev) => [newHistoryItem, ...prev]);

//       // Select it automatically
//       setSelectedHistory(newHistoryItem);


//       setHistory((prev) => [newHistoryItem, ...prev]);
//       setSelectedHistory(newHistoryItem);

//     } catch (err) {
//       console.error("Error:", err);
//       alert("Failed to upload CSV");
//     }

//     setLoading(false);
//   };

//   const resetView = () => {
//     setCsvFile(null);
//     setGraphData([]);
//     setPerSecondData([]);
//     setFrameSeries([]);
//     setSummary(null);
//     setMinThreshold("");
//     setMaxThreshold("");
//     setShowTimeDetails(false);
//     setShowPerSecondDetails(false);
//     setShowFrameDetails(false);

//     setRawGraphData([]);
//     setRawPerSecondData([]);
//     setRawFrameSeries([]);
//     setAvailableDates([]);
//     setSelectedDate("");
//     setTimeout(() => {
//       window.location.reload();
//     }, 150);
//   };

//   // helpers for detail panels
//   const getTimeAlerts = () =>
//     thresholdsActive ? graphData.filter((p) => p.count > parsedMax) : [];

//   const getTimeSafe = () =>
//     thresholdsActive
//       ? graphData.filter(
//         (p) => p.count >= parsedMin && p.count <= parsedMax
//       )
//       : [];

//   const getPerSecondAlerts = () =>
//     thresholdsActive
//       ? perSecondData.filter((p) => p.avg_count > parsedMax)
//       : [];

//   const getPerSecondSafe = () =>
//     thresholdsActive
//       ? perSecondData.filter(
//         (p) => p.avg_count >= parsedMin && p.avg_count <= parsedMax
//       )
//       : [];

//   const getFrameAlerts = () =>
//     thresholdsActive ? frameSeries.filter((p) => p.count > parsedMax) : [];

//   const getFrameSafe = () =>
//     thresholdsActive
//       ? frameSeries.filter(
//         (p) => p.count >= parsedMin && p.count <= parsedMax
//       )
//       : [];

//   return (
//     <>
//       <div style={styles.page}>
//         <div style={styles.content}>
//           {/* TOP BAR */}
//           <div style={styles.topBar}>
//             <button style={styles.backBtn} onClick={() => navigate(-1)}>
//               ← Back
//             </button>

//             <div style={styles.brandBlock}>
//               <div style={styles.brand}>VigilNet Dashboard</div>
//               <div style={styles.header}>Historical Crowd Analytics</div>
//               <div style={styles.subtitle}>
//                 Upload model outputs as CSV and explore time-based crowd trends
//                 for research, debugging, and reporting.
//               </div>
//             </div>
//           </div>

//           {/* THEORY CARD */}
//           <div style={styles.theoryCard}>
//             <div style={styles.theoryTitle}>Why this dashboard matters</div>
//             <div style={styles.theoryHeading}>
//               From raw CSV logs to decisions you can defend
//             </div>
//             <p style={styles.theoryText}>
//               VigilNet’s historical dashboard turns date + timestamp + count
//               logs into a visual narrative. Instead of scanning thousands of
//               rows in Excel or terminal prints, you get an immediate sense of
//               where crowd density spikes, how long risk levels persist, and
//               whether your models behave consistently across days and events.
//             </p>
//             <ul style={styles.theoryList}>
//               <li>
//                 <strong>Model evaluation:</strong> overlay time and crowd count
//                 to spot drift, undercounting, or saturation.
//               </li>
//               <li>
//                 <strong>Operational insight:</strong> identify peak windows,
//                 entry bottlenecks, and exit clearing times.
//               </li>
//               <li>
//                 <strong>Professional reporting:</strong> graphs exported from
//                 here can go directly into BTP reports, papers, or presentations.
//               </li>
//             </ul>
//           </div>

//           {/* MAIN ROW */}
//           <div style={styles.mainRow}>
//             {/* LEFT COLUMN: HISTORY */}
//             <div style={styles.leftCol}>
//               <div style={styles.historyCard}>
//                 <div style={styles.historyTitle}>Upload history</div>
//                 <div style={styles.historySub}>
//                   Recent CSV files processed on this dashboard.
//                 </div>

//                 {history.length === 0 ? (
//                   <div style={styles.historyEmpty}>
//                     No files analyzed yet. Upload a CSV to start building your
//                     history.
//                   </div>
//                 ) : (
//                   <>
//                     <ul style={styles.historyList}>
//                       {history.map((item, idx) => (
//                         <li
//                           key={idx}
//                           style={{
//                             ...styles.historyItem,
//                             cursor: "pointer",
//                             backgroundColor:
//                               selectedHistory === item
//                                 ? "rgba(15,23,42,0.8)"
//                                 : "transparent",
//                           }}
//                           onClick={() => setSelectedHistory(item)}
//                         >
//                           <div style={styles.historyName}>{item.name}</div>
//                           <div style={styles.historyMeta}>
//                             {item.points} points · {item.uploadedAt}
//                           </div>
//                         </li>
//                       ))}
//                     </ul>

//                     {selectedHistory && (
//                       <div
//                         style={{
//                           marginTop: 10,
//                           paddingTop: 8,
//                           borderTop: "1px solid rgba(15,23,42,0.9)",
//                           fontSize: 12,
//                           color: "#9ca3af",
//                         }}
//                       >
//                         <div>
//                           Min count:{" "}
//                           {selectedHistory.minCount?.toFixed
//                             ? selectedHistory.minCount.toFixed(2)
//                             : selectedHistory.minCount}
//                         </div>
//                         <div>
//                           Max count:{" "}
//                           {selectedHistory.maxCount?.toFixed
//                             ? selectedHistory.maxCount.toFixed(2)
//                             : selectedHistory.maxCount}
//                         </div>
//                         <div>Total points: {selectedHistory.points}</div>
//                       </div>
//                     )}
//                   </>
//                 )}
//               </div>
//             </div>


//             {/* RIGHT COLUMN: UPLOAD + GRAPHS */}
//             <div style={styles.rightCol}>
//               {/* CSV Upload + Thresholds + Date Filter */}
//               <div style={styles.graphBox}>
//                 <div style={styles.graphTitle}>Upload CSV file</div>
//                 <div style={styles.graphSubtitle}>
//                   Expected format:&nbsp;
//                   <code>date, timestamp_ns, frame_index, count</code>
//                 </div>

//                 <input
//                   type="file"
//                   accept=".csv"
//                   onChange={(e) => setCsvFile(e.target.files[0])}
//                   style={styles.fileInput}
//                 />

//                 <div style={{ display: "flex", gap: 10, marginTop: 5 }}>
//                   <button onClick={handleCSVUpload} style={styles.actionBtn}>
//                     {loading ? "Processing..." : "Upload & Process"}
//                   </button>

//                   <button onClick={resetView} style={styles.resetBtn}>
//                     Clear / Refresh
//                   </button>
//                 </div>

//                 {csvFile && (
//                   <div style={styles.selectedFile}>
//                     Selected file: {csvFile.name}
//                   </div>
//                 )}

//                 {/* Threshold controls */}
//                 <div style={styles.thresholdRow}>
//                   <span>Thresholds:</span>
//                   <span>Min (safe start)</span>
//                   <input
//                     type="number"
//                     step="0.01"
//                     value={minThreshold}
//                     onChange={(e) => setMinThreshold(e.target.value)}
//                     style={styles.thresholdInput}
//                     placeholder="e.g. 100"
//                   />
//                   <span>Max (alert)</span>
//                   <input
//                     type="number"
//                     step="0.01"
//                     value={maxThreshold}
//                     onChange={(e) => setMaxThreshold(e.target.value)}
//                     style={styles.thresholdInput}
//                     placeholder="e.g. 150"
//                   />
//                 </div>

//                 {/* Date filter */}
//                 {availableDates.length > 0 && (
//                   <div style={styles.thresholdRow}>
//                     <span>Filter by date:</span>
//                     <select
//                       value={selectedDate}
//                       onChange={(e) => {
//                         const value = e.target.value;
//                         setSelectedDate(value);
//                         applyDateFilter(
//                           value,
//                           rawGraphData,
//                           rawPerSecondData,
//                           rawFrameSeries
//                         );
//                       }}
//                       style={styles.dateSelect}
//                     >
//                       <option value="">All Month</option>
//                       {availableDates.map((d) => (
//                         <option key={d} value={d}>
//                           {d}
//                         </option>
//                       ))}
//                     </select>
//                     <button
//                       style={styles.filterBtn}
//                       onClick={() =>
//                         applyDateFilter(
//                           selectedDate,
//                           rawGraphData,
//                           rawPerSecondData,
//                           rawFrameSeries
//                         )
//                       }
//                     >
//                       FILTER
//                     </button>
//                   </div>
//                 )}

//                 {summary && (
//                   <div style={styles.summaryRow}>
//                     <span style={styles.summaryChip}>
//                       Min count: {summary.min_count.toFixed(2)}
//                     </span>
//                     <span style={styles.summaryChip}>
//                       Max count: {summary.max_count.toFixed(2)}
//                     </span>
//                     <span style={styles.summaryChip}>
//                       Mean count: {summary.mean_count.toFixed(2)}
//                     </span>
//                     <span style={styles.summaryChip}>
//                       Points: {summary.num_points}
//                     </span>
//                   </div>
//                 )}
//               </div>

//               {/* GRAPH 1: Time-series trend */}
//               <div style={styles.graphBox}>
//                 <div style={styles.graphTitle}>
//                   Crowd trends (time-series)
//                 </div>
//                 <div style={styles.graphSubtitle}>
//                   time_sec vs count – green = within safe range, red = above
//                   alert threshold.
//                 </div>

//                 <button
//                   style={styles.detailBtn}
//                   onClick={() => setShowTimeDetails((prev) => !prev)}
//                 >
//                   {showTimeDetails ? "Hide details" : "Show details"}
//                 </button>

//                 <div style={styles.graphArea}>
//                   {graphData.length === 0 ? (
//                     <div style={styles.graphPlaceholder}>
//                       Upload a CSV file to render the time–series chart.
//                     </div>
//                   ) : (
//                     <ResponsiveContainer width="100%" height="100%">
//                       <LineChart data={graphData}>
//                         <XAxis
//                           dataKey="timestamp"
//                           tick={{ fill: "#93c5fd", fontSize: 11 }}
//                           label={{
//                             value: "Time (sec)",
//                             position: "insideBottom",
//                             offset: -4,
//                             fill: "#6b7280",
//                             fontSize: 11,
//                           }}
//                         />
//                         <YAxis
//                           tick={{ fill: "#93c5fd", fontSize: 11 }}
//                           label={{
//                             value: "Count",
//                             angle: -90,
//                             position: "insideLeft",
//                             fill: "#6b7280",
//                             fontSize: 11,
//                           }}
//                         />
//                         <Tooltip
//                           contentStyle={{
//                             backgroundColor: "#020617",
//                             border: "1px solid #1e293b",
//                             borderRadius: 8,
//                             fontSize: 11,
//                           }}
//                           labelStyle={{ color: "#e5e7eb" }}
//                         />
//                         {thresholdsActive && (
//                           <>
//                             <ReferenceLine
//                               y={parsedMin}
//                               stroke={GREEN}
//                               strokeDasharray="3 3"
//                               label={{
//                                 value: "Min safe",
//                                 fill: GREEN,
//                                 fontSize: 10,
//                               }}
//                             />
//                             <ReferenceLine
//                               y={parsedMax}
//                               stroke={RED}
//                               strokeDasharray="3 3"
//                               label={{
//                                 value: "Max alert",
//                                 fill: RED,
//                                 fontSize: 10,
//                               }}
//                             />
//                           </>
//                         )}
//                         <Line
//                           type="monotone"
//                           dataKey="count"
//                           stroke={BLUE}
//                           strokeWidth={2}
//                           dot={(props) => renderColoredDot(props)}
//                           activeDot={{ r: 5 }}
//                         />
//                       </LineChart>
//                     </ResponsiveContainer>
//                   )}
//                 </div>

//                 {showTimeDetails && thresholdsActive && (
//                   <div style={styles.detailPanel}>
//                     <div style={styles.detailSectionTitle}>
//                       Alert points (count &gt; max)
//                     </div>
//                     {getTimeAlerts().length === 0 ? (
//                       <div>No alert points.</div>
//                     ) : (
//                       getTimeAlerts().map((p, idx) => (
//                         <div key={`ta-${idx}`}>
//                           date = {p.date}, t = {p.timestamp}, count ={" "}
//                           {p.count.toFixed(3)}
//                         </div>
//                       ))
//                     )}
//                     <div style={styles.detailSectionTitle}>
//                       Safe points (between min &amp; max)
//                     </div>
//                     {getTimeSafe().length === 0 ? (
//                       <div>No safe points based on current thresholds.</div>
//                     ) : (
//                       getTimeSafe().map((p, idx) => (
//                         <div key={`ts-${idx}`}>
//                           date = {p.date}, t = {p.timestamp}, count ={" "}
//                           {p.count.toFixed(3)}
//                         </div>
//                       ))
//                     )}
//                   </div>
//                 )}
//               </div>

//               {/* GRAPH 2: Per-second average (bar chart) */}
//               <div style={styles.graphBox}>
//                 <div style={styles.graphTitle}>
//                   Per-second average crowd level
//                 </div>
//                 <div style={styles.graphSubtitle}>
//                   Bars show average count per whole second – green safe, red
//                   alert.
//                 </div>

//                 <button
//                   style={styles.detailBtn}
//                   onClick={() =>
//                     setShowPerSecondDetails((prev) => !prev)
//                   }
//                 >
//                   {showPerSecondDetails ? "Hide details" : "Show details"}
//                 </button>

//                 <div style={styles.graphArea}>
//                   {perSecondData.length === 0 ? (
//                     <div style={styles.graphPlaceholder}>
//                       Upload a CSV to see per-second averages.
//                     </div>
//                   ) : (
//                     <ResponsiveContainer width="100%" height="100%">
//                       <BarChart data={perSecondData}>
//                         <CartesianGrid strokeDasharray="3 3" />
//                         <XAxis
//                           dataKey="second"
//                           tick={{ fill: "#93c5fd", fontSize: 11 }}
//                           label={{
//                             value: "Second",
//                             position: "insideBottom",
//                             offset: -4,
//                             fill: "#6b7280",
//                             fontSize: 11,
//                           }}
//                         />
//                         <YAxis
//                           tick={{ fill: "#93c5fd", fontSize: 11 }}
//                           label={{
//                             value: "Avg Count",
//                             angle: -90,
//                             position: "insideLeft",
//                             fill: "#6b7280",
//                             fontSize: 11,
//                           }}
//                         />
//                         <Tooltip
//                           contentStyle={{
//                             backgroundColor: "#020617",
//                             border: "1px solid #1e293b",
//                             borderRadius: 8,
//                             fontSize: 11,
//                           }}
//                           labelStyle={{ color: "#e5e7eb" }}
//                         />
//                         <Legend />
//                         <Bar dataKey="avg_count" name="Avg count">
//                           {perSecondData.map((entry, index) => {
//                             const v = entry.avg_count;
//                             let fillColor = "#60a5fa";
//                             if (thresholdsActive) {
//                               if (v > parsedMax) fillColor = RED;
//                               else if (v >= parsedMin && v <= parsedMax)
//                                 fillColor = GREEN;
//                               else fillColor = AMBER;
//                             }
//                             return (
//                               <Cell key={`cell-${index}`} fill={fillColor} />
//                             );
//                           })}
//                         </Bar>
//                         {thresholdsActive && (
//                           <>
//                             <ReferenceLine
//                               y={parsedMin}
//                               stroke={GREEN}
//                               strokeDasharray="3 3"
//                             />
//                             <ReferenceLine
//                               y={parsedMax}
//                               stroke={RED}
//                               strokeDasharray="3 3"
//                             />
//                           </>
//                         )}
//                       </BarChart>
//                     </ResponsiveContainer>
//                   )}
//                 </div>

//                 {showPerSecondDetails && thresholdsActive && (
//                   <div style={styles.detailPanel}>
//                     <div style={styles.detailSectionTitle}>
//                       Alert seconds (avg &gt; max)
//                     </div>
//                     {getPerSecondAlerts().length === 0 ? (
//                       <div>No alert seconds.</div>
//                     ) : (
//                       getPerSecondAlerts().map((p, idx) => (
//                         <div key={`pa-${idx}`}>
//                           date = {p.date}, second = {p.second}, avg ={" "}
//                           {p.avg_count.toFixed(3)}
//                         </div>
//                       ))
//                     )}
//                     <div style={styles.detailSectionTitle}>
//                       Safe seconds (between min &amp; max)
//                     </div>
//                     {getPerSecondSafe().length === 0 ? (
//                       <div>No safe seconds for current thresholds.</div>
//                     ) : (
//                       getPerSecondSafe().map((p, idx) => (
//                         <div key={`ps-${idx}`}>
//                           date = {p.date}, second = {p.second}, avg ={" "}
//                           {p.avg_count.toFixed(3)}
//                         </div>
//                       ))
//                     )}
//                   </div>
//                 )}
//               </div>

//               {/* GRAPH 3: Frame index vs count */}
//               <div style={styles.graphBox}>
//                 <div style={styles.graphTitle}>
//                   Frame index vs crowd count
//                 </div>
//                 <div style={styles.graphSubtitle}>
//                   Helps you see how model behaves as frames progress.
//                 </div>

//                 <button
//                   style={styles.detailBtn}
//                   onClick={() => setShowFrameDetails((prev) => !prev)}
//                 >
//                   {showFrameDetails ? "Hide details" : "Show details"}
//                 </button>

//                 <div style={styles.graphArea}>
//                   {frameSeries.length === 0 ? (
//                     <div style={styles.graphPlaceholder}>
//                       Upload a CSV to see frame-based trend.
//                     </div>
//                   ) : (
//                     <ResponsiveContainer width="100%" height="100%">
//                       <LineChart data={frameSeries}>
//                         <XAxis
//                           dataKey="frame_index"
//                           tick={{ fill: "#93c5fd", fontSize: 11 }}
//                           label={{
//                             value: "Frame index",
//                             position: "insideBottom",
//                             offset: -4,
//                             fill: "#6b7280",
//                             fontSize: 11,
//                           }}
//                         />
//                         <YAxis
//                           tick={{ fill: "#93c5fd", fontSize: 11 }}
//                           label={{
//                             value: "Count",
//                             angle: -90,
//                             position: "insideLeft",
//                             fill: "#6b7280",
//                             fontSize: 11,
//                           }}
//                         />
//                         <Tooltip
//                           contentStyle={{
//                             backgroundColor: "#020617",
//                             border: "1px solid #1e293b",
//                             borderRadius: 8,
//                             fontSize: 11,
//                           }}
//                           labelStyle={{ color: "#e5e7eb" }}
//                         />
//                         {thresholdsActive && (
//                           <>
//                             <ReferenceLine
//                               y={parsedMin}
//                               stroke={GREEN}
//                               strokeDasharray="3 3"
//                             />
//                             <ReferenceLine
//                               y={parsedMax}
//                               stroke={RED}
//                               strokeDasharray="3 3"
//                             />
//                           </>
//                         )}
//                         <Line
//                           type="monotone"
//                           dataKey="count"
//                           stroke={CYAN}
//                           strokeWidth={2}
//                           dot={(props) => renderColoredDot(props)}
//                           activeDot={{ r: 5 }}
//                         />
//                       </LineChart>
//                     </ResponsiveContainer>
//                   )}
//                 </div>

//                 {showFrameDetails && thresholdsActive && (
//                   <div style={styles.detailPanel}>
//                     <div style={styles.detailSectionTitle}>
//                       Alert frames (count &gt; max)
//                     </div>
//                     {getFrameAlerts().length === 0 ? (
//                       <div>No alert frames.</div>
//                     ) : (
//                       getFrameAlerts().map((p, idx) => (
//                         <div key={`fa-${idx}`}>
//                           date = {p.date}, frame = {p.frame_index}, count ={" "}
//                           {p.count.toFixed(3)}
//                         </div>
//                       ))
//                     )}
//                     <div style={styles.detailSectionTitle}>
//                       Safe frames (between min &amp; max)
//                     </div>
//                     {getFrameSafe().length === 0 ? (
//                       <div>No safe frames for current thresholds.</div>
//                     ) : (
//                       getFrameSafe().map((p, idx) => (
//                         <div key={`fs-${idx}`}>
//                           date = {p.date}, frame = {p.frame_index}, count ={" "}
//                           {p.count.toFixed(3)}
//                         </div>
//                       ))
//                     )}
//                   </div>
//                 )}
//               </div>
//             </div>
//           </div>
//         </div>
//       </div>
//     </>
//   );
// }

// export default Dashboard;




// main_page code
// import React, { useState, useEffect, useRef } from "react";
// import { useNavigate } from "react-router-dom";

// function useWindowWidth() {
//   const [width, setWidth] = useState(window.innerWidth);
//   useEffect(() => {
//     const handleResize = () => setWidth(window.innerWidth);
//     window.addEventListener("resize", handleResize);
//     return () => window.removeEventListener("resize", handleResize);
//   }, []);
//   return width;
// }

// function MainPage({ user }) {
//   const navigate = useNavigate();

//   const [systemSettingsOpen, setSystemSettingsOpen] = useState(false);
//   const systemSettingsRef = useRef(null);

//   const [selectedSystemCamera, setSelectedSystemCamera] = useState("");
//   const [uploadedVideoURL, setUploadedVideoURL] = useState(null);

//   // map: pairId -> crowd_txt content
//   const [crowdTxtMap, setCrowdTxtMap] = useState({});

//   const width = useWindowWidth();
//   const isMobile = width < 700;


//   // --- SAMPLE CSV STREAM (for sample right side) ---
//   const sampleCsvText = `
// Date,Timestamp_ns,Frame_index,Count
// 2025-03-01,00:00:000,0,150.114
// 2025-03-01,00:00:400,10,138.108
// 2025-03-01,00:00:800,20,160.500
// 2025-03-01,00:01:200,30,149.882
// 2025-03-01,00:01:600,40,152.774
// 2025-03-01,00:02:000,50,141.339
// 2025-03-01,00:02:400,60,159.002
// 2025-03-01,00:02:800,70,147.661
// 2025-03-01,00:03:200,80,153.441
// 2025-03-01,00:03:600,90,148.997
// 2025-03-01,00:04:000,100,157.884
// 2025-03-01,00:04:400,110,142.660
// 2025-03-01,00:04:800,120,151.225
// 2025-03-01,00:05:200,130,158.331
// 2025-03-01,00:05:600,140,144.880
// 2025-03-01,00:06:000,150,156.102
// 2025-03-01,00:06:400,160,139.556
// 2025-03-01,00:06:800,170,153.774
// 2025-03-01,00:07:200,180,157.330
// 2025-03-01,00:07:600,190,143.901
// 2025-03-01,00:08:000,200,159.447
// `.trim();

//   const sampleCsvLines = sampleCsvText.split("\n");
//   const [sampleCsvIndex, setSampleCsvIndex] = useState(1); // start after header
//   const [sampleCsvDisplay, setSampleCsvDisplay] = useState([sampleCsvLines[0]]);

//   // simulate real-time CSV writing for sample right pane
//   useEffect(() => {
//     const interval = setInterval(() => {
//       setSampleCsvDisplay((prev) => {
//         // When we reach end, reset to only header again
//         if (sampleCsvIndex >= sampleCsvLines.length) {
//           setSampleCsvIndex(1);
//           return [sampleCsvLines[0]];
//         }
//         // otherwise append next line
//         const nextLine = sampleCsvLines[sampleCsvIndex];
//         setSampleCsvIndex((i) => i + 1);
//         return [...prev, nextLine];
//       });
//     }, 400); // every 400ms, adjust speed if you like

//     return () => clearInterval(interval);
//   }, [sampleCsvLines.length, sampleCsvIndex]);

//   const styles = {
//     page: {
//       backgroundColor: "#020617",
//       minHeight: "100vh",
//       padding: isMobile ? "18px 4vw" : "32px 56px",
//       fontFamily:
//         "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
//       color: "#e5e7eb",
//       display: "flex",
//       flexDirection: "column",
//       gap: isMobile ? 18 : 28,
//       position: "relative",
//       animation: "fadeInDown 0.6s",
//       boxSizing: "border-box",
//     },

//     welcomeRow: {
//       display: "flex",
//       flexDirection: isMobile ? "column" : "row",
//       justifyContent: "space-between",
//       alignItems: isMobile ? "flex-start" : "center",
//       marginBottom: isMobile ? 12 : 20,
//       width: "100%",
//     },
//     brandTitle: {
//       fontSize: isMobile ? 14 : 16,
//       textTransform: "uppercase",
//       letterSpacing: "0.22em",
//       color: "#9ca3af",
//       marginBottom: 4,
//     },
//     welcomeText: {
//       fontSize: isMobile ? 20 : 26,
//       fontWeight: 600,
//       color: "#e5e7eb",
//     },
//     welcomeSubtext: {
//       marginTop: 4,
//       fontSize: 13,
//       color: "#6b7280",
//     },

//     cameraPairsContainer: {
//       display: "flex",
//       flexDirection: "column",
//       gap: isMobile ? 12 : 24,
//       flexWrap: "nowrap",
//       animation: "fadeIn 0.7s",
//       width: "100%",
//     },
//     cameraPair: {
//       display: isMobile ? "block" : "flex",
//       gap: isMobile ? 12 : 20,
//       alignItems: isMobile ? "stretch" : "flex-start",
//       width: "100%",
//       marginBottom: isMobile ? 10 : 0,
//     },
//     cameraCard: {
//       backgroundColor: "#020617",
//       borderRadius: 16,
//       boxShadow: "0 16px 40px rgba(15,23,42,0.9)",
//       overflow: "hidden",
//       paddingBottom: 16,
//       animation: "fadeInUp 0.7s",
//       flex: isMobile ? "unset" : "1 1 50%",
//       minWidth: isMobile ? "100%" : 360,
//       position: "relative",
//       border: "1px solid #1f2937",
//       transition: "transform 0.2s ease, box-shadow 0.2s ease",
//     },
//     cameraTitle: {
//       padding: isMobile ? "10px 12px 4px" : "14px 18px 4px",
//       borderBottom: "1px solid #111827",
//       backgroundColor: "#020617",
//     },
//     cameraNameInput: {
//       width: "100%",
//       padding: isMobile ? 7 : 9,
//       fontSize: isMobile ? 13 : 15,
//       borderRadius: 10,
//       border: "1px solid #1f2937",
//       backgroundColor: "#020617",
//       color: "#e5e7eb",
//       fontWeight: 500,
//       outline: "none",
//       boxSizing: "border-box",
//     },
//     cameraFrame: {
//       width: "100%",
//       height: isMobile ? 180 : 360,
//       backgroundColor: "#020617",
//       borderRadius: 14,
//       display: "flex",
//       justifyContent: "center",
//       alignItems: "center",
//       overflow: "hidden",
//       position: "relative",
//       marginTop: 8,
//       marginBottom: 10,
//       padding: 8,
//       boxSizing: "border-box",
//     },
//     cameraVideo: {
//       width: "100%",
//       height: "100%",
//       objectFit: "contain",
//       borderRadius: "14px",
//       backgroundColor: "#000",
//     },

//     // CSV view on right side (non-sample model card)
//     csvBox: {
//       width: "100%",
//       height: "100%",
//       borderRadius: 12,
//       border: "1px solid #1f2937",
//       backgroundColor: "#020617",
//       padding: 10,
//       boxSizing: "border-box",
//       display: "flex",
//       flexDirection: "column",
//     },
//     csvTitle: {
//       fontSize: 13,
//       fontWeight: 600,
//       color: "#93c5fd",
//       marginBottom: 6,
//     },
//     csvContent: {
//       flex: 1,
//       fontSize: 11,
//       lineHeight: 1.4,
//       color: "#e5e7eb",
//       whiteSpace: "pre-wrap",
//       overflowY: "auto",
//       paddingRight: 4,
//     },

//     cameraSettings: {
//       marginTop: 8,
//       padding: isMobile ? "0 10px" : "0 18px",
//       fontSize: 13,
//     },
//     toggleSwitch: {
//       marginTop: 10,
//       padding: isMobile ? "0 10px" : "0 18px",
//       display: "flex",
//       alignItems: "center",
//       justifyContent: "space-between",
//     },
//     toggleLabel: {
//       fontWeight: 500,
//       fontSize: isMobile ? 13 : 14,
//       color: "#e5e7eb",
//     },

//     deleteCameraBtn: {
//       position: "absolute",
//       top: 20,
//       right: 20,
//       backgroundColor: "#b91c1c",
//       border: "1px solid #fecaca",
//       borderRadius: 999,
//       color: "#fee2e2",
//       cursor: "pointer",
//       padding: "6px 12px",
//       fontWeight: 600,
//       fontSize: 11,
//       zIndex: 10,
//     },

//     buttonRow: {
//       marginTop: isMobile ? 20 : 28,
//       display: "flex",
//       justifyContent: "center",
//       gap: isMobile ? 10 : 16,
//       flexWrap: "wrap",
//       animation: "fadeInUp 0.7s",
//     },
//     actionButton: {
//       backgroundColor: "#1d4ed8",
//       color: "#e5e7eb",
//       borderRadius: 999,
//       padding: isMobile ? "11px 14px" : "12px 20px",
//       fontSize: isMobile ? 14 : 15,
//       fontWeight: 600,
//       cursor: "pointer",
//       border: "none",
//       boxShadow: "0 12px 30px rgba(37,99,235,0.35)",
//       letterSpacing: "0.03em",
//       minWidth: isMobile ? "100%" : 210,
//     },

//     addCameraBtn: {
//       marginTop: 16,
//       backgroundColor: "#111827",
//       color: "#e5e7eb",
//       borderRadius: 999,
//       padding: isMobile ? "10px 14px" : "11px 20px",
//       fontSize: 14,
//       fontWeight: 500,
//       cursor: "pointer",
//       border: "1px solid #374151",
//       minWidth: isMobile ? "100%" : 260,
//       alignSelf: "center",
//     },

//     sidePanel: {
//       position: isMobile ? "static" : "fixed",
//       top: isMobile ? undefined : 96,
//       right: isMobile ? undefined : 28,
//       width: isMobile ? "100%" : 340,
//       backgroundColor: "#020617",
//       boxShadow: "0 16px 45px rgba(15,23,42,0.95)",
//       borderRadius: 20,
//       padding: isMobile ? 14 : 18,
//       zIndex: 90,
//       animation: "fadeInUp 0.5s",
//       color: "#e5e7eb",
//       border: "1px solid #1f2937",
//       boxSizing: "border-box",
//       marginTop: isMobile ? 16 : 0,
//     },
//     sidePanelTitle: {
//       fontWeight: 600,
//       fontSize: isMobile ? 15 : 17,
//       marginBottom: 10,
//       color: "#93c5fd",
//     },
//     closeBtn: {
//       marginTop: 10,
//       width: "100%",
//       backgroundColor: "#111827",
//       color: "#e5e7eb",
//       border: "1px solid #374151",
//       borderRadius: 999,
//       padding: "7px 10px",
//       fontSize: 13,
//       cursor: "pointer",
//     },

//     systemSelect: {
//       width: "100%",
//       padding: 8,
//       borderRadius: 10,
//       border: "1px solid #374151",
//       backgroundColor: "#020617",
//       color: "#e5e7eb",
//       marginBottom: 12,
//       fontSize: 13,
//     },

//     uploadInput: {
//       marginTop: 4,
//       marginBottom: 10,
//       fontSize: 13,
//     },

//     "@keyframes fadeInDown": {
//       from: { opacity: 0, transform: "translateY(-24px)" },
//       to: { opacity: 1, transform: "translateY(0)" },
//     },
//     "@keyframes fadeInUp": {
//       from: { opacity: 0, transform: "translateY(24px)" },
//       to: { opacity: 1, transform: "translateY(0)" },
//     },
//     "@keyframes fadeIn": {
//       from: { opacity: 0 },
//       to: { opacity: 1 },
//     },
//   };

//   const [cameraPairs, setCameraPairs] = useState([
//     {
//       pairId: 0, // base sample cameras (do not touch)
//       cameras: [
//         {
//           id: 0,
//           name: "Camera 0 - Sample Video",
//           src: "/people.mp4",
//           on: true,
//           uploadedFile: null,
//         },
//         {
//           id: 1,
//           name: "Camera 0b - Sample Output",
//           src: "/output_with_heatmap.gif",
//           on: true,
//           uploadedFile: null,
//         },
//       ],
//     },
//   ]);

//   const [availableCameras] = useState([
//     "Integrated Webcam",
//     "USB Camera 1",
//     "USB Camera 2",
//     "Virtual Camera",
//   ]);

//   useEffect(() => {
//     function handleClickOutside(event) {
//       if (
//         systemSettingsRef.current &&
//         !systemSettingsRef.current.contains(event.target) &&
//         systemSettingsOpen
//       ) {
//         setSystemSettingsOpen(false);
//       }
//     }

//     document.addEventListener("mousedown", handleClickOutside);
//     return () => {
//       document.removeEventListener("mousedown", handleClickOutside);
//     };
//   }, [systemSettingsOpen]);

//   const toggleCamera = (pairId, cameraId) => {
//     setCameraPairs((pairs) =>
//       pairs.map((pair) =>
//         pair.pairId === pairId
//           ? {
//             ...pair,
//             cameras: pair.cameras.map((cam) =>
//               cam.id === cameraId ? { ...cam, on: !cam.on } : cam
//             ),
//           }
//           : pair
//       )
//     );
//   };

//   const updateCameraName = (pairId, cameraId, newName) => {
//     setCameraPairs((pairs) =>
//       pairs.map((pair) =>
//         pair.pairId === pairId
//           ? {
//             ...pair,
//             cameras: pair.cameras.map((cam) =>
//               cam.id === cameraId ? { ...cam, name: newName } : cam
//             ),
//           }
//           : pair
//       )
//     );
//   };

//   // Upload video for REAL camera (left) => backend processes, we show CSV on right
//   const uploadVideoFile = async (pairId, cameraId, file) => {
//     const formData = new FormData();
//     formData.append("video", file);

//     try {
//       const res = await fetch("http://0.0.0.0:8000/process_video/", {
//         method: "POST",
//         body: formData,
//       });

//       // even if backend returns output_video, we ignore it for non-sample
//       await res.json().catch(() => null);

//       // Left side (real) shows uploaded video as before
//       setCameraPairs((pairs) =>
//         pairs.map((pair) =>
//           pair.pairId === pairId
//             ? {
//               ...pair,
//               cameras: pair.cameras.map((cam) =>
//                 cam.id === cameraId
//                   ? {
//                     ...cam,
//                     src: URL.createObjectURL(file),
//                     uploadedFile: file,
//                   }
//                   : cam
//               ),
//             }
//             : pair
//         )
//       );

//       // Right side (model) shows live crowd_txt CSV
//       fetch("http://0.0.0.0:8000/crowd_txt/")
//         .then((r) => r.text())
//         .then((txt) => {
//           setCrowdTxtMap((prev) => ({
//             ...prev,
//             [pairId]: txt,
//           }));
//         })
//         .catch(() => { });
//     } catch (err) {
//       console.error("Error processing video:", err);
//     }
//   };

//   const addCameraPair = () => {
//     const newPairId = cameraPairs.length
//       ? cameraPairs[cameraPairs.length - 1].pairId + 1
//       : 1;
//     const baseCamId = cameraPairs.reduce(
//       (max, pair) => Math.max(max, ...pair.cameras.map((c) => c.id)),
//       0
//     );

//     setCameraPairs((prev) => [
//       ...prev,
//       {
//         pairId: newPairId,
//         cameras: [
//           {
//             id: baseCamId + 1,
//             name: `Camera ${newPairId} - Real`,
//             src: "https://placeimg.com/640/480/nature",
//             on: true,
//             uploadedFile: null,
//           },
//           {
//             id: baseCamId + 2,
//             name: `Camera ${newPairId}b - Crowd CSV`,
//             src: "",
//             on: true,
//             uploadedFile: null,
//           },
//         ],
//       },
//     ]);
//   };

//   const deleteCameraPair = (pairId) => {
//     if (pairId === 0) return; // don't delete sample pair
//     setCameraPairs((pairs) => pairs.filter((pair) => pair.pairId !== pairId));
//     setCrowdTxtMap((prev) => {
//       const copy = { ...prev };
//       delete copy[pairId];
//       return copy;
//     });
//   };

//   const addSystemCamera = () => {
//     if (!selectedSystemCamera) return;
//     const newPairId = cameraPairs.length
//       ? cameraPairs[cameraPairs.length - 1].pairId + 1
//       : 1;

//     setCameraPairs((prev) => [
//       ...prev,
//       {
//         pairId: newPairId,
//         cameras: [
//           {
//             id: newPairId * 2 - 1,
//             name: `${selectedSystemCamera} - Real`,
//             src: "https://placeimg.com/640/480/tech",
//             on: true,
//           },
//           {
//             id: newPairId * 2,
//             name: `${selectedSystemCamera}b - Crowd CSV`,
//             src: "",
//             on: true,
//           },
//         ],
//       },
//     ]);
//     setSelectedSystemCamera("");
//     setSystemSettingsOpen(false);
//   };

//   const addUploadedVideo = () => {
//     if (!uploadedVideoURL) return;
//     const newPairId = cameraPairs.length
//       ? cameraPairs[cameraPairs.length - 1].pairId + 1
//       : 1;

//     setCameraPairs((prev) => [
//       ...prev,
//       {
//         pairId: newPairId,
//         cameras: [
//           {
//             id: newPairId * 2 - 1,
//             name: `Uploaded Video ${newPairId} - Real`,
//             src: uploadedVideoURL,
//             on: true,
//           },
//           {
//             id: newPairId * 2,
//             name: `Uploaded Video ${newPairId}b - Crowd CSV`,
//             src: "",
//             on: true,
//           },
//         ],
//       },
//     ]);
//     setUploadedVideoURL(null);
//     setSystemSettingsOpen(false);
//   };

//   return (
//     <div style={styles.page}>
//       {/* Top: welcome */}
//       <div style={styles.welcomeRow}>
//         <div>
//           <div style={styles.brandTitle}>VigilNet</div>
//           <div style={styles.welcomeText}>Welcome to VigilNet</div>
//           <div style={styles.welcomeSubtext}>
//             Intelligent CCTV and crowd monitoring console
//           </div>
//         </div>
//       </div>

//       {/* Camera pairs */}
//       <div style={styles.cameraPairsContainer}>
//         {cameraPairs.map((pair) => {
//           const isCamera0 = pair.pairId === 0;

//           return (
//             <div key={pair.pairId} style={styles.cameraPair}>
//               {pair.cameras.map((cam) => {
//                 const isModelVideo = cam.name.toLowerCase().includes("b");

//                 return (
//                   <div key={cam.id} style={styles.cameraCard}>
//                     <div style={styles.cameraTitle}>
//                       <input
//                         type="text"
//                         value={cam.name}
//                         onChange={(e) =>
//                           updateCameraName(pair.pairId, cam.id, e.target.value)
//                         }
//                         style={styles.cameraNameInput}
//                         disabled={isCamera0}
//                       />
//                     </div>

//                     <div style={styles.cameraFrame}>
//                       {cam.on ? (
//                         // RIGHT card logic
//                         isModelVideo ? (
//                           // SAMPLE pair (pairId 0): show simulated CSV stream
//                           isCamera0 ? (
//                             <div style={styles.csvBox}>
//                               <div style={styles.csvTitle}>Sample Crowd CSV (simulated live)</div>
//                               <pre style={styles.csvContent}>
//                                 {sampleCsvDisplay.join("\n")}
//                               </pre>
//                             </div>
//                           ) : (
//                             // NON-SAMPLE pair right side: backend crowd_txt
//                             <div style={styles.csvBox}>
//                               <div style={styles.csvTitle}>Crowd CSV (live)</div>
//                               <pre style={styles.csvContent}>
//                                 {crowdTxtMap[pair.pairId] ||
//                                   "Upload a video on the left to stream crowd.csv from backend."}
//                               </pre>
//                             </div>
//                           )
//                         ) : (
//                           // LEFT (real) camera: normal video/image feed
//                           <>
//                             {cam.src.endsWith(".mp4") || cam.src.endsWith(".webm") ? (
//                               <video
//                                 autoPlay
//                                 muted
//                                 loop
//                                 style={styles.cameraVideo}
//                                 src={cam.src}
//                               />
//                             ) : (
//                               <img
//                                 style={styles.cameraVideo}
//                                 alt={`${cam.name} Feed`}
//                                 src={cam.src}
//                               />
//                             )}
//                           </>
//                         )
//                       ) : (
//                         <div
//                           style={{
//                             color: "#6b7280",
//                             fontStyle: "italic",
//                             fontSize: 13,
//                           }}
//                         >
//                           Camera Off
//                         </div>
//                       )}
//                     </div>


//                     <div style={styles.toggleSwitch}>
//                       <label style={styles.toggleLabel}>{cam.name} On/Off</label>
//                       <input
//                         type="checkbox"
//                         checked={cam.on}
//                         onChange={() => toggleCamera(pair.pairId, cam.id)}
//                       />
//                     </div>

//                     {/* Upload for real videos of non-sample pairs */}
//                     {!isModelVideo && !isCamera0 && (
//                       <div style={styles.cameraSettings}>
//                         <label style={{ fontSize: 13 }}>
//                           Upload Video:
//                           <input
//                             type="file"
//                             accept="video/*"
//                             onChange={(e) => {
//                               if (e.target.files && e.target.files[0]) {
//                                 uploadVideoFile(
//                                   pair.pairId,
//                                   cam.id,
//                                   e.target.files[0]
//                                 );
//                               }
//                             }}
//                             style={styles.uploadInput}
//                           />
//                         </label>
//                       </div>
//                     )}

//                     {/* Delete whole pair (non-sample, only on left card) */}
//                     {!isCamera0 && cam.id === pair.cameras[0].id && (
//                       <button
//                         onClick={() => deleteCameraPair(pair.pairId)}
//                         style={styles.deleteCameraBtn}
//                         title="Delete this camera pair"
//                       >
//                         Delete
//                       </button>
//                     )}
//                   </div>
//                 );
//               })}
//             </div>
//           );
//         })}
//       </div>

//       {/* Add camera pair */}
//       <button style={styles.addCameraBtn} onClick={addCameraPair}>
//         + Add More Camera Pair
//       </button>

//       {/* Action buttons */}
//       <div style={styles.buttonRow}>
//         <button
//           style={styles.actionButton}
//           onClick={() => setSystemSettingsOpen((v) => !v)}
//         >
//           System Settings
//         </button>

//         <button
//           style={styles.actionButton}
//           onClick={() => navigate("/dashboard")}
//         >
//           Analytics Dashboard
//         </button>

//         <button
//           style={styles.actionButton}
//           onClick={() => navigate("/crowd_count_photo")}
//         >
//           Crowd Counting from Photo
//         </button>
//       </div>

//       {/* System Settings panel (no sounds now) */}
//       {systemSettingsOpen && (
//         <div style={styles.sidePanel} ref={systemSettingsRef}>
//           <div style={styles.sidePanelTitle}>System Settings</div>

//           <label style={{ fontWeight: 600, fontSize: 13 }}>Select Camera:</label>
//           <select
//             value={selectedSystemCamera}
//             onChange={(e) => setSelectedSystemCamera(e.target.value)}
//             style={styles.systemSelect}
//           >
//             <option value="">-- Choose a camera --</option>
//             {availableCameras.map((cam, idx) => (
//               <option key={idx} value={cam}>
//                 {cam}
//               </option>
//             ))}
//           </select>

//           <button
//             disabled={!selectedSystemCamera}
//             onClick={addSystemCamera}
//             style={{
//               ...styles.actionButton,
//               width: "100%",
//               opacity: selectedSystemCamera ? 1 : 0.5,
//               marginBottom: 10,
//             }}
//           >
//             Add Selected Camera
//           </button>

//           <label style={{ fontWeight: 600, fontSize: 13 }}>Upload Video:</label>
//           <input
//             type="file"
//             accept="video/*"
//             onChange={(e) => {
//               if (e.target.files && e.target.files[0]) {
//                 const fileURL = URL.createObjectURL(e.target.files[0]);
//                 setUploadedVideoURL(fileURL);
//               }
//             }}
//             style={styles.uploadInput}
//           />

//           {uploadedVideoURL && (
//             <>
//               <video
//                 src={uploadedVideoURL}
//                 controls
//                 style={{
//                   width: "100%",
//                   borderRadius: 12,
//                   marginBottom: 10,
//                   marginTop: 4,
//                 }}
//               />
//               <button
//                 onClick={addUploadedVideo}
//                 style={{ ...styles.actionButton, width: "100%" }}
//               >
//                 Add Uploaded Video as Camera Pair
//               </button>
//             </>
//           )}

//           <button
//             onClick={() => setSystemSettingsOpen(false)}
//             style={styles.closeBtn}
//           >
//             Close
//           </button>
//         </div>
//       )}
//     </div>
//   );
// }

// export default MainPage;


// dashboard page
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
  const renderColoredDot = (props) => {
    const { cx, cy, payload } = props;
    const value = payload.count;
    let fill = "#60a5fa";

    if (thresholdsActive) {
      if (value > parsedMax) fill = RED; // alert
      else if (value >= parsedMin && value <= parsedMax) fill = GREEN; // safe
      else fill = AMBER; // below min
    }

    return (
      <circle cx={cx} cy={cy} r={3} fill={fill} stroke={BG} strokeWidth={1} />
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
      const res = await fetch("http://127.0.0.1:8000/upload-csv", {
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






// anaylitice
# # main.py
# import os
# import time
# import threading
# import csv
# from datetime import datetime


# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Query
# from fastapi.middleware.cors import CORSMiddleware
# from starlette.responses import JSONResponse

# # ---------------- SETTINGS ----------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "crowd_counting.pth")

# USE_GPU = True
# USE_FP16_IF_CUDA = True
# FRAME_RESIZE = (512, 384)
# SKIP_FRAMES = 10
# SMOOTH_WINDOW = 3
# FLUSH_INTERVAL = 2.0
# PRINT_EVERY = 50
# UPSAMPLE_DMAP = False
# SCALE_BY_AREA = True
# # ------------------------------------------

# device = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")
# print("Device:", device)

# # ---------- Camera hinderance helpers ----------
# _hinder_counters = {
#     "frozen": 0,
#     "covered": 0,
#     "low_contrast": 0,
#     "obstructed": 0,
# }
# FROZEN_DIFF_THRESH = 2.0
# FROZEN_FRAMES_THRESH = 5
# COVERED_MEAN_THRESH = 15
# COVERED_FRAMES_THRESH = 3
# LOW_CONTRAST_STD_THRESH = 10.0
# LOW_CONTRAST_FRAMES_THRESH = 5
# OBSTRUCT_EDGE_DENSITY_THRESH = 0.005
# OBSTRUCT_FRAMES_THRESH = 5
# DARK_PCT_THRESH = 0.50

# _prev_gray_for_hinder = None


# def check_camera_hinder(frame_bgr):
#     global _prev_gray_for_hinder, _hinder_counters
#     gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#     mean = float(gray.mean())
#     std = float(gray.std())

#     if _prev_gray_for_hinder is None:
#         diff_mean = 255.0
#     else:
#         diff = cv2.absdiff(gray, _prev_gray_for_hinder)
#         diff_mean = float(diff.mean())

#     edges = cv2.Canny(gray, 50, 150)
#     edge_density = float(np.mean(edges > 0))
#     dark_pct = float(np.mean(gray < 30))

#     if diff_mean < FROZEN_DIFF_THRESH:
#         _hinder_counters["frozen"] += 1
#     else:
#         _hinder_counters["frozen"] = 0

#     if mean < COVERED_MEAN_THRESH and std < LOW_CONTRAST_STD_THRESH:
#         _hinder_counters["covered"] += 1
#     else:
#         _hinder_counters["covered"] = 0

#     if std < LOW_CONTRAST_STD_THRESH:
#         _hinder_counters["low_contrast"] += 1
#     else:
#         _hinder_counters["low_contrast"] = 0

#     if edge_density < OBSTRUCT_EDGE_DENSITY_THRESH or dark_pct >= DARK_PCT_THRESH:
#         _hinder_counters["obstructed"] += 1
#     else:
#         _hinder_counters["obstructed"] = 0

#     reason = ""
#     if _hinder_counters["frozen"] >= FROZEN_FRAMES_THRESH:
#         reason = "camera_frozen"
#     elif _hinder_counters["covered"] >= COVERED_FRAMES_THRESH:
#         reason = "lens_covered_or_extremely_dark"
#     elif _hinder_counters["obstructed"] >= OBSTRUCT_FRAMES_THRESH:
#         reason = "obstructed_or_low_edge_density"
#     elif _hinder_counters["low_contrast"] >= LOW_CONTRAST_FRAMES_THRESH:
#         reason = "low_contrast"

#     _prev_gray_for_hinder = gray.copy()

#     return bool(reason), reason


# def enhance_frame(frame_bgr,
#                   gamma=1.6,
#                   use_clahe=True,
#                   clahe_clip=2.0,
#                   clahe_tile=(8, 8),
#                   denoise=False):
#     img = frame_bgr.copy()
#     if gamma != 1.0:
#         invGamma = 1.0 / gamma
#         table = np.array([((i / 255.0) ** invGamma) * 255
#                           for i in np.arange(256)]).astype("uint8")
#         img = cv2.LUT(img, table)

#     if use_clahe:
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         h, s, v = cv2.split(hsv)
#         clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
#         v_clahe = clahe.apply(v)
#         hsv_clahe = cv2.merge((h, s, v_clahe))
#         img = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

#     if denoise:
#         img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

#     img = np.clip(img, 0, 255).astype(np.uint8)
#     return img


# # ---------- MCNN model ----------
# class MC_CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.column1 = nn.Sequential(
#             nn.Conv2d(3, 8, 9, padding="same"),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(8, 16, 7, padding="same"),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, 7, padding="same"),
#             nn.ReLU(),
#             nn.Conv2d(32, 16, 7, padding="same"),
#             nn.ReLU(),
#             nn.Conv2d(16, 8, 7, padding="same"),
#             nn.ReLU(),
#         )

#         self.column2 = nn.Sequential(
#             nn.Conv2d(3, 10, 7, padding="same"),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(10, 20, 5, padding="same"),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(20, 40, 5, padding="same"),
#             nn.ReLU(),
#             nn.Conv2d(40, 20, 5, padding="same"),
#             nn.ReLU(),
#             nn.Conv2d(20, 10, 5, padding="same"),
#             nn.ReLU(),
#         )

#         self.column3 = nn.Sequential(
#             nn.Conv2d(3, 12, 5, padding="same"),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(12, 24, 3, padding="same"),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(24, 48, 3, padding="same"),
#             nn.ReLU(),
#             nn.Conv2d(48, 24, 3, padding="same"),
#             nn.ReLU(),
#             nn.Conv2d(24, 12, 3, padding="same"),
#             nn.ReLU(),
#         )

#         self.fusion_layer = nn.Conv2d(30, 1, 1, padding=0)

#     def forward(self, img_tensor):
#         x1 = self.column1(img_tensor)
#         x2 = self.column2(img_tensor)
#         x3 = self.column3(img_tensor)
#         x = torch.cat((x1, x2, x3), 1)
#         x = self.fusion_layer(x)
#         return x


# def load_model(path, device):
#     m = MC_CNN().to(device).eval()
#     ckpt = torch.load(path, map_location=device)
#     if isinstance(ckpt, dict) and "state_dict" in ckpt:
#         state = ckpt["state_dict"]
#     elif isinstance(ckpt, dict):
#         state = ckpt
#     elif isinstance(ckpt, nn.Module):
#         ckpt.to(device).eval()
#         return ckpt
#     else:
#         raise RuntimeError("Unsupported checkpoint format")

#     new_state = {}
#     for k, v in state.items():
#         nk = k[len("module."):] if k.startswith("module.") else k
#         new_state[nk] = v

#     m.load_state_dict(new_state, strict=False)
#     return m


# model = load_model(MODEL_PATH, device)
# if device.type == "cuda" and USE_FP16_IF_CUDA:
#     model.half()


# def frame_to_tensor(frame_bgr):
#     if FRAME_RESIZE is not None:
#         frame_bgr = cv2.resize(frame_bgr, FRAME_RESIZE, interpolation=cv2.INTER_AREA)
#     rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#     arr = np.asarray(rgb, dtype=np.float32) / 255.0
#     arr = np.transpose(arr, (2, 0, 1))
#     tensor = torch.from_numpy(arr).unsqueeze(0)
#     if device.type == "cuda" and USE_FP16_IF_CUDA:
#         tensor = tensor.half().to(device, non_blocking=True)
#     else:
#         tensor = tensor.to(device, non_blocking=True)
#     return tensor


# def ms_to_time_str(t_ms):
#     ms = int(t_ms % 1000)
#     total_seconds = int(t_ms // 1000)
#     minutes = total_seconds // 60
#     seconds = total_seconds % 60
#     return f"{minutes:02d}:{seconds:02d}:{ms:03d}"


# # ---------- per-run tracking ----------
# RUNS = {}  # run_id -> {"csv_path": str, "done": bool}
# RUNS_LOCK = threading.Lock()


# def process_video_to_csv(video_path: str, out_csv_path: str):
#     global _hinder_counters, _prev_gray_for_hinder
#     _hinder_counters = {k: 0 for k in _hinder_counters}
#     _prev_gray_for_hinder = None

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Cannot open video:", video_path)
#         return

#     csv_file = open(out_csv_path, "w", newline="", buffering=1)
#     writer = csv.writer(csv_file)

#     # 👉 today's date in yyyy-mm-dd format
#     today_str = datetime.now().strftime("%Y-%m-%d")

#     # 👉 updated header with date first
#     writer.writerow(["date", "timestamp_ms", "frame_index", "count", "alert"])

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

#             # 👉 write date as first column now
#             writer.writerow([
#                 today_str,              # date (yyyy-mm-dd)
#                 time_str,               # timestamp_ms (mm:ss:ms string)
#                 processed,              # frame_index
#                 f"{display_count:.3f}", # count
#                 alert_field             # alert
#             ])

#             processed += 1

#             if time.time() - last_flush > FLUSH_INTERVAL:
#                 csv_file.flush()
#                 last_flush = time.time()

#             if processed % PRINT_EVERY == 0:
#                 elapsed = time.time() - start_wall
#                 fps_eff = processed / elapsed if elapsed > 0 else 0
#                 print(
#                     f"Processed {processed} frames (effective FPS: {fps_eff:.2f}), "
#                     f"last count {display_count:.2f}"
#                 )

#     csv_file.flush()
#     csv_file.close()
#     cap.release()
#     print("Finished. Wrote counts to:", out_csv_path)


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


# # ---------- FastAPI ----------
# # app = FastAPI()
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


