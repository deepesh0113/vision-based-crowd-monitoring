import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

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
  const now = new Date();

  const [fromYear, setFromYear] = useState(now.getFullYear());
  const [fromMonth, setFromMonth] = useState(now.getMonth() + 1);
  const [fromDay, setFromDay] = useState(now.getDate());
  const [fromHour, setFromHour] = useState(0);
  const [toYear, setToYear] = useState(now.getFullYear());
  const [toMonth, setToMonth] = useState(now.getMonth() + 1);
  const [toDay, setToDay] = useState(now.getDate());
  const [toHour, setToHour] = useState(23);
  const [filterHover, setFilterHover] = useState(false);
  const [backHover, setBackHover] = useState(false);

  const yearOptions = Array.from({ length: 6 }, (_, i) => 2020 + i);
  const monthOptions = Array.from({ length: 12 }, (_, i) => i + 1);
  const daysInMonth = (year, month) => new Date(year, month, 0).getDate();
  const fromDayOptions = Array.from(
    { length: daysInMonth(fromYear, fromMonth) },
    (_, i) => i + 1
  );
  const toDayOptions = Array.from(
    { length: daysInMonth(toYear, toMonth) },
    (_, i) => i + 1
  );
  const hourOptions = Array.from({ length: 24 }, (_, i) => i);

  const width = useWindowWidth();
  const isMobile = width < 700;

  const styles = {
    page: {
      minHeight: "100vh",
      backgroundColor: "#020617",
      fontFamily:
        "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      display: "flex",
      justifyContent: "center",
      padding: isMobile ? "16px 4vw 32px" : "24px 24px 40px",
      boxSizing: "border-box",
    },
    content: {
      width: "100%",
      maxWidth: 1040,
      display: "flex",
      flexDirection: "column",
      gap: isMobile ? 16 : 20,
      animation: "fadeIn 0.6s ease-in",
    },
    topBar: {
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      gap: 12,
    },
    backBtn: {
      borderRadius: 999,
      border: "1px solid #1f2937",
      backgroundColor: backHover ? "#0b1120" : "#020617",
      color: "#9ca3af",
      padding: "7px 16px",
      fontSize: 13,
      cursor: "pointer",
      display: "flex",
      alignItems: "center",
      gap: 6,
      boxShadow: backHover ? "0 0 0 1px #1f2937" : "none",
      transition: "background 0.15s ease, box-shadow 0.15s ease, color 0.15s ease",
      whiteSpace: "nowrap",
    },
    titleBlock: {
      flex: 1,
      display: "flex",
      flexDirection: "column",
      gap: 4,
      alignItems: isMobile ? "flex-start" : "flex-start",
    },
    brand: {
      fontSize: 12,
      textTransform: "uppercase",
      letterSpacing: "0.2em",
      color: "#6b7280",
    },
    header: {
      fontWeight: 600,
      fontSize: isMobile ? "1.3rem" : "1.7rem",
      letterSpacing: "0.03em",
      color: "#e5e7eb",
    },
    subtitle: {
      fontSize: 12,
      color: "#6b7280",
    },

    theory: {
      backgroundColor: "#020617",
      borderRadius: 16,
      padding: isMobile ? "14px 14px" : "18px 20px",
      border: "1px solid #1f2937",
      boxShadow: "0 16px 40px rgba(15,23,42,0.9)",
      color: "#d1d5db",
      fontSize: isMobile ? 13 : 14,
      lineHeight: 1.7,
    },
    theoryStrong: {
      fontWeight: 600,
      color: "#93c5fd",
    },

    filterBox: {
      backgroundColor: "#020617",
      borderRadius: 16,
      border: "1px solid #1f2937",
      boxShadow: "0 16px 40px rgba(15,23,42,0.9)",
      padding: isMobile ? "12px 12px" : "16px 20px",
      display: "flex",
      flexDirection: isMobile ? "column" : "row",
      alignItems: isMobile ? "flex-start" : "center",
      flexWrap: "wrap",
      gap: isMobile ? 10 : 16,
      fontSize: 13,
    },
    filterRow: {
      display: "flex",
      flexWrap: "wrap",
      alignItems: "center",
      gap: 8,
    },
    label: {
      color: "#9ca3af",
      fontWeight: 500,
      marginRight: 4,
      whiteSpace: "nowrap",
    },
    select: {
      padding: "6px 8px",
      borderRadius: 10,
      border: "1px solid #1f2937",
      backgroundColor: "#020617",
      color: "#e5e7eb",
      fontSize: 13,
      minWidth: 70,
      outline: "none",
    },
    filterButton: {
      backgroundColor: "#1d4ed8",
      color: "#f9fafb",
      cursor: "pointer",
      border: "none",
      fontWeight: 600,
      padding: "8px 18px",
      borderRadius: 999,
      letterSpacing: "0.06em",
      boxShadow: filterHover
        ? "0 12px 30px rgba(37,99,235,0.45)"
        : "0 10px 24px rgba(37,99,235,0.32)",
      transition: "box-shadow 0.15s ease, transform 0.1s ease",
      transform: filterHover ? "translateY(-1px)" : "translateY(0)",
      fontSize: 13,
      whiteSpace: "nowrap",
      marginLeft: isMobile ? 0 : "auto",
      width: isMobile ? "100%" : "auto",
      textAlign: "center",
    },

    graphBox: {
      backgroundColor: "#020617",
      borderRadius: 16,
      border: "1px solid #1f2937",
      boxShadow: "0 18px 42px rgba(15,23,42,0.95)",
      padding: isMobile ? "16px 12px 24px" : "18px 22px 28px",
      minHeight: isMobile ? 220 : 280,
      display: "flex",
      flexDirection: "column",
    },
    graphTitle: {
      fontSize: isMobile ? 14 : 15,
      color: "#93c5fd",
      fontWeight: 600,
      marginBottom: 8,
    },
    graphRange: {
      fontSize: 12,
      color: "#6b7280",
      marginBottom: 16,
    },
    graphPlaceholder: {
      flex: 1,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontSize: 13,
      color: "#60a5fa",
      borderRadius: 12,
      border: "1px dashed #1f2937",
    },
  };

  const handleFilter = () => {
    alert(
      `Filtering data from ${fromDay}-${fromMonth}-${fromYear} ${fromHour}:00 to ${toDay}-${toMonth}-${toYear} ${toHour}:00`
    );
  };

  return (
    <>
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      <div style={styles.page}>
        <div style={styles.content}>
          {/* Top bar */}
          <div style={styles.topBar}>
            <button
              style={styles.backBtn}
              onMouseEnter={() => setBackHover(true)}
              onMouseLeave={() => setBackHover(false)}
              onClick={() => navigate(-1)}
            >
              <span>←</span>
              <span>Back</span>
            </button>

            <div style={styles.titleBlock}>
              <div style={styles.brand}>VigilNet</div>
              <div style={styles.header}>Historical Dashboard</div>
              <div style={styles.subtitle}>
                Explore past crowd patterns and time-window behaviour.
              </div>
            </div>
          </div>

          {/* Theory / description card */}
          <div style={styles.theory}>
            <span style={styles.theoryStrong}>How does the Dashboard help?</span>
            <br />
            The VigilNet dashboard lets you track, compare, and analyze crowd events
            over any time window. Choose a date–time range and inspect how density
            evolved across hours, days, or weeks.
            <br />
            <br />
            <span style={styles.theoryStrong}>How do the models work?</span>
            <br />
            Deep learning and time-series models (for example, LSTM-style sequence
            models and regression baselines) learn patterns from historical
            density maps. They can highlight abnormal spikes, recurring peaks, and
            low-traffic intervals to support proactive planning.
            <br />
            <br />
            <span style={styles.theoryStrong}>Why visualize?</span>
            <br />
            Visual charts quickly reveal peaks, lulls, and correlations with
            events or alerts. With filters, operators can compare crowd profiles by
            day, month, or hour – similar to how research dashboards analyse
            experiments.
          </div>

          {/* Filter card */}
          <div style={styles.filterBox}>
            <div style={styles.filterRow}>
              <span style={styles.label}>From</span>
              <select
                style={styles.select}
                value={fromDay}
                onChange={(e) => setFromDay(Number(e.target.value))}
              >
                {fromDayOptions.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
              <select
                style={styles.select}
                value={fromMonth}
                onChange={(e) => setFromMonth(Number(e.target.value))}
              >
                {monthOptions.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
              <select
                style={styles.select}
                value={fromYear}
                onChange={(e) => setFromYear(Number(e.target.value))}
              >
                {yearOptions.map((y) => (
                  <option key={y} value={y}>
                    {y}
                  </option>
                ))}
              </select>
              <select
                style={styles.select}
                value={fromHour}
                onChange={(e) => setFromHour(Number(e.target.value))}
              >
                {hourOptions.map((h) => (
                  <option key={h} value={h}>
                    {h}:00
                  </option>
                ))}
              </select>
            </div>

            <div style={styles.filterRow}>
              <span style={styles.label}>To</span>
              <select
                style={styles.select}
                value={toDay}
                onChange={(e) => setToDay(Number(e.target.value))}
              >
                {toDayOptions.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
              <select
                style={styles.select}
                value={toMonth}
                onChange={(e) => setToMonth(Number(e.target.value))}
              >
                {monthOptions.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
              <select
                style={styles.select}
                value={toYear}
                onChange={(e) => setToYear(Number(e.target.value))}
              >
                {yearOptions.map((y) => (
                  <option key={y} value={y}>
                    {y}
                  </option>
                ))}
              </select>
              <select
                style={styles.select}
                value={toHour}
                onChange={(e) => setToHour(Number(e.target.value))}
              >
                {hourOptions.map((h) => (
                  <option key={h} value={h}>
                    {h}:00
                  </option>
                ))}
              </select>
            </div>

            <button
              style={styles.filterButton}
              onMouseEnter={() => setFilterHover(true)}
              onMouseLeave={() => setFilterHover(false)}
              onClick={handleFilter}
            >
              FILTER DATA
            </button>
          </div>

          {/* Graph card */}
          <div style={styles.graphBox}>
            <div style={styles.graphTitle}>Crowd trends over selected interval</div>
            <div style={styles.graphRange}>
              {fromDay}-{fromMonth}-{fromYear} {fromHour}:00 → {toDay}-{toMonth}-
              {toYear} {toHour}:00
            </div>

            <div style={styles.graphPlaceholder}>
              {/* Hook up Recharts / Chart.js / custom SVG here */}
              [Chart area – plug in line / area chart for density vs time]
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default Dashboard;
