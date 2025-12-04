import React, { useMemo, useState } from "react";

export default function CsvViewer({ csvText, fallbackMessage, styleOverride = {} }) {
  const [showRaw, setShowRaw] = useState(false);
  const [rowsPerPage, setRowsPerPage] = useState(100);
  const [page, setPage] = useState(0);

  const parsed = useMemo(() => {
    if (!csvText || !csvText.trim()) return { headers: [], rows: [] };
    const lines = csvText.trim().split(/\r?\n/).filter(Boolean);
    const headers = lines[0].split(",").map(h => h.trim());
    const rows = lines.slice(1).map(line => {
      const cols = line.split(",").map(c => c.trim());
      const obj = {};
      headers.forEach((h, i) => (obj[h] = cols[i] ?? ""));
      return obj;
    });
    return { headers, rows };
  }, [csvText]);

  const totalPages = Math.max(1, Math.ceil(parsed.rows.length / rowsPerPage));
  const pageRows = parsed.rows.slice(page * rowsPerPage, (page + 1) * rowsPerPage);

  const formatCount = v => {
    const n = parseFloat(v);
    if (isNaN(n)) return v;
    if (Math.abs(n - Math.round(n)) < 0.001) return Math.round(n);
    return n.toFixed(2);
  };

  const badgeForAlert = alert => {
    if (!alert || alert === "NONE" || alert === "null") return null;
    const a = alert.toUpperCase();
    let style = { background: "#fde68a", color: "#92400e" };
    if (a.includes("HINDER") || a.includes("CAMERA")) style = { background: "#fecaca", color: "#7f1d1d" };
    if (a.includes("LOW_LIGHT")) style = { background: "#bbf7d0", color: "#065f46" };
    return (
      <span style={{ padding: "2px 6px", borderRadius: 6, fontSize: 11, fontWeight: 600, ...style }}>
        {alert}
      </span>
    );
  };

  const baseStyle = {
    flex: 1,
    fontSize: 11,
    lineHeight: 1.4,
    color: "#e5e7eb",
    whiteSpace: "pre-wrap",
    overflowY: "auto",
    paddingRight: 4,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, 'Courier New', monospace",
    padding: 10,
    ...styleOverride,
  };

  if (!csvText || !csvText.trim()) {
    return <div style={baseStyle}>{fallbackMessage}</div>;
  }

  return (
    <div style={baseStyle}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
        <button onClick={() => setShowRaw(s => !s)} style={buttonStyle}>
          {showRaw ? "Pretty view" : "Raw CSV"}
        </button>
        <button
          onClick={() => {
            const blob = new Blob([csvText], { type: "text/csv" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "output_with_hinderance.csv";
            a.click();
            URL.revokeObjectURL(url);
          }}
          style={buttonStyle}
        >
          Download CSV
        </button>

        <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
          <label style={{ fontSize: 11, opacity: 0.8 }}>Rows / page</label>
          <select value={rowsPerPage} onChange={e => { setRowsPerPage(parseInt(e.target.value, 10)); setPage(0); }} style={{ fontSize: 12 }}>
            {[20,50,100,200].map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
      </div>

      {showRaw ? (
        <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{csvText}</pre>
      ) : (
        <>
          <div style={{ overflowX: "auto", borderRadius: 8 }}>
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 700 }}>
              <thead>
                <tr>
                  {parsed.headers.map(h => (
                    <th key={h} style={{ textAlign: "left", padding: "8px 10px", fontSize: 11, opacity: 0.8 }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pageRows.map((r, i) => {
                  const alert = (r.alert ?? r.Alert ?? r.AlerT ?? "").toString();
                  const alertFlag = alert && alert.toUpperCase() !== "NONE" && alert !== "";
                  return (
                    <tr key={i} style={{
                      background: alertFlag ? "linear-gradient(90deg, rgba(255,235,238,0.03), transparent)" : "transparent"
                    }}>
                      {parsed.headers.map((h, j) => {
                        const val = r[h] ?? r[parsed.headers[j]] ?? "";
                        if (h.toLowerCase().includes("crowd")) {
                          return <td key={j} style={tdStyle}><strong>{formatCount(val)}</strong></td>;
                        }
                        if (h.toLowerCase().includes("alert")) {
                          return <td key={j} style={tdStyle}>{badgeForAlert(val) || <span style={{opacity:0.7}}>None</span>}</td>;
                        }
                        return <td key={j} style={tdStyle}>{val}</td>;
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div style={{ display: "flex", gap: 8, marginTop: 8, alignItems: "center" }}>
            <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0} style={smallControlStyle}>◀ Prev</button>
            <div style={{ fontSize: 12, opacity: 0.9 }}>
              Page {page + 1} / {totalPages} — {parsed.rows.length} rows
            </div>
            <button onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))} disabled={page >= totalPages - 1} style={smallControlStyle}>Next ▶</button>
            <div style={{ marginLeft: "auto", fontSize: 12, opacity: 0.8 }}>Showing {pageRows.length} rows</div>
          </div>
        </>
      )}
    </div>
  );
}

const buttonStyle = {
  padding: "6px 10px",
  borderRadius: 6,
  background: "rgba(255,255,255,0.04)",
  border: "1px solid rgba(255,255,255,0.04)",
  color: "inherit",
  cursor: "pointer",
  fontSize: 12
};

const tdStyle = {
  padding: "8px 10px",
  fontSize: 12,
  verticalAlign: "middle",
  borderBottom: "1px dashed rgba(255,255,255,0.03)",
};

const smallControlStyle = {
  padding: "4px 8px",
  borderRadius: 6,
  background: "transparent",
  border: "1px solid rgba(255,255,255,0.04)",
  color: "inherit",
  cursor: "pointer",
  fontSize: 12
};
