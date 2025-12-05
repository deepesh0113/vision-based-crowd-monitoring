import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, CartesianGrid, Legend, Cell, ReferenceLine,
} from 'recharts';

function useWindowWidth() {
  const [width, setWidth] = useState(window.innerWidth);
  useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  return width;
}

function Dashboard() {
  const navigate = useNavigate();
  const [csvFile, setCsvFile] = useState(null);

  // Filtered data used by day-wise graphs
  const [graphData, setGraphData] = useState([]);
  const [perSecondData, setPerSecondData] = useState([]);
  const [distributionData, setDistributionData] = useState([]); // NEW: Count distribution
  const [summary, setSummary] = useState(null);

  // Raw unfiltered data for filters
  const [rawGraphData, setRawGraphData] = useState([]);
  const [rawPerSecondData, setRawPerSecondData] = useState([]);

  // Day-wise filter
  const [availableDates, setAvailableDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState('');

  // Month-wise filter
  const [availableMonths, setAvailableMonths] = useState([]);
  const [selectedMonth, setSelectedMonth] = useState('');
  const [monthDailyData, setMonthDailyData] = useState([]);

  // Year-wise filter
  const [availableYears, setAvailableYears] = useState([]);
  const [selectedYear, setSelectedYear] = useState('');
  const [yearMonthlyData, setYearMonthlyData] = useState([]);

  // View mode: day, month, year
  const [viewMode, setViewMode] = useState('day');

  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [selectedHistory, setSelectedHistory] = useState(null);

  // Thresholds
  const [minThreshold, setMinThreshold] = useState('');
  const [maxThreshold, setMaxThreshold] = useState('');

  // Details toggles
  const [showTimeDetails, setShowTimeDetails] = useState(false);
  const [showPerSecondDetails, setShowPerSecondDetails] = useState(false);
  const [showDistributionDetails, setShowDistributionDetails] = useState(false); // NEW
  const [showMonthDetails, setShowMonthDetails] = useState(false);
  const [showMonthDetails2, setShowMonthDetails2] = useState(false);
  const [showMonthDetails3, setShowMonthDetails3] = useState(false);
  const [showYearDetails, setShowYearDetails] = useState(false);
  const [showYearDetails2, setShowYearDetails2] = useState(false);
  const [showYearDetails3, setShowYearDetails3] = useState(false);

  const width = useWindowWidth();
  const isMobile = width < 700;

  const parsedMin = parseFloat(minThreshold);
  const parsedMax = parseFloat(maxThreshold);
  const thresholdsActive = !Number.isNaN(parsedMin) && !Number.isNaN(parsedMax);

  // Theme colors
  const BG = '#020617';
  const BLUE = '#3b82f6';
  const CYAN = '#0ea5e9';
  const GREEN = '#22c55e';
  const RED = '#ef4444';
  const AMBER = '#facc15';
  const LENSALERT = '#a855f7'; // violet for lenscoveredorextremelydark
  const FREEZEALERT = '#f97316'; // orange for camerafrozen

  const styles = {
    page: {
      minHeight: '100vh',
      backgroundColor: BG,
      fontFamily: 'Inter, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
      display: 'flex',
      justifyContent: 'center',
      padding: isMobile ? '18px 4vw 32px' : '28px 40px',
      boxSizing: 'border-box',
    },
    content: {
      width: '100%',
      maxWidth: '1120px',
      display: 'flex',
      flexDirection: 'column',
      gap: isMobile ? '16px' : '20px',
      animation: 'fadeIn 0.5s ease-in',
    },
    topBar: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'flex-start',
      gap: '16px',
      flexWrap: 'wrap',
    },
    backBtn: {
      borderRadius: '999px',
      border: '1px solid #1e293b',
      backgroundColor: 'transparent',
      color: '#9ca3af',
      padding: '7px 16px',
      fontSize: '13px',
      cursor: 'pointer',
    },
    brandBlock: {
      display: 'flex',
      flexDirection: 'column',
      gap: '2px',
      textAlign: 'left',
    },
    brand: {
      fontSize: '11px',
      textTransform: 'uppercase',
      letterSpacing: '0.23em',
      color: '#6b7280',
    },
    header: {
      fontWeight: '600',
      fontSize: isMobile ? '1.25rem' : '1.6rem',
      color: '#e5e7eb',
    },
    subtitle: {
      fontSize: '12px',
      color: '#9ca3af',
    },
    theoryCard: {
      background: 'radial-gradient(circle at top left, #020617, #020617 60%)',
      borderRadius: '18px',
      padding: isMobile ? '14px 14px 18px' : '18px',
      boxShadow: '0 24px 52px rgba(15,23,42,0.9)',
      border: '1px solid #111827',
    },
    theoryTitle: {
      fontSize: '14px',
      textTransform: 'uppercase',
      letterSpacing: '0.18em',
      color: '#9ca3af',
      marginBottom: '8px',
    },
    theoryHeading: {
      fontSize: '15px',
      fontWeight: '600',
      color: '#e5e7eb',
      marginBottom: '8px',
    },
    theoryText: {
      fontSize: '13px',
      color: '#9ca3af',
      lineHeight: '1.65',
    },
    theoryList: {
      marginTop: '10px',
      paddingLeft: '18px',
      fontSize: '13px',
      color: '#9ca3af',
      lineHeight: '1.6',
    },
    mainRow: {
      display: 'flex',
      flexDirection: isMobile ? 'column' : 'row',
      gap: '18px',
      alignItems: 'flex-start',
    },
    leftCol: {
      flex: isMobile ? 'unset' : '0 0 280px',
      width: isMobile ? '100%' : '280px',
      display: 'flex',
      flexDirection: 'column',
      gap: '14px',
    },
    rightCol: {
      flex: '1',
      display: 'flex',
      flexDirection: 'column',
      gap: '14px',
    },
    historyCard: {
      background: 'radial-gradient(circle at top, #020617, #020617 70%)',
      borderRadius: '18px',
      padding: isMobile ? '14px 14px 16px' : '16px',
      boxShadow: '0 22px 50px rgba(15,23,42,0.9)',
      border: '1px solid #111827',
    },
    historyTitle: {
      fontSize: '14px',
      color: '#93c5fd',
      fontWeight: '600',
      marginBottom: '6px',
    },
    historySub: {
      fontSize: '12px',
      color: '#6b7280',
      marginBottom: '10px',
    },
    historyList: {
      margin: '0',
      padding: '0',
      listStyle: 'none',
      maxHeight: '260px',
      overflowY: 'auto',
    },
    historyItem: {
      padding: '7px 0',
      borderBottom: '1px solid rgba(15,23,42,0.9)',
    },
    historyName: {
      fontSize: '13px',
      color: '#e5e7eb',
      marginBottom: '2px',
      wordBreak: 'break-all',
    },
    historyMeta: {
      fontSize: '11px',
      color: '#9ca3af',
    },
    historyEmpty: {
      fontSize: '12px',
      color: '#6b7280',
      marginTop: '6px',
    },
    graphBox: {
      background: 'radial-gradient(circle at top right, #020617, #020617 70%)',
      borderRadius: '18px',
      border: '1px solid #111827',
      boxShadow: '0 24px 52px rgba(15,23,42,0.95)',
      padding: isMobile ? '16px 12px 20px' : '18px 24px',
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
    },
    graphTitle: {
      fontSize: '14px',
      color: BLUE,
      fontWeight: '600',
    },
    graphSubtitle: {
      fontSize: '12px',
      color: '#6b7280',
    },
    fileInput: {
      marginTop: '8px',
      marginBottom: '10px',
      padding: '7px',
      borderRadius: '10px',
      background: BG,
      color: '#e5e7eb',
      border: '1px solid #111827',
      fontSize: '13px',
    },
    resetBtn: {
      backgroundColor: '#10b981',
      color: '#fff',
      padding: '8px 20px',
      borderRadius: '999px',
      cursor: 'pointer',
      border: 'none',
      fontWeight: '600',
    },
    actionBtn: {
      backgroundColor: CYAN,
      color: '#0b1120',
      cursor: 'pointer',
      border: 'none',
      fontWeight: '600',
      padding: '7px 16px',
      borderRadius: '999px',
      fontSize: '13px',
    },
    selectedFile: {
      marginTop: '8px',
      fontSize: '12px',
      color: '#93c5fd',
      wordBreak: 'break-all',
    },
    graphArea: {
      width: '100%',
      height: '320px',
      marginTop: '4px',
    },
    graphPlaceholder: {
      flex: '1',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '13px',
      color: CYAN,
      borderRadius: '12px',
      border: '1px dashed #1e293b',
    },
    summaryRow: {
      display: 'flex',
      flexWrap: 'wrap',
      gap: '10px',
      fontSize: '12px',
      color: '#9ca3af',
      marginTop: '4px',
    },
    summaryChip: {
      padding: '4px 10px',
      borderRadius: '999px',
      border: '1px solid #1e293b',
    },
    thresholdRow: {
      display: 'flex',
      flexWrap: 'wrap',
      gap: '8px',
      marginTop: '10px',
      alignItems: 'center',
      fontSize: '12px',
      color: '#9ca3af',
    },
    thresholdInput: {
      width: '90px',
      padding: '4px 8px',
      borderRadius: '999px',
      border: '1px solid #1f2937',
      backgroundColor: '#020617',
      color: '#e5e7eb',
      fontSize: '12px',
      outline: 'none',
    },
    detailBtn: {
      alignSelf: 'flex-end',
      marginTop: '4px',
      padding: '4px 10px',
      fontSize: '11px',
      borderRadius: '999px',
      border: '1px solid #1f2937',
      backgroundColor: '#020617',
      color: '#9ca3af',
      cursor: 'pointer',
    },
    detailPanel: {
      marginTop: '8px',
      padding: '8px 10px',
      borderRadius: '12px',
      border: '1px solid #1f2937',
      backgroundColor: '#020617',
      fontSize: '11px',
      maxHeight: '160px',
      overflowY: 'auto',
      lineHeight: '1.5',
    },
    detailSectionTitle: {
      fontWeight: '600',
      marginTop: '4px',
      marginBottom: '2px',
      fontSize: '11px',
    },
    dateSelect: {
      padding: '4px 8px',
      borderRadius: '999px',
      border: '1px solid #1f2937',
      backgroundColor: '#020617',
      color: '#e5e7eb',
      fontSize: '12px',
      outline: 'none',
    },
    filterBtn: {
      backgroundColor: BLUE,
      color: '#f9fafb',
      border: 'none',
      borderRadius: '999px',
      padding: '6px 14px',
      fontSize: '12px',
      cursor: 'pointer',
      fontWeight: '500',
    },
    viewToggleRow: {
      display: 'flex',
      flexWrap: 'wrap',
      gap: '8px',
      marginTop: '10px',
      alignItems: 'center',
      fontSize: '12px',
      color: '#9ca3af',
    },
    viewToggleBtn: {
      padding: '5px 12px',
      borderRadius: '999px',
      border: '1px solid #1f2937',
      backgroundColor: '#020617',
      color: '#9ca3af',
      fontSize: '12px',
      cursor: 'pointer',
    },
    viewToggleBtnActive: {
      backgroundColor: BLUE,
      color: '#f9fafb',
      borderColor: BLUE,
    },
  };

  // custom dot renderer for line charts based on thresholds + alert column
  const renderColoredDot = (props) => {
    const { cx, cy, payload } = props;
    const value = payload.count;
    const alertType = payload.alert ? String(payload.alert).trim() : "";

    let fill = "#60a5fa";  // default blue
    let radius = 3;

    // 1) camera alerts override thresholds
    if (alertType === "lens_covered_or_extremely_dark") {
      fill = LENSALERT;
      radius = 5;
    } else if (alertType === "camera_frozen") {
      fill = FREEZEALERT;
      radius = 5;
    } else if (thresholdsActive) {
      // 2) if no alert string, fall back to threshold colors
      if (value > parsedMax) fill = RED;                    // above alert
      else if (value >= parsedMin && value <= parsedMax) fill = GREEN; // safe
      else fill = AMBER;                                    // below min
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


  // Filter helpers
  // Day-wise filter (single date)
  const applyDateFilter = (date, recordsSrc, perSecondSrc) => {
    if (!date) {
      setGraphData(recordsSrc);
      setPerSecondData(perSecondSrc);
      computeDistribution(recordsSrc); // NEW: compute distribution for all dates
      return;
    }

    const filteredRecords = recordsSrc.filter(r => r.date === date);
    const filteredPerSecond = perSecondSrc.filter(r => r.date === date);
    setGraphData(filteredRecords);
    setPerSecondData(filteredPerSecond);
    computeDistribution(filteredRecords); // NEW: compute distribution for selected date
  };

  // NEW: Compute count distribution for histogram
  const computeDistribution = (records) => {
    if (!records.length) {
      setDistributionData([]);
      return;
    }

    // Bin counts into 10 buckets
    const minCount = Math.min(...records.map(r => r.count));
    const maxCount = Math.max(...records.map(r => r.count));
    const binSize = (maxCount - minCount) / 10 || 1;

    const bins = Array(10).fill(0);
    records.forEach(r => {
      const binIndex = Math.min(Math.floor((r.count - minCount) / binSize), 9);
      bins[binIndex]++;
    });

    const distribution = bins.map((count, index) => ({
      bin: index + 1,
      range: `${Math.round(minCount + index * binSize)}-${Math.round(minCount + (index + 1) * binSize)}`,
      frequency: count,
      count: minCount + (index + 0.5) * binSize
    }));

    setDistributionData(distribution);
  };

  // Month-wise filter (aggregate per date inside selected month)
  const applyMonthFilter = (monthKey, recordsSrc) => {
    if (!monthKey) {
      setMonthDailyData([]);
      return;
    }

    // recordsSrc has date, timestamp, count
    const monthRecords = recordsSrc.filter(r => r.date.startsWith(monthKey));
    const dayMap = {};

    monthRecords.forEach(r => {
      const d = r.date; // YYYY-MM-DD
      if (!dayMap[d]) {
        dayMap[d] = { date: d, dayLabel: d.slice(8, 10), total: 0, max: -Infinity, n: 0 };
      }
      dayMap[d].total += r.count;
      dayMap[d].max = Math.max(dayMap[d].max, r.count);
      dayMap[d].n += 1;
    });

    const dailyArr = Object.values(dayMap)
      .map(d => ({
        date: d.date,
        day: d.dayLabel,
        avg_count: d.total / d.n,
        max_count: d.max,
        total_count: d.total,
      }))
      .sort((a, b) => a.date.localeCompare(b.date));

    setMonthDailyData(dailyArr);



    // Year-wise filter (aggregate per month inside selected year)
    const applyYearFilter = (yearKey, recordsSrc) => {
      if (!yearKey) {
        setYearMonthlyData([]);
        return;
      }

      // recordsSrc has date, timestamp, count
      const yearRecords = recordsSrc.filter(r => r.date.startsWith(yearKey));
      const monthMap = {};

      yearRecords.forEach(r => {
        const monthKey = r.date.slice(0, 7); // YYYY-MM
        const shortMonth = monthKey.slice(5, 7); // 01, 02, ...
        if (!monthMap[monthKey]) {
          monthMap[monthKey] = {
            month: shortMonth, // for x-axis
            monthLabel: monthKey, // full YYYY-MM
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
        .map(m => ({
          month: m.month,
          monthLabel: m.monthLabel,
          avg_count: m.total / m.n,
          max_count: m.max,
          total_count: m.total,
        }))
        .sort((a, b) => a.monthLabel.localeCompare(b.monthLabel));

      setYearMonthlyData(monthlyArr);

    };

    // Upload handler
    const handleCSVUpload = async () => {
      if (!csvFile) {
        alert('Please select a CSV file first!');
        return;
      }

      const formData = new FormData();
      formData.append('file', csvFile);
      setLoading(true);

      try {
        const res = await fetch('http://127.0.0.1:8000/dashboard/upload-csv', {
          method: 'POST',
          body: formData,
        });
        const data = await res.json();

        if (data.status === 'error') {
          alert(data.message || 'Error processing CSV');
          setLoading(false);
          return;
        }

        const records = data.records;
        const perSecondAll = data.persecond;
        const summaryObj = data.summary || null;

        // Store raw unfiltered
        setRawGraphData(records);
        setRawPerSecondData(perSecondAll);

        // Collect unique dates from records
        const uniqueDates = Array.from(new Set(records.map(r => r.date).filter(Boolean)));
        setAvailableDates(uniqueDates);
        const initialDate = uniqueDates[0];
        setSelectedDate(initialDate);

        // Collect unique months from records (YYYY-MM)
        const uniqueMonths = Array.from(
          new Set(records.map(r => r.date ? r.date.slice(0, 7) : null).filter(Boolean))
        );
        uniqueMonths.sort();
        setAvailableMonths(uniqueMonths);
        const initialMonth = uniqueMonths[0];
        setSelectedMonth(initialMonth);

        // Collect unique years
        const uniqueYears = Array.from(
          new Set(records.map(r => r.date ? r.date.slice(0, 4) : null).filter(Boolean))
        );
        uniqueYears.sort();
        setAvailableYears(uniqueYears);
        const initialYear = uniqueYears[0];
        setSelectedYear(initialYear);

        // Default to day view
        setViewMode('day');

        // Apply initial day filter
        applyDateFilter(initialDate, records, perSecondAll);

        // Compute month aggregation for initial month (for when user switches view)
        applyMonthFilter(initialMonth, records);
        applyYearFilter(initialYear, records);

        setSummary(summaryObj);

        // History entry
        const minVal = summaryObj && typeof summaryObj.mincount !== 'undefined' ? summaryObj.mincount : null;
        const maxVal = summaryObj && typeof summaryObj.maxcount !== 'undefined' ? summaryObj.maxcount : null;
        const newHistoryItem = {
          name: csvFile.name,
          uploadedAt: new Date().toLocaleString(),
          points: records.length,
          minCount: minVal,
          maxCount: maxVal,
        };
        setHistory(prev => [newHistoryItem, ...prev]);
        setSelectedHistory(newHistoryItem);

      } catch (err) {
        console.error('Error:', err);
        alert('Failed to upload CSV');
      } finally {
        setLoading(false);
      }
    };

    const resetView = () => {
      setCsvFile(null);
      setGraphData([]);
      setPerSecondData([]);
      setDistributionData([]);
      setSummary(null);
      setMinThreshold('');
      setMaxThreshold('');
      setShowTimeDetails(false);
      setShowPerSecondDetails(false);
      setShowDistributionDetails(false);
      setRawGraphData([]);
      setRawPerSecondData([]);
      setAvailableDates([]);
      setSelectedDate('');
      setAvailableMonths([]);
      setSelectedMonth('');
      setMonthDailyData([]);
      setViewMode('day');
      setTimeout(() => window.location.reload(), 150);
    };
    // Helpers for detail panels

    // Day-wise: time-series
    const getTimeAlerts = () =>
      thresholdsActive ? graphData.filter(p => p.count > parsedMax) : [];

    const getTimeSafe = () =>
      thresholdsActive
        ? graphData.filter(p => p.count >= parsedMin && p.count <= parsedMax)
        : [];

    // Day-wise: per-second
    const getPerSecondAlerts = () =>
      thresholdsActive
        ? perSecondData.filter(p => p.avg_count > parsedMax)
        : [];

    const getPerSecondSafe = () =>
      thresholdsActive
        ? perSecondData.filter(
          p => p.avg_count >= parsedMin && p.avg_count <= parsedMax
        )
        : [];

    // Day-wise: distribution (histogram)
    const getDistributionAlerts = () =>
      thresholdsActive
        ? distributionData.filter(d => d.count > parsedMax)
        : [];

    const getDistributionSafe = () =>
      thresholdsActive
        ? distributionData.filter(
          d => d.count >= parsedMin && d.count <= parsedMax
        )
        : [];

    // Month-wise detail helpers (use monthDailyData)
    const getMonthAlertDays = () =>
      thresholdsActive
        ? monthDailyData.filter(d => d.max_count > parsedMax)
        : [];

    const getMonthSafeDays = () =>
      thresholdsActive
        ? monthDailyData.filter(
          d => d.avg_count >= parsedMin && d.avg_count <= parsedMax
        )
        : [];

    // Year-wise detail helpers (use yearMonthlyData)
    const getYearAlertMonths = () =>
      thresholdsActive
        ? yearMonthlyData.filter(m => m.max_count > parsedMax)
        : [];

    const getYearSafeMonths = () =>
      thresholdsActive
        ? yearMonthlyData.filter(
          m => m.avg_count >= parsedMin && m.avg_count <= parsedMax
        )
        : [];

    return (
      <div style={styles.page}>
        <div style={styles.content}>
          {/* TOP BAR */}
          <div style={styles.topBar}>
            <button style={styles.backBtn} onClick={() => navigate(-1)}>
              Back
            </button>
            <div style={styles.brandBlock}>
              <div style={styles.brand}>VigilNet Dashboard</div>
              <div style={styles.header}>Historical Crowd Analytics</div>
              <div style={styles.subtitle}>
                Upload model outputs as CSV and explore time-based crowd trends for research, debugging, and reporting.
              </div>
            </div>
          </div>

          {/* THEORY CARD */}
          <div style={styles.theoryCard}>
            <div style={styles.theoryTitle}>Why this dashboard matters</div>
            <div style={styles.theoryHeading}>From raw CSV logs to decisions you can defend</div>
            <p style={styles.theoryText}>
              VigilNet's historical dashboard turns <code>date,timestamp_ns,count,alert</code> logs into a visual narrative.
              Zoom into a single day or step back to see how whole months behave, spotting spikes, quiet days, and recurring patterns.
            </p>
            <ul style={styles.theoryList}>
              <li><strong>Day-wise:</strong> inspect how counts evolve within a day (timestamp-wise, per-second, distribution)</li>
              <li><strong>Month-wise:</strong> summarise how each day in a month behaves using average, peak, and total crowd counts</li>
            </ul>
          </div>

          {/* MAIN ROW */}
          <div style={styles.mainRow}>
            {/* LEFT COLUMN - HISTORY */}
            <div style={styles.leftCol}>
              <div style={styles.historyCard}>
                <div style={styles.historyTitle}>Upload history</div>
                <div style={styles.historySub}>Recent CSV files processed on this dashboard.</div>
                {history.length === 0 ? (
                  <div style={styles.historyEmpty}>
                    No files analyzed yet. Upload a CSV to start building your history.
                  </div>
                ) : (
                  <ul style={styles.historyList}>
                    {history.map((item, idx) => (
                      <li
                        key={idx}
                        style={{
                          ...styles.historyItem,
                          cursor: 'pointer',
                          backgroundColor: selectedHistory === item ? 'rgba(15,23,42,0.8)' : 'transparent',
                        }}
                        onClick={() => setSelectedHistory(item)}
                      >
                        <div style={styles.historyName}>{item.name}</div>
                        <div style={styles.historyMeta}>
                          {item.points} points • {item.uploadedAt}
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
                {selectedHistory && (
                  <div style={{ marginTop: '10px', paddingTop: '8px', borderTop: '1px solid rgba(15,23,42,0.9)', fontSize: '12px', color: '#9ca3af' }}>
                    <div>Min count: {selectedHistory.minCount?.toFixed ? selectedHistory.minCount.toFixed(2) : selectedHistory.minCount}</div>
                    <div>Max count: {selectedHistory.maxCount?.toFixed ? selectedHistory.maxCount.toFixed(2) : selectedHistory.maxCount}</div>
                    <div>Total points: {selectedHistory.points}</div>
                  </div>
                )}
              </div>
            </div>

            {/* RIGHT COLUMN - UPLOAD & GRAPHS */}
            <div style={styles.rightCol}>
              {/* CSV Upload + Thresholds + View Filters */}
              <div style={styles.graphBox}>
                <div style={styles.graphTitle}>Upload CSV file</div>
                <div style={styles.graphSubtitle}>
                  Expected format<br />
                  <code>date,timestamp_ns,count,alert</code><br /><br />
                  Legend additions:
                  <span style={{ color: LENSALERT, fontWeight: 'bold' }}>  ●</span> violet dot = lens covered or extremely dark<br />
                  <span style={{ color: FREEZEALERT, fontWeight: 'bold' }}>  ●</span> orange dot = camera frozen
                </div>
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => setCsvFile(e.target.files[0])}
                  style={styles.fileInput}
                />
                <div style={{ display: 'flex', gap: '10px', marginTop: '5px' }}>
                  <button onClick={handleCSVUpload} style={styles.actionBtn} disabled={loading}>
                    {loading ? 'Processing...' : 'Upload & Process'}
                  </button>
                  <button onClick={resetView} style={styles.resetBtn}>
                    Clear & Refresh
                  </button>
                </div>
                {csvFile && (
                  <div style={styles.selectedFile}>Selected file: {csvFile.name}</div>
                )}

                {/* Threshold controls */}
                <div style={styles.thresholdRow}>
                  <span>Thresholds</span>
                  <span>Min safe start</span>
                  <input
                    type="number"
                    step="0.01"
                    value={minThreshold}
                    onChange={(e) => setMinThreshold(e.target.value)}
                    style={styles.thresholdInput}
                    placeholder="e.g. 100"
                  />
                  <span>Max alert</span>
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
                  <span>View mode</span>
                  <button
                    style={{
                      ...styles.viewToggleBtn,
                      ...(viewMode === 'day' ? styles.viewToggleBtnActive : {}),
                    }}
                    onClick={() => {
                      setViewMode('day');
                      applyDateFilter(selectedDate, rawGraphData, rawPerSecondData);
                    }}
                  >
                    Day-wise
                  </button>
                  <button
                    style={{
                      ...styles.viewToggleBtn,
                      ...(viewMode === 'month' ? styles.viewToggleBtnActive : {}),
                    }}
                    onClick={() => {
                      setViewMode('month');
                      applyMonthFilter(selectedMonth, rawGraphData);
                    }}
                  >
                    Month-wise
                  </button>
                  <button
                    style={{
                      ...styles.viewToggleBtn,
                      ...(viewMode === 'year' ? styles.viewToggleBtnActive : {}),
                    }}
                    onClick={() => {
                      setViewMode('year');
                      applyYearFilter(selectedYear, rawGraphData);
                    }}
                  >
                    Year-wise
                  </button>
                </div>

                {/* Day-wise vs Month-wise filters */}
                {viewMode === 'day' && availableDates.length > 0 && (
                  <div style={styles.thresholdRow}>
                    <span>Filter by date</span>
                    <select
                      value={selectedDate}
                      onChange={(e) => {
                        const value = e.target.value;
                        setSelectedDate(value);
                        applyDateFilter(value, rawGraphData, rawPerSecondData);
                      }}
                      style={styles.dateSelect}
                    >
                      <option value="">All dates</option>
                      {availableDates.map(d => (
                        <option key={d} value={d}>{d}</option>
                      ))}
                    </select>
                    <button
                      style={styles.filterBtn}
                      onClick={() => applyDateFilter(selectedDate, rawGraphData, rawPerSecondData)}
                    >
                      FILTER
                    </button>
                  </div>
                )}

                {viewMode === 'month' && availableMonths.length > 0 && (
                  <div style={styles.thresholdRow}>
                    <span>Filter by month</span>
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
                      {availableMonths.map(m => (
                        <option key={m} value={m}>{m}</option>
                      ))}
                    </select>
                    <button
                      style={styles.filterBtn}
                      onClick={() => applyMonthFilter(selectedMonth, rawGraphData)}
                    >
                      FILTER
                    </button>
                  </div>
                )}

                {viewMode === 'year' && availableYears.length > 0 && (
                  <div style={styles.thresholdRow}>
                    <span>Filter by year</span>
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
                      {availableYears.map(y => (
                        <option key={y} value={y}>{y}</option>
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

                {/* Summary stats */}
                {summary && (
                  <div style={styles.summaryRow}>
                    <span style={styles.summaryChip}>Min count: {summary.mincount.toFixed(2)}</span>
                    <span style={styles.summaryChip}>Max count: {summary.maxcount.toFixed(2)}</span>
                    <span style={styles.summaryChip}>Mean count: {summary.meancount.toFixed(2)}</span>
                    <span style={styles.summaryChip}>Points: {summary.numpoints}</span>
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

                  {/* DAY-WISE GRAPH 3: NEW Count Distribution Histogram (replaces frame graph) */}
                  <div style={styles.graphBox}>
                    <div style={styles.graphTitle}>Count distribution histogram (selected day)</div>
                    <div style={styles.graphSubtitle}>
                      Frequency of crowd counts in 10 bins. Reveals distribution patterns and outliers.
                    </div>
                    <button
                      style={styles.detailBtn}
                      onClick={() => setShowDistributionDetails(prev => !prev)}
                    >
                      {showDistributionDetails ? 'Hide details' : 'Show details'}
                    </button>
                    <div style={styles.graphArea}>
                      {distributionData.length === 0 ? (
                        <div style={styles.graphPlaceholder}>
                          Upload a CSV and select a date to see count distribution.
                        </div>
                      ) : (
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={distributionData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                              dataKey="range"
                              tick={{ fill: '#93c5fd', fontSize: 11 }}
                              label={{ value: 'Count Range', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 11 }}
                            />
                            <YAxis
                              tick={{ fill: '#93c5fd', fontSize: 11 }}
                              label={{ value: 'Frequency', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 11 }}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: '#020617',
                                border: '1px solid #1e293b',
                                borderRadius: '8px',
                                fontSize: '11px',
                              }}
                              labelStyle={{ color: '#e5e7eb' }}
                            />
                            <Legend />
                            <Bar dataKey="frequency" name="Occurrences" fill="#60a5fa">
                              {distributionData.map((entry, index) => {
                                const v = entry.count;
                                let fillColor = '#60a5fa';
                                if (thresholdsActive) {
                                  if (v > parsedMax) fillColor = RED;
                                  else if (v >= parsedMin && v <= parsedMax) fillColor = GREEN;
                                  else fillColor = AMBER;
                                }
                                return <Cell key={`dist-${index}`} fill={fillColor} />;
                              })}
                            </Bar>
                            {thresholdsActive && (
                              <>
                                <ReferenceLine y={parsedMin} stroke={GREEN} strokeDasharray="3 3" label={{ position: 'top' }} />
                                <ReferenceLine y={parsedMax} stroke={RED} strokeDasharray="3 3" label={{ position: 'top' }} />
                              </>
                            )}
                          </BarChart>
                        </ResponsiveContainer>
                      )}
                    </div>
                    {showDistributionDetails && thresholdsActive && (
                      <div style={styles.detailPanel}>
                        <div style={styles.detailSectionTitle}>Alert bins (avg count &gt; max)</div>
                        {getDistributionAlerts().length === 0 ? (
                          <div>No alert bins.</div>
                        ) : (
                          getDistributionAlerts().map((d, idx) => (
                            <div key={`da-${idx}`}>
                              range = {d.range}, freq = {d.frequency}, avg ={" "}
                              {d.count.toFixed(1)}
                            </div>
                          ))
                        )}

                        <div style={styles.detailSectionTitle}>
                          Safe bins (avg between min &amp; max)
                        </div>
                        {getDistributionSafe().length === 0 ? (
                          <div>No safe bins for current thresholds.</div>
                        ) : (
                          getDistributionSafe().map((d, idx) => (
                            <div key={`ds-${idx}`}>
                              range = {d.range}, freq = {d.frequency}, avg ={" "}
                              {d.count.toFixed(1)}
                            </div>
                          ))
                        )}
                      </div>
                    )}

                  </div>
                </>
              ) : viewMode === "month" ? (
                <>
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
      </div>
    );
  }
}
export default Dashboard;

