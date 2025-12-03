// import React, { useState, useEffect } from 'react';

// // Custom hook for window width
// function useWindowWidth() {
//   const [width, setWidth] = useState(window.innerWidth);
//   useEffect(() => {
//     const handleResize = () => setWidth(window.innerWidth);
//     window.addEventListener('resize', handleResize);
//     return () => window.removeEventListener('resize', handleResize);
//   }, []);
//   return width;
// }

// function Report() {
//   const width = useWindowWidth();
//   const isMobile = width < 700;

//   const images = ['/project.jpg', '/project1.jpg', '/project2.webp'];
//   const [currentImg, setCurrentImg] = useState(0);

//   useEffect(() => {
//     const intervalId = setInterval(() => {
//       setCurrentImg((prev) => (prev + 1) % images.length);
//     }, 5000);
//     return () => clearInterval(intervalId);
//   }, [images.length]);

//   const styles = {
//     container: {
//       backgroundColor: '#111',
//       color: '#fff',
//       minHeight: '100vh',
//       fontFamily: "'Segoe UI', Arial, sans-serif",
//       padding: isMobile ? '16px 8px' : '40px 36px',
//       maxWidth: isMobile ? '99vw' : '900px',
//       margin: 'auto',
//       lineHeight: 1.6,
//     },
//     header: {
//       display: isMobile ? 'block' : 'flex',
//       alignItems: isMobile ? 'flex-start' : 'center',
//       marginBottom: isMobile ? '18px' : '30px',
//       textAlign: isMobile ? 'center' : 'left',
//     },
//     image: {
//       width: isMobile ? '100%' : '120px',   // Full width on mobile, fixed width on desktop
//       height: isMobile ? 'auto' : '120px',  // Height auto for aspect ratio, fixed on desktop
//       maxWidth: '340px',                    // (Optional) limit max width for very large screens
//       borderRadius: '15px',
//       boxShadow: '0 0 28px #1e90ff88',
//       marginRight: isMobile ? '0' : '24px',
//       marginBottom: isMobile ? '16px' : '0',
//       objectFit: 'cover',
//       flexShrink: 0,
//       display: 'block',                     // Block for image to expand full width in parent
//     },
//     title: {
//       fontSize: isMobile ? '1.6rem' : '2.8rem',
//       fontWeight: 'bold',
//       color: '#1e90ff',
//       textTransform: 'uppercase',
//       textShadow: '0 0 18px rgba(30,144,255,0.7)',
//     },
//     sectionTitle: {
//       fontSize: isMobile ? '1.1rem' : '1.75rem',
//       fontWeight: '700',
//       marginTop: isMobile ? '28px' : '40px',
//       marginBottom: isMobile ? '10px' : '14px',
//       color: '#1e90ffcc',
//       borderBottom: '2px solid #1e90ff',
//       paddingBottom: '6px',
//     },
//     paragraph: {
//       fontSize: isMobile ? '0.98rem' : '1.1rem',
//       marginBottom: '16px',
//       color: '#ddd',
//       textAlign: isMobile ? 'left' : 'inherit',
//     },
//     list: {
//       marginLeft: isMobile ? '14px' : '20px',
//       marginBottom: '16px',
//       color: '#ccc',
//       fontSize: isMobile ? '0.95rem' : '1.05rem',
//       textAlign: isMobile ? 'left' : 'inherit',
//     },
//     footer: {
//       textAlign: 'center',
//       color: '#bbb',
//       marginTop: '38px',
//       fontSize: isMobile ? '0.95rem' : '1.05rem',
//       letterSpacing: '0.8px',
//     }
//   };

//   return (
//     <div style={styles.container}>
//       <header style={styles.header}>
//         <img src={images[currentImg]} alt="Project screenshot" style={styles.image} />
//         <h1 style={styles.title}>VigilNet Crowd Management System Report</h1>
//       </header>

//       <section>
//         <h2 style={styles.sectionTitle}>Abstract</h2>
//         <p style={styles.paragraph}>
//           VigilNet is an AI-powered real-time crowd management system designed to ensure safety during
//           large-scale gatherings by accurately monitoring crowd density, detecting anomalies, and
//           verifying camera integrity through tamper detection. This project integrates advanced deep learning
//           models and intuitive web-based dashboards offering critical alerts with minimal latency for rapid response.
//         </p>
//       </section>

//       <section>
//         <h2 style={styles.sectionTitle}>Introduction</h2>
//         <p style={styles.paragraph}>
//           In an era of increasing urbanization and public events, the need for effective crowd safety
//           solutions is paramount. VigilNet offers a robust automated surveillance solution that
//           overcomes the limitations of manual monitoring by leveraging AI algorithms for continuous,
//           precise crowd analysis and proactive hazard detection.
//         </p>
//       </section>

//       <section>
//         <h2 style={styles.sectionTitle}>Literature Review</h2>
//         <p style={styles.paragraph}>
//           Contemporary research features multiple computer vision approaches for crowd analysis,
//           such as YOLO for fast object detection and MCNN for density estimation. While these provide strong
//           foundations, they lack full real-time integration and tamper detection, limiting their practical
//           deployment. VigilNet builds on these to deliver an end-to-end operational system.
//         </p>
//       </section>

//       <section>
//         <h2 style={styles.sectionTitle}>Objectives</h2>
//         <ul style={styles.list}>
//           <li>Develop a real-time crowd density estimation framework.</li>
//           <li>Implement anomaly detection to flag potential hazards dynamically.</li>
//           <li>Detect and alert any camera tampering for continuous monitoring reliability.</li>
//           <li>Create a user-friendly dashboard with low-latency alert mechanisms.</li>
//           <li>Enable scalable deployment across multiple cameras and locations.</li>
//         </ul>
//       </section>

//       <section>
//         <h2 style={styles.sectionTitle}>System Architecture and Methodology</h2>
//         <p style={styles.paragraph}>
//           VigilNet processes live video streams through modular AI inference engines combining YOLOv8, ResNet50,
//           and Multi-Column CNN models. Alerts propagate via a Node.js backend, with users interacting through
//           a React frontend dashboard featuring live heatmaps, historical logs, and snapshot capture. The architecture
//           prioritizes modularity and low latency.
//         </p>
//       </section>

//       <section>
//         <h2 style={styles.sectionTitle}>Data and Models</h2>
//         <p style={styles.paragraph}>
//           The system leverages annotated crowd datasets for model training, achieving high accuracy in a variety of
//           environments by combining detection and density estimation. Models were optimized for efficient real-time
//           inference suitable for edge and cloud deployments.
//         </p>
//       </section>

//       <section>
//         <h2 style={styles.sectionTitle}>Implementation Details</h2>
//         <p style={styles.paragraph}>
//           Using React for frontend UI and Node.js backend, VigilNet integrates AI inference with alert management.
//           Features include multi-camera support, anomaly thresholds, snapshot functionality, timeline navigation,
//           and support for RTSP and cloud video sources.
//         </p>
//       </section>

//       <section>
//         <h2 style={styles.sectionTitle}>Feature Description</h2>
//         <ul style={styles.list}>
//           <li>Real-time crowd density heatmaps and numerical counts.</li>
//           <li>Anomaly detection based on motion and behavioral analysis.</li>
//           <li>Camera tamper detection with instant alerting.</li>
//           <li>Latency under 250 ms guaranteeing fast responses.</li>
//           <li>Multi-platform integrations including SMS, webhooks, and REST APIs.</li>
//         </ul>
//       </section>

//       <section>
//         <h2 style={styles.sectionTitle}>Results and Performance Evaluation</h2>
//         <p style={styles.paragraph}>
//           VigilNet demonstrated detection accuracy exceeding 99% and responsive alerting within required latencies.
//           Field trials validate effectiveness in varying crowd scenarios. Models and system components were benchmarked
//           for scalability considerations.
//         </p>
//       </section>

//       <section>
//         <h2 style={styles.sectionTitle}>Discussions and Challenges</h2>
//         <p style={styles.paragraph}>
//           Challenges involved handling occlusions in dense crowds, minimizing false positives in anomaly alerts, and
//           ensuring privacy compliance. Robust asynchronous processing and error handling were developed to maintain system resilience.
//         </p>
//       </section>

//       <section>
//         <h2 style={styles.sectionTitle}>Conclusions and Future Work</h2>
//         <p style={styles.paragraph}>
//           VigilNet presents a comprehensive solution for crowd safety leveraging AI. Future directions include
//           predictive analytics, enhanced privacy features, multi-camera synchronization, and further latency reduction
//           to improve real-world applicability and user experience.
//         </p>
//       </section>

//       {/* Footer line at the bottom, not sticky/fixed */}
//       <div style={styles.footer}>
//         © 2025 VigilNet Project Team
//       </div>
//     </div>
//   );
// }

// export default Report;
import React, { useState, useEffect } from "react";

// Custom hook for window width
function useWindowWidth() {
  const [width, setWidth] = useState(window.innerWidth);
  useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);
  return width;
}

function Report() {
  const width = useWindowWidth();
  const isMobile = width < 700;

  const images = ["/project.jpg", "/project1.jpg", "/project2.webp"];
  const [currentImg, setCurrentImg] = useState(0);

  useEffect(() => {
    const intervalId = setInterval(() => {
      setCurrentImg((prev) => (prev + 1) % images.length);
    }, 5000);
    return () => clearInterval(intervalId);
  }, [images.length]);

  const styles = {
    page: {
      backgroundColor: "#020617",
      color: "#e5e7eb",
      minHeight: "100vh",
      fontFamily:
        "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      padding: isMobile ? "18px 4vw 28px" : "28px 32px 40px",
      display: "flex",
      justifyContent: "center",
      boxSizing: "border-box",
    },
    container: {
      width: "100%",
      maxWidth: 960,
      margin: "0 auto",
      backgroundColor: "#020617",
      borderRadius: 18,
      border: "1px solid #1f2937",
      boxShadow: "0 22px 50px rgba(15,23,42,0.95)",
      padding: isMobile ? "18px 16px 22px" : "26px 26px 30px",
      lineHeight: 1.7,
      animation: "reportFadeIn 0.6s ease-in",
    },
    header: {
      display: isMobile ? "block" : "flex",
      alignItems: isMobile ? "flex-start" : "center",
      gap: isMobile ? 12 : 18,
      marginBottom: isMobile ? 16 : 20,
    },
    image: {
      width: isMobile ? "100%" : 130,
      height: isMobile ? "auto" : 130,
      maxWidth: 360,
      borderRadius: 16,
      boxShadow: "0 18px 40px rgba(15,23,42,0.9)",
      objectFit: "cover",
      flexShrink: 0,
    },
    headerText: {
      flex: 1,
      minWidth: 0,
    },
    label: {
      fontSize: 11,
      textTransform: "uppercase",
      letterSpacing: "0.18em",
      color: "#6b7280",
      marginBottom: 4,
    },
    title: {
      fontSize: isMobile ? "1.55rem" : "2.2rem",
      fontWeight: 600,
      color: "#e5e7eb",
      letterSpacing: "0.03em",
      marginBottom: 4,
    },
    subtitle: {
      fontSize: 13,
      color: "#9ca3af",
    },

    sectionTitle: {
      fontSize: isMobile ? "1.05rem" : "1.2rem",
      fontWeight: 600,
      marginTop: isMobile ? 18 : 22,
      marginBottom: 6,
      color: "#93c5fd",
      borderBottom: "1px solid #1f2937",
      paddingBottom: 4,
    },
    paragraph: {
      fontSize: isMobile ? 13 : 14,
      marginBottom: 10,
      color: "#d1d5db",
      textAlign: "left",
    },
    list: {
      marginLeft: isMobile ? 16 : 22,
      marginBottom: 10,
      color: "#d1d5db",
      fontSize: isMobile ? 13 : 14,
      paddingLeft: 0,
    },
    listItem: {
      marginBottom: 4,
    },
    footer: {
      textAlign: "center",
      color: "#6b7280",
      marginTop: isMobile ? 22 : 26,
      fontSize: 11,
      letterSpacing: "0.16em",
      textTransform: "uppercase",
    },
  };

  return (
    <>
      <style>{`
        @keyframes reportFadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      <div style={styles.page}>
        <div style={styles.container}>
          <header style={styles.header}>
            <img
              src={images[currentImg]}
              alt="Project screenshot"
              style={styles.image}
            />
            <div style={styles.headerText}>
              <div style={styles.label}>VigilNet</div>
              <h1 style={styles.title}>Crowd Management System – Report</h1>
              <div style={styles.subtitle}>
                Project overview, methodology, and evaluation summary.
              </div>
            </div>
          </header>

          <section>
            <h2 style={styles.sectionTitle}>Abstract</h2>
            <p style={styles.paragraph}>
              VigilNet is an AI-powered real-time crowd management system designed
              to ensure safety during large-scale gatherings by accurately
              monitoring crowd density, detecting anomalies, and verifying camera
              integrity through tamper detection. This project integrates advanced
              deep learning models and intuitive web-based dashboards, offering
              critical alerts with minimal latency for rapid response.
            </p>
          </section>

          <section>
            <h2 style={styles.sectionTitle}>Introduction</h2>
            <p style={styles.paragraph}>
              In an era of increasing urbanization and public events, the need for
              effective crowd safety solutions is paramount. VigilNet offers a
              robust automated surveillance solution that overcomes the limitations
              of manual monitoring by leveraging AI algorithms for continuous,
              precise crowd analysis and proactive hazard detection.
            </p>
          </section>

          <section>
            <h2 style={styles.sectionTitle}>Literature Review</h2>
            <p style={styles.paragraph}>
              Contemporary research features multiple computer vision approaches
              for crowd analysis, such as YOLO for fast object detection and MCNN
              for density estimation. While these provide strong foundations, many
              lack full real-time integration and tamper detection, limiting their
              practical deployment. VigilNet builds on these ideas to deliver an
              end-to-end operational system.
            </p>
          </section>

          <section>
            <h2 style={styles.sectionTitle}>Objectives</h2>
            <ul style={styles.list}>
              <li style={styles.listItem}>
                Develop a real-time crowd density estimation framework.
              </li>
              <li style={styles.listItem}>
                Implement anomaly detection to flag potential hazards dynamically.
              </li>
              <li style={styles.listItem}>
                Detect and alert any camera tampering for continuous monitoring
                reliability.
              </li>
              <li style={styles.listItem}>
                Create a user-friendly dashboard with low-latency alert mechanisms.
              </li>
              <li style={styles.listItem}>
                Enable scalable deployment across multiple cameras and locations.
              </li>
            </ul>
          </section>

          <section>
            <h2 style={styles.sectionTitle}>System Architecture and Methodology</h2>
            <p style={styles.paragraph}>
              VigilNet processes live video streams through modular AI inference
              engines combining YOLOv8, ResNet50, and Multi-Column CNN models.
              Alerts propagate via a backend service layer, with users interacting
              through a web dashboard featuring live heatmaps, historical logs, and
              snapshot capture. The architecture prioritizes modularity, low
              latency, and ease of integration.
            </p>
          </section>

          <section>
            <h2 style={styles.sectionTitle}>Data and Models</h2>
            <p style={styles.paragraph}>
              The system leverages annotated crowd datasets for model training,
              achieving high accuracy in a variety of environments by combining
              detection and density estimation. Models are optimized for efficient
              real-time inference suitable for both edge and cloud deployments.
            </p>
          </section>

          <section>
            <h2 style={styles.sectionTitle}>Implementation Details</h2>
            <p style={styles.paragraph}>
              Using a modern web stack, VigilNet integrates AI inference with alert
              management. Features include multi-camera support, configurable
              anomaly thresholds, snapshot functionality, timeline navigation, and
              support for RTSP and cloud video sources to match real-world CCTV
              infrastructure.
            </p>
          </section>

          <section>
            <h2 style={styles.sectionTitle}>Feature Description</h2>
            <ul style={styles.list}>
              <li style={styles.listItem}>
                Real-time crowd density heatmaps and numerical counts.
              </li>
              <li style={styles.listItem}>
                Anomaly detection based on motion and behavioural analysis.
              </li>
              <li style={styles.listItem}>
                Camera tamper detection with instant alerting.
              </li>
              <li style={styles.listItem}>
                Latency targets under ~250 ms for fast responses.
              </li>
              <li style={styles.listItem}>
                Multi-platform integrations including SMS, webhooks, and REST APIs.
              </li>
            </ul>
          </section>

          <section>
            <h2 style={styles.sectionTitle}>Results and Performance Evaluation</h2>
            <p style={styles.paragraph}>
              VigilNet demonstrated high detection accuracy and responsive alerting
              within the required latency budget. Field-style tests on varied crowd
              scenarios validate the system&apos;s robustness, while benchmarking
              highlights scalability across different deployment configurations.
            </p>
          </section>

          <section>
            <h2 style={styles.sectionTitle}>Discussions and Challenges</h2>
            <p style={styles.paragraph}>
              Key challenges include handling occlusions in dense crowds,
              minimizing false positives in anomaly alerts, and ensuring privacy
              compliance. Robust asynchronous processing, careful threshold tuning,
              and defensive error handling were used to maintain system resilience.
            </p>
          </section>

          <section>
            <h2 style={styles.sectionTitle}>Conclusions and Future Work</h2>
            <p style={styles.paragraph}>
              VigilNet presents a comprehensive solution for crowd safety leveraging
              modern AI techniques. Future work includes predictive analytics,
              enhanced privacy-preserving pipelines, multi-camera synchronization,
              and further latency reduction to strengthen real-world applicability
              and user experience.
            </p>
          </section>

          <div style={styles.footer}>© 2025 VigilNet Project Team</div>
        </div>
      </div>
    </>
  );
}

export default Report;
