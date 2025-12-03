import React, { useState, useEffect } from "react";

function useWindowWidth() {
  const [width, setWidth] = useState(window.innerWidth);

  useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return width;
}

function FAQ() {
  const width = useWindowWidth();
  const isMobile = width < 700;

  const styles = {
    page: {
      backgroundColor: "#020617",
      color: "#e5e7eb",
      minHeight: "100vh",
      padding: isMobile ? "20px 4vw 32px" : "28px 32px 40px",
      fontFamily:
        "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      display: "flex",
      justifyContent: "center",
      boxSizing: "border-box",
    },
    contentOuter: {
      width: "100%",
      maxWidth: 1040,
      display: "flex",
      flexDirection: "column",
      gap: isMobile ? 18 : 20,
      animation: "fadeIn 0.6s ease-in",
    },
    topRow: {
      display: "flex",
      flexDirection: isMobile ? "column" : "row",
      gap: isMobile ? 18 : 24,
      alignItems: isMobile ? "flex-start" : "flex-start",
      justifyContent: "space-between",
    },
    // Left text block
    textBlock: {
      flex: 1,
      minWidth: isMobile ? "0" : "260px",
    },
    brand: {
      fontSize: 12,
      textTransform: "uppercase",
      letterSpacing: "0.2em",
      color: "#6b7280",
      marginBottom: 4,
    },
    heading: {
      fontSize: isMobile ? "1.4rem" : "1.9rem",
      fontWeight: 600,
      color: "#e5e7eb",
      marginBottom: 4,
      letterSpacing: "0.03em",
    },
    subtext: {
      fontSize: isMobile ? 13 : 14,
      color: "#9ca3af",
      maxWidth: 520,
      lineHeight: 1.7,
    },

    // Right image (optional visual)
    imageBox: {
      flex: isMobile ? "unset" : "0 0 260px",
      display: "flex",
      alignItems: "center",
      justifyContent: isMobile ? "flex-start" : "flex-end",
      width: "100%",
      marginTop: isMobile ? 8 : 0,
    },
    faqImage: {
      width: isMobile ? "55%" : "80%",
      maxWidth: 260,
      borderRadius: 18,
      boxShadow: "0 18px 40px rgba(15,23,42,0.95)",
      opacity: 0.92,
      objectFit: "cover",
    },

    // FAQ list
    faqListWrapper: {
      marginTop: isMobile ? 10 : 16,
    },
    faqList: {
      width: "100%",
      maxWidth: 820,
      textAlign: "left",
      display: "flex",
      flexDirection: "column",
      gap: isMobile ? 10 : 12,
    },
    faqItem: {
      backgroundColor: "#020617",
      border: "1px solid #1f2937",
      borderRadius: 14,
      padding: isMobile ? "12px 12px" : "13px 16px",
      boxShadow: "0 14px 32px rgba(15,23,42,0.9)",
      transition: "transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s",
      cursor: "default",
    },
    questionRow: {
      display: "flex",
      alignItems: "flex-start",
      justifyContent: "space-between",
      gap: 10,
    },
    question: {
      fontSize: isMobile ? 14 : 15,
      fontWeight: 600,
      color: "#bfdbfe",
    },
    answer: {
      marginTop: 6,
      fontSize: isMobile ? 13 : 14,
      color: "#d1d5db",
      lineHeight: 1.7,
    },
  };

  const handleMouseEnter = (e) => {
    e.currentTarget.style.transform = "translateY(-2px)";
    e.currentTarget.style.boxShadow = "0 18px 40px rgba(15,23,42,0.95)";
    e.currentTarget.style.borderColor = "#1d4ed8";
  };

  const handleMouseLeave = (e) => {
    e.currentTarget.style.transform = "translateY(0)";
    e.currentTarget.style.boxShadow = "0 14px 32px rgba(15,23,42,0.9)";
    e.currentTarget.style.borderColor = "#1f2937";
  };

  const faqs = [
    {
      q: "1. What is VigilNet Crowd Management?",
      a: "VigilNet is a real-time crowd monitoring and alerting system that uses AI-powered analytics to detect anomalies and provide actionable insights.",
    },
    {
      q: "2. How do I integrate my cameras or NVR systems?",
      a: "You can integrate RTSP streams, NVRs, or cloud-based video sources directly through the VigilNet dashboard with minimal configuration.",
    },
    {
      q: "3. Does VigilNet support multi-camera and multi-location monitoring?",
      a: "Yes, VigilNet can handle multiple cameras across different sites simultaneously, making it suitable for large-scale events or smart city deployments.",
    },
    {
      q: "4. What research areas does this project contribute to?",
      a: "VigilNet contributes to computer vision for crowd analysis, anomaly detection, deep learning architectures, and edge computing for real-time video analytics.",
    },
    {
      q: "5. Which AI models are commonly used in VigilNet for crowd analysis?",
      a: "Techniques like YOLO for object detection and CSRNet-style networks for crowd density estimation are used to achieve high accuracy in real-time monitoring.",
    },
    {
      q: "6. How does VigilNet handle privacy and compliance?",
      a: "The system can anonymize faces and supports deployments that process only metadata or anonymized frames, helping align with privacy regulations when required.",
    },
    {
      q: "7. Can VigilNet be extended for academic or research purposes?",
      a: "Yes, VigilNet exposes REST APIs and uses a modular backend, making it easy to plug in experimental models or connect to custom data pipelines.",
    },
    {
      q: "8. What technologies power VigilNet’s backend?",
      a: "The backend typically combines Python-based AI services with a web server layer and WebSockets for real-time alerts, and can be deployed on scalable cloud infrastructure.",
    },
    {
      q: "9. How can VigilNet assist in emergency planning or event management research?",
      a: "By providing real-time density heatmaps and anomaly alerts, VigilNet helps researchers evaluate evacuation strategies, event safety, and urban flow planning.",
    },
    {
      q: "10. Does VigilNet work in low-light or adverse weather conditions?",
      a: "Pre-processing techniques such as low-light enhancement and noise reduction help maintain detection quality even in challenging camera conditions.",
    },
  ];

  return (
    <>
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      <div style={styles.page}>
        <div style={styles.contentOuter}>
          {/* Top row: text + image */}
          <div style={styles.topRow}>
            <div style={styles.textBlock}>
              <div style={styles.brand}>VigilNet</div>
              <h1 style={styles.heading}>Frequently Asked Questions</h1>
              <p style={styles.subtext}>
                Answers to both operational and research-focused questions
                around the VigilNet Crowd Management system – deployment,
                models, and use-cases.
              </p>
            </div>

            <div style={styles.imageBox}>
              <img
                src="/faq.jpg"
                alt="FAQ illustration"
                style={styles.faqImage}
              />
            </div>
          </div>

          {/* FAQ list */}
          <div style={styles.faqListWrapper}>
            <div style={styles.faqList}>
              {faqs.map((faq, index) => (
                <div
                  key={index}
                  style={styles.faqItem}
                  onMouseEnter={handleMouseEnter}
                  onMouseLeave={handleMouseLeave}
                >
                  <div style={styles.questionRow}>
                    <div style={styles.question}>{faq.q}</div>
                  </div>
                  <div style={styles.answer}>{faq.a}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default FAQ;
