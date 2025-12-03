import React, { useState, useEffect } from "react";

// Responsive: custom hook for screen size
function useWindowWidth() {
  const [width, setWidth] = useState(window.innerWidth);
  useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);
  return width;
}

function Contact() {
  // Array of image paths
  const images = ["/contact_us_bg.jpg", "/contact1.png", "/contact2.png"];
  const [imgIdx, setImgIdx] = useState(0);

  // Responsive setup
  const width = useWindowWidth();
  const isMobile = width < 700;

  // Cycle image every 3 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setImgIdx((prev) => (prev + 1) % images.length);
    }, 3000);
    return () => clearInterval(interval);
  }, [images.length]);

  const styles = {
    page: {
      backgroundColor: "#020617",
      color: "#e5e7eb",
      minHeight: "100vh",
      padding: isMobile ? "18px 4vw 28px" : "24px 32px 40px",
      fontFamily:
        "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      boxSizing: "border-box",
    },
    container: {
      width: "100%",
      maxWidth: isMobile ? "100%" : 880,
      backgroundColor: "#020617",
      borderRadius: 18,
      boxShadow: "0 20px 50px rgba(15,23,42,0.95)",
      border: "1px solid #1f2937",
      display: "flex",
      flexDirection: isMobile ? "column" : "row",
      overflow: "hidden",
    },

    // Left side image
    left: {
      flex: isMobile ? "unset" : 1,
      minWidth: 0,
      backgroundColor: "#020617",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: isMobile ? "14px 10px 0" : "16px 0 16px 16px",
    },
    image: {
      width: isMobile ? "82%" : "100%",
      maxWidth: 340,
      height: isMobile ? "auto" : 260,
      objectFit: "cover",
      borderRadius: isMobile ? 16 : 16,
      boxShadow: "0 18px 40px rgba(15,23,42,0.9)",
      transition: "opacity 0.6s ease",
      opacity: 0.96,
    },

    // Right side content
    right: {
      flex: isMobile ? "unset" : 1.15,
      padding: isMobile ? "18px 6vw 22px" : "24px 26px 26px",
      display: "flex",
      flexDirection: "column",
      justifyContent: "center",
      alignItems: isMobile ? "center" : "flex-start",
      borderLeft: isMobile ? "none" : "1px solid #1f2937",
    },
    brand: {
      fontSize: 11,
      textTransform: "uppercase",
      letterSpacing: "0.2em",
      color: "#6b7280",
      marginBottom: 4,
      textAlign: isMobile ? "center" : "left",
    },
    heading: {
      fontSize: isMobile ? "1.5rem" : "2rem",
      fontWeight: 600,
      color: "#e5e7eb",
      marginBottom: 6,
      letterSpacing: "0.03em",
      textAlign: isMobile ? "center" : "left",
    },
    subtext: {
      fontSize: isMobile ? 13 : 14,
      color: "#9ca3af",
      lineHeight: 1.7,
      maxWidth: 460,
      textAlign: isMobile ? "center" : "left",
    },

    button: {
      display: "inline-block",
      padding: isMobile ? "11px 20px" : "11px 24px",
      fontSize: isMobile ? 14 : 14,
      fontWeight: 600,
      color: "#f9fafb",
      backgroundColor: "#1d4ed8",
      border: "none",
      borderRadius: 999,
      textDecoration: "none",
      boxShadow: "0 14px 32px rgba(37,99,235,0.4)",
      cursor: "pointer",
      margin: isMobile ? "18px 0 14px" : "18px 0 14px",
      letterSpacing: "0.06em",
      textAlign: "center",
      whiteSpace: "nowrap",
    },

    infoBlock: {
      width: "100%",
      marginTop: 4,
    },
    smallLabel: {
      fontSize: isMobile ? 11 : 12,
      color: "#9ca3af",
      margin: "10px 0 2px",
      fontWeight: 500,
      letterSpacing: "0.14em",
      textTransform: "uppercase",
      textAlign: isMobile ? "center" : "left",
    },
    infoRow: {
      fontSize: isMobile ? 13 : 14,
      margin: "4px 0 0 0",
      textAlign: isMobile ? "center" : "left",
      wordBreak: "break-word",
    },
    link: {
      color: "#60a5fa",
      textDecoration: "none",
      fontWeight: 500,
    },
    caption: {
      marginTop: isMobile ? 18 : 20,
      color: "#9ca3af",
      fontSize: isMobile ? 12 : 13,
      lineHeight: 1.7,
      textAlign: isMobile ? "center" : "left",
    },
    footer: {
      textAlign: isMobile ? "center" : "left",
      color: "#6b7280",
      fontSize: isMobile ? 11 : 12,
      marginTop: 14,
      letterSpacing: "0.14em",
      textTransform: "uppercase",
    },
  };

  return (
    <>
      <style>{`
        @keyframes fadeInContact {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      <div style={styles.page}>
        <div style={styles.container}>
          {/* Left: Rotating Image */}
          <div style={styles.left}>
            <img
              src={images[imgIdx]}
              alt="Contact"
              style={styles.image}
              loading="lazy"
            />
          </div>

          {/* Right: Content */}
          <div style={styles.right}>
            <div style={styles.brand}>VigilNet</div>
            <h1 style={styles.heading}>Contact the Project Team</h1>
            <p style={styles.subtext}>
              Reach out for technical questions, research discussions, or
              collaborations around crowd monitoring and intelligent CCTV
              analytics.
            </p>

            {/* Send Email to all three addresses */}
            <a
              href="mailto:deepesh.kumar.ug22@nsut.ac.in,prateek.dhanker.ug22@nsut.ac.in,gaurav.kumar.ug22@nsut.ac.in"
              style={styles.button}
              title="Send your query to all team members"
            >
              📧 Send Email to Team
            </a>

            <div style={styles.infoBlock}>
              {/* Emails */}
              <p style={styles.smallLabel}>Email</p>
              <div style={styles.infoRow}>
                <a
                  href="mailto:deepesh.kumar.ug22@nsut.ac.in"
                  style={styles.link}
                >
                  deepesh.kumar.ug22@nsut.ac.in
                </a>
              </div>
              <div style={styles.infoRow}>
                <a
                  href="mailto:prateek.dhanker.ug22@nsut.ac.in"
                  style={styles.link}
                >
                  prateek.dhanker.ug22@nsut.ac.in
                </a>
              </div>
              <div style={styles.infoRow}>
                <a
                  href="mailto:gaurav.kumar.ug22@nsut.ac.in"
                  style={styles.link}
                >
                  gaurav.kumar.ug22@nsut.ac.in
                </a>
              </div>

              {/* Phone Numbers */}
              <p style={styles.smallLabel}>Phone</p>
              <div style={styles.infoRow}>
                <a href="tel:+917982460774" style={styles.link}>
                  +91-79824 60774
                </a>
              </div>
              <div style={styles.infoRow}>
                <a href="tel:+917988898595" style={styles.link}>
                  +91-79888 98595
                </a>
              </div>
              <div style={styles.infoRow}>
                <a href="tel:+919599871719" style={styles.link}>
                  +91-95998 71719
                </a>
              </div>
            </div>

            <div style={styles.caption}>
              We usually respond quickly for academic discussions, BTP queries,
              and deployment-related support. Feel free to share logs, screenshots,
              or datasets when relevant.
            </div>
            <div style={styles.footer}>© 2025 VigilNet Project Team</div>
          </div>
        </div>
      </div>
    </>
  );
}

export default Contact;
