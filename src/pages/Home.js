// import React, { useState, useEffect } from 'react';
// import { Link } from 'react-router-dom';

// function Home() {
//   const [showIntro, setShowIntro] = useState(true);
//   const [density, setDensity] = useState('Normal');
//   const [lastUpdate, setLastUpdate] = useState(new Date().toLocaleTimeString());
//   const [activeNewsIdx, setActiveNewsIdx] = useState(null);
//   const [activeFeatureIdx, setActiveFeatureIdx] = useState(null);
//   const [activeFlashIdx, setActiveFlashIdx] = useState(null);

//   // Simulate density updates every 5 seconds
//   useEffect(() => {
//     if (!showIntro) {
//       const densities = ['Low', 'Normal', 'High'];
//       const interval = setInterval(() => {
//         const randomIndex = Math.floor(Math.random() * densities.length);
//         setDensity(densities[randomIndex]);
//         setLastUpdate(new Date().toLocaleTimeString());
//       }, 5000);
//       return () => clearInterval(interval);
//     }
//   }, [showIntro]);

//   const handleGetStarted = () => setShowIntro(false);
//   const getDensityColor = () => {
//     if (density === 'High') return 'red';
//     if (density === 'Normal') return 'orange';
//     return 'green';
//   };

//   // const handleViewTimeline = () => {
//   //   window.open("/crowd_log.txt", "Timeline", "width=600,height=400");
//   // };

//   // const handleSnapshot = () => {
//   //   window.open("/output_with_heatmap.gif", "Snapshot", "width=800,height=600");
//   // };

//   // Data arrays (news, features, flashcards)
//   const newsData = [
//     {
//       headline: "Tamil Nadu Rally Stampede — 40 Dead",
//       details:
//         "The Tamil Nadu rally tragedy in Karur on September 27, 2025, was one of India's worst political gathering disasters. Over 40 people died, including children and women, and more than 124 were injured when tens of thousands surged forward during actor-politician Vijay's delayed appearance. Chaotic conditions and collapsed barricades led to multiple crowd crushes. Rescue was hindered by blocked ambulances and jammed exits. The state launched a formal inquiry and announced compensation. The incident exposed major lapses in planning and spotlighted urgent reforms needed for mass event safety in India."
//     },
//     {
//       headline: "RCB Stadium Chaos — 11 Dead",
//       details:
//         "On June 3, 2025, celebrations at Bengaluru's M. Chinnaswamy Stadium turned deadly after a victory parade drew thousands of RCB fans. Security failed to control crowd surges at entry gates, resulting in 11 confirmed deaths and more than 50 injuries, many from asphyxiation and being trampled. Confusion over ticket checks and inadequate exit routes escalated risks. The Karnataka government ordered an inquiry and offered compensation. Experts now emphasize better stadium design, staff training, and real-time crowd monitoring to prevent future sports-related disasters."
//     },
//     {
//       headline: "Mahakumbh Tragedy — 30+ Dead",
//       details:
//         "At the Mahakumbh Mela in Prayagraj, January 2025, a crowd surge on a bathing day turned tragic, leaving more than 30 people dead according to official and media reports. Millions gathered on narrow passageways with insufficient barriers, causing stampedes at riverbanks. Most victims were suffocated or trampled in the rush. Police and local officials faced criticism for inadequate crowd management. Security has since been heightened and stricter protocols adopted for remaining festival days to protect worshippers."
//     },
//     {
//       headline: "Delhi Station Rush — 18 Dead",
//       details:
//         "On February 15, 2025, overcrowding at New Delhi Railway Station during a Mahakumbh rush resulted in a deadly stampede. Official tolls confirmed 18 deaths, with most victims being women and children, and dozens more injured. Passengers fell from footbridges and onto the tracks amid mass confusion. Rescue efforts restored order swiftly, but the event led to public outcry for improved crowd control and better early-warning procedures at major transport hubs. Railway authorities have committed to upgrading station safety protocols."
//     }
//   ];

//   const newsFlashcards = [
//     {
//       image: "/news1.webp",
//       title: "Tamil Nadu Rally",
//       summary: "Stampede at a political rally in Tamil Nadu leaves 40 dead, 120+ injured.",
//       details:
//         "Actor Vijay’s political event in Karur turned tragic when barricades collapsed after hours of delay, triggering a massive crowd surge. Emergency teams struggled to access the site. This incident calls for regulatory action on mass gathering planning and emergency readiness."
//     },
//     {
//       image: "/news2.jpg",
//       title: "RCB Stadium Chaos",
//       summary: "RCB’s victory celebration in Chinnaswamy Stadium turns tragic — 11 fans dead in stampede.",
//       details:
//         "Lack of exit routes and poor ticket-management led to panic post-IPL finals. Hundreds injured, raising questions on stadium design and crisis response protocols. State officials promised future reforms for fan safety."
//     },
//     {
//       image: "/news3.jpg",
//       title: "Mahakumbh Tragedy",
//       summary: "Stampede during Mahakumbh Mela leaves 30+ pilgrims dead and dozens injured.",
//       details:
//         "Crowd surges overwhelmed police barricades at the Prayagraj riverbank. Root causes were a lack of crowd-flow monitoring and early-warning signals. Intensive reforms in security practices now implemented for subsequent festival days."
//     },
//     {
//       image: "/news4.jpg",
//       title: "Delhi Station Rush",
//       summary: "New Delhi Railway Station stampede amid festival rush — 18 dead, several injured.",
//       details:
//         "Festive crowd mismanagement led to falls from bridges and onto the tracks. Railways have announced an overhaul in platform surveillance and crowd control measures to prevent future tragedies."
//     }
//   ];

//   const keyFeatures = [
//     {
//       title: "Crowd Density Mapping",
//       summary: "Heatmaps and counts per zone — visualize occupancy in real time.",
//       content:
//         "Real-time crowd counts and heatmap visualization allow rapid intervention in high-risk zones, preventing stampede risk. Advanced computer vision ensures accuracy, is scalable for large venues, and runs with low latency for instant alerting."
//     },
//     {
//       title: "Anomaly Detection",
//       summary: "Detect unusual motion patterns, fights, or loitering with adjustable sensitivity.",
//       content:
//         "AI-powered detection flags sudden motion, panic surges, fights, and loitering. Adjustable thresholds and rich analytics help organizers respond to emergencies before they escalate, saving lives and reducing property damage."
//     },
//     {
//       title: "Camera Tamper Detection",
//       summary: "Automatic detection of lens occlusion, defocus, or vandalism.",
//       content:
//         "Automated alerts for lens blockages, focus changes, or vandalism keep surveillance systems reliable throughout any event. Machine learning distinguishes between accidental and malicious tampering for adaptive responses."
//     }
//   ];

//   return (
//     <div>
//       {showIntro ? (
//         <div className="intro-screen">
//           <video autoPlay loop muted className="intro-video">
//             <source src="/home_video.mp4" type="video/mp4" />
//             Your browser does not support the video tag.
//           </video>
//           <div className="intro-content">
//             <h1>Vigilnet Crowd Management</h1>
//             <button className="btn" onClick={handleGetStarted}>Get Started</button>
//           </div>
//         </div>
//       ) : (
//         <div className="home-layout">
//           <div className="top-section">
//             {/* Main description and stats */}
//             <div className="left-side">
//               <div className="description-container">
//                 <div className="description-text">
//                   <h1>VigilNet — Intelligent crowd monitoring & instant alerts</h1>
//                   <p>
//                     Detect crowd density, abnormal motion, and camera tampering in real time.
//                     Send alerts to staff, automate workflows, and keep venues safe with low-latency models.
//                   </p>
//                 </div>
//                 <div className="stats-container">
//                   <div className="stat-card">
//                     <strong>99.8%</strong>
//                     <p>Detection accuracy</p>
//                   </div>
//                   <div className="stat-card">
//                     <strong>&lt;250ms</strong>
//                     <p>Alert latency</p>
//                   </div>
//                   <div className="stat-card">
//                     <strong>24/7</strong>
//                     <p>Uptime</p>
//                   </div>
//                 </div>
//               </div>
//             </div>

//             {/* Camera status and controls */}
//             <div className="right-side">
//               <div className="camera-section">
//                 <div className="camera-container">
//                   <video autoPlay loop muted playsInline className="camera-video">
//                     <source src="/people.mp4" type="video/mp4" />
//                     Your browser does not support the video tag.
//                   </video>
//                   <div className="camera-info">
//                     <div className="camera-status">
//                       <span style={{ backgroundColor: getDensityColor() }} className="status-indicator"></span>
//                       <span className="status-label">{density} Risk</span>
//                     </div>
//                     <div className="camera-details">
//                       <p><strong>Camera:</strong> Main Entrance</p>
//                       <p><strong>Status:</strong> Online</p>
//                       <p><strong>Last Updated:</strong> {lastUpdate}</p>
//                       {/* <div className="camera-controls">
//                         <button className="control-btn" onClick={handleSnapshot}>Snapshot</button>
//                         <button className="control-btn" onClick={handleViewTimeline}>View Timeline</button>
//                       </div> */}
//                     </div>
//                   </div>
//                 </div>
//               </div>
//             </div>
//           </div>

//           {/* News Flashcards Section */}
//           <div className="latest-news">
//             <h2>Latest News</h2>
//             <p>Stay informed with the latest crowd safety incidents and reports from across India.</p>
//             <div className="feature-cards">
//               {newsFlashcards.map((news, idx) => (
//                 <div
//                   className="feature-card"
//                   key={idx}
//                   style={{ cursor: "pointer", marginBottom: "1em" }}
//                   onClick={() =>
//                     setActiveFlashIdx(activeFlashIdx === idx ? null : idx)
//                   }
//                 >
//                   <img src={news.image} alt={news.title} />
//                   <h3 style={{ color: "blue" }}>
//                     {news.title}
//                   </h3>
//                   <p>{news.summary}</p>
//                   {activeFlashIdx === idx && (
//                     <div
//                       style={{
//                         marginTop: '0.5em',
//                         background: 'rgba(174, 191, 208, 0.3)',
//                         color: '#fff',
//                         padding: '12px',
//                         borderRadius: '5px'
//                       }}
//                     >
//                       <p>{news.details}</p>
//                     </div>
//                   )}
//                 </div>
//               ))}
//             </div>
//             {/* News Headlines with popout on click */}
//             <div className="news-headlines" style={{ marginTop: "2em" }}>
//               <h3>More Headlines</h3>
//               <ul>
//                 {newsData.map((news, idx) => (
//                   <li key={idx}>
//                     <strong
//                       style={{ color: "blue", cursor: "pointer" }}
//                       onClick={() =>
//                         setActiveNewsIdx(activeNewsIdx === idx ? null : idx)
//                       }
//                     >
//                       {news.headline}
//                     </strong>
//                     {activeNewsIdx === idx && (
//                       <div
//                         style={{
//                           background: '#000',
//                           color: '#fff',
//                           padding: '12px',
//                           borderRadius: '5px',
//                           marginTop: '0.5em'
//                         }}
//                       >
//                         <p>{news.details}</p>
//                       </div>
//                     )}
//                   </li>
//                 ))}
//               </ul>

//             </div>
//           </div>

//           {/* Key Features Section as interactive cards */}
//           <div className="key-features" style={{ marginTop: "2em" }}>
//             <h2>Key Features</h2>
//             <p>Everything you need for real-time crowd intelligence and fast incident response.</p>
//             <div className="feature-grid">
//               {keyFeatures.map((feature, idx) => (
//                 <div
//                   className="feature-item"
//                   key={idx}
//                   style={{
//                     marginBottom: "1em",
//                     cursor: "pointer"
//                   }}
//                   onClick={() =>
//                     setActiveFeatureIdx(activeFeatureIdx === idx ? null : idx)
//                   }
//                 >
//                   <span className="icon">
//                     {idx === 0 ? '📊' : idx === 1 ? '⚠️' : '🔒'}
//                   </span>
//                   <h3
//                     style={{
//                       color: "green",
//                       fontWeight: 600,
//                       marginBottom: "0.35em"
//                     }}
//                   >
//                     {feature.title}
//                   </h3>
//                   <p>{feature.summary}</p>
//                   {activeFeatureIdx === idx && (
//                     <div
//                       style={{
//                         marginTop: '0.5em',
//                         background: 'rgba(30, 144, 255, 0.3)',
//                         color: '#fff',
//                         padding: '12px',
//                         borderRadius: '5px'
//                       }}
//                     >
//                       <p>{feature.content}</p>
//                     </div>
//                   )}
//                 </div>
//               ))}
//             </div>

//             <div className="integrations">
//               <h3>Integrations</h3>
//               <p>
//                 Works with common NVRs, RTSP streams, and cloud video sources.
//                 Webhooks, SMS & email alerts, and REST API.
//               </p>
//               <div className="integration-tags">
//                 <span>RTSP</span>
//                 <span>Webhooks</span>
//                 <span>Slack</span>
//                 <span>SMS</span>
//                 <span>On-prem</span>
//               </div>
//             </div>
//           </div>
//         </div>
//       )}
//     </div>
//   );
// }

// export default Home;


import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";

function Home() {
  const [showIntro, setShowIntro] = useState(true);
  const [density, setDensity] = useState("Normal");
  const [lastUpdate, setLastUpdate] = useState(
    new Date().toLocaleTimeString()
  );
  const [activeNewsIdx, setActiveNewsIdx] = useState(null);
  const [activeFeatureIdx, setActiveFeatureIdx] = useState(null);
  const [activeFlashIdx, setActiveFlashIdx] = useState(null);

  useEffect(() => {
    if (!showIntro) {
      const densities = ["Low", "Normal", "High"];
      const interval = setInterval(() => {
        const randomIndex = Math.floor(Math.random() * densities.length);
        setDensity(densities[randomIndex]);
        setLastUpdate(new Date().toLocaleTimeString());
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [showIntro]);

  const handleGetStarted = () => setShowIntro(false);

  const getDensityColor = () => {
    if (density === "High") return "#f97373";
    if (density === "Normal") return "#facc15";
    return "#22c55e";
  };

  const newsData = [
    {
      headline: "Tamil Nadu Rally Stampede — 40 Dead",
      details:
        "The Tamil Nadu rally tragedy in Karur on September 27, 2025, was one of India's worst political gathering disasters. Over 40 people died, including children and women, and more than 124 were injured when tens of thousands surged forward during actor-politician Vijay's delayed appearance. Chaotic conditions and collapsed barricades led to multiple crowd crushes. Rescue was hindered by blocked ambulances and jammed exits. The state launched a formal inquiry and announced compensation. The incident exposed major lapses in planning and spotlighted urgent reforms needed for mass event safety in India.",
    },
    {
      headline: "RCB Stadium Chaos — 11 Dead",
      details:
        "On June 3, 2025, celebrations at Bengaluru's M. Chinnaswamy Stadium turned deadly after a victory parade drew thousands of RCB fans. Security failed to control crowd surges at entry gates, resulting in 11 confirmed deaths and more than 50 injuries, many from asphyxiation and being trampled. Confusion over ticket checks and inadequate exit routes escalated risks. The Karnataka government ordered an inquiry and offered compensation. Experts now emphasize better stadium design, staff training, and real-time crowd monitoring to prevent future sports-related disasters.",
    },
    {
      headline: "Mahakumbh Tragedy — 30+ Dead",
      details:
        "At the Mahakumbh Mela in Prayagraj, January 2025, a crowd surge on a bathing day turned tragic, leaving more than 30 people dead according to official and media reports. Millions gathered on narrow passageways with insufficient barriers, causing stampedes at riverbanks. Most victims were suffocated or trampled in the rush. Police and local officials faced criticism for inadequate crowd management. Security has since been heightened and stricter protocols adopted for remaining festival days to protect worshippers.",
    },
    {
      headline: "Delhi Station Rush — 18 Dead",
      details:
        "On February 15, 2025, overcrowding at New Delhi Railway Station during a Mahakumbh rush resulted in a deadly stampede. Official tolls confirmed 18 deaths, with most victims being women and children, and dozens more injured. Passengers fell from footbridges and onto the tracks amid mass confusion. Rescue efforts restored order swiftly, but the event led to public outcry for improved crowd control and better early-warning procedures at major transport hubs. Railway authorities have committed to upgrading station safety protocols.",
    },
  ];

  const newsFlashcards = [
    {
      image: "/news1.webp",
      title: "Tamil Nadu Rally",
      summary:
        "Stampede at a political rally in Tamil Nadu leaves 40 dead, 120+ injured.",
      details:
        "Actor Vijay’s political event in Karur turned tragic when barricades collapsed after hours of delay, triggering a massive crowd surge. Emergency teams struggled to access the site. This incident calls for regulatory action on mass gathering planning and emergency readiness.",
    },
    {
      image: "/news2.jpg",
      title: "RCB Stadium Chaos",
      summary:
        "RCB’s victory celebration in Chinnaswamy Stadium turns tragic — 11 fans dead in stampede.",
      details:
        "Lack of exit routes and poor ticket-management led to panic post-IPL finals. Hundreds injured, raising questions on stadium design and crisis response protocols. State officials promised future reforms for fan safety.",
    },
    {
      image: "/news3.jpg",
      title: "Mahakumbh Tragedy",
      summary:
        "Stampede during Mahakumbh Mela leaves 30+ pilgrims dead and dozens injured.",
      details:
        "Crowd surges overwhelmed police barricades at the Prayagraj riverbank. Root causes were a lack of crowd-flow monitoring and early-warning signals. Intensive reforms in security practices now implemented for subsequent festival days.",
    },
    {
      image: "/news4.jpg",
      title: "Delhi Station Rush",
      summary:
        "New Delhi Railway Station stampede amid festival rush — 18 dead, several injured.",
      details:
        "Festive crowd mismanagement led to falls from bridges and onto the tracks. Railways have announced an overhaul in platform surveillance and crowd control measures to prevent future tragedies.",
    },
  ];

  const keyFeatures = [
    {
      title: "Crowd Density Mapping",
      summary: "Heatmaps and counts per zone — visualize occupancy in real time.",
      content:
        "Real-time crowd counts and heatmap visualization allow rapid intervention in high-risk zones, preventing stampede risk. Advanced computer vision ensures accuracy, is scalable for large venues, and runs with low latency for instant alerting.",
    },
    {
      title: "Anomaly Detection",
      summary: "Detect unusual motion patterns, fights, or loitering with adjustable sensitivity.",
      content:
        "AI-powered detection flags sudden motion, panic surges, fights, and loitering. Adjustable thresholds and rich analytics help organizers respond to emergencies before they escalate, saving lives and reducing property damage.",
    },
    {
      title: "Camera Tamper Detection",
      summary: "Automatic detection of lens occlusion, defocus, or vandalism.",
      content:
        "Automated alerts for lens blockages, focus changes, or vandalism keep surveillance systems reliable throughout any event. Machine learning distinguishes between accidental and malicious tampering for adaptive responses.",
    },
  ];

  return (
    <div className="home-page">
      {showIntro ? (
        <div className="intro-screen">
          <video autoPlay loop muted className="intro-video">
            <source src="/home_video.mp4" type="video/mp4" />
            Your browser does not support the video tag.
          </video>
          <div className="intro-content">
            <h1 className="intro-title">VigilNet Crowd Management</h1>
            <button className="btn intro-btn" onClick={handleGetStarted}>
              Get Started
            </button>
          </div>
        </div>
      ) : (
        <div className="home-layout">
          {/* HERO ROW */}
          <section className="hero-row">
            <div className="hero-left">
              <div className="description-card">
                <div className="description-text">
                  <h1>VigilNet — Intelligent crowd monitoring & instant alerts</h1>
                  <p>
                    Detect crowd density, abnormal motion, and camera tampering
                    in real time. Send alerts to staff, automate workflows, and
                    keep venues safe with low-latency models.
                  </p>
                </div>

                <div className="stats-grid">
                  <div className="stat-card">
                    <strong>99.8%</strong>
                    <p>Detection accuracy</p>
                  </div>
                  <div className="stat-card">
                    <strong>&lt;250ms</strong>
                    <p>Alert latency</p>
                  </div>
                  <div className="stat-card">
                    <strong>24/7</strong>
                    <p>Uptime</p>
                  </div>
                </div>

                <div className="hero-links">
                  <Link to="/main_page" className="hero-link-btn">
                    Open VigilNet Console
                  </Link>
                </div>
              </div>
            </div>

            <div className="hero-right">
              <div className="camera-card">
                <video
                  autoPlay
                  loop
                  muted
                  playsInline
                  className="camera-video"
                >
                  <source src="/people.mp4" type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
                <div className="camera-meta">
                  <div className="camera-status-row">
                    <div className="density-chip">
                      <span
                        className="density-dot"
                        style={{ backgroundColor: getDensityColor() }}
                      />
                      <span className="density-label">{density} risk</span>
                    </div>
                    <span className="camera-online">● Online</span>
                  </div>
                  <div className="camera-details">
                    <p>
                      <span className="label">Camera:</span> Main Entrance
                    </p>
                    <p>
                      <span className="label">Last update:</span> {lastUpdate}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* LATEST NEWS SECTION */}
          <section className="section-block latest-news">
            <div className="section-header">
              <h2>Latest crowd safety incidents</h2>
              <p>
                Recent Indian crowd disasters that motivate VigilNet’s real-time
                monitoring and early-warning design.
              </p>
            </div>

            <div className="news-card-grid">
              {newsFlashcards.map((news, idx) => (
                <div
                  key={idx}
                  className="news-card"
                  onClick={() =>
                    setActiveFlashIdx(activeFlashIdx === idx ? null : idx)
                  }
                >
                  <img src={news.image} alt={news.title} />
                  <h3>{news.title}</h3>
                  <p className="news-summary">{news.summary}</p>
                  {activeFlashIdx === idx && (
                    <div className="news-details-card">
                      <p>{news.details}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="news-headlines">
              <h3>More headlines</h3>
              <ul>
                {newsData.map((news, idx) => (
                  <li key={idx}>
                    <button
                      type="button"
                      className="headline-btn"
                      onClick={() =>
                        setActiveNewsIdx(activeNewsIdx === idx ? null : idx)
                      }
                    >
                      {news.headline}
                    </button>
                    {activeNewsIdx === idx && (
                      <div className="headline-details">
                        <p>{news.details}</p>
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          </section>

          {/* KEY FEATURES SECTION */}
          <section className="section-block key-features">
            <div className="section-header">
              <h2>Key features</h2>
              <p>
                Modules designed for real-time crowd intelligence, operations,
                and research-led safety planning.
              </p>
            </div>

            <div className="feature-grid">
              {keyFeatures.map((feature, idx) => (
                <div
                  key={idx}
                  className="feature-card-iit"
                  onClick={() =>
                    setActiveFeatureIdx(
                      activeFeatureIdx === idx ? null : idx
                    )
                  }
                >
                  <span className="feature-icon">
                    {idx === 0 ? "📊" : idx === 1 ? "⚠️" : "🔒"}
                  </span>
                  <h3>{feature.title}</h3>
                  <p className="feature-summary">{feature.summary}</p>
                  {activeFeatureIdx === idx && (
                    <div className="feature-details">
                      <p>{feature.content}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="integrations">
              <h3>Integrations</h3>
              <p>
                Works with common NVRs, RTSP streams, and cloud video sources.
                Webhooks, SMS &amp; email alerts, and REST APIs for downstream
                automation.
              </p>
              <div className="integration-tags">
                <span>RTSP</span>
                <span>Webhooks</span>
                <span>Slack</span>
                <span>SMS</span>
                <span>On-prem</span>
              </div>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

export default Home;
