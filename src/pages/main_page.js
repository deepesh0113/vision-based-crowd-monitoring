import React, { useState, useEffect, useRef } from "react";
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

function MainPage({ user }) {
  const navigate = useNavigate();

  const [alertOpen, setAlertOpen] = useState(false);
  const [systemSettingsOpen, setSystemSettingsOpen] = useState(false);

  const alertRef = useRef(null);
  const systemSettingsRef = useRef(null);

  const [selectedSound, setSelectedSound] = useState("sound1");
  const [volume, setVolume] = useState(50);

  const [selectedSystemCamera, setSelectedSystemCamera] = useState("");
  const [uploadedVideoURL, setUploadedVideoURL] = useState(null);

  const width = useWindowWidth();
  const isMobile = width < 700;

  const audioRef = useRef(null);

  const styles = {
    page: {
      backgroundColor: "#020617", // serious dark
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

    // TOP WELCOME ROW (no role/account)
    welcomeRow: {
      display: "flex",
      flexDirection: isMobile ? "column" : "row",
      justifyContent: "space-between",
      alignItems: isMobile ? "flex-start" : "center",
      marginBottom: isMobile ? 12 : 20,
      width: "100%",
    },
    brandTitle: {
      fontSize: isMobile ? 14 : 16,
      textTransform: "uppercase",
      letterSpacing: "0.22em",
      color: "#9ca3af",
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
      backgroundColor: "#020617",
      borderRadius: 16,
      boxShadow: "0 16px 40px rgba(15,23,42,0.9)",
      overflow: "hidden",
      paddingBottom: 16,
      animation: "fadeInUp 0.7s",
      flex: isMobile ? "unset" : "1 1 50%",
      minWidth: isMobile ? "100%" : 360,
      position: "relative",
      border: "1px solid #1f2937",
      transition: "transform 0.2s ease, box-shadow 0.2s ease",
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
      height: isMobile ? 180 : 360,
      backgroundColor: "#020617",
      borderRadius: 14,
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      overflow: "hidden",
      position: "relative",
      marginTop: 8,
      marginBottom: 10,
    },
    cameraVideo: {
      width: "100%",
      height: "100%",
      objectFit: "contain",
      borderRadius: "14px",
      backgroundColor: "#000",
    },

    cameraSettings: {
      marginTop: 8,
      padding: isMobile ? "0 10px" : "0 18px",
      fontSize: 13,
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
    volumeDisplay: {
      fontWeight: 600,
      marginLeft: isMobile ? 6 : 10,
      color: "#93baff",
      fontSize: 13,
    },

    deleteCameraBtn: {
      position: "absolute",
      top: 20,
      right: 20,
      backgroundColor: "#b91c1c",
      border: "1px solid #fecaca",
      borderRadius: 999,
      color: "#fee2e2",
      cursor: "pointer",
      padding: "6px 12px",
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
      backgroundColor: "#1d4ed8",
      color: "#e5e7eb",
      borderRadius: 999,
      padding: isMobile ? "11px 14px" : "12px 20px",
      fontSize: isMobile ? 14 : 15,
      fontWeight: 600,
      cursor: "pointer",
      border: "none",
      boxShadow: "0 12px 30px rgba(37,99,235,0.35)",
      letterSpacing: "0.03em",
      minWidth: isMobile ? "100%" : 210,
    },

    addCameraBtn: {
      marginTop: 16,
      backgroundColor: "#111827",
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

    alertPanel: {
      position: isMobile ? "static" : "fixed",
      top: isMobile ? undefined : 96,
      right: isMobile ? undefined : 28,
      width: isMobile ? "100%" : 340,
      backgroundColor: "#020617",
      boxShadow: "0 16px 45px rgba(15,23,42,0.95)",
      borderRadius: 20,
      padding: isMobile ? 14 : 18,
      zIndex: 90,
      animation: "fadeInUp 0.5s",
      color: "#e5e7eb",
      border: "1px solid #1f2937",
      boxSizing: "border-box",
      marginTop: isMobile ? 16 : 0,
    },
    alertTitle: {
      fontWeight: 600,
      fontSize: isMobile ? 15 : 17,
      marginBottom: 10,
      color: "#93c5fd",
    },
    volumeInput: {
      width: "100%",
      height: 6,
      borderRadius: 4,
      background: "#020617",
      cursor: "pointer",
      marginTop: 4,
    },

    closeBtn: {
      marginTop: 10,
      width: "100%",
      backgroundColor: "#111827",
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
    },

    "@keyframes fadeInDown": {
      from: { opacity: 0, transform: "translateY(-24px)" },
      to: { opacity: 1, transform: "translateY(0)" },
    },
    "@keyframes fadeInUp": {
      from: { opacity: 0, transform: "translateY(24px)" },
      to: { opacity: 1, transform: "translateY(0)" },
    },
    "@keyframes fadeIn": {
      from: { opacity: 0 },
      to: { opacity: 1 },
    },
  };

  const [cameraPairs, setCameraPairs] = useState([
    {
      pairId: 0, // base cameras - locked from delete
      cameras: [
        {
          id: 0,
          name: "Camera 0 - Sample Video",
          src: "/people.mp4",
          on: true,
          alertSound: "sound1",
          volume: 50,
          uploadedFile: null,
        },
        {
          id: 1,
          name: "Camera 0b - Output Video",
          src: "/output_with_heatmap.gif",
          on: true,
          alertSound: "sound1",
          volume: 50,
          uploadedFile: null,
        },
      ],
    },
  ]);

  const [availableCameras] = useState([
    "Integrated Webcam",
    "USB Camera 1",
    "USB Camera 2",
    "Virtual Camera",
  ]);

  useEffect(() => {
    function handleClickOutside(event) {
      if (alertRef.current && !alertRef.current.contains(event.target) && alertOpen) {
        setAlertOpen(false);
      }
      if (
        systemSettingsRef.current &&
        !systemSettingsRef.current.contains(event.target) &&
        systemSettingsOpen
      ) {
        setSystemSettingsOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);

    // play / stop global alert preview
    if (alertOpen && audioRef.current) {
      audioRef.current.src = `/sounds/${selectedSound}.mp3`;
      audioRef.current.volume = volume / 100;
      audioRef.current.loop = true;
      audioRef.current.load();
      audioRef.current.play().catch((e) => {
        console.log("Alert sound play prevented:", e);
      });
    } else if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [alertOpen, systemSettingsOpen, selectedSound, volume]);

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

  const setCameraAlertSound = (pairId, cameraId, sound) => {
    setCameraPairs((pairs) =>
      pairs.map((pair) =>
        pair.pairId === pairId
          ? {
              ...pair,
              cameras: pair.cameras.map((cam) =>
                cam.id === cameraId ? { ...cam, alertSound: sound } : cam
              ),
            }
          : pair
      )
    );
    playSound(sound);
  };

  const changeVolume = (pairId, cameraId, vol) => {
    setCameraPairs((pairs) =>
      pairs.map((pair) =>
        pair.pairId === pairId
          ? {
              ...pair,
              cameras: pair.cameras.map((cam) =>
                cam.id === cameraId ? { ...cam, volume: vol } : cam
              ),
            }
          : pair
      )
    );
  };

  const uploadVideoFile = async (pairId, cameraId, file) => {
    const formData = new FormData();
    formData.append("video", file);

    // fixed double http:// bug
    const res = await fetch("http://0.0.0.0:8000/process_video/", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    setCameraPairs((pairs) =>
      pairs.map((pair) =>
        pair.pairId === pairId
          ? {
              ...pair,
              cameras: pair.cameras.map((cam) =>
                cam.id === cameraId
                  ? { ...cam, src: URL.createObjectURL(file), uploadedFile: file }
                  : cam.id === cameraId + 1
                  ? {
                      ...cam,
                      src: `http://0.0.0.0:8000/download_output?path=${encodeURIComponent(
                        data.output_video
                      )}`,
                    }
                  : cam
              ),
            }
          : pair
      )
    );

    fetch("http://0.0.0.0:8000/crowd_txt/")
      .then((res) => res.text())
      .then((txt) => {
        // you can store & show this txt somewhere if needed
      })
      .catch(() => {});
  };

  const playSound = (sound) => {
    if (audioRef.current) {
      audioRef.current.src = `/sounds/${sound}.mp3`;
      audioRef.current.load();
      audioRef.current.volume = 1;
      audioRef.current.play().catch((e) => {
        console.log("Audio playback failed:", e);
      });
      setTimeout(() => {
        if (audioRef.current) {
          audioRef.current.pause();
          audioRef.current.currentTime = 0;
        }
      }, 3000);
    }
  };

  const addCameraPair = () => {
    const newPairId = cameraPairs.length
      ? cameraPairs[cameraPairs.length - 1].pairId + 1
      : 1;
    const baseCamId = cameraPairs.reduce(
      (max, pair) => Math.max(max, ...pair.cameras.map((c) => c.id)),
      0
    );

    setCameraPairs((prev) => [
      ...prev,
      {
        pairId: newPairId,
        cameras: [
          {
            id: baseCamId + 1,
            name: `Camera ${newPairId} - Real`,
            src: "https://placeimg.com/640/480/nature",
            on: true,
            alertSound: false,
            volume: 50,
            uploadedFile: null,
          },
          {
            id: baseCamId + 2,
            name: `Camera ${newPairId}b - Model Video`,
            src: "https://placeimg.com/640/480/tech",
            on: true,
            alertSound: false,
            volume: 50,
            uploadedFile: null,
          },
        ],
      },
    ]);
  };

  const deleteCameraPair = (pairId) => {
    if (pairId === 0) return;
    setCameraPairs((pairs) => pairs.filter((pair) => pair.pairId !== pairId));
  };

  const addSystemCamera = () => {
    if (!selectedSystemCamera) return;
    const newPairId = cameraPairs.length
      ? cameraPairs[cameraPairs.length - 1].pairId + 1
      : 1;

    setCameraPairs((prev) => [
      ...prev,
      {
        pairId: newPairId,
        cameras: [
          {
            id: newPairId * 2 - 1,
            name: `${selectedSystemCamera} - Real`,
            src: "https://placeimg.com/640/480/tech",
            on: true,
          },
          {
            id: newPairId * 2,
            name: `${selectedSystemCamera} - Model Video`,
            src: "https://placeimg.com/640/480/arch",
            on: true,
          },
        ],
      },
    ]);
    setSelectedSystemCamera("");
    setSystemSettingsOpen(false);
  };

  const addUploadedVideo = () => {
    if (!uploadedVideoURL) return;
    const newPairId = cameraPairs.length
      ? cameraPairs[cameraPairs.length - 1].pairId + 1
      : 1;

    setCameraPairs((prev) => [
      ...prev,
      {
        pairId: newPairId,
        cameras: [
          {
            id: newPairId * 2 - 1,
            name: `Uploaded Video ${newPairId} - Real`,
            src: uploadedVideoURL,
            on: true,
          },
          {
            id: newPairId * 2,
            name: `Uploaded Video ${newPairId}b - Model Video`,
            src: uploadedVideoURL,
            on: true,
          },
        ],
      },
    ]);
    setUploadedVideoURL(null);
    setSystemSettingsOpen(false);
  };

  return (
    <div style={styles.page}>
      {/* Top: welcome, no role/account */}
      <div style={styles.welcomeRow}>
        <div>
          <div style={styles.brandTitle}>VigilNet</div>
          <div style={styles.welcomeText}>Welcome to VigilNet</div>
          <div style={styles.welcomeSubtext}>
            Intelligent CCTV and crowd monitoring console
          </div>
        </div>
      </div>

      {/* Camera pairs */}
      <div style={styles.cameraPairsContainer}>
        {cameraPairs.map((pair) => {
          const isCamera0 = pair.pairId === 0;

          return (
            <div key={pair.pairId} style={styles.cameraPair}>
              {pair.cameras.map((cam) => {
                const isModelVideo = cam.name.toLowerCase().includes("b");

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
                        disabled={isCamera0}
                      />
                    </div>

                    <div style={styles.cameraFrame}>
                      {cam.on ? (
                        <>
                          {cam.src.endsWith(".mp4") ||
                          cam.src.endsWith(".webm") ? (
                            <video
                              autoPlay
                              muted
                              loop
                              style={styles.cameraVideo}
                              src={cam.src}
                            />
                          ) : (
                            <img
                              style={styles.cameraVideo}
                              alt={`${cam.name} Feed`}
                              src={cam.src}
                            />
                          )}
                        </>
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

                    {isModelVideo && !isCamera0 && (
                      <>
                        <div style={styles.cameraSettings}>
                          <span
                            style={{ fontWeight: 600, marginRight: 6, fontSize: 13 }}
                          >
                            Alert Sound:
                          </span>
                          {["sound1", "sound2", "sound3"].map((sound) => (
                            <label
                              key={sound}
                              style={{ marginRight: 10, cursor: "pointer" }}
                            >
                              <input
                                type="radio"
                                name={`alertSound-${pair.pairId}-${cam.id}`}
                                value={sound}
                                checked={cam.alertSound === sound}
                                onChange={() =>
                                  setCameraAlertSound(pair.pairId, cam.id, sound)
                                }
                              />{" "}
                              {sound}
                            </label>
                          ))}
                        </div>

                        <div style={styles.cameraSettings}>
                          <span
                            style={{ fontWeight: 600, marginRight: 6, fontSize: 13 }}
                          >
                            Volume:
                          </span>
                          <input
                            type="range"
                            min="0"
                            max="100"
                            value={cam.volume}
                            onChange={(e) =>
                              changeVolume(pair.pairId, cam.id, +e.target.value)
                            }
                          />
                          <span style={styles.volumeDisplay}>
                            {cam.volume}%
                          </span>
                        </div>
                      </>
                    )}

                    {!isModelVideo && !isCamera0 && (
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
                        </label>
                      </div>
                    )}

                    {!isCamera0 && cam.id === pair.cameras[0].id && (
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
          );
        })}
      </div>

      {/* Single hidden audio element for sounds */}
      <audio ref={audioRef} />

      {/* Add camera pair */}
      <button style={styles.addCameraBtn} onClick={addCameraPair}>
        + Add More Camera Pair
      </button>

      {/* Action buttons */}
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

        <button
          style={styles.actionButton}
          onClick={() => setAlertOpen((v) => !v)}
        >
          View Alerts
        </button>
      </div>

      {/* Alert panel */}
      {alertOpen && (
        <div style={styles.alertPanel} ref={alertRef}>
          <div style={styles.alertTitle}>Alert Sound Settings</div>

          <div>
            <input
              type="radio"
              id="sound1"
              name="globalAlertSound"
              value="sound1"
              checked={selectedSound === "sound1"}
              onChange={() => setSelectedSound("sound1")}
            />
            <label htmlFor="sound1"> Sound 1</label>
          </div>

          <div>
            <input
              type="radio"
              id="sound2"
              name="globalAlertSound"
              value="sound2"
              checked={selectedSound === "sound2"}
              onChange={() => setSelectedSound("sound2")}
            />
            <label htmlFor="sound2"> Sound 2</label>
          </div>

          <div>
            <input
              type="radio"
              id="sound3"
              name="globalAlertSound"
              value="sound3"
              checked={selectedSound === "sound3"}
              onChange={() => setSelectedSound("sound3")}
            />
            <label htmlFor="sound3"> Sound 3</label>
          </div>

          <label htmlFor="volume" style={{ fontWeight: 600, fontSize: 13 }}>
            Volume: {volume}%
          </label>
          <input
            type="range"
            id="volume"
            min="0"
            max="100"
            value={volume}
            onChange={(e) => setVolume(+e.target.value)}
            style={styles.volumeInput}
          />

          <button
            onClick={() => setAlertOpen(false)}
            style={styles.closeBtn}
          >
            Close
          </button>
        </div>
      )}

      {/* System Settings panel */}
      {systemSettingsOpen && (
        <div style={styles.alertPanel} ref={systemSettingsRef}>
          <div style={styles.alertTitle}>System Settings</div>

          <label style={{ fontWeight: 600, fontSize: 13 }}>Select Camera:</label>
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

          <label style={{ fontWeight: 600, fontSize: 13 }}>Upload Video:</label>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                const fileURL = URL.createObjectURL(e.target.files[0]);
                setUploadedVideoURL(fileURL);
              }
            }}
            style={styles.uploadInput}
          />

          {uploadedVideoURL && (
            <>
              <video
                src={uploadedVideoURL}
                controls
                style={{
                  width: "100%",
                  borderRadius: 12,
                  marginBottom: 10,
                  marginTop: 4,
                }}
              />
              <button
                onClick={addUploadedVideo}
                style={{ ...styles.actionButton, width: "100%" }}
              >
                Add Uploaded Video as Camera Pair
              </button>
            </>
          )}

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
