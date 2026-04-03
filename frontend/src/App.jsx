import React, { useState, useEffect, useCallback } from "react";
import DualRouteMap from "./components/DualRouteMap";
import SafetyBar    from "./components/SafetyBar";
import AlphaSlider  from "./components/AlphaSlider";
import ReportButton from "./components/ReportButton";
import useRoute     from "./hooks/useRoute";
import useHeatmap   from "./hooks/useHeatmap";

const LANDMARKS = [
  { name: "MG Road",        lat: 12.9767, lon: 77.6009 },
  { name: "Majestic",       lat: 12.9767, lon: 77.5713 },
  { name: "Koramangala",    lat: 12.9352, lon: 77.6245 },
  { name: "Indiranagar",    lat: 12.9718, lon: 77.6412 },
  { name: "Jayanagar",      lat: 12.9220, lon: 77.5833 },
  { name: "Shivajinagar",   lat: 12.9839, lon: 77.5929 },
  { name: "Whitefield",     lat: 12.9698, lon: 77.7499 },
  { name: "Electronic City",lat: 12.8458, lon: 77.6603 },
];

export default function App() {
  const [origin,      setOrigin]      = useState(null);
  const [destination, setDestination] = useState(null);
  const [alpha,       setAlpha]       = useState(0.7);
  const [hour,        setHour]        = useState(new Date().getHours());
  const [selectMode,  setSelectMode]  = useState("origin");
  const [showHeatmap, setShowHeatmap] = useState(true);

  const { routes, loading, error, fetchRoutes, clearRoutes } = useRoute();
  const { heatmapData, fetchHeatmap } = useHeatmap();

  // Load heatmap on mount
  useEffect(() => { fetchHeatmap(3000); }, [fetchHeatmap]);

  // Fetch route whenever origin, destination, alpha, or hour changes
  useEffect(() => {
    if (origin && destination) {
      fetchRoutes({
        originLat: origin.lat,
        originLon: origin.lon,
        destLat:   destination.lat,
        destLon:   destination.lon,
        alpha,
        hour,
      });
    }
  }, [origin, destination, alpha, hour]);

  const handleMapClick = useCallback((e) => {
    const { lat, lng } = e.latlng;
    if (selectMode === "origin") {
      setOrigin({ lat, lon: lng });
      setSelectMode("dest");
    } else {
      setDestination({ lat, lon: lng });
      setSelectMode("origin");
    }
  }, [selectMode]);

  // Set origin landmark
  const setOriginLandmark = (lm) => {
    setOrigin({ lat: lm.lat, lon: lm.lon });
    setSelectMode("dest");
  };

  // Set destination landmark
  const setDestLandmark = (lm) => {
    setDestination({ lat: lm.lat, lon: lm.lon });
    setSelectMode("origin");
  };

  const handleClear = () => {
    setOrigin(null);
    setDestination(null);
    clearRoutes();
    setSelectMode("origin");
  };

  return (
    <div style={{
      display: "flex", height: "100vh",
      background: "#0f172a", color: "#f1f5f9",
      fontFamily: "system-ui, sans-serif", fontSize: 14,
      overflow: "hidden",
    }}>

      {/* ── Sidebar ── */}
      <div style={{
        width: 340, minWidth: 340,
        background: "#0f172a",
        borderRight: "1px solid #1e293b",
        display: "flex", flexDirection: "column",
        overflow: "hidden",
      }}>

        {/* Header */}
        <div style={{ padding: "20px 20px 16px", borderBottom: "1px solid #1e293b" }}>
          <div style={{ fontSize: 20, fontWeight: 800, color: "#22c55e", marginBottom: 4 }}>
            Fear-Free Navigator
          </div>
          <div style={{ color: "#475569", fontSize: 12 }}>
            AI-powered safe routing · Bengaluru
          </div>
        </div>

        {/* Content */}
        <div style={{ flex: 1, overflow: "auto", padding: 16 }}>

          {/* Step indicator */}
          <div style={{
            background: "#1e293b", borderRadius: 8,
            padding: "10px 14px", marginBottom: 12,
            fontSize: 13, color: "#94a3b8",
            display: "flex", alignItems: "center", gap: 8,
          }}>
            <div style={{
              width: 10, height: 10, borderRadius: "50%", flexShrink: 0,
              background: selectMode === "origin" ? "#22c55e" : "#3b82f6",
            }} />
            {selectMode === "origin"
              ? "Step 1: Select start point below or click map"
              : "Step 2: Select destination below or click map"
            }
          </div>

          {/* Origin / Dest display */}
          {(origin || destination) && (
            <div style={{
              background: "#1e293b", borderRadius: 8,
              padding: "12px 14px", marginBottom: 12, fontSize: 12,
            }}>
              {origin && (
                <div style={{ marginBottom: 4 }}>
                  <span style={{ color: "#22c55e" }}>Start: </span>
                  <span style={{ color: "#94a3b8" }}>
                    {origin.lat.toFixed(4)}, {origin.lon.toFixed(4)}
                  </span>
                </div>
              )}
              {destination && (
                <div>
                  <span style={{ color: "#3b82f6" }}>End: </span>
                  <span style={{ color: "#94a3b8" }}>
                    {destination.lat.toFixed(4)}, {destination.lon.toFixed(4)}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Quick landmarks — TWO ROWS: origin + dest */}
          <div style={{ marginBottom: 16 }}>
            <div style={{
              fontSize: 11, color: "#475569",
              marginBottom: 6, textTransform: "uppercase", letterSpacing: 1,
            }}>
              Start point
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
              {LANDMARKS.map(lm => (
                <button key={"o_"+lm.name} onClick={() => setOriginLandmark(lm)}
                  style={{
                    background: origin?.lat === lm.lat ? "#22c55e22" : "#1e293b",
                    border: `1px solid ${origin?.lat === lm.lat ? "#22c55e" : "#334155"}`,
                    borderRadius: 6, color: origin?.lat === lm.lat ? "#22c55e" : "#94a3b8",
                    fontSize: 11, padding: "4px 10px", cursor: "pointer",
                  }}>
                  {lm.name}
                </button>
              ))}
            </div>

            <div style={{
              fontSize: 11, color: "#475569",
              marginBottom: 6, textTransform: "uppercase", letterSpacing: 1,
            }}>
              Destination
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {LANDMARKS.map(lm => (
                <button key={"d_"+lm.name} onClick={() => setDestLandmark(lm)}
                  style={{
                    background: destination?.lat === lm.lat ? "#3b82f622" : "#1e293b",
                    border: `1px solid ${destination?.lat === lm.lat ? "#3b82f6" : "#334155"}`,
                    borderRadius: 6, color: destination?.lat === lm.lat ? "#3b82f6" : "#94a3b8",
                    fontSize: 11, padding: "4px 10px", cursor: "pointer",
                  }}>
                  {lm.name}
                </button>
              ))}
            </div>
          </div>

          {/* Alpha slider */}
          <AlphaSlider alpha={alpha} onChange={setAlpha} />

          {/* Hour selector */}
          <div style={{
            background: "#1e293b", borderRadius: 12,
            padding: "12px 16px", marginBottom: 16,
          }}>
            <div style={{
              display: "flex", justifyContent: "space-between",
              alignItems: "center", marginBottom: 8,
            }}>
              <span style={{ color: "#94a3b8", fontSize: 13 }}>Time of day</span>
              <span style={{ color: "#f1f5f9", fontWeight: 700, fontSize: 14 }}>
                {String(hour).padStart(2, "0")}:00
                {hour >= 20 || hour <= 5 ? " 🌙" : " ☀️"}
              </span>
            </div>
            <input type="range" min="0" max="23" step="1" value={hour}
              onChange={e => setHour(parseInt(e.target.value))}
              style={{ width: "100%", accentColor: "#6366f1" }}
            />
          </div>

          {/* Loading */}
          {loading && (
            <div style={{ textAlign: "center", color: "#6366f1", padding: 16, fontSize: 13 }}>
              Computing safest route...
            </div>
          )}

          {/* Error */}
          {error && (
            <div style={{
              background: "#ef444411", border: "1px solid #ef4444",
              borderRadius: 8, padding: "10px 14px",
              color: "#ef4444", fontSize: 13, marginBottom: 12,
            }}>
              {error}
            </div>
          )}

          {/* Route comparison */}
          {routes && !loading && (
            <SafetyBar
              safeRoute  = {routes.safe_route}
              fastRoute  = {routes.fast_route}
              comparison = {routes.comparison}
            />
          )}

          {/* Report button */}
          {origin && <ReportButton lat={origin.lat} lon={origin.lon} />}

          {/* Controls */}
          <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
            <button onClick={() => setShowHeatmap(h => !h)} style={{
              flex: 1, padding: "10px", borderRadius: 8,
              border: `1px solid ${showHeatmap ? "#6366f1" : "#334155"}`,
              background: showHeatmap ? "#6366f122" : "transparent",
              color: showHeatmap ? "#6366f1" : "#94a3b8",
              cursor: "pointer", fontSize: 12,
            }}>
              {showHeatmap ? "Hide" : "Show"} Heatmap
            </button>
            <button onClick={handleClear} style={{
              flex: 1, padding: "10px", borderRadius: 8,
              border: "1px solid #334155", background: "transparent",
              color: "#94a3b8", cursor: "pointer", fontSize: 12,
            }}>
              Clear
            </button>
          </div>
        </div>

        {/* Footer */}
        <div style={{
          padding: "12px 16px", borderTop: "1px solid #1e293b",
          fontSize: 11, color: "#334155", textAlign: "center",
        }}>
          Green = Safe Route · Blue dashed = Fast Route
        </div>
      </div>

      {/* ── Map ── */}
      <div style={{ flex: 1, position: "relative" }}>
        <DualRouteMap
          routes      = {routes}
          origin      = {origin}
          destination = {destination}
          heatmapData = {heatmapData}
          showHeatmap = {showHeatmap}
          onMapClick  = {handleMapClick}
        />
      </div>
    </div>
  );
}