import React from "react";

export default function AlphaSlider({ alpha, onChange }) {
  const label =
    alpha >= 0.8 ? "Max Safety"   :
    alpha >= 0.6 ? "Safety First" :
    alpha >= 0.4 ? "Balanced"     :
    alpha >= 0.2 ? "Speed First"  :
                   "Max Speed";

  const color =
    alpha >= 0.7 ? "#22c55e" :
    alpha >= 0.4 ? "#f59e0b" :
                   "#ef4444";

  return (
    <div style={{
      background: "#1e293b",
      borderRadius: 12,
      padding: "16px 20px",
      marginBottom: 16,
    }}>
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: 8,
      }}>
        <span style={{ color: "#94a3b8", fontSize: 13 }}>
          Safety preference
        </span>
        <span style={{
          color,
          fontWeight: 700,
          fontSize: 14,
          background: color + "22",
          padding: "2px 10px",
          borderRadius: 20,
        }}>
          {label}
        </span>
      </div>

      <input
        type="range"
        min="0" max="1" step="0.1"
        value={alpha}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: "100%", accentColor: color }}
      />

      <div style={{
        display: "flex",
        justifyContent: "space-between",
        fontSize: 11,
        color: "#475569",
        marginTop: 4,
      }}>
        <span>Max Speed</span>
        <span>Max Safety</span>
      </div>
    </div>
  );
}