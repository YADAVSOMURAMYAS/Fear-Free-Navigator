import React from "react";

export default function SegmentTooltip({ segment }) {
  if (!segment) return null;

  const { safety_score, safety_grade, highway, name, travel_time_s } = segment;

  const colors = {
    A: "#22c55e", B: "#84cc16",
    C: "#f59e0b", D: "#ef4444", E: "#7f1d1d",
  };
  const color = colors[safety_grade] || "#94a3b8";

  return (
    <div style={{
      background: "#0f172a",
      border: `1px solid ${color}`,
      borderRadius: 8,
      padding: "10px 14px",
      minWidth: 180,
      fontSize: 13,
    }}>
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: 6,
      }}>
        <span style={{ color: "#94a3b8" }}>Safety</span>
        <span style={{
          color,
          fontWeight: 700,
          fontSize: 16,
        }}>
          {safety_score?.toFixed(1)} ({safety_grade})
        </span>
      </div>

      {name && (
        <div style={{ color: "#e2e8f0", marginBottom: 4 }}>
          {name}
        </div>
      )}

      <div style={{ color: "#64748b", fontSize: 11 }}>
        {highway} · {travel_time_s?.toFixed(0)}s travel
      </div>
    </div>
  );
}