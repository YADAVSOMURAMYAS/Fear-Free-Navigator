import React from "react";

function GradeChip({ grade, score }) {
  const colors = {
    A: "#22c55e", B: "#84cc16",
    C: "#f59e0b", D: "#ef4444", E: "#7f1d1d",
  };
  const color = colors[grade] || "#94a3b8";
  return (
    <span style={{
      background: color + "22",
      color,
      border: `1px solid ${color}`,
      borderRadius: 20,
      padding: "2px 12px",
      fontWeight: 700,
      fontSize: 13,
    }}>
      {grade} · {score}
    </span>
  );
}

export default function SafetyBar({ safeRoute, fastRoute, comparison }) {
  if (!safeRoute || !fastRoute) return null;

  const gain   = comparison?.safety_gain_points || 0;
  const cost   = comparison?.time_penalty_min   || 0;
  const worthIt= comparison?.safer_route_worth_it;

  return (
    <div style={{
      background: "#1e293b",
      borderRadius: 12,
      padding: "16px 20px",
      marginBottom: 16,
    }}>
      <div style={{
        fontSize: 13,
        color: "#94a3b8",
        marginBottom: 12,
        fontWeight: 600,
        textTransform: "uppercase",
        letterSpacing: 1,
      }}>
        Route Comparison
      </div>

      {/* Safe route */}
      <div style={{
        background: "#22c55e11",
        border: "1px solid #22c55e44",
        borderRadius: 8,
        padding: "12px 16px",
        marginBottom: 8,
      }}>
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}>
          <span style={{ color: "#22c55e", fontWeight: 700, fontSize: 14 }}>
            Safe Route
          </span>
          <GradeChip
            grade={safeRoute.safety_grade}
            score={safeRoute.avg_safety_score.toFixed(1)}
          />
        </div>
        <div style={{ color: "#64748b", fontSize: 12, marginTop: 6 }}>
          {safeRoute.total_time_min.toFixed(1)} min ·{" "}
          {safeRoute.total_dist_km.toFixed(1)} km ·{" "}
          {safeRoute.dangerous_count} dangerous segments
        </div>
      </div>

      {/* Fast route */}
      <div style={{
        background: "#3b82f611",
        border: "1px solid #3b82f644",
        borderRadius: 8,
        padding: "12px 16px",
        marginBottom: 12,
      }}>
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}>
          <span style={{ color: "#3b82f6", fontWeight: 700, fontSize: 14 }}>
            Fast Route
          </span>
          <GradeChip
            grade={fastRoute.safety_grade}
            score={fastRoute.avg_safety_score.toFixed(1)}
          />
        </div>
        <div style={{ color: "#64748b", fontSize: 12, marginTop: 6 }}>
          {fastRoute.total_time_min.toFixed(1)} min ·{" "}
          {fastRoute.total_dist_km.toFixed(1)} km ·{" "}
          {fastRoute.dangerous_count} dangerous segments
        </div>
      </div>

      {/* Verdict */}
      <div style={{
        background: worthIt ? "#22c55e11" : "#f59e0b11",
        border: `1px solid ${worthIt ? "#22c55e44" : "#f59e0b44"}`,
        borderRadius: 8,
        padding: "10px 16px",
        fontSize: 13,
        color: worthIt ? "#22c55e" : "#f59e0b",
      }}>
        <strong>
          +{gain.toFixed(1)} safety pts
        </strong>
        {" "}for{" "}
        <strong>
          +{cost.toFixed(1)} min
        </strong>
        {" · "}
        {comparison?.recommendation}
      </div>
    </div>
  );
}