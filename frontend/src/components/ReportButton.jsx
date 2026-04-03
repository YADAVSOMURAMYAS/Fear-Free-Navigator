import React, { useState } from "react";
import axios from "axios";

export default function ReportButton({ lat, lon }) {
  const [open, setOpen]       = useState(false);
  const [desc, setDesc]       = useState("");
  const [category, setCategory] = useState("unsafe");
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!desc || desc.length < 5) return;
    setLoading(true);
    try {
      await axios.post("/report", {
        lat, lon,
        description: desc,
        category,
        hour: new Date().getHours(),
      });
      setSubmitted(true);
      setTimeout(() => {
        setOpen(false);
        setSubmitted(false);
        setDesc("");
      }, 2000);
    } catch (err) {
      console.error("Report failed:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        style={{
          background: "#ef4444",
          color: "#fff",
          border: "none",
          borderRadius: 8,
          padding: "10px 16px",
          cursor: "pointer",
          fontSize: 13,
          fontWeight: 600,
          width: "100%",
          marginTop: 8,
        }}
      >
        Report Unsafe Area
      </button>

      {open && (
        <div style={{
          position: "fixed",
          inset: 0,
          background: "rgba(0,0,0,0.7)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 9999,
        }}>
          <div style={{
            background: "#1e293b",
            borderRadius: 16,
            padding: 24,
            width: 320,
            boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
          }}>
            {submitted ? (
              <div style={{
                textAlign: "center",
                color: "#22c55e",
                fontSize: 16,
                padding: 20,
              }}>
                Report submitted. Thank you!
              </div>
            ) : (
              <>
                <div style={{
                  color: "#f1f5f9",
                  fontWeight: 700,
                  fontSize: 16,
                  marginBottom: 16,
                }}>
                  Report Unsafe Area
                </div>

                <select
                  value={category}
                  onChange={e => setCategory(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "8px 12px",
                    borderRadius: 8,
                    background: "#0f172a",
                    color: "#f1f5f9",
                    border: "1px solid #334155",
                    marginBottom: 12,
                    fontSize: 13,
                  }}
                >
                  <option value="unsafe">General Unsafe</option>
                  <option value="poor_lighting">Poor Lighting</option>
                  <option value="harassment">Harassment</option>
                  <option value="crime">Crime</option>
                  <option value="other">Other</option>
                </select>

                <textarea
                  value={desc}
                  onChange={e => setDesc(e.target.value)}
                  placeholder="Describe what makes this area unsafe..."
                  rows={3}
                  style={{
                    width: "100%",
                    padding: "8px 12px",
                    borderRadius: 8,
                    background: "#0f172a",
                    color: "#f1f5f9",
                    border: "1px solid #334155",
                    marginBottom: 12,
                    fontSize: 13,
                    resize: "none",
                    boxSizing: "border-box",
                  }}
                />

                <div style={{ display: "flex", gap: 8 }}>
                  <button
                    onClick={() => setOpen(false)}
                    style={{
                      flex: 1,
                      padding: "10px",
                      borderRadius: 8,
                      border: "1px solid #334155",
                      background: "transparent",
                      color: "#94a3b8",
                      cursor: "pointer",
                      fontSize: 13,
                    }}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleSubmit}
                    disabled={loading || desc.length < 5}
                    style={{
                      flex: 1,
                      padding: "10px",
                      borderRadius: 8,
                      border: "none",
                      background: desc.length < 5 ? "#334155" : "#ef4444",
                      color: "#fff",
                      cursor: desc.length < 5 ? "not-allowed" : "pointer",
                      fontSize: 13,
                      fontWeight: 600,
                    }}
                  >
                    {loading ? "Submitting..." : "Submit"}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}