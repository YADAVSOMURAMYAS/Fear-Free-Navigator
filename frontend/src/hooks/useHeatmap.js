import { useState, useCallback, useRef } from "react";
import api from "../api";

export default function useHeatmap() {
  const [heatmapData, setHeatmapData] = useState([]);
  const [loading,     setLoading]     = useState(false);
  const abortRef = useRef(null);
  const reqIdRef = useRef(0);

  const fetchHeatmap = useCallback(async (
    sampleN = 3000,
    city    = "Bengaluru",
  ) => {
    if (abortRef.current) {
      abortRef.current.abort();
    }
    const controller = new AbortController();
    abortRef.current = controller;
    const reqId = ++reqIdRef.current;

    setLoading(true);
    try {
      const { data } = await api.get("/heatmap", {
        signal: controller.signal,
        params: { sample_n: sampleN, city },
      });
      const points = data.points.map(([lat, lon, score]) => [
        lat, lon, score / 100,
      ]);
      if (reqId === reqIdRef.current) {
        setHeatmapData(points);
      }
    } catch (err) {
      if (err?.code === "ERR_CANCELED" || err?.name === "CanceledError") {
        return;
      }
      if (err?.response?.status === 409) {
        return;
      }
      setHeatmapData([]);
    } finally {
      if (reqId === reqIdRef.current) {
        setLoading(false);
      }
    }
  }, []);

  return { heatmapData, loading, fetchHeatmap };
}