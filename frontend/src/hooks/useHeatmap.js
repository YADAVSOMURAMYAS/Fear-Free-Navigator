import { useState, useCallback } from "react";
import api from "../api";

export default function useHeatmap() {
  const [heatmapData, setHeatmapData] = useState([]);
  const [loading,     setLoading]     = useState(false);

  const fetchHeatmap = useCallback(async (
    sampleN = 3000,
    city    = "Bengaluru",
  ) => {
    setLoading(true);
    try {
      const { data } = await api.get("/heatmap", {
        params: { sample_n: sampleN, city },
      });
      const points = data.points.map(([lat, lon, score]) => [
        lat, lon, score / 100,
      ]);
      setHeatmapData(points);
    } catch (err) {
    } finally {
      setLoading(false);
    }
  }, []);

  return { heatmapData, loading, fetchHeatmap };
}