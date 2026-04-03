import { useState, useCallback } from "react";
import axios from "axios";

export default function useHeatmap() {
  const [heatmapData, setHeatmapData] = useState([]);
  const [loading, setLoading]         = useState(false);

  const fetchHeatmap = useCallback(async (sampleN = 3000) => {
    setLoading(true);
    try {
      const { data } = await axios.get("/heatmap", {
        params: { sample_n: sampleN },
      });
      // Convert to [lat, lon, intensity] format for Leaflet.heat
      const points = data.points.map(([lat, lon, score]) => [
        lat, lon, score / 100,
      ]);
      setHeatmapData(points);
    } catch (err) {
      console.error("Heatmap fetch failed:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  return { heatmapData, loading, fetchHeatmap };
}