import { useState, useCallback } from "react";
import axios from "axios";

export default function useRoute() {
  const [routes, setRoutes]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

  const fetchRoutes = useCallback(async ({
    originLat, originLon,
    destLat,   destLon,
    alpha = 0.7,
    hour  = new Date().getHours(),
  }) => {
    setLoading(true);
    setError(null);

    try {
      const { data } = await axios.get("/route", {
        params: {
          origin_lat: originLat,
          origin_lon: originLon,
          dest_lat:   destLat,
          dest_lon:   destLon,
          alpha,
          hour,
        },
      });
      setRoutes(data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        "Failed to fetch routes. Is the API running?"
      );
    } finally {
      setLoading(false);
    }
  }, []);

  const clearRoutes = useCallback(() => {
    setRoutes(null);
    setError(null);
  }, []);

  return { routes, loading, error, fetchRoutes, clearRoutes };
}