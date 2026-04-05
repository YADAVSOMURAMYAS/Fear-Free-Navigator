import { useState, useCallback, useRef } from "react";
import api from "../api";

export default function useRoute() {
  const [routes,  setRoutes]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);
  const abortRef = useRef(null);
  const reqIdRef = useRef(0);

  const fetchRoutes = useCallback(async ({
    originLat, originLon,
    destLat,   destLon,
    alpha      = 0.7,
    hour       = new Date().getHours(),
    city       = "Bengaluru",
    autoDetect = false,
    mode       = "car",
  }) => {
    if (abortRef.current) {
      abortRef.current.abort();
    }
    const controller = new AbortController();
    abortRef.current = controller;
    const reqId = ++reqIdRef.current;

    setLoading(true);
    setError(null);

    try {
      const { data } = await api.get("/route", {
        signal: controller.signal,
        params: {
          origin_lat:  originLat,
          origin_lon:  originLon,
          dest_lat:    destLat,
          dest_lon:    destLon,
          alpha,
          hour,
          city:        city || "Bengaluru",
          auto_detect: autoDetect || false,
          mode:        mode || "car",
        },
      });

      // Check for service unavailable
      if (data.error === "service_unavailable") {
        setError(data.message);
        return data;
      }

      if (reqId === reqIdRef.current) {
        setRoutes(data);
      }
      return data;

    } catch (err) {
      if (err?.code === "ERR_CANCELED" || err?.name === "CanceledError") {
        return null;
      }

      if (err?.response?.status === 409) {
        // Request was superseded by a newer city selection.
        return null;
      }

      const detail = err.response?.data?.detail || "";

      // Handle same point error gracefully
      if (err.response?.status === 400) {
        setError("Origin and destination are the same point. Please select different locations.");
        return null;
      }

      setError(
        detail || "Failed to fetch routes. Is the API running?"
      );
      return null;
    } finally {
      if (reqId === reqIdRef.current) {
        setLoading(false);
      }
    }
  }, []);

  const clearRoutes = useCallback(() => {
    setRoutes(null);
    setError(null);
  }, []);

  return { routes, loading, error, fetchRoutes, clearRoutes };
}