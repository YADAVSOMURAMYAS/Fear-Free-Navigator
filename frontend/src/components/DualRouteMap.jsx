import React, { useEffect } from "react";
import {
  MapContainer, TileLayer, Polyline,
  Marker, Popup, useMap, useMapEvents,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require("leaflet/dist/images/marker-icon-2x.png"),
  iconUrl:       require("leaflet/dist/images/marker-icon.png"),
  shadowUrl:     require("leaflet/dist/images/marker-shadow.png"),
});

const BENGALURU_CENTER = [12.9716, 77.5946];

// ── Click handler component ────────────────────────────────────────────────────
function MapClickHandler({ onMapClick }) {
  useMapEvents({
    click: (e) => {
      if (onMapClick) onMapClick(e);
    },
  });
  return null;
}

// ── Fly to route bounds ────────────────────────────────────────────────────────
function FlyToBounds({ routes }) {
  const map = useMap();
  useEffect(() => {
    if (!routes) return;
    const coords = routes.safe_route?.coords || [];
    if (coords.length < 2) return;
    try {
      const bounds = L.latLngBounds(coords);
      map.flyToBounds(bounds, { padding: [40, 40], duration: 1 });
    } catch (e) {}
  }, [map, routes]);
  return null;
}

// ── Heatmap layer ──────────────────────────────────────────────────────────────
function HeatmapLayer({ data }) {
  const map = useMap();
  useEffect(() => {
    if (!data || data.length === 0) return;
    if (!L.heatLayer) return;
    const heat = L.heatLayer(data, {
      radius: 20, blur: 15, maxZoom: 17, max: 1.0,
      gradient: {
        0.0: "#22c55e", 0.4: "#84cc16",
        0.6: "#f59e0b", 0.8: "#ef4444", 1.0: "#7f1d1d",
      },
    }).addTo(map);
    return () => { try { map.removeLayer(heat); } catch(e){} };
  }, [map, data]);
  return null;
}

// ── Custom colored markers ─────────────────────────────────────────────────────
const greenIcon = new L.Icon({
  iconUrl: "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png",
  iconSize: [25, 41], iconAnchor: [12, 41],
  popupAnchor: [1, -34], shadowSize: [41, 41],
});

const blueIcon = new L.Icon({
  iconUrl: "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png",
  iconSize: [25, 41], iconAnchor: [12, 41],
  popupAnchor: [1, -34], shadowSize: [41, 41],
});

// ── Main map component ─────────────────────────────────────────────────────────
export default function DualRouteMap({
  routes, origin, destination,
  heatmapData, showHeatmap, onMapClick,
}) {
  return (
    <MapContainer
      center={BENGALURU_CENTER}
      zoom={13}
      style={{ height: "100%", width: "100%", borderRadius: 12 }}
    >
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        attribution='&copy; <a href="https://carto.com">CARTO</a>'
      />

      {/* Click handler */}
      <MapClickHandler onMapClick={onMapClick} />

      {/* Fly to bounds when route changes */}
      {routes && <FlyToBounds routes={routes} />}

      {/* Heatmap */}
      {showHeatmap && heatmapData.length > 0 && (
        <HeatmapLayer data={heatmapData} />
      )}

      {/* Safe route — solid green */}
      {routes?.safe_route?.coords && (
        <Polyline
          positions={routes.safe_route.coords}
          pathOptions={{ color: "#22c55e", weight: 6, opacity: 0.9 }}
        >
          <Popup>
            Safe Route — Score: {routes.safe_route.avg_safety_score.toFixed(1)}/100
            <br/>Time: {routes.safe_route.total_time_min.toFixed(1)} min
            <br/>Distance: {routes.safe_route.total_dist_km.toFixed(1)} km
          </Popup>
        </Polyline>
      )}

      {/* Fast route — blue dashed */}
      {routes?.fast_route?.coords && (
        <Polyline
          positions={routes.fast_route.coords}
          pathOptions={{
            color: "#3b82f6", weight: 4,
            opacity: 0.7, dashArray: "10 8",
          }}
        >
          <Popup>
            Fast Route — Score: {routes.fast_route.avg_safety_score.toFixed(1)}/100
            <br/>Time: {routes.fast_route.total_time_min.toFixed(1)} min
            <br/>Distance: {routes.fast_route.total_dist_km.toFixed(1)} km
          </Popup>
        </Polyline>
      )}

      {/* Origin marker */}
      {origin && (
        <Marker
          position={[origin.lat, origin.lon]}
          icon={greenIcon}
        >
          <Popup>
            Start<br/>
            {origin.lat.toFixed(4)}, {origin.lon.toFixed(4)}
          </Popup>
        </Marker>
      )}

      {/* Destination marker */}
      {destination && (
        <Marker
          position={[destination.lat, destination.lon]}
          icon={blueIcon}
        >
          <Popup>
            Destination<br/>
            {destination.lat.toFixed(4)}, {destination.lon.toFixed(4)}
          </Popup>
        </Marker>
      )}
    </MapContainer>
  );
}