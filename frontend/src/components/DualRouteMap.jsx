import React, { useEffect } from "react";
import {
  MapContainer, TileLayer, Polyline,
  Marker, Popup, useMap, useMapEvents, Circle,
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

// ── Click handler ──────────────────────────────────────────────────────────────
function MapClickHandler({ onMapClick }) {
  useMapEvents({ click: (e) => { if (onMapClick) onMapClick(e); } });
  return null;
}

// ── Fly to city center when city changes ───────────────────────────────────────
function FlyToCity({ center }) {
  const map = useMap();
  useEffect(() => {
    if (!center) return;
    map.flyTo(center, 13, { duration: 1.5 });
  }, [map, center]);
  return null;
}

// ── Fly to route bounds when route computed ────────────────────────────────────
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

// ── Heatmap ────────────────────────────────────────────────────────────────────
function HeatmapLayer({ data }) {
  const map = useMap();
  useEffect(() => {
    if (!data || data.length === 0) return;
    if (!L.heatLayer) return;
    const heat = L.heatLayer(data, {
      radius: 20, blur: 15, maxZoom: 17, max: 1.0,
      gradient: {
        0.0: "#22c55e", 0.4: "#84cc16",
        0.6: "#f59e0b", 0.8: "#ef4444",
        1.0: "#7f1d1d",
      },
    }).addTo(map);
    return () => { try { map.removeLayer(heat); } catch(e){} };
  }, [map, data]);
  return null;
}

// ── Live location control ──────────────────────────────────────────────────────
function LocateControl({ onLocate }) {
  const map = useMap();
  const [locating, setLocating] = React.useState(false);

  const handleClick = (e) => {
    e.stopPropagation();
    setLocating(true);
    map.locate({ setView: true, maxZoom: 15 });
    map.once("locationfound", (e) => {
      setLocating(false);
      onLocate({ lat: e.latlng.lat, lon: e.latlng.lng });
    });
    map.once("locationerror", () => {
      setLocating(false);
      alert("Location access denied.");
    });
  };

  return (
    <div
      onClick={handleClick}
      style={{
        position:     "absolute",
        top:          80,
        left:         10,
        zIndex:       1000,
        background:   locating ? "#22c55e22" : "#1e293b",
        border:       "1px solid #22c55e",
        borderRadius: 8,
        padding:      "8px 14px",
        cursor:       "pointer",
        color:        "#22c55e",
        fontWeight:   700,
        fontSize:     13,
        userSelect:   "none",
      }}
    >
      {locating ? "Locating..." : "My Location"}
    </div>
  );
}

// ── Custom icons ───────────────────────────────────────────────────────────────
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

const goldIcon = new L.Icon({
  iconUrl: "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-gold.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png",
  iconSize: [25, 41], iconAnchor: [12, 41],
  popupAnchor: [1, -34], shadowSize: [41, 41],
});

// ── Main component ─────────────────────────────────────────────────────────────
export default function DualRouteMap({
  routes, origin, destination,
  userLocation, heatmapData,
  showHeatmap, onMapClick, onLocate,
  mapStyleUrl, currentStep, directions,
  cityCenter,
}) {
  return (
    <MapContainer
      center={cityCenter || BENGALURU_CENTER}
      zoom={13}
      style={{ height: "100%", width: "100%", borderRadius: 12 }}
    >
      <TileLayer
        url={mapStyleUrl || "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"}
        attribution='&copy; CARTO'
      />

      <MapClickHandler onMapClick={onMapClick} />
      <LocateControl onLocate={onLocate} />

      {/* Fly to city when city selector changes */}
      {cityCenter && <FlyToCity center={cityCenter} />}

      {/* Fly to route bounds when route computed */}
      {routes && <FlyToBounds routes={routes} />}

      {/* Safety heatmap */}
      {showHeatmap && heatmapData.length > 0 && (
        <HeatmapLayer data={heatmapData} />
      )}

      {/* Safe route — solid green */}
      {routes?.safe_route?.coords && (
        <Polyline
          positions={routes.safe_route.coords}
          pathOptions={{ color: "#22c55e", weight: 6, opacity: 0.95 }}
        >
          <Popup>
            <b style={{color:"#22c55e"}}>Safe Route</b><br/>
            Score: {routes.safe_route.avg_safety_score.toFixed(1)}/100<br/>
            Time: {routes.safe_route.total_time_min.toFixed(1)} min<br/>
            Distance: {routes.safe_route.total_dist_km.toFixed(1)} km
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
            <b style={{color:"#3b82f6"}}>Fast Route</b><br/>
            Score: {routes.fast_route.avg_safety_score.toFixed(1)}/100<br/>
            Time: {routes.fast_route.total_time_min.toFixed(1)} min<br/>
            Distance: {routes.fast_route.total_dist_km.toFixed(1)} km
          </Popup>
        </Polyline>
      )}

      {/* User live location */}
      {userLocation && (
        <>
          <Circle
            center={[userLocation.lat, userLocation.lon]}
            radius={30}
            pathOptions={{
              color: "#6366f1", fillColor: "#6366f1",
              fillOpacity: 0.3, weight: 2,
            }}
          />
          <Marker
            position={[userLocation.lat, userLocation.lon]}
            icon={goldIcon}
          >
            <Popup>Your current location</Popup>
          </Marker>
        </>
      )}

      {/* Origin marker */}
      {origin && (
        <Marker position={[origin.lat, origin.lon]} icon={greenIcon}>
          <Popup>
            <b>Start</b><br/>
            {origin.lat.toFixed(4)}, {origin.lon.toFixed(4)}
          </Popup>
        </Marker>
      )}

      {/* Destination marker */}
      {destination && (
        <Marker position={[destination.lat, destination.lon]} icon={blueIcon}>
          <Popup>
            <b>Destination</b><br/>
            {destination.lat.toFixed(4)}, {destination.lon.toFixed(4)}
          </Popup>
        </Marker>
      )}
    </MapContainer>
  );
}