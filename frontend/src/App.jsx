import React, {
  useState, useEffect, useCallback, useRef
} from "react";
import api from "./api";
import DualRouteMap  from "./components/DualRouteMap";
import SafetyBar     from "./components/SafetyBar";
import AlphaSlider   from "./components/AlphaSlider";
import ReportButton  from "./components/ReportButton";
import useRoute      from "./hooks/useRoute";
import useHeatmap    from "./hooks/useHeatmap";


const CITY_CENTERS = {
  "Bengaluru":          [12.9716, 77.5946],
  "Mumbai":             [19.0760, 72.8777],
  "Delhi":              [28.6139, 77.2090],
  "Chennai":            [13.0827, 80.2707],
  "Hyderabad":          [17.3850, 78.4867],
  "Kolkata":            [22.5726, 88.3639],
  "Pune":               [18.5204, 73.8567],
  "Ahmedabad":          [23.0225, 72.5714],
  "Surat":              [21.1702, 72.8311],
  "Jaipur":             [26.9124, 75.7873],
  "Lucknow":            [26.8467, 80.9462],
  "Kanpur":             [26.4499, 80.3319],
  "Nagpur":             [21.1458, 79.0882],
  "Bhopal":             [23.2599, 77.4126],
  "Indore":             [22.7196, 75.8577],
  "Patna":              [25.5941, 85.1376],
  "Ranchi":             [23.3441, 85.3096],
  "Guwahati":           [26.1445, 91.7362],
  "Chandigarh":         [30.7333, 76.7794],
  "Amritsar":           [31.6340, 74.8723],
  "Ludhiana":           [30.9010, 75.8573],
  "Jodhpur":            [26.2389, 73.0243],
  "Kota":               [25.2138, 75.8648],
  "Jaipur":             [26.9124, 75.7873],
  "Vadodara":           [22.3072, 73.1812],
  "Rajkot":             [22.3039, 70.8022],
  "Nashik":             [19.9975, 73.7898],
  "Aurangabad":         [19.8762, 75.3433],
  "Kolhapur":           [16.7050, 74.2433],
  "Solapur":            [17.6599, 75.9064],
  "Visakhapatnam":      [17.6868, 83.2185],
  "Vijayawada":         [16.5062, 80.6480],
  "Warangal":           [17.9784, 79.5941],
  "Tirupati":           [13.6288, 79.4192],
  "Kochi":              [9.9312,  76.2673],
  "Thiruvananthapuram": [8.5241,  76.9366],
  "Kozhikode":          [11.2588, 75.7804],
  "Coimbatore":         [11.0168, 76.9558],
  "Madurai":            [9.9252,  78.1198],
  "Tiruchirappalli":    [10.7905, 78.7047],
  "Salem":              [11.6643, 78.1460],
  "Mysuru":             [12.2958, 76.6394],
  "Hubli":              [15.3647, 75.1240],
  "Mangalore":          [12.9141, 74.8560],
  "Bhubaneswar":        [20.2961, 85.8245],
  "Gwalior":            [26.2183, 78.1828],
  "Jabalpur":           [23.1815, 79.9864],
  "Raipur":             [21.2514, 81.6296],
  "Dehradun":           [30.3165, 78.0322],
  "Siliguri":           [26.7271, 88.3953],
  "Meerut":             [28.9845, 77.7064],
  "Agra":               [27.1767, 78.0081],
  "Varanasi":           [25.3176, 82.9739],
  "Allahabad":          [25.4358, 81.8463],
  "Kanpur":             [26.4499, 80.3319],
  "Faridabad":          [28.4089, 77.3178],
  "Gurugram":           [28.4595, 77.0266],
  "Noida":              [28.5355, 77.3910],
  "Ghaziabad":          [28.6692, 77.4538],
  "Bikaner":            [28.0229, 73.3119],
  "Aligarh":            [27.8974, 78.0880],
  "Bhavnagar":          [21.7645, 72.1519],
  "Jamnagar":           [22.4707, 70.0577],
  "Guntur":             [16.3067, 80.4365],
  "Nellore":            [14.4426, 79.9865],
  "Kurnool":            [15.8281, 78.0373],
  "Tirunelveli":        [8.7139,  77.7567],
  "Thrissur":           [10.5276, 76.2144],
};

const LANDMARKS = {
  "Bengaluru": [
    {name:"MG Road",      lat:12.9767,lon:77.6009},
    {name:"Koramangala",  lat:12.9352,lon:77.6245},
    {name:"Indiranagar",  lat:12.9718,lon:77.6412},
    {name:"Majestic",     lat:12.9767,lon:77.5713},
    {name:"Whitefield",   lat:12.9698,lon:77.7499},
  ],
  "Mumbai": [
    {name:"CST",          lat:18.9398,lon:72.8355},
    {name:"Bandra",       lat:19.0596,lon:72.8295},
    {name:"Andheri",      lat:19.1197,lon:72.8468},
    {name:"Dharavi",      lat:19.0380,lon:72.8528},
    {name:"Powai",        lat:19.1197,lon:72.9056},
  ],
  "Delhi": [
    {name:"Connaught Place",lat:28.6315,lon:77.2167},
    {name:"Lajpat Nagar",  lat:28.5700,lon:77.2433},
    {name:"Karol Bagh",    lat:28.6519,lon:77.1900},
    {name:"Dwarka",        lat:28.5921,lon:77.0460},
    {name:"Saket",         lat:28.5244,lon:77.2066},
  ],
  "Chennai": [
    {name:"T Nagar",       lat:13.0418,lon:80.2341},
    {name:"Anna Nagar",    lat:13.0850,lon:80.2101},
    {name:"Adyar",         lat:13.0012,lon:80.2565},
    {name:"Velachery",     lat:12.9789,lon:80.2208},
    {name:"Central",       lat:13.0827,lon:80.2707},
  ],
  "Hyderabad": [
    {name:"Charminar",     lat:17.3616,lon:78.4747},
    {name:"Hitech City",   lat:17.4504,lon:78.3805},
    {name:"Banjara Hills", lat:17.4126,lon:78.4483},
    {name:"Secunderabad",  lat:17.4399,lon:78.4983},
    {name:"Gachibowli",    lat:17.4401,lon:78.3489},
  ],
};

const DEFAULT_LANDMARKS = [
  {name:"City Center",    lat:0,lon:0},
];

const MAP_STYLES = [
  {id:"dark",    label:"Dark",      url:"https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"},
  {id:"light",   label:"Light",     url:"https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"},
  {id:"satellite",label:"Satellite",url:"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"},
  {id:"terrain", label:"Terrain",   url:"https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"},
  {id:"streets", label:"Streets",   url:"https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"},
];

const TRAVEL_MODES = [
  {id:"car",        label:"🚗 Car",       alpha:0.7},
  {id:"motorcycle", label:"🏍️ Bike",      alpha:0.75},
  {id:"walking",    label:"🚶 Walk",      alpha:0.9},
  {id:"cycling",    label:"🚴 Cycle",     alpha:0.85},
];

export default function App() {
  const [cityCenter, setCityCenter] = useState([12.9716, 77.5946]);
  const [cityLoading, setCityLoading] = useState(false);
  const [origin,         setOrigin]         = useState(null);
  const [destination,    setDestination]    = useState(null);
  const [alpha,          setAlpha]          = useState(0.7);
  const [hour,           setHour]           = useState(new Date().getHours());
  const [selectMode,     setSelectMode]     = useState("origin");
  const [showHeatmap,    setShowHeatmap]    = useState(true);
  const [selectedCity,   setSelectedCity]   = useState("Bengaluru");
  const currentCityRef = useRef("Bengaluru");

// Keep ref in sync with state
useEffect(() => {
  currentCityRef.current = selectedCity;
}, [selectedCity]);
  const [availCities,    setAvailCities]    = useState(["Bengaluru"]);
  const [travelMode,     setTravelMode]     = useState("car");
  const [mapStyle,       setMapStyle]       = useState(MAP_STYLES[0]);
  const [womenMode,      setWomenMode]      = useState(false);
  const [userLocation,   setUserLocation]   = useState(null);
  const [activeDir,      setActiveDir]      = useState(null);
  const [currentStep,    setCurrentStep]    = useState(0);
  const [tripStarted,    setTripStarted]    = useState(false);
  const [llmMessage,     setLlmMessage]     = useState("");
  const [serviceError,   setServiceError]   = useState(null);
  const [showDirPanel,   setShowDirPanel]   = useState(false);

  const { routes, loading, error, fetchRoutes, clearRoutes } = useRoute();
  const { heatmapData, fetchHeatmap } = useHeatmap();

  const effectiveAlpha = womenMode ? 0.9 : alpha;

  // Load available cities
  useEffect(() => {
  fetchHeatmap(3000);
  api.get("/cities").then(r => {
    if (r.data.cities?.length > 0) {
      setAvailCities(r.data.cities);
    }
  }).catch(() => {});
}, [fetchHeatmap]);
// Reload heatmap when city changes
useEffect(() => {
  fetchHeatmap(3000, selectedCity);
}, [selectedCity, fetchHeatmap]);

  // Auto-fetch routes
  useEffect(() => {
  if (origin && destination) {
    setServiceError(null);
    fetchRoutes({
      originLat:  origin.lat,
      originLon:  origin.lon,
      destLat:    destination.lat,
      destLon:    destination.lon,
      alpha:      effectiveAlpha,
      hour,
      city:       currentCityRef.current,
      mode:       travelMode,
      autoDetect: false,
    }).then(result => {
      if (result?.error === "service_unavailable") {
        setServiceError(result);
      }
    });
  }
}, [origin, destination, effectiveAlpha, hour, selectedCity, travelMode]);

  // LLM trip messages
  useEffect(() => {
    if (!routes || !tripStarted) return;
    const dirs = routes.safe_route?.directions || [];
    if (dirs.length === 0) return;

    const step = dirs[currentStep] || dirs[0];
    const score= step.safety_score || 50;

    if (score < 30) {
      setLlmMessage(
        `⚠️ Caution! You're entering a high-risk zone. `+
        `${step.instruction}. Stay alert and keep to well-lit areas.`
      );
    } else if (score > 70) {
      setLlmMessage(
        `✅ Safe stretch ahead. ${step.instruction}. `+
        `This area has good lighting and commercial activity.`
      );
    } else {
      setLlmMessage(`📍 ${step.instruction}`);
    }
  }, [currentStep, routes, tripStarted]);

  const handleMapClick = useCallback((e) => {
    const {lat, lng} = e.latlng;
    if (selectMode === "origin") {
      setOrigin({lat, lon: lng});
      setSelectMode("dest");
    } else {
      setDestination({lat, lon: lng});
      setSelectMode("origin");
    }
  }, [selectMode]);

  const handleLocate = useCallback((loc) => {
  setUserLocation(loc);
  setOrigin(loc);
  setSelectMode("dest");

  // Detect city and update BEFORE routing
  api.get("/cities/detect", {
    params: { lat: loc.lat, lon: loc.lon }
  }).then(r => {
    if (r.data.city) {
      setSelectedCity(r.data.city);
      // City selector will trigger re-route automatically
    }
  }).catch(() => {});
}, []);

  const handleClear = () => {
    setOrigin(null);
    setDestination(null);
    setUserLocation(null);
    setServiceError(null);
    setTripStarted(false);
    setCurrentStep(0);
    setLlmMessage("");
    setShowDirPanel(false);
    clearRoutes();
    setSelectMode("origin");
  };

  const startTrip = () => {
    if (!routes) return;
    setTripStarted(true);
    setShowDirPanel(true);
    setCurrentStep(0);
    setLlmMessage("🚀 Trip started! Follow the green route for maximum safety.");
    setActiveDir(routes.safe_route?.directions || []);
  };

  const shareWhatsApp = () => {
    if (!userLocation) return;
    const msg = encodeURIComponent(
      `🚨 I'm sharing my live location for safety.\n`+
      `Current location: https://maps.google.com/?q=${userLocation.lat},${userLocation.lon}\n`+
      `I'm using Fear-Free Navigator. Track me here:\n`+
      `https://maps.google.com/?q=${userLocation.lat},${userLocation.lon}`
    );
    window.open(`https://wa.me/?text=${msg}`, "_blank");
  };

  const landmarks = LANDMARKS[selectedCity] || DEFAULT_LANDMARKS;
  const isHighRisk = routes?.safe_route?.avg_safety_score < 40;

  return (
    <div style={{
      display:"flex", height:"100vh",
      background:"#0f172a", color:"#f1f5f9",
      fontFamily:"system-ui,sans-serif", fontSize:14,
      overflow:"hidden",
    }}>

      {/* ── Sidebar ── */}
      <div style={{
        width:340, minWidth:340,
        background:"#0f172a",
        borderRight:"1px solid #1e293b",
        display:"flex", flexDirection:"column",
        overflow:"hidden",
      }}>

        {/* Header */}
        <div style={{
          padding:"16px 20px 12px",
          borderBottom:"1px solid #1e293b",
          background:"linear-gradient(135deg,#0f172a,#1e293b)",
        }}>
          <div style={{
            fontSize:18,fontWeight:800,
            color:"#22c55e",marginBottom:2,
          }}>
            Fear-Free Navigator
          </div>
          <div style={{color:"#475569",fontSize:11}}>
            AI-powered safe routing · {selectedCity}
          </div>
        </div>

        {/* Scrollable content */}
        <div style={{flex:1,overflow:"auto",padding:12}}>

          {/* Service error */}
          {serviceError && (
            <div style={{
              background:"#f59e0b11",border:"1px solid #f59e0b",
              borderRadius:8,padding:"10px 14px",marginBottom:12,
              fontSize:12,
            }}>
              <div style={{color:"#f59e0b",fontWeight:700,marginBottom:4}}>
                Service Unavailable
              </div>
              <div style={{color:"#94a3b8"}}>
                {serviceError.message}
              </div>
              {serviceError.suggestion && (
                <div
                  style={{
                    color:"#22c55e",marginTop:6,cursor:"pointer",
                    textDecoration:"underline",
                  }}
                  onClick={() => setSelectedCity(serviceError.suggestion.split(": ")[1])}
                >
                  Switch to {serviceError.suggestion}
                </div>
              )}
            </div>
          )}

          {/* City selector */}
          <div style={{
            background:"#1e293b",borderRadius:8,
            padding:"10px 14px",marginBottom:10,
          }}>
            <div style={{
              fontSize:10,color:"#475569",marginBottom:6,
              textTransform:"uppercase",letterSpacing:1,
            }}>
              City
            </div>
            <select
              value={selectedCity}
              onChange={e => {
  const newCity = e.target.value;
  setSelectedCity(newCity);
  currentCityRef.current = newCity;
  handleClear();

  // Fly to new city
  const center = CITY_CENTERS[newCity] || [20.5937, 78.9629];
  setCityCenter(center);

  // Show loading
  setCityLoading(true);
  setTimeout(() => setCityLoading(false), 5000);
}}
              style={{
                width:"100%",padding:"6px 10px",borderRadius:6,
                background:"#0f172a",color:"#f1f5f9",
                border:"1px solid #334155",fontSize:13,cursor:"pointer",
              }}
            >
              {availCities.map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>

          {/* Travel mode */}
          <div style={{
            background:"#1e293b",borderRadius:8,
            padding:"10px 14px",marginBottom:10,
          }}>
            <div style={{
              fontSize:10,color:"#475569",marginBottom:8,
              textTransform:"uppercase",letterSpacing:1,
            }}>
              Travel Mode
            </div>
            <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
              {TRAVEL_MODES.map(m => (
                <button
                  key={m.id}
                  onClick={() => setTravelMode(m.id)}
                  style={{
                    flex:1,padding:"6px 4px",borderRadius:6,
                    border:`1px solid ${travelMode===m.id?"#22c55e":"#334155"}`,
                    background:travelMode===m.id?"#22c55e22":"transparent",
                    color:travelMode===m.id?"#22c55e":"#94a3b8",
                    cursor:"pointer",fontSize:11,fontWeight:600,
                    whiteSpace:"nowrap",
                  }}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          {/* Step indicator */}
          <div style={{
            background:"#1e293b",borderRadius:8,
            padding:"8px 14px",marginBottom:10,
            fontSize:12,color:"#94a3b8",
            display:"flex",alignItems:"center",gap:8,
          }}>
            <div style={{
              width:8,height:8,borderRadius:"50%",flexShrink:0,
              background:selectMode==="origin"?"#22c55e":"#3b82f6",
            }}/>
            {selectMode==="origin"
              ? "Set start point"
              : "Set destination"}
          </div>

          {/* Locations display */}
          {(origin||destination) && (
            <div style={{
              background:"#1e293b",borderRadius:8,
              padding:"10px 14px",marginBottom:10,fontSize:12,
            }}>
              {origin && (
                <div style={{marginBottom:4}}>
                  <span style={{color:"#22c55e"}}>Start: </span>
                  <span style={{color:"#94a3b8"}}>
                    {origin.lat.toFixed(4)}, {origin.lon.toFixed(4)}
                  </span>
                </div>
              )}
              {destination && (
                <div>
                  <span style={{color:"#3b82f6"}}>End: </span>
                  <span style={{color:"#94a3b8"}}>
                    {destination.lat.toFixed(4)}, {destination.lon.toFixed(4)}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Landmarks */}
          <div style={{marginBottom:10}}>
            <div style={{
              fontSize:10,color:"#475569",marginBottom:6,
              textTransform:"uppercase",letterSpacing:1,
            }}>
              Start
            </div>
            <div style={{display:"flex",flexWrap:"wrap",gap:5,marginBottom:8}}>
              {landmarks.map(lm => (
                <button
                  key={"o_"+lm.name}
                  onClick={() => {
                    setOrigin({lat:lm.lat,lon:lm.lon});
                    setSelectMode("dest");
                  }}
                  style={{
                    background:origin?.lat===lm.lat?"#22c55e22":"#1e293b",
                    border:`1px solid ${origin?.lat===lm.lat?"#22c55e":"#334155"}`,
                    borderRadius:5,
                    color:origin?.lat===lm.lat?"#22c55e":"#94a3b8",
                    fontSize:10,padding:"3px 8px",cursor:"pointer",
                  }}
                >
                  {lm.name}
                </button>
              ))}
            </div>
            <div style={{
              fontSize:10,color:"#475569",marginBottom:6,
              textTransform:"uppercase",letterSpacing:1,
            }}>
              Destination
            </div>
            <div style={{display:"flex",flexWrap:"wrap",gap:5}}>
              {landmarks.map(lm => (
                <button
                  key={"d_"+lm.name}
                  onClick={() => {
                    setDestination({lat:lm.lat,lon:lm.lon});
                    setSelectMode("origin");
                  }}
                  style={{
                    background:destination?.lat===lm.lat?"#3b82f622":"#1e293b",
                    border:`1px solid ${destination?.lat===lm.lat?"#3b82f6":"#334155"}`,
                    borderRadius:5,
                    color:destination?.lat===lm.lat?"#3b82f6":"#94a3b8",
                    fontSize:10,padding:"3px 8px",cursor:"pointer",
                  }}
                >
                  {lm.name}
                </button>
              ))}
            </div>
          </div>

          {/* Women safety mode */}
          <div
            onClick={() => setWomenMode(w => !w)}
            style={{
              background:womenMode?"#ec489911":"#1e293b",
              border:`1px solid ${womenMode?"#ec4899":"#334155"}`,
              borderRadius:8,padding:"10px 14px",marginBottom:10,
              cursor:"pointer",display:"flex",alignItems:"center",gap:10,
            }}
          >
            <div style={{
              width:32,height:32,borderRadius:"50%",
              background:womenMode?"#ec4899":"#334155",
              display:"flex",alignItems:"center",
              justifyContent:"center",fontSize:16,flexShrink:0,
            }}>
              👩
            </div>
            <div>
              <div style={{
                color:womenMode?"#ec4899":"#f1f5f9",
                fontWeight:700,fontSize:12,
              }}>
                Women Safety Mode
              </div>
              <div style={{color:"#475569",fontSize:10}}>
                {womenMode
                  ? "Active — max safety (α=0.9)"
                  : "Tap to enable"}
              </div>
            </div>
          </div>

          {/* Alpha slider */}
          {!womenMode && <AlphaSlider alpha={alpha} onChange={setAlpha}/>}

          {/* Hour slider */}
          <div style={{
            background:"#1e293b",borderRadius:8,
            padding:"10px 14px",marginBottom:10,
          }}>
            <div style={{
              display:"flex",justifyContent:"space-between",
              alignItems:"center",marginBottom:6,
            }}>
              <span style={{color:"#94a3b8",fontSize:12}}>Time of day</span>
              <span style={{color:"#f1f5f9",fontWeight:700,fontSize:13}}>
                {String(hour).padStart(2,"0")}:00
                {hour>=20||hour<=5?" 🌙":" ☀️"}
              </span>
            </div>
            <input type="range" min="0" max="23" step="1" value={hour}
              onChange={e => setHour(parseInt(e.target.value))}
              style={{width:"100%",accentColor:"#6366f1"}}
            />
          </div>

          {/* City loading */}
{cityLoading && (
  <div style={{
    background:   "#6366f111",
    border:       "1px solid #6366f1",
    borderRadius: 8,
    padding:      "8px 14px",
    marginBottom: 10,
    fontSize:     12,
    color:        "#6366f1",
    textAlign:    "center",
  }}>
    Loading {selectedCity} road network...
  </div>
)}

{/* Route loading */}
{loading && (
  <div style={{
    textAlign:"center",color:"#6366f1",
    padding:12,fontSize:12,
  }}>
    Computing safest route...
  </div>
)}

          {/* Error */}
          {error && !serviceError && (
            <div style={{
              background:"#ef444411",border:"1px solid #ef4444",
              borderRadius:8,padding:"8px 12px",
              color:"#ef4444",fontSize:12,marginBottom:10,
            }}>
              {error}
            </div>
          )}

          {/* Route comparison */}
          {routes && !loading && !serviceError && (
            <SafetyBar
              safeRoute  = {routes.safe_route}
              fastRoute  = {routes.fast_route}
              comparison = {routes.comparison}
            />
          )}

          {/* LLM message */}
          {tripStarted && llmMessage && (
            <div style={{
              background:"#6366f111",border:"1px solid #6366f1",
              borderRadius:8,padding:"10px 14px",marginBottom:10,
              fontSize:12,color:"#c4b5fd",
            }}>
              {llmMessage}
            </div>
          )}

          {/* Start trip button */}
          {routes && !loading && !tripStarted && (
            <button
              onClick={startTrip}
              style={{
                width:"100%",padding:"10px",borderRadius:8,
                background:"#22c55e",border:"none",
                color:"#fff",fontWeight:700,fontSize:13,
                cursor:"pointer",marginBottom:8,
              }}
            >
              Start Safe Trip
            </button>
          )}

          {/* High risk — WhatsApp share */}
          {(isHighRisk || womenMode) && userLocation && (
            <button
              onClick={shareWhatsApp}
              style={{
                width:"100%",padding:"10px",borderRadius:8,
                background:"#25D366",border:"none",
                color:"#fff",fontWeight:700,fontSize:12,
                cursor:"pointer",marginBottom:8,
                display:"flex",alignItems:"center",
                justifyContent:"center",gap:8,
              }}
            >
              📱 Share Live Location on WhatsApp
            </button>
          )}

          {/* Report button */}
          {origin && <ReportButton lat={origin.lat} lon={origin.lon}/>}

          {/* Controls row */}
          <div style={{display:"flex",gap:6,marginTop:8}}>
            <button
              onClick={() => setShowHeatmap(h => !h)}
              style={{
                flex:1,padding:"8px",borderRadius:6,
                border:`1px solid ${showHeatmap?"#6366f1":"#334155"}`,
                background:showHeatmap?"#6366f122":"transparent",
                color:showHeatmap?"#6366f1":"#94a3b8",
                cursor:"pointer",fontSize:11,
              }}
            >
              {showHeatmap?"Hide":"Show"} Heatmap
            </button>
            <button
              onClick={() => setShowDirPanel(d => !d)}
              style={{
                flex:1,padding:"8px",borderRadius:6,
                border:`1px solid ${showDirPanel?"#f59e0b":"#334155"}`,
                background:showDirPanel?"#f59e0b22":"transparent",
                color:showDirPanel?"#f59e0b":"#94a3b8",
                cursor:"pointer",fontSize:11,
              }}
            >
              Directions
            </button>
            <button
              onClick={handleClear}
              style={{
                flex:1,padding:"8px",borderRadius:6,
                border:"1px solid #334155",background:"transparent",
                color:"#94a3b8",cursor:"pointer",fontSize:11,
              }}
            >
              Clear
            </button>
          </div>
        </div>

        {/* Footer */}
        <div style={{
          padding:"8px 16px",borderTop:"1px solid #1e293b",
          fontSize:10,color:"#334155",textAlign:"center",
        }}>
          Green = Safe · Blue dashed = Fast · {selectedCity}
        </div>
      </div>

      {/* ── Main area ── */}
      <div style={{flex:1,display:"flex",flexDirection:"column",position:"relative"}}>

        {/* Map style selector */}
        <div style={{
          position:"absolute",top:10,right:10,zIndex:1000,
          display:"flex",gap:4,
          background:"#0f172acc",borderRadius:8,padding:6,
        }}>
          {MAP_STYLES.map(s => (
            <button
              key={s.id}
              onClick={() => setMapStyle(s)}
              style={{
                padding:"4px 10px",borderRadius:6,
                border:`1px solid ${mapStyle.id===s.id?"#22c55e":"#334155"}`,
                background:mapStyle.id===s.id?"#22c55e22":"transparent",
                color:mapStyle.id===s.id?"#22c55e":"#94a3b8",
                cursor:"pointer",fontSize:11,fontWeight:600,
              }}
            >
              {s.label}
            </button>
          ))}
        </div>

        {/* Directions panel */}
        {showDirPanel && routes?.safe_route?.directions && (
          <div style={{
            position:"absolute",top:10,left:10,zIndex:1000,
            width:280,maxHeight:"70vh",overflow:"auto",
            background:"#0f172aee",borderRadius:12,
            padding:12,border:"1px solid #1e293b",
          }}>
            <div style={{
              color:"#22c55e",fontWeight:700,
              fontSize:13,marginBottom:10,
            }}>
              Turn-by-Turn Directions
            </div>
            {routes.safe_route.directions.map((dir, i) => (
              <div
                key={i}
                onClick={() => setCurrentStep(i)}
                style={{
                  padding:"8px 10px",borderRadius:6,
                  marginBottom:4,cursor:"pointer",
                  background:currentStep===i?"#22c55e22":"#1e293b",
                  border:`1px solid ${currentStep===i?"#22c55e":"#334155"}`,
                }}
              >
                <div style={{
                  display:"flex",gap:8,
                  alignItems:"center",
                }}>
                  <span style={{fontSize:16}}>
                    {_dirIcon(dir.type)}
                  </span>
                  <div>
                    <div style={{
                      color:"#f1f5f9",fontSize:11,
                      fontWeight:currentStep===i?700:400,
                    }}>
                      {dir.instruction}
                    </div>
                    <div style={{color:"#475569",fontSize:10,marginTop:2}}>
                      {dir.distance_m > 0
                        ? `${dir.distance_m.toFixed(0)}m`
                        : ""
                      }
                      {dir.safety_score && (
                        <span style={{
                          marginLeft:6,
                          color:dir.safety_color,
                        }}>
                          Safety: {dir.safety_score.toFixed(0)}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Map */}
        <div style={{flex:1}}>
          <DualRouteMap
            routes       = {routes}
            origin       = {origin}
            destination  = {destination}
            userLocation = {userLocation}
            heatmapData  = {heatmapData}
            showHeatmap  = {showHeatmap}
            onMapClick   = {handleMapClick}
            onLocate     = {handleLocate}
            mapStyleUrl  = {mapStyle.url}
            currentStep  = {currentStep}
            directions   = {routes?.safe_route?.directions}
            cityCenter   = {cityCenter}   
          />
        </div>
      </div>
    </div>
  );
}

function _dirIcon(type) {
  const icons = {
    start:"🚀", arrive:"🏁",
    left:"↰", right:"↱",
    sharp_left:"⬅️", sharp_right:"➡️",
    slight_left:"↖️", slight_right:"↗️",
    straight:"⬆️", uturn:"↩️",
  };
  return icons[type] || "⬆️";
}