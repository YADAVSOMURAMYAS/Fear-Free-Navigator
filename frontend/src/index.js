import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// Load leaflet.heat
require("leaflet.heat");
window.L = L;

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);