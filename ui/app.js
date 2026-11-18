// PathSnap evaluator UI: lists trips from the API, sends the chosen trip id, renders the result.
// The reference route is never sent to the browser; only server-computed metrics are shown.
const $ = (id) => document.getElementById(id);
let API = "";
const map = L.map("map").setView([49.87, 8.65], 12);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "&copy; OpenStreetMap" }).addTo(map);
let layers = { gps: L.layerGroup(), hmm: L.layerGroup(), clf: L.layerGroup() };
let ctrl = null, legend = null;

function setStatus(ok, text) { const s = $("status"); s.className = "pill " + (ok ? "on" : "off"); s.textContent = text; }

async function connect() {
  API = $("apiUrl").value.replace(/\/$/, "");
  try {
    const h = await (await fetch(API + "/health")).json();
    setStatus(true, "connected: " + h.model);
    const t = await (await fetch(API + "/trips")).json();
    renderTrips(t.trips);
  } catch (e) {
    setStatus(false, "connection failed");
    $("tripHint").innerHTML = '<span class="err">Cannot reach ' + API + ' (' + e + ')</span>';
  }
}

function renderTrips(trips) {
  $("tripHint").textContent = trips.length + " trips available. Click one to run the pipeline.";
  const ul = $("tripList"); ul.innerHTML = "";
  trips.forEach((t) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.innerHTML = "Trip " + t.trip_id + "<small>" + t.n_gps + " GPS points</small>";
    b.onclick = () => runTrip(t.trip_id, b);
    li.appendChild(b); ul.appendChild(li);
  });
}

const pct = (x) => (x == null || isNaN(x) ? "n/a" : (100 * x).toFixed(2) + "%");

async function runTrip(id, btn) {
  document.querySelectorAll("#tripList button").forEach((b) => b.classList.remove("sel"));
  btn.classList.add("sel");
  $("results").className = ""; $("results").textContent = "Running pipeline on server...";
  try {
    const r = await fetch(API + "/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trip_id: id }) });
    if (!r.ok) throw new Error("HTTP " + r.status);
    show(await r.json());
  } catch (e) { $("results").innerHTML = '<span class="err">Request failed: ' + e + "</span>"; }
}

function show(d) {
  Object.values(layers).forEach((l) => l.clearLayers());
  if (ctrl) map.removeControl(ctrl);
  if (legend) map.removeControl(legend);
  d.gps.forEach((p) => L.circleMarker(p, { radius: 2.5, color: "#1565c0", weight: 1, fillOpacity: 0.8 }).addTo(layers.gps));
  L.polyline(d.gps, { color: "#90caf9", weight: 1, opacity: 0.6 }).addTo(layers.gps);
  d.classifier_only_route.forEach((s) => L.polyline(s, { color: "#e53935", weight: 3, opacity: 0.7, dashArray: "4 6" }).addTo(layers.clf));
  d.predicted_route.forEach((s) => L.polyline(s, { color: "#2e7d32", weight: 5, opacity: 0.9 }).addTo(layers.hmm));
  layers.gps.addTo(map); layers.hmm.addTo(map);
  ctrl = L.control.layers(null, { "GPS observations": layers.gps, "Predicted route (classifier + HMM)": layers.hmm, "Classifier-only arcs": layers.clf }, { collapsed: false }).addTo(map);
  legend = L.control({ position: "bottomright" });
  legend.onAdd = () => { const div = L.DomUtil.create("div", "legend");
    div.innerHTML = '<i style="background:#1565c0"></i>GPS observations<br><i style="background:#2e7d32"></i>Predicted route (HMM)<br><i style="background:#e53935"></i>Classifier-only'; return div; };
  legend.addTo(map);
  map.fitBounds(L.latLngBounds(d.gps), { padding: [30, 30] });

  const m = d.metrics;
  if (m.available) {
    $("results").innerHTML = "<table><tr><th>Trip " + d.trip_id + " (" + d.n_obs + " obs)</th><th>Classifier</th><th>+ HMM</th></tr>" +
      row("Point accuracy", m.clf.acc, m.hmm.acc) + row("Route continuity", m.clf.continuity, m.hmm.continuity) +
      row("Route F1 (arc set)", m.clf.route_f1, m.hmm.route_f1) + row("Route precision", m.clf.route_precision, m.hmm.route_precision) +
      row("Route recall", m.clf.route_recall, m.hmm.route_recall) + "</table>" +
      '<p class="hint">Model: ' + d.model + "</p>";
  } else { $("results").textContent = "No reference available for this trip."; }
  const t = d.timing;
  $("timing").textContent = "Server time " + t.total_s.toFixed(2) + " s (candidates " + t.candidates_s.toFixed(2) + ", classifier " + t.classifier_s.toFixed(2) + ", HMM " + t.hmm_s.toFixed(2) + ")";
}
function row(name, a, b) { return "<tr><td>" + name + "</td><td>" + pct(a) + "</td><td class='" + (b >= a ? "good" : "") + "'>" + pct(b) + "</td></tr>"; }

$("connect").onclick = connect;
const q = new URLSearchParams(location.search).get("api"); if (q) $("apiUrl").value = q;
connect();
