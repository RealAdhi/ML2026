"""Folium map generation for PathSnap (GPS, road network, reference route, predictions)."""
from __future__ import annotations

import folium
import numpy as np


def _arc_line(trip, a):
    u, v = trip.arcs[a]
    return [(trip.node_lat[u], trip.node_lon[u]), (trip.node_lat[v], trip.node_lon[v])]


def trip_map(trip, network_arcs=None, ref_arcs=None, clf_arcs=None, hmm_arcs=None, max_gps=4000, title=""):
    """Interactive map with toggleable layers: GPS, road network, reference, classifier-only, classifier+HMM."""
    m = folium.Map(location=[float(np.median(trip.lat)), float(np.median(trip.lon))], zoom_start=14,
                   tiles="OpenStreetMap", control_scale=True)
    if network_arcs is not None:
        g = folium.FeatureGroup(name="Road network (candidate arcs)", show=True)
        for a in np.unique(network_arcs):
            folium.PolyLine(_arc_line(trip, a), color="#9e9e9e", weight=2, opacity=0.6).add_to(g)
        g.add_to(m)
    if ref_arcs is not None:
        g = folium.FeatureGroup(name="Reference route", show=True)
        for a in ref_arcs:
            folium.PolyLine(_arc_line(trip, a), color="#ff9800", weight=8, opacity=0.55).add_to(g)
        g.add_to(m)
    if clf_arcs is not None:
        g = folium.FeatureGroup(name="Predicted: classifier only", show=False)
        prev = None
        for a in clf_arcs:
            if a != prev:
                folium.PolyLine(_arc_line(trip, a), color="#e53935", weight=4, opacity=0.8, dash_array="6 6").add_to(g)
            prev = a
        g.add_to(m)
    if hmm_arcs is not None:
        g = folium.FeatureGroup(name="Predicted: classifier + HMM", show=True)
        prev = None
        for a in hmm_arcs:
            if a != prev:
                folium.PolyLine(_arc_line(trip, a), color="#2e7d32", weight=4, opacity=0.95).add_to(g)
            prev = a
        g.add_to(m)
    g = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(trip.lat) // max_gps)
    for i in range(0, len(trip.lat), step):
        folium.CircleMarker((trip.lat[i], trip.lon[i]), radius=2, color="#1565c0", fill=True, fill_opacity=0.8,
                            popup=f"obs {i}, t={trip.t[i]:.0f}s").add_to(g)
    g.add_to(m)
    folium.Marker((trip.lat[0], trip.lon[0]), tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((trip.lat[-1], trip.lon[-1]), tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = f"""<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 10px;
      border:1px solid #888;border-radius:4px;font-size:12px;line-height:1.6">
      <b>{title}</b><br>
      <span style="color:#1565c0">&#9679;</span> GPS observations<br>
      <span style="color:#9e9e9e">&#9473;</span> road network (candidate arcs)<br>
      <span style="color:#ff9800">&#9473;</span> reference route<br>
      <span style="color:#e53935">- -</span> classifier-only prediction<br>
      <span style="color:#2e7d32">&#9473;</span> classifier + HMM prediction</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    m.fit_bounds([[float(trip.lat.min()), float(trip.lon.min())], [float(trip.lat.max()), float(trip.lon.max())]])
    return m
