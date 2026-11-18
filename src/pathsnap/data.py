"""Raw data loading and projection helpers for PathSnap."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import Transformer

def local_projector(lon0: float, lat0: float):
    """Per-trip azimuthal-equidistant projection (metres) centred on the trip.

    The trips are spread over several continents, so a single UTM zone would be wrong.
    Returns (forward, inverse) transformers.
    """
    crs = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    return (Transformer.from_crs(4326, crs, always_xy=True),
            Transformer.from_crs(crs, 4326, always_xy=True))


@dataclass
class Trip:
    trip_id: str
    lon: np.ndarray
    lat: np.ndarray
    t: np.ndarray
    px: np.ndarray          # projected GPS x (m)
    py: np.ndarray
    node_x: np.ndarray      # projected node coords (m)
    node_y: np.ndarray
    node_lon: np.ndarray
    node_lat: np.ndarray
    arcs: np.ndarray        # (n_arcs, 2) from/to node ids
    route: np.ndarray | None  # reference arc ids (None when labels withheld)
    inv: object = None        # projected -> lon/lat transformer


def list_trips(root) -> list[str]:
    root = Path(root)
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def _load2(path, dtype=float, ncol=2):
    a = np.loadtxt(path, dtype=dtype, ndmin=2)
    return a.reshape(-1, ncol)


def load_trip(root, trip_id: str, with_route: bool = True) -> Trip:
    base = Path(root) / trip_id / trip_id
    track = _load2(f"{base}.track", float, 3)
    nodes = _load2(f"{base}.nodes", float, 2)
    arcs = _load2(f"{base}.arcs", int, 2)
    route = None
    rp = Path(f"{base}.route")
    if with_route and rp.exists():
        route = np.loadtxt(rp, dtype=int, ndmin=1)
    good = np.isfinite(track[:, :2]).all(axis=1)
    lon0, lat0 = np.median(track[good, 0]), np.median(track[good, 1])
    fwd, inv = local_projector(lon0, lat0)
    px, py = fwd.transform(track[:, 0], track[:, 1])
    nx_, ny_ = fwd.transform(nodes[:, 0], nodes[:, 1])
    return Trip(trip_id, track[:, 0], track[:, 1], track[:, 2], np.asarray(px), np.asarray(py),
                np.asarray(nx_), np.asarray(ny_), nodes[:, 0], nodes[:, 1], arcs, route, inv)
