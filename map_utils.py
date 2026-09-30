"""
Map helpers: city coordinates, travel-guide markers, lightweight geocoding.
Used by the Streamlit app and the embedded AI-TRAVEL-GUIDE map component.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any
from urllib.parse import quote

import requests

# Major Indian destinations (lat, lon) — aligned with app CITIES list
INDIA_CITY_COORDS: dict[str, tuple[float, float]] = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Kochi": (9.9312, 76.2673),
    "Goa": (15.2993, 74.1240),
    "Jaipur": (26.9124, 75.7873),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Bhopal": (23.2599, 77.4126),
    "Lucknow": (26.8467, 80.9462),
    "Patna": (25.5941, 85.1376),
    "Bhubaneswar": (20.2961, 85.8245),
    "Guwahati": (26.1445, 91.7362),
    "Srinagar": (34.0837, 74.7973),
    "Chandigarh": (30.7333, 76.7794),
    "Indore": (22.7196, 75.8577),
    "Nagpur": (21.1458, 79.0882),
    "Varanasi": (25.3176, 82.9739),
    "Manali": (32.2396, 77.1887),
    "Shimla": (31.1048, 77.1734),
    "Darjeeling": (27.0360, 88.2627),
    "Udaipur": (24.5854, 73.7125),
    "Mysuru": (12.2958, 76.6394),
    "Amritsar": (31.6340, 74.8723),
    "Rishikesh": (30.0869, 78.2676),
    "Haridwar": (29.9457, 78.1642),
    "Ooty": (11.4064, 76.6932),
    "Munnar": (10.0889, 77.0595),
    "Port Blair": (11.6234, 92.7265),
    "Ladakh": (34.1526, 77.5771),
}

MARKER_COLORS = {
    "destination": "#e63946",
    "attraction": "#1d3557",
    "gem": "#2a9d8f",
    "itinerary": "#e9c46a",
    "hotel": "#457b9d",
    "restaurant": "#f4a261",
    "default": "#6c757d",
}


def get_city_coords(destination: str) -> tuple[float, float] | None:
    key = (destination or "").strip()
    if not key:
        return None
    if key in INDIA_CITY_COORDS:
        return INDIA_CITY_COORDS[key]
    for name, coords in INDIA_CITY_COORDS.items():
        if name.lower() == key.lower():
            return coords
    return None


@lru_cache(maxsize=64)
def geocode_place(place_name: str, near_city: str = "") -> tuple[float, float] | None:
    """Resolve a place name near an Indian city via Nominatim (cached)."""
    place_name = (place_name or "").strip()
    if not place_name or len(place_name) < 3:
        return None

    query = f"{place_name}, {near_city}, India" if near_city else f"{place_name}, India"
    try:
        time.sleep(1.05)  # Nominatim usage policy: max 1 req/sec
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "IndiaTripPlanner/1.0"},
            timeout=8,
        )
        if resp.status_code != 200:
            return None
        results = resp.json()
        if not results:
            return None
        return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        return None


def _offset_coords(
    base: tuple[float, float], index: int, total: int = 8
) -> tuple[float, float]:
    """Spread markers around the city center when geocoding is skipped."""
    import math

    angle = (index / max(total, 1)) * 2 * math.pi
    radius = 0.018 * (1 + index // 5)
    return base[0] + radius * math.cos(angle), base[1] + radius * math.sin(angle)


def _add_marker(
    markers: list[dict[str, Any]],
    seen: set[str],
    name: str,
    mtype: str,
    desc: str,
    destination: str,
    coords: tuple[float, float] | None,
    *,
    base_coords: tuple[float, float] | None = None,
    offset_index: int = 0,
    try_geocode: bool = False,
) -> None:
    name = (name or "").strip()
    if not name or name.lower() in seen:
        return
    seen.add(name.lower())

    lat, lon = (None, None)
    if coords:
        lat, lon = coords
    elif try_geocode:
        resolved = geocode_place(name, destination)
        if resolved:
            lat, lon = resolved
    if lat is None and base_coords:
        lat, lon = _offset_coords(base_coords, offset_index)

    if lat is None:
        return

    markers.append(
        {
            "name": name,
            "lat": lat,
            "lng": lon,
            "type": mtype,
            "color": MARKER_COLORS.get(mtype, MARKER_COLORS["default"]),
            "desc": (desc or "")[:220],
        }
    )


def guide_to_map_markers(guide: dict, destination: str, max_markers: int = 12) -> list[dict]:
    """Build map marker list from a Gemini / fallback travel guide."""
    markers: list[dict[str, Any]] = []
    seen: set[str] = set()
    dest_coords = get_city_coords(destination)

    if dest_coords:
        lat, lon = dest_coords
        markers.append(
            {
                "name": destination,
                "lat": lat,
                "lng": lon,
                "type": "destination",
                "color": MARKER_COLORS["destination"],
                "desc": guide.get("overview", "")[:220],
            }
        )
        seen.add(destination.lower())

    offset_i = 0
    for item in (guide.get("attractions") or [])[:4]:
        if len(markers) >= max_markers:
            break
        _add_marker(
            markers, seen,
            item.get("name", ""),
            "attraction",
            item.get("desc", ""),
            destination,
            None,
            base_coords=dest_coords,
            offset_index=offset_i,
            try_geocode=True,
        )
        offset_i += 1

    for item in (guide.get("gems") or [])[:3]:
        if len(markers) >= max_markers:
            break
        _add_marker(
            markers, seen,
            item.get("name", ""),
            "gem",
            item.get("desc", "") or item.get("why_special", ""),
            destination,
            None,
            base_coords=dest_coords,
            offset_index=offset_i,
            try_geocode=True,
        )
        offset_i += 1

    for day_places in (guide.get("itineraries") or {}).values():
        for item in (day_places or [])[:2]:
            if len(markers) >= max_markers:
                break
            _add_marker(
                markers, seen,
                item.get("place", ""),
                "itinerary",
                item.get("desc", ""),
                destination,
                None,
                base_coords=dest_coords,
                offset_index=offset_i,
            )
            offset_i += 1

    for item in (guide.get("hotels") or [])[:2]:
        if len(markers) >= max_markers:
            break
        _add_marker(
            markers, seen, item.get("name", ""), "hotel", item.get("desc", ""),
            destination, None, base_coords=dest_coords, offset_index=offset_i,
        )
        offset_i += 1

    for item in (guide.get("restaurants") or [])[:2]:
        if len(markers) >= max_markers:
            break
        _add_marker(
            markers, seen, item.get("name", ""), "restaurant", item.get("desc", ""),
            destination, None, base_coords=dest_coords, offset_index=offset_i,
        )
        offset_i += 1

    return markers


def build_map_payload(
    destination: str,
    guide: dict | None = None,
    route_from: str | None = None,
) -> dict[str, Any]:
    """Payload passed into the embedded Leaflet HTML component."""
    center = get_city_coords(destination)
    markers: list[dict] = []

    if guide:
        markers = guide_to_map_markers(guide, destination)
    elif center:
        lat, lon = center
        markers = [
            {
                "name": destination,
                "lat": lat,
                "lng": lon,
                "type": "destination",
                "color": MARKER_COLORS["destination"],
                "desc": f"Explore {destination}, India",
            }
        ]

    route_line: list[list[float]] | None = None
    if route_from and center:
        origin = get_city_coords(route_from)
        if origin:
            route_line = [
                [origin[0], origin[1]],
                [center[0], center[1]],
            ]

    zoom = 11 if len(markers) <= 3 else 10
    if center:
        center_lat, center_lon = center
    elif markers:
        center_lat = sum(m["lat"] for m in markers) / len(markers)
        center_lon = sum(m["lng"] for m in markers) / len(markers)
    else:
        center_lat, center_lon = 20.5937, 78.9629  # India center
        zoom = 5

    return {
        "destination": destination,
        "center": [center_lat, center_lon],
        "zoom": zoom,
        "markers": markers,
        "route": route_line,
        "mapLabel": (guide or {}).get("mapLabel", f"{destination} Region"),
    }


def google_maps_directions_url(origin: str, destination: str) -> str:
    return (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={quote(origin + ', India')}"
        f"&destination={quote(destination + ', India')}"
    )


def google_maps_place_url(place: str, destination: str) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={quote(place + ', ' + destination + ', India')}"
