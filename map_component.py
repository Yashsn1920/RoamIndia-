"""
Streamlit component: embed AI-TRAVEL-GUIDE style Leaflet map.
"""

from __future__ import annotations

import json
import os

import streamlit as st
import streamlit.components.v1 as components

from map_utils import build_map_payload, google_maps_directions_url, google_maps_place_url

_COMPONENT_HTML = os.path.join(
    os.path.dirname(__file__), "components", "travel_map.html"
)


def render_travel_map(
    destination: str,
    guide: dict | None = None,
    route_from: str | None = None,
    height: int = 480,
) -> None:
    """Render interactive map with optional travel-guide markers."""
    payload = build_map_payload(destination, guide=guide, route_from=route_from)
    with open(_COMPONENT_HTML, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__MAP_DATA__", json.dumps(payload))
    components.html(html, height=height, scrolling=False)


def render_map_links(destination: str, guide: dict | None = None) -> None:
    """Quick links to external maps (Google Maps, full tourist site)."""
    cols = st.columns(3)
    with cols[0]:
        st.link_button(
            "Open in Google Maps",
            f"https://www.google.com/maps/search/?api=1&query={destination}%2C+India",
            use_container_width=True,
        )
    with cols[1]:
        if guide and guide.get("attractions"):
            first = guide["attractions"][0].get("name", "")
            if first:
                st.link_button(
                    f"Directions to {first[:20]}…",
                    google_maps_place_url(first, destination),
                    use_container_width=True,
                )
    with cols[2]:
        st.link_button(
            "Full tourist map site",
            "http://localhost:3000/dashboard.html",
            use_container_width=True,
            help="Start AI-TRAVEL-GUIDE: cd AI-TRAVEL-GUIDE && npm start",
        )
