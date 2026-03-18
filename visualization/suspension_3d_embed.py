"""
visualization/suspension_3d_embed.py — Streamlit Integration for 3D Suspension Viz
====================================================================================
Drop-in module that embeds the Three.js 3D suspension visualizer into the
existing Streamlit dashboard.

Usage A — Standalone tab (add to dashboard.py sidebar router):
    from visualization.suspension_3d_embed import render_3d_suspension
    # In the sidebar router:
    if mode == "Suspension 3D":
        render_3d_suspension()

Usage B — Inject into existing SuspensionVisualizer.render() as a 2D/3D toggle:
    from visualization.suspension_3d_embed import render_3d_suspension
    # At the top of SuspensionVisualizer.render(), add:
    view_mode = st.radio('VIEW MODE', ['2D Schematic', '3D Interactive'],
                          horizontal=True)
    if view_mode == '3D Interactive':
        render_3d_suspension()
        return
    # ... rest of existing 2D render code ...
"""
from __future__ import annotations
import os
import streamlit as st
import streamlit.components.v1 as components


def _find_html_path() -> str:
    """Locate the suspension_3d_ter26.html file relative to this module."""
    candidates = [
        # Same directory as this module
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'suspension_3d_ter26.html'),
        # Project root / visualization
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'visualization', 'suspension_3d_ter26.html'),
        # Project root directly
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'suspension_3d_ter26.html'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"suspension_3d_ter26.html not found. Searched:\n"
        + "\n".join(f"  · {c}" for c in candidates)
    )


@st.cache_resource
def _load_html() -> str:
    """Load and cache the HTML content (only reads disk once per session)."""
    path = _find_html_path()
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def render_3d_suspension(height: int = 720) -> None:
    """
    Render the 3D interactive suspension visualizer inside a Streamlit app.

    Parameters
    ----------
    height : int
        Pixel height of the embedded iframe. Default 720 fits most monitors
        while leaving room for the Streamlit sidebar and header.
    """
    html_content = _load_html()
    components.html(html_content, height=height, scrolling=False)