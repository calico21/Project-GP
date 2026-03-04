"""
simulator/rerun_compat.py
─────────────────────────────────────────────────────────────────────────────
Rerun SDK version compatibility layer.

The Rerun API has changed significantly across versions:
  v0.9–0.14:  set_time_seconds / rr.Scalars / rr.TextDocument
  v0.15–0.16: set_time_nanos / rr.TimeSeriesScalar / rr.TextLog
  v0.17+:     set_time_nanos still works, further archetype changes

This module auto-detects the installed version and provides a stable
interface used throughout the Project-GP simulator.

Usage:
    from simulator.rerun_compat import rr, rr_set_time, rr_scalar, rr_text
"""

import rerun as rr
import numpy as np
import math

# ── Detect version ────────────────────────────────────────────────────────────
try:
    _VER = tuple(int(x) for x in rr.__version__.split('.')[:2])
except Exception:
    _VER = (0, 0)

# ── Time API ──────────────────────────────────────────────────────────────────

def rr_set_time(sim_time: float, frame_id: int = 0):
    """Set both a wall-time and frame-counter timeline, version-safe."""
    # Try each API variant in order of recency
    try:
        rr.set_time_nanos("sim_time", int(sim_time * 1e9))
    except AttributeError:
        try:
            rr.set_time_seconds("sim_time", sim_time)
        except AttributeError:
            pass
    try:
        rr.set_time_sequence("frame", frame_id)
    except AttributeError:
        pass


# ── Scalar logging ─────────────────────────────────────────────────────────────

def rr_scalar(entity_path: str, value: float):
    """Log a scalar time-series value, version-safe."""
    try:
        rr.log(entity_path, rr.Scalars(value))
        return
    except AttributeError:
        pass
    try:
        rr.log(entity_path, rr.TimeSeriesScalar(value))
        return
    except AttributeError:
        pass
    try:
        rr.log(entity_path, rr.Scalar(value))
        return
    except AttributeError:
        pass
    # Fallback: skip (no scalar support in this version)


# ── Text logging ───────────────────────────────────────────────────────────────

def rr_text(entity_path: str, text: str):
    """Log a text document / HUD string, version-safe."""
    try:
        rr.log(entity_path, rr.TextDocument(text))
        return
    except AttributeError:
        pass
    try:
        rr.log(entity_path, rr.TextLog(text))
        return
    except AttributeError:
        pass


# ── 3D primitives — thin wrappers with consistent signature ────────────────────

def rr_points3d(entity_path: str, positions, colors=None, radii=None):
    kwargs = {}
    if colors  is not None: kwargs['colors']  = colors
    if radii   is not None: kwargs['radii']   = radii
    rr.log(entity_path, rr.Points3D(positions, **kwargs))


def rr_boxes3d(entity_path: str, half_sizes, colors=None, centers=None):
    kwargs = {}
    if colors  is not None: kwargs['colors']  = colors
    if centers is not None: kwargs['centers'] = centers
    rr.log(entity_path, rr.Boxes3D(half_sizes=half_sizes, **kwargs))


def rr_lines3d(entity_path: str, strips, colors=None):
    kwargs = {}
    if colors is not None: kwargs['colors'] = colors
    rr.log(entity_path, rr.LineStrips3D(strips, **kwargs))


def rr_arrows3d(entity_path: str, origins, vectors, colors=None):
    kwargs = {}
    if colors is not None: kwargs['colors'] = colors
    rr.log(entity_path, rr.Arrows3D(origins=origins, vectors=vectors, **kwargs))


def rr_transform3d(entity_path: str, translation=None, rotation_quat_xyzw=None,
                    rotation_axis=None, rotation_angle_rad=None):
    """Log a 3D transform, version-safe."""
    if rotation_quat_xyzw is not None:
        rot = rr.Quaternion(xyzw=rotation_quat_xyzw)
        rr.log(entity_path, rr.Transform3D(translation=translation, rotation=rot))
    elif rotation_axis is not None and rotation_angle_rad is not None:
        rot = rr.RotationAxisAngle(
            axis=rotation_axis,
            angle=rr.Angle(rad=rotation_angle_rad),
        )
        rr.log(entity_path, rr.Transform3D(
            translation=translation or [0, 0, 0], rotation=rot,
        ))
    else:
        rr.log(entity_path, rr.Transform3D(translation=translation or [0, 0, 0]))


def rr_clear(entity_path: str):
    try:
        rr.log(entity_path, rr.Clear(recursive=False))
    except Exception:
        pass


# ── Init helper ────────────────────────────────────────────────────────────────

def rr_init(app_id: str, spawn: bool = True):
    """Initialise Rerun, version-safe."""
    try:
        rr.init(app_id, spawn=spawn)
    except TypeError:
        # Older versions may not support spawn kwarg
        rr.init(app_id)
        if spawn:
            try:
                rr.spawn()
            except Exception:
                pass
