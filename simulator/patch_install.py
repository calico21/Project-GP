#!/usr/bin/env python3
"""
simulator/patch_install.py
Run this ONCE from your project root to install the fixed simulator files:

    cd ~/FS_Driver_Setup_Optimizer
    python simulator/patch_install.py

What it does:
  1. Overwrites simulator/track_builder.py   — fixes arcsin NaN in chicane()
  2. Overwrites simulator/rerun_compat.py    — new: Rerun version shim
  3. Overwrites simulator/visualizer_client.py — fixes set_time_seconds API
"""

import os, sys, shutil, textwrap

ROOT = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
TRACK_BUILDER = r'''
import numpy as np
import json
import os
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field

try:
    from simulator.lap_timer import TimingLine, LapTimer
except ImportError:
    from lap_timer import TimingLine, LapTimer


@dataclass
class Cone:
    x: float; y: float; color: str


@dataclass
class Track:
    name: str
    cx: np.ndarray; cy: np.ndarray; cpsi: np.ndarray; ck: np.ndarray
    width_left: np.ndarray; width_right: np.ndarray
    total_length: float
    cones_left:  List[Cone] = field(default_factory=list)
    cones_right: List[Cone] = field(default_factory=list)
    cones_orange:List[Cone] = field(default_factory=list)
    finish_line: Optional[TimingLine] = None
    sector_lines:List[TimingLine] = field(default_factory=list)
    start_x: float = 0.0; start_y: float = 0.0; start_yaw: float = 0.0

    def get_start_pose(self): return self.start_x, self.start_y, self.start_yaw

    def get_closest_point(self, x, y):
        dx = self.cx - x; dy = self.cy - y
        dists = np.sqrt(dx*dx + dy*dy)
        idx = int(np.argmin(dists))
        return idx, float(dists[idx])

    def get_local_curvature(self, x, y):
        idx, _ = self.get_closest_point(x, y)
        return float(self.ck[idx])

    def to_json(self, path):
        d = {'name':self.name,
             'cx':self.cx.tolist(),'cy':self.cy.tolist(),
             'cpsi':self.cpsi.tolist(),'ck':self.ck.tolist(),
             'width_left':self.width_left.tolist(),'width_right':self.width_right.tolist(),
             'total_length':self.total_length,
             'start_x':self.start_x,'start_y':self.start_y,'start_yaw':self.start_yaw,
             'cones_left': [{'x':c.x,'y':c.y,'color':c.color} for c in self.cones_left],
             'cones_right':[{'x':c.x,'y':c.y,'color':c.color} for c in self.cones_right],
             'cones_orange':[{'x':c.x,'y':c.y,'color':c.color} for c in self.cones_orange]}
        with open(path,'w') as f: json.dump(d, f, indent=2)
        print(f"[TrackBuilder] Saved -> {path}")

    @classmethod
    def from_json(cls, path):
        with open(path) as f: d = json.load(f)
        return cls(name=d['name'],
                   cx=np.array(d['cx']),cy=np.array(d['cy']),
                   cpsi=np.array(d['cpsi']),ck=np.array(d['ck']),
                   width_left=np.array(d['width_left']),
                   width_right=np.array(d['width_right']),
                   total_length=d['total_length'],
                   start_x=d.get('start_x',0.0),start_y=d.get('start_y',0.0),
                   start_yaw=d.get('start_yaw',0.0),
                   cones_left =[Cone(**c) for c in d.get('cones_left', [])],
                   cones_right=[Cone(**c) for c in d.get('cones_right',[])],
                   cones_orange=[Cone(**c) for c in d.get('cones_orange',[])])


class TrackBuilder:
    CONE_SPACING_STRAIGHT = 5.0
    CONE_SPACING_CORNER   = 3.0
    DS = 0.5

    def __init__(self, name='custom', default_half_width=3.5,
                 start_x=0.0, start_y=0.0, start_yaw=0.0):
        self.name = name; self.default_hw = default_half_width
        self.start_x = start_x; self.start_y = start_y; self.start_yaw = start_yaw
        self._x = start_x; self._y = start_y; self._psi = start_yaw
        self._pts_x=[start_x]; self._pts_y=[start_y]; self._pts_psi=[start_yaw]
        self._pts_k=[0.0]; self._pts_wl=[default_half_width]; self._pts_wr=[default_half_width]
        self._sector_arclens=[]; self._current_arclen=0.0

    def straight(self, length, half_width=None):
        hw = half_width or self.default_hw
        n  = max(2, int(length / self.DS))
        for _ in range(n):
            self._x += self.DS*np.cos(self._psi); self._y += self.DS*np.sin(self._psi)
            self._pts_x.append(self._x); self._pts_y.append(self._y)
            self._pts_psi.append(self._psi); self._pts_k.append(0.0)
            self._pts_wl.append(hw); self._pts_wr.append(hw)
            self._current_arclen += self.DS
        return self

    def arc(self, radius, angle_deg, direction='L', half_width=None):
        hw   = half_width or self.default_hw
        sign = 1.0 if direction.upper()=='L' else -1.0
        k    = sign / max(radius, 0.5)
        arc_len = abs(np.deg2rad(angle_deg)) * radius
        n    = max(2, int(arc_len / self.DS))
        dpsi = (np.deg2rad(angle_deg)*sign) / n
        for _ in range(n):
            self._psi += dpsi
            self._x += self.DS*np.cos(self._psi - dpsi/2)
            self._y += self.DS*np.sin(self._psi - dpsi/2)
            self._pts_x.append(self._x); self._pts_y.append(self._y)
            self._pts_psi.append(self._psi); self._pts_k.append(k)
            self._pts_wl.append(hw); self._pts_wr.append(hw)
            self._current_arclen += self.DS
        return self

    def chicane(self, offset=3.0, length=20.0, half_width=None):
        """
        FIX: correct chord-sagitta radius formula + arcsin argument clamped to [0,1].
        r = (half_chord² + sagitta²) / (2 * sagitta)
        sin_a = half_chord / r  — guaranteed ≤ 1 by construction of r.
        """
        hw     = half_width or self.default_hw
        half_L = length / 2.0
        r      = (half_L**2 + offset**2) / (2.0 * max(offset, 0.1))
        r      = max(r, half_L * 1.01)          # guarantee r > half-chord
        sin_a  = min(half_L / r, 1.0)           # clamp: arcsin safe
        a      = np.degrees(np.arcsin(sin_a))
        self.arc(r, a, 'L', hw).arc(r, a*2, 'R', hw).arc(r, a, 'L', hw)
        return self

    def add_sector_split(self):
        self._sector_arclens.append(self._current_arclen); return self

    def build(self):
        from scipy.ndimage import gaussian_filter1d
        cx=np.array(self._pts_x); cy=np.array(self._pts_y)
        cpsi=np.array(self._pts_psi); ck=np.array(self._pts_k)
        wl=np.array(self._pts_wl); wr=np.array(self._pts_wr)
        total_length = float(len(cx))*self.DS
        ck = gaussian_filter1d(ck, sigma=3.0)

        cones_left=[]; cones_right=[]; cones_orange=[]; last_s=-99.0
        for i in range(len(cx)):
            spacing = self.CONE_SPACING_CORNER if abs(ck[i])>0.03 else self.CONE_SPACING_STRAIGHT
            s = i*self.DS
            if s - last_s >= spacing:
                pn = cpsi[i]+np.pi/2
                cones_left.append(Cone(cx[i]+wl[i]*np.cos(pn), cy[i]+wl[i]*np.sin(pn),'yellow'))
                cones_right.append(Cone(cx[i]-wr[i]*np.cos(pn), cy[i]-wr[i]*np.sin(pn),'blue'))
                last_s = s

        pn_sf = cpsi[0]+np.pi/2
        fl = TimingLine(x0=cx[0]+wl[0]*np.cos(pn_sf), y0=cy[0]+wl[0]*np.sin(pn_sf),
                        x1=cx[0]-wr[0]*np.cos(pn_sf), y1=cy[0]-wr[0]*np.sin(pn_sf), label="S/F")
        cones_orange += [Cone(fl.x0,fl.y0,'big_orange'), Cone(fl.x1,fl.y1,'big_orange')]

        sector_lines=[]
        for arc_s in self._sector_arclens:
            i = min(int(arc_s/self.DS), len(cx)-1)
            pn = cpsi[i]+np.pi/2
            sl = TimingLine(x0=cx[i]+wl[i]*np.cos(pn), y0=cy[i]+wl[i]*np.sin(pn),
                            x1=cx[i]-wr[i]*np.cos(pn), y1=cy[i]-wr[i]*np.sin(pn),
                            label=f"S{len(sector_lines)+1}")
            sector_lines.append(sl)
            cones_orange += [Cone(sl.x0,sl.y0,'big_orange'), Cone(sl.x1,sl.y1,'big_orange')]

        return Track(name=self.name, cx=cx, cy=cy, cpsi=cpsi, ck=ck,
                     width_left=wl, width_right=wr, total_length=total_length,
                     cones_left=cones_left, cones_right=cones_right, cones_orange=cones_orange,
                     finish_line=fl, sector_lines=sector_lines,
                     start_x=self.start_x, start_y=self.start_y, start_yaw=self.start_yaw)


def build_skidpad():
    R=9.125
    b=TrackBuilder("Skidpad",default_half_width=1.5,start_x=0.0,start_y=0.0,start_yaw=np.pi/2)
    b.straight(15.0); b.arc(R,360,'L',1.5); b.arc(R,360,'L',1.5)
    b.straight(2.0);  b.arc(R,360,'R',1.5); b.arc(R,360,'R',1.5); b.straight(15.0)
    return b.build()

def build_fsg_autocross():
    b=TrackBuilder("FSG Autocross",default_half_width=3.5,start_x=0.0,start_y=0.0,start_yaw=0.0)
    b.straight(30); b.arc(9,180,'L'); b.straight(20); b.arc(18,90,'R')
    b.straight(12); b.arc(9,90,'L'); b.straight(10); b.arc(8,90,'R'); b.straight(10)
    b.add_sector_split()
    b.chicane(3.0,18.0); b.straight(15); b.chicane(3.5,20.0); b.straight(12)
    b.add_sector_split()
    b.arc(25,120,'L',4.0); b.straight(40); b.arc(9,90,'R'); b.straight(10)
    b.arc(9,90,'L'); b.straight(25)
    return b.build()

def build_endurance_lap():
    b=TrackBuilder("Endurance Lap",default_half_width=4.0,start_x=0.0,start_y=0.0,start_yaw=0.0)
    b.straight(80); b.arc(18,90,'L'); b.straight(12); b.arc(18,90,'R'); b.straight(70)
    b.arc(10,90,'L'); b.straight(10); b.arc(10,90,'R'); b.add_sector_split()
    b.straight(80); b.arc(12,90,'L'); b.straight(10); b.arc(12,90,'L'); b.straight(80)
    b.arc(8,180,'R'); b.add_sector_split()
    b.straight(80); b.arc(16,90,'L'); b.straight(14); b.arc(16,90,'R'); b.straight(80)
    return b.build()

def build_acceleration():
    b=TrackBuilder("Acceleration 75m",default_half_width=5.0,start_x=0.0,start_y=0.0,start_yaw=0.0)
    b.straight(80); return b.build()

TRACK_REGISTRY={'skidpad':build_skidpad,'fsg_autocross':build_fsg_autocross,
                'endurance_lap':build_endurance_lap,'acceleration':build_acceleration}

def get_track(name):
    if name in TRACK_REGISTRY:
        t = TRACK_REGISTRY[name]()
        print(f"[TrackBuilder] Built '{name}': {t.total_length:.0f}m | "
              f"{len(t.cones_left)+len(t.cones_right)} cones | {len(t.sector_lines)} sectors")
        return t
    elif os.path.isfile(name):
        return Track.from_json(name)
    raise ValueError(f"Unknown track '{name}'. Options: {list(TRACK_REGISTRY.keys())}")
'''.strip()


# ─────────────────────────────────────────────────────────────────────────────
RERUN_COMPAT = r'''
"""
simulator/rerun_compat.py  —  Rerun SDK version compatibility shim.
Handles API differences between rerun 0.9 and 0.20+.
"""
import rerun as rr
import numpy as np
import math

try:
    _VER = tuple(int(x) for x in rr.__version__.split('.')[:2])
except Exception:
    _VER = (0, 0)

def rr_set_time(sim_time: float, frame_id: int = 0):
    try:    rr.set_time_nanos("sim_time", int(sim_time * 1e9))
    except AttributeError:
        try: rr.set_time_seconds("sim_time", sim_time)
        except AttributeError: pass
    try:    rr.set_time_sequence("frame", frame_id)
    except AttributeError: pass

def rr_scalar(path: str, value: float):
    for cls_name in ('Scalars', 'TimeSeriesScalar', 'Scalar'):
        cls = getattr(rr, cls_name, None)
        if cls is not None:
            try: rr.log(path, cls(value)); return
            except Exception: pass

def rr_text(path: str, text: str):
    for cls_name in ('TextDocument', 'TextLog'):
        cls = getattr(rr, cls_name, None)
        if cls is not None:
            try: rr.log(path, cls(text)); return
            except Exception: pass

def rr_init(app_id: str, spawn: bool = True):
    try:    rr.init(app_id, spawn=spawn)
    except TypeError:
        rr.init(app_id)
        if spawn:
            try: rr.spawn()
            except Exception: pass

def rr_points3d(path, positions, colors=None, radii=None):
    kw = {}
    if colors is not None: kw['colors'] = colors
    if radii  is not None: kw['radii']  = radii
    rr.log(path, rr.Points3D(positions, **kw))

def rr_boxes3d(path, half_sizes, colors=None, centers=None):
    kw = {}
    if colors  is not None: kw['colors']  = colors
    if centers is not None: kw['centers'] = centers
    rr.log(path, rr.Boxes3D(half_sizes=half_sizes, **kw))

def rr_lines3d(path, strips, colors=None):
    kw = {}
    if colors is not None: kw['colors'] = colors
    rr.log(path, rr.LineStrips3D(strips, **kw))

def rr_arrows3d(path, origins, vectors, colors=None):
    kw = {}
    if colors is not None: kw['colors'] = colors
    rr.log(path, rr.Arrows3D(origins=origins, vectors=vectors, **kw))

def rr_transform3d(path, translation=None, rotation_quat_xyzw=None,
                    rotation_axis=None, rotation_angle_rad=None):
    if rotation_quat_xyzw is not None:
        rot = rr.Quaternion(xyzw=rotation_quat_xyzw)
        rr.log(path, rr.Transform3D(translation=translation, rotation=rot))
    elif rotation_axis is not None and rotation_angle_rad is not None:
        rot = rr.RotationAxisAngle(axis=rotation_axis,
                                   angle=rr.Angle(rad=rotation_angle_rad))
        rr.log(path, rr.Transform3D(translation=translation or [0,0,0], rotation=rot))
    else:
        rr.log(path, rr.Transform3D(translation=translation or [0,0,0]))

def rr_clear(path: str):
    try: rr.log(path, rr.Clear(recursive=False))
    except Exception: pass
'''.strip()


# ─────────────────────────────────────────────────────────────────────────────
VISUALIZER = r'''
"""simulator/visualizer_client.py  v2.1 — fixed Rerun API + track NaN"""
import os, sys, time, socket, math
from collections import deque
import numpy as np

try:
    import rerun as rr
except ImportError:
    print("ERROR: pip install rerun-sdk"); sys.exit(1)

_D = os.path.dirname(os.path.abspath(__file__))
if _D not in sys.path: sys.path.insert(0, _D)
from rerun_compat import (rr_set_time, rr_scalar, rr_text, rr_init,
                           rr_points3d, rr_boxes3d, rr_lines3d,
                           rr_arrows3d, rr_transform3d, rr_clear)
from scipy.spatial.transform import Rotation as SciRot

_ROOT = os.path.dirname(_D)
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
try:
    from simulator.sim_protocol import TelemetryFrame, TX_BYTES, DEFAULT_HOST, DEFAULT_PORT_SEND
    from simulator.track_builder import get_track
except ImportError:
    from sim_protocol import TelemetryFrame, TX_BYTES, DEFAULT_HOST, DEFAULT_PORT_SEND
    from track_builder import get_track

VISUALIZER_HZ=60; TRAIL_LENGTH=500; GG_HISTORY=2000
FORCE_ARROW_SCALE=0.0008; FZ_BAR_SCALE=0.0004
TIRE_TEMP_COLD=40.0; TIRE_TEMP_OPTIMAL=80.0; TIRE_TEMP_HOT=110.0
LF=0.68; LR=0.92; TRACK_W=1.20; R_TIRE=0.2032; W_TIRE=0.10

def _lerp(t,c0,c1):
    return [max(0,min(255,int(c0[i]+(c1[i]-c0[i])*float(t)))) for i in range(3)]+[255]

def tire_temp_color(T):
    T=float(T)
    if T<TIRE_TEMP_COLD: return [50,50,220,255]
    if T<TIRE_TEMP_OPTIMAL: return _lerp((T-TIRE_TEMP_COLD)/(TIRE_TEMP_OPTIMAL-TIRE_TEMP_COLD),(50,50,220),(50,220,50))
    if T<TIRE_TEMP_HOT:     return _lerp((T-TIRE_TEMP_OPTIMAL)/(TIRE_TEMP_HOT-TIRE_TEMP_OPTIMAL),(50,220,50),(220,50,50))
    return [220,50,50,255]

def grip_color(u):
    u=float(u)
    if u<0.70: return _lerp(u/0.70,(30,200,30),(200,200,30))
    return _lerp(min((u-0.70)/0.30,1.0),(200,200,30),(220,30,30))

def speed_color(s,mx=100.0):
    t=min(float(s)/max(float(mx),1.0),1.0)
    if t<0.5: return _lerp(t*2,(30,50,220),(30,220,50))
    return _lerp((t-0.5)*2,(30,220,50),(220,30,30))

class SimVisualizer:
    def __init__(self,host=DEFAULT_HOST,port=DEFAULT_PORT_SEND,
                 track_name='fsg_autocross',app_id='ProjectGP_DigitalTwin'):
        self.host=host; self.port=port; self.app_id=app_id
        self._trail_x=deque(maxlen=TRAIL_LENGTH); self._trail_y=deque(maxlen=TRAIL_LENGTH)
        self._trail_spd=deque(maxlen=TRAIL_LENGTH)
        self._gg_lat=deque(maxlen=GG_HISTORY); self._gg_lon=deque(maxlen=GG_HISTORY)
        self._prev_fid=-1; self._gg_init=False
        self.track=None
        try:
            self.track=get_track(track_name)
            print(f"[Viz] Track: {self.track.name}  {self.track.total_length:.0f}m")
        except Exception as e: print(f"[Viz] No track ({e})")

    def _safe(self,fn,*a,**kw):
        try: fn(*a,**kw)
        except Exception: pass

    def _init_rerun(self):
        rr_init(self.app_id,spawn=True)
        rr_set_time(0.0,0)
        print(f"[Viz] Rerun {rr.__version__} ready.")
        self._safe(rr_boxes3d,"world/ground",half_sizes=[[500,500,0.02]],colors=[[25,25,25,255]])
        if self.track: self._log_track_static()

    def _log_track_static(self):
        cx,cy=self.track.cx,self.track.cy
        pts=[[float(cx[i]),float(cy[i]),0.01] for i in range(0,len(cx),3)]
        if pts: self._safe(rr_points3d,"world/track/centreline",pts,colors=[[70,70,70,160]]*len(pts),radii=0.05)
        for path,cones,col in [
            ("world/track/cones_left",  self.track.cones_left,  [255,220,0,255]),
            ("world/track/cones_right", self.track.cones_right, [0,50,255,255]),
            ("world/track/cones_orange",self.track.cones_orange,[255,120,0,255])]:
            cp=[[c.x,c.y,R_TIRE] for c in cones]
            if cp: self._safe(rr_points3d,path,cp,colors=[col]*len(cp),radii=0.13)
        for sl in self.track.sector_lines:
            self._safe(rr_lines3d,f"world/track/sector_{sl.label}",
                       [[[sl.x0,sl.y0,0.05],[sl.x1,sl.y1,0.05]]],colors=[[200,200,0,200]])

    def _log_frame(self,tf):
        rr_set_time(float(tf.sim_time),int(tf.frame_id))
        q=SciRot.from_euler('zyx',[float(tf.yaw),float(tf.pitch),float(tf.roll)]).as_quat()
        self._safe(rr_transform3d,"world/vehicle",
                   translation=[float(tf.x),float(tf.y),float(tf.z)],
                   rotation_quat_xyzw=list(q))
        self._safe(rr_boxes3d,"world/vehicle/chassis",half_sizes=[[1.10,0.40,0.20]],colors=[[200,40,40,255]])

        for name,(lx,ly,lz,T,slip,Fz,Fy,sa) in {
            'fl':(LF,  TRACK_W/2, tf.z_fl,tf.T_fl,tf.slip_fl,tf.Fz_fl,tf.Fy_fl,tf.delta),
            'fr':(LF, -TRACK_W/2, tf.z_fr,tf.T_fr,tf.slip_fr,tf.Fz_fr,tf.Fy_fr,tf.delta),
            'rl':(-LR, TRACK_W/2, tf.z_rl,tf.T_rl,tf.slip_rl,tf.Fz_rl,tf.Fy_rl,0.0),
            'rr':(-LR,-TRACK_W/2, tf.z_rr,tf.T_rr,tf.slip_rr,tf.Fz_rr,tf.Fy_rr,0.0)}.items():
            b=f"world/vehicle/{name}"
            self._safe(rr_transform3d,b,translation=[float(lx),float(ly),float(lz)],
                       rotation_axis=[0,0,1],rotation_angle_rad=float(sa))
            self._safe(rr_boxes3d,f"{b}/tire",half_sizes=[[R_TIRE,W_TIRE,R_TIRE]],
                       colors=[tire_temp_color(float(T))])
            util=float(tf.grip_util_f if 'f' in name else tf.grip_util_r)
            self._safe(rr_points3d,f"{b}/grip",[[0,0,R_TIRE+0.06]],
                       colors=[grip_color(util)],radii=max(0.05,util*0.13))
            al=float(Fy)*FORCE_ARROW_SCALE
            if abs(al)>0.03:
                self._safe(rr_arrows3d,f"{b}/Fy",origins=[[0,0,0]],vectors=[[0,al,0]],colors=[grip_color(util)])
            self._safe(rr_lines3d,f"{b}/Fz",[[[0,0,0],[0,0,float(Fz)*FZ_BAR_SCALE]]],colors=[[100,150,250,200]])
            if abs(float(slip))>0.01:
                arc=[[R_TIRE*math.cos(a),R_TIRE*math.sin(a),0] for a in np.linspace(0,float(slip),8)]
                self._safe(rr_lines3d,f"{b}/slip",[arc],colors=[[255,200,0,200] if abs(float(slip))>0.1 else [200,200,0,140]])

        self._trail_x.append(float(tf.x)); self._trail_y.append(float(tf.y)); self._trail_spd.append(float(tf.speed_kmh))
        if len(self._trail_x)>2:
            tp=[[float(x),float(y),0.02] for x,y in zip(self._trail_x,self._trail_y)]
            tc=[speed_color(float(s)) for s in self._trail_spd]
            self._safe(rr_points3d,"world/track/trail",tp,colors=tc,radii=0.06)

        self._gg_lat.append(float(tf.lat_g)); self._gg_lon.append(float(tf.lon_g))
        if len(self._gg_lat)>1:
            gp=[[float(la),float(lo),0] for la,lo in zip(self._gg_lat,self._gg_lon)]
            gc=[grip_color(math.sqrt(la**2+lo**2)/1.4) for la,lo in zip(self._gg_lat,self._gg_lon)]
            self._safe(rr_points3d,"gg_diagram/trace",gp,colors=gc,radii=0.04)
            self._safe(rr_points3d,"gg_diagram/current",[[float(tf.lat_g),float(tf.lon_g),0]],colors=[[255,255,255,255]],radii=0.10)
            if not self._gg_init:
                self._safe(rr_lines3d,"gg_diagram/boundary",
                           [[[1.45*math.cos(a),1.45*math.sin(a),0] for a in np.linspace(0,2*math.pi,64)]],
                           colors=[[100,100,100,180]])
                self._gg_init=True

        for path,val in {
            "suspension/fl":float(tf.z_fl)-0.30,"suspension/fr":float(tf.z_fr)-0.30,
            "suspension/rl":float(tf.z_rl)-0.30,"suspension/rr":float(tf.z_rr)-0.30,
            "tires/T_fl":float(tf.T_fl),"tires/T_fr":float(tf.T_fr),
            "tires/T_rl":float(tf.T_rl),"tires/T_rr":float(tf.T_rr),
            "loads/Fz_fl":float(tf.Fz_fl),"loads/Fz_fr":float(tf.Fz_fr),
            "loads/Fz_rl":float(tf.Fz_rl),"loads/Fz_rr":float(tf.Fz_rr),
            "slip/alpha_fl":math.degrees(float(tf.slip_fl)),"slip/alpha_rl":math.degrees(float(tf.slip_rl)),
            "slip/kappa_rl":float(tf.kappa_rl),"slip/kappa_rr":float(tf.kappa_rr),
            "aero/downforce":float(tf.downforce),"aero/drag":float(tf.drag),
            "perf/speed_kmh":float(tf.speed_kmh),"perf/lat_g":float(tf.lat_g),
            "perf/lon_g":float(tf.lon_g),"perf/yaw_rate":float(tf.yaw_rate_deg),
            "perf/grip_util_f":float(tf.grip_util_f),"perf/grip_util_r":float(tf.grip_util_r),
            "perf/energy_kj":float(tf.energy_kj),
            "timing/lap_time":float(tf.lap_time),"timing/lap_number":float(tf.lap_number)}.items():
            self._safe(rr_scalar,path,val)

        def ft(t):
            t=max(0.0,float(t)); return f"{int(t//60):02d}:{t%60:06.3f}"
        hud="\n".join([
            f"SPD  {tf.speed_kmh:5.1f} km/h",
            f"LAT  {tf.lat_g:+.3f}G   LON  {tf.lon_g:+.3f}G",
            f"YAW  {tf.yaw_rate_deg:+.1f} deg/s","─────────────────────",
            f"LAP {tf.lap_number+1}  SEC {tf.sector+1}  {ft(tf.lap_time)}","─────────────────────",
            f"Fz FL:{tf.Fz_fl:5.0f} FR:{tf.Fz_fr:5.0f} N",
            f"   RL:{tf.Fz_rl:5.0f} RR:{tf.Fz_rr:5.0f} N",
            f"T° FL:{tf.T_fl:.0f} FR:{tf.T_fr:.0f} RL:{tf.T_rl:.0f} RR:{tf.T_rr:.0f}",
            f"DF {tf.downforce:.0f}N  Drag {tf.drag:.0f}N  E {tf.energy_kj:.1f}kJ"])
        self._safe(rr_text,"hud/telemetry",hud)
        self._prev_fid=int(tf.frame_id)

    def run(self):
        self._init_rerun()
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0',self.port)); sock.settimeout(0.016)
        print(f"[Viz] Listening on :{self.port}  — waiting for server…")
        frames_rx=0; last_t=time.perf_counter()
        try:
            while True:
                t0=time.perf_counter()
                try:
                    data,_=sock.recvfrom(TX_BYTES+16)
                    tf=TelemetryFrame.from_bytes(data)
                    if tf:
                        if self._prev_fid>=0 and tf.frame_id-self._prev_fid>5:
                            print(f"\r[Viz] ~{tf.frame_id-self._prev_fid} drops",end='')
                        self._log_frame(tf); frames_rx+=1
                except socket.timeout: pass
                except Exception: pass
                now=time.perf_counter()
                if now-last_t>2.0 and frames_rx>0:
                    print(f"\r[Viz] {frames_rx/(now-last_t+1e-9):.0f}Hz | frame {self._prev_fid:6d}",end='',flush=True)
                    frames_rx=0; last_t=now
                st=(1.0/VISUALIZER_HZ)-(time.perf_counter()-t0)
                if st>0: time.sleep(st)
        except KeyboardInterrupt: print("\n[Viz] Done.")
        finally: sock.close()

def main():
    import argparse, socket as _s
    p=argparse.ArgumentParser(); p.add_argument('--track',default='fsg_autocross')
    p.add_argument('--host',default=DEFAULT_HOST); p.add_argument('--port',type=int,default=DEFAULT_PORT_SEND)
    a=p.parse_args()
    SimVisualizer(host=a.host,port=a.port,track_name=a.track).run()

if __name__=='__main__': main()
'''.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Write files
# ─────────────────────────────────────────────────────────────────────────────

FILES = {
    'track_builder.py'    : TRACK_BUILDER,
    'rerun_compat.py'     : RERUN_COMPAT,
    'visualizer_client.py': VISUALIZER,
}

print("=" * 55)
print("  Project-GP Simulator — patch installer")
print("=" * 55)

for fname, content in FILES.items():
    dst = os.path.join(ROOT, fname)
    # Backup old file
    if os.path.exists(dst):
        bak = dst + '.bak'
        shutil.copy2(dst, bak)
        print(f"  Backup: {fname} -> {fname}.bak")
    with open(dst, 'w') as f:
        f.write(content + '\n')
    print(f"  Written: {dst}")

print()
print("Verifying track build (no warnings expected)…")
import warnings
warnings.filterwarnings('error')
sys.path.insert(0, ROOT)
try:
    from track_builder import get_track
    for nm in ('fsg_autocross', 'skidpad', 'endurance_lap'):
        t = get_track(nm)
        print(f"  {nm}: {t.total_length:.0f}m  OK")
    print()
    print("All patches applied successfully.")
    print()
    print("Run the visualizer:")
    print("  python simulator/visualizer_client.py --track fsg_autocross")
except Exception as e:
    print(f"  ERROR during verification: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)
