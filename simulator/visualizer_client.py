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

        # Wheel local Z = corner_world_z - cg_world_z = (z_fl + tire_r) - (Z + tire_r) = z_fl - Z
        # This is the body-frame vertical offset from CG to each corner (≈0 when flat)
        _cg_z = float(tf.z)
        for name,(lx,ly,lz,T,slip,Fz,Fy,sa) in {
            'fl':(LF,  TRACK_W/2, float(tf.z_fl)-_cg_z, tf.T_fl,tf.slip_fl,tf.Fz_fl,tf.Fy_fl,tf.delta),
            'fr':(LF, -TRACK_W/2, float(tf.z_fr)-_cg_z, tf.T_fr,tf.slip_fr,tf.Fz_fr,tf.Fy_fr,tf.delta),
            'rl':(-LR, TRACK_W/2, float(tf.z_rl)-_cg_z, tf.T_rl,tf.slip_rl,tf.Fz_rl,tf.Fy_rl,0.0),
            'rr':(-LR,-TRACK_W/2, float(tf.z_rr)-_cg_z, tf.T_rr,tf.slip_rr,tf.Fz_rr,tf.Fy_rr,0.0)}.items():
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
            # Suspension travel = corner height relative to CG (body-frame vertical displacement)
            "suspension/fl": float(tf.z_fl)-float(tf.z),
            "suspension/fr": float(tf.z_fr)-float(tf.z),
            "suspension/rl": float(tf.z_rl)-float(tf.z),
            "suspension/rr": float(tf.z_rr)-float(tf.z),
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