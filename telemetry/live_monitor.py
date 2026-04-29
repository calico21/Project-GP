"""
telemetry/live_monitor.py  (v2 — fixed)
"""
from __future__ import annotations
import math, threading, time, warnings
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

import matplotlib
for _b in ('TkAgg','Qt5Agg','QtAgg','WxAgg','MacOSX'):
    try: matplotlib.use(_b); break
    except Exception: continue

import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

try:
    import jax, jax.numpy as jnp; _JAX = True
except ImportError:
    _JAX = False


@dataclass
class _OptState:
    xk:        object = None
    iteration: int    = 0
    al_iter:   int    = 0
    loss_hist: list   = field(default_factory=list)
    grad_hist: list   = field(default_factory=list)
    viol_hist: list   = field(default_factory=list)
    dirty:     bool   = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class LiveOptMonitor:
    BG='#0A0E14'; PANEL='#0F1620'; FG='#E0E6ED'; MUTED='#6B7280'; GRID='#1A2332'
    SPEED_C='#00D4FF'; STEER_C='#FFB300'; FORCE_C='#22D67A'
    VIOL_C='#FF3344'; LOSS_C='#B266FF'; GRAD_C='#FF6F00'

    def __init__(self, solver, track_arrays, *, update_interval=2.0,
                 save_frames=False, mu=1.40):
        self._solver=solver; self._N=solver.N; self._dt=solver.dt_control
        self._mu=mu; self._interval=update_interval; self._save=save_frames
        self._frame=0
        (self._s_full, self._k_full, self._tx_full, self._ty_full,
         self._tpsi_full, self._wl_full, self._wr_full) = [
            np.asarray(a, dtype=np.float32) for a in track_arrays]
        self._k_n=self._x_n=self._y_n=self._psi_n=self._wl_n=self._wr_n=None
        self._x0_jax=self._sp_jax=None; self._ctx=False
        self._state=_OptState(); self._last_render=0.0
        self._fig=None; self._axes={}; self._artists={}

    def start(self):
        plt.ion(); self._build_figure(); plt.pause(0.1)
        print(f"[LiveMonitor] Window open — updates every {self._interval:.0f}s")

    def set_al_iter(self, al_iter):
        with self._state.lock:
            self._state.al_iter=al_iter; self._state.iteration=0

    def record_violation(self, viol):
        with self._state.lock: self._state.viol_hist.append(float(viol))

    def close(self):
        if self._fig: plt.ioff(); plt.close(self._fig)

    def __call__(self, xk):
        now = time.perf_counter()
        with self._state.lock:
            self._state.xk=xk.copy(); self._state.iteration+=1; self._state.dirty=True
        if now - self._last_render >= self._interval:
            self._last_render=now; self._render()

    def _set_solve_context(self, x0, sp, k_n, x_n, y_n, psi_n, wl_n, wr_n):
        self._k_n=np.asarray(k_n,np.float32); self._x_n=np.asarray(x_n,np.float32)
        self._y_n=np.asarray(y_n,np.float32); self._psi_n=np.asarray(psi_n,np.float32)
        self._wl_n=np.asarray(wl_n,np.float32); self._wr_n=np.asarray(wr_n,np.float32)
        if _JAX:
            self._x0_jax=jnp.array(x0,dtype=jnp.float32)
            self._sp_jax=jnp.array(sp,dtype=jnp.float32)
        self._ctx=True

    def _build_figure(self):
        plt.rcParams.update({
            'figure.facecolor':self.BG,'axes.facecolor':self.PANEL,
            'savefig.facecolor':self.BG,'axes.edgecolor':self.GRID,
            'axes.labelcolor':self.FG,'axes.titlecolor':self.FG,
            'xtick.color':self.FG,'ytick.color':self.FG,'text.color':self.FG,
            'grid.color':self.GRID,'grid.alpha':0.35,'grid.linewidth':0.4,
            'axes.grid':True,'font.size':8,'axes.titlesize':9,
            'axes.titleweight':'bold','legend.frameon':False,'legend.fontsize':7,
            'lines.linewidth':1.2,
        })
        self._fig = plt.figure(figsize=(16,9), facecolor=self.BG,
                               num='Project-GP — Diff-WMPC Live')
        try: self._fig.canvas.manager.set_window_title('Project-GP — Diff-WMPC Live')
        except: pass

        gs  = mgridspec.GridSpec(2,2,figure=self._fig,
                                  left=0.05,right=0.97,top=0.93,bottom=0.07,
                                  hspace=0.35,wspace=0.30,height_ratios=[2.2,1])
        gsr = mgridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=gs[0,1],hspace=0.40)

        ax_map   = self._fig.add_subplot(gs[0,0])
        ax_speed = self._fig.add_subplot(gsr[0])
        ax_ctrl  = self._fig.add_subplot(gsr[1], sharex=ax_speed)
        ax_gg    = self._fig.add_subplot(gsr[2])
        ax_conv  = self._fig.add_subplot(gs[1,:])
        ax_speed.tick_params(axis='x',labelbottom=False)

        for ax in (ax_map,ax_speed,ax_ctrl,ax_gg,ax_conv):
            ax.set_facecolor(self.PANEL)

        ax_map.set_aspect('equal',adjustable='datalim')
        ax_map.set_title('Racing Line  (colour = speed)'); ax_map.set_xlabel('X [m]'); ax_map.set_ylabel('Y [m]')
        ax_speed.set_title('Speed [km/h]'); ax_speed.set_ylabel('km/h')
        ax_ctrl.set_title('Controls  δ [°] / F [kN]'); ax_ctrl.set_xlabel('Step')
        ax_gg.set_title('G-G Envelope'); ax_gg.set_xlabel('Lat G'); ax_gg.set_ylabel('Lon G')
        ax_conv.set_title('Convergence'); ax_conv.set_xlabel('L-BFGS-B iteration')
        self._axes=dict(map=ax_map,speed=ax_speed,ctrl=ax_ctrl,gg=ax_gg,conv=ax_conv)

        # static track
        self._draw_static_track()

        # racing line collection
        lc=LineCollection([],cmap='plasma',norm=Normalize(0,25),lw=2.8,zorder=5,capstyle='round')
        ax_map.add_collection(lc)
        cb=self._fig.colorbar(lc,ax=ax_map,fraction=0.025,pad=0.02)
        cb.set_label('Speed [m/s]',color=self.FG); cb.ax.tick_params(colors=self.FG)
        cb.outline.set_edgecolor(self.GRID)
        car_dot,=ax_map.plot([],[],  'o',color=self.FG,ms=8,zorder=7,
                             markeredgecolor=self.BG,markeredgewidth=1.2)
        self._artists.update(lc=lc,car=car_dot)

        # friction ceiling
        # Use same first-N-steps of track that the MPC horizon will cover
        n=self._N
        ks=np.abs(self._k_full[:n])+1e-4
        vc=np.minimum(np.sqrt(self._mu*9.81/ks),self._solver.V_limit)*3.6
        sc=np.arange(n)
        ax_speed.fill_between(sc,vc,color=self.VIOL_C,alpha=0.08,zorder=0)
        ax_speed.plot(sc,vc,color=self.VIOL_C,lw=0.7,ls='--',alpha=0.55,label='μ limit',zorder=1)
        ax_speed.legend(loc='upper right')

        steps=np.arange(self._N)
        spd_l, =ax_speed.plot(steps,np.zeros(self._N),color=self.SPEED_C,lw=1.4,zorder=3)
        steer_l,=ax_ctrl.plot(steps,np.zeros(self._N),color=self.STEER_C,lw=1.0,label='δ [°]')
        force_l,=ax_ctrl.plot(steps,np.zeros(self._N),color=self.FORCE_C,lw=1.0,label='F [kN]')
        ax_ctrl.axhline(0,color=self.GRID,lw=0.4); ax_ctrl.legend(loc='upper right')
        self._artists.update(spd=spd_l,steer=steer_l,force=force_l)

        # G-G
        ax_gg.add_patch(Circle((0,0),self._mu,fill=False,edgecolor=self.VIOL_C,lw=1.5,ls='--',zorder=2))
        ax_gg.add_patch(Circle((0,0),self._mu*0.7,fill=False,edgecolor=self.MUTED,lw=0.6,ls=':',zorder=1))
        ax_gg.axhline(0,color=self.GRID,lw=0.4); ax_gg.axvline(0,color=self.GRID,lw=0.4)
        lim=self._mu*1.35; ax_gg.set_xlim(-lim,lim); ax_gg.set_ylim(-lim,lim)
        gg_sc=ax_gg.scatter([],[],s=3,c=[],cmap='plasma',vmin=0,vmax=25,edgecolors='none',zorder=3)
        self._artists['gg_sc']=gg_sc

        # convergence
        ax_conv2=ax_conv.twinx(); ax_conv2.set_facecolor(self.PANEL)
        cl,=ax_conv.plot([],[],color=self.LOSS_C,lw=1.2,label='Loss (norm.)')
        cg,=ax_conv.plot([],[],color=self.GRAD_C,lw=0.8,ls='--',label='‖∇‖ (norm.)')
        cv,=ax_conv2.plot([],[],color=self.VIOL_C,lw=1.1,label='Constraint viol.',zorder=5)
        ax_conv.set_ylabel('Loss / ‖∇‖  (norm.)'); ax_conv2.set_ylabel('Max violation',color=self.VIOL_C)
        ax_conv2.tick_params(axis='y',colors=self.VIOL_C)
        ax_conv.legend([cl,cg,cv],[l.get_label() for l in (cl,cg,cv)],loc='upper right')
        self._artists.update(conv_loss=cl,conv_grad=cg,conv_viol=cv,ax_conv2=ax_conv2)

        txt=self._fig.text(0.50,0.974,'Initialising…',ha='center',va='top',
                           color=self.FG,fontsize=10,fontweight='bold',
                           transform=self._fig.transFigure)
        self._artists['title']=txt
        self._fig.canvas.draw()

    def _draw_static_track(self):
        ax=self._axes['map']
        nx=-np.sin(self._tpsi_full); ny=np.cos(self._tpsi_full)
        xl=self._tx_full+self._wl_full*nx; yl=self._ty_full+self._wl_full*ny
        xr=self._tx_full-self._wr_full*nx; yr=self._ty_full-self._wr_full*ny
        ax.fill(np.concatenate([xl,xr[::-1]]),np.concatenate([yl,yr[::-1]]),
                color='#131A24',alpha=0.9,zorder=0)
        ax.plot(xl,yl,color='#2A3A4C',lw=0.8,zorder=1)
        ax.plot(xr,yr,color='#2A3A4C',lw=0.8,zorder=1)
        ax.plot(self._tx_full,self._ty_full,color=self.MUTED,lw=0.5,ls=':',alpha=0.4,zorder=2)
        ax.plot([xl[0],xr[0]],[yl[0],yr[0]],color=self.FG,lw=2.2,zorder=5,solid_capstyle='round')
        ax.annotate(' S/F',(self._tx_full[0],self._ty_full[0]),color=self.FG,fontsize=8,fontweight='bold',zorder=6)
        ax.autoscale_view()
        x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
        xm=(x1-x0)*0.05; ym=(y1-y0)*0.05
        ax.set_xlim(x0-xm,x1+xm); ax.set_ylim(y0-ym,y1+ym)

    def _render(self):
        if not _JAX or not self._ctx:
            self._flush(); return
        with self._state.lock:
            xk=self._state.xk; it=self._state.iteration; al_it=self._state.al_iter
            loss_h=list(self._state.loss_hist); grad_h=list(self._state.grad_hist)
            viol_h=list(self._state.viol_hist); self._state.dirty=False
        if xk is None: self._flush(); return

        # ── decode wavelet coefficients → time-domain controls ───────────────
        try:
            coeffs = jnp.clip(
                jnp.array(xk.reshape(self._N, 2), dtype=jnp.float32),
                -25000., 25000.)
            U_jax  = self._solver._db4_idwt(coeffs)
            steer  = np.array(U_jax[:, 0])
            force  = np.array(U_jax[:, 1])
        except Exception as e:
            warnings.warn(f"[LiveMonitor] IDWT failed: {e}"); self._flush(); return

        # ── REAL trajectory from solver ───────────────────────────────────────
        try:
            _, x_traj, _, _, _ = self._solver._simulate_trajectory(
                coeffs,                        # wavelet_coeffs (N,2) — already decoded above
                self._x0_jax,                  # x0
                self._sp_jax,                  # setup_params
                jnp.array(self._k_n),          # track_k
                jnp.array(self._x_n),          # track_x
                jnp.array(self._y_n),          # track_y
                jnp.array(self._psi_n),        # track_psi
                jnp.array(self._wl_n),         # track_w_left
                jnp.array(self._wr_n),         # track_w_right
                1.0,                           # lmuy_scale
                0.0,                           # wind_yaw
            )
            X = np.array(x_traj)               # (N, state_dim)

            xw = X[:, 0]    # STATE_X  = 0
            yw = X[:, 1]    # STATE_Y  = 1
            v  = X[:, 14]   # STATE_VX = 14

            dt      = self._dt
            a_lon   = np.gradient(v) / (dt * 9.81)
            psi     = X[:, 5]                  # STATE_YAW = 5  (was 2 — wrong)
            psi_dot = np.gradient(psi) / dt
            a_lat   = np.abs(psi_dot * v) / 9.81

        except Exception as e:
            warnings.warn(f"[LiveMonitor] real rollout failed: {e}")
            return
        steps=np.arange(self._N)
        pts=np.column_stack([xw,yw]); segs=np.stack([pts[:-1],pts[1:]],axis=1)
        self._artists['lc'].set_segments(segs); self._artists['lc'].set_array(0.5*(v[:-1]+v[1:]))
        self._artists['car'].set_data([xw[-1]],[yw[-1]])
        self._artists['spd'].set_data(steps,v*3.6)
        self._axes['speed'].relim(); self._axes['speed'].autoscale_view()
        self._artists['steer'].set_data(steps,np.degrees(steer))
        self._artists['force'].set_data(steps,force/1e3)
        self._axes['ctrl'].relim(); self._axes['ctrl'].autoscale_view()
        gg_pts = np.column_stack([a_lat, a_lon])
        self._artists['gg_sc'].set_offsets(gg_pts)
        self._artists['gg_sc'].set_array(v[:len(gg_pts)])
        # (a_lat, a_lon already computed by _kinematic_rollout)
        def _nm(lst):
            a=np.array(lst,dtype=np.float64); mx=np.nanmax(np.abs(a)) if a.size else 1.
            return a/(mx+1e-30)
        if len(loss_h)>1:
            ix=np.arange(len(loss_h)); self._artists['conv_loss'].set_data(ix,_nm(loss_h))
            self._axes['conv'].relim(); self._axes['conv'].autoscale_view()
        if len(grad_h)>1:
            self._artists['conv_grad'].set_data(np.arange(len(grad_h)),_nm(grad_h))
        if len(viol_h)>0:
            vx=np.linspace(0,max(len(loss_h),1),len(viol_h))
            self._artists['conv_viol'].set_data(vx,viol_h)
            self._artists['ax_conv2'].relim(); self._artists['ax_conv2'].autoscale_view()
        g_combined = np.hypot(a_lat, a_lon)
        fok=100.*float(np.mean(g_combined<=self._mu))
        self._artists['title'].set_text(
            f'AL {al_it}  ·  iter {it:>4d}  ·  '
            f'v_max={float(np.max(v))*3.6:.1f} km/h  '
            f'G_max={float(np.max(g_combined)):.2f}  '
            f'inside μ={fok:.0f}%  [kinematic display]')
        self._flush()
        if self._save:
            Path('figs').mkdir(exist_ok=True)
            self._fig.savefig(f'figs/frame_{self._frame:04d}.png',dpi=100,bbox_inches='tight')
            self._frame+=1

    def _flush(self):
        try:
            self._fig.canvas.draw_idle(); self._fig.canvas.flush_events(); plt.pause(0.001)
        except Exception: pass


def attach_monitor_to_solver(solver, track_arrays, **kw) -> LiveOptMonitor:
    monitor=LiveOptMonitor(solver,track_arrays,**kw)
    _orig=solver.solve

    def _wrap(*args,**kwargs):
        import scipy.optimize as _sco
        from models.vehicle_dynamics import (
            DifferentiableMultiBodyVehicle, build_default_setup_28,
            compute_equilibrium_suspension)

        _keys=('track_s','track_k','track_x','track_y','track_psi','track_w_left','track_w_right')
        _get=lambda i,k: (args[i] if i<len(args) else kwargs.get(k))
        raw_k=np.asarray(_get(1,'track_k'),np.float32)
        raw_x=np.asarray(_get(2,'track_x'),np.float32)
        raw_y=np.asarray(_get(3,'track_y'),np.float32)
        raw_psi=np.asarray(_get(4,'track_psi'),np.float32)
        raw_wl=np.asarray(_get(5,'track_w_left'),np.float32)
        raw_wr=np.asarray(_get(6,'track_w_right'),np.float32)

        N=solver.N; s0=np.linspace(0,1,len(raw_k))

        # ── Replicate solve()'s horizon-distance frac calculation ─────────────
        # solve() resamples to frac of the lap, not the full lap.
        # The monitor must use the same frac or _simulate_trajectory gets
        # waypoints ~23m apart instead of ~0.75m apart → immediate boundary crash.
        raw_s     = np.asarray(_get(0,'track_s'), np.float32)
        track_total_len = float(raw_s[-1]) if float(raw_s[-1]) > 1.0 \
                          else float(np.sum(np.sqrt(np.diff(raw_x)**2 +
                                                    np.diff(raw_y)**2)))
        k0_abs       = abs(float(np.mean(raw_k[:max(1, len(raw_k)//8)]))) + 1e-4
        v_est        = min(math.sqrt(solver.mu_friction * 9.81 / k0_abs),
                           solver.V_limit)
        horizon_dist = v_est * N * solver.dt_control
        frac         = min(horizon_dist / (track_total_len + 1e-6), 1.0)

        sw = np.linspace(0, frac, N)    # ← was np.linspace(0, 1, N)
        def _i(a): return np.interp(sw,s0,a).astype(np.float32)
        k_n=_i(raw_k); x_n=_i(raw_x); y_n=_i(raw_y)
        psi_n=_i(np.unwrap(raw_psi)); wl_n=_i(raw_wl); wr_n=_i(raw_wr)

        sp=kwargs.get('setup_params',None)
        sp=build_default_setup_28(solver.vp) if sp is None else np.asarray(sp,np.float32)
        k0=abs(float(k_n[0]))+1e-4; v0=min(math.sqrt(solver.mu_friction*9.81/k0),solver.V_limit)
        x0=DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0,vx0=v0)
        try:
            z=compute_equilibrium_suspension(sp,solver.vp)
            x0=x0.at[6].set(float(z[0])).at[7].set(float(z[1])).at[8].set(float(z[2])).at[9].set(float(z[3]))
        except Exception: pass

        # Give context BEFORE solve
        monitor._set_solve_context(x0,sp,k_n,x_n,y_n,psi_n,wl_n,wr_n)

        # Render warm-start immediately
        try:
            Uw=solver._build_physics_warmstart(
                jnp.array(k_n),jnp.array(psi_n),x0,jnp.array(sp))
            wc=solver._db4_dwt(Uw)
            with monitor._state.lock:
                monitor._state.xk=np.array(wc.flatten()); monitor._state.dirty=True
            monitor._last_render=0.; monitor._render()
            monitor._artists['title'].set_text('Warm-start  (optimisation starting…)')
            monitor._flush()
        except Exception as e:
            warnings.warn(f"[LiveMonitor] warm-start preview: {e}")

        # Patch scipy.optimize.minimize
        _om=_sco.minimize; al_ctr=[0]
        def _mon_min(fun,x0_opt,*a,**kw2):
            al_ctr[0]+=1; monitor.set_al_iter(al_ctr[0])
            with monitor._state.lock:
                monitor._state.loss_hist.clear(); monitor._state.grad_hist.clear()
            def _rec(xk_inner):
                val,grad=fun(xk_inner)
                with monitor._state.lock:
                    monitor._state.loss_hist.append(float(val) if np.isfinite(float(val)) else float('nan'))
                    monitor._state.grad_hist.append(float(np.linalg.norm(grad)) if np.all(np.isfinite(grad)) else float('nan'))
                    monitor._state.xk=xk_inner.copy(); monitor._state.dirty=True
                # Render on every function eval (throttled by wall-clock interval)
                now=time.perf_counter()
                if now - monitor._last_render >= monitor._interval:
                    monitor._last_render=now; monitor._render()
                return val,grad
            kw2['callback']=monitor
            return _om(_rec,x0_opt,*a,**kw2)
        _sco.minimize=_mon_min
        try:
            result=_orig(*args,**kwargs)
        finally:
            _sco.minimize=_om
            with monitor._state.lock: monitor._state.dirty=True
            monitor._last_render=0.; monitor._render()
        return result

    solver.solve=_wrap
    return monitor