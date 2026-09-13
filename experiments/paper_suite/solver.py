"""Independent experimental solvers for the production 216-variable GLRK residual."""
from __future__ import annotations
import time
import jax
import jax.numpy as jnp
from experiments.full_vehicle_gradient_audit import N_STAGE, stage_residual, production_stages

def initial_stages(vehicle, x, controls, setup):
    dx = vehicle._compute_derivatives(x, controls, setup)
    return jnp.concatenate((dx, dx))

def _info(vehicle, x, controls, setup, dt, z, history, converged, elapsed):
    f = stage_residual(vehicle,x,controls,setup,dt,z)
    k1,k2=z.reshape(2,-1); x1=x+dt*(.25*k1+(.25-jnp.sqrt(3)/6)*k2); x2=x+dt*((.25+jnp.sqrt(3)/6)*k1+.25*k2)
    raw1=vehicle._compute_derivatives(x1,controls,setup); raw2=vehicle._compute_derivatives(x2,controls,setup)
    return {"stages": z, "iterations": len(history), "converged": converged, "initial_residual_inf": history[0]["residual_inf"],
            "final_residual_inf": float(jnp.max(jnp.abs(f))), "final_residual_l2": float(jnp.linalg.norm(f)), "residual_history":history,
            "wall_time_s":time.perf_counter()-elapsed, "branch_signature":{"stage_rate_clip_counts":[int(jnp.sum(jnp.abs(raw1)>=500)),int(jnp.sum(jnp.abs(raw2)>=500))]}}

def solve_stages_picard(vehicle,x,controls,setup,dt,tol=1e-10,max_iterations=128,iterations=None):
    z=initial_stages(vehicle,x,controls,setup); hist=[]; start=time.perf_counter(); limit=iterations or max_iterations
    ok=False
    for i in range(limit):
        f=stage_residual(vehicle,x,controls,setup,dt,z); ri=float(jnp.max(jnp.abs(f))); hist.append({"iteration":i,"residual_inf":ri,"residual_l2":float(jnp.linalg.norm(f))})
        if ri < tol: ok=True; break
        z=z-f
    return _info(vehicle,x,controls,setup,dt,z,hist,ok,start)

def solve_stages_newton(vehicle,x,controls,setup,dt,tol=1e-10,max_iterations=15,iterations=None):
    z=initial_stages(vehicle,x,controls,setup); hist=[]; start=time.perf_counter(); limit=iterations or max_iterations; ok=False
    residual=lambda zz: stage_residual(vehicle,x,controls,setup,dt,zz)
    for i in range(limit):
        f=residual(z); ri=float(jnp.max(jnp.abs(f))); hist.append({"iteration":i,"residual_inf":ri,"residual_l2":float(jnp.linalg.norm(f))})
        if ri < tol: ok=True; break
        jac=jax.jacrev(residual)(z)
        try: delta=jnp.linalg.solve(jac, f)
        except Exception: break
        # Backtracking is part of this reference solver only; it guards an invalid Newton step.
        candidate=z-delta
        if not bool(jnp.all(jnp.isfinite(candidate))): break
        z=candidate
    return _info(vehicle,x,controls,setup,dt,z,hist,ok,start)

def local_picard_jacobian(vehicle,x,controls,setup,dt,z):
    # G=z-F, exactly the production fixed-point map.
    g=lambda zz: zz-stage_residual(vehicle,x,controls,setup,dt,zz)
    J=jax.jacrev(g)(z); sv=jnp.linalg.svd(J,compute_uv=False); ev=jnp.linalg.eigvals(J)
    return {"spectral_radius":float(jnp.max(jnp.abs(ev))),"spectral_norm":float(sv[0]),"infinity_norm":float(jnp.max(jnp.sum(jnp.abs(J),axis=1))),"largest_singular_value":float(sv[0]),"condition_number":float(sv[0]/(sv[-1]+1e-30))}
