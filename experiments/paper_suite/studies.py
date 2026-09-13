"""Executable studies.  Unsupported learned comparisons are recorded as unavailable."""
from __future__ import annotations
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from config.vehicles.ter27 import vehicle_params_ter27 as VP
from config.tire_coeffs import tire_coeffs as TC
from models.vehicle_dynamics import DEFAULT_SETUP, SETUP_LB, SETUP_UB, SETUP_NAMES, DifferentiableMultiBodyVehicle
from experiments.full_vehicle_gradient_audit import (AuditCase, _full_setup, _state_objective, converged_stages, explicit_ift, finite_difference, implicit_context, make_case, production_stages, reconstruct_from_stages, stage_residual)
from .common import SuiteConfig, finite_metrics, write_result
from .solver import solve_stages_picard, solve_stages_newton, local_picard_jacobian

def _case(cfg): return make_case(AuditCase(n_setup_params=2, dt=cfg.dt))
def _controls(n, dtype=jnp.float64):
    t=jnp.arange(n,dtype=dtype); return jnp.stack((.03+.01*jnp.sin(t/4),jnp.full(n,10.,dtype),jnp.full(n,10.,dtype),jnp.full(n,10.,dtype),jnp.full(n,10.,dtype),jnp.zeros(n,dtype)),axis=1)

def solver_study(cfg: SuiteConfig):
    vehicle,x,u,s,_=_case(cfg); cases=[("nominal",x,u,s)]
    for i in range(5 if cfg.mode != "smoke" else 1):
        cases += [(f"state_{i}",x.at[14+i].add(.02*(i+1)),u,s),(f"control_{i}",x,u.at[i%6].add(.02*(i+1)),s),(f"setup_{i}",x,u,s.at[i].add(20.*(i+1)))]
    rows=[]
    for label,xx,uu,ss in cases:
        ref=solve_stages_picard(vehicle,xx,uu,ss,cfg.dt,tol=1e-10,max_iterations=128)
        for method, values, fn in (("picard",cfg.picard_iterations(),solve_stages_picard),("newton",(1,2,3) if cfg.mode=="smoke" else (1,2,3,4,5,6,8,10,15),solve_stages_newton)):
            for n in values:
                info=fn(vehicle,xx,uu,ss,cfg.dt,iterations=n)
                rows.append({"case":label,"method":method,"budget":n,"iterations":info["iterations"],"converged":info["converged"],"residual_inf":info["final_residual_inf"],"residual_l2":info["final_residual_l2"],"wall_time_s":info["wall_time_s"],"root_distance_l2":float(jnp.linalg.norm(info["stages"]-ref["stages"])),**info["branch_signature"]})
        rows.append({"case":label,"method":"local_picard","budget":0,**local_picard_jacobian(vehicle,xx,uu,ss,cfg.dt,ref["stages"])})
    return write_result(cfg,"solver",{"stage_dimension":216,"state_dimension":108,"local_contraction_interpretation":"Local observation only; spectral radius below one is not a global guarantee."},rows=rows)

def gradient_study(cfg: SuiteConfig):
    vehicle,x,u,s,target=_case(cfg)
    indexes=jnp.array([0,1] if cfg.mode=="smoke" else [0,1,2,3,4,5,6,7,8,9,12,13,14,15,16,17])
    names=[SETUP_NAMES[int(i)] for i in indexes]; theta=s[indexes]
    def full(th): return s.at[indexes].set(th)
    def objective(th):
        setup=full(th); z=production_stages(vehicle,x,u,setup,cfg.dt); return _state_objective(vehicle,reconstruct_from_stages(x,z,cfg.dt),target,setup,"benchmark")
    ad=jax.grad(objective)(theta); zstar, info=converged_stages(vehicle,x,u,s,cfg.dt,tol=1e-9,max_iter=128)
    # Independent root-map IFT for an arbitrary (non-contiguous) parameter subset.
    F=lambda z,th: stage_residual(vehicle,x,u,full(th),cfg.dt,z)
    fz=jax.jacrev(F,0)(zstar,theta); ft=jax.jacrev(F,1)(zstar,theta); dz=jnp.linalg.solve(fz,-ft)
    G=lambda z,th: _state_objective(vehicle,reconstruct_from_stages(x,z,cfg.dt),target,full(th),"benchmark")
    ift=jax.jacrev(G,1)(zstar,theta)+jax.jacrev(G,0)(zstar,theta)@dz
    rows=[]
    for eps in ((1e-3,1e-4) if cfg.mode=="smoke" else (1e-2,1e-3,1e-4,1e-5,1e-6)):
        fd=finite_difference(objective,theta,eps)
        for i,name in enumerate(names):
            near=abs(float(fd[i]))<1e-10; rows.append({"parameter":name,"epsilon":eps,"production_ad":float(ad[i]),"ift":float(ift[i]),"central_fd":float(fd[i]),"absolute_error_ad_fd":abs(float(ad[i]-fd[i])),"relative_error_ad_fd":None if near else abs(float(ad[i]-fd[i]))/abs(float(fd[i])),"near_zero_sensitivity":near,"sign_agreement":bool(jnp.sign(ad[i])==jnp.sign(fd[i]))})
    return write_result(cfg,"gradients",{"converged_root":info,"ad_vs_ift":finite_metrics(ad,ift),"parameters_evaluated":names,"ift_identity_l2":float(jnp.linalg.norm(fz@dz+ft)),"near_zero_rule":"relative error omitted where |central FD| < 1e-10"},rows=rows)

def baseline_study(cfg: SuiteConfig):
    from .baseline_real import run_baselines, OBS
    seeds=[cfg.seed] if cfg.mode != "paper" else [cfg.seed, cfg.seed+1, cfg.seed+2]
    rows=[]; runs=[]
    for seed in seeds:
        run_rows, info=run_baselines(cfg.mode,seed,cfg.dt)
        for row in run_rows: row["seed"]=seed
        rows.extend(run_rows); runs.append(info)
    audit={"dataset":{"training_samples_per_seed":runs[0]["n_train"],"validation_samples_per_seed":runs[0]["n_validation"],"test":"four deterministic production rollouts"},"seeds":seeds,"epochs":runs[0]["epochs"],"optimizer":"Adam","learning_rate":3e-4,"weight_decay":0.0,"dt_s":cfg.dt,"horizon_steps":runs[0]["horizon"],"state_subset":"28 mechanical states [q,p=M_diag*v], evaluated as [q,v]","controls":6,"setup_dimensions":28,"models":sorted({r["model"] for r in rows if "model" in r})}
    base=Path(cfg.output); base.mkdir(parents=True,exist_ok=True)
    import json
    (base/"fairness_audit.json").write_text(json.dumps(audit,indent=2))
    return write_result(cfg,"baseline",{**runs[0],"common_observables":OBS,"reference":"production 108-state vehicle projected to 28 mechanical states","fairness_audit":"fairness_audit.json","seed_count":len(seeds)},status="complete",rows=rows)

def energy_study(cfg: SuiteConfig):
    from .energy import run_energy
    rows, info=run_energy(cfg.mode,cfg.dt)
    return write_result(cfg,"energy",info,status="complete",rows=rows)

def ablation_study(cfg: SuiteConfig):
    variants={"A0":"unconstrained matched baseline requires trained checkpoint","A1":"remove ICNN constraints requires retraining","A2":"value-only grounding requires retraining","A3":"PSD dissipation ablation not independently configurable","A4":"FiLM removal requires retraining","A5":"explicit RK4 adapter requires a matched trained/evaluation protocol","A7":"energy-density normalization toggle is not exposed","A9":"full PassiveHNet production map"}
    rows=[{"variant":k,"status":"available" if k=="A9" else "unavailable","reason":v,"seeds":0 if k!="A9" else 1} for k,v in variants.items()]
    return write_result(cfg,"ablations",{"honesty_note":"No ablation is assigned performance values without a matched trained model and protocol."},status="partial",rows=rows)

def setup_optimization(cfg: SuiteConfig):
    vehicle,x,u,s,_=_case(cfg); idx=jnp.array([0,1]); controls=_controls(2 if cfg.mode=="smoke" else min(10,cfg.horizon()))
    def loss(theta):
        setup=s.at[idx].set(theta); state=x; vals=[]
        for c in controls:
            state=vehicle.simulate_step(state,c,setup,dt=cfg.dt,n_substeps=1); vals.append(state[3]**2+state[6]**2+state[8]**2)
        return jnp.mean(jnp.stack(vals))+1e-10*jnp.sum((theta-s[idx])**2)
    theta=s[idx]; rows=[]
    for it in range(2 if cfg.mode=="smoke" else 8):
        value,grad=jax.value_and_grad(loss)(theta); new=jnp.clip(theta-2e4*grad,SETUP_LB[idx],SETUP_UB[idx]); rows.append({"iteration":it,"objective":float(value),"gradient_norm":float(jnp.linalg.norm(grad)),"k_f":float(theta[0]),"k_r":float(theta[1]),"constraint_violation":float(jnp.sum(jnp.maximum(SETUP_LB[idx]-theta,0)+jnp.maximum(theta-SETUP_UB[idx],0))),"gradient_method":"production_ad"}); theta=new
    return write_result(cfg,"setup_optimization",{"objective":"mean roll/heave proxy under fixed combined control; no unsupported splitter observable","initial_setup":s[idx],"final_setup":theta,"success":bool(jnp.all(jnp.isfinite(theta)) )},rows=rows)

def extrapolation_study(cfg: SuiteConfig):
    rows=[]
    for split,vxs in (("interpolation",(10.,20.,30.)),("velocity_extrapolation",(35.,40.))):
        for vx in vxs:
            v=DifferentiableMultiBodyVehicle(VP,TC); x=v.make_initial_state(vx0=vx); nxt=v.simulate_step(x,jnp.array([.03,10,10,10,10,0.]),DEFAULT_SETUP,dt=cfg.dt,n_substeps=1); rows.append({"split":split,"vx_mps":vx,"finite":bool(jnp.all(jnp.isfinite(nxt))),"state_norm":float(jnp.linalg.norm(nxt)),"one_step_change_l2":float(jnp.linalg.norm(nxt-x))})
    return write_result(cfg,"extrapolation",{"status_note":"Observational production-map generalization diagnostic; learned interpolation/extrapolation RMSE unavailable without training split/checkpoints."},status="partial",rows=rows)

def compute_study(cfg: SuiteConfig):
    vehicle,x,u,s,_=_case(cfg); rows=[]
    for name,fn in (("production_forward",lambda:vehicle.simulate_step(x,u,s,dt=cfg.dt,n_substeps=1)),("picard",lambda:solve_stages_picard(vehicle,x,u,s,cfg.dt,iterations=2)),("newton",lambda:solve_stages_newton(vehicle,x,u,s,cfg.dt,iterations=1))):
        fn(); samples=[]
        for _ in range(2 if cfg.mode=="smoke" else 5):
            t=time.perf_counter(); fn(); samples.append((time.perf_counter()-t)*1000)
        rows.append({"method":name,"warm_runtime_mean_ms":float(np.mean(samples)),"warm_runtime_std_ms":float(np.std(samples)),"p50_ms":float(np.percentile(samples,50)),"p95_ms":float(np.percentile(samples,95)),"samples":len(samples)})
    return write_result(cfg,"compute",{"timing_note":"Warm Python/JAX timing; compilation time is intentionally excluded."},rows=rows)
