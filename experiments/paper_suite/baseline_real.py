"""Fair, executable mechanical-subsystem baseline protocol.

The learned baselines in :mod:`benchmarks.baselines` are 28-state models.
Their second 14 states are generalized momentum; the vehicle exposes velocity,
so this adapter performs the documented diagonal ``p=Mv`` conversion.
"""
from __future__ import annotations
import time
import jax, jax.numpy as jnp
import numpy as np
from benchmarks.baselines import BASELINE_REGISTRY
from config.tire_coeffs import tire_coeffs as TC
from config.vehicles.ter27 import vehicle_params_ter27 as VP
from models.vehicle_dynamics import DEFAULT_SETUP, DifferentiableMultiBodyVehicle

OBS = {"vx":14,"vy":15,"yaw_rate":19,"roll":3,"pitch":4,"front_heave":6,"rear_heave":8}

def _to_baseline(vehicle, state):
    return jnp.concatenate((state[:14], vehicle.M_diag * state[14:28]))
def _from_baseline(vehicle, state):
    return jnp.concatenate((state[:14], state[14:28] / vehicle.M_diag))

def manoeuvres(n, dtype=jnp.float32):
    t=jnp.arange(n,dtype=dtype)
    return {
      "straight":jnp.stack((jnp.zeros(n,dtype),jnp.full(n,8.,dtype),jnp.full(n,8.,dtype),jnp.full(n,8.,dtype),jnp.full(n,8.,dtype),jnp.zeros(n,dtype)),1),
      "steering_transient":jnp.stack((.06*jnp.sin(t/3),jnp.full(n,8.,dtype),jnp.full(n,8.,dtype),jnp.full(n,8.,dtype),jnp.full(n,8.,dtype),jnp.zeros(n,dtype)),1),
      "sustained_lateral":jnp.stack((jnp.full(n,.05,dtype),jnp.full(n,8.,dtype),jnp.full(n,8.,dtype),jnp.full(n,8.,dtype),jnp.full(n,8.,dtype),jnp.zeros(n,dtype)),1),
      "brake_steer":jnp.stack((.04*jnp.sin(t/2),jnp.zeros(n,dtype),jnp.zeros(n,dtype),jnp.full(n,40.,dtype),jnp.full(n,40.,dtype),jnp.zeros(n,dtype)),1),
    }

def make_training_data(vehicle, n, seed, dt):
    """Generate actual one-step labels from production vehicle dynamics."""
    key=jax.random.PRNGKey(seed); rows=[]
    for i in range(n):
        key,ka,kb=jax.random.split(key,3); vx=jax.random.uniform(ka,(),minval=10.,maxval=30.)
        x=vehicle.make_initial_state(vx0=float(vx)); x=x.at[15].set(jax.random.uniform(kb,(),minval=-1.,maxval=1.))
        u=jnp.array([jax.random.uniform(ka,(),minval=-.05,maxval=.05),8.,8.,8.,8.,0.],dtype=x.dtype)
        xn=vehicle.simulate_step(x,u,DEFAULT_SETUP,dt=dt,n_substeps=1)
        rows.append((_to_baseline(vehicle,x),u,_to_baseline(vehicle,xn),DEFAULT_SETUP))
    return {"x":jnp.stack([r[0] for r in rows]),"u":jnp.stack([r[1] for r in rows]),"x_next":jnp.stack([r[2] for r in rows]),"setup":jnp.stack([r[3] for r in rows])}

def _metrics(ref,pred):
    d=np.asarray(pred-ref); ref=np.asarray(ref)
    block={"q":float(np.sqrt(np.mean(d[:,:14]**2))),"v":float(np.sqrt(np.mean(d[:,14:28]**2)))}
    return {"rmse":float(np.sqrt(np.mean(d*d))),"mae":float(np.mean(np.abs(d))),"nrmse":float(np.sqrt(np.mean(d*d))/(np.ptp(ref)+1e-12)),"max_abs_error":float(np.max(np.abs(d))),"final_error":float(np.linalg.norm(d[-1])),"block_rmse":block, **{f"rmse_{name}":float(np.sqrt(np.mean(d[:,idx]**2))) for name,idx in OBS.items()}}

def run_baselines(mode, seed, dt):
    # Paper mode is deliberately a real pilot, not the old smoke configuration.
    n_train={"smoke":4,"standard":32,"paper":1000}[mode]; n_val={"smoke":1,"standard":8,"paper":200}[mode]; epochs={"smoke":1,"standard":5,"paper":50}[mode]; horizon={"smoke":3,"standard":20,"paper":200}[mode]
    vehicle=DifferentiableMultiBodyVehicle(VP,TC); all_data=make_training_data(vehicle,n_train+n_val,seed,dt)
    train={k:v[:n_train] for k,v in all_data.items()}; validation={k:v[n_train:] for k,v in all_data.items()}; output=[]; histories={}
    for mi,(name,cls) in enumerate(BASELINE_REGISTRY.items()):
        model=cls(dt=dt); start=time.perf_counter()
        try:
            info=model.train(jax.random.PRNGKey(seed+mi),train,n_epochs=epochs,lr=3e-4,batch_size=min(n_train,16)); params=model.params
            val_loss=float(model.loss_fn(params, validation["x"], validation["u"], validation["x_next"], validation["setup"]))
            histories[name]={"train_loss":float(info["final_loss"]),"validation_loss":val_loss,"train_time_s":time.perf_counter()-start}
            for m,controls in manoeuvres(horizon,train["x"].dtype).items():
                x=vehicle.make_initial_state(vx0=20.); ref=[]
                for u in controls: x=vehicle.simulate_step(x,u,DEFAULT_SETUP,dt=dt,n_substeps=1); ref.append(x[:28])
                predp=model.predict_trajectory(params,_to_baseline(vehicle,vehicle.make_initial_state(vx0=20.)),controls,DEFAULT_SETUP)
                pred=jax.vmap(lambda z:_from_baseline(vehicle,z))(predp); finite=bool(jnp.all(jnp.isfinite(pred)))
                for h in sorted(set((1,min(10,horizon),min(50,horizon),min(100,horizon),horizon))):
                    output.append({"model":name,"manoeuvre":m,"horizon_steps":h,"failure":not finite,"nan_inf_count":int(jnp.size(pred)-jnp.sum(jnp.isfinite(pred))),"runtime_s":time.perf_counter()-start,**_metrics(jnp.stack(ref)[:h],pred[:h])})
        except Exception as exc:
            output.append({"model":name,"status":"failed","reason":repr(exc)})
    return output,{"n_train":n_train,"n_validation":n_val,"epochs":epochs,"horizon":horizon,"representation":"[q, p=M_diag*v]; results converted back to [q,v]","training":histories}
