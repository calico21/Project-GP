import sys
import os
import numpy as np
import pandas as pd
import time
import torch

# --- BOTORCH & GPYTORCH IMPORTS (The SOTA Upgrade) ---
try:
    import gpytorch
    from botorch.models import SingleTaskMultiFidelityGP, ModelListGP
    from gpytorch.mlls import SumMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
    from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
    from botorch.optim import optimize_acqf
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from botorch.utils.sampling import draw_sobol_samples
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
except ImportError as e:
    print(f"[Error] BoTorch Import Failed: {e}")
    sys.exit(1)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from models.vehicle_dynamics import MultiBodyVehicle
    from data.configs.vehicle_params import vehicle_params as VP_DICT
    from data.configs.tire_coeffs import tire_coeffs as TP_DICT
except ImportError as e:
    print(f"[Error] Import Failed: {e}")
    sys.exit(1)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

class MultiFidelitySetupOptimizer:
    """
    State-of-the-Art Multi-Objective Multi-Fidelity Bayesian Optimizer (BoTorch).
    Uses Auto-Regressive Co-Kriging to learn residuals between an algebraic proxy 
    and the heavy 14-DOF Thermal ODE solver.
    """
    def __init__(self):
        # Setup Parameter Bounds: [k_f, k_r, arb_f, arb_r, c_f, c_r, h_cg, fidelity]
        self.bounds = torch.tensor([
            [15000., 15000., 0.,    0.,    1000., 1000., 0.25, 0.0],
            [60000., 60000., 2000., 1500., 5000., 5000., 0.35, 1.0]
        ], **tkwargs)
        
        self.var_keys = ['k_f', 'k_r', 'arb_f', 'arb_r', 'c_f', 'c_r', 'h_cg']
        
        # --- THE FIX: RELAXED REFERENCE POINT ---
        # Adjusting the baseline so BoTorch doesn't prune valid transient setups
        self.ref_point = torch.tensor([0.5, -2.0], **tkwargs) 

    def evaluate_algebraic_proxy(self, X_tensor):
        """
        LOW-FIDELITY EVALUATOR (fidelity = 0.0)
        Vectorized steady-state calculation. Extremely cheap.
        """
        k_f, k_r = X_tensor[:, 0], X_tensor[:, 1]
        arb_f, arb_r = X_tensor[:, 2], X_tensor[:, 3]
        c_f, c_r = X_tensor[:, 4], X_tensor[:, 5]
        h_cg = X_tensor[:, 6]

        total_k_f = k_f + arb_f
        total_k_r = k_r + arb_r
        lltd = total_k_f / (total_k_f + total_k_r)
        
        grip_proxy = 1.6 - 2.5 * (h_cg - 0.25) - 1.2 * torch.abs(lltd - 0.5)
        
        c_total = c_f + c_r
        overshoot_proxy = 0.8 - 0.00005 * c_total + 1.5 * torch.abs(lltd - 0.5)
        
        return torch.stack([grip_proxy, -overshoot_proxy], dim=1)

    def evaluate_physics(self, X_tensor):
        """
        HIGH-FIDELITY EVALUATOR (fidelity = 1.0)
        Executes the expensive 14-DOF MultiBody ODE integration.
        """
        Y_list = []
        for i in range(X_tensor.shape[0]):
            x = X_tensor[i].cpu().numpy()
            params = [x[0], x[1], x[2], x[3], x[4], x[5]]
            vehicle = MultiBodyVehicle(VP_DICT, TP_DICT)
            
            dt, T_max = 0.005, 3.0 # Extended slightly for the full slalom
            steps = int(T_max / dt)
            x_curr = np.zeros(17) 
            x_curr[3] = 20.0 
            
            yaw_rates, lat_accels = [], []
            crashed = False

            for t in np.linspace(0, T_max, steps):
                # --- THE FIX: DYNAMIC SLALOM MANEUVER ---
                # A 1 Hz sine sweep violently shifts weight, forcing dampers and ARBs to work.
                steer = 0.15 * np.sin(2.0 * np.pi * 1.0 * t) if t > 0.2 else 0.0
                
                try:
                    x_next = vehicle.simulate_step(x_curr, [steer, 0], params, dt=dt)
                    if not np.all(np.isfinite(x_next)): crashed = True; break
                    
                    vx, r = x_next[3], x_next[5]
                    if abs(r) > 5.0 or abs(vx) > 100.0: crashed = True; break
                    
                    lat_accels.append(vx * r / 9.81)
                    yaw_rates.append(r)
                    x_curr = x_next
                except RuntimeError:
                    crashed = True
                    break
                    
            if crashed or len(lat_accels) < 10:
                Y_list.append([0.0, -5.0]) 
                continue
            
            # Metric 1: Maximize average absolute lateral Gs during the slalom
            f_score = np.mean(np.abs(lat_accels)) 
            
            # Metric 2: Minimize overshoot/instability peak variance
            peak_yaw = np.max(np.abs(yaw_rates))
            steady_yaw = np.mean(np.abs(yaw_rates[-int(steps*0.2):]))
            overshoot = 0.0 if steady_yaw < 0.01 else (peak_yaw - steady_yaw) / steady_yaw
            
            Y_list.append([f_score, -overshoot]) 
            
        return torch.tensor(Y_list, **tkwargs)

    def initialize_multifidelity_surrogate(self, n_lf=500, n_hf=10):
        print(f"\n[BoTorch] Phase 1: Co-Kriging Initialization...")
        print(f"   > Evaluating {n_lf} Low-Fidelity (Algebraic) Samples")
        train_X_lf = draw_sobol_samples(bounds=self.bounds, n=n_lf, q=1).squeeze(1)
        train_X_lf[:, -1] = 0.0
        train_Y_lf = self.evaluate_algebraic_proxy(train_X_lf)

        print(f"   > Evaluating {n_hf} High-Fidelity (14-DOF Thermal) Samples")
        train_X_hf = draw_sobol_samples(bounds=self.bounds, n=n_hf, q=1).squeeze(1)
        train_X_hf[:, -1] = 1.0 
        train_Y_hf = self.evaluate_physics(train_X_hf)

        train_X = torch.cat([train_X_lf, train_X_hf])
        train_Y = torch.cat([train_Y_lf, train_Y_hf])
        
        return train_X, train_Y

    def build_model_list(self, train_X, train_Y):
        models = []
        for i in range(train_Y.shape[-1]):
            train_Y_i = train_Y[..., i:i+1]
            model = SingleTaskMultiFidelityGP(
                train_X, train_Y_i,
                data_fidelities=[7], 
                input_transform=Normalize(d=train_X.shape[-1]),
                outcome_transform=Standardize(m=1)
            )
            models.append(model)
        
        model_list = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
        return model_list, mll

    def run(self, n_iterations=20):
        train_X, train_Y = self.initialize_multifidelity_surrogate()
        
        print(f"\n[BoTorch] Phase 2: Hunting Target-Fidelity Pareto Front (qLogNEHVI)...")
        for iteration in range(n_iterations):
            model, mll = self.build_model_list(train_X, train_Y)
            fit_gpytorch_mll(mll)
            
            acq_func = qLogNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=self.ref_point,
                X_baseline=train_X[train_X[:, -1] == 1.0],
                prune_baseline=True
            )
            
            fixed_acq_func = FixedFeatureAcquisitionFunction(
                acq_function=acq_func,
                d=8,
                columns=[7], 
                values=[1.0] 
            )
            
            candidates, _ = optimize_acqf(
                acq_function=fixed_acq_func,
                bounds=self.bounds[:, :-1], 
                q=1,
                num_restarts=5,
                raw_samples=128,
            )
            
            new_X = torch.cat([candidates.detach(), torch.tensor([[1.0]], **tkwargs)], dim=-1)
            new_Y = self.evaluate_physics(new_X)
            
            print(f"   > Active Learning Iter {iteration+1:02d}: Evaluated HF Obj=[Grip: {new_Y[0,0]:.3f} Gs, Stability: {new_Y[0,1]:.3f}]")
            
            train_X = torch.cat([train_X, new_X])
            train_Y = torch.cat([train_Y, new_Y])

        hf_mask = train_X[:, -1] == 1.0
        hf_X, hf_Y = train_X[hf_mask], train_Y[hf_mask]
        
        pareto_mask = is_non_dominated(hf_Y)
        pareto_X = hf_X[pareto_mask].cpu().numpy()
        pareto_Y = hf_Y[pareto_mask].cpu().numpy()
        
        pareto_Y[:, 1] = -pareto_Y[:, 1] 
        final_pop = [{k: v for k, v in zip(self.var_keys, row[:-1])} for row in pareto_X]
        return final_pop, pareto_Y

if __name__ == "__main__":
    optimizer = MultiFidelitySetupOptimizer()
    final_pop, final_obj = optimizer.run(n_iterations=20)
    
    df = pd.DataFrame(final_pop)
    df['Lat_G_Score'] = final_obj[:, 0]
    df['Stability_Overshoot'] = final_obj[:, 1]
    
    print("\n[BoTorch] Multi-Fidelity Optimization Complete. Target-Fidelity Pareto Front:")
    print(df.sort_values('Lat_G_Score', ascending=False).to_string(index=False))
    
    out_file = os.path.join(project_root, 'optimization_results.csv')
    df.to_csv(out_file, index=False)
    print(f"\n[Success] Results mapped via Multi-Fidelity Tensors and saved to {out_file}")