import sys
import os
import numpy as np
import pandas as pd
import time
from scipy.stats import norm

# --- SCILKIT-LEARN FOR KRIGING ---
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
except ImportError:
    print("[Error] sklearn missing. Please install scikit-learn.")
    sys.exit(1)

# --- IMPORT PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS ---
try:
    from models.vehicle_dynamics import MultiBodyVehicle
    from data.configs.vehicle_params import vehicle_params as VP_DICT
    from data.configs.tire_coeffs import tire_coeffs as TP_DICT
except ImportError as e:
    print(f"[Error] Import Failed: {e}")
    sys.exit(1)

# --- PYMOO IMPORTS ---
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination


class MultiFidelitySurrogate:
    """
    Autoregressive Co-Kriging Surrogate Model.
    Predicts: Y_high(x) = Y_low(x) + GP_residual(x)
    """
    def __init__(self):
        # Matern kernel is ideal for physical processes
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
        self.gp_grip = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-4)
        self.gp_stab = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-4)
        self.is_trained = False
        
    def low_fidelity_model(self, ind):
        """
        Cheap, 0-DOF Algebraic Approximation (Microseconds to compute).
        Maps setup trends but lacks transient accuracy.
        """
        k_f, k_r = ind['k_f'], ind['k_r']
        arb_f, arb_r = ind['arb_f'], ind['arb_r']
        h_cg = ind['h_cg']
        
        # Total lateral load transfer approximation
        roll_stiff_f = k_f + arb_f * 50
        roll_stiff_r = k_r + arb_r * 50
        balance = roll_stiff_f / (roll_stiff_f + roll_stiff_r + 1e-6)
        
        # Grip proxy: lower CG is better, extreme balance is bad
        ideal_balance = 0.55
        grip = -1.6 + (h_cg - 0.25)*2.0 + abs(balance - ideal_balance)*1.5
        
        # Stability proxy: Front stiffness increases understeer (stability)
        stability = max(0.0, 0.2 - (balance - 0.5)*0.5)
        
        return np.array([grip, stability])

    def train_residuals(self, X_train, Y_high):
        """Trains the GPs on the error between High and Low fidelity."""
        Y_low = np.array([self.low_fidelity_model(self._vec_to_dict(x)) for x in X_train])
        Y_res = Y_high - Y_low
        
        self.gp_grip.fit(X_train, Y_res[:, 0])
        self.gp_stab.fit(X_train, Y_res[:, 1])
        self.is_trained = True

    def predict(self, x, return_std=False):
        """Combined Multi-Fidelity Prediction."""
        ind = self._vec_to_dict(x)
        y_low = self.low_fidelity_model(ind)
        
        if not self.is_trained:
            if return_std: return y_low, np.array([1.0, 1.0])
            return y_low
            
        res_g, std_g = self.gp_grip.predict(x.reshape(1, -1), return_std=True)
        res_s, std_s = self.gp_stab.predict(x.reshape(1, -1), return_std=True)
        
        y_pred = y_low + np.array([res_g[0], res_s[0]])
        
        if return_std:
            return y_pred, np.array([std_g[0], std_s[0]])
        return y_pred

    def _vec_to_dict(self, x):
        keys = ['k_f', 'k_r', 'arb_f', 'arb_r', 'c_f', 'c_r', 'h_cg']
        return {k: v for k, v in zip(keys, x)}


class ActiveLearningOptimizer:
    """
    Manages the Bayesian Active Learning Loop using Expected Improvement (EI).
    """
    def __init__(self, bounds):
        self.bounds = bounds
        self.surrogate = MultiFidelitySurrogate()
        self.X_history = []
        self.Y_history = []
        
    def expected_improvement(self, X_candidate, gp, current_best):
        """Calculates EI for a set of candidate points."""
        mu, std = gp.predict(X_candidate, return_std=True)
        with np.errstate(divide='warn'):
            Z = (current_best - mu) / (std + 1e-9)
            ei = (current_best - mu) * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0.0] = 0.0
        return ei

    def run_high_fidelity(self, x):
        """Executes the expensive 14-DOF MultiBody ODE integration."""
        ind = self.surrogate._vec_to_dict(x)
        params = [ind['k_f'], ind['k_r'], ind['arb_f'], ind['arb_r'], ind['c_f'], ind['c_r']]
        vehicle = MultiBodyVehicle(VP_DICT, TP_DICT)
        
        dt, T_max = 0.005, 2.5
        steps = int(T_max / dt)
        x_curr = np.zeros(14) # Upgraded 14-DOF thermal vector
        x_curr[3] = 20.0 
        
        yaw_rates, lat_accels = [], []
        crashed = False

        for t in np.linspace(0, T_max, steps):
            steer = 0.1 if t > 0.2 else 0.0
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
                
        if crashed or len(lat_accels) < 10: return [0.0, 5.0]
        
        steady_ay = np.mean(lat_accels[-int(steps*0.2):])
        f_score = -abs(steady_ay) 
        
        peak_yaw = np.max(np.abs(yaw_rates))
        steady_yaw = np.mean(np.abs(yaw_rates[-int(steps*0.2):]))
        overshoot = 0.0 if steady_yaw < 0.01 else (peak_yaw - steady_yaw) / steady_yaw
            
        return [f_score, overshoot]

    def build_surrogate(self, initial_samples=5, active_iterations=10):
        print("\n[BayesianOpt] Phase 1: Building Multi-Fidelity Surrogate...")
        
        # 1. LHS Initial Sampling
        xl, xu = self.bounds
        for _ in range(initial_samples):
            x = np.random.uniform(xl, xu)
            y = self.run_high_fidelity(x)
            self.X_history.append(x)
            self.Y_history.append(y)
            
        self.surrogate.train_residuals(np.array(self.X_history), np.array(self.Y_history))
        
        # 2. Active Learning via Expected Improvement
        print(f"[BayesianOpt] Phase 2: Active Learning ({active_iterations} Iterations)...")
        for i in range(active_iterations):
            # Generate 10,000 cheap candidates to evaluate EI
            X_cand = np.random.uniform(xl, xu, (10000, len(xl)))
            
            # Use scalarized metric for EI (e.g., focus heavily on grip)
            current_best = np.min(np.array(self.Y_history)[:, 0])
            ei_scores = self.expected_improvement(X_cand, self.surrogate.gp_grip, current_best)
            
            best_cand = X_cand[np.argmax(ei_scores)]
            
            # Execute Expensive Model ONLY on the point with max EI
            print(f"   > Iter {i+1}: Running High-Fidelity Physics on max EI candidate...")
            y_true = self.run_high_fidelity(best_cand)
            
            self.X_history.append(best_cand)
            self.Y_history.append(y_true)
            self.surrogate.train_residuals(np.array(self.X_history), np.array(self.Y_history))
            
        print("[BayesianOpt] Autoregressive Co-Kriging Trained Successfully.")
        return self.surrogate


class SurrogateVehicleSetupProblem(ElementwiseProblem):
    """
    NSGA-II Problem that evaluates the fast Multi-Fidelity Surrogate 
    instead of the heavy ODE physics engine.
    """
    def __init__(self, surrogate, xl, xu):
        self.surrogate = surrogate
        self.var_keys = ['k_f', 'k_r', 'arb_f', 'arb_r', 'c_f', 'c_r', 'h_cg']
        super().__init__(n_var=7, n_obj=2, n_ieq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # Microsecond evaluation using Co-Kriging predictions
        y_pred = self.surrogate.predict(x)
        out["F"] = [y_pred[0], y_pred[1]]


class SetupOptimizer:
    def __init__(self, pop_size=100, generations=50):
        self.pop_size = pop_size
        self.generations = generations
        self.xl = np.array([15000., 15000., 0.,   0.,   1000., 1000., 0.25])
        self.xu = np.array([60000., 60000., 2000.,1500., 5000., 5000., 0.35])

    def run(self):
        # 1. Train the Multi-Fidelity Surrogate using Active Learning
        bo_manager = ActiveLearningOptimizer(bounds=(self.xl, self.xu))
        trained_surrogate = bo_manager.build_surrogate(initial_samples=5, active_iterations=15)
        
        # 2. Run genetic algorithm instantly on the Surrogate
        print(f"\n[Optimizer] Initializing NSGA-II on Surrogate Surface (Pop={self.pop_size}, Gen={self.generations})...")
        problem = SurrogateVehicleSetupProblem(trained_surrogate, self.xl, self.xu)
        
        algorithm = NSGA2(
            pop_size=self.pop_size,
            n_offsprings=40,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", self.generations),
            seed=1,
            verbose=False 
        )
        
        print(f"[Optimizer] GA Complete. Found {len(res.X)} Pareto-optimal setups.")
        
        final_pop = [{k: v for k, v in zip(problem.var_keys, row)} for row in res.X]
        return final_pop, res.F

if __name__ == "__main__":
    opt = SetupOptimizer(pop_size=100, generations=50) # Now we can afford huge populations!
    final_pop, final_obj = opt.run()
    
    df = pd.DataFrame(final_pop)
    df['Lat_G_Score'] = final_obj[:, 0]
    df['Stability_Overshoot'] = final_obj[:, 1]
    
    out_file = os.path.join(project_root, 'optimization_results.csv')
    df.to_csv(out_file, index=False)
    print(f"\n[Success] Results saved to {out_file}")