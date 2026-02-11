import yaml
import os

def load_vehicle_config(config_dir="config/vehicle_params"):
    """
    Reads chassis, suspension, and aero YAMLs and merges them into
    a single flat dictionary for the physics engine.
    """
    params = {}
    
    # List of files to merge
    files = ['chassis.yaml', 'suspension.yaml', 'aero.yaml']
    
    for filename in files:
        path = os.path.join(config_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing config file: {path}")
            
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            
            # Flatten structure (Simpler for the solver)
            if filename == 'suspension.yaml':
                params['k_spring'] = [
                    data['front']['spring_rate'], data['front']['spring_rate'], # FL, FR
                    data['rear']['spring_rate'],  data['rear']['spring_rate']   # RL, RR
                ]
                params['c_damper'] = [2000, 2000, 2000, 2000] # Default damper rates
                params['track_width_f'] = data['front']['track_width']
                params['track_width_r'] = data['rear']['track_width']
                params['wheelbase_f'] = 0.8
                params['wheelbase_r'] = 0.75
                params['tire_radius'] = 0.23 # m
                
                # --- FIX: Add Vertical Tire Stiffness ---
                # Default to 150,000 N/m (Typical for 13" FSAE Tire) if not in YAML
                params['k_tire'] = data.get('tire_vertical_stiffness', 150000.0) 
                
                # Pacejka Defaults (Tire Grip Parameters)
                params.update({'Dy': 1.6, 'Cy': 1.3, 'By': 8.0})
                params.update({'Dx': 1.6, 'Cx': 1.3, 'Bx': 10.0})

            elif filename == 'aero.yaml':
                params['Cl'] = data.get('cla', 3.0)
                params['Cd'] = data.get('cda', 1.0)
                
            else: # Chassis
                params.update(data)
                if 'inertia' in data:
                    params['Ixx'] = data['inertia']['Ixx']
                    params['Iyy'] = data['inertia']['Iyy']
                    params['Izz'] = data['inertia']['Izz']
                
                # Add defaults for missing Chassis params
                params['unsprung_mass'] = params.get('unsprung_mass', 15.0) 
                params['wheel_inertia'] = params.get('wheel_inertia', 0.5) 
                params['max_torque'] = params.get('max_torque', 250)
                params['max_brake'] = params.get('max_brake', 1000)

    return params