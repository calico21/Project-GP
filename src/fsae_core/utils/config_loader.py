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
            # e.g., suspension['front']['spring_rate'] -> params['k_spring_f']
            if filename == 'suspension.yaml':
                params['k_spring'] = [
                    data['front']['spring_rate'], data['front']['spring_rate'], # FL, FR
                    data['rear']['spring_rate'],  data['rear']['spring_rate']   # RL, RR
                ]
                params['c_damper'] = [2000, 2000, 2000, 2000] # Default placeholders if not in YAML
                params['track_width_f'] = data['front']['track_width']
                params['track_width_r'] = data['rear']['track_width']
                # Assume wheelbase is split 50/50 roughly if not specified, 
                # or read from chassis.yaml if you added it there.
                params['wheelbase_f'] = 0.8
                params['wheelbase_r'] = 0.75
                params['tire_radius'] = 0.23 # m
                
                # Pacejka Defaults (if not in YAML)
                params.update({'Dy': 1.6, 'Cy': 1.3, 'By': 8.0})
                params.update({'Dx': 1.6, 'Cx': 1.3, 'Bx': 10.0})

            elif filename == 'aero.yaml':
                params['Cl'] = data.get('cla', 3.0)
                params['Cd'] = data.get('cda', 1.0)
                
            else: # Chassis
                params.update(data)
                # Ensure inertia is accessible directly
                if 'inertia' in data:
                    params['Ixx'] = data['inertia']['Ixx']
                    params['Iyy'] = data['inertia']['Iyy']
                    params['Izz'] = data['inertia']['Izz']
                
                # Add dummy variables for Unsprung Mass if missing
                params['unsprung_mass'] = 15.0 # kg per corner
                params['wheel_inertia'] = 0.5 
                params['max_torque'] = 250 # Nm
                params['max_brake'] = 1000 # Nm

    return params