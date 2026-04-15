from acados_sim_builder import AcadosSimBuilder
from system_dynamics import DynamicBicycleSystemDynamics, KinematicBicycleSystemDynamics, BlendedBicycleSystemDynamics

if __name__ == "__main__":
    blended_b = BlendedBicycleSystemDynamics()
    dynamic_b = DynamicBicycleSystemDynamics()

    # nothing uses this. this is shit. a cul de sac

    #AcadosSimBuilder(blended_b, model_type="explicit", T=0.001).generate("gen/blended_bicycle_explicit")
    #AcadosSimBuilder(blended_b, model_type="timescaling_implicit", T=1).generate("gen/blended_bicycle_timescaling_implicit")
    #AcadosSimBuilder(dynamic_b, model_type="timescaling_implicit", T=1).generate("gen/dynamic_bicycle_timescaling_implicit")