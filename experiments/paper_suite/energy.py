"""Adapters for the repository's existing controlled pH and vehicle energy labs."""
from __future__ import annotations
from experiments.energy_lab import run_part_1, run_part_2

def run_energy(mode, dt):
    # The controlled benchmark is the existing research benchmark, not a new toy.
    tf={"smoke":.05,"standard":.5,"paper":5.0}[mode]
    toy=run_part_1(T_final=tf,dt=dt)
    vehicle=run_part_2(duration=tf,dt=dt)
    rows=[]
    for r in toy:
        mono=sum(1 for a,b in zip(r["H_trajectory"],r["H_trajectory"][1:]) if b>a+1e-10)
        rows.append({"domain":"controlled_pH","case":r["name"],"initial_energy":r["H0"],"final_energy":r["H_final"],"delta_H":r["delta_H"],"relative_energy_drift":r["delta_H"]/max(abs(r["H0"]),1e-12),"power_balance_residual":r["balance_residual"],"input_work":r["W_supply"],"dissipation":r["W_diss"],"monotonicity_violations":mono})
    for r in vehicle:
        e=r["E_total"]; rows.append({"domain":"vehicle_observational","case":r["name"],"initial_energy":float(e[0]),"final_energy":float(e[-1]),"delta_H":float(e[-1]-e[0]),"relative_energy_drift":float((e[-1]-e[0])/max(abs(e[0]),1e-12)),"power_balance_residual":None,"input_work":None,"dissipation":None,"monotonicity_violations":None})
    return rows,{"toy_cases":len(toy),"vehicle_note":"Observational external-force/actuation diagnostic, not a global passivity proof."}
