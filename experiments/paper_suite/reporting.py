"""Automatic plot/table generation from structured artifacts only."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import SuiteConfig, write_result

def _rows(path):
    p=path / (path.name + ".csv")
    return list(csv.DictReader(p.open())) if p.exists() else []

def figures(cfg: SuiteConfig):
    base=Path(cfg.output); out=base/"figures"; out.mkdir(parents=True,exist_ok=True); made=[]
    specs=[("solver","fig04_solver_convergence","budget","residual_inf","Picard/Newton residual convergence"), ("baseline","fig07_baseline_rollout","horizon_steps","rmse","Baseline rollout error"), ("energy","fig09_energy_balance_toy","case","relative_energy_drift","Energy balance diagnostic"), ("compute","fig23_compute_cost","method","warm_runtime_mean_ms","Warm runtime")]
    for study,name,xkey,ykey,title in specs:
        rows=_rows(base/study)
        rows=[r for r in rows if xkey in r and ykey in r and r[ykey] not in ("",None)]
        if not rows: continue
        fig,ax=plt.subplots(figsize=(5.2,3.2)); groups={}
        for r in rows: groups.setdefault(r.get("method",study),[]).append(r)
        for label,rs in groups.items():
            try: ax.plot([float(r[xkey]) if xkey!="method" else i for i,r in enumerate(rs)], [float(r[ykey]) for r in rs],marker="o",label=label)
            except ValueError:
                # Categorical studies (e.g. named energy cases) retain their data.
                ax.bar(range(len(rs)), [float(r[ykey]) for r in rs], label=label)
                ax.set_xticks(range(len(rs)), [r[xkey] for r in rs], rotation=25, ha="right")
        if ykey=="residual_inf": ax.set_yscale("log")
        ax.set(title=title,xlabel=xkey,ylabel=ykey); ax.grid(True,alpha=.3)
        if groups: ax.legend()
        fig.tight_layout()
        for ext in ("png","pdf"): fig.savefig(out/f"{name}.{ext}",dpi=300)
        plt.close(fig); made += [f"{name}.png",f"{name}.pdf"]
    return write_result(cfg,"figures",{"generated":made},status="completed")

def tables(cfg: SuiteConfig):
    base=Path(cfg.output); out=base/"tables"; out.mkdir(parents=True,exist_ok=True); made=[]
    mapping={"solver":"table_solver.tex","baseline":"table_baselines.tex","energy":"table_energy.tex","ablations":"table_ablations.tex","gradients":"table_gradients.tex","setup_optimization":"table_setup_optimization.tex","compute":"table_compute.tex","extrapolation":"table_extrapolation.tex"}
    for study,filename in mapping.items():
        rows=_rows(base/study)
        if not rows: continue
        cols=list(rows[0])[:8]
        tex=["\\begin{tabular}{"+"l"*len(cols)+"}"," \\toprule"," & ".join(c.replace("_","\\_") for c in cols)+" \\\\ \\midrule"]
        tex += [" & ".join(str(r.get(c,"-")).replace("_","\\_") for c in cols)+" \\\\" for r in rows]
        tex += ["\\bottomrule","\\end{tabular}"]
        (out/filename).write_text("\n".join(tex)); made.append(filename)
    return write_result(cfg,"tables",{"generated":made},status="completed")
