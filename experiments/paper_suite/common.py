"""Shared configuration, provenance, serialisation, and numerical helpers."""
from __future__ import annotations

import csv, hashlib, json, platform, subprocess, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "paper_suite"

@dataclass(frozen=True)
class SuiteConfig:
    mode: str = "smoke"
    seed: int = 0
    output: str = str(DEFAULT_OUT)
    dt: float = .005
    def horizon(self) -> int: return {"smoke": 3, "standard": 20, "paper": 200}[self.mode]
    def picard_iterations(self) -> tuple[int, ...]:
        return (1, 2, 4) if self.mode == "smoke" else (1, 2, 4, 8, 16, 32, 64, 128)

def git_commit() -> str | None:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception: return None

def metadata(config: SuiteConfig, experiment: str, elapsed_s: float | None = None) -> dict[str, Any]:
    return {"git_commit": git_commit(), "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version, "jax_version": jax.__version__, "platform": platform.platform(),
            "dtype": str(jax.config.jax_default_dtype_bits), "seed": config.seed,
            "experiment_config": asdict(config), "experiment": experiment, "elapsed_wall_s": elapsed_s}

def jsonable(x: Any) -> Any:
    if isinstance(x, (np.ndarray, jax.Array)): return np.asarray(x).tolist()
    if isinstance(x, (np.floating, np.integer)): return x.item()
    if isinstance(x, Path): return str(x)
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    return x

def write_result(config: SuiteConfig, name: str, metrics: dict, status="completed", rows=None) -> Path:
    start = time.perf_counter()
    out = Path(config.output) / name; out.mkdir(parents=True, exist_ok=True)
    report = metadata(config, name, time.perf_counter() - start) | {"status": status, "metrics": metrics}
    p = out / f"{name}.json"; p.write_text(json.dumps(jsonable(report), indent=2, sort_keys=True))
    if rows is not None:
        rows = jsonable(rows)
        if rows:
            with (out / f"{name}.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
                writer.writeheader(); writer.writerows(rows)
    return p

def finite_metrics(a, b, eps=1e-30):
    a, b = jnp.asarray(a), jnp.asarray(b); d = a-b
    return {"absolute_l2": float(jnp.linalg.norm(d)), "relative": float(jnp.linalg.norm(d)/(jnp.linalg.norm(b)+eps)),
            "cosine": float(jnp.vdot(a,b)/(jnp.linalg.norm(a)*jnp.linalg.norm(b)+eps))}

def manifest(config: SuiteConfig) -> Path:
    base = Path(config.output); files = []
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.name not in {"manifest.json", "final_manifest.json"}:
            files.append({"path": str(p.relative_to(base)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    payload=json.dumps(jsonable(metadata(config, "manifest") | {"artifacts": files}), indent=2)
    p = base / "manifest.json"; p.write_text(payload)
    # Kept separately for the paper hand-off while retaining backward compatibility.
    (base / "final_manifest.json").write_text(payload)
    return p
