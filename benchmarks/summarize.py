"""Side-by-side summary of reference vs optimised benchmark runs.

Usage: python benchmarks/summarize.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path(__file__).resolve().parent / "results"


def load(name):
    """Load a result file; ``a.json+b.json`` concatenates the runs of both."""
    parts = [RES / n for n in name.split("+")]
    if not all(p.exists() for p in parts):
        return None
    data = [json.loads(p.read_text()) for p in parts]
    merged = dict(data[0])
    merged["runs"] = sorted((r for d in data for r in d["runs"]), key=lambda r: r["seed"])
    return merged


def table(title, files, metrics, better):
    runs = {label: load(f) for label, f in files}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        return
    print("\n## " + title)
    first = next(iter(runs.values()))
    print("config: " + json.dumps(first["config"]))
    seeds = [r["seed"] for r in first["runs"]]
    print("\n| metric ({} is better) | ".format(better) +
          " | ".join(runs) + " |")
    print("|---" * (len(runs) + 1) + "|")
    for m in metrics:
        cells = []
        for v in runs.values():
            x = np.array([r[m] for r in v["runs"]])
            cells.append("{:.4f} ± {:.4f}".format(x.mean(), x.std(ddof=1) if len(x) > 1 else 0))
        print("| {} | ".format(m) + " | ".join(cells) + " |")

    print("\nper-seed values:")
    for m in metrics:
        for label, v in runs.items():
            print("  {:>10s} {:<14s} ".format(m, label) +
                  " ".join("{:.6f}".format(r[m]) for r in v["runs"]))

    ref_key = next((k for k in runs if k.startswith("reference")), None)
    if ref_key:
        ref = runs[ref_key]["runs"]
        for label, v in runs.items():
            if label == ref_key:
                continue
            print("\n{} minus {} (paired over seeds {}):".format(label, ref_key, seeds))
            for m in metrics:
                a = np.array([r[m] for r in v["runs"]])
                b = np.array([r[m] for r in ref])
                d = a - b
                p = stats.ttest_rel(a, b).pvalue if np.any(d != 0) else 1.0
                print("  {:>10s}: mean diff {:+.3e}, max |diff| {:.3e}, paired t-test p={:.3f}".format(
                    m, d.mean(), np.abs(d).max(), p))

    print("\ntime per run (s):")
    for label, v in runs.items():
        for key in ("train_time_s", "test_time_s"):
            if key in v["runs"][0]:
                x = np.array([r[key] for r in v["runs"]])
                print("  {:<22s} {:<12s} {:9.2f} ± {:.2f}".format(label, key, x.mean(), x.std()))


table("uDTW: ECG5000 forecasting (UCR), 5 seeds",
      [("reference CPU", "ecg5000_ref_cpu.json"), ("fast CPU", "ecg5000_fast_cpu.json"),
       ("fast CUDA", "ecg5000_fast_cuda.json")],
      ["mse", "dtw", "sdtw_div"], "lower")
table("JEANIE: NW-UCLA cross-view 5-way 1-shot, 5 seeds",
      [("reference CPU", "nwucla_ref_cpu.json+nwucla_ref_cpu_s34.json"),
       ("fast CPU", "nwucla_fast_cpu.json"), ("fast CUDA", "nwucla_fast_cuda.json")],
      ["accuracy"], "higher")
table("uDTW float64 equivalence check: ECG5000, seeds 0-1",
      [("reference CPU", "ecg5000_f64_ref_cpu.json"), ("fast CPU", "ecg5000_f64_fast_cpu.json"),
       ("fast CUDA", "ecg5000_f64_fast_cuda.json")],
      ["mse", "dtw", "sdtw_div", "final_train_loss"], "lower")
table("JEANIE float64 equivalence check: NW-UCLA, seed 0",
      [("reference CPU", "nwucla_f64_ref_cpu.json"), ("fast CPU", "nwucla_f64_fast_cpu.json"),
       ("fast CUDA", "nwucla_f64_fast_cuda.json")],
      ["accuracy", "final_train_loss"], "higher")
