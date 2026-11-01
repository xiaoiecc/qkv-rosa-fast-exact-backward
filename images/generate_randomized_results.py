"""Plot final measured backward and training costs; no extrapolated timings."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
native = json.loads((ROOT / "benchmarks/results/random_final_native.json").read_text())
training = json.loads((ROOT / "research/results/training_round_final_summary.json").read_text())
assert native["status"] == "PASS" and training["status"] == "COMPLETE"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.axisbelow": True,
    "svg.fonttype": "none",
})
fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.4))
fig.patch.set_facecolor("#fbfcfe")
colors = ["#273c59", "#168f88", "#e29b33"]
labels = ["Exact", "Approx. / all channels", "Approx. / 8 channels"]
methods = ["exact", "final32", "final_channels8"]
cases = [
    next(r for r in native["rows"] if r["name"] == "prefix_copy_fresh" and r["order"] == "second"),
    next(r for r in native["rows"] if r["name"] == "prefix_copy64_fresh" and r["order"] == "second"),
]
ax = axes[0]
ys = []
texts = []
for group, row in enumerate(cases):
    for j, method in enumerate(methods):
        y = group * 4 + j
        value = row["backward"][method]["median_ms"]
        ax.barh(y, value, color=colors[j], height=.69)
        ax.text(value + 4, y, f"{value:.2f}", va="center", fontsize=10)
        ys.append(y); texts.append(labels[j])
    ax.text(0, group * 4 - .67, f"T = {row['T']:,}, D = {row['D']}", weight="bold", fontsize=12)
ax.set_yticks(ys, texts)
ax.set_xlim(0, 365)
ax.set_ylim(6.7, -1.1)
ax.set_xlabel("Complete native backward (ms; 9-sample median)")
ax.set_title("A. Captured long-prefix adjoints", loc="left", weight="bold", pad=16)
ax.grid(axis="x", alpha=.18)
ax.tick_params(axis="y", length=0)

ax = axes[1]
ys, texts = [], []
for group, length in enumerate((2048, 4096)):
    for j, method in enumerate(("exact", "final", "final_channels")):
        rows = [r for r in training["rows"] if r["T"] == length and
                r["order"] == "second" and r["seed"] == 1101 and r["method"] == method]
        y = group * 4 + j
        ys.append(y); texts.append(labels[j])
        if not rows:
            ax.text(.15, y, "not measured", va="center", color="#687589", fontsize=10)
            continue
        value = rows[0]["seconds"]
        ax.barh(y, value, color=colors[j], height=.69)
        ax.text(value + .22, y, f"{value:.3f}", va="center", fontsize=10)
    ax.text(0, group * 4 - .67, f"T = {length:,}, D = 64", weight="bold", fontsize=12)
ax.set_yticks(ys, texts)
ax.set_xlim(0, 23.6)
ax.set_ylim(6.7, -1.1)
ax.set_xlabel("20 complete forward / backward / Adam steps (s)")
ax.set_title("B. Ordinary training on prefix-copy stress", loc="left", weight="bold", pad=16)
ax.grid(axis="x", alpha=.18)
ax.tick_params(axis="y", length=0)
fig.suptitle("Shared long intervals: large savings with an explicit approximation choice",
             x=.05, ha="left", fontsize=16, weight="bold")
fig.text(.05, .075,
         "CPU, one numerical thread, second-order credit. Approximation: 32 residual rows. "
         "Panel B uses discovery seed 1101.\n"
         "These are structured stress inputs. Other inputs are slower or have large gradient errors; "
         "see the full 26-case report and fresh seeds.",
         fontsize=10, color="#45536a", linespacing=1.5)
fig.subplots_adjust(left=.16, right=.97, top=.82, bottom=.22, wspace=.63)
for suffix in ("png", "svg"):
    path = ROOT / f"images/randomized-results.{suffix}"
    fig.savefig(path,
                dpi=175, facecolor=fig.get_facecolor(), bbox_inches="tight")
    if suffix == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
print("Saved randomized-results.png and randomized-results.svg")
