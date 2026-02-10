#!/usr/bin/env python3
"""
Plot checkpoint strategy scaling results.

Reads checkpoint_scaling.csv (produced by test_gretl_checkpoint_scaling)
and generates two figures:
  1. Recomputation overhead vs graph length
  2. Peak memory usage vs graph length

Usage:
    python plot_checkpoint_scaling.py [path/to/checkpoint_scaling.csv]
"""

import sys
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    """Return {strategy: {col: [values]}}."""
    data = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row["strategy"]
            data[s]["N"].append(int(row["N"]))
            data[s]["recomputations"].append(int(row["recomputations"]))
            data[s]["peak_stored"].append(int(row["peak_stored"]))
            data[s]["stores"].append(int(row["stores"]))
            data[s]["evictions"].append(int(row["evictions"]))
            data[s]["wall_seconds"].append(float(row["wall_seconds"]))
            data[s]["budget_or_period"].append(int(row["budget_or_period"]))
    return data


STYLE = {
    "StoreAll":      dict(color="tab:green",  marker="s", linestyle="--", linewidth=2),
    "Wang":          dict(color="tab:blue",   marker="o", linestyle="-",  linewidth=2),
    "StrummWalther": dict(color="tab:orange", marker="^", linestyle="-",  linewidth=2),
    "Periodic":      dict(color="tab:red",    marker="D", linestyle="-",  linewidth=2),
}

BUDGET = 1000


def plot_recomputations(ax, data):
    """Recomputation count vs graph length."""
    for strategy, sdata in data.items():
        N = np.array(sdata["N"])
        recomps = np.array(sdata["recomputations"])
        st = STYLE.get(strategy, {})

        if strategy == "StoreAll":
            # StoreAll has 0 recomps — show feasible region only
            feasible = N <= BUDGET
            ax.plot(N[feasible], recomps[feasible], label="StoreAll (0 recomps, feasible)",
                    markersize=8, **st)
            if np.any(~feasible):
                ax.plot(N[~feasible], recomps[~feasible], label="StoreAll (exceeds budget)",
                        alpha=0.3, markersize=8, **{**st, "linestyle": ":"})
        else:
            ax.plot(N, recomps, label=strategy, markersize=6, **st)

    ax.set_xlabel("Graph length  N", fontsize=12)
    ax.set_ylabel("Recomputations during backprop", fontsize=12)
    ax.set_title(f"Compute overhead  (memory budget = {BUDGET})", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)   # handles 0 gracefully
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)


def plot_recomp_ratio(ax, data):
    """Recomputation ratio (recomps / N) vs graph length."""
    for strategy, sdata in data.items():
        N = np.array(sdata["N"], dtype=float)
        recomps = np.array(sdata["recomputations"], dtype=float)
        ratio = recomps / N
        st = STYLE.get(strategy, {})

        if strategy == "StoreAll":
            feasible = N <= BUDGET
            ax.plot(N[feasible], ratio[feasible], label="StoreAll (feasible)",
                    markersize=8, **st)
            if np.any(~feasible):
                ax.plot(N[~feasible], ratio[~feasible], label="StoreAll (exceeds budget)",
                        alpha=0.3, markersize=8, **{**st, "linestyle": ":"})
        else:
            ax.plot(N, ratio, label=strategy, markersize=6, **st)

    ax.set_xlabel("Graph length  N", fontsize=12)
    ax.set_ylabel("Recomputations / N", fontsize=12)
    ax.set_title(f"Normalised compute overhead  (budget = {BUDGET})", fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)


def plot_peak_memory(ax, data):
    """Peak stored checkpoints vs graph length."""
    for strategy, sdata in data.items():
        N = np.array(sdata["N"])
        peak = np.array(sdata["peak_stored"])
        st = STYLE.get(strategy, {})
        ax.plot(N, peak, label=strategy, markersize=6, **st)

    ax.axhline(y=BUDGET, color="red", linestyle="--", alpha=0.5, linewidth=1.5,
               label=f"Budget = {BUDGET}")
    ax.set_xlabel("Graph length  N", fontsize=12)
    ax.set_ylabel("Peak stored checkpoints", fontsize=12)
    ax.set_title("Memory usage vs graph length", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)


def plot_wall_time(ax, data):
    """Wall-clock time vs graph length."""
    for strategy, sdata in data.items():
        N = np.array(sdata["N"])
        wall = np.array(sdata["wall_seconds"])
        st = STYLE.get(strategy, {})

        if strategy == "StoreAll":
            feasible = N <= BUDGET
            ax.plot(N[feasible], wall[feasible], label="StoreAll (feasible)",
                    markersize=8, **st)
            if np.any(~feasible):
                ax.plot(N[~feasible], wall[~feasible], label="StoreAll (exceeds budget)",
                        alpha=0.3, markersize=8, **{**st, "linestyle": ":"})
        else:
            ax.plot(N, wall, label=strategy, markersize=6, **st)

    ax.set_xlabel("Graph length  N", fontsize=12)
    ax.set_ylabel("Wall-clock time  (s)", fontsize=12)
    ax.set_title("Total time (forward + backprop)", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoint_scaling.csv"
    data = read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_recomputations(axes[0, 0], data)
    plot_recomp_ratio(axes[0, 1], data)
    plot_peak_memory(axes[1, 0], data)
    plot_wall_time(axes[1, 1], data)

    fig.suptitle("Checkpoint Strategy Scaling Study\n"
                 f"Linear chain,  State<double,double>,  memory budget = {BUDGET}",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = csv_path.replace(".csv", ".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
