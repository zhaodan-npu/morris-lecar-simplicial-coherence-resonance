"""Build the new figure for Reviewer 1, comment 1.

Panel (a): deterministic bifurcation diagram of the single Morris-Lecar node in
the applied current, with the subcritical Hopf point, the fold of limit cycles,
the bistable window, and the working point I = 88.
Panel (b): optimal ISI coefficient of variation CV* against the input current.
Panel (c): the corresponding optimal noise intensity D_v*.

Everything is drawn in black with distinct markers, line styles and line
widths, so the figure survives black-and-white printing.

Inputs:
  --bifurcation-dir   output of morris_lecar_hopf_distance_analysis.py
  --analysis-dir      output of aggregate_current_bifurcation_distance.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = (
    # label in the data, legend text, marker, line style, filled?
    (("control", 1.5), "Pairwise, $\\alpha = 1.5$", "o", "--", False),
    (("higher_order", 1.5), "Simplicial, $\\alpha = 1.5$", "o", "-", True),
    (("control", 2.0), "Pairwise, $\\alpha = 2.0$", "s", "--", False),
    (("higher_order", 2.0), "Simplicial, $\\alpha = 2.0$", "s", "-", True),
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def draw_bifurcation(axis, bifurcation_dir):
    summary = json.loads((bifurcation_dir / "hopf_distance_summary.json").read_text(encoding="utf-8"))
    equilibrium = read_csv(bifurcation_dir / "branch_equilibrium.csv")
    cycles = read_csv(bifurcation_dir / "branch_limit_cycles.csv")

    current_hopf = summary["hopf"]["current"]
    fold = summary["fold_of_limit_cycles"]["current"]

    axis.set_xlim(75.0, 100.0)
    axis.set_ylim(-62.0, 52.0)
    axis.axvspan(fold, current_hopf, facecolor="none", edgecolor="0.55", hatch="///", linewidth=0.0)

    handles = []
    currents = np.array([float(row["current"]) for row in equilibrium])
    v_star = np.array([float(row["v_star"]) for row in equilibrium])
    stable = np.array([row["stable"] == "True" for row in equilibrium])
    for flag, style, width, legend in ((True, "-", 1.6, "equilibrium, stable"),
                                       (False, "--", 1.4, "equilibrium, unstable")):
        selection = stable == flag
        if not selection.any():
            continue
        breaks = np.where(np.diff(np.flatnonzero(selection)) > 1)[0]
        first = True
        for piece in np.split(np.flatnonzero(selection), breaks + 1):
            line, = axis.plot(currents[piece], v_star[piece], color="k",
                              linestyle=style, linewidth=width,
                              label=legend if first else None)
            if first:
                handles.append(line)
            first = False

    for branch, style, width, legend in (("stable", "-", 2.8, "limit cycle, stable"),
                                         ("unstable", ":", 1.6, "limit cycle, unstable")):
        rows = [row for row in cycles if row["branch"] == branch]
        if not rows:
            continue
        order = np.argsort([float(row["current"]) for row in rows])
        c = np.array([float(rows[i]["current"]) for i in order])
        for index, key in enumerate(("v_max", "v_min")):
            line, = axis.plot(c, np.array([float(rows[i][key]) for i in order]),
                              color="k", linestyle=style, linewidth=width,
                              label=legend if index == 0 else None)
            if index == 0:
                handles.append(line)

    axis.axvline(88.0, color="k", linestyle="-", linewidth=0.7)
    axis.annotate("$I = 88$", xy=(87.9, -27.4), xytext=(76.5, -44.0), fontsize=8,
                  arrowprops=dict(arrowstyle="->", lw=0.7))
    axis.annotate("$I_{\\rm fold}$", xy=(fold, 47.0), xytext=(fold - 7.5, 44.0), fontsize=8,
                  arrowprops=dict(arrowstyle="->", lw=0.7))
    axis.annotate("$I_{\\rm H}$", xy=(current_hopf, 47.0), xytext=(current_hopf + 2.6, 44.0), fontsize=8,
                  arrowprops=dict(arrowstyle="->", lw=0.7))
    axis.set_xlabel("$I$")
    axis.set_ylabel("$V$")
    axis.legend(handles=handles, frameon=False, loc="upper left",
                bbox_to_anchor=(0.02, 0.76), fontsize=6.5, handlelength=2.4,
                borderaxespad=0.0, labelspacing=0.35)


def draw_metric(axis, rows, key, low_key, high_key, ylabel, fold, current_hopf):
    axis.axvspan(fold, current_hopf, facecolor="none", edgecolor="0.6", hatch="///", linewidth=0.0)
    for (label, alpha), legend, marker, style, filled in SERIES:
        selection = [row for row in rows
                     if row["label"] == label and abs(float(row["alpha"]) - alpha) < 1e-9]
        selection.sort(key=lambda row: float(row["current"]))
        if not selection:
            continue
        x = np.array([float(row["current"]) for row in selection])
        y = np.array([float(row[key]) for row in selection])
        lo = np.array([float(row[low_key]) for row in selection])
        hi = np.array([float(row[high_key]) for row in selection])
        axis.errorbar(x, y, yerr=np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)]),
                      color="k", linestyle=style, linewidth=1.3, marker=marker, markersize=4.5,
                      markerfacecolor="k" if filled else "w", markeredgecolor="k",
                      capsize=2.0, elinewidth=0.8, label=legend)
    axis.axvline(88.0, color="k", linestyle="-", linewidth=0.8)
    axis.set_xlabel("$I$")
    axis.set_ylabel(ylabel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bifurcation-dir", default="bifurcation_analysis")
    parser.add_argument("--analysis-dir", default="current_bifurcation_distance_analysis")
    parser.add_argument("--output-stem", default="fig_current_bifurcation_distance")
    args = parser.parse_args()

    bifurcation_dir = Path(args.bifurcation_dir)
    analysis_dir = Path(args.analysis_dir)
    summary = json.loads((bifurcation_dir / "hopf_distance_summary.json").read_text(encoding="utf-8"))
    fold = summary["fold_of_limit_cycles"]["current"]
    current_hopf = summary["hopf"]["current"]

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 7.5})
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))

    draw_bifurcation(axes[0], bifurcation_dir)

    summary_path = analysis_dir / "cr_vs_current.csv"
    if summary_path.exists():
        rows = read_csv(summary_path)
        draw_metric(axes[1], rows, "cv_star", "cv_star_ci_low", "cv_star_ci_high",
                    "${\\rm CV}^{*}$", fold, current_hopf)
        draw_metric(axes[2], rows, "dv_star", "dv_star_ci_low", "dv_star_ci_high",
                    "$D_v^{*}$", fold, current_hopf)
        axes[1].legend(frameon=False, loc="best")
    else:
        for axis in axes[1:]:
            axis.text(0.5, 0.5, "run aggregate_current_bifurcation_distance.py first",
                      ha="center", va="center", transform=axis.transAxes, fontsize=8)

    for letter, axis in zip("abc", axes):
        axis.text(-0.16, 1.04, f"({letter})", transform=axis.transAxes, fontsize=11, va="bottom")

    figure.tight_layout()
    for suffix in (".png", ".pdf"):
        figure.savefig(args.output_stem + suffix, dpi=300)
    print(f"figure written to {args.output_stem}.png / .pdf")


if __name__ == "__main__":
    main()
