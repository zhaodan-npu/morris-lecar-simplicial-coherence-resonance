"""Plot unaltered paired trajectories and node-1 empirical ISI distributions."""

from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "analysis"
FIG = HERE / "figures"


def tex_number(x):
    coefficient, exponent = f"{x:.3e}".split("e")
    return rf"{coefficient}\times10^{{{int(exponent)}}}"


def style():
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 17,
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "axes.labelsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 14, "axes.linewidth": 1.4, "legend.frameon": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "xtick.minor.width": 1., "ytick.minor.width": 1.,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def axes():
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    fig.subplots_adjust(left=.18, right=.94, bottom=.20, top=.83)
    ax.tick_params(which="both", top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    return fig, ax


def save(fig, ax, stem):
    ax.legend(loc="lower left", bbox_to_anchor=(-.01, 1.015), ncol=2,
              handlelength=1.7, columnspacing=.9, handletextpad=.4, borderaxespad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xlabels = [label for value, label in zip(ax.get_xticks(), ax.get_xticklabels()) if ax.get_xlim()[0] <= value <= ax.get_xlim()[1]]
    ylabels = [label for value, label in zip(ax.get_yticks(), ax.get_yticklabels()) if ax.get_ylim()[0] <= value <= ax.get_ylim()[1]]
    for artist in (ax.xaxis.label, ax.yaxis.label, ax.get_legend(), *xlabels, *ylabels):
        if not artist.get_visible():
            continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, (stem, str(artist), box)
    for suffix in ("pdf", "eps", "png"):
        fig.savefig(FIG / f"{stem}.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    summary = json.loads((OUT / "summary.json").read_text())
    assert summary["integer_operator_equal_all_networks"] and summary["all_spike_checks_passed"]
    macros = [rf"\newcommand{{\EqMatrixError}}{{{tex_number(summary['operator_max_abs_error'])}}}"]
    hashes = []
    for case in summary["cases"]:
        name = f"alpha{str(case['alpha']).replace('.', '')}"
        trajectory = OUT / f"{name}_trajectory.npz"
        raw = np.load(trajectory)
        t = raw["t"]
        use = (t >= 3000.) & (t <= 4000.)
        fig, ax = axes()
        for mode, label, color, ls in (("simplex", "Simplex sum", "#c62828", "-"),
                                        ("projected", "Weighted pairwise", "#2166ac", (0, (5, 3)))):
            ax.plot(t[use], raw[f"voltage_{mode}"][use], label=label, color=color, linestyle=ls, lw=1.9)
        ax.set(xlabel=r"$t$ (ms)", ylabel=r"$V_1$ (mV)", xlim=(3000, 4000))
        ax.margins(y=.08)
        save(fig, ax, f"{name}_voltage")
        isi_file = OUT / f"{name}_selected_node_isi.csv"
        with isi_file.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        fig, ax = axes()
        max_isi = 0.
        for mode, label, color, ls in (("simplex", "Simplex sum", "#c62828", "-"),
                                        ("projected", "Weighted pairwise", "#2166ac", (0, (5, 3)))):
            samples = np.sort([float(row["isi"]) for row in rows if row["mode"] == mode])
            assert len(samples) > 0
            max_isi = max(max_isi, float(samples[-1]))
            x = np.r_[0., samples, samples[-1]*1.05]
            y = np.r_[0., np.arange(1, len(samples)+1)/len(samples), 1.]
            ax.step(x, y, where="post", color=color, linestyle=ls, lw=2., label=label)
        ax.set(xlabel="ISI (ms)", ylabel="Empirical CDF", xlim=(0, max_isi*1.05), ylim=(-.025, 1.025))
        ax.set_yticks([0, .25, .5, .75, 1])
        save(fig, ax, f"{name}_isi")
        suffix = "Levy" if case["alpha"] == 1.5 else "Gaussian"
        macros.extend([
            rf"\newcommand{{\EqVoltage{suffix}}}{{{tex_number(case['max_all_nodes_all_steps_voltage_abs_error'])}}}",
            rf"\newcommand{{\EqCV{suffix}}}{{{case['selected_node']['cv_simplex']:.6f}}}",
            rf"\newcommand{{\EqISI{suffix}}}{{{case['selected_node']['isis_simplex']}}}",
        ])
        for path in (trajectory, isi_file):
            hashes.append({"path": str(path.relative_to(HERE)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT / "numbers.tex").write_text("\n".join(macros)+"\n", encoding="utf-8")
    (OUT / "plot_manifest.json").write_text(json.dumps({"inputs": hashes, "voltage_window": [3000, 4000],
         "voltage_subsampling": False, "isi_window": [3000, 15000], "isi_binning": False,
         "selected_node": 1, "panel_letters": False, "n_panels": 4}, indent=2), encoding="utf-8")
    print("Four panels exported, with full-resolution traces and unbinned empirical ISI CDFs.")


if __name__ == "__main__":
    main()
