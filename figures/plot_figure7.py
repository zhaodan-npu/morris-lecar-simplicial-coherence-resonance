"""Restyle the six robustness panels from the original stored optimal values."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUITE = ROOT / "paper_suite_true_simplicial_prl_supplement_Dv"
COLORS = {"control": "#36454f", "higher_order": "#c62828"}
LABELS = {"control": "Pairwise", "higher_order": "Simplicial"}
MARKERS = {"control": "o", "higher_order": "s"}
GROUPS = {
    "N": ("size_N", r"$N$", [60, 80, 100, 120], ["60", "80", "100", "120"]),
    "pfill": ("pfill_", r"$p_{\mathrm{fill}}$", [0, .25, .5, .75], ["0", "0.25", "0.50", "0.75"]),
    "pedge": ("pedge_", r"$p_{\mathrm{edge}}$", [.1, .15, .2], ["0.10", "0.15", "0.20"]),
}
SPECS = [("a", "N", "cv"), ("b", "pfill", "cv"), ("c", "pedge", "cv"),
         ("d", "N", "Dv"), ("e", "pfill", "Dv"), ("f", "pedge", "Dv")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes():
    paths = [ROOT / "Coherence_Resonance.pdf", ROOT / "plot_completed_Dv_experiments.py",
             SUITE / "merged_summary.json", SUITE / "summary_figures" / "supplement_best_summary.csv"]
    paths.extend(SUITE / "summary_figures" / f"supplement_best_{'CV' if m == 'cv' else 'Dv'}_summary_{g}.pdf"
                 for _, g, m in SPECS)
    return [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in paths]


def read_data():
    summary = json.loads((SUITE / "merged_summary.json").read_text(encoding="utf-8"))
    with (SUITE / "summary_figures" / "supplement_best_summary.csv").open(newline="", encoding="utf-8") as handle:
        original = list(csv.DictReader(handle))
    rows = []
    for group, (prefix, _, ticks, _) in GROUPS.items():
        selected = [r for r in original if r["experiment"].startswith(prefix)]
        assert len(selected) == 2 * len(ticks)
        for alpha in (1.5, 2.):
            ordered = sorted((r for r in selected if float(r["alpha"]) == alpha),
                             key=lambda r: float(r["experiment"][len(prefix):]))
            np.testing.assert_array_equal([float(r["experiment"][len(prefix):]) for r in ordered], ticks)
            for row in ordered:
                for label in COLORS:
                    key = "control" if label == "control" else "higher"
                    best = summary[row["experiment"]]["best_by_alpha"][str(alpha)][label]
                    cv, dv = float(row[f"{key}_cv"]), float(row[f"{key}_Dv"])
                    np.testing.assert_allclose(cv, best["best_cv_value"], rtol=0, atol=1e-12)
                    np.testing.assert_allclose(dv, best["best_noise_intensity_dv"], rtol=0, atol=1e-12)
                    rows.append({"group": group, "experiment": row["experiment"],
                                 "x": float(row["experiment"][len(prefix):]),
                                 "alpha": alpha, "label": label, "cv": cv, "Dv": dv})
    assert len(rows) == 44
    return rows


def style():
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 18,
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "axes.labelsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 16, "legend.frameon": False, "axes.linewidth": 1.4,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "xtick.minor.width": 1., "ytick.minor.width": 1.,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": None,
    })


def draw(spec, rows, output_dir=None):
    output_dir = HERE if output_dir is None else Path(output_dir)
    letter, group, metric = spec
    _, xlabel, ticks, ticklabels = GROUPS[group]
    span = ticks[-1] - ticks[0]
    xlim = [ticks[0] - .05 * span, ticks[-1] + .05 * span]
    ylim = [0, .85 if metric == "cv" else 1.05]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    fig.subplots_adjust(left=.18, right=.97, bottom=.18, top=.78)
    lines = []
    for alpha in (1.5, 2.):
        for label in COLORS:
            subset = sorted((r for r in rows if r["group"] == group and r["alpha"] == alpha
                             and r["label"] == label), key=lambda r: r["x"])
            x = np.array([r["x"] for r in subset])
            y = np.array([r[metric] for r in subset])
            line, = ax.plot(x, y, color=COLORS[label], marker=MARKERS[label],
                            linestyle="--" if label == "control" else "-", linewidth=2.2,
                            markersize=5.8, markeredgewidth=1.2, markeredgecolor=COLORS[label],
                            markerfacecolor=COLORS[label] if alpha == 1.5 else "white",
                            label=rf"{LABELS[label]}, $\alpha={alpha:.1f}$")
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y)
            lines.append(line)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(ticks, ticklabels)
    ax.set_yticks([0, .2, .4, .6, .8] if metric == "cv" else [0, .25, .50, .75, 1.])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\mathrm{CV}^{*}$" if metric == "cv" else r"$D_v^*$")
    ax.xaxis.set_minor_locator(AutoMinorLocator(4 if group != "pfill" else 5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=True, right=True)
    ax.grid(False)
    legend = fig.legend(lines, [line.get_label() for line in lines], ncol=2,
                        loc="upper center", bbox_to_anchor=(.51, .995),
                        handlelength=1.6, handletextpad=.45, columnspacing=1.,
                        labelspacing=.4, borderaxespad=0.)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lb = legend.get_window_extent(renderer)
    assert not lb.overlaps(ax.bbox) and lb.x0 >= 0 and lb.x1 <= fig.bbox.x1
    boxes = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, (
            letter, text.get_text(), box.bounds)
        boxes.append({"text": text.get_text(), "bbox_pixels": list(box.bounds)})
    for axis in (ax.xaxis, ax.yaxis):
        labels = axis.get_ticklabels()
        for first, second in zip(labels, labels[1:]):
            assert not first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer))
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"fig7{letter}.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)
    return {"panel": letter, "group": group, "metric": metric, "xlim": xlim, "ylim": ylim,
            "n_points": 4 * len(ticks), "pdf_sha256": sha(output_dir / f"fig7{letter}.pdf"),
            "legend_outside_axes": True, "no_clipped_text": True,
            "no_tick_overlap": True, "panel_letters": False, "text_boxes": boxes}


def main():
    before = input_hashes()
    rows = read_data()
    style()
    records = [draw(spec, rows) for spec in SPECS]
    assert before == input_hashes()
    (HERE / "analysis").mkdir(exist_ok=True)
    data_path = HERE / "analysis" / "plot_data.csv"
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "analysis" / "manifest.json").write_text(json.dumps(
        {"inputs": before, "script_sha256": sha(Path(__file__)), "plot_data_sha256": sha(data_path),
         "panels": records, "source_inputs_unchanged": True, "optima_reselected": False,
         "error_bars_added": False, "simulations_run": False}, indent=2), encoding="utf-8")
    print("Exported six Figure 7 panels with external legends; 88 optimal values retained.")


if __name__ == "__main__":
    main()
