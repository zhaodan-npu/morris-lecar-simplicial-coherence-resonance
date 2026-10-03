"""Give Figure 8 the same external-legend style as the preceding figures."""

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUITE = ROOT / "paper_suite_true_simplicial_prl_supplement_Dv"
sys.path.insert(0, str(HERE.parent / "figure7"))
from plot_figure7 import style, sha, COLORS, LABELS, MARKERS


def inputs():
    paths = [ROOT / "Coherence_Resonance.pdf", ROOT / "plot_completed_Dv_experiments.py",
             HERE.parent / "figure7" / "plot_figure7.py", SUITE / "merged_summary.json",
             SUITE / "summary_figures" / "supplement_best_summary.csv"]
    paths.extend(SUITE / "summary_figures" / f"supplement_alpha_sweep_{metric}.pdf" for metric in ("CV", "Dv"))
    return [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in paths]


def read_data():
    with (SUITE / "summary_figures" / "supplement_best_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = sorted((r for r in csv.DictReader(handle) if r["experiment"] == "alpha_sweep"),
                      key=lambda r: float(r["alpha"]))
    summary = json.loads((SUITE / "merged_summary.json").read_text())["alpha_sweep"]["best_by_alpha"]
    np.testing.assert_array_equal([float(r["alpha"]) for r in rows], [1.2, 1.5, 1.8, 2.])
    data = []
    for row in rows:
        for label in COLORS:
            key = "control" if label == "control" else "higher"
            cv, dv = float(row[f"{key}_cv"]), float(row[f"{key}_Dv"])
            best = summary[row["alpha"]][label]
            np.testing.assert_allclose(cv, best["best_cv_value"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(dv, best["best_noise_intensity_dv"], rtol=0, atol=1e-12)
            data.append({"alpha": float(row["alpha"]), "label": label, "cv": cv, "Dv": dv})
    return data


def draw(letter, metric, rows, output_dir=None):
    output_dir = HERE if output_dir is None else Path(output_dir)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    fig.subplots_adjust(left=.18, right=.97, bottom=.18, top=.80)
    lines = []
    for label in COLORS:
        selected = sorted((r for r in rows if r["label"] == label), key=lambda r: r["alpha"])
        x = np.array([r["alpha"] for r in selected])
        y = np.array([r[metric] for r in selected])
        line, = ax.plot(x, y, color=COLORS[label], marker=MARKERS[label],
                        linestyle="--" if label == "control" else "-", linewidth=2.2,
                        markersize=5.8, markeredgewidth=1.2, markeredgecolor=COLORS[label],
                        markerfacecolor=COLORS[label], label=LABELS[label])
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), y)
        lines.append(line)
    ax.set_xlim(1.15, 2.05)
    ax.set_ylim(0, .85 if metric == "cv" else 1.05)
    ax.set_xticks([1.2, 1.5, 1.8, 2.], ["1.2", "1.5", "1.8", "2.0"])
    ax.set_yticks([0, .2, .4, .6, .8] if metric == "cv" else [0, .25, .5, .75, 1.])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\mathrm{CV}^{*}$" if metric == "cv" else r"$D_v^*$")
    ax.xaxis.set_minor_locator(AutoMinorLocator(3))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=True, right=True)
    ax.grid(False)
    legend = fig.legend(lines, [LABELS[label] for label in COLORS], ncol=2,
                        loc="upper center", bbox_to_anchor=(.575, .96),
                        handlelength=1.6, handletextpad=.45, columnspacing=1., borderaxespad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert not legend.get_window_extent(renderer).overlaps(ax.bbox)
    boxes = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, (
            letter, text.get_text(), box.bounds)
        boxes.append({"text": text.get_text(), "bbox_pixels": list(box.bounds)})
    for axis in (ax.xaxis, ax.yaxis):
        ticks = axis.get_ticklabels()
        for first, second in zip(ticks, ticks[1:]):
            assert not first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer))
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"fig8{letter}.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)
    return {"panel": letter, "metric": metric, "xlim": [1.15, 2.05],
            "ylim": [0, .85 if metric == "cv" else 1.05], "n_points": 8,
            "pdf_sha256": sha(output_dir / f"fig8{letter}.pdf"), "legend_outside_axes": True,
            "no_clipped_text": True, "no_tick_overlap": True, "panel_letters": False,
            "text_boxes": boxes}


def main():
    before = inputs()
    rows = read_data()
    style()
    panels = [draw(letter, metric, rows) for letter, metric in (("a", "cv"), ("b", "Dv"))]
    assert before == inputs()
    (HERE / "analysis").mkdir(exist_ok=True)
    path = HERE / "analysis" / "plot_data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "analysis" / "manifest.json").write_text(json.dumps(
        {"inputs": before, "script_sha256": sha(Path(__file__)), "plot_data_sha256": sha(path),
         "panels": panels, "optima_reselected": False, "error_bars_added": False,
         "source_inputs_unchanged": True, "simulations_run": False}, indent=2), encoding="utf-8")
    print("Exported fig8a and fig8b with shared Times typography and external legends.")


if __name__ == "__main__":
    main()
