"""Restyle Figure 9 using the original activation/excursion aggregates."""

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "paper_suite_mechanism_supplement_Dv_exp1_full_Dv030" / "mechanism_figures"
sys.path.insert(0, str(HERE.parent / "figure7"))
from plot_figure7 import style, sha, COLORS, LABELS

SPECS = [("a", "1.5", "mean"), ("b", "2.0", "mean"),
         ("c", "1.5", "cv"), ("d", "2.0", "cv")]
OPTIMAL_DV = {("1.5", "control"): .56, ("1.5", "higher_order"): .16,
              ("2.0", "control"): 1., ("2.0", "higher_order"): .30}
DV = [.04, .09, .16, .25, .30, .40, .56, .75, 1., 1.56, 2.25, 4., 9.]


def source_name(letter, alpha, mode):
    return f"exp1_figureA_{letter}_activation_excursion_{mode}_alpha{alpha.replace('.', '')}.pdf"


def inputs():
    paths = [ROOT / "Coherence_Resonance.pdf", ROOT / "plot_mechanism_supplement.py",
             HERE.parent / "figure7" / "plot_figure7.py", SOURCE / "activation_excursion_aggregated.csv"]
    paths.extend(SOURCE / source_name(*spec) for spec in SPECS)
    return [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in paths]


def read_data():
    with (SOURCE / "activation_excursion_aggregated.csv").open(newline="", encoding="utf-8") as handle:
        data = [row for row in csv.DictReader(handle) if row["v_low"] == "-20.0"]
    rows = []
    for letter, alpha, mode in SPECS:
        for label in COLORS:
            group = sorted((row for row in data if row["alpha"] == alpha and row["label"] == label),
                           key=lambda row: float(row["noise_intensity_dv"]))
            np.testing.assert_allclose([float(row["noise_intensity_dv"]) for row in group], DV, rtol=0, atol=1e-12)
            for phase in ("activation", "excursion"):
                metric = f"{phase}_{'mean_mean' if mode == 'mean' else 'cv_mean'}"
                for row in group:
                    rows.append({"panel": letter, "alpha": alpha, "label": label,
                                 "phase": phase, "metric": metric, "Dv": float(row["noise_intensity_dv"]),
                                 "value": float(row[metric]), "v_low": -20.,
                                 "source_n_rows": int(row["n_rows"]),
                                 "source_metric_n": int(row[metric[:-5] + "_n"])})
    assert len(rows) == 208
    assert all(np.isfinite(row["value"]) for row in rows)
    return rows


def draw(spec, rows):
    letter, alpha, mode = spec
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    fig.subplots_adjust(left=.18, right=.97, bottom=.18, top=.73)
    lines, labels, marks = [], [], []
    for label in COLORS:
        color = COLORS[label]
        target = OPTIMAL_DV[(alpha, label)]
        ax.axvline(target, color=color, lw=.85, ls=":", alpha=.60, zorder=0)
        for phase, symbol, marker, linestyle in (("activation", "T_a", "o", "-"),
                                                 ("excursion", "T_e", "s", "--")):
            selected = [r for r in rows if r["panel"] == letter and r["label"] == label and r["phase"] == phase]
            x = np.array([r["Dv"] for r in selected])
            y = np.array([r["value"] for r in selected])
            # Times lacks extensible angle brackets; use its native chevrons for means.
            quantity = rf"${{<}}{symbol}{{>}}$" if mode == "mean" else rf"$\mathrm{{CV}}({symbol})$"
            line, = ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, lw=2.2,
                            ms=5.8, mew=1.2, mec=color, mfc=color)
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y)
            lines.append(line)
            labels.append(f"{LABELS[label]}, {quantity}")
            index = np.flatnonzero(np.isclose(x, target, rtol=0, atol=1e-12)).item()
            ax.plot([target], [y[index]], marker=marker, ms=8., mfc="white", mec=color,
                    mew=1.2, linestyle="none", zorder=5)
            marks.append({"label": label, "phase": phase, "Dv": target, "value": float(y[index])})

    ax.set_xscale("log")
    ax.set_xlim(.035, 10.5)
    # Match the original 5% vertical margins, including its untruncated mean curves.
    values = np.array([r["value"] for r in rows if r["panel"] == letter])
    span = np.ptp(values)
    ylim = [float(values.min() - .05 * span), float(values.max() + .05 * span)]
    ax.set_ylim(ylim)
    ax.set_xticks([.04, .1, .25, .56, 1., 4., 9.], ["0.04", "0.10", "0.25", "0.56", "1", "4", "9"])
    yticks = {"a": [0, 200, 400, 600, 800], "b": [0, 100, 200, 300, 400],
              "c": [.8, 1., 1.2, 1.4, 1.6, 1.8], "d": [2, 3, 4, 5]}[letter]
    ax.set_yticks(yticks)
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=True, right=True)
    ax.grid(False)
    ax.set_xlabel(r"$D_v$")
    ax.set_ylabel(r"${<}T_a{>},\ {<}T_e{>}$" if mode == "mean"
                  else r"$\mathrm{CV}(T_a),\ \mathrm{CV}(T_e)$")
    condition = ax.text(0, 1.025, rf"$\alpha={alpha}$", transform=ax.transAxes, fontsize=18,
                        ha="left", va="bottom")
    legend = fig.legend(lines, labels, ncol=2, loc="upper center", bbox_to_anchor=(.55, .995),
                        handlelength=1.5, handletextpad=.4, columnspacing=.9,
                        borderaxespad=0, labelspacing=.3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box, condition_box = legend.get_window_extent(renderer), condition.get_window_extent(renderer)
    assert not legend_box.overlaps(ax.bbox)
    assert not condition_box.overlaps(ax.bbox)
    assert not condition_box.overlaps(legend_box), (letter, condition_box.bounds, legend_box.bounds)
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
        fig.savefig(HERE / f"fig9{letter}.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)
    return {"panel": letter, "alpha": alpha, "mode": mode, "n_points": 52,
            "source_pdf": str((SOURCE / source_name(*spec)).relative_to(ROOT)),
            "xlim": [.035, 10.5], "ylim": ylim, "optimal_markers": marks,
            "pdf_sha256": sha(HERE / f"fig9{letter}.pdf"), "legend_outside_axes": True,
            "condition_outside_axes": True, "no_clipped_text": True, "no_tick_overlap": True,
            "panel_letters": False, "text_boxes": boxes}


def main():
    before = inputs()
    rows = read_data()
    style()
    panels = [draw(spec, rows) for spec in SPECS]
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
    print("Exported fig9a-d: 208 original values, external legends, original optimum markers.")


if __name__ == "__main__":
    main()
