"""Restyle the four variance-fraction panels without changing their statistics."""

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

SPECS = [("a", "1.5", "control"), ("b", "2.0", "control"),
         ("c", "1.5", "higher_order"), ("d", "2.0", "higher_order")]
CURVES = [("activation", -30., "#2b6cb0", "o", "-"),
          ("activation", -20., "#222222", "o", "-"),
          ("excursion", -20., "#707070", "s", "--"),
          ("activation", -10., "#b22222", "o", "-")]
OPTIMAL_DV = {("1.5", "control"): .56, ("1.5", "higher_order"): .16,
              ("2.0", "control"): 1., ("2.0", "higher_order"): .30}
DV = [.04, .09, .16, .25, .30, .40, .56, .75, 1., 1.56, 2.25, 4., 9.]


def source_name(letter, alpha, network):
    label = "pairwise" if network == "control" else "simplicial"
    return f"exp1_figureB_{letter}_variance_fraction_{label}_alpha{alpha.replace('.', '')}.pdf"


def input_hashes():
    paths = [ROOT / "Coherence_Resonance.pdf", ROOT / "plot_mechanism_supplement.py",
             HERE.parent / "figure7" / "plot_figure7.py", SOURCE / "activation_excursion_aggregated.csv"]
    paths.extend(SOURCE / source_name(*spec) for spec in SPECS)
    return [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in paths]


def read_data():
    with (SOURCE / "activation_excursion_aggregated.csv").open(newline="", encoding="utf-8") as handle:
        original = list(csv.DictReader(handle))
    rows = []
    for letter, alpha, network in SPECS:
        for phase, vlow, color, marker, linestyle in CURVES:
            group = sorted((r for r in original if r["alpha"] == alpha and r["label"] == network
                            and float(r["v_low"]) == vlow), key=lambda r: float(r["noise_intensity_dv"]))
            np.testing.assert_allclose([float(r["noise_intensity_dv"]) for r in group], DV, rtol=0, atol=1e-12)
            metric = f"{phase}_variance_fraction"
            for row in group:
                rows.append({"panel": letter, "alpha": alpha, "network": network,
                             "phase": phase, "v_low": vlow, "Dv": float(row["noise_intensity_dv"]),
                             "value": float(row[f"{metric}_mean"]), "source_metric": f"{metric}_mean",
                             "source_n_rows": int(row["n_rows"]), "source_metric_n": int(row[f"{metric}_n"]),
                             "color": color, "marker": marker, "linestyle": linestyle})
    assert len(rows) == 208
    assert all(np.isfinite(r["value"]) for r in rows)
    return rows


def draw(spec, rows):
    letter, alpha, network = spec
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    fig.subplots_adjust(left=.18, right=.97, bottom=.18, top=.70)
    handles, labels = [], []
    for phase, vlow, color, marker, linestyle in CURVES:
        group = [r for r in rows if r["panel"] == letter and r["phase"] == phase and r["v_low"] == vlow]
        x, y = np.array([r["Dv"] for r in group]), np.array([r["value"] for r in group])
        line, = ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, lw=2.2,
                        ms=5.8, mew=1.2, mec=color, mfc=color)
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), y)
        handles.append(line)
        sub = "a" if phase == "activation" else "e"
        labels.append(rf"$f_{sub},\ V_{{\mathrm{{low}}}}={vlow:.0f}$")
    optimum = OPTIMAL_DV[(alpha, network)]
    ax.axvline(optimum, color=COLORS[network], lw=.85, ls=":", alpha=.60, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(.035, 10.5)
    values = np.array([r["value"] for r in rows if r["panel"] == letter])
    span = np.ptp(values)
    ylim = [float(values.min() - .05 * span), float(values.max() + .05 * span)]
    ax.set_ylim(ylim)
    ax.set_xticks([.04, .1, .25, .56, 1., 4., 9.], ["0.04", "0.10", "0.25", "0.56", "1", "4", "9"])
    ax.set_yticks([0, .2, .4, .6, .8, 1.], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=True, right=True)
    ax.grid(False)
    ax.set_xlabel(r"$D_v$")
    ax.set_ylabel(r"$f_a,\ f_e$")
    condition = ax.text(0, 1.025, rf"{LABELS[network]}, $\alpha={alpha}$", transform=ax.transAxes,
                        fontsize=18, ha="left", va="bottom")
    optimal_text = ax.text(1, 1.025, rf"$D_v^*={optimum:.2f}$", transform=ax.transAxes,
                           fontsize=18, color=COLORS[network], ha="right", va="bottom")
    # Matplotlib fills legend columns first; show the three thresholds in reading order.
    order = [0, 3, 1, 2]
    legend = fig.legend([handles[i] for i in order], [labels[i] for i in order], ncol=2,
                        loc="upper center", bbox_to_anchor=(.55, .995), handlelength=1.5,
                        handletextpad=.4, columnspacing=.9, borderaxespad=0, labelspacing=.3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    header_boxes = [obj.get_window_extent(renderer) for obj in (legend, condition, optimal_text)]
    for index, box in enumerate(header_boxes):
        assert not box.overlaps(ax.bbox)
        assert all(not box.overlaps(other) for other in header_boxes[index + 1:]), (
            letter, [b.bounds for b in header_boxes])
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
        fig.savefig(HERE / f"fig10{letter}.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)
    return {"panel": letter, "alpha": alpha, "network": network, "n_points": 52,
            "source_pdf": str((SOURCE / source_name(*spec)).relative_to(ROOT)),
            "xlim": [.035, 10.5], "ylim": ylim, "optimal_Dv": optimum,
            "pdf_sha256": sha(HERE / f"fig10{letter}.pdf"), "legend_outside_axes": True,
            "condition_and_optimum_outside_axes": True, "no_clipped_text": True,
            "no_tick_overlap": True, "panel_letters": False, "text_boxes": boxes}


def main():
    before = input_hashes()
    rows = read_data()
    style()
    panels = [draw(spec, rows) for spec in SPECS]
    assert before == input_hashes()
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
    print("Exported fig10a-d: 208 original values and four original optimal-noise lines.")


if __name__ == "__main__":
    main()
