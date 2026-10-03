"""Restyle Figure 11 while preserving the original dual-axis data and limits."""

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, FuncFormatter, LogLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "paper_suite_mechanism_supplement_Dv_exp1_full_Dv030" / "mechanism_figures"
sys.path.insert(0, str(HERE.parent / "figure7"))
from plot_figure7 import style, sha, COLORS, LABELS

SPECS = [("a", "1.5", "control"), ("b", "2.0", "control"),
         ("c", "1.5", "higher_order"), ("d", "2.0", "higher_order")]
CURVES = [("stcc_mean", "left", "#222222", "o", "-", r"$\mathrm{STCC}$"),
          ("simplex_stcc_mean", "left", "#c62828", "s", "--", r"$\mathrm{STCC}_{\triangle}$"),
          ("synchrony_index", "right", "#2b6cb0", "^", "-.", r"$\chi$")]
OPTIMAL_DV = {("1.5", "control"): .56, ("1.5", "higher_order"): .16,
              ("2.0", "control"): 1., ("2.0", "higher_order"): .30}
DV = [.04, .09, .16, .25, .30, .40, .56, .75, 1., 1.56, 2.25, 4., 9.]


def source_name(letter, alpha, network):
    label = "pairwise" if network == "control" else "simplicial"
    return f"exp2_figureAB_{letter}_stcc_chi_{label}_alpha{alpha.replace('.', '')}.pdf"


def input_hashes():
    paths = [ROOT / "Coherence_Resonance.pdf", ROOT / "plot_mechanism_supplement.py",
             HERE.parent / "figure7" / "plot_figure7.py", SOURCE / "binned_synchrony_aggregated.csv"]
    paths.extend(SOURCE / source_name(*spec) for spec in SPECS)
    return [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in paths]


def read_data():
    with (SOURCE / "binned_synchrony_aggregated.csv").open(newline="", encoding="utf-8") as handle:
        original = [r for r in csv.DictReader(handle) if r["bin_size"] == "10.0"]
    rows = []
    for letter, alpha, network in SPECS:
        group = sorted((r for r in original if r["alpha"] == alpha and r["label"] == network),
                       key=lambda r: float(r["noise_intensity_dv"]))
        np.testing.assert_allclose([float(r["noise_intensity_dv"]) for r in group], DV, rtol=0, atol=1e-12)
        for metric, side, color, marker, linestyle, _ in CURVES:
            for row in group:
                rows.append({"panel": letter, "alpha": alpha, "network": network,
                             "metric": metric, "axis": side, "Dv": float(row["noise_intensity_dv"]),
                             "value": float(row[f"{metric}_mean"]), "bin_size": 10.,
                             "source_n_rows": int(row["n_rows"]), "source_metric_n": int(row[f"{metric}_n"]),
                             "color": color, "marker": marker, "linestyle": linestyle})
    assert len(rows) == 156
    missing = [(r["panel"], r["metric"], r["Dv"], r["source_metric_n"])
               for r in rows if not np.isfinite(r["value"])]
    assert missing == [("d", "simplex_stcc_mean", .04, 0), ("d", "simplex_stcc_mean", .09, 0)]
    return rows


def limits(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    span = np.ptp(values)
    return [float(values.min() - .05 * span), float(values.max() + .05 * span)]


def draw(spec, rows):
    letter, alpha, network = spec
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    fig.subplots_adjust(left=.18, right=.82, bottom=.18, top=.75)
    right = ax.twinx()
    handles = []
    for metric, side, color, marker, linestyle, label in CURVES:
        group = [r for r in rows if r["panel"] == letter and r["metric"] == metric]
        x, y = np.array([r["Dv"] for r in group]), np.array([r["value"] for r in group])
        target = ax if side == "left" else right
        line, = target.plot(x, y, color=color, marker=marker, linestyle=linestyle,
                            lw=2.2, ms=5.8, mew=1.2, mec=color, mfc=color, label=label)
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), y)
        handles.append(line)
    optimum = OPTIMAL_DV[(alpha, network)]
    ax.axvline(optimum, color=COLORS[network], lw=.85, ls=":", alpha=.60, zorder=0)
    ylims = {side: limits([r["value"] for r in rows if r["panel"] == letter and r["axis"] == side])
             for side in ("left", "right")}
    if network == "control":
        # Preserve the original fixed right-axis range used for small chi variations.
        ylims["right"] = [.0995, .1025]
    ax.set_xscale("log")
    ax.set_xlim(.035, 10.5)
    ax.set_ylim(ylims["left"])
    right.set_ylim(ylims["right"])
    left_ticks = {"a": [0, .0004, .0008, .0012], "b": [-.0015, 0, .0015],
                  "c": [0, .15, .30, .45], "d": [0, .15, .30, .45]}[letter]
    right_ticks = {"a": [.100, .101, .102], "b": [.100, .101, .102],
                   "c": [.2, .4], "d": [.2, .4, .6]}[letter]
    ax.set_yticks(left_ticks)
    right.set_yticks(right_ticks)
    if network == "control":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value * 1000:.1f}"))
        ax.set_ylabel(r"STCC ($10^{-3}$)")
        right.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_ylabel("STCC")
        right.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel(r"$D_v$")
    right.set_ylabel(r"$\chi$", color="#2b6cb0", labelpad=5)
    ax.set_xticks([.04, .1, .25, .56, 1., 4., 9.], ["0.04", "0.10", "0.25", "0.56", "1", "4", "9"])
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    for axis in (ax, right):
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        axis.grid(False)
    ax.tick_params(axis="x", which="both", top=True)
    ax.tick_params(axis="y", which="both", left=True, right=False)
    right.tick_params(axis="y", which="both", left=False, right=True, colors="#2b6cb0")
    right.spines["right"].set_color("#2b6cb0")
    condition = ax.text(0, 1.025, rf"{LABELS[network]}, $\alpha={alpha}$", transform=ax.transAxes,
                        fontsize=18, ha="left", va="bottom")
    optimal_text = ax.text(1, 1.025, rf"$D_v^*={optimum:.2f}$", transform=ax.transAxes,
                           fontsize=18, color=COLORS[network], ha="right", va="bottom")
    legend = fig.legend(handles, [line.get_label() for line in handles], ncol=3,
                        loc="upper center", bbox_to_anchor=(.5, .995), handlelength=1.5,
                        handletextpad=.4, columnspacing=.9, borderaxespad=0)
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
    for axis in (ax.xaxis, ax.yaxis, right.yaxis):
        ticks = axis.get_ticklabels()
        for first, second in zip(ticks, ticks[1:]):
            assert not first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer))
    for suffix in ("pdf", "png"):
        fig.savefig(HERE / f"fig11{letter}.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)
    return {"panel": letter, "alpha": alpha, "network": network, "n_grid_points": 39,
            "n_points": sum(np.isfinite(r["value"]).item() for r in rows if r["panel"] == letter),
            "source_pdf": str((SOURCE / source_name(*spec)).relative_to(ROOT)),
            "xlim": [.035, 10.5], "ylim_left": ylims["left"], "ylim_right": ylims["right"],
            "left_tick_values": left_ticks, "right_tick_values": right_ticks, "optimal_Dv": optimum,
            "pdf_sha256": sha(HERE / f"fig11{letter}.pdf"), "legend_outside_axes": True,
            "condition_and_optimum_outside_axes": True, "no_clipped_text": True,
            "no_tick_overlap": True, "panel_letters": False, "text_boxes": boxes}


def main():
    before = input_hashes()
    rows = read_data()
    style()
    # Times does not include the hollow triangle; keep the original symbol with STIX.
    plt.rcParams["mathtext.fallback"] = "stix"
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
         "source_inputs_unchanged": True, "simulations_run": False,
         "math_font_exception": "STIX hollow triangle only"}, indent=2), encoding="utf-8")
    print("Exported fig11a-d: 154 finite values, two preserved missing values, unchanged dual-axis ranges.")


if __name__ == "__main__":
    main()
