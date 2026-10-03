"""Restyle the six panels of current Figure 3 using verified source values."""

from pathlib import Path
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COLORS = {"control": "#36454f", "higher_order": "#c62828"}
MARKERS = {"control": "o", "higher_order": "s"}
LABELS = {"control": "Pairwise", "higher_order": "Simplicial"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_style():
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 18,
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "axes.labelsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 16, "legend.frameon": False,
        "axes.linewidth": 1.4, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "xtick.minor.width": 1., "ytick.minor.width": 1.,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": None,
    })


def draw(spec, rows):
    letter, alpha = spec["panel"], spec["alpha"]
    divisor, metric = spec["display_divisor"], spec["metric"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    fig.subplots_adjust(left=.18, right=.97, bottom=.18, top=.80)
    handles = []
    for label in COLORS:
        group = [r for r in rows if r["panel"] == letter and r["label"] == label]
        x = np.array([float(r["D_v"]) for r in group])
        y = np.array([float(r["mean"]) for r in group]) / divisor
        lower = np.array([float(r["lower"]) for r in group]) / divisor
        upper = np.array([float(r["upper"]) for r in group]) / divisor
        container = ax.errorbar(
            x, y, yerr=np.array([y - lower, upper - y]),
            color=COLORS[label], marker=MARKERS[label], markersize=5.5,
            linewidth=2., markeredgewidth=1.1, capsize=3, elinewidth=1.2,
            label=LABELS[label], zorder=3)
        np.testing.assert_array_equal(container.lines[0].get_ydata(), y)
        segments = np.array(container.lines[2][0].get_segments())
        np.testing.assert_allclose(segments[:, 0, 1], lower, rtol=0, atol=1e-14)
        np.testing.assert_allclose(segments[:, 1, 1], upper, rtol=0, atol=1e-14)
        handles.append(container)
        optimum = spec["optima"][label]["dv"]
        selected = np.flatnonzero(np.isclose(x, optimum, rtol=0, atol=1e-12))
        assert len(selected) == 1
        if metric == "isi_cv":
            # Retain best_row_for's original mean spike-count eligibility rule.
            spike_counts = {float(r["D_v"]): float(r["mean"]) for r in rows
                            if r["metric"] == "spike_count_total"
                            and float(r["alpha"]) == alpha and r["label"] == label}
            eligible = np.array([spike_counts[dv] >= 1000 for dv in x]) & np.isfinite(y)
            candidates = np.flatnonzero(eligible)
            assert selected[0] == candidates[np.argmin(y[candidates])]
            np.testing.assert_allclose(y[selected[0]], spec["optima"][label]["cv"],
                                       rtol=0, atol=1e-12)
        ax.axvline(optimum, color=COLORS[label], linestyle=":", linewidth=1.2, alpha=.75)
        ax.plot([optimum], [y[selected[0]]], marker=MARKERS[label],
                markersize=8., markerfacecolor="white", markeredgecolor=COLORS[label],
                markeredgewidth=1.4, linestyle="none", zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(.035, 10.5)
    ax.set_xticks([.04, .1, .3, 1., 4., 9.])
    ax.set_xticklabels(["0.04", "0.10", "0.30", "1", "4", "9"])
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * .1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, spec["y_max"] / divisor)
    ax.set_xlabel(r"$D_v$")
    if metric == "isi_cv":
        ax.set_ylabel("CV")
        ax.set_yticks([0, .5, 1., 1.5, 2.])
    elif metric == "spike_count_total":
        ax.set_ylabel(r"Spike count ($10^3$)")
        ax.set_yticks([0, 20, 40, 60])
    else:
        ax.set_ylabel(r"$t_{\mathrm{cor}}$ (s)")
        ax.set_yticks([0, .4, .8, 1.2] if alpha == 1.5 else [0, 1, 2, 3, 4])
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=True, right=True)
    ax.grid(False)
    header_offset = .025 if metric == "isi_cv" else 0.
    condition = fig.text(.18, .913 + header_offset, rf"$\alpha={alpha:.1f}$",
                         ha="left", va="center", fontsize=18)
    legend = fig.legend(handles, [LABELS[k] for k in COLORS], loc="upper right",
                        bbox_to_anchor=(.98, .96 + header_offset), ncol=2, handlelength=1.5,
                        handletextpad=.4, columnspacing=.9, borderaxespad=0)
    optimum_notes = []
    if metric == "isi_cv":
        for label, position in zip(COLORS, (.18, .59)):
            optimum = spec["optima"][label]
            note = (rf"$\mathrm{{CV}}_{{\mathrm{{opt}}}}={optimum['cv']:.3f}"
                    rf"\ (D_v={optimum['dv']:.2f})$")
            optimum_notes.append(fig.text(position, .847, note, fontsize=15,
                                          color=COLORS[label], ha="left", va="center"))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_boxes = []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, (
            letter, artist.get_text(), box)
        text_boxes.append({"text": artist.get_text(), "bbox_pixels": list(box.bounds)})
    lb, cb = legend.get_window_extent(renderer), condition.get_window_extent(renderer)
    assert not lb.overlaps(ax.bbox) and not cb.overlaps(ax.bbox)
    assert not lb.overlaps(cb), (letter, lb, cb)
    note_boxes = [note.get_window_extent(renderer) for note in optimum_notes]
    for box in note_boxes:
        assert not any(box.overlaps(other) for other in (ax.bbox, lb, cb)), (
            letter, box.bounds, ax.bbox.bounds, lb.bounds, cb.bounds)
    for first, second in zip(note_boxes, note_boxes[1:]):
        assert not first.overlaps(second)
    ticks = ax.get_xticklabels()
    for a, b in zip(ticks, ticks[1:]):
        assert not a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer))
    for suffix in ("pdf", "png"):
        fig.savefig(HERE / f"fig3{letter}.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)
    return {"panel": letter, "pdf_sha256": digest(HERE / f"fig3{letter}.pdf"),
            "legend_outside_data": True, "condition_outside_data": True,
            "no_text_clipped": True, "no_tick_text_overlap": True,
            "text_boxes": text_boxes, "panel_letters": False,
            "optimum_values_outside_data": bool(optimum_notes),
            "optimum_min_mean_spike_count": 1000,
            "script_sha256": digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", choices=list("abcdef"), default=list("abcdef"))
    selected = set(parser.parse_args().panels)
    manifest = json.loads((HERE / "analysis" / "manifest.json").read_text())
    for record in manifest["inputs"]:
        assert digest(ROOT / record["path"]) == record["sha256"]
    with (HERE / "analysis" / "plot_data.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    apply_style()
    layout_path = HERE / "analysis" / "layout_verification.json"
    records = {}
    if layout_path.exists():
        previous = json.loads(layout_path.read_text())
        records = {r["panel"]: r for r in previous["panels"]}
        for record in records.values():
            record.setdefault("script_sha256", previous["script_sha256"])
    unchanged = {path: digest(path) for letter in "abcdef" if letter not in selected
                 for suffix in ("pdf", "png")
                 if (path := HERE / f"fig3{letter}.{suffix}").exists()}
    for spec in manifest["panels"]:
        if spec["panel"] in selected:
            records[spec["panel"]] = draw(spec, rows)
    assert all(digest(path) == value for path, value in unchanged.items())
    layout_path.write_text(json.dumps(
        {"plot_data_sha256": digest(HERE / "analysis" / "plot_data.csv"),
         "script_sha256": digest(Path(__file__)),
         "panels": [records[key] for key in sorted(records)]}, indent=2), encoding="utf-8")
    print(f"Exported Figure 3 panels: {', '.join(sorted(selected))}. Unselected files unchanged.")


if __name__ == "__main__":
    main()
