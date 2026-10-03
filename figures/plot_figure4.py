"""Move Figure 4 legends outside the axes without changing estimates or masks."""

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL = HERE.parent / "comment2"
sys.path.insert(0, str(ORIGINAL))
from redraw_comment2_cv import read_data, LABELS, COLORS, MARKERS, DISPLAY


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    paths = [ROOT / "Coherence_Resonance.pdf", ORIGINAL / "redraw_comment2_cv.py",
             ORIGINAL / "analyze_comment2.py", ORIGINAL / "analysis" / "curves.csv",
             ORIGINAL / "analysis" / "summary.json", ORIGINAL / "analysis" / "input_manifest.json"]
    paths.extend(ORIGINAL / "figures" / f"fig_R1_cv_alpha{alpha}_revised.pdf" for alpha in (15, 20))
    return [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in paths]


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


def draw(panel, alpha, rows):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    fig.subplots_adjust(left=.18, right=.97, bottom=.18, top=.75)
    handles, plotted, incomplete = [], [], []
    for label in LABELS:
        group = sorted((r for r in rows if r["alpha"] == alpha and r["label"] == label),
                       key=lambda r: r["Dv"])
        x = np.array([r["Dv"] for r in group])
        y = np.array([r["CV_mean"] for r in group])
        eligible = np.array([r["eligible_for_minimum"] for r in group])
        complete = np.array([r["valid_cv_count"] == r["total_paths"] for r in group])
        color, marker = COLORS[label], MARKERS[label]
        ls = "--" if label == "control" else "-"
        # Retain the original exclusions and do not connect across excluded points.
        line, = ax.plot(x, np.where(eligible, y, np.nan), color=color, ls=ls, lw=2.2, zorder=2)
        np.testing.assert_array_equal(line.get_ydata(), np.where(eligible, y, np.nan))
        for mask, fill in ((eligible & complete, color), (eligible & ~complete, "white")):
            ax.plot(x[mask], y[mask], ls="none", marker=marker, ms=5.8,
                    mfc=fill, mec=color, mew=1.2, zorder=4)
        lo = np.array([r["CV_ci_low"] for r in group])
        hi = np.array([r["CV_ci_high"] for r in group])
        bars = ax.vlines(x[eligible], lo[eligible], hi[eligible], color=color, lw=1.1, zorder=3)
        segments = np.array(bars.get_segments())
        np.testing.assert_array_equal(segments[:, 0, 1], lo[eligible])
        np.testing.assert_array_equal(segments[:, 1, 1], hi[eligible])
        ax.plot(x[eligible], lo[eligible], "_", color=color, ms=5)
        ax.plot(x[eligible], hi[eligible], "_", color=color, ms=5)
        handles.append(Line2D([], [], color=color, ls=ls, lw=2.2, marker=marker,
                              ms=5.8, label=DISPLAY[label]))
        plotted.extend({"label": label, "Dv": r["Dv"], "CV_mean": r["CV_mean"]}
                       for r in group if r["eligible_for_minimum"])
        incomplete.extend({"label": label, "Dv": r["Dv"], "valid_cv_count": r["valid_cv_count"],
                           "total_paths": r["total_paths"]} for r in group
                          if r["eligible_for_minimum"] and r["valid_cv_count"] < r["total_paths"])
    ax.set_xscale("log")
    ax.set_xlim(.035, 10.5)
    ax.set_xticks([.04, .1, .3, 1, 4, 9], ["0.04", "0.10", "0.30", "1", "4", "9"])
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * .1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 2.1)
    ax.set_yticks([0, .5, 1, 1.5, 2])
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel(r"$D_v$")
    ax.set_ylabel("CV")
    ax.tick_params(which="both", top=True, right=True)
    ax.grid(False)
    condition = fig.text(.18, .92, rf"$\alpha={alpha:.1f}$", va="center", fontsize=18)
    legend = fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.985, .995),
                        borderaxespad=0, handlelength=1.5, handletextpad=.5, labelspacing=.25)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lb, cb = legend.get_window_extent(renderer), condition.get_window_extent(renderer)
    assert not lb.overlaps(ax.bbox) and not cb.overlaps(ax.bbox) and not lb.overlaps(cb)
    boxes = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, (
            text.get_text(), box.bounds)
        boxes.append({"text": text.get_text(), "bbox_pixels": list(box.bounds)})
    ticks = ax.get_xticklabels()
    for first, second in zip(ticks, ticks[1:]):
        assert not first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer))
    for suffix in ("pdf", "png"):
        fig.savefig(HERE / f"fig4{panel}.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)
    return {"panel": panel, "alpha": alpha, "pdf_sha256": sha(HERE / f"fig4{panel}.pdf"),
            "legend_outside_axes": True, "condition_outside_axes": True,
            "no_clipped_text": True, "no_tick_overlap": True, "panel_letters": False,
            "plotted": plotted, "incomplete": incomplete, "text_boxes": boxes}


def main():
    before = inputs()
    rows = read_data()
    style()
    panels = [draw(panel, alpha, rows) for panel, alpha in (("a", 1.5), ("b", 2.))]
    assert [len(p["plotted"]) for p in panels] == [58, 52]
    assert [(r["Dv"], r["valid_cv_count"]) for r in panels[1]["incomplete"]] == [
        (.09, 2143), (.12, 4047), (.16, 4311)]
    assert not panels[0]["incomplete"]
    assert before == inputs()
    (HERE / "analysis").mkdir(exist_ok=True)
    (HERE / "analysis" / "layout_verification.json").write_text(json.dumps(
        {"inputs": before, "script_sha256": sha(Path(__file__)), "panels": panels,
         "source_data_unchanged": True, "error_bars_recomputed": False,
         "legend_labels": DISPLAY, "activity_rule": "Original mean total network spikes >= 1000"},
        indent=2), encoding="utf-8")
    print("Exported fig4a and fig4b with external legends; 110 means and confidence intervals unchanged.")


if __name__ == "__main__":
    main()
