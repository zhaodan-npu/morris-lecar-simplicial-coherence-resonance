"""Restyle current Figure 5 without changing summary values or curve encodings."""

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
COLORS = {"control": "#36454f", "higher_order": "#c62828"}
LABELS = {"control": "Pairwise", "higher_order": "Simplicial"}
MARKERS = {"control": "o", "higher_order": "s"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def draw(panel, rows, output_dir=None):
    output_dir = HERE if output_dir is None else Path(output_dir)
    metric = "cv" if panel == "a" else "dv"
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    fig.subplots_adjust(left=.18, right=.97, bottom=.18, top=.78)
    lines = []
    for alpha in (1.5, 2.0):
        for label in COLORS:
            group = sorted((r for r in rows if float(r["alpha"]) == alpha and r["label"] == label),
                           key=lambda r: float(r["sigma2"]))
            x = np.array([float(r["sigma2"]) for r in group])
            y = np.array([float(r[metric]) for r in group])
            line, = ax.plot(x, y, color=COLORS[label], marker=MARKERS[label],
                            linestyle="--" if label == "control" else "-", linewidth=2.2,
                            markersize=5.8, markeredgewidth=1.2, markeredgecolor=COLORS[label],
                            markerfacecolor=COLORS[label] if alpha == 1.5 else "white",
                            label=rf"{LABELS[label]}, $\alpha={alpha:.1f}$")
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y)
            lines.append(line)
    ax.set_xlim(-.005, .165)
    ax.set_xticks([0, .04, .08, .12, .16], ["0", "0.04", "0.08", "0.12", "0.16"])
    ax.set_xlabel(r"$\sigma_2$")
    ax.set_ylabel(r"$\mathrm{CV}^{*}$" if panel == "a" else r"$D_v^*$")
    ax.set_ylim(0, .8 if panel == "a" else 1.2)
    ax.set_yticks([0, .2, .4, .6, .8] if panel == "a" else [0, .3, .6, .9, 1.2])
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=True, right=True)
    ax.grid(False)
    legend = fig.legend(lines, [line.get_label() for line in lines], ncol=2,
                        loc="upper center", bbox_to_anchor=(.51, .995),
                        handlelength=1.6, handletextpad=.45, columnspacing=1.,
                        labelspacing=.4, borderaxespad=0.)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    assert not legend_box.overlaps(ax.bbox)
    assert legend_box.x0 >= 0 and legend_box.x1 <= fig.bbox.x1
    boxes = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, (
            text.get_text(), box.bounds)
        boxes.append({"text": text.get_text(), "bbox_pixels": list(box.bounds)})
    for axis in (ax.xaxis, ax.yaxis):
        ticks = axis.get_ticklabels()
        for first, second in zip(ticks, ticks[1:]):
            assert not first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer))
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"fig5{panel}.{suffix}", dpi=600, facecolor="white")
    plt.close(fig)
    return {"panel": panel, "pdf_sha256": sha(output_dir / f"fig5{panel}.pdf"),
            "legend_outside_axes": True, "no_clipped_text": True,
            "no_tick_overlap": True, "panel_letters": False, "n_values": 68,
            "error_bars_added": False, "text_boxes": boxes}


def main():
    manifest = json.loads((HERE / "analysis" / "manifest.json").read_text())
    for record in manifest["inputs"]:
        assert sha(ROOT / record["path"]) == record["sha256"]
    with (HERE / "analysis" / "plot_data.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    style()
    results = [draw(panel, rows) for panel in "ab"]
    (HERE / "analysis" / "layout_verification.json").write_text(json.dumps(
        {"script_sha256": sha(Path(__file__)), "plot_data_sha256": sha(HERE / "analysis" / "plot_data.csv"),
         "panels": results}, indent=2), encoding="utf-8")
    print("Exported fig5a and fig5b as separate PDF and 600 dpi PNG files.")


if __name__ == "__main__":
    main()
