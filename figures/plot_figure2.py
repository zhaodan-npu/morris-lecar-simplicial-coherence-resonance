"""Restyle the four original Figure 2 conditions without selecting new examples."""

from pathlib import Path
from dataclasses import asdict
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator, NullLocator
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import plot_figure1_dynamics_revised as original

RESULTS = ROOT / "paper_suite_true_simplicial_array_prl_N100_Dv_refined"
SOURCE_FIGURES = RESULTS / "formal_figures"
ANALYSIS = HERE / "analysis"
RED = "#c62828"
BLACK = "#202020"
SIZE = (4.8, 7.4)
DATA_KEYS = ("times", "voltage_frames", "snapshot_v", "per_node_cv", "frequency")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_manifest():
    paths = [ROOT / "Coherence_Resonance.pdf",
             ROOT / "plot_figure1_dynamics_revised.py",
             ROOT / "plot_alpha_noise_cv_panel_like_reference.py",
             ROOT / "morris_lecar_paired_alpha_compare_simplicial.py",
             ROOT / "morris_lecar_simplicial_paper_suite.py",
             RESULTS / "candidate39_main" / "aggregated_metrics.csv",
             SOURCE_FIGURES / "fig1_dynamics_reselected_summary.csv"]
    paths += [SOURCE_FIGURES / f"fig1{c.letter}_dynamics_reselected_symbols.pdf"
              for c in original.CASES]
    return [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in paths]


def load_data(inputs):
    cache = ANALYSIS / "data_manifest.json"
    if cache.exists():
        meta = json.loads(cache.read_text(encoding="utf-8"))
        assert meta["inputs"] == inputs, "Source changed; do not silently reuse this cache."
        items = []
        for case, record in zip(original.CASES, meta["cases"]):
            path = ANALYSIS / f"data_{case.letter}.npz"
            assert sha256(path) == record["cache_sha256"]
            with np.load(path) as data:
                item = {key: data[key].copy() for key in DATA_KEYS}
            item.update(record["scalars"])
            item["case"] = case
            items.append(item)
        return items

    # Replay the existing illustration generator with its original seed and protocol.
    # This is not a new parameter sweep or a replacement for production statistics.
    print("Restoring the four original illustrative realizations (seed 20260509).", flush=True)
    params, items = original.prepare_data(RESULTS, 20260509, 1400.0, 400.0)
    with (SOURCE_FIGURES / "fig1_dynamics_reselected_summary.csv").open(
            encoding="utf-8", newline="") as handle:
        expected = {r["panel"]: r for r in csv.DictReader(handle)}
    pairs = {"snapshot_time": "snapshot_time_ms", "snapshot_mean": "snapshot_mean_mV",
             "snapshot_std": "snapshot_std_mV", "snapshot_q05": "snapshot_q05_mV",
             "snapshot_q95": "snapshot_q95_mV",
             "snapshot_active_fraction": "snapshot_active_fraction",
             "snapshot_score": "snapshot_selection_score", "ensemble_cv": "ensemble_cv",
             "ensemble_spikes": "ensemble_spike_count"}
    records = []
    for item in items:
        case = item["case"]
        for key, field in pairs.items():
            np.testing.assert_allclose(item[key], float(expected[case.letter][field]),
                                       rtol=1e-9, atol=1e-8, err_msg=f"{case.letter}: {key}")
        path = ANALYSIS / f"data_{case.letter}.npz"
        np.savez_compressed(path, **{k: item[k] for k in DATA_KEYS})
        scalars = {k: v for k, v in item.items() if k not in (*DATA_KEYS, "case")}
        records.append({"case": asdict(case), "scalars": scalars,
                        "cache_sha256": sha256(path), "summary_matches_original": True})
        print(f"Panel {case.letter}: original snapshot and summary reproduced.", flush=True)
    cache.write_text(json.dumps({"inputs": inputs, "params": asdict(params),
                                "cases": records}, indent=2), encoding="utf-8")
    return items


def style():
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 16,
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "axes.labelsize": 19, "xtick.labelsize": 16, "ytick.labelsize": 16,
        "axes.linewidth": 1.3, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "xtick.minor.width": 1., "ytick.minor.width": 1.,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": None,
    })


def check_layout(fig, data_axes, annotations):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, (
            artist.get_text(), box, fig.bbox)
        boxes.append({"text": artist.get_text(), "bbox_pixels": list(box.bounds)})
    for text in annotations:
        box = text.get_window_extent(renderer)
        assert all(not box.overlaps(ax.bbox) for ax in data_axes), text.get_text()
    return boxes


def draw(item):
    case = item["case"]
    fig = plt.figure(figsize=SIZE)
    left, width = .19, .57
    top = fig.add_axes([left, .62, width, .285])
    mid = fig.add_axes([left, .36, width, .17])
    bottom = fig.add_axes([left, .10, width, .17])
    cax = fig.add_axes([.80, .63, .025, .265])
    times = item["times"]
    nodes = np.arange(1, 101)
    heat = top.imshow(item["voltage_frames"], origin="lower", aspect="auto",
                      extent=[1, 100, times[0], times[-1]], cmap="turbo",
                      vmin=-45, vmax=35, interpolation="nearest", rasterized=True)
    top.set(xlim=(1, 100), ylim=(0, 1000), yticks=[0, 500, 1000], ylabel=r"$t$ (ms)")
    top.set_xticks([])
    top.axhline(item["snapshot_time"], color=BLACK, linewidth=2.2)
    top.axhline(item["snapshot_time"], color="white", linewidth=1.1, linestyle=(0, (4, 2)))
    colorbar = fig.colorbar(heat, cax=cax, ticks=[-25, 0, 25])
    colorbar.set_label(r"$V$ (mV)", fontsize=18, labelpad=7)
    colorbar.ax.tick_params(labelsize=15, length=4, direction="in")

    mid.plot(nodes, item["snapshot_v"], ".", color=RED, markersize=5)
    mid.axhline(0, color="#888888", linewidth=.8)
    mid.set(xlim=(0, 101), ylim=(-55, 35), yticks=[-40, 0, 30],
            ylabel=r"$V_i(t_s)$ (mV)")
    mid.set_xticks([])
    finite = np.isfinite(item["per_node_cv"])
    bottom.plot(nodes[finite], item["per_node_cv"][finite], ".", color=BLACK, markersize=4.8)
    bottom.axhline(item["ensemble_cv"], color=RED, linewidth=1.8)
    bottom.set(xlim=(0, 101), ylim=(0, 1.25), xticks=[0, 50, 100], yticks=[0, .5, 1.0],
               xlabel=r"Node $i$", ylabel=r"$\mathrm{CV}_i$")
    for ax in (top, mid, bottom):
        ax.grid(False)
        ax.tick_params(which="both", top=True, right=True)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(NullLocator())
    bottom.xaxis.set_minor_locator(AutoMinorLocator(5))

    annotations = [
        fig.text(left, .95, rf"$\alpha={case.alpha:.1f},\quad D_v={case.dv:.2f}$",
                 fontsize=18, ha="left", va="center"),
        fig.text(left, .565, rf"$t_s={item['snapshot_time']:.2f}$ ms",
                 fontsize=16, ha="left", va="center"),
        fig.text(left, .307,
                 rf"$\overline{{\mathrm{{CV}}}}={item['ensemble_cv']:.2f},\quad"
                 rf"N_{{\mathrm{{sp}}}}={item['ensemble_spikes']/1000:.1f}\times10^3$",
                 fontsize=16, ha="left", va="center"),
    ]
    layout = check_layout(fig, (top, mid, bottom), annotations)
    stem = HERE / f"fig2{case.letter}_cvbar"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white",
                metadata={"Title": f"Figure 2 condition {case.letter}",
                          "Subject": "Original data; revised typography and spacing"})
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    np.testing.assert_array_equal(heat.get_array(), item["voltage_frames"])
    plt.close(fig)
    return {"panel": case.letter, "file": stem.name + ".pdf", "page_inches": SIZE,
            "pdf_sha256": sha256(stem.with_suffix(".pdf")),
            "snapshot_time_ms": item["snapshot_time"],
            "finite_node_cvs": int(finite.sum()), "node_count": 100,
            "data_bounds": {"snapshot_min": float(item["snapshot_v"].min()),
                            "snapshot_max": float(item["snapshot_v"].max()),
                            "cv_max": float(np.nanmax(item["per_node_cv"]))},
            "text_inside_page": True, "annotations_outside_data_axes": True,
            "panel_letters": False, "layout": layout}


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    inputs = input_manifest()
    items = load_data(inputs)
    style()
    outputs = [draw(item) for item in items]
    assert inputs == input_manifest(), "An input was modified while plotting."
    (ANALYSIS / "plot_verification.json").write_text(
        json.dumps({"inputs": inputs, "outputs": outputs,
                    "no_new_parameter_sweep": True, "source_pdf_unchanged": True,
                    "plot_script_sha256": sha256(Path(__file__))}, indent=2), encoding="utf-8")
    print("Four separate PDFs and previews exported; source files unchanged.", flush=True)


if __name__ == "__main__":
    main()
