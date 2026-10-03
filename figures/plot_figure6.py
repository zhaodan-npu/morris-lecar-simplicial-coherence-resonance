"""Restyle the six Figure 6 heatmaps, retaining source values and color limits."""

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.text import Text
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, LogLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUITE = ROOT / "paper_suite_sigma2_Dv_phase_refined_N100"
sys.path.insert(0, str(ROOT))
from plot_sigma2_noise_phase_diagram import collect, log_edges_from_centers, edges_from_centers

SPECS = [
    ("a", 1.5, "cv_higher_order", "cv", "viridis_r", .2, 1.1, "CV", "white"),
    ("b", 1.5, "spike_higher_order", "spikes", "magma", None, None, r"Spike count ($10^3$)", None),
    ("c", 1.5, "delta_cv", "delta_cv", "coolwarm", -.1, .5, r"$\Delta\mathrm{CV}$", "black"),
    ("d", 2., "cv_higher_order", "cv", "viridis_r", .2, 1.1, "CV", "white"),
    ("e", 2., "spike_higher_order", "spikes", "magma", None, None, r"Spike count ($10^3$)", None),
    ("f", 2., "delta_cv", "delta_cv", "coolwarm", -.1, .5, r"$\Delta\mathrm{CV}$", "black"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes():
    paths = [ROOT / "Coherence_Resonance.pdf", ROOT / "plot_sigma2_noise_phase_diagram.py",
             SUITE / "phase_figures" / "sigma2_noise_phase_summary.json"]
    paths.extend(sorted(SUITE.glob("sigma2_*/aggregated_metrics.csv")))
    paths.extend(SUITE / "phase_figures" / f"sigma2_noise_phase_alpha_{s[1]:g}_{s[3]}_prlstyle.pdf"
                 for s in SPECS)
    return [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in paths]


def style():
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 18,
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "axes.labelsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16,
        "axes.linewidth": 1.4, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "xtick.minor.width": 1., "ytick.minor.width": 1.,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": None,
    })


def draw(spec, matrix, sigma, dv, ridge, arrays):
    letter, alpha, metric, _, cmap_name, vmin, vmax, cbar_label, ridge_color = spec
    xedges, yedges = log_edges_from_centers(dv), edges_from_centers(sigma)
    values = np.ma.masked_invalid(matrix)
    norm = Normalize(vmin=vmin, vmax=vmax)
    norm.autoscale_None(values)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#eeeeee")
    fig = plt.figure(figsize=(5.6, 4.2))
    ax = fig.add_axes([.16, .18, .58, .70])
    cax = fig.add_axes([.775, .18, .033, .70])
    # Face-colored edges suppress PDF viewer hairlines without interpolating cells.
    mesh = ax.pcolormesh(xedges, yedges, values, cmap=cmap, norm=norm, shading="flat",
                         edgecolors="face", linewidth=.35, antialiased=False)
    np.testing.assert_array_equal(mesh.get_array().data, values.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(mesh.get_array()), np.ma.getmaskarray(values))
    if ridge_color:
        line, = ax.plot(ridge, sigma, color=ridge_color, marker=".", lw=2., ms=5., mew=1.)
        np.testing.assert_array_equal(line.get_xdata(), ridge)
        np.testing.assert_array_equal(line.get_ydata(), sigma)
    ax.set_xscale("log")
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])
    ax.set_xticks([.04, .25, 1, 4, 9], ["0.04", "0.25", "1", "4", "9"])
    ax.set_yticks([0, .04, .08, .12, .16], ["0.00", "0.04", "0.08", "0.12", "0.16"])
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * .1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.set_xlabel(r"$D_v$")
    ax.set_ylabel(r"$\sigma_2$")
    ax.tick_params(which="both", top=True, right=True)
    ax.grid(False)
    condition = fig.text(.16, .95, rf"$\alpha={alpha:.1f}$", ha="left", va="center", fontsize=18)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(cbar_label, labelpad=8)
    if metric == "spike_higher_order":
        cbar.set_ticks([20000, 40000, 60000] if alpha == 1.5 else [10000, 20000, 30000, 40000])
        cbar.formatter = FuncFormatter(lambda value, position: f"{value / 1000:g}")
    elif metric == "cv_higher_order":
        cbar.set_ticks([.2, .4, .6, .8, 1.])
    else:
        cbar.set_ticks([-.1, 0, .1, .2, .3, .4, .5])
    cbar.update_ticks()
    cbar.ax.tick_params(which="both", direction="in", width=1.2, length=5, labelsize=16)
    cbar.outline.set_linewidth(1.2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert not condition.get_window_extent(renderer).overlaps(ax.bbox)
    boxes = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, (
            letter, text.get_text(), box.bounds)
        boxes.append({"text": text.get_text(), "bbox_pixels": list(box.bounds)})
    for axis in (ax.xaxis, ax.yaxis, cax.yaxis):
        ticks = axis.get_ticklabels()
        for first, second in zip(ticks, ticks[1:]):
            assert not first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer))
    label_box = cax.yaxis.label.get_window_extent(renderer)
    assert not any(label_box.overlaps(t.get_window_extent(renderer)) for t in cax.get_yticklabels())
    for suffix in ("pdf", "png"):
        fig.savefig(HERE / f"fig6{letter}.{suffix}", dpi=600, facecolor="white")
    arrays[f"{letter}_values"] = matrix
    arrays[f"{letter}_mask"] = np.ma.getmaskarray(values)
    arrays[f"{letter}_rgb"] = cmap(norm(values))[:, :, :3]
    arrays[f"{letter}_ridge_dv"] = ridge if ridge_color else np.array([])
    plt.close(fig)
    return {"panel": letter, "alpha": alpha, "metric": metric, "cmap": cmap_name,
            "vmin": float(norm.vmin), "vmax": float(norm.vmax),
            "masked_cells": int(np.ma.getmaskarray(values).sum()), "cells": int(matrix.size),
            "ridge_color": ridge_color, "spike_tick_divisor": 1000 if metric == "spike_higher_order" else 1,
            "no_text_clipped": True, "no_tick_overlap": True, "condition_outside_axes": True,
            "panel_letters": False, "pdf_sha256": sha(HERE / f"fig6{letter}.pdf"), "text_boxes": boxes}


def main():
    before = input_hashes()
    sigma, dv, alphas, data, summary = collect(SUITE, min_spikes=100.)
    assert alphas == [1.5, 2.]
    assert len(sigma) == 17 and len(dv) == 20
    saved = json.loads((SUITE / "phase_figures" / "sigma2_noise_phase_summary.json").read_text())
    assert len(saved) == len(summary) == 34
    for original, current in zip(saved, summary):
        assert set(original) == set(current)
        for key in original:
            np.testing.assert_allclose(original[key], current[key], rtol=0, atol=1e-12, equal_nan=True)
    style()
    sigma, dv = np.array(sigma), np.array(dv)
    arrays = {"sigma": sigma, "dv": dv, "xedges": log_edges_from_centers(dv),
              "yedges": edges_from_centers(sigma)}
    records = []
    for spec in SPECS:
        alpha, metric = spec[1:3]
        ridge = np.array([r["best_higher_order_Dv"] for r in summary if r["alpha"] == alpha])
        records.append(draw(spec, data[alpha][metric], sigma, dv, ridge, arrays))
    assert before == input_hashes()
    (HERE / "analysis").mkdir(exist_ok=True)
    np.savez_compressed(HERE / "analysis" / "plot_data.npz", **arrays)
    (HERE / "analysis" / "manifest.json").write_text(json.dumps(
        {"inputs": before, "script_sha256": sha(Path(__file__)), "min_spikes": 100,
         "panels": records, "source_data_unchanged": True, "simulations_run": False,
         "interpolation": False, "plot_data_sha256": sha(HERE / "analysis" / "plot_data.npz")},
        indent=2), encoding="utf-8")
    print("Exported six panels; original summary ridges reproduced and all input hashes unchanged.")


if __name__ == "__main__":
    main()
