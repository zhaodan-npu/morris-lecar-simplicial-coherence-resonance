from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import morris_lecar_paired_alpha_compare_simplicial as base
import morris_lecar_simplicial_paper_suite as suite
from plot_alpha_noise_cv_panel_like_reference import load_aggregated, lookup, simulate_node_panel


@dataclass(frozen=True)
class Case:
    letter: str
    alpha: float
    dv: float


CASES = (
    Case("a", 1.5, 0.25),
    Case("b", 1.5, 0.75),
    Case("c", 2.0, 0.25),
    Case("d", 2.0, 1.00),
)

RED = "#c62828"
BLACK = "#202020"
GRAY = "#666666"
SNAPSHOT_QUANTILES = np.array([0.05, 0.25, 0.50, 0.75, 0.95])


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 1.25,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.minor.size": 2.3,
            "ytick.minor.size": 2.3,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: mpl.axes.Axes, *, minor: bool = True) -> None:
    ax.grid(False)
    if minor:
        ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)


def select_representative_snapshot(
    times: np.ndarray,
    voltage_frames: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Select a typical frame by one common, phase-agnostic rule.

    Each frame is represented by five spatial voltage quantiles.  The selected
    frame is the one closest to the component-wise median quantile vector,
    after robust scaling.  The first and last 10% of the retained window are
    excluded to avoid boundary effects.  This replaces the old arbitrary
    midpoint snapshot and applies exactly the same rule to every condition.
    """
    frame_quantiles = np.quantile(voltage_frames, SNAPSHOT_QUANTILES, axis=1).T
    target = np.median(frame_quantiles, axis=0)
    robust_scale = np.quantile(frame_quantiles, 0.75, axis=0) - np.quantile(
        frame_quantiles, 0.25, axis=0
    )
    robust_scale = np.maximum(robust_scale, 1.0)
    scores = np.sum(((frame_quantiles - target) / robust_scale) ** 2, axis=1)

    window = float(times[-1] - times[0])
    valid = (times >= times[0] + 0.10 * window) & (times <= times[-1] - 0.10 * window)
    valid_indices = np.flatnonzero(valid)
    selected_index = int(valid_indices[np.argmin(scores[valid])])
    return selected_index, scores


def save_pair(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def prepare_data(result_root: Path, seed: int, trace_total: float, trace_warmup: float):
    rows = load_aggregated(result_root / "candidate39_main" / "aggregated_metrics.csv")
    config = suite.PaperSuiteConfig(
        n_nodes=100,
        n_networks=1,
        n_trials_per_network=1,
        t_total=trace_total,
        t_warmup=trace_warmup,
        base_seed=seed,
    )
    params = suite.candidate39_params(config)
    network = base.build_simplicial_network(
        params,
        np.random.default_rng(params.base_seed + base.stable_seed("paper_network", 0)),
    )

    prepared = []
    for case in CASES:
        scale = base.noise_scale_from_voltage_noise_intensity(case.dv, case.alpha, params.capacitance)
        times, voltage_frames, _, per_node_cv, frequency = simulate_node_panel(
            params,
            network,
            case.alpha,
            scale,
            seed,
        )
        snapshot_index, snapshot_scores = select_representative_snapshot(times, voltage_frames)
        snapshot_v = voltage_frames[snapshot_index].copy()
        prepared.append(
            {
                "case": case,
                "times": times,
                "voltage_frames": voltage_frames,
                "snapshot_index": snapshot_index,
                "snapshot_time": float(times[snapshot_index]),
                "snapshot_v": snapshot_v,
                "snapshot_score": float(snapshot_scores[snapshot_index]),
                "snapshot_mean": float(np.mean(snapshot_v)),
                "snapshot_std": float(np.std(snapshot_v)),
                "snapshot_q05": float(np.quantile(snapshot_v, 0.05)),
                "snapshot_q95": float(np.quantile(snapshot_v, 0.95)),
                "snapshot_active_fraction": float(np.mean(snapshot_v > params.spike_threshold)),
                "per_node_cv": per_node_cv,
                "frequency": frequency,
                "ensemble_cv": lookup(rows, case.alpha, case.dv, "higher_order", "isi_cv_mean"),
                "ensemble_spikes": lookup(rows, case.alpha, case.dv, "higher_order", "spike_count_total_mean"),
            }
        )
    return params, prepared


def draw_column(
    axes: list[mpl.axes.Axes] | np.ndarray,
    item: dict,
    params,
    *,
    show_ylabels: bool,
    show_panel_letter: bool,
):
    case: Case = item["case"]
    times = item["times"]
    voltage_frames = item["voltage_frames"]
    snapshot_v = item["snapshot_v"]
    snapshot_time = item["snapshot_time"]
    per_node_cv = item["per_node_cv"]
    x_nodes = np.arange(1, params.n_nodes + 1)

    ax_top, ax_mid, ax_bottom = axes
    image = ax_top.imshow(
        voltage_frames,
        aspect="auto",
        origin="lower",
        extent=[1, params.n_nodes, times[0], times[-1]],
        cmap="turbo",
        vmin=-45,
        vmax=35,
        interpolation="nearest",
        rasterized=True,
    )
    ax_top.set_xlim(1, params.n_nodes)
    ax_top.set_xticks([])
    if show_ylabels:
        ax_top.set_ylabel(r"$t$")
        ax_top.set_yticks([0, 500, 1000])
    else:
        ax_top.set_yticks([])
    ax_top.axhline(snapshot_time, color=BLACK, linewidth=2.2, alpha=0.70, zorder=4)
    ax_top.axhline(
        snapshot_time,
        color="white",
        linewidth=1.05,
        linestyle=(0, (4, 2)),
        zorder=5,
    )
    snapshot_y = (snapshot_time - times[0]) / (times[-1] - times[0])
    ax_top.text(
        0.97,
        min(snapshot_y + 0.018, 0.91),
        rf"$t_s={snapshot_time:.0f}$ ms",
        transform=ax_top.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.8,
        color="white",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.58, "pad": 1.0},
        zorder=6,
    )
    ax_top.text(
        0.04,
        0.96,
        rf"$\alpha={case.alpha:.1f},\ D_v={case.dv:.2f}$",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    if show_panel_letter:
        ax_top.text(-0.20, 1.10, rf"({case.letter})", transform=ax_top.transAxes, fontsize=12, fontweight="bold")
    style_axis(ax_top, minor=False)

    ax_mid.plot(x_nodes, snapshot_v, ".", color=RED, markersize=3.2)
    ax_mid.axhline(0.0, color=GRAY, linewidth=0.9)
    ax_mid.set_xlim(0, params.n_nodes + 1)
    ax_mid.set_ylim(-55.0, 35.0)
    ax_mid.set_xticks([])
    ax_mid.set_yticks([-40, 0, 30])
    if show_ylabels:
        ax_mid.set_ylabel(r"$V_i(t_s)$")
    else:
        ax_mid.set_yticklabels([])
    style_axis(ax_mid)

    finite = np.isfinite(per_node_cv)
    ax_bottom.plot(x_nodes[finite], per_node_cv[finite], ".", color=BLACK, markersize=3.0)
    ax_bottom.axhline(item["ensemble_cv"], color=RED, linewidth=1.5)
    ax_bottom.set_xlim(0, params.n_nodes + 1)
    ax_bottom.set_ylim(0.0, 1.25)
    ax_bottom.set_xticks([0, 50, 100])
    ax_bottom.set_yticks([0.0, 0.5, 1.0])
    ax_bottom.set_xlabel(r"Node $i$")
    if show_ylabels:
        ax_bottom.set_ylabel(r"$CV_i$")
    else:
        ax_bottom.set_yticklabels([])
    ax_bottom.text(
        0.05,
        0.91,
        rf"$CV_{{\rm ens}}={item['ensemble_cv']:.2f}$" + "\n" + rf"$N_{{\rm sp}}={item['ensemble_spikes']/1000:.1f}\times10^3$",
        transform=ax_bottom.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color="#333333",
    )
    style_axis(ax_bottom)
    return image


def make_composite(params, prepared: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(7.15, 5.05),
        gridspec_kw={"height_ratios": [1.15, 0.72, 0.78]},
    )
    fig.subplots_adjust(left=0.075, right=0.925, bottom=0.095, top=0.955, hspace=0.48, wspace=0.16)
    image = None
    for col, item in enumerate(prepared):
        image = draw_column(
            axes[:, col],
            item,
            params,
            show_ylabels=(col == 0),
            show_panel_letter=True,
        )
    cax = fig.add_axes([0.94, 0.705, 0.012, 0.205])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label(r"$V$", fontsize=10)
    colorbar.ax.tick_params(direction="in", width=0.9, length=3.5, labelsize=8)
    save_pair(fig, out_dir / "fig1_dynamics_reselected_symbols")


def make_separate_columns(params, prepared: list[dict], out_dir: Path) -> None:
    for item in prepared:
        case: Case = item["case"]
        with mpl.rc_context(
            {
                "font.size": 15,
                "axes.labelsize": 17,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
            }
        ):
            fig, axes = plt.subplots(
                3,
                1,
                figsize=(3.35, 5.6),
                gridspec_kw={"height_ratios": [1.15, 0.72, 0.78]},
            )
            fig.subplots_adjust(left=0.21, right=0.82, bottom=0.09, top=0.96, hspace=0.55)
            image = draw_column(
                axes,
                item,
                params,
                show_ylabels=True,
                show_panel_letter=False,
            )
            cax = fig.add_axes([0.86, 0.735, 0.028, 0.17])
            colorbar = fig.colorbar(image, cax=cax)
            colorbar.set_label(r"$V$", fontsize=14)
            colorbar.ax.tick_params(direction="in", width=0.9, length=3.5, labelsize=12)
            save_pair(fig, out_dir / f"fig1{case.letter}_dynamics_reselected_symbols")


def write_summary(prepared: list[dict], out_dir: Path) -> None:
    path = out_dir / "fig1_dynamics_reselected_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "panel",
                "alpha",
                "D_v",
                "snapshot_time_ms",
                "snapshot_mean_mV",
                "snapshot_std_mV",
                "snapshot_q05_mV",
                "snapshot_q95_mV",
                "snapshot_active_fraction",
                "snapshot_selection_score",
                "ensemble_cv",
                "ensemble_spike_count",
            ),
        )
        writer.writeheader()
        for item in prepared:
            case: Case = item["case"]
            writer.writerow(
                {
                    "panel": case.letter,
                    "alpha": case.alpha,
                    "D_v": case.dv,
                    "snapshot_time_ms": item["snapshot_time"],
                    "snapshot_mean_mV": item["snapshot_mean"],
                    "snapshot_std_mV": item["snapshot_std"],
                    "snapshot_q05_mV": item["snapshot_q05"],
                    "snapshot_q95_mV": item["snapshot_q95"],
                    "snapshot_active_fraction": item["snapshot_active_fraction"],
                    "snapshot_selection_score": item["snapshot_score"],
                    "ensemble_cv": item["ensemble_cv"],
                    "ensemble_spike_count": item["ensemble_spikes"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 1 with reproducibly selected representative voltage snapshots.")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "paper_suite_true_simplicial_array_prl_N100_Dv_refined",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "paper_suite_true_simplicial_array_prl_N100_Dv_refined" / "formal_figures",
    )
    parser.add_argument("--trace-total", type=float, default=1400.0)
    parser.add_argument("--trace-warmup", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=20260509)
    args = parser.parse_args()

    apply_style()
    params, prepared = prepare_data(args.result_root, args.seed, args.trace_total, args.trace_warmup)
    make_composite(params, prepared, args.output_dir)
    make_separate_columns(params, prepared, args.output_dir)
    write_summary(prepared, args.output_dir)
    print(f"Wrote reselected Figure 1 to {args.output_dir}")


if __name__ == "__main__":
    main()
