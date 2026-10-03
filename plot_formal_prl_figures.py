from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(os.environ.get("ML_FORMAL_ROOT", Path(__file__).resolve().parent / "paper_suite_true_simplicial_array_prl_N100_Dv_refined"))
FIG_DIR = ROOT / "formal_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ("control", "higher_order")
COLORS = {"control": "#36454f", "higher_order": "#c62828"}
MARKERS = {"control": "o", "higher_order": "s"}
ALPHA_TITLES = {1.5: r"$\alpha=1.5$ (Levy)", 2.0: r"$\alpha=2.0$ (Gaussian)"}

mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 13,
        "axes.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return float("nan")
    return float(value)


def sem(row: dict, metric: str) -> float:
    std = f(row, f"{metric}_std")
    n = f(row, "total_realizations")
    if not math.isfinite(std) or not math.isfinite(n) or n <= 0:
        return float("nan")
    return std / math.sqrt(n)


def noise_intensity_dv(noise_scale: float, alpha: float, capacitance: float = 20.0) -> float:
    return (noise_scale / capacitance) ** alpha


def row_noise_intensity_dv(row: dict) -> float:
    value = row.get("noise_intensity_dv", "")
    if value not in ("", None):
        return float(value)
    return noise_intensity_dv(f(row, "noise_scale"), f(row, "alpha"))


def subset(rows: list[dict], alpha: float, label: str) -> list[dict]:
    out = [row for row in rows if abs(f(row, "alpha") - alpha) < 1e-12 and row["label"] == label]
    return sorted(out, key=row_noise_intensity_dv)


def configure_axes(ax):
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def plot_metric_panel(ax, rows: list[dict], alpha: float, metric: str, ylabel: str, use_sem: bool = True):
    for label in LABELS:
        data = subset(rows, alpha, label)
        x = [row_noise_intensity_dv(row) for row in data]
        y = [f(row, f"{metric}_mean") for row in data]
        yerr = [sem(row, metric) if use_sem else f(row, f"{metric}_std") for row in data]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=MARKERS[label],
            color=COLORS[label],
            linewidth=1.8,
            markersize=4.8,
            capsize=2.6,
            label=label.replace("_", " "),
        )
    ax.set_ylabel(ylabel)
    configure_axes(ax)


def save_figure(fig, name: str, tight: bool = True):
    bbox_kwargs = {"bbox_inches": "tight", "pad_inches": 0.03} if tight else {"bbox_inches": None}
    for suffix, kwargs in (
        ("png", {"dpi": 600, **bbox_kwargs}),
        ("pdf", bbox_kwargs),
    ):
        path = FIG_DIR / f"{name}.{suffix}"
        try:
            fig.savefig(path, **kwargs)
        except PermissionError:
            fig.savefig(FIG_DIR / f"{name}_prlstyle.{suffix}", **kwargs)
    plt.close(fig)


def best_row_for(rows: list[dict], alpha: float, label: str, metric: str = "isi_cv", min_spikes: float = 1000.0):
    data = [
        row
        for row in subset(rows, alpha, label)
        if math.isfinite(f(row, f"{metric}_mean"))
        and math.isfinite(f(row, "spike_count_total_mean"))
        and f(row, "spike_count_total_mean") >= min_spikes
    ]
    return min(data, key=lambda row: f(row, f"{metric}_mean")) if data else None


def format_dv(value: float) -> str:
    if abs(value - round(value)) < 1e-10:
        return f"{value:.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def configure_fig1_axis(ax, ylabel: str, ylim: tuple[float, float], yticks: list[float], yticklabels: list[str] | None = None):
    configure_axes(ax)
    ax.set_xscale("log")
    ax.set_xlim(0.035, 10.5)
    ax.set_xticks([0.04, 0.1, 0.3, 1.0, 4.0, 9.0])
    ax.set_xticklabels(["0.04", "0.10", "0.30", "1", "4", "9"])
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    ax.set_xlabel(r"$D_v$")
    ax.set_ylabel(ylabel)


def plot_fig1_single_panel(
    rows: list[dict],
    alpha: float,
    metric: str,
    ylabel: str,
    name: str,
    ylim: tuple[float, float],
    yticks: list[float],
    yticklabels: list[str] | None,
    annotate_cv: bool,
):
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.18, top=0.96)
    annotation_lines = []
    for label in LABELS:
        data = subset(rows, alpha, label)
        x = [row_noise_intensity_dv(row) for row in data]
        y = [f(row, f"{metric}_mean") for row in data]
        yerr = [sem(row, metric) for row in data]
        display_label = "Pairwise" if label == "control" else "Simplicial"
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=MARKERS[label],
            color=COLORS[label],
            linewidth=2.2,
            markersize=6.2,
            markeredgewidth=1.2,
            capsize=3.2,
            label=display_label,
        )
        best = best_row_for(rows, alpha, label)
        if best is not None:
            best_x = row_noise_intensity_dv(best)
            best_y = f(best, f"{metric}_mean")
            ax.axvline(best_x, color=COLORS[label], linestyle=":", linewidth=1.25, alpha=0.72)
            ax.plot(
                [best_x],
                [best_y],
                marker=MARKERS[label],
                markersize=9.0,
                markerfacecolor="white",
                markeredgecolor=COLORS[label],
                markeredgewidth=1.6,
                linestyle="none",
                zorder=6,
            )
            if annotate_cv:
                prefix = "P" if label == "control" else "S"
                annotation_lines.append((COLORS[label], rf"{prefix}: $D_v^*={format_dv(best_x)}$, $CV^*={f(best, 'isi_cv_mean'):.3f}$"))
    ax.text(0.055, 0.90, rf"$\alpha={alpha:.1f}$", transform=ax.transAxes, ha="left", va="top", fontsize=18)
    configure_fig1_axis(ax, ylabel, ylim, yticks, yticklabels)
    ax.legend(loc="upper right", handlelength=2.2, borderaxespad=0.3)
    if annotate_cv:
        for idx, (color, text) in enumerate(annotation_lines):
            ax.text(
                0.055,
                0.77 - 0.09 * idx,
                text,
                transform=ax.transAxes,
                color=color,
                fontsize=12.5,
                ha="left",
                va="top",
            )
    save_figure(fig, name, tight=False)


def plot_main_candidate():
    rows = read_rows(ROOT / "candidate39_main" / "aggregated_metrics.csv")
    cv_values = [f(row, "isi_cv_mean") for row in rows if math.isfinite(f(row, "isi_cv_mean"))]
    spike_values = [f(row, "spike_count_total_mean") for row in rows if math.isfinite(f(row, "spike_count_total_mean"))]
    cv_ylim = (0.0, max(2.1, math.ceil(max(cv_values) * 10.0) / 10.0))
    spike_ylim = (0.0, math.ceil(max(spike_values) / 10000.0) * 10000.0)
    cv_ticks = [0.0, 0.5, 1.0, 1.5, 2.0]
    spike_ticks = [0, 20000, 40000, 60000]
    spike_ticklabels = ["0", "20", "40", "60"]

    plot_fig1_single_panel(rows, 1.5, "isi_cv", "CV", "fig1a_main_candidate39_cv_alpha15", cv_ylim, cv_ticks, None, True)
    plot_fig1_single_panel(rows, 2.0, "isi_cv", "CV", "fig1b_main_candidate39_cv_alpha20", cv_ylim, cv_ticks, None, True)
    plot_fig1_single_panel(rows, 1.5, "spike_count_total", r"Spike count ($10^3$)", "fig1c_main_candidate39_spikes_alpha15", spike_ylim, spike_ticks, spike_ticklabels, False)
    plot_fig1_single_panel(rows, 2.0, "spike_count_total", r"Spike count ($10^3$)", "fig1d_main_candidate39_spikes_alpha20", spike_ylim, spike_ticks, spike_ticklabels, False)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex="col", sharey="row")
    for col, alpha in enumerate((1.5, 2.0)):
        for row_idx, (metric, ylabel, ylim, yticks) in enumerate(
            (
                ("isi_cv", "CV", cv_ylim, cv_ticks),
                ("spike_count_total", "Spike count", spike_ylim, spike_ticks),
            )
        ):
            ax = axes[row_idx, col]
            for label in LABELS:
                data = subset(rows, alpha, label)
                x = [row_noise_intensity_dv(item) for item in data]
                y = [f(item, f"{metric}_mean") for item in data]
                yerr = [sem(item, metric) for item in data]
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    marker=MARKERS[label],
                    color=COLORS[label],
                    linewidth=2.0,
                    markersize=5.2,
                    capsize=2.8,
                    label="Pairwise" if label == "control" else "Simplicial",
                )
                best = best_row_for(rows, alpha, label)
                if best is not None:
                    best_x = row_noise_intensity_dv(best)
                    best_y = f(best, f"{metric}_mean")
                    ax.axvline(best_x, color=COLORS[label], linestyle=":", linewidth=1.0, alpha=0.6)
                    ax.plot([best_x], [best_y], marker=MARKERS[label], markersize=7.0, markerfacecolor="white", markeredgecolor=COLORS[label], linestyle="none")
            ax.text(0.055, 0.88, rf"$\alpha={alpha:.1f}$", transform=ax.transAxes, ha="left", va="top", fontsize=15)
            configure_fig1_axis(ax, ylabel, ylim, yticks)
            if col > 0:
                ax.set_ylabel("")
            if row_idx == 0:
                ax.set_xlabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "fig1_main_candidate39_cv_spikes")


def collect_best_from_summary(experiments: list[str]) -> list[dict]:
    summary = json.loads((ROOT / "merged_summary.json").read_text(encoding="utf-8"))
    rows = []
    for exp in experiments:
        item = summary[exp]
        for alpha_key, alpha_data in item["best_by_alpha"].items():
            alpha = float(alpha_key)
            control = alpha_data["control"]
            higher = alpha_data["higher_order"]
            rows.append(
                {
                    "experiment": exp,
                    "alpha": alpha,
                    "control_noise": control["best_noise_scale"],
                    "control_Dv": control.get("best_noise_intensity_dv", noise_intensity_dv(control["best_noise_scale"], alpha)),
                    "control_cv": control["best_cv_value"],
                    "higher_noise": higher["best_noise_scale"],
                    "higher_Dv": higher.get("best_noise_intensity_dv", noise_intensity_dv(higher["best_noise_scale"], alpha)),
                    "higher_cv": higher["best_cv_value"],
                    "delta_cv": control["best_cv_value"] - higher["best_cv_value"],
                    "control_spikes": control["best_spike_count"],
                    "higher_spikes": higher["best_spike_count"],
                    "mean_simplices": item["mean_simplex_count"],
                }
            )
    return rows


def sigma2_experiments():
    items = []
    for path in ROOT.glob("sigma2_*"):
        if not (path / "summary.json").exists():
            continue
        try:
            sigma = float(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        items.append((sigma, path.name))
    return sorted(items, key=lambda item: item[0])


def configure_sigma2_axis(ax, ylim: tuple[float, float], yticks: list[float]):
    configure_axes(ax)
    ax.set_xlim(-0.005, 0.165)
    ax.set_xticks([0.00, 0.04, 0.08, 0.12, 0.16])
    ax.set_xticklabels(["0", "0.04", "0.08", "0.12", "0.16"])
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xlabel(r"$\sigma_2$")
    ax.set_ylabel("CV")


def configure_sigma2_dv_axis(ax, ylim: tuple[float, float], yticks: list[float]):
    configure_axes(ax)
    ax.set_xlim(-0.005, 0.165)
    ax.set_xticks([0.00, 0.04, 0.08, 0.12, 0.16])
    ax.set_xticklabels(["0", "0.04", "0.08", "0.12", "0.16"])
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xlabel(r"$\sigma_2$")
    ax.set_ylabel(r"$D_v^*$")


def plot_sigma2_single_panel(
    rows: list[dict],
    sigma: list[float],
    alpha: float,
    metric: str,
    ylabel_kind: str,
    name: str,
    ylim: tuple[float, float],
    yticks: list[float],
):
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.18, top=0.96)
    data = [row for row in rows if row["alpha"] == alpha]
    ax.plot(
        sigma,
        [row[f"control_{metric}"] for row in data],
        marker=MARKERS["control"],
        linestyle="--",
        color=COLORS["control"],
        linewidth=2.2,
        markersize=6.2,
        markeredgewidth=1.2,
        label="Pairwise",
    )
    ax.plot(
        sigma,
        [row[f"higher_{metric}"] for row in data],
        marker=MARKERS["higher_order"],
        linestyle="-",
        color=COLORS["higher_order"],
        linewidth=2.2,
        markersize=6.2,
        markeredgewidth=1.2,
        label="Simplicial",
    )
    ax.text(0.06, 0.16, rf"$\alpha={alpha:.1f}$", transform=ax.transAxes, ha="left", va="bottom", fontsize=18)
    if ylabel_kind == "cv":
        configure_sigma2_axis(ax, ylim, yticks)
    else:
        configure_sigma2_dv_axis(ax, ylim, yticks)
    ax.legend(loc="upper right", handlelength=2.2, borderaxespad=0.3)
    save_figure(fig, name, tight=False)


def plot_sigma2_all_alpha_panel(
    rows: list[dict],
    sigma: list[float],
    metric: str,
    ylabel_kind: str,
    name: str,
    ylim: tuple[float, float],
    yticks: list[float],
):
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.18, top=0.96)
    for alpha, marker_face in ((1.5, "filled"), (2.0, "open")):
        data = [row for row in rows if row["alpha"] == alpha]
        for label, linestyle in (("control", "--"), ("higher_order", "-")):
            markerfacecolor = COLORS[label] if marker_face == "filled" else "white"
            display_label = "Pairwise" if label == "control" else "Simplicial"
            ax.plot(
                sigma,
                [row[f"{'control' if label == 'control' else 'higher'}_{metric}"] for row in data],
                marker=MARKERS[label],
                linestyle=linestyle,
                color=COLORS[label],
                linewidth=2.2,
                markersize=6.4,
                markerfacecolor=markerfacecolor,
                markeredgecolor=COLORS[label],
                markeredgewidth=1.3,
                label=rf"{display_label}, $\alpha={alpha:.1f}$",
            )
    if ylabel_kind == "cv":
        configure_sigma2_axis(ax, ylim, yticks)
    else:
        configure_sigma2_dv_axis(ax, ylim, yticks)
    ax.legend(loc="best", handlelength=2.2, borderaxespad=0.3, fontsize=11.5)
    save_figure(fig, name, tight=False)


def plot_sigma2_sweep():
    sigma_items = sigma2_experiments()
    experiments = [name for _, name in sigma_items]
    rows = collect_best_from_summary(experiments)
    sigma = [value for value, _ in sigma_items]

    cv_values = []
    for row in rows:
        cv_values.extend([row["control_cv"], row["higher_cv"]])
    ymax = max(0.8, math.ceil(max(cv_values) * 10.0) / 10.0)
    cv_ylim = (0.0, ymax)
    cv_yticks = [0.0, 0.2, 0.4, 0.6, 0.8] if ymax <= 0.8 else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    dv_values = []
    for row in rows:
        dv_values.extend([row["control_Dv"], row["higher_Dv"]])
    dv_ymax = max(1.0, math.ceil(max(dv_values) * 10.0) / 10.0)
    dv_ylim = (0.0, dv_ymax)
    dv_yticks = [0.0, 0.3, 0.6, 0.9, 1.2] if dv_ymax <= 1.2 else [0.0, 0.5, 1.0, 1.5]

    plot_sigma2_all_alpha_panel(rows, sigma, "cv", "cv", "fig2a_sigma2_sweep_cv_all_alpha", cv_ylim, cv_yticks)
    plot_sigma2_all_alpha_panel(rows, sigma, "Dv", "dv", "fig2b_sigma2_sweep_Dv_all_alpha", dv_ylim, dv_yticks)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
    for ax, metric, axis_config, ylim_item, yticks_item in (
        (axes[0], "cv", configure_sigma2_axis, cv_ylim, cv_yticks),
        (axes[1], "Dv", configure_sigma2_dv_axis, dv_ylim, dv_yticks),
    ):
        for alpha, marker_face in ((1.5, "filled"), (2.0, "open")):
            data = [row for row in rows if row["alpha"] == alpha]
            for label, linestyle in (("control", "--"), ("higher_order", "-")):
                markerfacecolor = COLORS[label] if marker_face == "filled" else "white"
                display_label = "Pairwise" if label == "control" else "Simplicial"
                ax.plot(
                    sigma,
                    [row[f"{'control' if label == 'control' else 'higher'}_{metric}"] for row in data],
                    marker=MARKERS[label],
                    linestyle=linestyle,
                    color=COLORS[label],
                    linewidth=2.0,
                    markersize=5.6,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=COLORS[label],
                    markeredgewidth=1.2,
                    label=rf"{display_label}, $\alpha={alpha:.1f}$",
                )
        axis_config(ax, ylim_item, yticks_item)
    axes[0].legend(loc="best", handlelength=2.2, borderaxespad=0.3, fontsize=10.5)
    fig.tight_layout()
    save_figure(fig, "fig2_sigma2_sweep_best_cv")


def plot_current_robustness():
    experiments = ["current_87.0", "current_88.0", "current_89.0"]
    rows = collect_best_from_summary(experiments)
    currents = [87.0, 88.0, 89.0]

    cv_values = [row[key] for row in rows for key in ("control_cv", "higher_cv") if math.isfinite(row[key])]
    cv_ylim = (0.0, max(0.85, math.ceil(max(cv_values) * 10.0) / 10.0))
    cv_ticks = [0.0, 0.2, 0.4, 0.6, 0.8]
    dv_values = [row[key] for row in rows for key in ("control_Dv", "higher_Dv") if math.isfinite(row[key])]
    dv_ylim = (0.0, max(1.05, math.ceil(max(dv_values) * 4.0) / 4.0))
    dv_ticks = [0.0, 0.25, 0.50, 0.75, 1.00]

    def ordered_data(alpha: float) -> list[dict]:
        data = [row for row in rows if abs(row["alpha"] - alpha) < 1e-12]
        return sorted(data, key=lambda row: float(row["experiment"].split("_", 1)[1]))

    def draw_metric_panel(
        ax,
        metric: str,
        ylabel: str,
        ylim: tuple[float, float],
        yticks: list[float],
        legend_fontsize: float = 11.0,
        legend_outside: bool = False,
    ):
        for alpha in (1.5, 2.0):
            data = ordered_data(alpha)
            markerfacecolor = None if alpha == 1.5 else "white"
            for label, key in (("control", f"control_{metric}"), ("higher_order", f"higher_{metric}")):
                display_label = "Pairwise" if label == "control" else "Simplicial"
                ax.plot(
                    currents,
                    [row[key] for row in data],
                    linestyle="--" if label == "control" else "-",
                    marker=MARKERS[label],
                    color=COLORS[label],
                    linewidth=2.2,
                    markersize=6.4,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=COLORS[label],
                    markeredgewidth=1.2,
                    label=rf"{display_label}, $\alpha={alpha:.1f}$",
                )
        configure_axes(ax)
        ax.set_xlim(86.7, 89.3)
        ax.set_xticks(currents)
        ax.set_xticklabels(["87", "88", "89"])
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_xlabel(r"$I$")
        ax.set_ylabel(ylabel)
        if legend_outside:
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=2,
                handlelength=1.6,
                columnspacing=0.8,
                handletextpad=0.4,
                borderaxespad=0.0,
                fontsize=legend_fontsize,
            )
        else:
            ax.legend(loc="best", handlelength=2.2, borderaxespad=0.3, fontsize=legend_fontsize)

    for metric, ylabel, ylim, yticks, name in (
        ("cv", "CV", cv_ylim, cv_ticks, "fig3a_current_robustness_cv"),
        ("Dv", r"$D_v^*$", dv_ylim, dv_ticks, "fig3b_current_robustness_Dv"),
    ):
        fig, ax = plt.subplots(figsize=(5.2, 3.9))
        fig.subplots_adjust(left=0.20, right=0.97, bottom=0.18, top=0.96)
        draw_metric_panel(ax, metric, ylabel, ylim, yticks)
        save_figure(fig, name, tight=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    draw_metric_panel(axes[0], "cv", "CV", cv_ylim, cv_ticks, legend_fontsize=6.2, legend_outside=True)
    draw_metric_panel(axes[1], "Dv", r"$D_v^*$", dv_ylim, dv_ticks, legend_fontsize=6.2, legend_outside=True)
    fig.tight_layout()
    save_figure(fig, "fig3_current_robustness_best_cv")


def plot_long_time_validation():
    rows_main = collect_best_from_summary(["candidate39_main"])
    rows_long = collect_best_from_summary(["candidate39_longer_time"])
    labels = ["main", "longer"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for alpha in (1.5, 2.0):
        data_main = [row for row in rows_main if row["alpha"] == alpha][0]
        data_long = [row for row in rows_long if row["alpha"] == alpha][0]
        axes[0].plot(
            labels,
            [data_main["control_cv"], data_long["control_cv"]],
            "o--",
            linewidth=1.3,
            markersize=4,
            label=f"control, alpha={alpha}",
        )
        axes[0].plot(
            labels,
            [data_main["higher_cv"], data_long["higher_cv"]],
            "s-",
            linewidth=1.8,
            markersize=4,
            label=f"higher, alpha={alpha}",
        )
        axes[1].plot(
            labels,
            [data_main["control_Dv"], data_long["control_Dv"]],
            "o--",
            linewidth=1.3,
            markersize=4,
            label=f"control, alpha={alpha}",
        )
        axes[1].plot(
            labels,
            [data_main["higher_Dv"], data_long["higher_Dv"]],
            "s-",
            linewidth=1.8,
            markersize=4,
            label=f"higher, alpha={alpha}",
        )

    axes[0].set_ylabel("Best ISI CV")
    axes[1].set_ylabel(r"Optimal $D_v^\ast$")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        configure_axes(ax)
    fig.tight_layout()
    save_figure(fig, "fig4_long_time_validation")


def write_best_summary_csv():
    experiments = [
        "candidate39_main",
        "candidate42_main",
        *[name for _, name in sigma2_experiments()],
        "current_87.0",
        "current_88.0",
        "current_89.0",
        "candidate39_longer_time",
    ]
    rows = collect_best_from_summary(experiments)
    path = FIG_DIR / "best_summary_table.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    plot_main_candidate()
    plot_sigma2_sweep()
    plot_current_robustness()
    plot_long_time_validation()
    write_best_summary_csv()
    print(f"Formal figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
