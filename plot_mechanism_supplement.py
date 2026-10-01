from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
import numpy as np


OPTIMAL_DV = {
    ("1.5", "control"): 0.56,
    ("1.5", "higher_order"): 0.16,
    ("2.0", "control"): 1.00,
    ("2.0", "higher_order"): 0.30,
}


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f(row, key):
    value = row.get(key, "")
    return np.nan if value in ("", None, "None") else float(value)


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, 0
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0, int(arr.size)


def collect_chunks(root: Path, filename: str):
    rows = []
    for path in sorted((root / "chunks").glob(f"network_*/{filename}")):
        rows.extend(read_csv(path))
    if filename == "binned_synchrony_metrics.csv":
        for row in rows:
            if "synchrony_variance_ratio" not in row:
                old_ratio = f(row, "synchrony_index")
                row["synchrony_variance_ratio"] = old_ratio
                row["synchrony_index"] = np.sqrt(old_ratio) if np.isfinite(old_ratio) and old_ratio >= 0 else np.nan
    return rows


def aggregate(rows, group_keys, metrics):
    out = []
    keys = sorted({tuple(row[key] for key in group_keys) for row in rows})
    for key in keys:
        subset = [row for row in rows if tuple(row[k] for k in group_keys) == key]
        item = dict(zip(group_keys, key))
        item["n_rows"] = len(subset)
        for metric in metrics:
            mean, std, n = mean_std([f(row, metric) for row in subset])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
            item[f"{metric}_n"] = n
        out.append(item)
    return out


def apply_publication_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 6.5,
        }
    )


def apply_prl_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 12,
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


def format_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks([0.04, 0.1, 0.25, 0.56, 1.0, 4.0, 9.0])
    ax.set_xticklabels(["0.04", "0.10", "0.25", "0.56", "1", "4", "9"])
    ax.grid(False)
    ax.tick_params(labelsize=8, direction="in", top=True, right=True)


def format_prl_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(0.035, 10.5)
    ax.set_xticks([0.04, 0.1, 0.25, 0.56, 1.0, 4.0, 9.0])
    ax.set_xticklabels(["0.04", "0.10", "0.25", "0.56", "1", "4", "9"])
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def save_prl_figure(fig, out_dir: Path, name: str):
    for suffix, kwargs in (
        ("pdf", {"bbox_inches": None}),
        ("png", {"dpi": 600, "bbox_inches": None}),
    ):
        path = out_dir / f"{name}.{suffix}"
        try:
            fig.savefig(path, **kwargs)
        except PermissionError:
            fig.savefig(out_dir / f"{name}_prlstyle.{suffix}", **kwargs)
    plt.close(fig)


def finite_best_index(x, y, spikes, min_spikes=1000.0):
    y_arr = np.asarray(y, dtype=float)
    s_arr = np.asarray(spikes, dtype=float)
    finite = np.isfinite(y_arr) & (s_arr > min_spikes)
    if not np.any(finite):
        return None
    idxs = np.where(finite)[0]
    return int(idxs[np.nanargmin(y_arr[finite])])


def index_at_dv(x, target):
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return None
    idx = int(np.argmin(np.abs(arr - target)))
    return idx if np.isfinite(arr[idx]) and abs(arr[idx] - target) < 1e-9 else None


def mark_optimal_point(ax, x, y, alpha, label, color, marker="o", size=6.5):
    target = OPTIMAL_DV[(alpha, label)]
    idx = index_at_dv(x, target)
    ax.axvline(target, color=color, lw=0.65, ls=":", alpha=0.42, zorder=0)
    if idx is not None and np.isfinite(y[idx]):
        ax.plot(
            [x[idx]],
            [y[idx]],
            marker=marker,
            ms=size,
            mfc="white",
            mec=color,
            mew=1.1,
            linestyle="none",
            zorder=5,
        )


def sem_from_row(row, metric):
    std = f(row, f"{metric}_std")
    n = f(row, f"{metric}_n")
    return std / np.sqrt(n) if np.isfinite(std) and np.isfinite(n) and n > 0 else np.nan


def add_optimal_dv_lines(ax, alpha):
    for label, color in (("control", "#666666"), ("higher_order", "#c62828")):
        target = OPTIMAL_DV[(alpha, label)]
        ax.axvline(target, color=color, lw=0.75, ls=":", alpha=0.65, zorder=0)


def plot_with_sem(ax, x, y, sem, color, ls, marker, label):
    ax.plot(x, y, marker=marker, ls=ls, color=color, lw=1.35, ms=3.4, label=label)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    sem_arr = np.asarray(sem, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(sem_arr)
    if np.any(valid):
        ax.fill_between(
            x_arr[valid],
            y_arr[valid] - sem_arr[valid],
            y_arr[valid] + sem_arr[valid],
            color=color,
            alpha=0.12,
            linewidth=0,
        )


def plot_activation(out_dir: Path, rows):
    vlow = "-20.0"
    data = [row for row in rows if row["v_low"] == vlow]
    apply_publication_style()
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.0), constrained_layout=True, sharex="col")
    colors = {"control": "#36454f", "higher_order": "#c62828"}
    labels = {"control": "control", "higher_order": "higher-order"}
    metrics = [
        ("isi_cv_mean", "ISI CV"),
        ("activation_cv_mean", r"$CV(T_a)$"),
        ("excursion_cv_mean", r"$CV(T_e)$"),
    ]
    for col, alpha in enumerate(("1.5", "2.0")):
        for label in ("control", "higher_order"):
            subset = [row for row in data if row["alpha"] == alpha and row["label"] == label]
            subset.sort(key=lambda row: f(row, "noise_intensity_dv"))
            x = [f(row, "noise_intensity_dv") for row in subset]
            for row_idx, (metric, ylabel) in enumerate(metrics):
                y = [f(row, metric) for row in subset]
                axes[row_idx, col].plot(
                    x,
                    y,
                    "o-",
                    color=colors[label],
                    lw=1.25,
                    ms=3.4,
                    label=labels[label],
                )
                spikes = [f(row, "spike_count_total_mean") for row in subset]
                best = finite_best_index(x, y, spikes)
                if best is not None:
                    axes[row_idx, col].plot(
                        [x[best]],
                        [y[best]],
                        marker="o",
                        ms=7.0,
                        mfc="none",
                        mec=colors[label],
                        mew=1.1,
                        linestyle="none",
                    )
        axes[0, col].text(
            0.04,
            0.92,
            rf"$\alpha={alpha}$",
            transform=axes[0, col].transAxes,
            fontsize=9,
            ha="left",
            va="top",
        )
        axes[2, col].set_xlabel(r"$D_v$")
        for row_idx, (_, ylabel) in enumerate(metrics):
            axes[row_idx, col].set_ylabel(ylabel)
            format_axis(axes[row_idx, col])
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(out_dir / "mechanism_activation_excursion.pdf")
    fig.savefig(out_dir / "mechanism_activation_excursion.png", dpi=300)
    fig.savefig(out_dir / "mechanism_activation_excursion_decomposition.pdf")
    fig.savefig(out_dir / "mechanism_activation_excursion_decomposition.png", dpi=300)
    plt.close(fig)


def plot_exp1_mean_cv(out_dir: Path, rows):
    data = [row for row in rows if row["v_low"] == "-20.0"]
    apply_prl_style()
    colors = {"control": "#36454f", "higher_order": "#c62828"}
    labels = {"control": "Pairwise", "higher_order": "Simplicial"}
    line_specs = {
        "activation": ("o", "-", r"$T_a$"),
        "excursion": ("s", "--", r"$T_e$"),
    }

    def subset_for(alpha: str, label: str):
        subset = [row for row in data if row["alpha"] == alpha and row["label"] == label]
        subset.sort(key=lambda row: f(row, "noise_intensity_dv"))
        return subset

    def draw_panel(ax, alpha: str, mode: str, legend_fontsize: float):
        for label in ("control", "higher_order"):
            subset = subset_for(alpha, label)
            x = [f(row, "noise_intensity_dv") for row in subset]
            color = colors[label]
            for quantity, (marker, linestyle, quantity_label) in line_specs.items():
                if mode == "mean":
                    metric = f"{quantity}_mean_mean"
                    display_quantity = rf"$\langle {quantity_label[1:-1]}\rangle$"
                else:
                    metric = f"{quantity}_cv_mean"
                    display_quantity = rf"$CV({quantity_label[1:-1]})$"
                y = [f(row, metric) for row in subset]
                ax.plot(
                    x,
                    y,
                    marker=marker,
                    linestyle=linestyle,
                    color=color,
                    lw=2.1,
                    ms=6.0,
                    markeredgecolor=color,
                    markeredgewidth=1.1,
                    label=rf"{labels[label]}, {display_quantity}",
                )
                mark_optimal_point(ax, x, y, alpha, label, color, marker=marker, size=8.0)
        format_prl_axis(ax)
        ax.set_xlabel(r"$D_v$")
        ax.text(0.055, 0.92, rf"$\alpha={alpha}$", transform=ax.transAxes, fontsize=18, ha="left", va="top")
        if mode == "mean":
            ax.set_ylabel(r"$\langle T_a\rangle,\ \langle T_e\rangle$")
        else:
            ax.set_ylabel(r"$CV(T_a),\ CV(T_e)$")
        ax.legend(loc="best", fontsize=legend_fontsize, handlelength=2.0, borderaxespad=0.3)

    panel_specs = [
        ("1.5", "mean", "exp1_figureA_a_activation_excursion_mean_alpha15"),
        ("2.0", "mean", "exp1_figureA_b_activation_excursion_mean_alpha20"),
        ("1.5", "cv", "exp1_figureA_c_activation_excursion_cv_alpha15"),
        ("2.0", "cv", "exp1_figureA_d_activation_excursion_cv_alpha20"),
    ]
    for alpha, mode, name in panel_specs:
        fig, ax = plt.subplots(figsize=(5.2, 3.9))
        fig.subplots_adjust(left=0.20, right=0.97, bottom=0.18, top=0.96)
        draw_panel(ax, alpha, mode, legend_fontsize=10.0)
        save_prl_figure(fig, out_dir, name)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.8))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.96, hspace=0.30, wspace=0.28)
    for ax, (alpha, mode, _) in zip(axes.ravel(), panel_specs):
        draw_panel(ax, alpha, mode, legend_fontsize=6.0)
    save_prl_figure(fig, out_dir, "exp1_figureA_activation_excursion_mean_cv")


def plot_exp1_variance_fraction(out_dir: Path, rows):
    apply_prl_style()
    colors = {"-30.0": "#2b6cb0", "-20.0": "#222222", "-10.0": "#b22222"}
    labels = {"control": "pairwise", "higher_order": "simplicial"}
    condition_colors = {"control": "#36454f", "higher_order": "#c62828"}

    def draw_panel(ax, alpha: str, label: str, legend_fontsize: float):
        for vlow in ("-30.0", "-20.0", "-10.0"):
            subset = [row for row in rows if row["alpha"] == alpha and row["label"] == label and row["v_low"] == vlow]
            subset.sort(key=lambda row: f(row, "noise_intensity_dv"))
            x = [f(row, "noise_intensity_dv") for row in subset]
            fa = [f(row, "activation_variance_fraction_mean") for row in subset]
            fe = [f(row, "excursion_variance_fraction_mean") for row in subset]
            ax.plot(
                x,
                fa,
                marker="o",
                linestyle="-",
                color=colors[vlow],
                lw=2.0,
                ms=5.6,
                markeredgecolor=colors[vlow],
                markeredgewidth=1.0,
                label=rf"$f_a,\ V_{{\rm low}}={float(vlow):.0f}$",
            )
            if vlow == "-20.0":
                ax.plot(
                    x,
                    fe,
                    marker="s",
                    linestyle="--",
                    color="#707070",
                    lw=1.9,
                    ms=5.4,
                    markeredgecolor="#707070",
                    markeredgewidth=1.0,
                    label=rf"$f_e,\ V_{{\rm low}}=-20$",
                )
        opt = OPTIMAL_DV[(alpha, label)]
        ax.axvline(opt, color=condition_colors[label], lw=0.9, ls=":", alpha=0.58, zorder=0)
        ax.text(
            0.055,
            0.92,
            rf"{labels[label]}, $\alpha={alpha}$",
            transform=ax.transAxes,
            fontsize=17,
            ha="left",
            va="top",
        )
        ax.text(opt, 0.055, rf"$D_v^*$", color=condition_colors[label], fontsize=14, ha="center", va="bottom")
        ax.set_xlabel(r"$D_v$")
        ax.set_ylabel(r"$f_a,\ f_e$")
        format_prl_axis(ax)
        ax.legend(loc="best", fontsize=legend_fontsize, handlelength=2.0, borderaxespad=0.3)

    panel_specs = [
        ("1.5", "control", "exp1_figureB_a_variance_fraction_pairwise_alpha15"),
        ("2.0", "control", "exp1_figureB_b_variance_fraction_pairwise_alpha20"),
        ("1.5", "higher_order", "exp1_figureB_c_variance_fraction_simplicial_alpha15"),
        ("2.0", "higher_order", "exp1_figureB_d_variance_fraction_simplicial_alpha20"),
    ]
    for alpha, label, name in panel_specs:
        fig, ax = plt.subplots(figsize=(5.2, 3.9))
        fig.subplots_adjust(left=0.18, right=0.97, bottom=0.18, top=0.96)
        draw_panel(ax, alpha, label, legend_fontsize=9.3)
        save_prl_figure(fig, out_dir, name)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.8))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.96, hspace=0.30, wspace=0.23)
    for ax, (alpha, label, _) in zip(axes.ravel(), panel_specs):
        draw_panel(ax, alpha, label, legend_fontsize=6.0)
    save_prl_figure(fig, out_dir, "exp1_figureB_variance_fraction_thresholds")


def plot_exp1_threshold_sensitivity(out_dir: Path, rows):
    apply_prl_style()
    colors = {"control": "#36454f", "higher_order": "#c62828"}
    labels = {"control": "pairwise", "higher_order": "simplicial"}
    vlow_styles = {
        "-30.0": ("o", "-", r"$V_{\rm low}=-30$"),
        "-20.0": ("s", "--", r"$V_{\rm low}=-20$"),
        "-10.0": ("^", ":", r"$V_{\rm low}=-10$"),
    }
    metrics = [
        ("activation_mean_mean", r"$\langle T_a\rangle$", "activation_mean"),
        ("excursion_mean_mean", r"$\langle T_e\rangle$", "excursion_mean"),
        ("activation_cv_mean", r"$CV(T_a)$", "activation_cv"),
        ("excursion_cv_mean", r"$CV(T_e)$", "excursion_cv"),
    ]

    def draw_panel(ax, alpha: str, metric: str, ylabel: str, legend_fontsize: float):
        for label in ("control", "higher_order"):
            for vlow, (marker, ls, vlabel) in vlow_styles.items():
                subset = [row for row in rows if row["alpha"] == alpha and row["label"] == label and row["v_low"] == vlow]
                subset.sort(key=lambda row: f(row, "noise_intensity_dv"))
                x = [f(row, "noise_intensity_dv") for row in subset]
                y = [f(row, metric) for row in subset]
                ax.plot(
                    x,
                    y,
                    marker=marker,
                    ls=ls,
                    color=colors[label],
                    lw=2.0,
                    ms=5.6,
                    markeredgecolor=colors[label],
                    markeredgewidth=1.0,
                    label=rf"{labels[label]}, {vlabel}",
                )
                mark_optimal_point(ax, x, y, alpha, label, colors[label], marker=marker, size=7.5)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"$D_v$")
        ax.text(0.055, 0.92, rf"$\alpha={alpha}$", transform=ax.transAxes, fontsize=17, ha="left", va="top")
        format_prl_axis(ax)
        ax.legend(loc="best", fontsize=legend_fontsize, handlelength=2.0, borderaxespad=0.3)

    for alpha in ("1.5", "2.0"):
        suffix = "alpha15" if alpha == "1.5" else "alpha20"
        panel_names = ["a", "b", "c", "d"]
        for panel, (metric, ylabel, short_name) in zip(panel_names, metrics):
            fig, ax = plt.subplots(figsize=(5.2, 3.9))
            fig.subplots_adjust(left=0.18, right=0.97, bottom=0.18, top=0.96)
            draw_panel(ax, alpha, metric, ylabel, legend_fontsize=9.2)
            save_prl_figure(fig, out_dir, f"exp1_threshold_sensitivity_{suffix}_{panel}_{short_name}")

        fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.8))
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.96, hspace=0.30, wspace=0.25)
        for ax, (metric, ylabel, _) in zip(axes.ravel(), metrics):
            draw_panel(ax, alpha, metric, ylabel, legend_fontsize=6.5)
        save_prl_figure(fig, out_dir, f"exp1_threshold_sensitivity_{suffix}")


def plot_binned(out_dir: Path, rows):
    data = [row for row in rows if row["bin_size"] == "10.0"]
    metrics = [
        ("stcc_mean_mean", "Mean STCC"),
        ("simplex_stcc_mean_mean", "Within-simplex STCC"),
        ("synchrony_index_mean", "Synchrony index"),
        ("population_entropy_bits_mean", "Population entropy"),
        ("population_fano_mean", "Population Fano"),
        ("cascade_size_mean_mean", "Cascade size"),
    ]
    colors = {"control": "#36454f", "higher_order": "#c62828"}
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.2), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for alpha, marker in (("1.5", "o"), ("2.0", "s")):
            for label, ls in (("control", "--"), ("higher_order", "-")):
                subset = [row for row in data if row["alpha"] == alpha and row["label"] == label]
                subset.sort(key=lambda row: f(row, "noise_intensity_dv"))
                ax.plot(
                    [f(row, "noise_intensity_dv") for row in subset],
                    [f(row, metric) for row in subset],
                    marker + ls,
                    color=colors[label],
                    lw=1.3,
                    ms=3.2,
                    label=f"{label}, $\\alpha={alpha}$",
                )
        ax.set_xlabel(r"$D_v$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=6)
    fig.savefig(out_dir / "mechanism_synchrony_cascade_information.pdf")
    fig.savefig(out_dir / "mechanism_synchrony_cascade_information.png", dpi=300)
    plt.close(fig)


def plot_exp2_expected_figures(out_dir: Path, rows):
    data = [row for row in rows if row["bin_size"] == "10.0"]
    apply_prl_style()
    label_names = {"control": "pairwise", "higher_order": "simplicial"}
    condition_colors = {"control": "#36454f", "higher_order": "#c62828"}
    curve_specs = [
        ("stcc_mean", r"$\mathrm{STCC}$", "#222222", "o", "-"),
        ("simplex_stcc_mean", r"$\mathrm{STCC}_{\triangle}$", "#c62828", "s", "--"),
    ]
    chi_color = "#2b6cb0"

    def draw_sem_curve(ax, x, y, sem, color, marker, linestyle, label):
        ax.plot(
            x,
            y,
            marker=marker,
            linestyle=linestyle,
            color=color,
            lw=2.1,
            ms=5.8,
            markeredgecolor=color,
            markeredgewidth=1.0,
            label=label,
        )

    def format_right_axis(ax):
        ax.grid(False)
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    def compact_y_ticks(ax, values, *, sci_small: bool = False, nbins: int = 4):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
        data_range = float(np.nanmax(arr) - np.nanmin(arr)) if arr.size else np.nan
        if arr.size and sci_small and np.nanmax(np.abs(arr)) < 0.01:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value * 1e3:.1f}"))
            ax.text(
                0.015,
                0.985,
                r"$\times 10^{-3}$",
                transform=ax.transAxes,
                fontsize=13,
                ha="left",
                va="top",
            )
        elif arr.size and data_range < 0.005 and np.nanmax(np.abs(arr)) < 0.2:
            center = round(float(np.nanmean(arr)), 3)
            ticks = [center - 0.001, center, center + 0.001]
            ax.set_ylim(center - 0.0015, center + 0.0015)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        elif arr.size and np.nanmax(np.abs(arr)) < 0.02:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        elif arr.size and np.nanmax(np.abs(arr)) < 0.2:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    def draw_panel(ax, alpha: str, label: str, legend_fontsize: float):
        subset = [row for row in data if row["alpha"] == alpha and row["label"] == label]
        subset.sort(key=lambda row: f(row, "noise_intensity_dv"))
        x = [f(row, "noise_intensity_dv") for row in subset]
        stcc_values = []
        for metric, legend_label, color, marker, linestyle in curve_specs:
            y = [f(row, f"{metric}_mean") for row in subset]
            stcc_values.extend(y)
            y_sem = [sem_from_row(row, metric) for row in subset]
            draw_sem_curve(ax, x, y, y_sem, color, marker, linestyle, legend_label)

        ax2 = ax.twinx()
        chi = [f(row, "synchrony_index_mean") for row in subset]
        chi_sem = [sem_from_row(row, "synchrony_index") for row in subset]
        draw_sem_curve(ax2, x, chi, chi_sem, chi_color, "^", "-.", r"$\chi$")

        opt = OPTIMAL_DV[(alpha, label)]
        ax.axvline(opt, color=condition_colors[label], lw=0.9, ls=":", alpha=0.55, zorder=0)
        ax.set_xlabel(r"$D_v$")
        ax.set_ylabel("STCC")
        ax2.set_ylabel(r"$\chi$", color=chi_color)
        ax2.tick_params(axis="y", colors=chi_color)
        ax2.spines["right"].set_color(chi_color)
        ax.text(
            0.055,
            0.92,
            rf"{label_names[label]}, $\alpha={alpha}$",
            transform=ax.transAxes,
            fontsize=17,
            ha="left",
            va="top",
        )
        format_prl_axis(ax)
        format_right_axis(ax2)
        compact_y_ticks(ax, stcc_values, sci_small=True)
        compact_y_ticks(ax2, chi, nbins=3)

        lines, legend_labels = ax.get_legend_handles_labels()
        lines2, legend_labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines + lines2,
            legend_labels + legend_labels2,
            loc="best",
            fontsize=legend_fontsize,
            handlelength=2.0,
            borderaxespad=0.3,
        )

    panel_specs = [
        ("1.5", "control", "exp2_figureAB_a_stcc_chi_pairwise_alpha15"),
        ("2.0", "control", "exp2_figureAB_b_stcc_chi_pairwise_alpha20"),
        ("1.5", "higher_order", "exp2_figureAB_c_stcc_chi_simplicial_alpha15"),
        ("2.0", "higher_order", "exp2_figureAB_d_stcc_chi_simplicial_alpha20"),
    ]
    for alpha, label, name in panel_specs:
        fig, ax = plt.subplots(figsize=(5.2, 3.9))
        fig.subplots_adjust(left=0.18, right=0.82, bottom=0.18, top=0.96)
        draw_panel(ax, alpha, label, legend_fontsize=9.2)
        save_prl_figure(fig, out_dir, name)

    fig = plt.figure(figsize=(10.4, 7.8))
    panel_positions = [
        (0.10, 0.58, 0.31, 0.34),
        (0.58, 0.58, 0.31, 0.34),
        (0.10, 0.12, 0.31, 0.34),
        (0.58, 0.12, 0.31, 0.34),
    ]
    for position, (alpha, label, _) in zip(panel_positions, panel_specs):
        ax = fig.add_axes(position)
        draw_panel(ax, alpha, label, legend_fontsize=6.3)
    save_prl_figure(fig, out_dir, "exp2_figureAB_stcc_chi_vs_Dv")


def main():
    parser = argparse.ArgumentParser(description="Merge and plot mechanism supplemental outputs.")
    parser.add_argument("--output-root", default="paper_suite_mechanism_supplement_Dv")
    args = parser.parse_args()

    root = Path(args.output_root)
    root = root if root.is_absolute() else Path(__file__).resolve().parent / root
    out_dir = root / "mechanism_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    activation = collect_chunks(root, "activation_excursion_metrics.csv")
    binned = collect_chunks(root, "binned_synchrony_metrics.csv")
    if not activation or not binned:
        raise FileNotFoundError(f"No mechanism chunk CSV files found under {root / 'chunks'}")

    activation_agg = aggregate(
        activation,
        ["alpha", "label", "noise_intensity_dv", "v_low"],
        [
            "isi_cv",
            "activation_mean",
            "activation_std",
            "activation_cv",
            "excursion_mean",
            "excursion_std",
            "excursion_cv",
            "ta_te_corr",
            "activation_variance_fraction",
            "excursion_variance_fraction",
            "cycle_mean",
            "cycle_std",
            "cycle_cv",
            "spike_count_total",
            "paired_cycle_count",
        ],
    )
    binned_agg = aggregate(
        binned,
        ["alpha", "label", "noise_intensity_dv", "bin_size"],
        [
            "stcc_mean",
            "simplex_stcc_mean",
            "synchrony_variance_ratio",
            "synchrony_index",
            "population_count_mean",
            "population_fano",
            "population_entropy_bits",
            "cascade_count",
            "cascade_size_mean",
            "cascade_size_cv",
            "cascade_duration_mean",
            "cascade_duration_cv",
        ],
    )
    write_csv(out_dir / "activation_excursion_aggregated.csv", activation_agg)
    write_csv(out_dir / "binned_synchrony_aggregated.csv", binned_agg)
    plot_activation(out_dir, activation_agg)
    plot_exp1_mean_cv(out_dir, activation_agg)
    plot_exp1_variance_fraction(out_dir, activation_agg)
    plot_exp1_threshold_sensitivity(out_dir, activation_agg)
    plot_binned(out_dir, binned_agg)
    plot_exp2_expected_figures(out_dir, binned_agg)
    print(f"Wrote mechanism figures to {out_dir}")


if __name__ == "__main__":
    main()
