from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


COLORS = {
    "sigma1": "#36454f",
    "sigma2": "#c62828",
}
MARKERS = {
    1.5: "o",
    2.0: "s",
}


def apply_prl_style():
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 16,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 11.5,
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


def configure_axes(ax):
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def save_figure(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        kwargs = {"bbox_inches": None}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(out_dir / f"{name}.{ext}", **kwargs)
    plt.close(fig)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sigma1_from_name(name: str) -> float:
    return float(name.split("_", 1)[1])


def sigma2_from_name(name: str) -> float:
    return float(name.split("_", 1)[1])


def collect_sigma1_rows(root: Path) -> list[dict]:
    merged = load_json(root / "merged_summary.json")
    array_jobs = load_json(root / "array_jobs.json")
    sigma_lookup = {item["name"]: float(item.get("sigma_1", sigma1_from_name(item["name"]))) for item in array_jobs}
    rows = []
    for exp_name, item in merged.items():
        sigma1 = sigma_lookup.get(exp_name, sigma1_from_name(exp_name))
        for alpha_key, alpha_data in item["best_by_alpha"].items():
            alpha = float(alpha_key)
            control = alpha_data["control"]
            if control is None:
                continue
            rows.append(
                {
                    "sweep": "sigma1",
                    "coupling": sigma1,
                    "alpha": alpha,
                    "cv": control["best_cv_value"],
                    "Dv": control["best_noise_intensity_dv"],
                    "spikes": control["best_spike_count"],
                }
            )
    return sorted(rows, key=lambda row: (row["alpha"], row["coupling"]))


def collect_sigma2_rows(root: Path) -> list[dict]:
    merged = load_json(root / "merged_summary.json")
    rows = []
    for exp_name, item in merged.items():
        if not exp_name.startswith("sigma2_"):
            continue
        sigma2 = sigma2_from_name(exp_name)
        for alpha_key, alpha_data in item["best_by_alpha"].items():
            alpha = float(alpha_key)
            higher = alpha_data["higher_order"]
            if higher is None:
                continue
            rows.append(
                {
                    "sweep": "sigma2",
                    "coupling": sigma2,
                    "alpha": alpha,
                    "cv": higher["best_cv_value"],
                    "Dv": higher["best_noise_intensity_dv"],
                    "spikes": higher["best_spike_count"],
                }
            )
    return sorted(rows, key=lambda row: (row["alpha"], row["coupling"]))


def write_summary(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sweep", "coupling", "alpha", "cv", "Dv", "spikes"])
        writer.writeheader()
        writer.writerows(rows)


def configure_coupling_axis(ax, ylabel: str, ylim: tuple[float, float], yticks: list[float]):
    configure_axes(ax)
    ax.set_xlim(-0.005, 0.165)
    ax.set_xticks([0.00, 0.04, 0.08, 0.12, 0.16])
    ax.set_xticklabels(["0", "0.04", "0.08", "0.12", "0.16"])
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xlabel(r"coupling strength")
    ax.set_ylabel(ylabel)


def plot_sigma1_metric(rows: list[dict], metric: str, ylabel: str, out_dir: Path, name: str):
    values = [row[metric] for row in rows if math.isfinite(row[metric])]
    if metric == "cv":
        ylim = (0.0, max(0.85, math.ceil(max(values) * 10.0) / 10.0))
        yticks = [0.0, 0.2, 0.4, 0.6, 0.8] if ylim[1] <= 0.9 else [0.0, 0.5, 1.0, 1.5, 2.0]
    else:
        ylim = (0.0, max(1.05, math.ceil(max(values) * 4.0) / 4.0))
        yticks = [0.0, 0.25, 0.50, 0.75, 1.00] if ylim[1] <= 1.05 else [0.0, 0.5, 1.0, 1.5]

    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.18, top=0.96)
    for alpha in (1.5, 2.0):
        data = [row for row in rows if abs(row["alpha"] - alpha) < 1e-12]
        markerfacecolor = COLORS["sigma1"] if alpha == 1.5 else "white"
        ax.plot(
            [row["coupling"] for row in data],
            [row[metric] for row in data],
            marker=MARKERS[alpha],
            color=COLORS["sigma1"],
            linewidth=2.2,
            markersize=6.4,
            markerfacecolor=markerfacecolor,
            markeredgecolor=COLORS["sigma1"],
            markeredgewidth=1.3,
            label=rf"$\alpha={alpha:.1f}$",
        )
    configure_coupling_axis(ax, ylabel, ylim, yticks)
    ax.set_xlabel(r"$\sigma_1$")
    ax.legend(loc="best", handlelength=2.2, borderaxespad=0.3)
    save_figure(fig, out_dir, name)


def plot_coupling_comparison(sigma1_rows: list[dict], sigma2_rows: list[dict], metric: str, ylabel: str, out_dir: Path, name: str):
    rows = sigma1_rows + sigma2_rows
    values = [row[metric] for row in rows if math.isfinite(row[metric])]
    if metric == "cv":
        ylim = (0.0, max(0.85, math.ceil(max(values) * 10.0) / 10.0))
        yticks = [0.0, 0.2, 0.4, 0.6, 0.8] if ylim[1] <= 0.9 else [0.0, 0.5, 1.0, 1.5, 2.0]
    else:
        ylim = (0.0, max(1.05, math.ceil(max(values) * 4.0) / 4.0))
        yticks = [0.0, 0.25, 0.50, 0.75, 1.00] if ylim[1] <= 1.05 else [0.0, 0.5, 1.0, 1.5]

    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.18, top=0.96)
    for sweep, color, linestyle, label_prefix in (
        ("sigma1", COLORS["sigma1"], "--", r"vary $\sigma_1$, $\sigma_2=0$"),
        ("sigma2", COLORS["sigma2"], "-", r"vary $\sigma_2$, $\sigma_1=0.002$"),
    ):
        source = sigma1_rows if sweep == "sigma1" else sigma2_rows
        for alpha in (1.5, 2.0):
            data = [row for row in source if abs(row["alpha"] - alpha) < 1e-12]
            markerfacecolor = color if alpha == 1.5 else "white"
            ax.plot(
                [row["coupling"] for row in data],
                [row[metric] for row in data],
                marker=MARKERS[alpha],
                linestyle=linestyle,
                color=color,
                linewidth=2.2,
                markersize=6.4,
                markerfacecolor=markerfacecolor,
                markeredgecolor=color,
                markeredgewidth=1.3,
                label=rf"{label_prefix}, $\alpha={alpha:.1f}$",
            )
    configure_coupling_axis(ax, ylabel, ylim, yticks)
    ax.legend(loc="best", handlelength=2.0, borderaxespad=0.3, fontsize=8.5)
    save_figure(fig, out_dir, name)


def main():
    parser = argparse.ArgumentParser(description="Plot sigma_1 sweep and compare it with the sigma_2 sweep.")
    parser.add_argument("--sigma1-root", default="paper_suite_sigma1_sweep_pairwise_N100_Dv_refined")
    parser.add_argument("--sigma2-root", default="paper_suite_true_simplicial_array_prl_N100_Dv_refined")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    sigma1_root = Path(args.sigma1_root)
    if not sigma1_root.is_absolute():
        sigma1_root = script_dir / sigma1_root
    sigma2_root = Path(args.sigma2_root)
    if not sigma2_root.is_absolute():
        sigma2_root = script_dir / sigma2_root
    out_dir = Path(args.output_dir) if args.output_dir else sigma1_root / "figures"
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir

    apply_prl_style()
    sigma1_rows = collect_sigma1_rows(sigma1_root)
    write_summary(sigma1_rows, out_dir / "sigma1_sweep_best_summary.csv")
    plot_sigma1_metric(sigma1_rows, "cv", "CV", out_dir, "fig_sigma1_sweep_cv_all_alpha")
    plot_sigma1_metric(sigma1_rows, "Dv", r"$D_v^*$", out_dir, "fig_sigma1_sweep_Dv_all_alpha")

    if (sigma2_root / "merged_summary.json").exists():
        sigma2_rows = collect_sigma2_rows(sigma2_root)
        write_summary(sigma1_rows + sigma2_rows, out_dir / "sigma1_vs_sigma2_best_summary.csv")
        plot_coupling_comparison(sigma1_rows, sigma2_rows, "cv", "CV", out_dir, "fig_sigma1_vs_sigma2_cv_all_alpha")
        plot_coupling_comparison(sigma1_rows, sigma2_rows, "Dv", r"$D_v^*$", out_dir, "fig_sigma1_vs_sigma2_Dv_all_alpha")

    print(f"Saved sigma_1 sweep figures to {out_dir}")


if __name__ == "__main__":
    main()
