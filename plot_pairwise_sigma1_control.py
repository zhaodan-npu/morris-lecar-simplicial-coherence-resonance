from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


LINE_SPECS = (
    {
        "name": "pairwise",
        "legend": "Pairwise",
        "color": "#36454f",
        "marker": "o",
        "prefix": "P",
    },
    {
        "name": "simplicial",
        "legend": "Simplicial",
        "color": "#c62828",
        "marker": "s",
        "prefix": "S",
    },
)


def apply_prl_style():
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 16,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 14,
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
    return float(value) if value not in ("", None, "None") else float("nan")


def sem(row: dict, metric: str) -> float:
    std = f(row, f"{metric}_std")
    n = f(row, "total_realizations")
    if not math.isfinite(std) or not math.isfinite(n) or n <= 0:
        return float("nan")
    return std / math.sqrt(n)


def row_dv(row: dict) -> float:
    return f(row, "noise_intensity_dv")


def subset(rows: list[dict], alpha: float, label: str) -> list[dict]:
    selected = [row for row in rows if abs(f(row, "alpha") - alpha) < 1e-12 and row["label"] == label]
    return sorted(selected, key=row_dv)


def configure_axis(
    ax,
    ylabel: str,
    ylim: tuple[float, float],
    yticks: list[float],
    yticklabels: list[str] | None = None,
):
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
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def best_row(rows: list[dict], alpha: float, label: str, min_spikes: float):
    data = [
        row
        for row in subset(rows, alpha, label)
        if math.isfinite(f(row, "isi_cv_mean"))
        and math.isfinite(f(row, "spike_count_total_mean"))
        and f(row, "spike_count_total_mean") >= min_spikes
    ]
    return min(data, key=lambda row: f(row, "isi_cv_mean")) if data else None


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        kwargs = {"dpi": 600, "bbox_inches": None} if ext == "png" else {"bbox_inches": None}
        fig.savefig(path, **kwargs)
    plt.close(fig)


def format_dv(value: float) -> str:
    if abs(value - round(value)) < 1e-10:
        return f"{value:.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def plot_group_panel(
    group: dict,
    alpha: float,
    metric: str,
    ylabel: str,
    ylim: tuple[float, float],
    yticks: list[float],
    yticklabels: list[str] | None,
    annotate_cv: bool,
    min_spikes: float,
    out_dir: Path,
    name: str,
):
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.18, top=0.96)
    annotation_lines = []
    for spec in LINE_SPECS:
        data = subset(group["source"][spec["name"]], alpha, group["data_labels"][spec["name"]])
        x = [row_dv(row) for row in data]
        y = [f(row, f"{metric}_mean") for row in data]
        yerr = [sem(row, metric) for row in data]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=spec["color"],
            marker=spec["marker"],
            linewidth=2.2,
            markersize=6.2,
            markeredgewidth=1.2,
            capsize=3.2,
            label=spec["legend"],
        )
        best = best_row(group["source"][spec["name"]], alpha, group["data_labels"][spec["name"]], min_spikes)
        if best is not None:
            best_x = row_dv(best)
            best_y = f(best, f"{metric}_mean")
            ax.axvline(best_x, color=spec["color"], linestyle=":", linewidth=1.25, alpha=0.72)
            ax.plot(
                [best_x],
                [best_y],
                marker=spec["marker"],
                markersize=9.0,
                markerfacecolor="white",
                markeredgecolor=spec["color"],
                markeredgewidth=1.6,
                linestyle="none",
                zorder=6,
            )
            if annotate_cv:
                annotation_lines.append(
                    (
                        spec["color"],
                        rf"{spec['prefix']}: $D_v^*={format_dv(best_x)}$, $CV^*={f(best, 'isi_cv_mean'):.3f}$",
                    )
                )
    ax.text(
        0.055,
        0.90,
        rf"$\alpha={alpha:.1f}$" + "\n" + group["label"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        linespacing=1.15,
        fontsize=18,
    )
    configure_axis(ax, ylabel, ylim, yticks, yticklabels)
    ax.legend(loc="upper right", handlelength=2.2, borderaxespad=0.3)
    if annotate_cv:
        for idx, (color, text) in enumerate(annotation_lines):
            ax.text(
                0.055,
                0.67 - 0.09 * idx,
                text,
                transform=ax.transAxes,
                color=color,
                fontsize=12.5,
                ha="left",
                va="top",
            )
    save(fig, out_dir, name)


def write_best_summary(groups: dict[str, dict], out_dir: Path, min_spikes: float):
    rows = []
    for group_name, group in groups.items():
        for alpha in (1.5, 2.0):
            for series_name, label, sigma2 in (
                ("pairwise", "control", 0.0),
                ("simplicial", "higher_order", 0.12),
            ):
                row = best_row(group["source"][series_name], alpha, label, min_spikes)
                if row is None:
                    continue
                rows.append(
                    {
                        "group": group_name,
                        "alpha": alpha,
                        "sigma_1": group["sigma_1"],
                        "series": series_name,
                        "sigma_2": sigma2,
                        "best_Dv": row_dv(row),
                        "best_CV": f(row, "isi_cv_mean"),
                        "best_spike_count": f(row, "spike_count_total_mean"),
                        "total_realizations": f(row, "total_realizations"),
                    }
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "sigma1_grouped_control_best_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Appendix plot: compare sigma_2=0 and sigma_2=0.12 within each sigma_1 baseline."
    )
    parser.add_argument(
        "--main-root",
        default="paper_suite_true_simplicial_array_prl_N100_Dv_refined",
        help="Root containing candidate39_main for sigma_1=0.002.",
    )
    parser.add_argument(
        "--strong-pairwise-root",
        default="paper_suite_pairwise_sigma1_012_N100_Dv_refined",
        help="Root containing sigma_1=0.12, sigma_2=0.",
    )
    parser.add_argument("--strong-pairwise-subdir", default="sigma1_0.120_sigma2_0.000")
    parser.add_argument(
        "--strong-simplicial-root",
        default="paper_suite_sigma1_012_sigma2_012_N100_Dv_refined",
        help="Root containing sigma_1=0.12, sigma_2=0.12.",
    )
    parser.add_argument("--strong-simplicial-subdir", default="sigma1_0.120_sigma2_0.120")
    parser.add_argument("--output-root", default="paper_suite_sigma1_grouped_appendix")
    parser.add_argument("--min-spikes", type=float, default=100.0)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    main_root = Path(args.main_root)
    if not main_root.is_absolute():
        main_root = script_dir / main_root
    strong_pairwise_root = Path(args.strong_pairwise_root)
    if not strong_pairwise_root.is_absolute():
        strong_pairwise_root = script_dir / strong_pairwise_root
    strong_simplicial_root = Path(args.strong_simplicial_root)
    if not strong_simplicial_root.is_absolute():
        strong_simplicial_root = script_dir / strong_simplicial_root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = script_dir / output_root

    main_rows = read_rows(main_root / "candidate39_main" / "aggregated_metrics.csv")
    strong_pairwise_rows = read_rows(strong_pairwise_root / args.strong_pairwise_subdir / "aggregated_metrics.csv")
    strong_simplicial_rows = read_rows(strong_simplicial_root / args.strong_simplicial_subdir / "aggregated_metrics.csv")
    out_dir = output_root / "figures"

    groups = {
        "sigma1_0.002": {
            "sigma_1": 0.002,
            "label": r"$\sigma_1=0.002$",
            "source": {
                "pairwise": main_rows,
                "simplicial": main_rows,
            },
            "data_labels": {
                "pairwise": "control",
                "simplicial": "higher_order",
            },
        },
        "sigma1_0.12": {
            "sigma_1": 0.12,
            "label": r"$\sigma_1=0.12$",
            "source": {
                "pairwise": strong_pairwise_rows,
                "simplicial": strong_simplicial_rows,
            },
            "data_labels": {
                "pairwise": "control",
                "simplicial": "higher_order",
            },
        },
    }

    apply_prl_style()
    cv_ylim = (0.0, 2.1)
    cv_ticks = [0.0, 0.5, 1.0, 1.5, 2.0]
    spike_ylim = (0.0, 60000.0)
    spike_ticks = [0, 20000, 40000, 60000]
    spike_ticklabels = ["0", "20", "40", "60"]
    for group_name, group in groups.items():
        for alpha_tag, alpha in (("alpha15", 1.5), ("alpha20", 2.0)):
            plot_group_panel(
                group,
                alpha,
                "isi_cv",
                r"$CV$",
                cv_ylim,
                cv_ticks,
                None,
                True,
                args.min_spikes,
                out_dir,
                f"{group_name}_cv_{alpha_tag}",
            )
            plot_group_panel(
                group,
                alpha,
                "spike_count_total",
                r"Spike count ($10^3$)",
                spike_ylim,
                spike_ticks,
                spike_ticklabels,
                False,
                args.min_spikes,
                out_dir,
                f"{group_name}_spikes_{alpha_tag}",
            )
    write_best_summary(groups, out_dir, args.min_spikes)
    print(f"Saved appendix figures and summary to {out_dir}")


if __name__ == "__main__":
    main()
