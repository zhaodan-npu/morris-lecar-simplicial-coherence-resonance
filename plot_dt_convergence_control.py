from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


SERIES = (
    {
        "key": "pairwise_dt005",
        "label": r"pairwise, $\Delta t=0.05$",
        "color": "#36454f",
        "marker": "o",
        "linestyle": "-",
    },
    {
        "key": "simplicial_dt005",
        "label": r"simplicial, $\Delta t=0.05$",
        "color": "#c62828",
        "marker": "s",
        "linestyle": "-",
    },
    {
        "key": "pairwise_dt001",
        "label": r"pairwise, $\Delta t=0.01$",
        "color": "#36454f",
        "marker": "o",
        "linestyle": "--",
    },
    {
        "key": "simplicial_dt001",
        "label": r"simplicial, $\Delta t=0.01$",
        "color": "#c62828",
        "marker": "s",
        "linestyle": "--",
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
            "savefig.bbox": "tight",
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


def build_series(main_rows: list[dict], dt_rows: list[dict], alpha: float) -> dict[str, list[dict]]:
    return {
        "pairwise_dt005": subset(main_rows, alpha, "control"),
        "simplicial_dt005": subset(main_rows, alpha, "higher_order"),
        "pairwise_dt001": subset(dt_rows, alpha, "control"),
        "simplicial_dt001": subset(dt_rows, alpha, "higher_order"),
    }


def configure_axis(ax, ylabel: str):
    ax.set_xscale("log")
    ax.set_xlim(0.035, 10.5)
    ax.set_xticks([0.04, 0.1, 0.3, 1.0, 4.0, 9.0])
    ax.set_xticklabels(["0.04", "0.10", "0.30", "1", "4", "9"])
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
        kwargs = {"dpi": 600} if ext == "png" else {}
        fig.savefig(out_dir / f"{name}.{ext}", **kwargs)
    plt.close(fig)


def plot_panel(data_by_key: dict[str, list[dict]], alpha: float, metric: str, ylabel: str, out_dir: Path, name: str):
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    fig.subplots_adjust(left=0.21, right=0.97, bottom=0.20, top=0.96)
    for spec in SERIES:
        data = data_by_key[spec["key"]]
        x = [row_dv(row) for row in data]
        y = [f(row, f"{metric}_mean") for row in data]
        yerr = [sem(row, metric) for row in data]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=spec["color"],
            marker=spec["marker"],
            linestyle=spec["linestyle"],
            linewidth=2.0,
            markersize=5.5,
            markeredgewidth=1.1,
            capsize=2.8,
            label=spec["label"],
        )
    ax.text(0.06, 0.92, rf"$\alpha={alpha:.1f}$", transform=ax.transAxes, ha="left", va="top")
    configure_axis(ax, ylabel)
    ax.legend(loc="best", handlelength=2.2, borderaxespad=0.25)
    save(fig, out_dir, name)


def write_best_summary(main_rows: list[dict], dt_rows: list[dict], out_dir: Path, min_spikes: float):
    rows = []
    for alpha in (1.5, 2.0):
        for name, source_rows, label, dt in (
            ("pairwise", main_rows, "control", 0.05),
            ("simplicial", main_rows, "higher_order", 0.05),
            ("pairwise", dt_rows, "control", 0.01),
            ("simplicial", dt_rows, "higher_order", 0.01),
        ):
            row = best_row(source_rows, alpha, label, min_spikes)
            if row is None:
                continue
            rows.append(
                {
                    "alpha": alpha,
                    "condition": name,
                    "dt": dt,
                    "best_Dv": row_dv(row),
                    "best_CV": f(row, "isi_cv_mean"),
                    "best_spike_count": f(row, "spike_count_total_mean"),
                    "total_realizations": f(row, "total_realizations"),
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "dt_convergence_best_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Plot dt=0.01 candidate39 convergence control against main dt=0.05.")
    parser.add_argument("--main-root", default="paper_suite_true_simplicial_array_prl_N100_Dv_refined")
    parser.add_argument("--dt-root", default="paper_suite_candidate39_dt001_N100_Dv_refined")
    parser.add_argument("--dt-subdir", default="candidate39_dt_0.010")
    parser.add_argument("--min-spikes", type=float, default=100.0)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    main_root = Path(args.main_root)
    if not main_root.is_absolute():
        main_root = script_dir / main_root
    dt_root = Path(args.dt_root)
    if not dt_root.is_absolute():
        dt_root = script_dir / dt_root

    main_rows = read_rows(main_root / "candidate39_main" / "aggregated_metrics.csv")
    dt_rows = read_rows(dt_root / args.dt_subdir / "aggregated_metrics.csv")
    out_dir = dt_root / "figures"

    apply_prl_style()
    for alpha_tag, alpha in (("alpha15", 1.5), ("alpha20", 2.0)):
        data = build_series(main_rows, dt_rows, alpha)
        plot_panel(data, alpha, "isi_cv", r"$CV$", out_dir, f"dt_convergence_cv_{alpha_tag}")
        plot_panel(data, alpha, "spike_count_total", r"$N_{\rm sp}$", out_dir, f"dt_convergence_spikes_{alpha_tag}")
    write_best_summary(main_rows, dt_rows, out_dir, args.min_spikes)
    print(f"Saved figures and summary to {out_dir}")


if __name__ == "__main__":
    main()
