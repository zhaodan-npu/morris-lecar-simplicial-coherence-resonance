from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import morris_lecar_paired_alpha_compare_simplicial as base
import morris_lecar_simplicial_paper_suite as suite


LABELS = ("control", "matched_pairwise", "higher_order")
DISPLAY = {
    "control": "Control",
    "matched_pairwise": "Matched pairwise",
    "higher_order": "Higher-order",
}
COLORS = {
    "control": "#666666",
    "matched_pairwise": "#1f77b4",
    "higher_order": "#c62828",
}
MARKERS = {"control": "o", "matched_pairwise": "s", "higher_order": "^"}
LINESTYLES = {"control": "--", "matched_pairwise": "-", "higher_order": "-"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row, key):
    value = row.get(key, "")
    return np.nan if value in ("", None, "None") else float(value)


def to_float(value):
    return np.nan if value in ("", None, "None") else float(value)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def merge_matched(root: Path, min_spikes: float):
    chunk_dirs = sorted((root / "chunks").glob("network_*"))
    if not chunk_dirs:
        raise FileNotFoundError(f"No chunk directories found under {root / 'chunks'}")

    all_rows = []
    network_rows = []
    params_dict = None
    for chunk_dir in chunk_dirs:
        all_rows.extend(read_json(chunk_dir / "all_realizations.json"))
        network_rows.extend(read_json(chunk_dir / "network_summary.json"))
        if params_dict is None:
            params_dict = read_json(chunk_dir / "parameters.json")

    params = base.PairedSimplicialParams(**params_dict)
    aggregated_rows = []
    for alpha in params.alphas:
        for noise_intensity_dv in params.noise_scales:
            subset = [
                row
                for row in all_rows
                if row["alpha"] == alpha
                and row["label"] == "matched_pairwise"
                and abs(row["noise_intensity_dv"] - noise_intensity_dv) < 1e-12
            ]
            scale = base.noise_scale_from_voltage_noise_intensity(noise_intensity_dv, alpha, params.capacitance)
            item = {
                "alpha": alpha,
                "label": "matched_pairwise",
                "noise_scale": scale,
                "noise_intensity_dv": noise_intensity_dv,
                "n_networks": len(chunk_dirs),
                "n_trials_per_network": params.n_trials_per_network,
                "total_realizations": len(subset),
            }
            for metric in (
                "isi_cv",
                "spike_count_total",
                "mean_activity_std",
                "edge_count",
                "clique_count",
                "simplex_count",
                "k1_actual",
                "k2_actual",
                "g1_original",
                "g2_original",
                "g_eff_target",
                "sigma1_matched",
            ):
                mean, std = base.summarize([row[metric] for row in subset])
                item[f"{metric}_mean"] = mean
                item[f"{metric}_std"] = std
            aggregated_rows.append(item)

    best = {}
    for alpha in sorted({row["alpha"] for row in aggregated_rows}):
        valid = [
            row
            for row in aggregated_rows
            if row["alpha"] == alpha
            and row["isi_cv_mean"] is not None
            and row["spike_count_total_mean"] is not None
            and row["spike_count_total_mean"] >= min_spikes
        ]
        row = min(valid, key=lambda item: item["isi_cv_mean"]) if valid else None
        best[str(alpha)] = None if row is None else {
            "best_noise_intensity_dv": row["noise_intensity_dv"],
            "best_noise_scale": row["noise_scale"],
            "best_cv_value": row["isi_cv_mean"],
            "best_spike_count": row["spike_count_total_mean"],
        }

    summary = {
        "parameters": params_dict,
        "n_chunk_networks": len(chunk_dirs),
        "best_by_alpha": {"matched_pairwise": best},
        "mean_edge_count": float(np.mean([row["edge_count"] for row in network_rows])),
        "mean_clique_count": float(np.mean([row["clique_count"] for row in network_rows])),
        "mean_simplex_count": float(np.mean([row["simplex_count"] for row in network_rows])),
        "mean_k1_actual": float(np.mean([row["k1_actual"] for row in network_rows])),
        "mean_k2_actual": float(np.mean([row["k2_actual"] for row in network_rows])),
        "mean_g_eff_target": float(np.mean([row["g_eff_target"] for row in network_rows])),
        "mean_sigma1_matched": float(np.mean([row["sigma1_matched"] for row in network_rows])),
    }

    (root / "network_summary.json").write_text(json.dumps(network_rows, indent=2), encoding="utf-8")
    (root / "aggregated_metrics.json").write_text(json.dumps(aggregated_rows, indent=2), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(root / "network_summary.csv", network_rows)
    write_csv(root / "aggregated_metrics.csv", aggregated_rows)
    return aggregated_rows, summary


def load_main_rows(main_root: Path):
    rows = read_csv(main_root / "candidate39_main" / "aggregated_metrics.csv")
    out = []
    for row in rows:
        if row["label"] not in {"control", "higher_order"}:
            continue
        out.append(
            {
                "alpha": f(row, "alpha"),
                "label": row["label"],
                "noise_intensity_dv": f(row, "noise_intensity_dv"),
                "isi_cv_mean": f(row, "isi_cv_mean"),
                "isi_cv_std": f(row, "isi_cv_std"),
                "spike_count_total_mean": f(row, "spike_count_total_mean"),
                "total_realizations": f(row, "total_realizations"),
            }
        )
    return out


def normalized_matched_rows(rows):
    return [
        {
            "alpha": to_float(row.get("alpha")),
            "label": row["label"],
            "noise_intensity_dv": to_float(row.get("noise_intensity_dv")),
            "isi_cv_mean": to_float(row.get("isi_cv_mean")),
            "isi_cv_std": to_float(row.get("isi_cv_std")),
            "spike_count_total_mean": to_float(row.get("spike_count_total_mean")),
            "total_realizations": to_float(row.get("total_realizations")),
        }
        for row in rows
    ]


def best_rows(rows, min_spikes):
    out = {}
    for alpha in sorted({row["alpha"] for row in rows}):
        out[str(alpha)] = {}
        for label in LABELS:
            valid = [
                row
                for row in rows
                if row["alpha"] == alpha
                and row["label"] == label
                and np.isfinite(row["isi_cv_mean"])
                and np.isfinite(row["spike_count_total_mean"])
                and row["spike_count_total_mean"] >= min_spikes
            ]
            out[str(alpha)][label] = min(valid, key=lambda row: row["isi_cv_mean"]) if valid else None
    return out


def apply_style():
    import matplotlib.pyplot as plt

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


def format_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks([0.04, 0.1, 0.25, 0.56, 1.0, 4.0, 9.0])
    ax.set_xticklabels(["0.04", "0.10", "0.25", "0.56", "1", "4", "9"])
    ax.tick_params(direction="in", top=True, right=True)


def plot_comparison(rows, out_dir: Path, min_spikes: float):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping matched-control figures.")
        return

    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    best = best_rows(rows, min_spikes)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True, sharey=True)
    for ax, alpha in zip(axes, (1.5, 2.0)):
        for label in LABELS:
            subset = [row for row in rows if abs(row["alpha"] - alpha) < 1e-12 and row["label"] == label]
            subset.sort(key=lambda row: row["noise_intensity_dv"])
            x = np.asarray([row["noise_intensity_dv"] for row in subset], dtype=float)
            y = np.asarray([row["isi_cv_mean"] for row in subset], dtype=float)
            std = np.asarray([row["isi_cv_std"] for row in subset], dtype=float)
            n = np.asarray([row["total_realizations"] for row in subset], dtype=float)
            spikes = np.asarray([row["spike_count_total_mean"] for row in subset], dtype=float)
            sem = np.divide(std, np.sqrt(n), out=np.full_like(std, np.nan), where=n > 0)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(sem) & (spikes >= min_spikes)
            ax.plot(
                x[valid],
                y[valid],
                marker=MARKERS[label],
                ls=LINESTYLES[label],
                color=COLORS[label],
                lw=1.35,
                ms=3.6,
                label=DISPLAY[label],
            )
            ax.fill_between(x[valid], y[valid] - sem[valid], y[valid] + sem[valid], color=COLORS[label], alpha=0.12, lw=0)
            best_row = best[str(alpha)].get(label)
            if best_row is not None:
                ax.plot(
                    [best_row["noise_intensity_dv"]],
                    [best_row["isi_cv_mean"]],
                    marker=MARKERS[label],
                    ms=6.8,
                    mfc="white",
                    mec=COLORS[label],
                    mew=1.1,
                    ls="none",
                )
        ax.text(0.04, 0.92, rf"$\alpha={alpha:.1f}$", transform=ax.transAxes, ha="left", va="top", fontsize=9)
        ax.set_xlabel(r"$D_v$")
        format_axis(ax)
    axes[0].set_ylabel("ISI CV")
    axes[0].legend(frameon=False, loc="best", handlelength=2.4)
    fig.savefig(out_dir / "matched_coupling_cv_vs_Dv.pdf")
    fig.savefig(out_dir / "matched_coupling_cv_vs_Dv.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.8), constrained_layout=True, sharey=True)
    for ax, alpha in zip(axes, (1.5, 2.0)):
        vals = [best[str(alpha)][label]["isi_cv_mean"] if best[str(alpha)][label] else np.nan for label in LABELS]
        bars = ax.bar(range(len(LABELS)), vals, color=[COLORS[label] for label in LABELS], width=0.62)
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(["Control", "Matched\npairwise", "HO"])
        ax.text(0.05, 0.92, rf"$\alpha={alpha:.1f}$", transform=ax.transAxes, ha="left", va="top", fontsize=9)
        for bar, value in zip(bars, vals):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
        control, matched, ho = vals
        if np.isfinite(control) and np.isfinite(matched) and np.isfinite(ho):
            ax.text(
                0.5,
                0.82,
                rf"$CV^*_{{HO}}-CV^*_{{matched}}={ho - matched:.3f}$",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7,
            )
        ax.tick_params(direction="in", top=True, right=True)
    axes[0].set_ylabel(r"Optimal ISI CV, $CV^\ast$")
    fig.savefig(out_dir / "matched_coupling_best_cv_bar.pdf")
    fig.savefig(out_dir / "matched_coupling_best_cv_bar.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Merge and plot matched effective-coupling control results.")
    parser.add_argument("--matched-root", default="paper_suite_matched_coupling_control_N100_Dv_refined")
    parser.add_argument("--main-root", default="paper_suite_true_simplicial_array_prl_N100_Dv_refined")
    parser.add_argument("--min-spikes", type=float, default=1000.0)
    args = parser.parse_args()

    root = Path(args.matched_root)
    root = root if root.is_absolute() else Path(__file__).resolve().parent / root
    main_root = Path(args.main_root)
    main_root = main_root if main_root.is_absolute() else Path(__file__).resolve().parent / main_root

    matched_rows, summary = merge_matched(root, args.min_spikes)
    comparison_rows = load_main_rows(main_root) + normalized_matched_rows(matched_rows)
    comparison_rows.sort(key=lambda row: (row["alpha"], row["label"], row["noise_intensity_dv"]))
    write_csv(root / "matched_control_comparison_aggregated.csv", comparison_rows)
    comparison_best = best_rows(comparison_rows, args.min_spikes)
    (root / "matched_control_comparison_summary.json").write_text(
        json.dumps(
            {
                "matched_summary": summary,
                "best_by_alpha": comparison_best,
                "min_spikes": args.min_spikes,
                "main_root": str(main_root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_comparison(comparison_rows, root / "matched_figures", args.min_spikes)
    print(f"Merged matched-control results saved to: {root}")


if __name__ == "__main__":
    main()
