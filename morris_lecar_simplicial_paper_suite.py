from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.dont_write_bytecode = True

import morris_lecar_paired_alpha_compare_simplicial as base


@dataclass
class PaperSuiteConfig:
    n_nodes: int = 100
    n_networks: int = 30
    n_trials_per_network: int = 12
    p_edge: float = 0.15
    p_fill_simplex: float = 0.5
    dt: float = 0.05
    t_total: float = 15000.0
    t_warmup: float = 3000.0
    alphas: tuple[float, ...] = (1.5, 2.0)
    beta: float = 0.0
    base_seed: int = 20260501
    spike_threshold_for_validity: float = 5.0


REFINED_DV_VALUES = (
    0.04,
    0.06,
    0.09,
    0.12,
    0.16,
    0.20,
    0.25,
    0.30,
    0.40,
    0.56,
    0.75,
    1.00,
    1.25,
    1.56,
    1.89,
    2.25,
    3.06,
    4.00,
    6.25,
    9.00,
)

REFINED_SIGMA2_SWEEP = tuple(round(0.01 * idx, 2) for idx in range(17))


def aggregate_rows(rows: list[dict], params: base.PairedSimplicialParams):
    aggregated_rows = []
    for alpha in params.alphas:
        for label in ("control", "higher_order"):
            for noise_intensity_dv in params.noise_scales:
                scale = base.noise_scale_from_voltage_noise_intensity(noise_intensity_dv, alpha, params.capacitance)
                subset = [
                    row
                    for row in rows
                    if row["alpha"] == alpha
                    and row["label"] == label
                    and abs(row["noise_intensity_dv"] - noise_intensity_dv) < 1e-12
                ]
                aggregated = {
                    "alpha": alpha,
                    "label": label,
                    "noise_scale": scale,
                    "noise_intensity_dv": noise_intensity_dv,
                    "n_networks": params.n_networks,
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
                ):
                    mean, std = base.summarize([row[metric] for row in subset])
                    aggregated[f"{metric}_mean"] = mean
                    aggregated[f"{metric}_std"] = std
                aggregated_rows.append(aggregated)
    return aggregated_rows


def best_rows_by_alpha(aggregated_rows: list[dict], min_spikes: float):
    out = {}
    for alpha in sorted({row["alpha"] for row in aggregated_rows}):
        out[str(alpha)] = {}
        for label in ("control", "higher_order"):
            valid = [
                row
                for row in aggregated_rows
                if row["alpha"] == alpha
                and row["label"] == label
                and row["isi_cv_mean"] is not None
                and row["spike_count_total_mean"] is not None
                and row["spike_count_total_mean"] >= min_spikes
            ]
            best = min(valid, key=lambda row: row["isi_cv_mean"]) if valid else None
            out[str(alpha)][label] = None if best is None else {
                "best_noise_scale": best["noise_scale"],
                "best_noise_intensity_dv": best["noise_intensity_dv"],
                "best_cv_value": best["isi_cv_mean"],
                "best_spike_count": best["spike_count_total_mean"],
            }
    return out


def run_paired_experiment(
    params: base.PairedSimplicialParams,
    output_dir: Path,
    save_all_rows: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    network_rows = []

    for network_idx in range(params.n_networks):
        network_rng = base.np.random.default_rng(params.base_seed + base.stable_seed("paper_network", network_idx))
        network = base.build_simplicial_network(params, network_rng)
        network_rows.append(
            {
                "network_idx": network_idx,
                "edge_count": network.edge_count,
                "clique_count": network.clique_count,
                "simplex_count": network.simplex_count,
            }
        )
        print(
            f"network {network_idx + 1:02d}/{params.n_networks}: "
            f"edges={network.edge_count}, 3-cliques={network.clique_count}, simplices={network.simplex_count}"
        )

        for alpha in params.alphas:
            for noise_idx, noise_intensity_dv in enumerate(params.noise_scales):
                scale = base.noise_scale_from_voltage_noise_intensity(noise_intensity_dv, alpha, params.capacitance)
                for trial_idx in range(params.n_trials_per_network):
                    init_seed = params.base_seed + base.stable_seed("init", network_idx, trial_idx)
                    noise_seed = params.base_seed + base.stable_seed("noise", alpha, noise_idx, network_idx, trial_idx)

                    for label, higher in (("control", False), ("higher_order", True)):
                        metrics = base.simulate_batch(
                            params,
                            network,
                            alpha=alpha,
                            noise_scale=scale,
                            higher_order_enabled=higher,
                            init_seed=init_seed,
                            noise_seed=noise_seed,
                        )
                        for path_idx in range(params.n_trials_per_network):
                            all_rows.append(
                                {
                                    "alpha": alpha,
                                    "label": label,
                                    "network_idx": network_idx,
                                    "trial_idx": trial_idx,
                                    "path_idx": path_idx,
                                    "noise_scale": scale,
                                    "noise_intensity_dv": noise_intensity_dv,
                                    "isi_cv": float(metrics["isi_cv"][path_idx]),
                                    "spike_count_total": float(metrics["spike_count_total"][path_idx]),
                                    "mean_activity_std": float(metrics["mean_activity_std"][path_idx]),
                                    "edge_count": float(metrics["edge_count"][path_idx]),
                                    "clique_count": float(metrics["clique_count"][path_idx]),
                                    "simplex_count": float(metrics["simplex_count"][path_idx]),
                                }
                            )

    aggregated_rows = aggregate_rows(all_rows, params)
    summary = {
        "parameters": asdict(params),
        "best_by_alpha": best_rows_by_alpha(aggregated_rows, min_spikes=5.0),
        "mean_edge_count": float(base.np.mean([row["edge_count"] for row in network_rows])),
        "mean_clique_count": float(base.np.mean([row["clique_count"] for row in network_rows])),
        "mean_simplex_count": float(base.np.mean([row["simplex_count"] for row in network_rows])),
    }

    (output_dir / "parameters.json").write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")
    (output_dir / "network_summary.json").write_text(json.dumps(network_rows, indent=2), encoding="utf-8")
    (output_dir / "aggregated_metrics.json").write_text(json.dumps(aggregated_rows, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    base.write_csv(output_dir / "network_summary.csv", network_rows)
    base.write_csv(output_dir / "aggregated_metrics.csv", aggregated_rows)

    if save_all_rows:
        (output_dir / "all_realizations.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    if should_make_plots():
        render_metrics_plot(output_dir / "metrics.png", aggregated_rows, output_dir.name)
    else:
        print(f"Data-only mode; skipping plot: {output_dir / 'metrics.png'}")
    return aggregated_rows, summary


def should_make_plots() -> bool:
    return os.environ.get("ML_MAKE_PLOTS", "").lower() in {"1", "true", "yes", "on"}


def render_metrics_plot(path: Path, aggregated_rows: list[dict], title: str):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(f"matplotlib is not installed; skipping plot: {path}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    color_map = {"control": "#455a64", "higher_order": "#c62828"}
    alphas = sorted({row["alpha"] for row in aggregated_rows})

    for col_idx, alpha in enumerate(alphas):
        subset_alpha = [row for row in aggregated_rows if row["alpha"] == alpha]
        for label in ("control", "higher_order"):
            rows = [row for row in subset_alpha if row["label"] == label]
            x = [row["noise_intensity_dv"] for row in rows]
            cv = [base.np.nan if row["isi_cv_mean"] is None else row["isi_cv_mean"] for row in rows]
            cv_err = [0.0 if row["isi_cv_std"] is None else row["isi_cv_std"] for row in rows]
            spikes = [row["spike_count_total_mean"] for row in rows]
            spike_err = [0.0 if row["spike_count_total_std"] is None else row["spike_count_total_std"] for row in rows]

            axes[0, col_idx].errorbar(x, cv, yerr=cv_err, fmt="o-", capsize=3, linewidth=1.8, color=color_map[label], label=label)
            axes[1, col_idx].errorbar(x, spikes, yerr=spike_err, fmt="o-", capsize=3, linewidth=1.8, color=color_map[label], label=label)

        axes[0, col_idx].set_title(f"alpha = {alpha}")
        axes[0, col_idx].set_ylabel("ISI CV")
        axes[1, col_idx].set_ylabel("Spike count")
        axes[1, col_idx].set_xlabel(r"$D_v$")
        axes[0, col_idx].grid(alpha=0.3)
        axes[1, col_idx].grid(alpha=0.3)
        axes[0, col_idx].legend(frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def candidate39_params(config: PaperSuiteConfig):
    return base.PairedSimplicialParams(
        n_nodes=config.n_nodes,
        n_networks=config.n_networks,
        n_trials_per_network=config.n_trials_per_network,
        p_edge=config.p_edge,
        p_fill_simplex=config.p_fill_simplex,
        sigma_1=0.002,
        sigma_2=0.12,
        current=88.0,
        dt=config.dt,
        t_total=config.t_total,
        t_warmup=config.t_warmup,
        alphas=config.alphas,
        beta=config.beta,
        noise_scales=REFINED_DV_VALUES,
        base_seed=config.base_seed + 39000,
    )


def candidate42_params(config: PaperSuiteConfig):
    return base.PairedSimplicialParams(
        n_nodes=config.n_nodes,
        n_networks=config.n_networks,
        n_trials_per_network=config.n_trials_per_network,
        p_edge=config.p_edge,
        p_fill_simplex=config.p_fill_simplex,
        sigma_1=0.005,
        sigma_2=0.12,
        current=88.0,
        dt=config.dt,
        t_total=config.t_total,
        t_warmup=config.t_warmup,
        alphas=config.alphas,
        beta=config.beta,
        noise_scales=REFINED_DV_VALUES,
        base_seed=config.base_seed + 42000,
    )


def run_main_compare(config: PaperSuiteConfig, root: Path):
    print("Running main comparison around candidate_39...")
    params = candidate39_params(config)
    return run_paired_experiment(params, root / "main_compare_candidate39", save_all_rows=False)


def run_candidate_compare(config: PaperSuiteConfig, root: Path):
    print("Running candidate_39 vs candidate_42 comparison...")
    run_paired_experiment(candidate39_params(config), root / "candidate39_compare", save_all_rows=False)
    run_paired_experiment(candidate42_params(config), root / "candidate42_compare", save_all_rows=False)


def run_sigma2_sweep(config: PaperSuiteConfig, root: Path):
    print("Running sigma2 sweep...")
    summary_rows = []
    for sigma2 in (0.0, 0.04, 0.08, 0.12):
        params = base.PairedSimplicialParams(
            n_nodes=config.n_nodes,
            n_networks=config.n_networks,
            n_trials_per_network=config.n_trials_per_network,
            p_edge=config.p_edge,
            p_fill_simplex=config.p_fill_simplex,
            sigma_1=0.002,
            sigma_2=sigma2,
            current=88.0,
            dt=config.dt,
            t_total=config.t_total,
            t_warmup=config.t_warmup,
            alphas=config.alphas,
            beta=config.beta,
            noise_scales=REFINED_DV_VALUES,
            base_seed=config.base_seed + int(1000 * sigma2) + 50000,
        )
        _, summary = run_paired_experiment(params, root / f"sigma2_{sigma2:0.2f}", save_all_rows=False)
        summary_rows.append(
            {
                "sigma_2": sigma2,
                "summary": summary,
            }
        )
    (root / "sigma2_sweep_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")


def run_current_robustness(config: PaperSuiteConfig, root: Path):
    print("Running current robustness...")
    summary_rows = []
    for current in (87.0, 88.0, 89.0):
        params = base.PairedSimplicialParams(
            n_nodes=config.n_nodes,
            n_networks=config.n_networks,
            n_trials_per_network=config.n_trials_per_network,
            p_edge=config.p_edge,
            p_fill_simplex=config.p_fill_simplex,
            sigma_1=0.002,
            sigma_2=0.12,
            current=current,
            dt=config.dt,
            t_total=config.t_total,
            t_warmup=config.t_warmup,
            alphas=config.alphas,
            beta=config.beta,
            noise_scales=REFINED_DV_VALUES,
            base_seed=config.base_seed + int(100 * current) + 60000,
        )
        _, summary = run_paired_experiment(params, root / f"current_{current:0.1f}", save_all_rows=False)
        summary_rows.append(
            {
                "current": current,
                "summary": summary,
            }
        )
    (root / "current_robustness_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Formal paper-oriented experiment suite for true simplicial Morris-Lecar CR.")
    parser.add_argument(
        "--mode",
        default="full_suite",
        choices=("full_suite", "main_compare", "candidate_compare", "sigma2_sweep", "current_robustness"),
        help="Which experiment block to run.",
    )
    parser.add_argument(
        "--output-root",
        default="paper_suite_true_simplicial",
        help="Output folder name, created under the script directory.",
    )
    parser.add_argument("--n-nodes", type=int, default=100, help="Number of Morris-Lecar nodes.")
    parser.add_argument("--n-networks", type=int, default=30, help="Number of network realizations.")
    parser.add_argument("--n-trials", type=int, default=12, help="Number of trial batches per network. Each batch contains this many paths.")
    parser.add_argument("--t-total", type=float, default=15000.0, help="Total simulation time.")
    parser.add_argument("--t-warmup", type=float, default=3000.0, help="Transient length to discard.")
    parser.add_argument("--seed", type=int, default=20260501, help="Base random seed.")
    args = parser.parse_args()

    config = PaperSuiteConfig(
        n_nodes=args.n_nodes,
        n_networks=args.n_networks,
        n_trials_per_network=args.n_trials,
        t_total=args.t_total,
        t_warmup=args.t_warmup,
        base_seed=args.seed,
    )

    output_root = Path(args.output_root)
    root = output_root if output_root.is_absolute() else Path(__file__).resolve().parent / output_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "suite_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    if args.mode in ("full_suite", "main_compare"):
        run_main_compare(config, root)
    if args.mode in ("full_suite", "candidate_compare"):
        run_candidate_compare(config, root)
    if args.mode in ("full_suite", "sigma2_sweep"):
        run_sigma2_sweep(config, root)
    if args.mode in ("full_suite", "current_robustness"):
        run_current_robustness(config, root)

    print(f"Paper suite results saved to: {root}")


if __name__ == "__main__":
    main()
