from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.dont_write_bytecode = True

import morris_lecar_paired_alpha_compare_simplicial as base


SIGMA2_VALUES = (
    0.00,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
    0.16,
)
DV_VALUES = (
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


def make_params(args, sigma2: float) -> base.PairedSimplicialParams:
    return base.PairedSimplicialParams(
        n_nodes=args.n_nodes,
        n_networks=args.n_networks,
        n_trials_per_network=args.n_trials,
        p_edge=0.15,
        p_fill_simplex=0.5,
        sigma_1=0.002,
        sigma_2=sigma2,
        current=88.0,
        dt=0.05,
        t_total=args.t_total,
        t_warmup=args.t_warmup,
        alphas=(1.5, 2.0),
        beta=0.0,
        noise_scales=DV_VALUES,
        base_seed=args.seed + int(round(100000 * sigma2)) + 71000,
    )


def run_one_network(params: base.PairedSimplicialParams, network_idx: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    network_rng = base.np.random.default_rng(params.base_seed + base.stable_seed("phase_network", network_idx))
    network = base.build_simplicial_network(params, network_rng)
    network_row = {
        "network_idx": network_idx,
        "edge_count": network.edge_count,
        "clique_count": network.clique_count,
        "simplex_count": network.simplex_count,
    }

    all_rows = []
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
                                "sigma_2": params.sigma_2,
                                "isi_cv": float(metrics["isi_cv"][path_idx]),
                                "spike_count_total": float(metrics["spike_count_total"][path_idx]),
                                "mean_activity_std": float(metrics["mean_activity_std"][path_idx]),
                                "edge_count": float(metrics["edge_count"][path_idx]),
                                "clique_count": float(metrics["clique_count"][path_idx]),
                                "simplex_count": float(metrics["simplex_count"][path_idx]),
                            }
                        )

    (output_dir / "parameters.json").write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")
    (output_dir / "network_summary.json").write_text(json.dumps([network_row], indent=2), encoding="utf-8")
    (output_dir / "all_realizations.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    base.write_csv(output_dir / "network_summary.csv", [network_row])
    base.write_csv(output_dir / "all_realizations.csv", all_rows)
    return network_row, all_rows


def main():
    parser = argparse.ArgumentParser(description="Sigma2-noise phase diagram array run for true simplicial Morris-Lecar CR.")
    parser.add_argument("task_id", type=int, help="Slurm array task id.")
    parser.add_argument("n_tasks", type=int, help="Total number of Slurm array tasks.")
    parser.add_argument("--output-root", default="paper_suite_sigma2_Dv_phase_refined_N100", help="Output root.")
    parser.add_argument("--n-nodes", type=int, default=100)
    parser.add_argument("--n-networks", type=int, default=30)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--t-total", type=float, default=15000.0)
    parser.add_argument("--t-warmup", type=float, default=3000.0)
    parser.add_argument("--seed", type=int, default=20260509)
    args = parser.parse_args()

    expected_tasks = len(SIGMA2_VALUES) * args.n_networks
    if args.n_tasks != expected_tasks:
        raise ValueError(f"Expected n_tasks={expected_tasks}, got {args.n_tasks}.")
    if args.task_id < 0 or args.task_id >= expected_tasks:
        raise ValueError(f"task_id must be in [0, {expected_tasks - 1}], got {args.task_id}.")

    sigma_idx = args.task_id // args.n_networks
    network_idx = args.task_id % args.n_networks
    sigma2 = SIGMA2_VALUES[sigma_idx]
    params = make_params(args, sigma2)

    output_root = Path(args.output_root)
    root = output_root if output_root.is_absolute() else Path(__file__).resolve().parent / output_root
    subdir = f"sigma2_{sigma2:0.3f}"
    experiment_root = root / subdir
    chunk_root = experiment_root / "chunks" / f"network_{network_idx:03d}"
    root.mkdir(parents=True, exist_ok=True)

    (root / "array_jobs.json").write_text(
        json.dumps(
            [
                {
                    "experiment_idx": idx,
                    "name": f"sigma2_{value:0.3f}",
                    "output_subdir": f"sigma2_{value:0.3f}",
                    "sigma_2": value,
                }
                for idx, value in enumerate(SIGMA2_VALUES)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "chunking.json").write_text(
        json.dumps(
            {
                "n_experiments": len(SIGMA2_VALUES),
                "n_networks": args.n_networks,
                "total_array_tasks": expected_tasks,
                "paths_per_alpha_label_noise_sigma2": args.n_networks * args.n_trials * args.n_trials,
                "sigma2_values": SIGMA2_VALUES,
                "noise_intensity_dv_values": DV_VALUES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Task {args.task_id}/{expected_tasks - 1}: "
        f"sigma2={sigma2:0.3f} ({sigma_idx + 1}/{len(SIGMA2_VALUES)}), network={network_idx}"
    )
    network_row, all_rows = run_one_network(params, network_idx, chunk_root)
    (chunk_root / "task_meta.json").write_text(
        json.dumps(
            {
                "task_id": args.task_id,
                "n_tasks": args.n_tasks,
                "experiment_idx": sigma_idx,
                "experiment_name": f"sigma2_{sigma2:0.3f}",
                "sigma_2": sigma2,
                "network_idx": network_idx,
                "network_summary": network_row,
                "n_rows": len(all_rows),
                "parameters": asdict(params),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Finished task {args.task_id}: sigma2={sigma2:0.3f}, network={network_idx}")


if __name__ == "__main__":
    main()
