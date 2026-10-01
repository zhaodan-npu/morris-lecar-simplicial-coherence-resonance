from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

import morris_lecar_paired_alpha_compare_simplicial as base
import morris_lecar_simplicial_paper_suite as suite


LABEL = "matched_pairwise"


def matched_pairwise_network(
    params: base.PairedSimplicialParams,
    network: base.SimplicialNetwork,
):
    degree = network.a1.sum(axis=1)
    k1_actual = float(np.mean(degree))
    k2_actual = float(np.mean(-np.diag(network.l2)))
    g1 = float(params.sigma_1 * k1_actual)
    g2 = float(params.sigma_2 * k2_actual)
    g_eff = float(g1 + g2)
    sigma1_matched = float(g_eff / k1_actual) if k1_actual > 0 else float(params.sigma_1)

    pairwise_laplacian = network.a1 - np.diag(degree)
    matched_network = base.SimplicialNetwork(
        a1=network.a1,
        pairwise_operator=sigma1_matched * pairwise_laplacian,
        cliques3=network.cliques3,
        simplices2=network.simplices2,
        a2=network.a2,
        l2=network.l2,
        edge_count=network.edge_count,
        clique_count=network.clique_count,
        simplex_count=network.simplex_count,
    )
    matched_params = replace(params, sigma_1=sigma1_matched, sigma_2=0.0)
    info = {
        "k1_actual": k1_actual,
        "k2_actual": k2_actual,
        "g1_original": g1,
        "g2_original": g2,
        "g_eff_target": g_eff,
        "sigma1_matched": sigma1_matched,
    }
    return matched_params, matched_network, info


def run_one_network(params: base.PairedSimplicialParams, network_idx: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    network_rng = base.np.random.default_rng(params.base_seed + base.stable_seed("paper_network", network_idx))
    network = base.build_simplicial_network(params, network_rng)
    matched_params, matched_network, match = matched_pairwise_network(params, network)
    network_row = {
        "network_idx": network_idx,
        "edge_count": network.edge_count,
        "clique_count": network.clique_count,
        "simplex_count": network.simplex_count,
        **match,
    }

    all_rows = []
    for alpha in params.alphas:
        for noise_idx, noise_intensity_dv in enumerate(params.noise_scales):
            scale = base.noise_scale_from_voltage_noise_intensity(noise_intensity_dv, alpha, params.capacitance)
            for trial_idx in range(params.n_trials_per_network):
                init_seed = params.base_seed + base.stable_seed("init", network_idx, trial_idx)
                noise_seed = params.base_seed + base.stable_seed("noise", alpha, noise_idx, network_idx, trial_idx)
                metrics = base.simulate_batch(
                    matched_params,
                    matched_network,
                    alpha=alpha,
                    noise_scale=scale,
                    higher_order_enabled=False,
                    init_seed=init_seed,
                    noise_seed=noise_seed,
                )
                for path_idx in range(params.n_trials_per_network):
                    all_rows.append(
                        {
                            "alpha": alpha,
                            "label": LABEL,
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
                            **match,
                        }
                    )

    (output_dir / "parameters.json").write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")
    (output_dir / "matched_parameters.json").write_text(json.dumps(asdict(matched_params), indent=2), encoding="utf-8")
    (output_dir / "network_summary.json").write_text(json.dumps([network_row], indent=2), encoding="utf-8")
    (output_dir / "all_realizations.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    base.write_csv(output_dir / "network_summary.csv", [network_row])
    base.write_csv(output_dir / "all_realizations.csv", all_rows)
    return network_row, all_rows


def main():
    parser = argparse.ArgumentParser(description="Matched effective-coupling pairwise control array task.")
    parser.add_argument("task_id", type=int, help="Slurm array task id; one task per network.")
    parser.add_argument("n_tasks", type=int, help="Total number of network tasks.")
    parser.add_argument("--output-root", default="paper_suite_matched_coupling_control_N100_Dv_refined")
    parser.add_argument("--n-nodes", type=int, default=100)
    parser.add_argument("--n-networks", type=int, default=30)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--t-total", type=float, default=15000.0)
    parser.add_argument("--t-warmup", type=float, default=3000.0)
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    if args.n_tasks != args.n_networks:
        raise ValueError(f"Expected n_tasks={args.n_networks}, got {args.n_tasks}.")
    if args.task_id < 0 or args.task_id >= args.n_networks:
        raise ValueError(f"task_id must be in [0, {args.n_networks - 1}], got {args.task_id}.")

    config = suite.PaperSuiteConfig(
        n_nodes=args.n_nodes,
        n_networks=args.n_networks,
        n_trials_per_network=args.n_trials,
        t_total=args.t_total,
        t_warmup=args.t_warmup,
        base_seed=args.seed,
    )
    params = suite.candidate39_params(config)
    root = Path(args.output_root)
    root = root if root.is_absolute() else Path(__file__).resolve().parent / root
    chunk_root = root / "chunks" / f"network_{args.task_id:03d}"
    root.mkdir(parents=True, exist_ok=True)

    (root / "chunking.json").write_text(
        json.dumps(
            {
                "experiment": "matched_effective_coupling_pairwise_control",
                "n_networks": args.n_networks,
                "total_array_tasks": args.n_networks,
                "paths_per_alpha_label_dv": args.n_networks * args.n_trials * args.n_trials,
                "base_condition_reused_for_control_and_ho": "candidate39_main",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Matched coupling task {args.task_id}/{args.n_networks - 1}")
    network_row, all_rows = run_one_network(params, args.task_id, chunk_root)
    (chunk_root / "task_meta.json").write_text(
        json.dumps(
            {
                "task_id": args.task_id,
                "n_tasks": args.n_tasks,
                "network_idx": args.task_id,
                "network_summary": network_row,
                "n_rows": len(all_rows),
                "parameters": asdict(params),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "Finished matched task "
        f"{args.task_id}: sigma1_matched={network_row['sigma1_matched']:.6g}, "
        f"g_eff={network_row['g_eff_target']:.6g}"
    )


if __name__ == "__main__":
    main()
