from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.dont_write_bytecode = True

import morris_lecar_paired_alpha_compare_simplicial as base
import morris_lecar_simplicial_paper_suite as suite
from morris_lecar_simplicial_paper_array_chunked import run_one_network


SIGMA1_SWEEP = (
    0.0,
    0.001,
    0.002,
    0.004,
    0.006,
    0.008,
    0.010,
    0.015,
    0.020,
    0.030,
    0.040,
    0.060,
    0.080,
    0.100,
    0.120,
    0.140,
    0.160,
)


def sigma1_dir_name(value: float) -> str:
    return f"sigma1_{value:0.3f}"


def build_jobs(config: suite.PaperSuiteConfig, sigma2: float):
    jobs = []
    for sigma1 in SIGMA1_SWEEP:
        jobs.append(
            {
                "name": sigma1_dir_name(sigma1),
                "sigma_1": sigma1,
                "params": base.PairedSimplicialParams(
                    n_nodes=config.n_nodes,
                    n_networks=config.n_networks,
                    n_trials_per_network=config.n_trials_per_network,
                    p_edge=config.p_edge,
                    p_fill_simplex=config.p_fill_simplex,
                    sigma_1=sigma1,
                    sigma_2=sigma2,
                    current=88.0,
                    dt=config.dt,
                    t_total=config.t_total,
                    t_warmup=config.t_warmup,
                    alphas=config.alphas,
                    beta=config.beta,
                    noise_scales=suite.REFINED_DV_VALUES,
                    # Keep the same graph/noise seeds across sigma_1 values.
                    base_seed=config.base_seed + 81000,
                ),
                "output_subdir": sigma1_dir_name(sigma1),
            }
        )
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Chunked array task for sigma_1 sweep at fixed sigma_2."
    )
    parser.add_argument("task_id", type=int, help="Slurm array task id.")
    parser.add_argument("n_tasks", type=int, help="Total number of Slurm array tasks.")
    parser.add_argument("--output-root", default="paper_suite_sigma1_sweep_pairwise_N100_Dv_refined")
    parser.add_argument("--n-nodes", type=int, default=100)
    parser.add_argument("--n-networks", type=int, default=30)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--t-total", type=float, default=15000.0)
    parser.add_argument("--t-warmup", type=float, default=3000.0)
    parser.add_argument("--sigma2", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    config = suite.PaperSuiteConfig(
        n_nodes=args.n_nodes,
        n_networks=args.n_networks,
        n_trials_per_network=args.n_trials,
        t_total=args.t_total,
        t_warmup=args.t_warmup,
        base_seed=args.seed,
    )
    jobs = build_jobs(config, sigma2=args.sigma2)
    expected_tasks = len(jobs) * args.n_networks
    if args.n_tasks != expected_tasks:
        raise ValueError(f"Expected n_tasks={expected_tasks}, got {args.n_tasks}.")
    if args.task_id < 0 or args.task_id >= expected_tasks:
        raise ValueError(f"task_id must be in [0, {expected_tasks - 1}], got {args.task_id}.")

    experiment_idx = args.task_id // args.n_networks
    network_idx = args.task_id % args.n_networks
    job = jobs[experiment_idx]

    output_root = Path(args.output_root)
    root = output_root if output_root.is_absolute() else Path(__file__).resolve().parent / output_root
    experiment_root = root / job["output_subdir"]
    chunk_root = experiment_root / "chunks" / f"network_{network_idx:03d}"
    root.mkdir(parents=True, exist_ok=True)

    (root / "array_jobs.json").write_text(
        json.dumps(
            [
                {
                    "experiment_idx": idx,
                    "name": item["name"],
                    "sigma_1": item["sigma_1"],
                    "sigma_2": args.sigma2,
                    "output_subdir": item["output_subdir"],
                }
                for idx, item in enumerate(jobs)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "chunking.json").write_text(
        json.dumps(
            {
                "n_experiments": len(jobs),
                "n_networks": args.n_networks,
                "total_array_tasks": expected_tasks,
                "paths_per_alpha_label_Dv": args.n_networks * args.n_trials * args.n_trials,
                "sigma_1_values": list(SIGMA1_SWEEP),
                "sigma_2_fixed": args.sigma2,
                "purpose": "Pure pairwise strength sweep to compare with the sigma_2 sweep.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Task {args.task_id}/{expected_tasks - 1}: "
        f"sigma_1={job['sigma_1']}, sigma_2={args.sigma2}, network={network_idx}"
    )
    network_row, all_rows = run_one_network(job["params"], network_idx, chunk_root)
    (chunk_root / "task_meta.json").write_text(
        json.dumps(
            {
                "task_id": args.task_id,
                "n_tasks": args.n_tasks,
                "experiment_idx": experiment_idx,
                "experiment_name": job["name"],
                "sigma_1": job["sigma_1"],
                "sigma_2": args.sigma2,
                "network_idx": network_idx,
                "network_summary": network_row,
                "n_rows": len(all_rows),
                "parameters": asdict(job["params"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Finished task {args.task_id}: {job['name']}, network={network_idx}")


if __name__ == "__main__":
    main()
