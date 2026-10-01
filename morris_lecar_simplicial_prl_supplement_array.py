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


MAIN_DV = (0.04, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.56, 0.75, 1.00, 1.25, 1.56, 1.89, 2.25, 3.06, 4.00, 6.25, 9.00)
FINE_DV = (0.12, 0.16, 0.20, 0.25, 0.30, 0.36, 0.42, 0.50, 0.60, 0.75, 1.00)


def make_params(
    config: suite.PaperSuiteConfig,
    *,
    n_nodes: int | None = None,
    p_edge: float | None = None,
    p_fill_simplex: float | None = None,
    sigma_1: float = 0.002,
    sigma_2: float = 0.12,
    current: float = 88.0,
    alphas: tuple[float, ...] = (1.5, 2.0),
    noise_scales: tuple[float, ...] = MAIN_DV,
    seed_offset: int = 0,
):
    return base.PairedSimplicialParams(
        n_nodes=config.n_nodes if n_nodes is None else n_nodes,
        n_networks=config.n_networks,
        n_trials_per_network=config.n_trials_per_network,
        p_edge=config.p_edge if p_edge is None else p_edge,
        p_fill_simplex=config.p_fill_simplex if p_fill_simplex is None else p_fill_simplex,
        sigma_1=sigma_1,
        sigma_2=sigma_2,
        current=current,
        dt=config.dt,
        t_total=config.t_total,
        t_warmup=config.t_warmup,
        alphas=alphas,
        beta=config.beta,
        noise_scales=noise_scales,
        base_seed=config.base_seed + seed_offset,
    )


def build_jobs(config: suite.PaperSuiteConfig):
    jobs = []

    for idx, n_nodes in enumerate((60, 80, 100, 120), start=1):
        jobs.append(
            {
                "name": f"size_N{n_nodes}",
                "params": make_params(config, n_nodes=n_nodes, seed_offset=10000 + idx),
                "output_subdir": f"size_N{n_nodes}",
            }
        )

    for idx, p_fill in enumerate((0.0, 0.25, 0.50, 0.75), start=1):
        jobs.append(
            {
                "name": f"pfill_{p_fill:0.2f}",
                "params": make_params(config, p_fill_simplex=p_fill, seed_offset=20000 + idx),
                "output_subdir": f"pfill_{p_fill:0.2f}",
            }
        )

    for idx, p_edge in enumerate((0.10, 0.15, 0.20), start=1):
        jobs.append(
            {
                "name": f"pedge_{p_edge:0.2f}",
                "params": make_params(config, p_edge=p_edge, seed_offset=30000 + idx),
                "output_subdir": f"pedge_{p_edge:0.2f}",
            }
        )

    jobs.append(
        {
            "name": "alpha_sweep",
            "params": make_params(
                config,
                alphas=(1.2, 1.5, 1.8, 2.0),
                seed_offset=40000,
            ),
            "output_subdir": "alpha_sweep",
        }
    )

    jobs.append(
        {
            "name": "fine_noise_window",
            "params": make_params(
                config,
                alphas=(1.5, 2.0),
                noise_scales=FINE_DV,
                seed_offset=50000,
            ),
            "output_subdir": "fine_noise_window",
        }
    )

    return jobs


def main():
    parser = argparse.ArgumentParser(description="Supplemental PRL checks for true simplicial Morris-Lecar CR.")
    parser.add_argument("task_id", type=int, help="Slurm array task id.")
    parser.add_argument("n_tasks", type=int, help="Total number of Slurm array tasks.")
    parser.add_argument("--output-root", default="paper_suite_true_simplicial_prl_supplement")
    parser.add_argument("--n-nodes", type=int, default=100)
    parser.add_argument("--n-networks", type=int, default=20)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--t-total", type=float, default=10000.0)
    parser.add_argument("--t-warmup", type=float, default=2000.0)
    parser.add_argument("--seed", type=int, default=20260503)
    args = parser.parse_args()

    config = suite.PaperSuiteConfig(
        n_nodes=args.n_nodes,
        n_networks=args.n_networks,
        n_trials_per_network=args.n_trials,
        t_total=args.t_total,
        t_warmup=args.t_warmup,
        base_seed=args.seed,
    )

    jobs = build_jobs(config)
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
                "paths_per_alpha_label_noise": args.n_networks * args.n_trials * args.n_trials,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Task {args.task_id}/{expected_tasks - 1}: "
        f"experiment={job['name']} ({experiment_idx}), network={network_idx}"
    )
    network_row, all_rows = run_one_network(job["params"], network_idx, chunk_root)
    (chunk_root / "task_meta.json").write_text(
        json.dumps(
            {
                "task_id": args.task_id,
                "n_tasks": args.n_tasks,
                "experiment_idx": experiment_idx,
                "experiment_name": job["name"],
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
