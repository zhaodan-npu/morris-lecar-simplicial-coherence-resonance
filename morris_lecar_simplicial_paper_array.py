from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.dont_write_bytecode = True

import morris_lecar_paired_alpha_compare_simplicial as base
import morris_lecar_simplicial_paper_suite as suite


def build_jobs(config: suite.PaperSuiteConfig):
    jobs = [
        {
            "name": "candidate39_main",
            "kind": "candidate",
            "params": suite.candidate39_params(config),
            "output_subdir": "candidate39_main",
        },
        {
            "name": "candidate42_main",
            "kind": "candidate",
            "params": suite.candidate42_params(config),
            "output_subdir": "candidate42_main",
        },
    ]

    for sigma2 in suite.REFINED_SIGMA2_SWEEP:
        jobs.append(
            {
                "name": f"sigma2_{sigma2:0.2f}",
                "kind": "candidate",
                "params": base.PairedSimplicialParams(
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
                    noise_scales=suite.REFINED_DV_VALUES,
                    base_seed=config.base_seed + 50000 + int(round(10000 * sigma2)),
                ),
                "output_subdir": f"sigma2_{sigma2:0.2f}",
            }
        )

    jobs.extend(
        [
        {
            "name": "current_87.0",
            "kind": "candidate",
            "params": base.PairedSimplicialParams(
                n_nodes=config.n_nodes,
                n_networks=config.n_networks,
                n_trials_per_network=config.n_trials_per_network,
                p_edge=config.p_edge,
                p_fill_simplex=config.p_fill_simplex,
                sigma_1=0.002,
                sigma_2=0.12,
                current=87.0,
                dt=config.dt,
                t_total=config.t_total,
                t_warmup=config.t_warmup,
                alphas=config.alphas,
                beta=config.beta,
                noise_scales=suite.REFINED_DV_VALUES,
                base_seed=config.base_seed + 60087,
            ),
            "output_subdir": "current_87.0",
        },
        {
            "name": "current_88.0",
            "kind": "candidate",
            "params": base.PairedSimplicialParams(
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
                noise_scales=suite.REFINED_DV_VALUES,
                base_seed=config.base_seed + 60088,
            ),
            "output_subdir": "current_88.0",
        },
        {
            "name": "current_89.0",
            "kind": "candidate",
            "params": base.PairedSimplicialParams(
                n_nodes=config.n_nodes,
                n_networks=config.n_networks,
                n_trials_per_network=config.n_trials_per_network,
                p_edge=config.p_edge,
                p_fill_simplex=config.p_fill_simplex,
                sigma_1=0.002,
                sigma_2=0.12,
                current=89.0,
                dt=config.dt,
                t_total=config.t_total,
                t_warmup=config.t_warmup,
                alphas=config.alphas,
                beta=config.beta,
                noise_scales=suite.REFINED_DV_VALUES,
                base_seed=config.base_seed + 60089,
            ),
            "output_subdir": "current_89.0",
        },
        {
            "name": "candidate39_longer_time",
            "kind": "candidate",
            "params": base.PairedSimplicialParams(
                n_nodes=config.n_nodes,
                n_networks=config.n_networks,
                n_trials_per_network=config.n_trials_per_network,
                p_edge=config.p_edge,
                p_fill_simplex=config.p_fill_simplex,
                sigma_1=0.002,
                sigma_2=0.12,
                current=88.0,
                dt=config.dt,
                t_total=max(config.t_total, 20000.0),
                t_warmup=max(config.t_warmup, 5000.0),
                alphas=config.alphas,
                beta=config.beta,
                noise_scales=suite.REFINED_DV_VALUES,
                base_seed=config.base_seed + 70039,
            ),
            "output_subdir": "candidate39_longer_time",
        },
        ]
    )
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Array-task launcher for the true simplicial Morris-Lecar paper suite.")
    parser.add_argument("task_id", type=int, help="Array task index.")
    parser.add_argument("n_tasks", type=int, help="Total number of tasks expected by the array.")
    parser.add_argument("--output-root", default="paper_suite_true_simplicial_array", help="Output root directory name.")
    parser.add_argument("--n-nodes", type=int, default=100)
    parser.add_argument("--n-networks", type=int, default=30)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--t-total", type=float, default=15000.0)
    parser.add_argument("--t-warmup", type=float, default=3000.0)
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

    jobs = build_jobs(config)
    if args.n_tasks != len(jobs):
        raise ValueError(f"Expected n_tasks={len(jobs)}, got {args.n_tasks}.")
    if args.task_id < 0 or args.task_id >= len(jobs):
        raise ValueError(f"task_id must be in [0, {len(jobs) - 1}], got {args.task_id}.")

    job = jobs[args.task_id]
    output_root = Path(args.output_root)
    root = output_root if output_root.is_absolute() else Path(__file__).resolve().parent / output_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "array_jobs.json").write_text(
        json.dumps(
            [
                {
                    "task_id": idx,
                    "name": item["name"],
                    "output_subdir": item["output_subdir"],
                }
                for idx, item in enumerate(jobs)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Running task {args.task_id}/{args.n_tasks - 1}: {job['name']}")
    print(f"Output root: {root}")

    output_dir = root / job["output_subdir"]
    suite.run_paired_experiment(job["params"], output_dir, save_all_rows=False)

    (output_dir / "task_meta.json").write_text(
        json.dumps(
            {
                "task_id": args.task_id,
                "n_tasks": args.n_tasks,
                "job_name": job["name"],
                "output_subdir": job["output_subdir"],
                "parameters": asdict(job["params"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Finished task {args.task_id}: {job['name']}")


if __name__ == "__main__":
    main()

