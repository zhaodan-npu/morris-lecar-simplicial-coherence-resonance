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


def make_params(args: argparse.Namespace) -> base.PairedSimplicialParams:
    config = suite.PaperSuiteConfig(
        n_nodes=args.n_nodes,
        n_networks=args.n_networks,
        n_trials_per_network=args.n_trials,
        t_total=args.t_total,
        t_warmup=args.t_warmup,
        base_seed=args.seed,
    )
    return base.PairedSimplicialParams(
        n_nodes=config.n_nodes,
        n_networks=config.n_networks,
        n_trials_per_network=config.n_trials_per_network,
        p_edge=config.p_edge,
        p_fill_simplex=config.p_fill_simplex,
        sigma_1=args.sigma1,
        sigma_2=args.sigma2,
        current=args.current,
        dt=config.dt,
        t_total=config.t_total,
        t_warmup=config.t_warmup,
        alphas=config.alphas,
        beta=config.beta,
        noise_scales=suite.REFINED_DV_VALUES,
        # Match candidate39_main so this control is paired with Fig. 1.
        base_seed=config.base_seed + 39000,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Chunked array task for sigma_1=0.12 controls with configurable sigma_2."
    )
    parser.add_argument("task_id", type=int, help="Slurm array task id.")
    parser.add_argument("n_tasks", type=int, help="Total number of Slurm array tasks.")
    parser.add_argument("--output-root", default="paper_suite_pairwise_sigma1_012_N100_Dv_refined")
    parser.add_argument("--output-subdir", default="sigma1_0.120_sigma2_0.000")
    parser.add_argument("--n-nodes", type=int, default=100)
    parser.add_argument("--n-networks", type=int, default=30)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--t-total", type=float, default=15000.0)
    parser.add_argument("--t-warmup", type=float, default=3000.0)
    parser.add_argument("--current", type=float, default=88.0)
    parser.add_argument("--sigma1", type=float, default=0.12)
    parser.add_argument("--sigma2", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    if args.n_tasks != args.n_networks:
        raise ValueError(f"Expected n_tasks={args.n_networks}, got {args.n_tasks}.")
    if args.task_id < 0 or args.task_id >= args.n_networks:
        raise ValueError(f"task_id must be in [0, {args.n_networks - 1}], got {args.task_id}.")

    params = make_params(args)
    output_root = Path(args.output_root)
    root = output_root if output_root.is_absolute() else Path(__file__).resolve().parent / output_root
    experiment_root = root / args.output_subdir
    chunk_root = experiment_root / "chunks" / f"network_{args.task_id:03d}"
    root.mkdir(parents=True, exist_ok=True)

    array_jobs = [
        {
            "experiment_idx": 0,
            "name": args.output_subdir,
            "output_subdir": args.output_subdir,
        }
    ]
    (root / "array_jobs.json").write_text(json.dumps(array_jobs, indent=2), encoding="utf-8")
    (root / "chunking.json").write_text(
        json.dumps(
            {
                "n_experiments": 1,
                "n_networks": args.n_networks,
                "total_array_tasks": args.n_networks,
                "paths_per_alpha_label_Dv": args.n_networks * args.n_trials * args.n_trials,
                "paired_with": "paper_suite_true_simplicial_array_prl_N100_Dv_refined/candidate39_main",
                "note": (
                    "When sigma_2=0, the control and higher_order labels are mathematically identical. "
                    "When sigma_2>0, use control as pairwise-only and higher_order as simplicial."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Task {args.task_id}/{args.n_tasks - 1}: "
        f"sigma_1={params.sigma_1}, sigma_2={params.sigma_2}, network={args.task_id}"
    )
    network_row, all_rows = run_one_network(params, args.task_id, chunk_root)
    (chunk_root / "task_meta.json").write_text(
        json.dumps(
            {
                "task_id": args.task_id,
                "n_tasks": args.n_tasks,
                "experiment_idx": 0,
                "experiment_name": args.output_subdir,
                "network_idx": args.task_id,
                "network_summary": network_row,
                "n_rows": len(all_rows),
                "parameters": asdict(params),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Finished task {args.task_id}: {args.output_subdir}")


if __name__ == "__main__":
    main()
