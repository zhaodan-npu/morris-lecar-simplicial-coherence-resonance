"""Array task for the input-current sweep requested by Reviewer 1 (comment 1).

The manuscript fixes the working point at I = 88, which lies in the monostable
excitable regime, 5.86 below the subcritical Hopf bifurcation I_H = 93.8576 and
0.31 below the fold of limit cycles I_fold = 88.3126 (see
morris_lecar_hopf_distance_analysis.py).  This script sweeps the input current
across the excitable range and into the bistable window so that the optimal CV
and the optimal noise intensity can be reported as functions of the distance
from the bifurcation.

Currents 87, 88 and 89 were already produced by the main array run
(paper_suite_true_simplicial_array_prl_N100_Dv_refined/current_8?.0) with the
same protocol and the same seed convention, base_seed = seed + 60000 + int(I),
so only the remaining currents are recomputed here.

Layout of the outputs is identical to
morris_lecar_simplicial_paper_array_chunked.py, so the existing merge script
merge_morris_lecar_simplicial_chunks.py works unchanged:

    <root>/current_<I>/chunks/network_<idx>/all_realizations.{json,csv}

One array task is one (current, network) pair.  Within a task the 2 x 20
(alpha, D_v) conditions may be distributed over several worker processes; the
random seeds depend only on (alpha, noise index, network index, trial index),
so the results do not depend on --workers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

sys.dont_write_bytecode = True

import morris_lecar_paired_alpha_compare_simplicial as base
import morris_lecar_simplicial_paper_suite as suite

DEFAULT_CURRENTS = (80.0, 82.0, 84.0, 86.0, 90.0, 91.0, 92.0, 93.0)
REUSED_CURRENTS = (87.0, 88.0, 89.0)  # already available from the main run


def build_jobs(config: suite.PaperSuiteConfig, currents):
    jobs = []
    for current in currents:
        jobs.append(
            {
                "name": f"current_{current:0.1f}",
                "output_subdir": f"current_{current:0.1f}",
                "params": base.PairedSimplicialParams(
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
                    noise_scales=suite.REFINED_DV_VALUES,
                    # same convention as the main run: 60087 / 60088 / 60089
                    base_seed=config.base_seed + 60000 + int(round(current)),
                ),
            }
        )
    return jobs


def run_condition(payload):
    """One (alpha, noise index) condition: 12 trials x 2 coupling labels."""
    params, network, network_idx, alpha, noise_idx, noise_intensity_dv = payload
    scale = base.noise_scale_from_voltage_noise_intensity(noise_intensity_dv, alpha, params.capacitance)
    rows = []
    for trial_idx in range(params.n_trials_per_network):
        init_seed = params.base_seed + base.stable_seed("init", network_idx, trial_idx)
        noise_seed = params.base_seed + base.stable_seed(
            "noise", alpha, noise_idx, network_idx, trial_idx
        )
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
                rows.append(
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
    return rows


def run_one_network(params, network_idx: int, output_dir: Path, workers: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    network_rng = base.np.random.default_rng(params.base_seed + base.stable_seed("paper_network", network_idx))
    network = base.build_simplicial_network(params, network_rng)
    network_row = {
        "network_idx": network_idx,
        "edge_count": network.edge_count,
        "clique_count": network.clique_count,
        "simplex_count": network.simplex_count,
    }

    payloads = [
        (params, network, network_idx, alpha, noise_idx, noise_intensity_dv)
        for alpha in params.alphas
        for noise_idx, noise_intensity_dv in enumerate(params.noise_scales)
    ]

    if workers <= 1:
        results = [run_condition(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_condition, payloads))

    all_rows = [row for chunk in results for row in chunk]

    (output_dir / "parameters.json").write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")
    (output_dir / "network_summary.json").write_text(json.dumps([network_row], indent=2), encoding="utf-8")
    (output_dir / "all_realizations.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    base.write_csv(output_dir / "network_summary.csv", [network_row])
    base.write_csv(output_dir / "all_realizations.csv", all_rows)
    return network_row, all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", type=int)
    parser.add_argument("n_tasks", type=int)
    parser.add_argument("--output-root", default="current_bifurcation_distance")
    parser.add_argument("--currents", default=",".join(f"{c:0.1f}" for c in DEFAULT_CURRENTS))
    parser.add_argument("--n-nodes", type=int, default=100)
    parser.add_argument("--n-networks", type=int, default=30)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--t-total", type=float, default=15000.0)
    parser.add_argument("--t-warmup", type=float, default=3000.0)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    currents = tuple(float(value) for value in args.currents.split(","))

    config = suite.PaperSuiteConfig(
        n_nodes=args.n_nodes,
        n_networks=args.n_networks,
        n_trials_per_network=args.n_trials,
        t_total=args.t_total,
        t_warmup=args.t_warmup,
        base_seed=args.seed,
    )
    jobs = build_jobs(config, currents)

    expected_tasks = len(jobs) * args.n_networks
    if args.n_tasks != expected_tasks:
        raise ValueError(f"Expected n_tasks={expected_tasks}, got {args.n_tasks}.")
    if not 0 <= args.task_id < expected_tasks:
        raise ValueError(f"task_id must be in [0, {expected_tasks - 1}], got {args.task_id}.")

    experiment_idx = args.task_id // args.n_networks
    network_idx = args.task_id % args.n_networks
    job = jobs[experiment_idx]

    output_root = Path(args.output_root)
    root = output_root if output_root.is_absolute() else Path(__file__).resolve().parent / output_root
    root.mkdir(parents=True, exist_ok=True)
    chunk_root = root / job["output_subdir"] / "chunks" / f"network_{network_idx:03d}"

    (root / "array_jobs.json").write_text(
        json.dumps(
            [{"experiment_idx": idx, "name": item["name"], "output_subdir": item["output_subdir"]}
             for idx, item in enumerate(jobs)],
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
                "currents": list(currents),
                "reused_currents": list(REUSED_CURRENTS),
                "paths_per_alpha_label_noise": args.n_networks * args.n_trials * args.n_trials,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    marker = chunk_root / "all_realizations.csv"
    if args.skip_existing and marker.exists() and marker.stat().st_size > 0:
        print(f"skip existing task {args.task_id}: {marker}")
        return

    print(f"Task {args.task_id}/{expected_tasks - 1}: {job['name']}, network={network_idx}, "
          f"workers={args.workers}")
    network_row, all_rows = run_one_network(job["params"], network_idx, chunk_root, args.workers)
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
                "workers": args.workers,
                "parameters": asdict(job["params"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Finished task {args.task_id}: {job['name']}, network={network_idx}")


if __name__ == "__main__":
    main()
