"""Run a short smoke test or one explicitly selected production network."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import morris_lecar_simplicial_paper_array as main_jobs
import morris_lecar_simplicial_paper_array_chunked as chunks
import morris_lecar_simplicial_paper_suite as suite
import morris_lecar_simplicial_prl_supplement_array as supplement
from merge_morris_lecar_simplicial_chunks import merge_experiment

ROOT = Path(__file__).resolve().parent


def jobs_for(family):
    if family == "main":
        return main_jobs.build_jobs(suite.PaperSuiteConfig())
    config = suite.PaperSuiteConfig(n_networks=20, n_trials_per_network=10,
                                   t_total=10000., t_warmup=2000., base_seed=20260503)
    return supplement.build_jobs(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "parameters", "smoke", "run-network", "merge"))
    parser.add_argument("--family", choices=("main", "supplement"), default="main")
    parser.add_argument("--experiment", default="candidate39_main")
    parser.add_argument("--network", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--min-spikes", type=float, default=5., help="Original summary selection threshold.")
    args = parser.parse_args()
    jobs = jobs_for(args.family)
    if args.command == "list":
        for family in ("main", "supplement"):
            for job in jobs_for(family):
                print(f"{family}: {job['name']} ({job['params'].n_networks} networks)")
        return
    if args.command == "smoke":
        params = replace(suite.candidate39_params(suite.PaperSuiteConfig()),
                         n_nodes=8, n_networks=1, n_trials_per_network=2,
                         t_total=200., t_warmup=40., noise_scales=(0., .25))
        output = args.output / "smoke"
        print("SMOKE TEST ONLY: reduced N, ensemble, duration, and noise grid.")
        _, rows = chunks.run_one_network(params, 0, output)
        aggregated = suite.aggregate_rows(rows, params)
        suite.base.write_csv(output / "aggregated_metrics.csv", aggregated)
        expected = len(params.alphas) * len(params.noise_scales) * 2 * 2 * 2
        assert len(rows) == expected
        print(f"Smoke test complete: {len(rows)} trajectory records in {output}")
        return
    selected = next((job for job in jobs if job["name"] == args.experiment), None)
    if selected is None:
        parser.error("Unknown experiment for this family; run 'python reproduce.py list'.")
    params = selected["params"]
    if args.command == "parameters":
        print(json.dumps(asdict(params), indent=2))
        return
    if args.command == "merge":
        if args.min_spikes < 0:
            parser.error("--min-spikes must be nonnegative")
        experiment_root = args.output / args.family / selected["output_subdir"]
        folders = sorted((experiment_root / "chunks").glob("network_*"))
        expected_names = [f"network_{i:03d}" for i in range(params.n_networks)]
        if [p.name for p in folders] != expected_names:
            parser.error("Missing or extra network chunks; refusing an incomplete production merge.")
        expected_params = json.loads(json.dumps(asdict(params)))
        for index, folder in enumerate(folders):
            if json.loads((folder / "parameters.json").read_text()) != expected_params:
                parser.error(f"Parameter mismatch in {folder.name}")
            network = json.loads((folder / "network_summary.json").read_text())
            if len(network) != 1 or network[0]["network_idx"] != index:
                parser.error(f"Network identity mismatch in {folder.name}")
            rows = json.loads((folder / "all_realizations.json").read_text())
            expected_count = len(params.alphas) * len(params.noise_scales) * 2 * params.n_trials_per_network**2
            keys = {(r["alpha"], r["noise_intensity_dv"], r["label"], r["trial_idx"], r["path_idx"])
                    for r in rows if r["network_idx"] == index}
            expected_keys = {(a, d, label, trial, path) for a in params.alphas for d in params.noise_scales
                             for label in ("control", "higher_order")
                             for trial in range(params.n_trials_per_network)
                             for path in range(params.n_trials_per_network)}
            if len(rows) != expected_count or keys != expected_keys:
                parser.error(f"Incomplete or duplicated trajectory records in {folder.name}")
        merge_experiment(experiment_root, min_spikes=args.min_spikes)
        print(f"Complete production ensemble merged: {experiment_root}")
        return
    if not 0 <= args.network < params.n_networks:
        parser.error(f"--network must be in [0, {params.n_networks - 1}]")
    root = args.output / args.family
    target = root / selected["output_subdir"] / "chunks" / f"network_{args.network:03d}"
    if (target / "all_realizations.json").exists():
        parser.error("Output already exists; use a new --output directory.")
    chunks.run_one_network(params, args.network, target)
    print(f"Production network finished: {target}")


if __name__ == "__main__":
    main()
