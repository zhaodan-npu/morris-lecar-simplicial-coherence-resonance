from __future__ import annotations

import argparse
import json
from pathlib import Path

import morris_lecar_paired_alpha_compare_simplicial as base
import morris_lecar_simplicial_paper_suite as suite


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def merge_experiment(experiment_root: Path, min_spikes: float):
    chunk_dirs = sorted((experiment_root / "chunks").glob("network_*"))
    if not chunk_dirs:
        raise FileNotFoundError(f"No chunk directories found under {experiment_root / 'chunks'}")

    all_rows = []
    network_rows = []
    params_dict = None
    for chunk_dir in chunk_dirs:
        all_rows.extend(load_json(chunk_dir / "all_realizations.json"))
        network_rows.extend(load_json(chunk_dir / "network_summary.json"))
        if params_dict is None:
            params_dict = load_json(chunk_dir / "parameters.json")

    params = base.PairedSimplicialParams(**params_dict)
    aggregated_rows = suite.aggregate_rows(all_rows, params)
    summary = {
        "parameters": params_dict,
        "n_chunk_networks": len(chunk_dirs),
        "best_by_alpha": suite.best_rows_by_alpha(aggregated_rows, min_spikes=min_spikes),
        "mean_edge_count": float(base.np.mean([row["edge_count"] for row in network_rows])),
        "mean_clique_count": float(base.np.mean([row["clique_count"] for row in network_rows])),
        "mean_simplex_count": float(base.np.mean([row["simplex_count"] for row in network_rows])),
    }

    (experiment_root / "network_summary.json").write_text(json.dumps(network_rows, indent=2), encoding="utf-8")
    (experiment_root / "aggregated_metrics.json").write_text(json.dumps(aggregated_rows, indent=2), encoding="utf-8")
    (experiment_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    base.write_csv(experiment_root / "network_summary.csv", network_rows)
    base.write_csv(experiment_root / "aggregated_metrics.csv", aggregated_rows)

    if should_make_plots():
        suite.render_metrics_plot(experiment_root / "metrics.png", aggregated_rows, experiment_root.name)
    return summary


def should_make_plots() -> bool:
    import os

    return os.environ.get("ML_MAKE_PLOTS", "").lower() in {"1", "true", "yes", "on"}


def main():
    parser = argparse.ArgumentParser(description="Merge chunked N=100 true simplicial Morris-Lecar array outputs.")
    parser.add_argument("--output-root", default="paper_suite_true_simplicial_array_prl_N100")
    parser.add_argument("--min-spikes", type=float, default=5.0)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    root = output_root if output_root.is_absolute() else Path(__file__).resolve().parent / output_root
    array_jobs = load_json(root / "array_jobs.json")

    merged = {}
    for job in array_jobs:
        experiment_root = root / job["output_subdir"]
        print(f"Merging {job['name']} from {experiment_root}")
        merged[job["name"]] = merge_experiment(experiment_root, min_spikes=args.min_spikes)

    (root / "merged_summary.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Merged results saved to: {root}")


if __name__ == "__main__":
    main()
