from __future__ import annotations

import argparse

import merge_morris_lecar_simplicial_chunks as merge_base


def main():
    parser = argparse.ArgumentParser(description="Merge PRL supplemental chunked outputs.")
    parser.add_argument("--output-root", default="paper_suite_true_simplicial_prl_supplement")
    parser.add_argument("--min-spikes", type=float, default=5.0)
    args = parser.parse_args()

    root = merge_base.Path(args.output_root)
    root = root if root.is_absolute() else merge_base.Path(__file__).resolve().parent / root
    array_jobs = merge_base.load_json(root / "array_jobs.json")

    merged = {}
    for job in array_jobs:
        experiment_root = root / job["output_subdir"]
        print(f"Merging {job['name']} from {experiment_root}")
        merged[job["name"]] = merge_base.merge_experiment(experiment_root, min_spikes=args.min_spikes)

    (root / "merged_summary.json").write_text(merge_base.json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Merged supplemental results saved to: {root}")


if __name__ == "__main__":
    main()
