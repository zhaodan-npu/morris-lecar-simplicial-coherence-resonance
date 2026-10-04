"""Prepare all six Figure 3 panels from numerical simulation summaries."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

GRID = (.04, .06, .09, .12, .16, .20, .25, .30, .40, .56, .75,
        1., 1.25, 1.56, 1.89, 2.25, 3.06, 4., 6.25, 9.)
LABELS = ("control", "higher_order")
SPECS = (
    ("a", 1.5, "isi_cv", 2.1, 1.), ("b", 2., "isi_cv", 2.1, 1.),
    ("c", 1.5, "spike_count_total", 60000., 1000.),
    ("d", 2., "spike_count_total", 60000., 1000.),
    ("e", 1.5, "tcor", 1.4, 1.), ("f", 2., "tcor", 4.2, 1.),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def index_rows(rows, correlation=False):
    expected = {(alpha, label, dv) for alpha in (1.5, 2.)
                for label in LABELS for dv in GRID}
    indexed = {}
    for row in rows:
        label = row["label"]
        if correlation:
            label = {"Pairwise": "control", "Simplicial": "higher_order"}[label]
            if int(row["n_networks"]) != 30 or float(row["observation_ms"]) != 117000.:
                raise ValueError("Correlation table does not use the production ensemble/duration")
            if GRID[int(row["dv_index"])] != float(row["noise_intensity_dv"]):
                raise ValueError("Noise index disagrees with the production grid")
        elif int(row["total_realizations"]) != 4320:
            raise ValueError("Main table does not use the production ensemble")
        key = (float(row["alpha"]), label, float(row["noise_intensity_dv"]))
        if key in indexed:
            raise ValueError(f"Duplicate condition: {key}")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Expected exactly 80 production conditions")
    return indexed


def build_rows(main_rows, correlation_rows):
    main = index_rows(main_rows)
    correlation = index_rows(correlation_rows, correlation=True)
    output, panels = [], []
    for letter, alpha, metric, limit, divisor in SPECS:
        optima = {}
        for label in LABELS:
            eligible = [main[alpha, label, dv] for dv in GRID
                        if float(main[alpha, label, dv]["spike_count_total_mean"]) >= 1000
                        and np.isfinite(float(main[alpha, label, dv]["isi_cv_mean"]))]
            if not eligible:
                raise ValueError(f"No eligible CV minimum for {alpha}, {label}")
            best = min(eligible, key=lambda row: float(row["isi_cv_mean"]))
            optima[label] = {"dv": float(best["noise_intensity_dv"]),
                             "cv": float(best["isi_cv_mean"])}
            for dv in GRID:
                row = main[alpha, label, dv]
                if metric == "tcor":
                    row = correlation[alpha, label, dv]
                    mean = float(row["floor_db_50000_ms"]) / 1000.
                    se = float(row["floor_db_jackknife_se_ms"]) / 1000.
                    source = "original_debiased_correlation_time_csv"
                    error = "delete_one_network_jackknife_standard_error"
                else:
                    mean = float(row[f"{metric}_mean"])
                    se = float(row[f"{metric}_std"]) / np.sqrt(float(row["total_realizations"]))
                    source = "original_aggregated_metrics_csv"
                    error = "original_std_over_sqrt_total_realizations"
                if not np.isfinite(mean) or not np.isfinite(se) or se < 0:
                    raise ValueError(f"Invalid plotted statistic: {letter}, {label}, {dv}")
                output.append(dict(panel=letter, alpha=alpha, metric=metric,
                                   label=label, D_v=dv, mean=mean, lower=mean-se,
                                   upper=mean+se, display_divisor=divisor,
                                   source_kind=source, error_kind=error))
        panels.append(dict(panel=letter, alpha=alpha, metric=metric, y_max=limit,
                           display_divisor=divisor, optima=optima))
    return output, panels


def prepare(main_csv, correlation_csv, output_dir, root):
    inputs = [dict(path=path.resolve().relative_to(root.resolve()).as_posix(),
                   sha256=digest(path)) for path in (main_csv, correlation_csv)]
    rows, panels = build_rows(read_rows(main_csv), read_rows(correlation_csv))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "plot_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(inputs=inputs, panels=panels, n_means=len(rows),
                    tcor_csv_not_available=False, no_simulations_run=True,
                    numerical_data_only=True,
                    correlation_mean_column="floor_db_50000_ms",
                    correlation_error_column="floor_db_jackknife_se_ms",
                    correlation_display_unit="seconds",
                    correlation_source_unit="milliseconds")
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared {len(rows)} means and error bars from numerical CSV tables.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-csv", type=Path, required=True)
    parser.add_argument("--correlation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Base for relative provenance paths")
    args = parser.parse_args()
    prepare(args.main_csv, args.correlation_csv, args.output_dir, args.root)


if __name__ == "__main__":
    main()
