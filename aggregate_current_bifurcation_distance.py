"""Aggregate the input-current sweep into CV*(I) and D_v*(I).

Reads the chunked array output of
morris_lecar_current_bifurcation_distance_array.py for the newly computed
currents, and the corresponding directories of the main run for the currents
that were already available (87, 88, 89).  For every
(current, alpha, coupling label) it reports

  * CV*    : the smallest network-averaged ISI coefficient of variation over
             the D_v grid, restricted to conditions with at least --min-spikes
             detected spikes,
  * D_v*   : the noise intensity at which that minimum is attained,

together with percentile bootstrap confidence intervals obtained by resampling
the network realizations.

Outputs (in --output-dir):
  cr_vs_current.csv      one row per (current, alpha, label)
  cv_curves.csv          network-averaged CV on the full D_v grid
  cr_vs_current.json     the same content plus run metadata
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_CURRENTS = (80.0, 82.0, 84.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0)
REUSED_CURRENTS = (87.0, 88.0, 89.0)


def load_current(chunks_dir: Path):
    """Return arrays (alpha, label, network, dv, cv, spikes) for one current."""
    alphas, labels, networks, dvs, cvs, spikes = [], [], [], [], [], []
    chunk_dirs = sorted(chunks_dir.glob("network_*"))
    if not chunk_dirs:
        raise FileNotFoundError(f"no network_* chunks under {chunks_dir}")
    for chunk in chunk_dirs:
        path = chunk / "all_realizations.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                alphas.append(float(row["alpha"]))
                labels.append(row["label"])
                networks.append(int(row["network_idx"]))
                dvs.append(float(row["noise_intensity_dv"]))
                value = row["isi_cv"]
                cvs.append(float(value) if value not in ("", "nan", "NaN") else np.nan)
                spikes.append(float(row["spike_count_total"]))
    return (np.array(alphas), np.array(labels), np.array(networks),
            np.array(dvs), np.array(cvs), np.array(spikes), len(chunk_dirs))


def network_means(cv, network, dv_mask, network_ids):
    """Mean CV per network for one (alpha, label, D_v) cell; NaN where empty."""
    out = np.full(len(network_ids), np.nan)
    for index, net in enumerate(network_ids):
        selection = cv[dv_mask & (network == net)]
        selection = selection[np.isfinite(selection)]
        if selection.size:
            out[index] = selection.mean()
    return out


def optimum(per_network_cv, mean_spikes, dv_grid, min_spikes, weights=None):
    """CV* and D_v* from per-network means, optionally with bootstrap weights."""
    if weights is None:
        curve = np.nanmean(per_network_cv, axis=1)
    else:
        curve = np.array([
            np.nansum(row * weights) / np.nansum(weights * np.isfinite(row))
            if np.any(np.isfinite(row)) else np.nan
            for row in per_network_cv
        ])
    valid = np.isfinite(curve) & (mean_spikes >= min_spikes)
    if not valid.any():
        return np.nan, np.nan, curve
    index = int(np.nanargmin(np.where(valid, curve, np.inf)))
    return float(curve[index]), float(dv_grid[index]), curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True,
                        help="results root written by the array script (contains current_*/chunks)")
    parser.add_argument("--reuse-root", default="",
                        help="main-run root holding current_87.0 / 88.0 / 89.0")
    parser.add_argument("--output-dir", default="current_bifurcation_distance_analysis")
    parser.add_argument("--currents", default=",".join(f"{c:0.1f}" for c in DEFAULT_CURRENTS))
    parser.add_argument("--min-spikes", type=float, default=5.0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    reuse_root = Path(args.reuse_root) if args.reuse_root else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    currents = [float(value) for value in args.currents.split(",")]
    summary_rows, curve_rows = [], []

    for current in currents:
        subdir = f"current_{current:0.1f}"
        candidate = run_root / subdir / "chunks"
        if not candidate.exists() and reuse_root is not None:
            candidate = reuse_root / subdir / "chunks"
        alpha, label, network, dv, cv, spikes, n_chunks = load_current(candidate)
        source = "new" if current not in REUSED_CURRENTS else "main run"
        print(f"I = {current:5.1f}: {len(cv)} realizations from {n_chunks} networks ({source})")

        dv_grid = np.unique(dv)
        network_ids = np.unique(network)

        for alpha_value in np.unique(alpha):
            for label_value in ("control", "higher_order"):
                base_mask = (alpha == alpha_value) & (label == label_value)
                per_network = np.empty((len(dv_grid), len(network_ids)))
                mean_spikes = np.empty(len(dv_grid))
                for index, dv_value in enumerate(dv_grid):
                    cell = base_mask & (np.abs(dv - dv_value) < 1e-12)
                    per_network[index] = network_means(cv, network, cell, network_ids)
                    mean_spikes[index] = spikes[cell].mean() if cell.any() else 0.0

                cv_star, dv_star, curve = optimum(per_network, mean_spikes, dv_grid, args.min_spikes)

                cv_samples, dv_samples = [], []
                for _ in range(args.bootstrap):
                    counts = rng.multinomial(len(network_ids), np.ones(len(network_ids)) / len(network_ids))
                    cv_b, dv_b, _ = optimum(per_network, mean_spikes, dv_grid, args.min_spikes,
                                            weights=counts.astype(float))
                    if np.isfinite(cv_b):
                        cv_samples.append(cv_b)
                        dv_samples.append(dv_b)

                def interval(samples):
                    if len(samples) < 20:
                        return float("nan"), float("nan")
                    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))

                cv_lo, cv_hi = interval(cv_samples)
                dv_lo, dv_hi = interval(dv_samples)

                summary_rows.append({
                    "current": current,
                    "alpha": float(alpha_value),
                    "label": label_value,
                    "cv_star": cv_star,
                    "cv_star_ci_low": cv_lo,
                    "cv_star_ci_high": cv_hi,
                    "dv_star": dv_star,
                    "dv_star_ci_low": dv_lo,
                    "dv_star_ci_high": dv_hi,
                    "n_networks": int(len(network_ids)),
                    "source": source,
                })
                for index, dv_value in enumerate(dv_grid):
                    curve_rows.append({
                        "current": current,
                        "alpha": float(alpha_value),
                        "label": label_value,
                        "noise_intensity_dv": float(dv_value),
                        "cv_mean": float(curve[index]) if np.isfinite(curve[index]) else "",
                        "cv_sem": float(np.nanstd(per_network[index], ddof=1)
                                        / np.sqrt(np.sum(np.isfinite(per_network[index]))))
                        if np.sum(np.isfinite(per_network[index])) > 1 else "",
                        "spike_count_mean": float(mean_spikes[index]),
                    })

    def write_csv(path, rows):
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "cr_vs_current.csv", summary_rows)
    write_csv(output_dir / "cv_curves.csv", curve_rows)
    (output_dir / "cr_vs_current.json").write_text(
        json.dumps({"currents": currents, "min_spikes": args.min_spikes,
                    "bootstrap": args.bootstrap, "rows": summary_rows}, indent=2),
        encoding="utf-8")

    print(f"\n{'I':>6} {'alpha':>6} {'label':>12} {'CV*':>8} {'95% CI':>18} {'Dv*':>7}")
    for row in summary_rows:
        print(f"{row['current']:6.1f} {row['alpha']:6.1f} {row['label']:>12} {row['cv_star']:8.4f} "
              f"[{row['cv_star_ci_low']:6.4f}, {row['cv_star_ci_high']:6.4f}] {row['dv_star']:7.2f}")
    print(f"\nWritten to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
