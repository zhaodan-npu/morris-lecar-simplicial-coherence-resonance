#!/usr/bin/env python3
"""Aggregate the long-record Fig. 2 voltage autocorrelation calculations.

For each (alpha, D_v, coupling) condition, the normalized voltage ACF is
first averaged across the 30 independent network realizations.  The reported
correlation time is then the integral of the absolute pooled ACF.  Network
jackknife errors therefore preserve the network as the independent sampling
unit and do not integrate the positive finite-sample floor of every network
separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


FILE_RE = re.compile(
    r"^coherence_a(?P<alpha>1\.5|2(?:\.0)?)_n(?P<network>\d{3})_"
    r"d(?P<dv_index>\d{2})_(?P<label>pairwise|simplicial)\.npz$"
)
CUTOFFS_MS = np.asarray([2000, 5000, 10000, 15000, 20000, 30000, 40000, 50000], dtype=float)
EXPECTED_NETWORKS = 30
EXPECTED_DV_INDICES = set(range(20))
EXPECTED_LABELS = {"pairwise", "simplicial"}


def integrate_abs(lag_ms: np.ndarray, acf: np.ndarray, cutoff_ms: float) -> float:
    mask = lag_ms <= cutoff_ms + 1.0e-9
    if mask.sum() < 2:
        raise ValueError(f"Too few lag samples at cutoff {cutoff_ms} ms")
    return float(np.trapz(np.abs(acf[mask]), lag_ms[mask]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-networks", type=int, default=EXPECTED_NETWORKS)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[float, int, str], list[tuple[int, Path]]] = {}
    unexpected: list[str] = []
    for path in sorted(args.results_dir.glob("*.npz")):
        match = FILE_RE.match(path.name)
        if not match:
            unexpected.append(path.name)
            continue
        key = (float(match["alpha"]), int(match["dv_index"]), match["label"])
        groups.setdefault(key, []).append((int(match["network"]), path))

    expected_keys = {
        (alpha, dv_index, label)
        for alpha in (1.5, 2.0)
        for dv_index in EXPECTED_DV_INDICES
        for label in EXPECTED_LABELS
    }
    missing_keys = sorted(expected_keys - set(groups))
    extra_keys = sorted(set(groups) - expected_keys)
    errors: list[str] = []
    if unexpected:
        errors.append(f"Unexpected filenames ({len(unexpected)}): {unexpected[:10]}")
    if missing_keys:
        errors.append(f"Missing condition groups ({len(missing_keys)}): {missing_keys[:10]}")
    if extra_keys:
        errors.append(f"Unexpected condition groups ({len(extra_keys)}): {extra_keys[:10]}")

    rows: list[dict[str, object]] = []
    pooled_payload: dict[str, np.ndarray] = {}
    reference_lag: np.ndarray | None = None

    for key in sorted(expected_keys):
        alpha, dv_index, label = key
        entries = sorted(groups.get(key, []))
        networks = [network for network, _ in entries]
        expected_ids = list(range(args.expected_networks))
        if entries and networks != expected_ids:
            errors.append(
                f"{key}: network IDs are {networks}; expected {expected_ids}"
            )
            if not args.allow_incomplete:
                continue

        acfs: list[np.ndarray] = []
        metadata: list[dict[str, object]] = []
        lag_ms: np.ndarray | None = None
        for network, path in entries:
            with np.load(path, allow_pickle=False) as data:
                current_lag = np.asarray(data["lag_ms"], dtype=float)
                current_acf = np.asarray(data["mean_acf"], dtype=float)
                meta = json.loads(str(data["metadata_json"].item()))
            if current_lag.shape != current_acf.shape:
                raise ValueError(f"{path}: lag and ACF shapes differ")
            if not np.all(np.isfinite(current_acf)):
                raise ValueError(f"{path}: ACF contains non-finite values")
            if lag_ms is None:
                lag_ms = current_lag
            elif not np.array_equal(lag_ms, current_lag):
                raise ValueError(f"{path}: lag grid differs within condition")
            expected_meta = {
                "alpha": alpha,
                "network_index": network,
                "dv_index": dv_index,
                "label": "Pairwise" if label == "pairwise" else "Simplicial",
                "n_trial_batches": 12,
                "n_paths_per_batch": 12,
                "n_trajectories_per_network": 144,
                "n_nodes": 100,
                "current": 88.0,
                "sigma_1": 0.002,
                "p_edge": 0.15,
                "p_fill_simplex": 0.5,
                "dt": 0.05,
                "sample_dt_ms": 0.5,
                "max_lag_ms": 50000.0,
                "t_total": 120000.0,
                "t_warmup": 3000.0,
            }
            for field, expected in expected_meta.items():
                if meta.get(field) != expected:
                    raise ValueError(
                        f"{path}: metadata {field}={meta.get(field)!r}, expected {expected!r}"
                    )
            expected_sigma2 = 0.0 if label == "pairwise" else 0.12
            if meta.get("sigma_2") != expected_sigma2:
                raise ValueError(f"{path}: sigma_2={meta.get('sigma_2')}, expected {expected_sigma2}")
            acfs.append(current_acf)
            metadata.append(meta)

        if not acfs or lag_ms is None:
            continue
        if reference_lag is None:
            reference_lag = lag_ms
            pooled_payload["lag_ms"] = lag_ms
        elif not np.array_equal(reference_lag, lag_ms):
            raise ValueError(f"{key}: lag grid differs from other conditions")

        stack = np.stack(acfs)
        pooled = stack.mean(axis=0)
        t_by_cutoff = {int(c): integrate_abs(lag_ms, pooled, c) for c in CUTOFFS_MS}
        n_networks = stack.shape[0]
        total_acf = stack.sum(axis=0)
        jackknife = np.asarray(
            [integrate_abs(lag_ms, (total_acf - stack[i]) / (n_networks - 1), 50000.0)
             for i in range(n_networks)]
        )
        jk_mean = float(jackknife.mean())
        jk_se = float(np.sqrt((n_networks - 1) / n_networks * np.sum((jackknife - jk_mean) ** 2)))
        tail_mask = (lag_ms >= 49000.0) & (lag_ms <= 50000.0)
        tail_abs_mean = float(np.mean(np.abs(pooled[tail_mask])))
        t40 = t_by_cutoff[40000]
        t50 = t_by_cutoff[50000]
        convergence_fraction_40_50 = float((t50 - t40) / t50)
        dv_values = {float(meta["noise_intensity_dv"]) for meta in metadata}
        if len(dv_values) != 1:
            raise ValueError(f"{key}: inconsistent D_v metadata: {sorted(dv_values)}")
        dv = dv_values.pop()

        row: dict[str, object] = {
            "alpha": alpha,
            "dv_index": dv_index,
            "noise_intensity_dv": dv,
            "label": "Pairwise" if label == "pairwise" else "Simplicial",
            "n_networks": n_networks,
            "n_trajectories_total": n_networks * 12 * 12,
            "tcor_50000_ms": t50,
            "jackknife_se_ms": jk_se,
            "tail_abs_mean_49000_50000": tail_abs_mean,
            "convergence_fraction_40000_50000": convergence_fraction_40_50,
        }
        for cutoff, value in t_by_cutoff.items():
            row[f"tcor_{cutoff}_ms"] = value
        rows.append(row)
        tag = f"a{alpha:.1f}_d{dv_index:02d}_{label}"
        pooled_payload[f"pooled_acf_{tag}"] = pooled
        pooled_payload[f"jackknife_tcor_50000_ms_{tag}"] = jackknife

    if errors and not args.allow_incomplete:
        raise SystemExit("Input validation failed:\n- " + "\n- ".join(errors))

    if not rows:
        raise SystemExit("No complete result groups were aggregated")
    fieldnames = list(rows[0].keys())
    csv_path = args.output_dir / "true_correlation_time_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output_dir / "pooled_acf_and_jackknife.npz", **pooled_payload)
    validation = {
        "input_results_dir": str(args.results_dir),
        "n_npz_files_seen": sum(len(v) for v in groups.values()),
        "n_condition_rows_written": len(rows),
        "expected_condition_rows": 80,
        "expected_networks_per_condition": args.expected_networks,
        "estimator": "integral_0_50s(abs(mean_across_networks(normalized_ACF)))",
        "uncertainty": "delete-one-network jackknife standard error",
        "errors": errors,
    }
    (args.output_dir / "aggregation_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))
    print(csv_path)


if __name__ == "__main__":
    main()
