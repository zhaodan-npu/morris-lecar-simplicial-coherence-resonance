#!/usr/bin/env python3
"""Noise-floor-aware analysis of the long-record Fig. 2 ACF calculations.

The raw estimator integral(abs(mean ACF)) has a positive finite-record floor.
This script retains the scientifically useful ordering (average normalized
ACFs before rectification), estimates the lag-dependent null folded-normal
floor from the 30 independent network ACFs, and reports two debiased numerical
integrals plus a damped-envelope tail extrapolation.  Network delete-one-out
jackknife uncertainties are recomputed for the debiased estimators.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


FILE_RE = re.compile(
    r"^coherence_a(?P<alpha>1\.5|2(?:\.0)?)_n(?P<network>\d{3})_"
    r"d(?P<dv_index>\d{2})_(?P<label>pairwise|simplicial)\.npz$"
)
CUTOFFS_MS = (2000.0, 5000.0, 10000.0, 20000.0, 30000.0, 40000.0, 50000.0)
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def trapz_to(lag_ms: np.ndarray, values: np.ndarray, cutoff_ms: float) -> float:
    mask = lag_ms <= cutoff_ms + 1.0e-9
    return float(np.trapz(values[mask], lag_ms[mask]))


def mean_and_se_from_sums(sum1: np.ndarray, sum2: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    mean = sum1 / n
    if n < 2:
        return mean, np.full_like(mean, np.nan)
    centered_ss = np.maximum(sum2 - n * mean * mean, 0.0)
    sample_var = centered_ss / (n - 1)
    return mean, np.sqrt(sample_var / n)


def debiased_amplitudes(mean: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folded_floor = SQRT_2_OVER_PI * se
    floor_subtracted = np.maximum(np.abs(mean) - folded_floor, 0.0)
    quadrature = np.sqrt(np.maximum(mean * mean - se * se, 0.0))
    return folded_floor, floor_subtracted, quadrature


def jackknife_se(values: np.ndarray) -> float:
    n = values.size
    center = float(np.mean(values))
    return float(np.sqrt((n - 1) / n * np.sum((values - center) ** 2)))


def estimate_frequency_hz(acf: np.ndarray, dt_ms: float) -> float:
    n = min(acf.size, int(round(10000.0 / dt_ms)) + 1)
    x = acf[:n] - np.mean(acf[:n])
    spectrum = np.abs(np.fft.rfft(x))
    frequency = np.fft.rfftfreq(n, d=dt_ms / 1000.0)
    band = (frequency >= 2.0) & (frequency <= 40.0)
    if not np.any(band):
        return float("nan")
    indices = np.flatnonzero(band)
    return float(frequency[indices[np.argmax(spectrum[band])]])


def fit_stretched_envelope(
    lag_ms: np.ndarray,
    acf: np.ndarray,
    se: np.ndarray,
) -> dict[str, float]:
    """Fit A exp[-(t/tau)^beta] to half-cycle maxima above 3 network SE."""
    dt_ms = float(lag_ms[1] - lag_ms[0])
    frequency_hz = estimate_frequency_hz(acf, dt_ms)
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        return {"fit_ok": 0.0}
    half_period_ms = 500.0 / frequency_hz
    bin_samples = max(3, int(round(half_period_ms / dt_ms)))
    absolute = np.abs(acf)
    peak_indices: list[int] = []
    for start in range(bin_samples, absolute.size - bin_samples, bin_samples):
        stop = min(start + bin_samples, absolute.size)
        local = start + int(np.argmax(absolute[start:stop]))
        peak_indices.append(local)
    if not peak_indices:
        return {"fit_ok": 0.0}
    idx = np.asarray(peak_indices, dtype=int)
    peak_t = lag_ms[idx]
    peak_a = absolute[idx]
    peak_se = np.maximum(se[idx], 1.0e-12)
    peak_snr = peak_a / peak_se
    signal = (peak_snr >= 3.0) & (peak_a > 1.0e-7)
    signal_idx = np.flatnonzero(signal)
    if signal_idx.size < 8:
        return {
            "fit_ok": 0.0,
            "fit_frequency_hz": frequency_hz,
            "fit_signal_last_ms": float(peak_t[signal_idx[-1]]) if signal_idx.size else 0.0,
        }
    last = int(signal_idx[-1])
    last_t = float(peak_t[last])
    minimum_fit_t = max(5.0 * half_period_ms, 0.15 * last_t)
    fit_mask = signal & (peak_t >= minimum_fit_t) & (peak_t <= last_t)
    fit_indices = np.flatnonzero(fit_mask)
    if fit_indices.size < 8:
        fit_indices = signal_idx[max(0, signal_idx.size - 20):]
    t = peak_t[fit_indices]
    y = np.log(np.maximum(peak_a[fit_indices], 1.0e-12))
    w = np.minimum(peak_snr[fit_indices], 20.0)
    best: tuple[float, float, float, float] | None = None
    for beta in np.linspace(0.50, 2.00, 151):
        z = np.power(t, beta)
        sw = float(np.sum(w))
        zbar = float(np.sum(w * z) / sw)
        ybar = float(np.sum(w * y) / sw)
        denominator = float(np.sum(w * (z - zbar) ** 2))
        if denominator <= 0.0:
            continue
        slope = float(np.sum(w * (z - zbar) * (y - ybar)) / denominator)
        if slope >= 0.0:
            continue
        intercept = ybar - slope * zbar
        residual = y - (intercept + slope * z)
        score = float(np.sum(w * residual * residual) / sw)
        tau_ms = float((-1.0 / slope) ** (1.0 / beta))
        if not np.isfinite(tau_ms) or tau_ms <= 0.0:
            continue
        if best is None or score < best[0]:
            best = (score, beta, tau_ms, intercept)
    if best is None:
        return {"fit_ok": 0.0, "fit_frequency_hz": frequency_hz, "fit_signal_last_ms": last_t}
    score, beta, tau_ms, intercept = best
    amplitude = float(np.exp(intercept))
    ybar = float(np.average(y, weights=w))
    ss_total = float(np.sum(w * (y - ybar) ** 2))
    fit_r2 = float(1.0 - score * np.sum(w) / ss_total) if ss_total > 0.0 else float("nan")

    # Numerically integrate the fitted envelope from the last signal-dominated
    # peak to an exponent of 30, then use <|cos|> = 2/pi for the oscillatory tail.
    tail_end_ms = max(last_t, tau_ms * (30.0 ** (1.0 / beta)))
    tail_end_ms = min(tail_end_ms, 10_000_000.0)
    tail_grid = np.linspace(last_t, tail_end_ms, 200_001)
    envelope = amplitude * np.exp(-np.power(tail_grid / tau_ms, beta))
    fitted_tail_ms = float((2.0 / math.pi) * np.trapz(envelope, tail_grid))
    return {
        "fit_ok": 1.0,
        "fit_frequency_hz": frequency_hz,
        "fit_half_period_ms": half_period_ms,
        "fit_signal_last_ms": last_t,
        "fit_n_peaks": float(fit_indices.size),
        "fit_beta": beta,
        "fit_tau_ms": tau_ms,
        "fit_amplitude": amplitude,
        "fit_log_r2": fit_r2,
        "fit_tail_ms": fitted_tail_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[float, int, str], list[tuple[int, Path]]] = {}
    for path in sorted(args.results_dir.glob("*.npz")):
        match = FILE_RE.match(path.name)
        if match:
            key = (float(match["alpha"]), int(match["dv_index"]), match["label"])
            groups.setdefault(key, []).append((int(match["network"]), path))
    expected_keys = {
        (alpha, dv_index, label)
        for alpha in (1.5, 2.0)
        for dv_index in range(20)
        for label in ("pairwise", "simplicial")
    }
    if set(groups) != expected_keys:
        raise SystemExit(f"Condition-key mismatch: missing={expected_keys-set(groups)}, extra={set(groups)-expected_keys}")

    rows: list[dict[str, object]] = []
    diagnostics: dict[str, np.ndarray] = {}
    for alpha, dv_index, label in sorted(expected_keys):
        entries = sorted(groups[(alpha, dv_index, label)])
        if [i for i, _ in entries] != list(range(30)):
            raise SystemExit(f"Missing network for {(alpha, dv_index,label)}")
        acfs: list[np.ndarray] = []
        lag_ms: np.ndarray | None = None
        metadata: list[dict[str, object]] = []
        for _, path in entries:
            with np.load(path, allow_pickle=False) as data:
                current_lag = np.asarray(data["lag_ms"], dtype=float)
                current_acf = np.asarray(data["mean_acf"], dtype=float)
                meta = json.loads(str(data["metadata_json"].item()))
            if lag_ms is None:
                lag_ms = current_lag
            elif not np.array_equal(lag_ms, current_lag):
                raise SystemExit(f"Lag mismatch in {(alpha,dv_index,label)}")
            acfs.append(current_acf)
            metadata.append(meta)
        assert lag_ms is not None
        stack = np.stack(acfs)
        n = stack.shape[0]
        sum1 = np.sum(stack, axis=0)
        sum2 = np.sum(stack * stack, axis=0)
        pooled, pooled_se = mean_and_se_from_sums(sum1, sum2, n)
        floor, floor_db, quad_db = debiased_amplitudes(pooled, pooled_se)
        observation_ms = float(metadata[0]["t_total"] - metadata[0]["t_warmup"])
        triangular = np.maximum(1.0 - lag_ms / observation_ms, 0.0)
        biased_pooled = pooled * triangular

        row: dict[str, object] = {
            "alpha": alpha,
            "dv_index": dv_index,
            "noise_intensity_dv": float(metadata[0]["noise_intensity_dv"]),
            "label": "Pairwise" if label == "pairwise" else "Simplicial",
            "n_networks": n,
            "observation_ms": observation_ms,
            "tail_raw_abs_mean_49000_50000": float(np.mean(np.abs(pooled[(lag_ms>=49000)&(lag_ms<=50000)]))),
            "tail_folded_floor_mean_49000_50000": float(np.mean(floor[(lag_ms>=49000)&(lag_ms<=50000)])),
            "tail_floor_db_mean_49000_50000": float(np.mean(floor_db[(lag_ms>=49000)&(lag_ms<=50000)])),
        }
        for cutoff in CUTOFFS_MS:
            token = int(cutoff)
            row[f"raw_unbiased_{token}_ms"] = trapz_to(lag_ms, np.abs(pooled), cutoff)
            row[f"raw_biased_{token}_ms"] = trapz_to(lag_ms, np.abs(biased_pooled), cutoff)
            row[f"floor_db_{token}_ms"] = trapz_to(lag_ms, floor_db, cutoff)
            row[f"quadrature_db_{token}_ms"] = trapz_to(lag_ms, quad_db, cutoff)

        loo_floor: list[float] = []
        loo_quad: list[float] = []
        for i in range(n):
            mean_i, se_i = mean_and_se_from_sums(sum1 - stack[i], sum2 - stack[i] * stack[i], n - 1)
            _, floor_i, quad_i = debiased_amplitudes(mean_i, se_i)
            loo_floor.append(trapz_to(lag_ms, floor_i, 50000.0))
            loo_quad.append(trapz_to(lag_ms, quad_i, 50000.0))
        row["floor_db_jackknife_se_ms"] = jackknife_se(np.asarray(loo_floor))
        row["quadrature_db_jackknife_se_ms"] = jackknife_se(np.asarray(loo_quad))

        fit = fit_stretched_envelope(lag_ms, pooled, pooled_se)
        row.update(fit)
        if fit.get("fit_ok", 0.0) == 1.0:
            signal_last = float(fit["fit_signal_last_ms"])
            row["envelope_extrapolated_ms"] = trapz_to(lag_ms, floor_db, signal_last) + float(fit["fit_tail_ms"])
        else:
            row["envelope_extrapolated_ms"] = float("nan")

        rows.append(row)
        tag = f"a{alpha:.1f}_d{dv_index:02d}_{label}"
        diagnostics[f"pooled_{tag}"] = pooled
        diagnostics[f"se_{tag}"] = pooled_se
        diagnostics[f"floor_db_amplitude_{tag}"] = floor_db
        diagnostics[f"quadrature_db_amplitude_{tag}"] = quad_db

    csv_path = args.output_dir / "true_correlation_time_debiased.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output_dir / "true_correlation_time_debiased_diagnostics.npz", lag_ms=lag_ms, **diagnostics)
    validation = {
        "conditions": len(rows),
        "networks_per_condition": 30,
        "primary_candidate": "lag-dependent folded-normal-floor-subtracted integral",
        "cross_checks": ["quadrature noise debiasing", "stretched-exponential envelope extrapolation", "biased/unbiased ACF"],
        "jackknife_unit": "network",
    }
    (args.output_dir / "debiased_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    print(csv_path)


if __name__ == "__main__":
    main()
