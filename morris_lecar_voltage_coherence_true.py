from __future__ import annotations

"""Voltage-based coherence metrics for the simplicial Morris--Lecar model.

The primary observable is the node-resolved membrane voltage.  The
population-mean membrane voltage is retained as a complementary collective
observable.

For each network realization and condition, normalized connected
autocorrelation functions are averaged over sampled nodes and paired
stochastic paths before the absolute correlation time is integrated.  A
spectral peak-to-background ratio and the FWHM of the dominant peak are
reported as complementary checks.

This module deliberately does not infer these quantities from ISIs: the
continuous voltage trajectory remains defined in the rare-spike regime where
ISI CV is poorly sampled.
"""

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

import morris_lecar_paired_alpha_compare_simplicial as base
from morris_lecar_simplicial_paper_suite import PaperSuiteConfig, REFINED_DV_VALUES, candidate39_params


LABELS = ("Pairwise", "Simplicial")


def exact_paper_params(n_paths: int) -> base.PairedSimplicialParams:
    """Return the Fig. 2 candidate-39 parameters with a configurable batch size."""
    config = PaperSuiteConfig(n_trials_per_network=n_paths)
    return candidate39_params(config)


def simulate_population_voltage(
    p: base.PairedSimplicialParams,
    network: base.SimplicialNetwork,
    *,
    alpha: float,
    noise_intensity_dv: float,
    higher_order_enabled: bool,
    init_seed: int,
    noise_seed: int,
    trace_stride: int = 10,
    sampled_nodes: int = 12,
) -> dict[str, np.ndarray]:
    """Retain decimated population-mean and representative node voltages."""
    n_steps = int(round(p.t_total / p.dt))
    warm_steps = int(round(p.t_warmup / p.dt))
    n_paths = p.n_trials_per_network
    n = p.n_nodes

    if trace_stride < 1:
        raise ValueError("trace_stride must be at least 1")
    n_kept = (n_steps - warm_steps + trace_stride - 1) // trace_stride
    traces = np.empty((n_paths, n_kept), dtype=np.float32)
    sampled_nodes = min(max(int(sampled_nodes), 1), n)
    node_indices = np.linspace(0, n - 1, sampled_nodes, dtype=int)
    node_traces = np.empty((n_paths, sampled_nodes, n_kept), dtype=np.float32)

    init_rng = np.random.default_rng(init_seed)
    noise_rng = np.random.default_rng(noise_seed)
    v = -60.0 + 2.0 * init_rng.standard_normal((n_paths, n))
    w = base.w_inf(v, p)
    prev_v = v.copy()
    operator = network.pairwise_operator + (p.sigma_2 * network.l2 if higher_order_enabled else 0.0)
    noise_scale = base.noise_scale_from_voltage_noise_intensity(
        noise_intensity_dv, alpha, p.capacitance
    )

    spike_count = np.zeros(n_paths, dtype=np.int32)
    write_idx = 0
    for step in range(n_steps):
        coupling = v @ operator.T
        i_ca = p.g_ca * base.m_inf(v, p) * (v - p.v_ca)
        i_k = p.g_k * w * (v - p.v_k)
        i_l = p.g_l * (v - p.v_l)
        dv = (p.current - i_ca - i_k - i_l + coupling) / p.capacitance
        dw = p.phi * (base.w_inf(v, p) - w) / np.maximum(base.tau_w(v, p), 1e-6)

        jump = base.levy_increment(
            noise_scale, alpha, p.beta, p.dt, (n_paths, n), noise_rng
        ) / p.capacitance
        v = np.clip(v + p.dt * dv + jump, -200.0, 200.0)
        w = np.clip(w + p.dt * dw, -0.5, 1.5)

        if step >= warm_steps:
            spike_count += np.count_nonzero(
                (prev_v < p.spike_threshold) & (v >= p.spike_threshold), axis=1
            )
            if (step - warm_steps) % trace_stride == 0:
                traces[:, write_idx] = np.mean(v, axis=1)
                node_traces[:, :, write_idx] = v[:, node_indices]
                write_idx += 1
        prev_v = v.copy()

    if write_idx != n_kept:
        raise RuntimeError(f"trace length mismatch: wrote {write_idx}, expected {n_kept}")
    return {
        "population_voltage": traces,
        "node_voltage": node_traces,
        "node_indices": node_indices,
        "spike_count_total": spike_count.astype(float),
    }


def normalized_acf_batch(traces: np.ndarray, max_lag_samples: int) -> np.ndarray:
    """Unbiased, connected, normalized ACF for every row, evaluated by FFT."""
    x = np.asarray(traces, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("traces must have shape (n_paths, n_samples)")
    x = x - np.mean(x, axis=1, keepdims=True)
    n = x.shape[1]
    max_lag_samples = min(int(max_lag_samples), n - 1)
    nfft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, n=nfft, axis=1)
    acov = np.fft.irfft(spectrum * spectrum.conj(), n=nfft, axis=1)[:, : max_lag_samples + 1]
    acov /= np.arange(n, n - max_lag_samples - 1, -1, dtype=float)[None, :]
    variance = acov[:, :1]
    valid = variance[:, 0] > 1e-14
    acf = np.full_like(acov, np.nan)
    acf[valid] = acov[valid] / variance[valid]
    return acf


def correlation_time_from_acf(acf: np.ndarray, sample_dt_ms: float) -> float:
    """Integral of |C(tau)/C(0)| over the supplied, predeclared lag window."""
    values = np.asarray(acf, dtype=float)
    if not np.all(np.isfinite(values)):
        return float("nan")
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(np.abs(values), dx=sample_dt_ms))


def mean_welch_psd(traces: np.ndarray, sample_dt_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean one-sided Welch PSD, implemented in NumPy for cluster portability."""
    fs_hz = 1000.0 / sample_dt_ms
    nperseg = min(8192, traces.shape[1])
    if nperseg < 256:
        raise ValueError("trace is too short for a stable spectrum")
    step = nperseg // 2
    starts = range(0, traces.shape[1] - nperseg + 1, step)
    window = np.hanning(nperseg)
    scale = 1.0 / (fs_hz * np.sum(window**2))
    psd_sum = np.zeros(nperseg // 2 + 1, dtype=float)
    count = 0
    for trace in np.asarray(traces, dtype=float):
        for start in starts:
            segment = trace[start : start + nperseg]
            segment = (segment - np.mean(segment)) * window
            periodogram = np.abs(np.fft.rfft(segment)) ** 2 * scale
            if nperseg % 2 == 0:
                periodogram[1:-1] *= 2.0
            else:
                periodogram[1:] *= 2.0
            psd_sum += periodogram
            count += 1
    if count == 0:
        raise ValueError("trace is too short to form a Welch segment")
    return np.fft.rfftfreq(nperseg, d=1.0 / fs_hz), psd_sum / count


def spectral_metrics(
    freqs_hz: np.ndarray,
    psd: np.ndarray,
    *,
    fmin_hz: float = 0.2,
    fmax_hz: float = 100.0,
) -> dict[str, float | bool]:
    """Dominant peak, peak/background ratio, FWHM and equivalent decay time."""
    mask = (freqs_hz >= fmin_hz) & (freqs_hz <= fmax_hz) & np.isfinite(psd)
    idx = np.flatnonzero(mask)
    if idx.size < 7:
        raise ValueError("insufficient frequency bins in the requested band")
    band = np.asarray(psd[idx], dtype=float)
    local_peak = int(np.argmax(band))
    peak_idx = int(idx[local_peak])
    peak_f = float(freqs_hz[peak_idx])
    peak_power = float(psd[peak_idx])

    # Estimate a robust local floor, excluding 20% around the dominant peak.
    exclusion = max(1.0, 0.2 * peak_f)
    background_mask = mask & (np.abs(freqs_hz - peak_f) > exclusion)
    background_values = psd[background_mask]
    background = float(np.median(background_values)) if background_values.size else float("nan")
    ratio = peak_power / max(background, np.finfo(float).tiny)
    snr_db = float(10.0 * np.log10(ratio))

    half_level = background + 0.5 * max(peak_power - background, 0.0)
    left = peak_idx
    while left > idx[0] and psd[left] > half_level:
        left -= 1
    right = peak_idx
    while right < idx[-1] and psd[right] > half_level:
        right += 1
    at_boundary = left == idx[0] or right == idx[-1]
    fwhm_hz = float(freqs_hz[right] - freqs_hz[left]) if right > left else float("nan")
    spectral_tau_ms = (
        float(1000.0 / (math.pi * fwhm_hz)) if fwhm_hz > 0.0 and not at_boundary else float("nan")
    )
    return {
        "peak_frequency_hz": peak_f,
        "peak_power": peak_power,
        "background_power": background,
        "peak_background_ratio": ratio,
        "peak_background_db": snr_db,
        "fwhm_hz": fwhm_hz,
        "spectral_tau_ms": spectral_tau_ms,
        "peak_at_band_boundary": bool(at_boundary or local_peak in (0, band.size - 1)),
    }


def condition_seeds(
    p: base.PairedSimplicialParams,
    network_idx: int,
    dv_idx: int,
    alpha: float,
    trial_idx: int | None,
) -> tuple[int, int]:
    """Return paired seeds, reproducing the paper trial hierarchy when requested."""
    if trial_idx is None:
        # Backward-compatible seeds used by the initial 12-path validation.
        init_seed = p.base_seed + base.stable_seed("coherence_init", network_idx, dv_idx, alpha)
        noise_seed = p.base_seed + base.stable_seed("coherence_noise", network_idx, dv_idx, alpha)
    else:
        # Exact seed hierarchy used by the production CV calculation.
        init_seed = p.base_seed + base.stable_seed("init", network_idx, trial_idx)
        noise_seed = p.base_seed + base.stable_seed(
            "noise", alpha, dv_idx, network_idx, trial_idx
        )
    return init_seed, noise_seed


def run_condition(args: argparse.Namespace) -> Path:
    p = exact_paper_params(args.n_paths)
    p.t_total = args.t_total
    p.t_warmup = args.t_warmup
    dv_values = tuple(args.dv_values) if args.dv_values else REFINED_DV_VALUES
    dv = float(dv_values[args.dv_index])
    label = LABELS[args.label_index]
    # Reproduce the exact topology realizations used by the production
    # candidate-39/Fig. 2 calculation.  The seed namespace is part of the
    # stochastic design: using a different label would preserve the graph
    # distribution but generate a different set of 30 networks.
    network_rng = np.random.default_rng(
        p.base_seed + base.stable_seed("paper_network", args.network_index)
    )
    network = base.build_simplicial_network(p, network_rng)
    init_seed, noise_seed = condition_seeds(
        p, args.network_index, args.dv_index, args.alpha, args.trial_index
    )
    simulation = simulate_population_voltage(
        p,
        network,
        alpha=args.alpha,
        noise_intensity_dv=dv,
        higher_order_enabled=(label == "Simplicial"),
        init_seed=init_seed,
        noise_seed=noise_seed,
        trace_stride=args.trace_stride,
        sampled_nodes=args.sampled_nodes,
    )

    population_traces = simulation["population_voltage"]
    node_traces = simulation["node_voltage"].reshape(-1, simulation["node_voltage"].shape[-1])
    sample_dt_ms = p.dt * args.trace_stride
    max_lag_samples = int(round(args.max_lag_ms / sample_dt_ms))
    acf_paths = normalized_acf_batch(node_traces, max_lag_samples)
    valid = np.all(np.isfinite(acf_paths), axis=1)
    mean_acf = np.mean(acf_paths[valid], axis=0) if np.any(valid) else np.full(max_lag_samples + 1, np.nan)
    tcor_ms = correlation_time_from_acf(mean_acf, sample_dt_ms)
    freqs_hz, mean_psd = mean_welch_psd(node_traces, sample_dt_ms)
    spec = spectral_metrics(freqs_hz, mean_psd)

    population_acf_paths = normalized_acf_batch(population_traces, max_lag_samples)
    population_valid = np.all(np.isfinite(population_acf_paths), axis=1)
    population_mean_acf = (
        np.mean(population_acf_paths[population_valid], axis=0)
        if np.any(population_valid)
        else np.full(max_lag_samples + 1, np.nan)
    )
    population_tcor_ms = correlation_time_from_acf(population_mean_acf, sample_dt_ms)
    population_freqs_hz, population_mean_psd = mean_welch_psd(population_traces, sample_dt_ms)
    population_spec = spectral_metrics(population_freqs_hz, population_mean_psd)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_token = "" if args.trial_index is None else f"_t{args.trial_index:02d}"
    stem = (
        f"coherence_a{args.alpha:g}_n{args.network_index:03d}_d{args.dv_index:02d}"
        f"{trial_token}_{label.lower()}"
    )
    output_path = out_dir / f"{stem}.npz"
    metadata = {
        "alpha": args.alpha,
        "network_index": args.network_index,
        "dv_index": args.dv_index,
        "trial_index": args.trial_index,
        "noise_intensity_dv": dv,
        "label": label,
        "n_paths": args.n_paths,
        "n_sampled_nodes": int(simulation["node_indices"].size),
        "sample_dt_ms": sample_dt_ms,
        "max_lag_ms": args.max_lag_ms,
        "t_total": p.t_total,
        "t_warmup": p.t_warmup,
        "n_nodes": p.n_nodes,
        "current": p.current,
        "sigma_1": p.sigma_1,
        "sigma_2": 0.0 if label == "Pairwise" else p.sigma_2,
        "p_edge": p.p_edge,
        "p_fill_simplex": p.p_fill_simplex,
        "dt": p.dt,
        "beta": p.beta,
        "init_seed": init_seed,
        "noise_seed": noise_seed,
        "correlation_time_ms": tcor_ms,
        "node_voltage_std_mean": float(np.mean(np.std(node_traces, axis=1, ddof=1))),
        "population_correlation_time_ms": population_tcor_ms,
        "population_voltage_std_mean": float(np.mean(np.std(population_traces, axis=1, ddof=1))),
        "spike_count_mean": float(np.mean(simulation["spike_count_total"])),
        "edge_count": network.edge_count,
        "clique_count": network.clique_count,
        "simplex_count": network.simplex_count,
        **spec,
        **{f"population_{key}": value for key, value in population_spec.items()},
    }
    np.savez_compressed(
        output_path,
        lag_ms=np.arange(mean_acf.size) * sample_dt_ms,
        mean_acf=mean_acf,
        frequency_hz=freqs_hz,
        mean_psd=mean_psd,
        population_mean_acf=population_mean_acf,
        population_frequency_hz=population_freqs_hz,
        population_mean_psd=population_mean_psd,
        node_indices=simulation["node_indices"],
        metadata_json=json.dumps(metadata),
    )
    print(json.dumps({"output": str(output_path), **metadata}, indent=2))
    return output_path


def _load_npz_record(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        arrays = {
            name: np.asarray(data[name])
            for name in (
                "lag_ms",
                "mean_acf",
                "frequency_hz",
                "mean_psd",
                "population_mean_acf",
                "population_frequency_hz",
                "population_mean_psd",
                "node_indices",
            )
        }
    return metadata, arrays


def _assert_same_grid(reference: np.ndarray, candidate: np.ndarray, name: str) -> None:
    if reference.shape != candidate.shape or not np.allclose(reference, candidate, rtol=0.0, atol=1e-12):
        raise ValueError(f"inconsistent {name} grid across trial batches")


def combine_trial_batches(args: argparse.Namespace) -> list[Path]:
    """Combine 12 paper-style trial batches into one network-level record per label."""
    if args.input_dir is None:
        raise ValueError("--input-dir is required in combine mode")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dv_values = tuple(args.dv_values) if args.dv_values else REFINED_DV_VALUES
    dv = float(dv_values[args.dv_index])
    outputs: list[Path] = []

    for label in LABELS:
        pattern = (
            f"coherence_a{args.alpha:g}_n{args.network_index:03d}_d{args.dv_index:02d}"
            f"_t*_{label.lower()}.npz"
        )
        paths = sorted(input_dir.glob(pattern))
        if len(paths) != args.n_trial_batches:
            raise RuntimeError(
                f"expected {args.n_trial_batches} trial batches for {label}, found {len(paths)}"
            )

        records = [_load_npz_record(path) for path in paths]
        metadata_rows = [record[0] for record in records]
        array_rows = [record[1] for record in records]
        trial_indices = sorted(int(row["trial_index"]) for row in metadata_rows)
        if trial_indices != list(range(args.n_trial_batches)):
            raise RuntimeError(f"unexpected trial indices for {label}: {trial_indices}")

        first = array_rows[0]
        for arrays in array_rows[1:]:
            _assert_same_grid(first["lag_ms"], arrays["lag_ms"], "lag")
            _assert_same_grid(first["frequency_hz"], arrays["frequency_hz"], "frequency")
            _assert_same_grid(
                first["population_frequency_hz"],
                arrays["population_frequency_hz"],
                "population frequency",
            )
            if not np.array_equal(first["node_indices"], arrays["node_indices"]):
                raise ValueError("inconsistent sampled node indices across trial batches")

        mean_acf = np.mean([arrays["mean_acf"] for arrays in array_rows], axis=0)
        mean_psd = np.mean([arrays["mean_psd"] for arrays in array_rows], axis=0)
        population_mean_acf = np.mean(
            [arrays["population_mean_acf"] for arrays in array_rows], axis=0
        )
        population_mean_psd = np.mean(
            [arrays["population_mean_psd"] for arrays in array_rows], axis=0
        )
        sample_dt_ms = float(metadata_rows[0]["sample_dt_ms"])
        tcor_ms = correlation_time_from_acf(mean_acf, sample_dt_ms)
        population_tcor_ms = correlation_time_from_acf(population_mean_acf, sample_dt_ms)
        spec = spectral_metrics(first["frequency_hz"], mean_psd)
        population_spec = spectral_metrics(
            first["population_frequency_hz"], population_mean_psd
        )

        metadata = dict(metadata_rows[0])
        metadata.update(
            {
                "trial_index": None,
                "n_trial_batches": args.n_trial_batches,
                "n_paths_per_batch": int(metadata_rows[0]["n_paths"]),
                "n_trajectories_per_network": int(
                    args.n_trial_batches * metadata_rows[0]["n_paths"]
                ),
                "aggregation_unit": "network",
                "correlation_time_ms": tcor_ms,
                "population_correlation_time_ms": population_tcor_ms,
                "node_voltage_std_mean": float(
                    np.mean([row["node_voltage_std_mean"] for row in metadata_rows])
                ),
                "population_voltage_std_mean": float(
                    np.mean([row["population_voltage_std_mean"] for row in metadata_rows])
                ),
                "spike_count_mean": float(
                    np.mean([row["spike_count_mean"] for row in metadata_rows])
                ),
                "init_seed": None,
                "noise_seed": None,
                "trial_init_seeds": [int(row["init_seed"]) for row in metadata_rows],
                "trial_noise_seeds": [int(row["noise_seed"]) for row in metadata_rows],
                **spec,
                **{f"population_{key}": value for key, value in population_spec.items()},
            }
        )
        stem = (
            f"coherence_a{args.alpha:g}_n{args.network_index:03d}_d{args.dv_index:02d}"
            f"_{label.lower()}"
        )
        output_path = output_dir / f"{stem}.npz"
        np.savez_compressed(
            output_path,
            lag_ms=first["lag_ms"],
            mean_acf=mean_acf,
            frequency_hz=first["frequency_hz"],
            mean_psd=mean_psd,
            population_mean_acf=population_mean_acf,
            population_frequency_hz=first["population_frequency_hz"],
            population_mean_psd=population_mean_psd,
            node_indices=first["node_indices"],
            metadata_json=json.dumps(metadata),
        )
        outputs.append(output_path)

    print(
        json.dumps(
            {
                "alpha": args.alpha,
                "network_index": args.network_index,
                "dv_index": args.dv_index,
                "noise_intensity_dv": dv,
                "n_trial_batches": args.n_trial_batches,
                "outputs": [str(path) for path in outputs],
            },
            indent=2,
        )
    )
    return outputs


def load_condition_rows(input_dir: Path, alpha: float) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(input_dir.glob(f"coherence_a{alpha:g}_n*_d*_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            row = json.loads(str(data["metadata_json"]))
        row["source_file"] = path.name
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((float(row["noise_intensity_dv"]), row["label"]), []).append(row)
    metric_names = (
        "correlation_time_ms",
        "peak_background_db",
        "spectral_tau_ms",
        "peak_frequency_hz",
        "fwhm_hz",
        "node_voltage_std_mean",
        "population_correlation_time_ms",
        "population_peak_background_db",
        "population_spectral_tau_ms",
        "population_voltage_std_mean",
        "spike_count_mean",
    )
    output: list[dict] = []
    for (dv, label), group in sorted(grouped.items()):
        item: dict[str, float | int | str] = {
            "noise_intensity_dv": dv,
            "label": label,
            "n_networks": len(group),
        }
        for name in metric_names:
            values = np.asarray([entry[name] for entry in group], dtype=float)
            values = values[np.isfinite(values)]
            item[f"{name}_mean"] = float(np.mean(values)) if values.size else float("nan")
            item[f"{name}_sem"] = (
                float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
            )
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def configure_prl_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_metric(summary: list[dict], metric: str, ylabel: str, output: Path, alpha: float) -> None:
    import matplotlib.pyplot as plt

    configure_prl_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    styles = {
        "Pairwise": dict(color="#374955", marker="o", mfc="#374955"),
        "Simplicial": dict(color="#d32f2f", marker="s", mfc="#d32f2f"),
    }
    for label in LABELS:
        rows = [row for row in summary if row["label"] == label]
        x = np.asarray([row["noise_intensity_dv"] for row in rows], dtype=float)
        y = np.asarray([row[f"{metric}_mean"] for row in rows], dtype=float)
        e = np.asarray([row[f"{metric}_sem"] for row in rows], dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=e,
            lw=1.25,
            ms=3.6,
            capsize=1.8,
            label=label,
            **styles[label],
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$D_v$")
    ax.set_ylabel(ylabel)
    ax.text(0.04, 0.94, rf"$\alpha={alpha:.1f}$", transform=ax.transAxes, ha="left", va="top")
    ax.legend(frameon=False, loc="best")
    ax.minorticks_on()
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.19, top=0.97)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_summary_outputs(summary: list[dict], input_dir: Path, alpha: float) -> None:
    alpha_token = f"{alpha:.1f}".replace(".", "p")
    plot_metric(
        summary,
        "correlation_time_ms",
        r"Correlation time $t_{\mathrm{cor}}$ (ms)",
        input_dir / f"figS_tcor_alpha{alpha_token}.pdf",
        alpha,
    )
    plot_metric(
        summary,
        "peak_background_db",
        "Spectral peak/background (dB)",
        input_dir / f"figS_spectral_ratio_alpha{alpha_token}.pdf",
        alpha,
    )


def run_aggregate(args: argparse.Namespace) -> None:
    input_dir = Path(args.output_dir)
    rows = load_condition_rows(input_dir, args.alpha)
    if not rows:
        raise FileNotFoundError(f"no condition files found in {input_dir}")
    raw_path = input_dir / f"coherence_alpha{args.alpha:g}_network_metrics.csv"
    summary_path = input_dir / f"coherence_alpha{args.alpha:g}_summary.csv"
    write_csv(raw_path, rows)
    summary = summarize_rows(rows)
    write_csv(summary_path, summary)
    plot_summary_outputs(summary, input_dir, args.alpha)
    print(json.dumps({"raw": str(raw_path), "summary": str(summary_path), "n_files": len(rows)}, indent=2))


def run_plot_summary(args: argparse.Namespace) -> None:
    input_dir = Path(args.output_dir)
    summary_path = input_dir / f"coherence_alpha{args.alpha:g}_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = read_csv(summary_path)
    plot_summary_outputs(summary, input_dir, args.alpha)
    print(json.dumps({"summary": str(summary_path), "n_rows": len(summary)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("condition", "combine", "aggregate", "plot-summary"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-dir")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--network-index", type=int, default=0)
    parser.add_argument("--dv-index", type=int, default=0)
    parser.add_argument("--trial-index", type=int)
    parser.add_argument("--n-trial-batches", type=int, default=12)
    parser.add_argument("--label-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--n-paths", type=int, default=12)
    parser.add_argument("--trace-stride", type=int, default=10)
    parser.add_argument("--sampled-nodes", type=int, default=12)
    parser.add_argument("--max-lag-ms", type=float, default=2000.0)
    parser.add_argument("--t-total", type=float, default=15000.0)
    parser.add_argument("--t-warmup", type=float, default=3000.0)
    parser.add_argument("--dv-values", type=float, nargs="*")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "condition":
        run_condition(args)
    elif args.mode == "combine":
        combine_trial_batches(args)
    elif args.mode == "plot-summary":
        run_plot_summary(args)
    else:
        run_aggregate(args)


if __name__ == "__main__":
    main()
