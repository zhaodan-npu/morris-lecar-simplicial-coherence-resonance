from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

import morris_lecar_paired_alpha_compare_simplicial as base


DV_VALUES = (0.04, 0.09, 0.16, 0.25, 0.30, 0.40, 0.56, 0.75, 1.00, 1.56, 2.25, 4.00, 9.00)
V_LOW_VALUES = (-30.0, -20.0, -10.0)
BIN_SIZES = (5.0, 10.0, 20.0, 50.0)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def cv(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    mean = float(np.mean(arr))
    return float(np.std(arr, ddof=1) / mean) if mean > 0 else np.nan


def corr(values_a, values_b):
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return np.nan
    a = a[valid]
    b = b[valid]
    if np.std(a, ddof=1) <= 0 or np.std(b, ddof=1) <= 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def variance_fraction(part, total):
    p = np.asarray(part, dtype=float)
    t = np.asarray(total, dtype=float)
    valid = np.isfinite(p) & np.isfinite(t)
    if valid.sum() < 3:
        return np.nan
    var_total = float(np.var(t[valid], ddof=1))
    if var_total <= 0:
        return np.nan
    return float(np.var(p[valid], ddof=1) / var_total)


def corr_summary(binary: np.ndarray, simplices: list[tuple[int, int, int]]):
    active = binary.sum(axis=1) > 1
    if active.sum() < 2:
        return np.nan, np.nan
    x = binary[active].astype(float)
    std = x.std(axis=1)
    valid = std > 0
    if valid.sum() < 2:
        return np.nan, np.nan
    original_idx = np.where(active)[0][valid]
    cmat = np.corrcoef(x[valid])
    iu = np.triu_indices(cmat.shape[0], 1)
    all_corr = float(np.nanmean(cmat[iu])) if iu[0].size else np.nan

    index_map = {node: idx for idx, node in enumerate(original_idx.tolist())}
    simplex_vals = []
    for i, j, k in simplices:
        if i in index_map and j in index_map and k in index_map:
            ii, jj, kk = index_map[i], index_map[j], index_map[k]
            simplex_vals.extend([cmat[ii, jj], cmat[ii, kk], cmat[jj, kk]])
    simplex_corr = float(np.nanmean(simplex_vals)) if simplex_vals else np.nan
    return all_corr, simplex_corr


def synchrony_stats(binary: np.ndarray):
    # binary shape: nodes x bins
    variances = binary.var(axis=1)
    denom = float(np.mean(variances))
    if denom <= 0:
        return np.nan, np.nan
    rate = binary.mean(axis=0)
    variance_ratio = float(rate.var() / denom)
    return variance_ratio, float(np.sqrt(max(variance_ratio, 0.0)))


def population_entropy(counts: np.ndarray):
    if counts.size == 0:
        return np.nan
    values, freq = np.unique(counts.astype(int), return_counts=True)
    prob = freq.astype(float) / float(freq.sum())
    return float(-np.sum(prob * np.log2(prob + 1e-15)))


def avalanche_stats(counts: np.ndarray):
    active = counts > 0
    sizes = []
    durations = []
    start = None
    total = 0
    for idx, is_active in enumerate(active.tolist()):
        if is_active and start is None:
            start = idx
            total = int(counts[idx])
        elif is_active:
            total += int(counts[idx])
        elif start is not None:
            sizes.append(total)
            durations.append(idx - start)
            start = None
            total = 0
    if start is not None:
        sizes.append(total)
        durations.append(len(active) - start)
    return {
        "cascade_count": len(sizes),
        "cascade_size_mean": float(np.mean(sizes)) if sizes else np.nan,
        "cascade_size_cv": cv(sizes),
        "cascade_duration_mean": float(np.mean(durations)) if durations else np.nan,
        "cascade_duration_cv": cv(durations),
    }


def compute_binned_metrics(spike_times_by_path, simplices, t_stat):
    rows = []
    for bin_size in BIN_SIZES:
        n_bins = max(1, int(np.ceil(t_stat / bin_size)))
        for path_idx, spike_times_by_node in enumerate(spike_times_by_path):
            binary = np.zeros((len(spike_times_by_node), n_bins), dtype=np.uint8)
            for node, times in enumerate(spike_times_by_node):
                if not times:
                    continue
                bins = np.asarray(times, dtype=float) // bin_size
                bins = bins[(bins >= 0) & (bins < n_bins)].astype(int)
                binary[node, np.unique(bins)] = 1
            counts = binary.sum(axis=0)
            all_corr, simplex_corr = corr_summary(binary, simplices)
            synchrony_variance_ratio, synchrony = synchrony_stats(binary)
            aval = avalanche_stats(counts)
            rows.append(
                {
                    "path_idx": path_idx,
                    "bin_size": bin_size,
                    "stcc_mean": all_corr,
                    "simplex_stcc_mean": simplex_corr,
                    "synchrony_variance_ratio": synchrony_variance_ratio,
                    "synchrony_index": synchrony,
                    "population_count_mean": float(np.mean(counts)),
                    "population_count_var": float(np.var(counts, ddof=1)) if counts.size > 1 else 0.0,
                    "population_fano": float(np.var(counts, ddof=1) / np.mean(counts)) if np.mean(counts) > 0 and counts.size > 1 else np.nan,
                    "population_entropy_bits": population_entropy(counts),
                    **aval,
                }
            )
    return rows


def simulate_mechanism_batch(
    params: base.PairedSimplicialParams,
    network: base.SimplicialNetwork,
    alpha: float,
    noise_intensity_dv: float,
    higher_order_enabled: bool,
    init_seed: int,
    noise_seed: int,
):
    n_steps = int(params.t_total / params.dt)
    n_paths = params.n_trials_per_network
    n = params.n_nodes
    t_stat = params.t_total - params.t_warmup

    scale = base.noise_scale_from_voltage_noise_intensity(noise_intensity_dv, alpha, params.capacitance)
    init_rng = np.random.default_rng(init_seed)
    noise_rng = np.random.default_rng(noise_seed)

    v = -60.0 + 2.0 * init_rng.standard_normal((n_paths, n))
    w = base.w_inf(v, params)
    prev_v = v.copy()
    operator = network.pairwise_operator + (params.sigma_2 * network.l2 if higher_order_enabled else 0.0)

    spike_times = [[[] for _ in range(n)] for _ in range(n_paths)]
    threshold_state = {vlow: v > vlow for vlow in V_LOW_VALUES}
    last_end = {vlow: np.full((n_paths, n), np.nan) for vlow in V_LOW_VALUES}
    last_excursion = {vlow: np.full((n_paths, n), np.nan) for vlow in V_LOW_VALUES}
    active_start = {vlow: np.full((n_paths, n), np.nan) for vlow in V_LOW_VALUES}
    ta_values = {vlow: [[] for _ in range(n_paths)] for vlow in V_LOW_VALUES}
    te_values = {vlow: [[] for _ in range(n_paths)] for vlow in V_LOW_VALUES}
    paired_ta_values = {vlow: [[] for _ in range(n_paths)] for vlow in V_LOW_VALUES}
    paired_te_values = {vlow: [[] for _ in range(n_paths)] for vlow in V_LOW_VALUES}
    cycle_values = {vlow: [[] for _ in range(n_paths)] for vlow in V_LOW_VALUES}

    for step in range(n_steps):
        coupling = v @ operator.T
        i_ca = params.g_ca * base.m_inf(v, params) * (v - params.v_ca)
        i_k = params.g_k * w * (v - params.v_k)
        i_l = params.g_l * (v - params.v_l)
        dv = (params.current - i_ca - i_k - i_l + coupling) / params.capacitance
        dw = params.phi * (base.w_inf(v, params) - w) / np.maximum(base.tau_w(v, params), 1e-6)
        jump = base.levy_increment(scale, alpha, params.beta, params.dt, (n_paths, n), noise_rng) / params.capacitance

        v = np.clip(v + params.dt * dv + jump, -200.0, 200.0)
        w = np.clip(w + params.dt * dw, -0.5, 1.5)

        current_time = (step + 1) * params.dt
        if current_time > params.t_warmup:
            t_post = current_time - params.t_warmup
            spike_crossing = (prev_v < params.spike_threshold) & (v >= params.spike_threshold)
            pidx, nidx = np.where(spike_crossing)
            for path, node in zip(pidx.tolist(), nidx.tolist()):
                spike_times[path][node].append(t_post)

            for vlow in V_LOW_VALUES:
                old_state = threshold_state[vlow]
                new_state = v > vlow
                onset = (~old_state) & new_state
                offset = old_state & (~new_state)

                p_on, n_on = np.where(onset)
                for path, node in zip(p_on.tolist(), n_on.tolist()):
                    if np.isfinite(last_end[vlow][path, node]):
                        ta = t_post - last_end[vlow][path, node]
                        if ta > 0:
                            ta_values[vlow][path].append(ta)
                            te_prev = last_excursion[vlow][path, node]
                            if np.isfinite(te_prev) and te_prev > 0:
                                paired_ta_values[vlow][path].append(ta)
                                paired_te_values[vlow][path].append(te_prev)
                                cycle_values[vlow][path].append(ta + te_prev)
                    active_start[vlow][path, node] = t_post

                p_off, n_off = np.where(offset)
                for path, node in zip(p_off.tolist(), n_off.tolist()):
                    start = active_start[vlow][path, node]
                    if np.isfinite(start):
                        te = t_post - start
                        if te > 0:
                            te_values[vlow][path].append(te)
                            last_excursion[vlow][path, node] = te
                    last_end[vlow][path, node] = t_post
                    active_start[vlow][path, node] = np.nan

                threshold_state[vlow] = new_state
        else:
            for vlow in V_LOW_VALUES:
                threshold_state[vlow] = v > vlow

        prev_v = v.copy()

    rows = []
    for path_idx in range(n_paths):
        spike_counts = [len(times) for times in spike_times[path_idx]]
        isi_all = []
        for times in spike_times[path_idx]:
            if len(times) >= 2:
                isi_all.extend(np.diff(np.asarray(times, dtype=float)).tolist())
        for vlow in V_LOW_VALUES:
            ta = ta_values[vlow][path_idx]
            te = te_values[vlow][path_idx]
            paired_ta = paired_ta_values[vlow][path_idx]
            paired_te = paired_te_values[vlow][path_idx]
            cycle = cycle_values[vlow][path_idx]
            rows.append(
                {
                    "metric_group": "activation_excursion",
                    "path_idx": path_idx,
                    "v_low": vlow,
                    "spike_count_total": int(np.sum(spike_counts)),
                    "isi_cv": cv(isi_all),
                    "activation_mean": summarize(ta)[0],
                    "activation_std": summarize(ta)[1],
                    "activation_cv": cv(ta),
                    "excursion_mean": summarize(te)[0],
                    "excursion_std": summarize(te)[1],
                    "excursion_cv": cv(te),
                    "ta_te_corr": corr(paired_ta, paired_te),
                    "activation_variance_fraction": variance_fraction(paired_ta, cycle),
                    "excursion_variance_fraction": variance_fraction(paired_te, cycle),
                    "cycle_mean": summarize(cycle)[0],
                    "cycle_std": summarize(cycle)[1],
                    "cycle_cv": cv(cycle),
                    "activation_count": len(ta),
                    "excursion_count": len(te),
                    "paired_cycle_count": len(cycle),
                }
            )

    binned_rows = compute_binned_metrics(spike_times, network.simplices2, t_stat)
    return rows, binned_rows


def make_params(args):
    return base.PairedSimplicialParams(
        n_nodes=args.n_nodes,
        n_networks=args.n_networks,
        n_trials_per_network=args.n_trials,
        p_edge=0.15,
        p_fill_simplex=0.5,
        sigma_1=0.002,
        sigma_2=0.12,
        current=88.0,
        dt=0.05,
        t_total=args.t_total,
        t_warmup=args.t_warmup,
        alphas=(1.5, 2.0),
        beta=0.0,
        noise_scales=DV_VALUES,
        base_seed=args.seed,
    )


def run_one_network(params, network_idx: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    network_rng = np.random.default_rng(params.base_seed + base.stable_seed("mechanism_network", network_idx))
    network = base.build_simplicial_network(params, network_rng)

    activation_rows = []
    binned_rows = []
    for alpha in params.alphas:
        for dv_idx, dv in enumerate(params.noise_scales):
            for trial_idx in range(params.n_trials_per_network):
                init_seed = params.base_seed + base.stable_seed("mechanism_init", network_idx, trial_idx)
                noise_seed = params.base_seed + base.stable_seed("mechanism_noise", alpha, dv_idx, network_idx, trial_idx)
                for label, higher in (("control", False), ("higher_order", True)):
                    act, binned = simulate_mechanism_batch(
                        params,
                        network,
                        alpha=alpha,
                        noise_intensity_dv=dv,
                        higher_order_enabled=higher,
                        init_seed=init_seed,
                        noise_seed=noise_seed,
                    )
                    common = {
                        "alpha": alpha,
                        "label": label,
                        "network_idx": network_idx,
                        "trial_idx": trial_idx,
                        "noise_intensity_dv": dv,
                        "noise_scale": base.noise_scale_from_voltage_noise_intensity(dv, alpha, params.capacitance),
                        "edge_count": network.edge_count,
                        "clique_count": network.clique_count,
                        "simplex_count": network.simplex_count,
                    }
                    activation_rows.extend([{**common, **row} for row in act])
                    binned_rows.extend([{**common, **row} for row in binned])

    network_rows = [
        {
            "network_idx": network_idx,
            "edge_count": network.edge_count,
            "clique_count": network.clique_count,
            "simplex_count": network.simplex_count,
        }
    ]
    (output_dir / "parameters.json").write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")
    (output_dir / "network_summary.json").write_text(json.dumps(network_rows, indent=2), encoding="utf-8")
    write_csv(output_dir / "network_summary.csv", network_rows)
    write_csv(output_dir / "activation_excursion_metrics.csv", activation_rows)
    write_csv(output_dir / "binned_synchrony_metrics.csv", binned_rows)
    return len(activation_rows), len(binned_rows)


def main():
    parser = argparse.ArgumentParser(description="Mechanistic supplemental array run: activation/excursion, synchrony, cascades, information metrics.")
    parser.add_argument("task_id", type=int)
    parser.add_argument("n_tasks", type=int)
    parser.add_argument("--output-root", default="paper_suite_mechanism_supplement_Dv")
    parser.add_argument("--n-nodes", type=int, default=100)
    parser.add_argument("--n-networks", type=int, default=30)
    parser.add_argument("--n-trials", type=int, default=4)
    parser.add_argument("--t-total", type=float, default=20000.0)
    parser.add_argument("--t-warmup", type=float, default=5000.0)
    parser.add_argument("--seed", type=int, default=20260517)
    args = parser.parse_args()

    if args.n_tasks != args.n_networks:
        raise ValueError(f"Expected n_tasks={args.n_networks}, got {args.n_tasks}.")
    if args.task_id < 0 or args.task_id >= args.n_networks:
        raise ValueError(f"task_id must be in [0, {args.n_networks - 1}], got {args.task_id}.")

    params = make_params(args)
    root = Path(args.output_root)
    root = root if root.is_absolute() else Path(__file__).resolve().parent / root
    chunk_root = root / "chunks" / f"network_{args.task_id:03d}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "chunking.json").write_text(
        json.dumps(
            {
                "n_networks": args.n_networks,
                "total_array_tasks": args.n_networks,
                "n_trials_per_network": args.n_trials,
                "paths_per_alpha_label_Dv": args.n_networks * args.n_trials * args.n_trials,
                "Dv_values": DV_VALUES,
                "v_low_values": V_LOW_VALUES,
                "bin_sizes": BIN_SIZES,
                "experiments": [
                    "activation_excursion_decomposition",
                    "network_synchronization",
                    "cascade_statistics",
                    "population_information_metrics",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Task {args.task_id}/{args.n_tasks - 1}: mechanism supplement network={args.task_id}")
    n_act, n_bin = run_one_network(params, args.task_id, chunk_root)
    print(f"Finished task {args.task_id}: activation_rows={n_act}, binned_rows={n_bin}")


if __name__ == "__main__":
    main()
