from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import morris_lecar_paired_alpha_compare_simplicial as base


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None, 0
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return float(np.mean(finite)), std, int(finite.size)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_params(dt: float):
    return base.PairedSimplicialParams(
        n_nodes=100,
        n_networks=3,
        n_trials_per_network=4,
        p_edge=0.15,
        p_fill_simplex=0.5,
        sigma_1=0.002,
        sigma_2=0.12,
        current=88.0,
        dt=dt,
        t_total=2500.0,
        t_warmup=500.0,
        alphas=(1.5, 2.0),
        noise_scales=(0.25, 0.75, 1.00),
        base_seed=20260507,
    )


def run_for_dt(dt: float):
    p = make_params(dt)
    rows = []
    network_rows = []
    dv_by_alpha = {
        1.5: (0.25, 0.75),
        2.0: (0.25, 1.00),
    }

    for network_idx in range(p.n_networks):
        network_seed = p.base_seed + base.stable_seed("dt_network", network_idx)
        network_rng = np.random.default_rng(network_seed)
        network = base.build_simplicial_network(p, network_rng)
        network_rows.append(
            {
                "dt": dt,
                "network_idx": network_idx,
                "edge_count": network.edge_count,
                "clique_count": network.clique_count,
                "simplex_count": network.simplex_count,
            }
        )
        print(
            f"dt={dt:g} network {network_idx + 1}/{p.n_networks}: "
            f"edges={network.edge_count}, simplices={network.simplex_count}",
            flush=True,
        )

        for alpha, dv_values in dv_by_alpha.items():
            for noise_intensity_dv in dv_values:
                noise_idx = list(p.noise_scales).index(noise_intensity_dv)
                scale = base.noise_scale_from_voltage_noise_intensity(noise_intensity_dv, alpha, p.capacitance)
                for trial_idx in range(p.n_trials_per_network):
                    init_seed = p.base_seed + base.stable_seed("dt_init", network_idx, trial_idx)
                    noise_seed = p.base_seed + base.stable_seed("dt_noise", dt, alpha, noise_idx, network_idx, trial_idx)
                    for label, higher in (("control", False), ("higher_order", True)):
                        metrics = base.simulate_batch(
                            p,
                            network,
                            alpha=alpha,
                            noise_scale=scale,
                            higher_order_enabled=higher,
                            init_seed=init_seed,
                            noise_seed=noise_seed,
                        )
                        for path_idx in range(p.n_trials_per_network):
                            rows.append(
                                {
                                    "dt": dt,
                                    "alpha": alpha,
                                    "noise_scale": scale,
                                    "noise_intensity_dv": noise_intensity_dv,
                                    "label": label,
                                    "network_idx": network_idx,
                                    "trial_idx": trial_idx,
                                    "path_idx": path_idx,
                                    "isi_cv": float(metrics["isi_cv"][path_idx]),
                                    "spike_count_total": float(metrics["spike_count_total"][path_idx]),
                                    "mean_activity_std": float(metrics["mean_activity_std"][path_idx]),
                                    "edge_count": float(metrics["edge_count"][path_idx]),
                                    "clique_count": float(metrics["clique_count"][path_idx]),
                                    "simplex_count": float(metrics["simplex_count"][path_idx]),
                                }
                            )

    return p, rows, network_rows


def aggregate(rows):
    out = []
    keys = sorted({(r["dt"], r["alpha"], r["noise_intensity_dv"], r["label"]) for r in rows})
    for dt, alpha, noise_intensity_dv, label in keys:
        subset = [
            r
            for r in rows
            if r["dt"] == dt
            and r["alpha"] == alpha
            and abs(r["noise_intensity_dv"] - noise_intensity_dv) < 1e-12
            and r["label"] == label
        ]
        cv_mean, cv_std, n_cv = mean_std([r["isi_cv"] for r in subset])
        sp_mean, sp_std, _ = mean_std([r["spike_count_total"] for r in subset])
        scale = base.noise_scale_from_voltage_noise_intensity(noise_intensity_dv, alpha, 20.0)
        out.append(
            {
                "dt": dt,
                "alpha": alpha,
                "noise_scale": scale,
                "noise_intensity_dv": noise_intensity_dv,
                "label": label,
                "total_paths": len(subset),
                "finite_cv_paths": n_cv,
                "isi_cv_mean": cv_mean,
                "isi_cv_std": cv_std,
                "spike_count_total_mean": sp_mean,
                "spike_count_total_std": sp_std,
            }
        )
    return out


def summarize_best(aggregated):
    summary = []
    for dt in sorted({r["dt"] for r in aggregated}):
        for alpha in sorted({r["alpha"] for r in aggregated}):
            row = {"dt": dt, "alpha": alpha}
            best = {}
            for label in ("control", "higher_order"):
                candidates = [
                    r
                    for r in aggregated
                    if r["dt"] == dt
                    and r["alpha"] == alpha
                    and r["label"] == label
                    and r["isi_cv_mean"] is not None
                    and r["spike_count_total_mean"] is not None
                    and r["spike_count_total_mean"] >= 100.0
                ]
                winner = min(candidates, key=lambda r: r["isi_cv_mean"]) if candidates else None
                best[label] = winner
                row[f"{label}_best_noise_scale"] = None if winner is None else winner["noise_scale"]
                row[f"{label}_best_noise_intensity_dv"] = None if winner is None else winner["noise_intensity_dv"]
                row[f"{label}_best_cv"] = None if winner is None else winner["isi_cv_mean"]
                row[f"{label}_best_spike_mean"] = None if winner is None else winner["spike_count_total_mean"]
            if best.get("control") and best.get("higher_order"):
                improvement = best["control"]["isi_cv_mean"] - best["higher_order"]["isi_cv_mean"]
                row["delta_cv"] = improvement
                row["relative_reduction"] = improvement / best["control"]["isi_cv_mean"]
            else:
                row["delta_cv"] = None
                row["relative_reduction"] = None
            summary.append(row)
    return summary


def main():
    out = Path("numerical_dt_validation") / "random_dt_convergence"
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_network_rows = []
    params = {}
    for dt in (0.05, 0.025):
        p, rows, network_rows = run_for_dt(dt)
        params[str(dt)] = asdict(p)
        all_rows.extend(rows)
        all_network_rows.extend(network_rows)

    aggregated = aggregate(all_rows)
    summary = summarize_best(aggregated)

    write_csv(out / "all_realizations.csv", all_rows)
    write_csv(out / "network_summary.csv", all_network_rows)
    write_csv(out / "aggregated_metrics.csv", aggregated)
    write_csv(out / "best_summary.csv", summary)
    (out / "parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    (out / "all_realizations.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    (out / "aggregated_metrics.json").write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    (out / "best_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBest CV summary with mean spike count >= 100")
    for row in summary:
        print(row)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
