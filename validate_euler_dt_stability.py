from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import morris_lecar_paired_alpha_compare_simplicial as base


def deterministic_network_trace(p, network, dt: float, higher_order: bool):
    q = base.PairedSimplicialParams(**{**p.__dict__, "dt": dt})
    rng = np.random.default_rng(20260507)
    v = -60.0 + 2.0 * rng.standard_normal(q.n_nodes)
    w = base.w_inf(v, q)
    operator = network.pairwise_operator + (q.sigma_2 * network.l2 if higher_order else 0.0)

    max_v = float(np.max(v))
    min_v = float(np.min(v))
    max_abs_dvdt = 0.0
    max_step = 0.0
    spike_count = 0
    prev_v = v.copy()
    tail_means = []
    n_steps = int(q.t_total / dt)
    warmup_step = int(q.t_warmup / dt)

    for step in range(n_steps):
        i_ca = q.g_ca * base.m_inf(v, q) * (v - q.v_ca)
        i_k = q.g_k * w * (v - q.v_k)
        i_l = q.g_l * (v - q.v_l)
        coupling = v @ operator.T
        dvdt = (q.current - i_ca - i_k - i_l + coupling) / q.capacitance
        dwdt = q.phi * (base.w_inf(v, q) - w) / np.maximum(base.tau_w(v, q), 1e-6)
        step_v = dt * dvdt

        v = np.clip(v + step_v, -200.0, 200.0)
        w = np.clip(w + dt * dwdt, -0.5, 1.5)

        crossed = (prev_v < q.spike_threshold) & (v >= q.spike_threshold)
        if step >= warmup_step:
            spike_count += int(np.sum(crossed))
            tail_means.append(float(np.mean(v)))

        prev_v = v.copy()
        max_v = max(max_v, float(np.max(v)))
        min_v = min(min_v, float(np.min(v)))
        max_abs_dvdt = max(max_abs_dvdt, float(np.max(np.abs(dvdt))))
        max_step = max(max_step, float(np.max(np.abs(step_v))))

    return {
        "dt": dt,
        "higher_order": higher_order,
        "final_mean_v": float(np.mean(v)),
        "final_std_v": float(np.std(v)),
        "tail_mean_v": float(np.mean(tail_means)),
        "tail_std_mean_v": float(np.std(tail_means)),
        "max_v": max_v,
        "min_v": min_v,
        "max_abs_dvdt": max_abs_dvdt,
        "max_euler_voltage_step": max_step,
        "spike_count_total": spike_count,
    }


def high_voltage_relaxation(p, dt: float, v0: float, w0: float):
    q = base.PairedSimplicialParams(**{**p.__dict__, "dt": dt, "n_nodes": 1})
    v = np.array([v0], dtype=float)
    w = np.array([w0], dtype=float)
    max_v = float(v0)
    min_v = float(v0)
    max_abs_dvdt = 0.0
    max_step = 0.0
    blow_up = False

    for _ in range(int(200.0 / dt)):
        i_ca = q.g_ca * base.m_inf(v, q) * (v - q.v_ca)
        i_k = q.g_k * w * (v - q.v_k)
        i_l = q.g_l * (v - q.v_l)
        dvdt = (q.current - i_ca - i_k - i_l) / q.capacitance
        dwdt = q.phi * (base.w_inf(v, q) - w) / np.maximum(base.tau_w(v, q), 1e-6)
        step_v = dt * dvdt
        v = v + step_v
        w = w + dt * dwdt
        if not np.isfinite(v).all() or abs(float(v[0])) > 1e6:
            blow_up = True
            break
        max_v = max(max_v, float(v[0]))
        min_v = min(min_v, float(v[0]))
        max_abs_dvdt = max(max_abs_dvdt, float(np.max(np.abs(dvdt))))
        max_step = max(max_step, float(np.max(np.abs(step_v))))

    return {
        "v0": v0,
        "w0": w0,
        "dt": dt,
        "final_v": float(v[0]),
        "final_w": float(w[0]),
        "max_v": max_v,
        "min_v": min_v,
        "max_abs_dvdt": max_abs_dvdt,
        "max_step": max_step,
        "blow_up": blow_up,
    }


def main():
    out = Path("numerical_dt_validation")
    out.mkdir(exist_ok=True)

    p = base.PairedSimplicialParams(
        n_nodes=100,
        n_networks=1,
        n_trials_per_network=1,
        p_edge=0.15,
        p_fill_simplex=0.5,
        sigma_1=0.002,
        sigma_2=0.12,
        current=88.0,
        t_total=3000.0,
        t_warmup=500.0,
        noise_scales=(0.0,),
    )
    network = base.build_simplicial_network(p, np.random.default_rng(20299501))

    dt_values = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    deterministic = [
        deterministic_network_trace(p, network, dt, higher_order)
        for dt in dt_values
        for higher_order in (False, True)
    ]

    high_voltage = [
        high_voltage_relaxation(p, dt, v0, w0)
        for dt in [0.1, 0.05, 0.025, 0.0125]
        for v0 in [80.0, 100.0, 120.0, 150.0]
        for w0 in [0.0, 0.5, 1.0]
    ]

    (out / "dt_validation_summary.json").write_text(
        json.dumps({"deterministic_no_noise": deterministic}, indent=2),
        encoding="utf-8",
    )
    (out / "high_voltage_relaxation.json").write_text(
        json.dumps(high_voltage, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out / 'dt_validation_summary.json'}")
    print(f"Wrote {out / 'high_voltage_relaxation.json'}")


if __name__ == "__main__":
    main()
