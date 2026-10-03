"""Independent simplex/edge implementations; no changes to production inputs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np
from numba import njit
import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import morris_lecar_paired_alpha_compare_simplicial as base

MAIN = ROOT / "paper_suite_true_simplicial_array_prl_N100_Dv_refined/candidate39_main"
OUT = HERE / "analysis"
CASES = ((1.5, .16), (2.0, .30))
NETWORK_INDEX = 0
NODE_INDEX = 0
INIT_SEED = 2026092901
NOISE_SEEDS = (2026092915, 2026092920)
BLOCK = 4096
INITIAL_VOLTAGE_TOLERANCE = 1e-6


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edge_projection(n, edges, triangles):
    # Build multiplicities from triangle lists, independently of net.a2/net.l2.
    counts = np.zeros((n, n), dtype=np.int64)
    incident = np.zeros(n, dtype=np.int64)
    for i, j, k in triangles:
        for u, v in ((i, j), (i, k), (j, k)):
            counts[u, v] += 1
            counts[v, u] += 1
        incident[[i, j, k]] += 1
    assert np.array_equal(counts.sum(axis=1), 2 * incident)
    mult = np.array([counts[i, j] for i, j in edges], dtype=np.int64)
    return counts, incident, mult


@njit(cache=False)
def coupling(v, edges, triangles, multiplicity, sigma1, sigma2, mode):
    out = np.zeros(v.size)
    for edge in range(edges.shape[0]):
        i, j = edges[edge]
        weight = sigma1
        if mode == 1:
            weight += .5 * sigma2 * multiplicity[edge]
        delta = weight * (v[j] - v[i])
        out[i] += delta
        out[j] -= delta
    if mode == 0:
        for face in range(triangles.shape[0]):
            i, j, k = triangles[face]
            out[i] += sigma2 * (.5 * (v[j] + v[k]) - v[i])
            out[j] += sigma2 * (.5 * (v[i] + v[k]) - v[j])
            out[k] += sigma2 * (.5 * (v[i] + v[j]) - v[k])
    return out


def algebra_tests():
    vi, vj, vk = sp.symbols("V_i V_j V_k")
    exact = sp.simplify((vj + vk) / 2 - vi - ((vj - vi) / 2 + (vk - vi) / 2))
    typo = sp.simplify((vj - vk) / 2 - vi - ((vj - vi) / 2 + (vk - vi) / 2))
    assert exact == 0 and typo == -vk
    edges = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int64)
    triangles = np.array([(0, 1, 2), (0, 1, 3)], dtype=np.int64)
    counts, incident, mult = edge_projection(4, edges, triangles)
    assert counts[0, 1] == 2 and counts[2, 3] == 0
    state = np.array([-60., 10., -30., 5.])
    a = coupling(state, edges, triangles, mult, .002, .12, 0)
    b = coupling(state, edges, triangles, mult, .002, .12, 1)
    assert np.max(np.abs(a - b)) < 1e-12
    uniform = coupling(state, edges, triangles, np.ones(len(edges), dtype=np.int64), .002, .12, 1)
    assert np.max(np.abs(a - uniform)) > 1
    dump_json(OUT / "algebra_tests.json", {
        "correct_identity_residual": str(exact), "reviewer_minus_sign_residual": str(typo),
        "motif_edges": edges.tolist(), "motif_filled_triangles": triangles.tolist(),
        "motif_edge_multiplicities": mult.tolist(), "motif_voltage": state.tolist(),
        "simplex_coupling": a.tolist(), "projected_coupling": b.tolist(),
        "once_per_edge_uniform_coupling": uniform.tolist(),
        "motif_max_projected_error": float(np.max(np.abs(a-b))),
    })


def audit_networks(p):
    rows, inputs = [], []
    selected = None
    for idx in range(p.n_networks):
        folder = MAIN / f"chunks/network_{idx:03d}"
        for filename in ("parameters.json", "network_summary.json"):
            path = folder / filename
            inputs.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})
        assert json.loads((folder / "parameters.json").read_text()) == vars(p)
        seed = p.base_seed + base.stable_seed("paper_network", idx)
        net = base.build_simplicial_network(p, np.random.default_rng(seed))
        expected = json.loads((folder / "network_summary.json").read_text())[0]
        for key in ("edge_count", "clique_count", "simplex_count"):
            assert getattr(net, key) == expected[key]
        edges = np.column_stack(np.where(np.triu(net.a1, 1) > 0)).astype(np.int64)
        triangles = np.asarray(net.simplices2, dtype=np.int64).reshape(-1, 3)
        counts, incident, mult = edge_projection(p.n_nodes, edges, triangles)
        assert np.array_equal(counts, net.a2)
        assert np.array_equal(incident, -np.diag(net.l2))
        skeleton = net.a1.astype(np.int64)
        # For the actual strengths, multiply the operator by 500 to use integers.
        assert p.sigma_1 == .002 and p.sigma_2 == .12
        integer_a = skeleton - np.diag(skeleton.sum(axis=1)) + 30 * counts - 60 * np.diag(incident)
        integer_weights = skeleton + 30 * counts
        integer_b = integer_weights - np.diag(integer_weights.sum(axis=1))
        assert np.array_equal(integer_a, integer_b)
        operator = net.pairwise_operator + p.sigma_2 * net.l2
        weights = p.sigma_1 * skeleton + .5 * p.sigma_2 * counts
        projected = weights - np.diag(weights.sum(axis=1))
        uniform = net.pairwise_operator + .5 * p.sigma_2 * (skeleton - np.diag(skeleton.sum(axis=1)))
        matrix_error = float(np.max(np.abs(operator - projected)))
        assert matrix_error < 1e-12
        rng = np.random.default_rng(2026092900 + idx)
        states = np.vstack((rng.uniform(-200, 200, (256, p.n_nodes)), np.full((1, p.n_nodes), -60.)))
        direct_edge_error = production_error = 0.
        for v in states:
            a = coupling(v, edges, triangles, mult, p.sigma_1, p.sigma_2, 0)
            b = coupling(v, edges, triangles, mult, p.sigma_1, p.sigma_2, 1)
            direct_edge_error = max(direct_edge_error, float(np.max(np.abs(a-b))))
            production_error = max(production_error, float(np.max(np.abs(a-operator@v))))
        assert direct_edge_error < 1e-9 and production_error < 1e-9
        rows.append({"network_idx": idx, "network_seed": seed, "edge_count": len(edges),
                     "simplex_count": len(triangles), "integer_operator_identical": True,
                     "matrix_max_abs_error": matrix_error,
                     "force_direct_vs_edges_max_abs_error": direct_edge_error,
                     "force_direct_vs_production_max_abs_error": production_error,
                     "uniform_once_per_edge_operator_max_abs_difference": float(np.max(np.abs(operator-uniform))),
                     "zero_simplex_edges": int(np.count_nonzero(mult == 0)),
                     "multiply_covered_edges": int(np.count_nonzero(mult > 1)),
                     "max_edge_multiplicity": int(mult.max(initial=0))})
        if idx == NETWORK_INDEX:
            selected = edges, triangles, mult, seed
    csv_write(OUT / "network_operator_audit.csv", rows)
    return rows, inputs, selected


@njit(cache=False)
def advance(v, w, jumps, start, dt, warmup, threshold, edges, triangles, mult, q, node):
    sigma1, sigma2, cap, gca, gk, gl, vca, vk, vl, v1, v2, v3, v4, phi, current = q
    steps, n = jumps.shape
    trace = np.empty((2, steps))
    crossings = np.zeros((2, steps, n), dtype=np.bool_)
    max_v = max_w = 0.
    max_v_step = start
    clip_counts = np.zeros((2, 2), dtype=np.int64)
    for s in range(steps):
        t = (start + s + 1) * dt
        for mode in range(2):
            c = coupling(v[mode], edges, triangles, mult, sigma1, sigma2, mode)
            for i in range(n):
                old_v, old_w = v[mode, i], w[mode, i]
                mi = .5 * (1 + np.tanh((old_v-v1)/v2))
                wi = .5 * (1 + np.tanh((old_v-v3)/v4))
                arg = min(40., max(-40., (old_v-v3)/(2*v4)))
                tau = 1 / np.cosh(arg)
                ca = gca * mi * (old_v-vca)
                potassium = gk * old_w * (old_v-vk)
                leak = gl * (old_v-vl)
                dv = (current-ca-potassium-leak+c[i])/cap
                dw = phi * (wi-old_w)/max(tau, 1e-6)
                new_v = old_v + dt*dv + jumps[s, i]
                new_w = old_w + dt*dw
                clip_counts[mode, 0] += int(new_v < -200. or new_v > 200.)
                clip_counts[mode, 1] += int(new_w < -.5 or new_w > 1.5)
                new_v = min(200., max(-200., new_v))
                new_w = min(1.5, max(-.5, new_w))
                v[mode, i], w[mode, i] = new_v, new_w
                if t > warmup:
                    crossings[mode, s, i] = old_v < threshold and new_v >= threshold
            trace[mode, s] = v[mode, node]
        for i in range(n):
            diff_v = abs(v[0, i]-v[1, i])
            if diff_v > max_v:
                max_v, max_v_step = diff_v, start+s+1
            max_w = max(max_w, abs(w[0, i]-w[1, i]))
    return trace, crossings, max_v, max_w, max_v_step, clip_counts


def cv_of(intervals):
    if len(intervals) < 3:
        return None
    return float(np.std(intervals, ddof=0)/(np.mean(intervals)+1e-12))


def integrator_conformance(p, selected):
    edges, triangles, mult, _ = selected
    q = np.array([p.sigma_1, p.sigma_2, p.capacitance, p.g_ca, p.g_k, p.g_l,
                  p.v_ca, p.v_k, p.v_l, p.v1, p.v2, p.v3, p.v4, p.phi, p.current])
    net = base.build_simplicial_network(p, np.random.default_rng(selected[3]))
    operator = net.pairwise_operator + p.sigma_2*net.l2
    rng = np.random.default_rng(2026092999)
    max_v = max_w = 0.
    for _ in range(16):
        v0 = rng.uniform(-100., 50., p.n_nodes)
        w0 = rng.uniform(-.1, .9, p.n_nodes)
        jump = rng.normal(0., .3, p.n_nodes)
        ca = p.g_ca*base.m_inf(v0, p)*(v0-p.v_ca)
        k = p.g_k*w0*(v0-p.v_k)
        leak = p.g_l*(v0-p.v_l)
        ref_v = np.clip(v0+p.dt*(p.current-ca-k-leak+v0@operator.T)/p.capacitance+jump, -200., 200.)
        ref_w = np.clip(w0+p.dt*p.phi*(base.w_inf(v0, p)-w0)/np.maximum(base.tau_w(v0, p), 1e-6), -.5, 1.5)
        v, w = np.stack((v0, v0)), np.stack((w0, w0))
        advance(v, w, jump[None, :], 0, p.dt, p.t_warmup, p.spike_threshold,
                edges, triangles, mult, q, NODE_INDEX)
        max_v = max(max_v, float(np.max(np.abs(v-ref_v))))
        max_w = max(max_w, float(np.max(np.abs(w-ref_w))))
    assert max_v < 1e-10 and max_w < 1e-12
    return {"states_tested": 16, "voltage_max_abs_error": max_v, "recovery_max_abs_error": max_w}


def run_case(p, selected, case_idx):
    alpha, dv = CASES[case_idx]
    edges, triangles, mult, net_seed = selected
    nsteps = int(p.t_total / p.dt)
    rng = np.random.default_rng(INIT_SEED)
    v0 = -60 + 2*rng.standard_normal(p.n_nodes)
    w0 = base.w_inf(v0, p)
    v, w = np.stack((v0, v0)), np.stack((w0, w0))
    nrng = np.random.default_rng(NOISE_SEEDS[case_idx])
    noise_hash = hashlib.sha256()
    q = np.array([p.sigma_1, p.sigma_2, p.capacitance, p.g_ca, p.g_k, p.g_l,
                  p.v_ca, p.v_k, p.v_l, p.v1, p.v2, p.v3, p.v4, p.phi, p.current])
    trace = np.empty((2, nsteps+1))
    trace[:, 0] = v[:, NODE_INDEX]
    events = [[[] for _ in range(p.n_nodes)] for _ in range(2)]
    max_v = max_w = 0.
    max_v_step = 0
    clips = np.zeros((2, 2), dtype=np.int64)
    scale = base.noise_scale_from_voltage_noise_intensity(dv, alpha, p.capacitance)
    started = time.perf_counter()
    for start in range(0, nsteps, BLOCK):
        length = min(BLOCK, nsteps-start)
        jumps = base.levy_increment(scale, alpha, p.beta, p.dt, (length, p.n_nodes), nrng)/p.capacitance
        noise_hash.update(jumps.tobytes())
        result = advance(v, w, jumps, start, p.dt, p.t_warmup, p.spike_threshold,
                         edges, triangles, mult, q, NODE_INDEX)
        tr, crosses, err_v, err_w, err_step, block_clips = result
        trace[:, start+1:start+length+1] = tr
        if err_v > max_v:
            max_v, max_v_step = err_v, int(err_step)
        max_w = max(max_w, err_w)
        clips += block_clips
        for mode in range(2):
            for i in range(p.n_nodes):
                indices = np.flatnonzero(crosses[mode, :, i]) + start + 1
                events[mode][i].extend(indices.tolist())
        if start == 0 or start // BLOCK % 20 == 0:
            print(f"alpha={alpha:g}, Dv={dv:g}: t={(start+length)*p.dt:g}, max|dV|={max_v:.3g}", flush=True)
    assert np.isfinite(v).all() and np.isfinite(w).all() and np.isfinite(trace).all()
    name = f"alpha{str(alpha).replace('.', '')}"
    np.savez_compressed(OUT / f"{name}_trajectory.npz", t=np.arange(nsteps+1)*p.dt,
                        voltage_simplex=trace[0], voltage_projected=trace[1])
    node_rows, spike_rows, isi_rows = [], [], []
    spike_identical = True
    for i in range(p.n_nodes):
        ea, eb = np.asarray(events[0][i]), np.asarray(events[1][i])
        identical = bool(np.array_equal(ea, eb))
        spike_identical &= identical
        ta, tb = ea*p.dt, eb*p.dt
        ia, ib = np.diff(ta), np.diff(tb)
        ca, cb = cv_of(ia), cv_of(ib)
        node_rows.append({"node_index_zero_based": i, "display_node": i+1,
                          "spikes_simplex": len(ea), "spikes_projected": len(eb),
                          "isis_simplex": len(ia), "isis_projected": len(ib),
                          "identical_spike_steps": identical,
                          "cv_simplex": ca, "cv_projected": cb,
                          "cv_abs_difference": abs(ca-cb) if ca is not None and cb is not None else None})
        for mode, steps in enumerate((ea, eb)):
            spike_rows.extend({"mode": ("simplex", "projected")[mode], "node_index_zero_based": i,
                               "step": int(step), "time": float(step*p.dt)} for step in steps)
        if i == NODE_INDEX:
            for mode, intervals in enumerate((ia, ib)):
                isi_rows.extend({"mode": ("simplex", "projected")[mode], "isi_index": j,
                                 "isi": float(x)} for j, x in enumerate(intervals))
    csv_write(OUT / f"{name}_node_statistics.csv", node_rows)
    if spike_rows:
        csv_write(OUT / f"{name}_spikes.csv", spike_rows)
    if isi_rows:
        csv_write(OUT / f"{name}_selected_node_isi.csv", isi_rows)
    case = {"alpha": alpha, "Dv": dv, "network_idx": NETWORK_INDEX,
            "network_seed": net_seed, "selected_node_index_zero_based": NODE_INDEX,
            "init_seed": INIT_SEED, "noise_seed": NOISE_SEEDS[case_idx],
            "max_all_nodes_all_steps_voltage_abs_error": float(max_v),
            "time_of_max_voltage_error": max_v_step*p.dt,
            "max_all_nodes_all_steps_recovery_abs_error": float(max_w),
            "all_nodes_identical_spike_steps": spike_identical,
            "max_defined_node_cv_abs_difference": max([x["cv_abs_difference"] for x in node_rows if x["cv_abs_difference"] is not None], default=None),
            "selected_node": node_rows[NODE_INDEX], "clipping_counts": clips.tolist(),
            "noise_increment_sha256": noise_hash.hexdigest(),
            "elapsed_seconds": time.perf_counter()-started,
            "initial_voltage_tolerance": INITIAL_VOLTAGE_TOLERANCE,
            "voltage_within_initial_tolerance": bool(max_v < INITIAL_VOLTAGE_TOLERANCE),
            "spike_check_passed": spike_identical}
    dump_json(OUT / f"{name}_summary.json", case)
    print(json.dumps(case, indent=2), flush=True)
    return case


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = MAIN / "chunks/network_000/parameters.json"
    params = json.loads(source.read_text())
    p = base.PairedSimplicialParams(**params)
    assert (p.n_nodes, p.dt, p.t_total, p.t_warmup) == (100, .05, 15000., 3000.)
    algebra_tests()
    audits, inputs, selected = audit_networks(p)
    conformance = integrator_conformance(p, selected)
    for name in ("morris_lecar_paired_alpha_compare_simplicial.py", "morris_lecar_simplicial_paper_array_chunked.py"):
        inputs.append({"path": name, "sha256": digest(ROOT/name)})
    cases = [run_case(p, selected, idx) for idx in range(len(CASES))]
    summary = {
        "scope": "Exact algebra and independent implementation check, not a new ensemble parameter sweep",
        "original_parameters": params, "validation_networks_for_operator": p.n_networks,
        "dynamical_networks": 1, "paired_noise_realizations_per_case": 1,
        "noise_block_size": BLOCK,
        "noise_protocol_note": "The original stable increment function is reused. Seeds and block shape define a new diagnostic realization, not a replay of archived trial/path seeds.",
        "cases_selected_before_simulation": "alpha=1.5,Dv=.16 and alpha=2,Dv=.30 are the archived simplicial CV minima; node 1/network 0 are fixed, not selected by agreement",
        "time_step_and_duration_unchanged": True,
        "integrator_vs_production_one_step": conformance,
        "arithmetic_note": "Independent summation orders in float64; no bitwise-identity claim for voltages",
        "integer_operator_equal_all_networks": all(x["integer_operator_identical"] for x in audits),
        "operator_max_abs_error": max(x["matrix_max_abs_error"] for x in audits),
        "force_direct_vs_edges_max_abs_error": max(x["force_direct_vs_edges_max_abs_error"] for x in audits),
        "force_direct_vs_production_max_abs_error": max(x["force_direct_vs_production_max_abs_error"] for x in audits),
        "uniform_once_per_edge_operator_max_abs_difference": max(x["uniform_once_per_edge_operator_max_abs_difference"] for x in audits),
        "cases": cases, "inputs": inputs,
        "verification_script_sha256": digest(Path(__file__)),
        "numpy_version": np.__version__,
        "all_spike_checks_passed": all(x["spike_check_passed"] for x in cases),
        "all_voltage_tolerance_checks_passed": all(x["voltage_within_initial_tolerance"] for x in cases),
        "initial_voltage_tolerance": INITIAL_VOLTAGE_TOLERANCE,
        "warning_policy": "The original 1e-6 voltage threshold is unchanged. Exceedance remains a warning, not a passed test. Exact equivalence follows from algebra, not this empirical threshold.",
    }
    for item in inputs:
        assert digest(ROOT/item["path"]) == item["sha256"], item
    dump_json(OUT / "summary.json", summary)
    assert summary["all_spike_checks_passed"], "See saved diagnostics; do not hide mismatched spike steps."
    print("PASS: integer operators, force evaluation, integrator conformance, and spike-step checks.", flush=True)
    if not summary["all_voltage_tolerance_checks_passed"]:
        print("WARNING: accumulated voltage difference exceeds the unchanged 1e-6 threshold; report the measured difference, not bitwise equality.", flush=True)


if __name__ == "__main__":
    main()
