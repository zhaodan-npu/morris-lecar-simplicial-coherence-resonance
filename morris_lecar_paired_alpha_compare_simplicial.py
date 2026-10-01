from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

plt = None

DEFAULT_DV_VALUES = (
    0.0,
    0.04,
    0.06,
    0.09,
    0.12,
    0.16,
    0.20,
    0.25,
    0.30,
    0.40,
    0.56,
    0.75,
    1.00,
    1.25,
    1.56,
    1.89,
    2.25,
    3.06,
    4.00,
    6.25,
    9.00,
)


def should_make_plots() -> bool:
    return os.environ.get("ML_MAKE_PLOTS", "").lower() in {"1", "true", "yes", "on"}


def load_pyplot():
    global plt
    if plt is None:
        import matplotlib.pyplot as pyplot

        plt = pyplot
    return plt


@dataclass
class PairedSimplicialParams:
    n_nodes: int = 60
    n_networks: int = 10
    n_trials_per_network: int = 10
    p_edge: float = 0.15
    p_fill_simplex: float = 0.5
    sigma_1: float = 0.005
    sigma_2: float = 0.08
    capacitance: float = 20.0
    g_ca: float = 4.4
    g_k: float = 8.0
    g_l: float = 2.0
    v_ca: float = 120.0
    v_k: float = -84.0
    v_l: float = -60.0
    v1: float = -1.2
    v2: float = 18.0
    v3: float = 2.0
    v4: float = 30.0
    phi: float = 0.04
    current: float = 86.0
    dt: float = 0.05
    t_total: float = 1800.0
    t_warmup: float = 300.0
    spike_threshold: float = 0.0
    alphas: tuple[float, ...] = (1.5, 2.0)
    beta: float = 0.0
    # Backward-compatible field name: values are D_v, converted to noise_scale inside alpha loops.
    noise_scales: tuple[float, ...] = DEFAULT_DV_VALUES
    base_seed: int = 20260422

    @property
    def total_realizations_per_condition(self) -> int:
        return self.n_networks * self.n_trials_per_network


@dataclass
class SimplicialNetwork:
    a1: np.ndarray
    pairwise_operator: np.ndarray
    cliques3: list[tuple[int, int, int]]
    simplices2: list[tuple[int, int, int]]
    a2: np.ndarray
    l2: np.ndarray
    edge_count: int
    clique_count: int
    simplex_count: int


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def m_inf(v: np.ndarray, p: PairedSimplicialParams):
    return 0.5 * (1.0 + np.tanh((v - p.v1) / p.v2))


def w_inf(v: np.ndarray, p: PairedSimplicialParams):
    return 0.5 * (1.0 + np.tanh((v - p.v3) / p.v4))


def tau_w(v: np.ndarray, p: PairedSimplicialParams):
    return 1.0 / np.cosh(np.clip((v - p.v3) / (2.0 * p.v4), -40.0, 40.0))


def build_simplicial_network(p: PairedSimplicialParams, rng: np.random.Generator) -> SimplicialNetwork:
    """
    Build a true simplicial higher-order network:
    1. sample a base pairwise graph A^(1)
    2. enumerate existing 3-cliques of the base graph
    3. fill only those cliques as 2-simplices with probability p_fill_simplex
    4. construct A^(2) and L^(2) from the filled 2-simplices

    Here:
      A^(2)_{ij} = number of filled 2-simplices containing both i and j,  i != j
      k_i^(2)    = number of filled 2-simplices incident to node i
      L^(2)      = 0.5 * A^(2) - diag(k^(2))

    This L^(2) reproduces the simplex contribution
      sum_(j,k in simplices containing i) [ (v_j + v_k)/2 - v_i ].
    """
    n = p.n_nodes
    upper = rng.random((n, n))
    a1 = np.triu((upper < p.p_edge).astype(float), 1)
    a1 = a1 + a1.T
    degree = a1.sum(axis=1)
    pairwise_operator = p.sigma_1 * (a1 - np.diag(degree))

    cliques3: list[tuple[int, int, int]] = []
    simplices2: list[tuple[int, int, int]] = []
    a2 = np.zeros((n, n), dtype=float)
    simplex_degree = np.zeros(n, dtype=float)

    for i in range(n - 2):
        neighbors_i = np.where(a1[i, i + 1 :] > 0)[0] + i + 1
        for j_idx, j in enumerate(neighbors_i[:-1]):
            common = neighbors_i[j_idx + 1 :]
            common = common[a1[j, common] > 0]
            for k in common.tolist():
                cliques3.append((i, j, k))
                if rng.random() >= p.p_fill_simplex:
                    continue
                simplices2.append((i, j, k))
                a2[i, j] += 1.0
                a2[j, i] += 1.0
                a2[i, k] += 1.0
                a2[k, i] += 1.0
                a2[j, k] += 1.0
                a2[k, j] += 1.0
                simplex_degree[i] += 1.0
                simplex_degree[j] += 1.0
                simplex_degree[k] += 1.0

    l2 = 0.5 * a2 - np.diag(simplex_degree)
    return SimplicialNetwork(
        a1=a1,
        pairwise_operator=pairwise_operator,
        cliques3=cliques3,
        simplices2=simplices2,
        a2=a2,
        l2=l2,
        edge_count=int(a1.sum() // 2),
        clique_count=len(cliques3),
        simplex_count=len(simplices2),
    )


def sample_alpha_stable(alpha: float, beta: float, size: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    if alpha == 2.0:
        return np.sqrt(2.0) * rng.standard_normal(size)

    u = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=size)
    w = rng.exponential(1.0, size=size)

    if abs(alpha - 1.0) > 1e-12:
        zeta = beta * np.tan(0.5 * np.pi * alpha)
        b = np.arctan(zeta) / alpha
        s = (1.0 + zeta * zeta) ** (1.0 / (2.0 * alpha))
        numerator = np.sin(alpha * (u + b))
        denominator = np.cos(u) ** (1.0 / alpha)
        frac = np.cos(u - alpha * (u + b)) / w
        return s * numerator / denominator * frac ** ((1.0 - alpha) / alpha)

    half_pi = 0.5 * np.pi
    term1 = (half_pi + beta * u) * np.tan(u)
    term2 = beta * np.log((half_pi * w * np.cos(u)) / (half_pi + beta * u))
    return (2.0 / np.pi) * (term1 - term2)


def levy_increment(scale: float, alpha: float, beta: float, dt: float, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    if scale == 0.0:
        return np.zeros(shape, dtype=float)
    z = sample_alpha_stable(alpha, beta, shape, rng)
    return scale * (dt ** (1.0 / alpha)) * z


def voltage_noise_intensity(noise_scale: float, alpha: float, capacitance: float) -> float:
    """Noise intensity in the membrane-voltage equation: D_v = (scale / C)^alpha."""
    return float((noise_scale / capacitance) ** alpha)


def noise_scale_from_voltage_noise_intensity(noise_intensity_dv: float, alpha: float, capacitance: float) -> float:
    """Convert membrane-voltage noise intensity D_v to the stable-law scale used by the integrator."""
    return float(capacitance * (noise_intensity_dv ** (1.0 / alpha)))


def simulate_batch(
    p: PairedSimplicialParams,
    network: SimplicialNetwork,
    alpha: float,
    noise_scale: float,
    higher_order_enabled: bool,
    init_seed: int,
    noise_seed: int,
):
    n_steps = int(p.t_total / p.dt)
    n_paths = p.n_trials_per_network
    n = p.n_nodes

    init_rng = np.random.default_rng(init_seed)
    noise_rng = np.random.default_rng(noise_seed)

    v = -60.0 + 2.0 * init_rng.standard_normal((n_paths, n))
    w = w_inf(v, p)
    prev_v = v.copy()

    operator = network.pairwise_operator + (p.sigma_2 * network.l2 if higher_order_enabled else 0.0)

    last_spike = np.full((n_paths, n), np.nan)
    spike_count = np.zeros(n_paths, dtype=np.int32)
    isi_lists = [[] for _ in range(n_paths)]
    mean_sum = np.zeros(n_paths)
    mean_sq_sum = np.zeros(n_paths)
    kept_steps = 0

    for step in range(n_steps):
        # As requested: keep the same matrix form for the coupling.
        coupling = v @ operator.T
        i_ca = p.g_ca * m_inf(v, p) * (v - p.v_ca)
        i_k = p.g_k * w * (v - p.v_k)
        i_l = p.g_l * (v - p.v_l)
        dv = (p.current - i_ca - i_k - i_l + coupling) / p.capacitance
        dw = p.phi * (w_inf(v, p) - w) / np.maximum(tau_w(v, p), 1e-6)

        jump = levy_increment(noise_scale, alpha, p.beta, p.dt, (n_paths, n), noise_rng) / p.capacitance
        v = v + p.dt * dv + jump
        w = w + p.dt * dw
        v = np.clip(v, -200.0, 200.0)
        w = np.clip(w, -0.5, 1.5)

        current_time = (step + 1) * p.dt
        crossing = (prev_v < p.spike_threshold) & (v >= p.spike_threshold)
        if current_time > p.t_warmup:
            path_idx, node_idx = np.where(crossing)
            for path, node in zip(path_idx.tolist(), node_idx.tolist()):
                previous = last_spike[path, node]
                if np.isfinite(previous):
                    isi_lists[path].append(current_time - previous)
                last_spike[path, node] = current_time
                spike_count[path] += 1

            mean_v = np.mean(v, axis=1)
            mean_sum += mean_v
            mean_sq_sum += mean_v**2
            kept_steps += 1

        prev_v = v.copy()

    isi_cv = np.full(n_paths, np.nan)
    for idx, intervals in enumerate(isi_lists):
        if len(intervals) >= 3:
            arr = np.asarray(intervals, dtype=float)
            isi_cv[idx] = float(np.std(arr) / (np.mean(arr) + 1e-12))

    mean_activity_std = np.sqrt(np.maximum(mean_sq_sum / max(kept_steps, 1) - (mean_sum / max(kept_steps, 1)) ** 2, 0.0))
    return {
        "isi_cv": isi_cv,
        "spike_count_total": spike_count.astype(float),
        "mean_activity_std": mean_activity_std,
        "edge_count": np.full(n_paths, network.edge_count, dtype=float),
        "clique_count": np.full(n_paths, network.clique_count, dtype=float),
        "simplex_count": np.full(n_paths, network.simplex_count, dtype=float),
    }


def summarize(values: list[float] | np.ndarray):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    return float(np.mean(finite)), float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0


def main():
    root = Path(__file__).resolve().parent / "paired_alpha_compare_simplicial_0_15_0_5"
    root.mkdir(parents=True, exist_ok=True)
    p = PairedSimplicialParams()

    print("Running paired simplicial Morris-Lecar alpha comparison...")
    print(p)

    all_rows = []
    aggregated_rows = []
    network_rows = []

    for network_idx in range(p.n_networks):
        network_rng = np.random.default_rng(p.base_seed + stable_seed("simplicial_network", network_idx))
        network = build_simplicial_network(p, network_rng)
        network_rows.append(
            {
                "network_idx": network_idx,
                "edge_count": network.edge_count,
                "clique_count": network.clique_count,
                "simplex_count": network.simplex_count,
            }
        )
        print(
            f"network {network_idx + 1:02d}/{p.n_networks}: "
            f"edges={network.edge_count}, 3-cliques={network.clique_count}, filled simplices={network.simplex_count}"
        )

        for alpha in p.alphas:
            for noise_idx, noise_intensity_dv in enumerate(p.noise_scales):
                scale = noise_scale_from_voltage_noise_intensity(noise_intensity_dv, alpha, p.capacitance)
                for trial_idx in range(p.n_trials_per_network):
                    init_seed = p.base_seed + stable_seed("init", network_idx, trial_idx)
                    noise_seed = p.base_seed + stable_seed("noise", alpha, noise_idx, network_idx, trial_idx)

                    for label, higher in (("control", False), ("higher_order", True)):
                        metrics = simulate_batch(
                            p,
                            network,
                            alpha=alpha,
                            noise_scale=scale,
                            higher_order_enabled=higher,
                            init_seed=init_seed,
                            noise_seed=noise_seed,
                        )
                        # Each call returns n_trials_per_network paths with shared init/noise seeds
                        # for the two labels, which guarantees paired comparison under the same
                        # base graph, same simplex fill, same initial states, and same noise draws.
                        for path_idx in range(p.n_trials_per_network):
                            all_rows.append(
                                {
                                    "alpha": alpha,
                                    "label": label,
                                    "network_idx": network_idx,
                                    "trial_idx": trial_idx,
                                    "path_idx": path_idx,
                                    "noise_scale": scale,
                                    "noise_intensity_dv": noise_intensity_dv,
                                    "isi_cv": float(metrics["isi_cv"][path_idx]),
                                    "spike_count_total": float(metrics["spike_count_total"][path_idx]),
                                    "mean_activity_std": float(metrics["mean_activity_std"][path_idx]),
                                    "edge_count": float(metrics["edge_count"][path_idx]),
                                    "clique_count": float(metrics["clique_count"][path_idx]),
                                    "simplex_count": float(metrics["simplex_count"][path_idx]),
                                }
                            )

    for alpha in p.alphas:
        for label in ("control", "higher_order"):
            for noise_intensity_dv in p.noise_scales:
                scale = noise_scale_from_voltage_noise_intensity(noise_intensity_dv, alpha, p.capacitance)
                subset = [
                    row
                    for row in all_rows
                    if row["alpha"] == alpha
                    and row["label"] == label
                    and abs(row["noise_intensity_dv"] - noise_intensity_dv) < 1e-12
                ]
                aggregated = {
                    "alpha": alpha,
                    "label": label,
                    "noise_scale": scale,
                    "noise_intensity_dv": noise_intensity_dv,
                    "n_networks": p.n_networks,
                    "n_trials_per_network": p.n_trials_per_network,
                    "total_realizations": len(subset),
                }
                for metric in ("isi_cv", "spike_count_total", "mean_activity_std", "edge_count", "clique_count", "simplex_count"):
                    mean, std = summarize([row[metric] for row in subset])
                    aggregated[f"{metric}_mean"] = mean
                    aggregated[f"{metric}_std"] = std
                aggregated_rows.append(aggregated)

    summary = {}
    for alpha in p.alphas:
        summary[str(alpha)] = {}
        for label in ("control", "higher_order"):
            valid = [row for row in aggregated_rows if row["alpha"] == alpha and row["label"] == label and row["isi_cv_mean"] is not None]
            best = min(valid, key=lambda row: row["isi_cv_mean"]) if valid else None
            summary[str(alpha)][label] = {
                "best_noise_scale": None if best is None else best["noise_scale"],
                "best_noise_intensity_dv": None if best is None else best["noise_intensity_dv"],
                "best_cv_value": None if best is None else best["isi_cv_mean"],
            }

    (root / "parameters.json").write_text(json.dumps(asdict(p), indent=2), encoding="utf-8")
    (root / "network_summary.json").write_text(json.dumps(network_rows, indent=2), encoding="utf-8")
    (root / "all_realizations.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    (root / "aggregated_metrics.json").write_text(json.dumps(aggregated_rows, indent=2), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(root / "network_summary.csv", network_rows)
    write_csv(root / "aggregated_metrics.csv", aggregated_rows)

    if should_make_plots():
        pyplot = load_pyplot()
        fig, axes = pyplot.subplots(2, 2, figsize=(11, 8.5), sharex=True)
        color_map = {"control": "#455a64", "higher_order": "#c62828"}
        for col_idx, alpha in enumerate(p.alphas):
            subset_alpha = [row for row in aggregated_rows if row["alpha"] == alpha]
            for label in ("control", "higher_order"):
                rows = [row for row in subset_alpha if row["label"] == label]
                x = [row["noise_intensity_dv"] for row in rows]
                cv = [np.nan if row["isi_cv_mean"] is None else row["isi_cv_mean"] for row in rows]
                cv_err = [0.0 if row["isi_cv_std"] is None else row["isi_cv_std"] for row in rows]
                spikes = [row["spike_count_total_mean"] for row in rows]
                spike_err = [0.0 if row["spike_count_total_std"] is None else row["spike_count_total_std"] for row in rows]
                axes[0, col_idx].errorbar(x, cv, yerr=cv_err, fmt="o-", capsize=3, linewidth=1.8, color=color_map[label], label=label)
                axes[1, col_idx].errorbar(x, spikes, yerr=spike_err, fmt="o-", capsize=3, linewidth=1.8, color=color_map[label], label=label)

            axes[0, col_idx].set_title(f"alpha = {alpha}")
            axes[0, col_idx].set_ylabel("ISI CV")
            axes[1, col_idx].set_ylabel("Spike count")
            axes[1, col_idx].set_xlabel(r"$D_v$")
            axes[0, col_idx].grid(alpha=0.3)
            axes[1, col_idx].grid(alpha=0.3)
            axes[0, col_idx].legend(frameon=False)

        fig.suptitle("Paired simplicial comparison: Levy alpha=1.5 vs Gaussian alpha=2.0")
        fig.tight_layout()
        fig.savefig(root / "paired_alpha_compare_simplicial.png", dpi=180)
        pyplot.close(fig)
    else:
        print("Data-only mode; skipping paired_alpha_compare_simplicial.png")
    print(f"Results will be saved to: {root}")


if __name__ == "__main__":
    main()
