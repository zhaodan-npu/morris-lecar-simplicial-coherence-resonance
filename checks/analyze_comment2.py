"""Reproduce the strength-matched control from existing raw trajectories."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "analysis"
FIG = HERE / "figures"
MAIN = ROOT / "paper_suite_true_simplicial_array_prl_N100_Dv_refined" / "candidate39_main"
MATCH = ROOT / "paper_suite_matched_coupling_control_N100_Dv_refined"
LABELS = ("control", "matched_pairwise", "higher_order")
ALPHAS = (1.5, 2.0)
N_NETWORKS = 30
N_PATHS = 144
N_BOOT = 20000
SEED = 20260929
MIN_SPIKES = 1000
DISPLAY = {"control": "Pairwise baseline", "matched_pairwise": "Matched pairwise", "higher_order": "Simplicial"}
COLORS = {"control": "#36454f", "matched_pairwise": "#2166ac", "higher_order": "#c62828"}
MARKERS = {"control": "o", "matched_pairwise": "^", "higher_order": "s"}


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def safe_ratio(x, y):
    return np.divide(x, y, out=np.full(np.broadcast_shapes(x.shape, y.shape), np.nan), where=y > 0)


def read_raw():
    params = json.loads((MAIN / "chunks/network_000/parameters.json").read_text())
    grid = np.array(params["noise_scales"], dtype=float)
    shape = (3, 2, N_NETWORKS, len(grid), N_PATHS)
    cvs = np.full(shape, np.nan)
    spikes = np.full(shape, np.nan)
    seen = np.zeros(shape, dtype=bool)
    provenance = []
    matching = []
    for n in range(N_NETWORKS):
        sub = Path("chunks") / f"network_{n:03d}"
        original = json.loads((MAIN / sub / "parameters.json").read_text())
        matched_original = json.loads((MATCH / sub / "parameters.json").read_text())
        assert original == params == matched_original, f"Unpaired parameters: {n}"
        meta = json.loads((MATCH / sub / "network_summary.json").read_text())[0]
        main_meta = json.loads((MAIN / sub / "network_summary.json").read_text())[0]
        for key in ("edge_count", "clique_count", "simplex_count"):
            assert meta[key] == main_meta[key], (n, key)
        mp = json.loads((MATCH / sub / "matched_parameters.json").read_text())
        expected = dict(params, sigma_1=meta["sigma1_matched"], sigma_2=0.0)
        assert mp == expected
        assert abs(meta["k1_actual"] - 2 * meta["edge_count"] / params["n_nodes"]) < 1e-12
        assert abs(meta["k2_actual"] - 3 * meta["simplex_count"] / params["n_nodes"]) < 1e-12
        meta["matching_error"] = abs(meta["sigma1_matched"] * meta["k1_actual"] - meta["g_eff_target"])
        assert meta["matching_error"] < 1e-12
        matching.append(meta)
        for directory in (MAIN, MATCH):
            path = directory / sub / "all_realizations.csv"
            provenance.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            with path.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    l = LABELS.index(row["label"])
                    a = ALPHAS.index(float(row["alpha"]))
                    d = int(np.flatnonzero(np.isclose(grid, float(row["noise_intensity_dv"]), rtol=0, atol=1e-12))[0])
                    t, p = int(row["trial_idx"]), int(row["path_idx"])
                    assert 0 <= t < 12 and 0 <= p < 12 and int(row["network_idx"]) == n
                    ix = (l, a, n, d, t * 12 + p)
                    assert not seen[ix], (path, ix)
                    seen[ix] = True
                    cvs[ix] = float(row["isi_cv"]) if row["isi_cv"] not in ("", "None") else np.nan
                    spikes[ix] = float(row["spike_count_total"])
    assert seen.all() and np.isfinite(spikes).all()
    assert np.all(cvs[np.isfinite(cvs)] >= 0)
    return params, grid, cvs, spikes, matching, provenance


def operator_check(params):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    import morris_lecar_paired_alpha_compare_simplicial as base
    p = base.PairedSimplicialParams(**params)
    errors = []
    for n in range(N_NETWORKS):
        rng = np.random.default_rng(p.base_seed + base.stable_seed("paper_network", n))
        net = base.build_simplicial_network(p, rng)
        weights = p.sigma_1 * net.a1 + 0.5 * p.sigma_2 * net.a2
        weighted_lap = weights - np.diag(weights.sum(axis=1))
        operator = net.pairwise_operator + p.sigma_2 * net.l2
        error = float(np.abs(operator - weighted_lap).max())
        assert error < 1e-12
        assert not np.any((weights > 0) & (net.a1 == 0))
        errors.append(error)
    return max(errors)


def analyze(params, grid, cvs, spikes, matching, provenance):
    # Resample entire networks jointly across noise points and all three conditions.
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(N_NETWORKS, np.full(N_NETWORKS, 1 / N_NETWORKS), size=N_BOOT)
    finite = np.isfinite(cvs)
    totals = np.where(finite, cvs, 0).sum(axis=-1)
    counts = finite.sum(axis=-1)
    spike_totals = spikes.sum(axis=-1)
    mean = safe_ratio(totals.sum(axis=2), counts.sum(axis=2))
    spike_mean = spike_totals.sum(axis=2) / (N_NETWORKS * N_PATHS)
    eligible = (spike_mean >= MIN_SPIKES) & np.isfinite(mean)
    boot = np.full((3, 2, N_BOOT, len(grid)), np.nan)
    best_boot = np.full((3, 2, N_BOOT), np.nan)
    grid_boot = np.full((3, 2, N_BOOT), np.nan)
    curves, best, network_rows = [], [], []
    sensitivity = []
    for l, label in enumerate(LABELS):
        for a, alpha in enumerate(ALPHAS):
            boot[l, a] = safe_ratio(weights @ totals[l, a], weights @ counts[l, a])
            bs = weights @ spike_totals[l, a] / (N_NETWORKS * N_PATHS)
            candidates = np.where((bs >= MIN_SPIKES) & np.isfinite(boot[l, a]), boot[l, a], np.inf)
            bi = candidates.argmin(axis=1)
            best_boot[l, a] = candidates[np.arange(N_BOOT), bi]
            grid_boot[l, a] = grid[bi]
            assert np.isfinite(best_boot[l, a]).all()
            ix = int(np.argmin(np.where(eligible[l, a], mean[l, a], np.inf)))
            q = np.percentile(best_boot[l, a], [2.5, 97.5])
            arr = cvs[l, a, :, ix].ravel()
            sp = spikes[l, a, :, ix].ravel()
            row = {"alpha": alpha, "label": label, "Dv_star": grid[ix], "CV_star": mean[l, a, ix],
                   "CV_star_ci_low": q[0], "CV_star_ci_high": q[1], "mean_spike_count": spike_mean[l, a, ix],
                   "valid_cv_count": int(np.isfinite(arr).sum()), "total_paths": len(arr),
                   "paths_below_1000_spikes": int((sp < MIN_SPIKES).sum())}
            best.append(row)
            for d, dv in enumerate(grid):
                b = boot[l, a, :, d]
                qd = np.percentile(b[np.isfinite(b)], [2.5, 97.5]) if np.isfinite(b).any() else [np.nan] * 2
                qs = np.percentile(bs[:, d], [2.5, 97.5])
                curves.append({"alpha": alpha, "label": label, "Dv": dv,
                    "CV_mean": mean[l, a, d], "CV_ci_low": qd[0], "CV_ci_high": qd[1],
                    "eligible_for_minimum": bool(eligible[l, a, d]),
                    "valid_cv_count": int(counts[l, a, :, d].sum()), "total_paths": N_NETWORKS * N_PATHS,
                    "mean_spike_count": spike_mean[l, a, d], "spike_ci_low": qs[0], "spike_ci_high": qs[1],
                    "fraction_paths_below1000": float((spikes[l, a, :, d] < MIN_SPIKES).mean())})
                for n in range(N_NETWORKS):
                    network_rows.append({"alpha": alpha, "label": label, "network": n, "Dv": dv,
                        "cv_sum": totals[l, a, n, d], "n_valid": counts[l, a, n, d],
                        "spike_count_sum": spike_totals[l, a, n, d], "n_paths": N_PATHS})
            good = finite[l, a] & (spikes[l, a] >= MIN_SPIKES)
            good_counts = good.sum(axis=(0, 2))
            good_sum = np.where(good, cvs[l, a], 0).sum(axis=(0, 2))
            good_mean = safe_ratio(good_sum, good_counts)
            valid_grid = eligible[l, a] & (good_counts >= 0.95 * N_NETWORKS * N_PATHS)
            si = int(np.argmin(np.where(valid_grid, good_mean, np.inf)))
            sensitivity.append({"alpha": alpha, "label": label,
                "rule": "each retained path >=1000 total spikes; >=95% of paths retained at eligible grid point",
                "Dv_star": grid[si], "CV_star": good_mean[si], "n_retained": int(good_counts[si]),
                "CV_at_primary_optimum": good_mean[ix], "n_at_primary_optimum": int(good_counts[ix])})
    differences = []
    for a, alpha in enumerate(ALPHAS):
        diff = best_boot[2, a] - best_boot[1, a]
        q = np.percentile(diff, [2.5, 97.5])
        b = {r["label"]: r for r in best if r["alpha"] == alpha}
        differences.append({"alpha": alpha, "contrast": "simplicial_minus_matched_optimal_CV",
            "estimate": b["higher_order"]["CV_star"] - b["matched_pairwise"]["CV_star"],
            "ci_low": q[0], "ci_high": q[1]})
    old = json.loads((MATCH / "matched_control_comparison_summary.json").read_text())
    for row in best:
        old_row = old["best_by_alpha"][str(row["alpha"])][row["label"]]
        assert abs(old_row["isi_cv_mean"] - row["CV_star"]) < 1e-12
        assert old_row["noise_intensity_dv"] == row["Dv_star"]
    sigmas = np.array([r["sigma1_matched"] for r in matching])
    summary = {"parameters": params, "n_raw_rows": int(cvs.size), "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": SEED, "bootstrap_unit": "paired network, same weights for conditions and noise points",
        "estimator": "arithmetic mean of finite path-level pooled-within-node-ISI CVs; ratio of sums in network bootstrap",
        "eligibility": "finite mean CV and mean total network spike count >=1000; reapplied within each bootstrap sample",
        "minima_scope": "discrete 20-point noise grid, not continuous optima; pointwise curve intervals",
        "mean_sigma1_matched": float(sigmas.mean()), "sigma1_matched_range": [float(sigmas.min()), float(sigmas.max())],
        "max_matching_error": max(r["matching_error"] for r in matching),
        "weighted_pairwise_operator_max_error": operator_check(params),
        "best": best, "differences": differences, "sensitivity": sensitivity,
        "new_simulations_run": False}
    OUT.mkdir(exist_ok=True)
    for name, rows in [("curves.csv", curves), ("optimal_summary.csv", best), ("network_statistics.csv", network_rows),
                       ("paired_differences.csv", differences), ("sensitivity.csv", sensitivity), ("network_matching.csv", matching)]:
        csv_write(OUT / name, rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (OUT / "input_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    macro_lines = ["% Generated by analyze_comment2.py; do not edit numerical values here."]
    codes = {"control": "Baseline", "matched_pairwise": "Matched", "higher_order": "Simplicial"}
    for r in best:
        code = codes[r["label"]] + ("Levy" if r["alpha"] == 1.5 else "Gauss")
        for suffix, val in [("CV", f"{r['CV_star']:.4f}"), ("Dv", f"{r['Dv_star']:g}"),
                            ("Lo", f"{r['CV_star_ci_low']:.4f}"), ("Hi", f"{r['CV_star_ci_high']:.4f}"),
                            ("Valid", str(r['valid_cv_count']))]:
            macro_lines.append(f"\\newcommand{{\\{code}{suffix}}}{{{val}}}")
    for r in differences:
        code = "Difference" + ("Levy" if r["alpha"] == 1.5 else "Gauss")
        for suffix, key in [("Mean", "estimate"), ("Lo", "ci_low"), ("Hi", "ci_high")]:
            macro_lines.append(f"\\newcommand{{\\{code}{suffix}}}{{{r[key]:.4f}}}")
    (OUT / "numbers.tex").write_text("\n".join(macro_lines) + "\n", encoding="ascii")
    return summary, curves


def plot(curves, summary):
    FIG.mkdir(exist_ok=True)
    font_path = font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16,
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
        "legend.fontsize": 12, "axes.linewidth": 1.5, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 1.3, "ytick.major.width": 1.3, "xtick.minor.width": 1,
        "ytick.minor.width": 1, "xtick.major.size": 6, "ytick.major.size": 6,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    for ai, alpha in enumerate(ALPHAS):
        for metric in ("cv", "spikes"):
            fig, ax = plt.subplots(figsize=(5.4, 4.1))
            fig.subplots_adjust(left=.16, right=.97, bottom=.18, top=.94)
            for label in LABELS:
                rows = [r for r in curves if r["label"] == label and r["alpha"] == alpha]
                x = np.array([r["Dv"] for r in rows])
                key, lo, hi = ("CV_mean", "CV_ci_low", "CV_ci_high") if metric == "cv" else ("mean_spike_count", "spike_ci_low", "spike_ci_high")
                scale = 1 if metric == "cv" else 1000
                y = np.array([r[key] for r in rows]) / scale
                lower = np.array([r[lo] for r in rows]) / scale
                upper = np.array([r[hi] for r in rows]) / scale
                valid = np.array([r["eligible_for_minimum"] for r in rows]) if metric == "cv" else np.ones(len(rows), bool)
                yp = np.where(valid, y, np.nan)
                # Intervals are drawn explicitly, so percentile asymmetry cannot produce negative yerr.
                ax.plot(x, yp, color=COLORS[label], marker=MARKERS[label], lw=2.2, ms=5.7,
                        mew=1.1, ls="--" if label == "control" else "-", label=DISPLAY[label], zorder=3)
                ax.vlines(x[valid], lower[valid], upper[valid], color=COLORS[label], lw=1.1, zorder=2)
                ax.plot(x[valid], lower[valid], "_", color=COLORS[label], ms=5)
                ax.plot(x[valid], upper[valid], "_", color=COLORS[label], ms=5)
                if metric == "cv":
                    invalid = ~valid & np.isfinite(y)
                    ax.plot(x[invalid], y[invalid], linestyle="none", marker=MARKERS[label], ms=5.7,
                            mfc="white", mec=COLORS[label], mew=1.0)
                    best = next(r for r in summary["best"] if r["alpha"] == alpha and r["label"] == label)
                    ax.plot(best["Dv_star"], best["CV_star"], marker=MARKERS[label], ms=10,
                            mfc="white", mec=COLORS[label], mew=1.8, zorder=5)
                    ax.vlines(best["Dv_star"], 0, best["CV_star"], color=COLORS[label], ls=":", lw=1.1)
            ax.set_xscale("log")
            ax.set_xlim(.035, 10.5)
            ax.set_xticks([.04, .1, .3, 1, 4, 9], ["0.04", "0.10", "0.30", "1", "4", "9"])
            ax.set_xlabel(r"$D_v$")
            ax.set_ylabel("CV" if metric == "cv" else r"Spike count ($10^3$)")
            ax.set_ylim(0, 2.1 if metric == "cv" else 65)
            ax.set_yticks([0, .5, 1, 1.5, 2] if metric == "cv" else [0, 20, 40, 60])
            ax.minorticks_on()
            ax.tick_params(which="both", top=True, right=True)
            ax.text(.05, .93, rf"$\alpha={alpha:.1f}$", transform=ax.transAxes, va="top", fontsize=18)
            ax.text(.99, .04, f"({chr(97 + ai + (2 if metric == 'spikes' else 0))})", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=18)
            legend_options = {"loc": "upper right"} if metric == "cv" else {"loc": "upper left", "bbox_to_anchor": (.035, .80)}
            ax.legend(**legend_options, borderaxespad=.35, handlelength=1.8, labelspacing=.25)
            stem = f"fig_R1_{metric}_alpha{str(alpha).replace('.', '')}"
            for suffix in ("pdf", "eps", "png"):
                fig.savefig(FIG / f"{stem}.{suffix}", dpi=450)
            plt.close(fig)
    (OUT / "figure_style.json").write_text(json.dumps({"font": font_path,
        "reference": "submitted PDF Fig. 2; plot_formal_prl_figures.py",
        "pointwise_confidence_level": .95, "separate_panels": True}, indent=2), encoding="utf-8")


def main():
    params, grid, cvs, spikes, matching, provenance = read_raw()
    summary, curves = analyze(params, grid, cvs, spikes, matching, provenance)
    plot(curves, summary)
    print(json.dumps({k: summary[k] for k in ("n_raw_rows", "mean_sigma1_matched", "best", "differences", "sensitivity")}, indent=2))


if __name__ == "__main__":
    main()
