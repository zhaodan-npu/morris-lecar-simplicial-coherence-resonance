"""Audit archived Fig. 2(b) data without changing simulations or spike rules."""

from __future__ import annotations

import csv
import hashlib
import json
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
SUITE = ROOT / "paper_suite_true_simplicial_array_prl_N100_Dv_refined"
SOURCES = {
    "original": SUITE / "candidate39_main",
    "longer": SUITE / "candidate39_longer_time",
    "smaller_dt": ROOT / "paper_suite_candidate39_dt001_N100_Dv_refined/candidate39_dt_0.010",
}
LABELS = ("control", "higher_order")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_suite(name, source):
    params = json.loads((source / "chunks/network_000/parameters.json").read_text())
    grid = params["noise_scales"]
    assert params["n_networks"] == 30 and params["n_trials_per_network"] == 12
    shape = (2, 30, len(grid), 144)
    cv, spikes = np.full(shape, np.nan), np.full(shape, np.nan)
    seen = np.zeros(shape, dtype=bool)
    manifest, networks = [], []
    for n in range(30):
        directory = source / f"chunks/network_{n:03d}"
        param_file = directory / "parameters.json"
        assert json.loads(param_file.read_text()) == params
        network_file = directory / "network_summary.json"
        networks.append(json.loads(network_file.read_text())[0])
        raw = directory / "all_realizations.csv"
        for path in (param_file, network_file, raw):
            manifest.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})
        with raw.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if float(row["alpha"]) != 2.0:
                    continue
                label = LABELS.index(row["label"])
                dv_index = grid.index(float(row["noise_intensity_dv"]))
                trial, path_index = int(row["trial_idx"]), int(row["path_idx"])
                assert int(row["network_idx"]) == n
                assert 0 <= trial < 12 and 0 <= path_index < 12
                ix = (label, n, dv_index, 12 * trial + path_index)
                assert not seen[ix]
                seen[ix] = True
                value = row["isi_cv"]
                cv[ix] = float(value) if value not in ("", "None") else np.nan
                spikes[ix] = float(row["spike_count_total"])
    assert seen.all() and np.isfinite(spikes).all()
    assert np.all(spikes >= 0) and np.all(spikes == np.floor(spikes))
    assert np.all(cv[np.isfinite(cv)] >= 0)
    rows, network_rows = [], []
    for li, label in enumerate(LABELS):
        for di, dv in enumerate(grid):
            values, counts = cv[li, :, di].ravel(), spikes[li, :, di].ravel()
            finite = np.isfinite(values)
            rows.append({
                "dataset": name, "alpha": 2.0, "label": label, "Dv": dv,
                "dt": params["dt"], "T": params["t_total"],
                "transient": params["t_warmup"],
                "observation_window": params["t_total"] - params["t_warmup"],
                "base_seed": params["base_seed"], "n_paths": int(values.size),
                "n_valid_cv": int(finite.sum()), "valid_cv_fraction": float(finite.mean()),
                "mean_cv_finite_only": float(values[finite].mean()) if finite.any() else None,
                "mean_spikes_all_paths": float(counts.mean()),
                "median_spikes": float(np.median(counts)),
                "spikes_q25": float(np.quantile(counts, .25)),
                "spikes_q75": float(np.quantile(counts, .75)),
                "min_spikes": int(counts.min()), "max_spikes": int(counts.max()),
                "zero_spike_paths": int((counts == 0).sum()),
                "paths_below_1000_spikes": int((counts < 1000).sum()),
            })
            for n in range(30):
                good = np.isfinite(cv[li, n, di])
                network_rows.append({"dataset": name, "label": label, "Dv": dv,
                                     "network": n, "n_paths": 144,
                                     "n_valid_cv": int(good.sum()),
                                     "cv_sum": float(cv[li, n, di, good].sum()),
                                     "total_spikes": int(spikes[li, n, di].sum())})
    with (source / "aggregated_metrics.csv").open(encoding="utf-8", newline="") as handle:
        old = {(r["label"], float(r["noise_intensity_dv"])): r
               for r in csv.DictReader(handle) if float(r["alpha"]) == 2}
    for row in rows:
        prior = old[row["label"], row["Dv"]]
        assert abs(row["mean_cv_finite_only"] - float(prior["isi_cv_mean"])) < 1e-11
        assert abs(row["mean_spikes_all_paths"] - float(prior["spike_count_total_mean"])) < 1e-9
        assert row["n_paths"] == int(prior["total_realizations"])
    return rows, network_rows, params, networks, manifest


def tables(rows):
    low = [r for r in rows if r["dataset"] == "original" and r["Dv"] <= .2]
    write_csv(OUT / "original_low_noise.csv", low)
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"网络 & $D_v$ & CV & 平均总spike数 & 有效CV/4320 & $<1000$的比例\\",
             r"\midrule"]
    for row in low:
        label = "成对" if row["label"] == "control" else "单纯形"
        lines.append(f"{label} & {row['Dv']:.2f} & {row['mean_cv_finite_only']:.3f} & "
                     f"{row['mean_spikes_all_paths']:.2f} & {row['n_valid_cv']} & "
                     f"{100*row['paths_below_1000_spikes']/row['n_paths']:.1f}\\%" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "low_noise_table.tex").write_text("\n".join(lines), encoding="utf-8")
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"网络 & $D_v$ & 原记录 & 较长记录 & 较小步长\\", r"\midrule"]
    for label, dv in (("control", .06), ("higher_order", .12),
                      ("control", 1.), ("higher_order", .30)):
        values = [next(r["mean_cv_finite_only"] for r in rows
                       if r["dataset"] == s and r["label"] == label and r["Dv"] == dv)
                  for s in SOURCES]
        display = "成对" if label == "control" else "单纯形"
        lines.append(f"{display} & {dv:.2f} & " + " & ".join(f"{v:.4f}" for v in values) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "comparison_table.tex").write_text("\n".join(lines), encoding="utf-8")


def plots(rows):
    font_path = font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                         "font.size": 18, "axes.labelsize": 21, "xtick.labelsize": 18,
                         "ytick.labelsize": 18, "legend.fontsize": 16, "axes.linewidth": 1.5,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.major.width": 1.3, "ytick.major.width": 1.3,
                         "xtick.major.size": 6, "ytick.major.size": 6,
                         "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    styles = {
        "original": ("#2166ac", "o", "-", r"$T=15000,\ \Delta t=0.05$"),
        "longer": ("#b2182b", "s", "--", r"$T=20000,\ \Delta t=0.05$"),
        "smaller_dt": ("#1b7837", "^", ":", r"$T=15000,\ \Delta t=0.01$"),
    }
    for label, suffix in (("control", "pairwise"), ("higher_order", "simplicial")):
        fig, ax = plt.subplots(figsize=(6.2, 4.7))
        for dataset, (color, marker, linestyle, legend) in styles.items():
            selected = [r for r in rows if r["dataset"] == dataset and r["label"] == label and r["Dv"] <= .3]
            ax.plot([r["Dv"] for r in selected], [r["mean_cv_finite_only"] for r in selected],
                    color=color, marker=marker, linestyle=linestyle, lw=2, ms=6.2,
                    markerfacecolor="white", markeredgewidth=1.5, label=legend)
        ax.set(xlabel=r"$D_v$", ylabel="CV", xlim=(.028, .312), ylim=(0, 2.8))
        ax.set_xticks([.04, .10, .20, .30], ["0.04", "0.10", "0.20", "0.30"])
        ax.set_yticks([0, .5, 1, 1.5, 2, 2.5])
        ax.minorticks_on()
        ax.tick_params(which="both", top=True, right=True)
        ax.legend(loc="upper right", handlelength=2, labelspacing=.28)
        fig.subplots_adjust(left=.14, right=.975, bottom=.15, top=.975)
        for extension in ("pdf", "eps", "png"):
            fig.savefig(FIG / f"low_noise_{suffix}.{extension}", dpi=600)
        plt.close(fig)
    return font_path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    rows, network_rows, manifest, params, networks = [], [], [], {}, {}
    for name, source in SOURCES.items():
        r, n, p, graph, m = read_suite(name, source)
        rows.extend(r)
        network_rows.extend(n)
        manifest.extend(m)
        params[name], networks[name] = p, graph
        print(f"Verified {name}: {len(r)} Gaussian groups, 4320 trajectories/group", flush=True)
    differences = {}
    for name in ("longer", "smaller_dt"):
        differences[name] = {k: [params['original'][k], v] for k, v in params[name].items()
                             if v != params['original'][k]}
    assert set(differences["longer"]) == {"t_total", "t_warmup", "base_seed"}
    assert set(differences["smaller_dt"]) == {"dt"}
    for n in range(30):
        for key in ("edge_count", "clique_count", "simplex_count"):
            assert networks["original"][n][key] == networks["smaller_dt"][n][key]
    write_csv(OUT / "all_statistics.csv", rows)
    write_csv(OUT / "network_statistics.csv", network_rows)
    tables(rows)
    font_path = plots(rows)
    for filename in ("morris_lecar_paired_alpha_compare_simplicial.py",
                     "morris_lecar_simplicial_paper_array_chunked.py",
                     "morris_lecar_simplicial_dt_convergence_array.py"):
        path = ROOT / filename
        manifest.append({"path": filename, "sha256": digest(path)})
    (OUT / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {"parameters": params, "parameter_differences_from_original": differences,
              "gaussian_rows_recomputed": 3*2*20*4320,
              "input_files_hashed": len(manifest), "font_path": font_path,
              "new_simulations_run": False,
              "longer_is_a_paired_duration_test": False,
              "smaller_dt_has_coupled_brownian_increments": False,
              "counts_are_whole_network_not_per_node": True,
              "raw_csv_has_individual_isi_counts": False,
              "numerical_convergence_established": False}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != 'parameters'}, indent=2), flush=True)


if __name__ == "__main__":
    main()
