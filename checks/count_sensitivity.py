"""Descriptive spike-count conditioning, not a CV bias correction."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "analysis"
SOURCE = ROOT / "paper_suite_true_simplicial_array_prl_N100_Dv_refined/candidate39_main/chunks"
THRESHOLDS = (0, 100, 1000)


def main():
    provenance = json.loads((OUT / "input_manifest.json").read_text())
    hashes = {str(Path(r["path"])): r["sha256"] for r in provenance}
    with (OUT / "all_statistics.csv").open(encoding="utf-8", newline="") as f:
        original = {(r["label"], float(r["Dv"])): r for r in csv.DictReader(f)
                    if r["dataset"] == "original"}
    groups, seen, inputs = {}, set(), []
    for network in range(30):
        path = SOURCE / f"network_{network:03d}/all_realizations.csv"
        relative = str(path.relative_to(ROOT))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == hashes[relative]
        inputs.append({"path": relative, "sha256": digest})
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if float(row["alpha"]) != 2.0:
                    continue
                key = (row["label"], float(row["noise_intensity_dv"]))
                ix = (*key, network, int(row["trial_idx"]), int(row["path_idx"]))
                assert ix not in seen and int(row["network_idx"]) == network
                seen.add(ix)
                cv = float(row["isi_cv"]) if row["isi_cv"] not in ("", "None") else np.nan
                groups.setdefault(key, []).append((cv, float(row["spike_count_total"])))
    assert len(groups) == 40 and len(seen) == 172800
    results = []
    for (label, dv), values in sorted(groups.items()):
        values = np.asarray(values)
        cv, spikes = values[:, 0], values[:, 1]
        finite = np.isfinite(cv)
        assert len(values) == 4320
        prior = original[label, dv]
        assert int(finite.sum()) == int(prior["n_valid_cv"])
        assert abs(float(cv[finite].mean()) - float(prior["mean_cv_finite_only"])) < 1e-12
        assert abs(float(spikes.mean()) - float(prior["mean_spikes_all_paths"])) < 1e-12
        for threshold in THRESHOLDS:
            keep = finite & (spikes >= threshold)
            results.append({"alpha": 2.0, "label": label, "Dv": dv,
                            "minimum_whole_network_spikes": threshold,
                            "total_trajectories": len(values),
                            "retained_trajectories": int(keep.sum()),
                            "retained_fraction": float(keep.mean()),
                            "conditional_mean_cv": float(cv[keep].mean()) if keep.any() else None})
    with (OUT / "count_sensitivity.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"$D_v$ & 原CV均值 & 总spike数$\geq1000$后的CV & 保留轨迹数\\",
             r"\midrule"]
    key_results = []
    for dv in (.12, .16, .2):
        selected = [r for r in results if r["label"] == "higher_order" and r["Dv"] == dv]
        baseline = next(r for r in selected if r["minimum_whole_network_spikes"] == 0)
        conditioned = next(r for r in selected if r["minimum_whole_network_spikes"] == 1000)
        lines.append(f"{dv:.2f} & {baseline['conditional_mean_cv']:.3f} & "
                     f"{conditioned['conditional_mean_cv']:.3f} & "
                     f"{conditioned['retained_trajectories']}/4320" + r"\\")
        key_results.append(conditioned)
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "count_sensitivity_table.tex").write_text("\n".join(lines), encoding="utf-8")
    report = {"inputs": inputs, "thresholds": THRESHOLDS,
              "gaussian_raw_records": len(seen), "groups_checked_against_prior_audit": len(groups),
              "new_simulations_run": False, "filtered_samples_are_different_populations": True,
              "count_threshold_proves_convergence": False,
              "entire_peak_proved_to_be_artifact": False,
              "peak_proved_to_be_genuine_dynamics": False,
              "key_results": key_results,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "count_sensitivity_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
