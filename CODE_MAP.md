# Result and Code Map

Figure numbers refer to the revised manuscript. Use `figures/render.py` for the
current presentation. The original numerical modules remain at the repository
root so that their imports and command-line interfaces are unchanged.

## From Equations to Statistics

`reproduce.py run-network` selects the saved production parameter set and calls
`run_one_network` in `morris_lecar_simplicial_paper_array_chunked.py`.
That routine constructs a random simplicial network with
`build_simplicial_network`, then calls `simulate_batch` for each alpha,
noise intensity, trial batch and coupling condition. Both functions are in
`morris_lecar_paired_alpha_compare_simplicial.py`.

`simulate_batch` evaluates the ionic currents and gating dynamics at each time
step, adds newly sampled alpha-stable increments, detects upward threshold
crossings after the transient, and computes trajectory-level ISI CV and spike
counts. `aggregate_rows` in `morris_lecar_simplicial_paper_suite.py` computes
ensemble statistics. `merge_experiment` writes the combined numerical tables
and noise-grid optimum summaries. These calculations do not read
`reference_data/`, plot coordinates, or manuscript files.

The simulation and analysis columns below identify numerical routines, whereas
the renderer and bundled-input columns describe a separate plotting stage.
The long-record source of Fig. 3(e,f) was recovered and verified on 2026-10-04;
its numerical table is now used directly, like the other panels.

## Paper Figures

| Result | Simulation / analysis | Revised renderer | Bundled inputs in `reference_data/` |
| --- | --- | --- | --- |
| Network construction | `build_simplicial_network` in `morris_lecar_paired_alpha_compare_simplicial.py` | `draw_construction_schematic.py`, via `--figure construction` | `parameters/main.json` is used with N=9 for the schematic only |
| Fig. 2: voltage snapshots and node CVs | `plot_figure1_dynamics_revised.py`; separate illustrative trajectories and production averages | `plot_figure2.py` | `figure2_*.npz`, `figure2_metadata.json` |
| Fig. 3(a-d): CV and spike counts | `morris_lecar_simplicial_paper_array_chunked.py`, `merge_morris_lecar_simplicial_chunks.py` | `plot_figure3.py` | `main_cv.csv`, `figure3.csv`, `figure3_metadata.json` |
| Fig. 3(e-f): correlation time | `morris_lecar_voltage_coherence_true.py` -> `aggregate_true_correlation_time.py` -> `analyze_true_correlation_time_debiased.py` -> `checks/prepare_figure3_numerical.py` | `plot_figure3.py` | `correlation_time_debiased.csv`: `floor_db_50000_ms` and `floor_db_jackknife_se_ms`, converted from ms to s |
| Fig. 4: strength-matched pairwise control | `morris_lecar_simplicial_matched_coupling_array.py`, `merge_plot_matched_coupling_control.py`, `checks/analyze_comment2.py` | `plot_figure4.py` | `matched_curves.csv`, `matched_summary.json`, `matched_network_statistics.csv` and companion tables |
| Fig. 5: coupling-strength sweep | sigma2 jobs in `morris_lecar_simplicial_paper_array.py`, chunked driver and merger | `plot_figure5.py` | `optimal_sigma2.csv` |
| Fig. 6: sigma2-Dv phase diagram | `morris_lecar_simplicial_sigma2_noise_phase_array.py`, `plot_sigma2_noise_phase_diagram.py` | `plot_figure6.py` | `phase_diagram.npz` |
| Fig. 7: N, filling and edge-probability sweeps | `morris_lecar_simplicial_prl_supplement_array.py`, `merge_prl_supplement_chunks.py` | `plot_figure7.py` | `optimal_structure.csv` |
| Fig. 8: alpha sweep | Same supplement driver and merger | `plot_figure8.py` | `optimal_alpha.csv` |
| Fig. 9: activation and excursion durations | `morris_lecar_simplicial_mechanism_supplement_array.py`, `plot_mechanism_supplement.py` | `plot_figure9.py` | `activation_excursion.csv` |
| Fig. 10: variance contributions | Same mechanism driver and analysis | `plot_figure10.py` | `variance_fractions.csv` |
| Fig. 11: spike-train correlations | Same mechanism driver and analysis | `plot_figure11.py` | `spike_correlations.csv` |

`figures/render.py --figure all` renders the schematic and all 40 numbered
panels. The copied plotting modules retain their original archive-oriented
`main()` functions. The portable entry point calls their drawing functions with
the bundled data; running the copied modules directly requires the original
research archive layout.

## Numerical Checks

| Calculation | Code | Inputs / command |
| --- | --- | --- |
| Exact weighted-edge equivalence | `checks/verify_equivalence.py`, `figures/plot_equivalence.py` | `python checks/run.py equivalence`; bundled parameters and metadata for all 30 networks |
| Matched-control bootstrap | `checks/analyze_comment2.py` | `python checks/run.py matched --archive /path/to/archive` |
| Low-noise activity and duration/timestep comparisons | `checks/analyze_peak.py`, `checks/count_sensitivity.py` | `python checks/run.py peak --archive /path/to/archive` |
| Smaller-timestep production calculation | `morris_lecar_simplicial_dt_convergence_array.py`, `plot_dt_convergence_control.py` | Full driver commands in REPRODUCIBILITY.md |
| Additional integration diagnostics | `validate_euler_dt_stability.py`, `validate_random_dt_convergence.py` | Original command-line interfaces (`--help`) |
| Long-record correlation-time analysis | `checks/run.py correlation`, original aggregation and floor-correction modules | `--archive /path/to/network_acfs`; 2400 combined network NPZ files, not voltage-trace tables |
| Deterministic working point | `morris_lecar_hopf_distance_analysis.py` | `python morris_lecar_hopf_distance_analysis.py --output-dir results/bifurcation` |
| Current-distance scan | `morris_lecar_current_bifurcation_distance_array.py`, `aggregate_current_bifurcation_distance.py`, `plot_current_bifurcation_distance.py` | Original command-line interfaces (`--help`) |
| Additional pairwise-strength controls | `morris_lecar_simplicial_pairwise_strong_control_array.py`, `morris_lecar_simplicial_sigma1_sweep_array.py`, `plot_pairwise_sigma1_control.py`, `plot_sigma1_sweep_figures.py` | Original command-line interfaces (`--help`); not substituted for the matched Fig. 4 control |
| Figure 3 numerical input preparation | `checks/prepare_figure3_numerical.py` | Main CV CSV and corrected correlation-time CSV; no manuscript file required |
| Historical figure-graphics audits | `checks/prepare_figure3.py`, `checks/prepare_figure5.py` | Retained for historical provenance and exported-graphics checks; not numerical simulation drivers |

The equivalence entry point reconstructs a small input directory from the saved
parameters and network metadata. It then runs the unchanged verification
routine, including the original voltage-error warning threshold. It does not
select a new network or reduce the integration duration.

## Production Archive Layout

The `matched` and `peak` checks require trajectory-level CSV files. Their input
directories under `--archive` are:

```text
paper_suite_true_simplicial_array_prl_N100_Dv_refined/
  candidate39_main/chunks/network_000/...network_029/
  candidate39_longer_time/chunks/network_000/...network_029/
paper_suite_matched_coupling_control_N100_Dv_refined/
  chunks/network_000/...network_029/
paper_suite_candidate39_dt001_N100_Dv_refined/
  candidate39_dt_0.010/chunks/network_000/...network_029/
```

Each chunk contains `parameters.json`, `network_summary.json` and
`all_realizations.csv`. The matched chunks additionally contain
`matched_parameters.json`; their parent directory contains
`matched_control_comparison_summary.json`. The peak checks also read each
experiment's `aggregated_metrics.csv` and hash the three relevant numerical
source modules in the archive root. Outputs go to a new directory under
`results/`, never to the input archive.

The bundled `matched_*` and `peak_*` CSV files are computed summaries, not a
replacement for the missing trajectory-level tables. The source manifest
records their original locations and hashes.
