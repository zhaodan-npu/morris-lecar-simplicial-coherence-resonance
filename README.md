# Coherence Resonance in Simplicial Morris-Lecar Networks

Numerical simulation and statistical analysis of Morris-Lecar networks driven
by alpha-stable Levy noise. The calculation starts from the model equations:
generate a network, integrate the stochastic dynamics, detect spikes, compute
interspike-interval statistics, and aggregate independent trajectories.

The simulation drivers generate their own numerical output. They do not require
the bundled reference data or any manuscript figures as input.

## Numerical Workflow

| Stage | Implementation |
| --- | --- |
| Erdos-Renyi graph, 3-cliques and filled 2-simplices | `build_simplicial_network` in `morris_lecar_paired_alpha_compare_simplicial.py` |
| Ionic currents, gating dynamics and alpha-stable noise increments | `m_inf`, `w_inf`, `tau_w`, `sample_alpha_stable` and `levy_increment` in the same module |
| Time integration, transient removal, spike detection and trajectory-level CV | `simulate_batch` in the same module |
| Paired simulations and saved trajectory statistics | `morris_lecar_simplicial_paper_array_chunked.py` |
| Ensemble averages and discrete-noise-grid minima | `morris_lecar_simplicial_paper_suite.py`, `merge_morris_lecar_simplicial_chunks.py` |
| Additional parameter scans and control experiments | Drivers listed below and in [CODE_MAP.md](CODE_MAP.md) |

Within each trajectory, intervals between successive spikes of the same neuron
are pooled across nodes. A CV is evaluated when at least three intervals are
available; otherwise it is recorded as missing. Finite trajectory-level CVs
are then averaged. The linear simplex coupling has an exact weighted pairwise
representation determined by filled-triangle multiplicities.

## Installation

Use Python 3.11 or later. Create and activate a virtual environment, then run:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Run the Main Experiment

```sh
python reproduce.py list
python reproduce.py parameters
python reproduce.py run-network --experiment candidate39_main --network 0
```

The last command integrates one network using the production parameters.
Run each network index from 0 through 29, then aggregate the complete ensemble:

```sh
python reproduce.py merge --experiment candidate39_main --min-spikes 5
```

The wrapper refuses an incomplete merge. Per-network output includes
`parameters.json`, `network_summary.json` and `all_realizations.csv`/`.json`.
The latter records computed CV, spike count and mean-voltage fluctuation
statistics for each trajectory. The merge writes `aggregated_metrics.csv`
and `summary.json` under `results/main/candidate39_main/`.

The main experiment uses N=100, I=88, sigma_1=0.002, sigma_2=0.12,
p_edge=0.15, p_fill=0.5, alpha in {1.5, 2.0}, dt=0.05 ms,
T=15000 ms and a 3000 ms transient. Each condition has 30 networks,
12 batches per network and 12 paths per batch, giving 4320 trajectories.
The full noise grid, ionic parameters and seeds are specified in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) and `parameters/main.json`.
The `--min-spikes 5` option retains the original summary selection rule;
the main figure optimum markers use the separate threshold documented there.

For a short installation check only: `python reproduce.py smoke`.
This check reduces the network size, duration and ensemble; it is not a
substitute for the production calculation.

## Parameter Scans and Controls

| Calculation | Numerical driver |
| --- | --- |
| Main comparison and coupling-strength scan | `morris_lecar_simplicial_paper_array_chunked.py` |
| Network size, triangle filling, edge probability and alpha scans | `morris_lecar_simplicial_prl_supplement_array.py` |
| Coupling-strength/noise phase diagram | `morris_lecar_simplicial_sigma2_noise_phase_array.py` |
| Strength-matched pairwise control | `morris_lecar_simplicial_matched_coupling_array.py` |
| Activation/excursion durations, variance contributions and spike correlations | `morris_lecar_simplicial_mechanism_supplement_array.py` |
| Smaller-time-step calculation | `morris_lecar_simplicial_dt_convergence_array.py` |
| Long-record voltage correlation time | `morris_lecar_voltage_coherence_true.py`, `aggregate_true_correlation_time.py`, `analyze_true_correlation_time_debiased.py` |
| Deterministic working point | `morris_lecar_hopf_distance_analysis.py` |

[REPRODUCIBILITY.md](REPRODUCIBILITY.md#full-experiment-commands) gives
the task ranges, commands and aggregation steps. `slurm/run_main.sh` is a
submission template; cluster account and partition settings are supplied by
the user. `python checks/run.py equivalence` independently compares the
triangle-sum and weighted-edge implementations.

The correlation-time calculation uses 120000 ms trajectories and the same
3000 ms transient. Its full sampling and postprocessing protocol is recorded
in `parameters/correlation_time.json` and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md#voltage-correlation-time).
`slurm/run_correlation.sh` runs the full-duration calculation; it is not the
short-record default of the historical solver's command-line interface.

## Repository Structure

| Location | Contents |
| --- | --- |
| Root numerical modules | Model, full simulation drivers, mergers and original analyses |
| `figures/` | Revised figure routines and the portable rendering entry point |
| `checks/` | Coupling equivalence, matched-control statistics and low-activity diagnostics |
| `parameters/` | Parameter records from saved production runs |
| `reference_data/` | Saved comparison inputs and network metadata; not required by the simulation drivers |
| `tests/` | Model, parameter, import and input-integrity tests |
| `slurm/` | Cluster-independent submission template |

[CODE_MAP.md](CODE_MAP.md) maps each result to its simulation, analysis and
figure inputs. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) specifies the model,
parameters, estimators, seeds, commands and remaining limitations.
`SOURCE_MANIFEST.json` records file hashes and source provenance.
[VALIDATION.md](VALIDATION.md) records the executed checks and their scope.

## Plotting and Validation Scope

Numerical generation and plotting are separate steps. The root-level analysis
scripts consume simulation output; `figures/render.py` is an optional renderer
of bundled comparison inputs, not a simulation command. See CODE_MAP.md for
each figure's data source. Publication plotting uses Times New Roman.

The numerical source files retain the original model, parameter grids and
estimators. Full trajectory tables are not bundled, and the complete production
ensemble has not been rerun during repository validation. All six Figure 3
panels now use numerical output tables, including the long-record,
noise-floor-corrected correlation times and network-jackknife errors for
panels (e,f). The correspondence with the archived results is checked in tests.
The full network-level autocorrelation archive is not bundled; its regeneration
and analysis commands are provided in REPRODUCIBILITY.md.

## Publication Status

This is a private, pre-publication repository. The article citation and reuse
license have not yet been assigned. Manuscript drafts, reviewer correspondence
and institutional access details are not part of this distribution.
