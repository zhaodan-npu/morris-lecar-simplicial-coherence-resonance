# Reproducibility Notes

## Model and Numerical Conventions

For each node, `C dV = (I - I_Ca - I_K - I_L + coupling) dt + s dL_alpha`
and `dw/dt = phi (w_inf(V)-w)/tau_w(V)`. The voltage noise intensity is
`D_v = (s/C)^alpha`, so the voltage increment is `(D_v*dt)^(1/alpha) Z`.
For alpha=2 the sampler uses `Z=sqrt(2)*N(0,1)`, giving increment variance
`2*D_v*dt`. The historical parameter field `noise_scales` stores D_v values,
not the current-noise scale s.

An undirected Erdos-Renyi graph is sampled; each existing 3-clique is filled
independently with probability p_fill. Unfilled cliques are not 2-simplices.
For node i each filled triangle contributes `sigma_2*((V_j+V_k)/2-V_i)`.
The implementation is linear: it is exactly representable by a weighted
pairwise operator with edge weights `sigma_1*A1_ij + sigma_2*A2_ij/2`, where
A2 counts filled-triangle multiplicities. Simplicial construction does not
make this particular coupling function dynamically irreducible.

The integrator is explicit Euler with alpha-stable increments. The production
implementation clips V to [-200,200] and w to [-0.5,1.5], and clips the cosh
argument and bounds the gating denominator as shown in the source. These
numerical safeguards are retained, not removed during release preparation.

## Parameters

All units follow the model and manuscript: time in ms and voltage in mV.

| Quantity | Production value |
| --- | --- |
| C; g_Ca, g_K, g_L | 20; 4.4, 8, 2 |
| V_Ca, V_K, V_L | 120, -84, -60 |
| V1, V2, V3, V4; phi | -1.2, 18, 2, 30; 0.04 |
| N; I | 100; 88 |
| p_edge; p_fill | 0.15; 0.5 |
| sigma_1; sigma_2 (main condition) | 0.002; 0.12 |
| alpha; beta | {1.5,2.0}; 0 |
| dt; T; discarded transient | 0.05; 15000; 3000 |
| Networks; batches/network; paths/batch | 30; 12; 12 |
| Spike threshold | upward crossing of 0 mV |

The main D_v grid is:
`0.04, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.56, 0.75,
1.00, 1.25, 1.56, 1.89, 2.25, 3.06, 4.00, 6.25, 9.00`.

`parameters/main.json` is copied from a saved production chunk. Its base seed
20299501 includes the candidate offset added to the suite seed 20260501.
The wrapper's full main parameters are tested against this saved file.
`python reproduce.py parameters --family supplement --experiment size_N100`
prints the corresponding structure-scan parameters: 20 networks, 10 batches
of 10 paths, T=10000, transient=2000. Mechanism runs use 30 networks,
4 batches of 4 paths, T=20000 and transient=5000; see `parameters/mechanism.json`.

## Seeds and Ensemble Size

Each main condition contains `30*12*12=4320` trajectories, not `30*12`.
The parameter `n_trials_per_network` is used for both the outer trial loop
and the number of paths in a vectorized batch. The function
`total_realizations_per_condition` in the historical dataclass does not account
for this nested batching; use the saved row counts and the expression above.

Network, initial-state, and noise seeds are deterministically derived by
`stable_seed` in the source. Pairwise and simplex-enabled conditions use the
same graph, filled triangles, initial states, and noise draws. Changing the
ordering of the noise grid changes noise seeds because the noise index enters
the seed. Cross-platform floating-point and library differences can accumulate;
statistical reproducibility is not a guarantee of bitwise trajectory identity.

## Statistics

The ISIs are differences between successive spikes of the same neuron after
the transient. They are pooled across nodes within each trajectory. A trajectory
CV is computed when at least three intervals exist, using population standard
deviation (`ddof=0`) and the source's `mean+1e-12` denominator. Finite trajectory
CVs are then averaged. This is neither the mean of node CVs nor a CV computed
after pooling intervals across all trajectories. Missing CVs are not zeros.

CV* is a minimum over the retained discrete noise grid of the ensemble mean
CV. Historical summaries use a minimum mean total spike count of 5; main
figure optimum markers use 1000 and phase-diagram masking uses 100. These
are distinct analysis settings. Counts refer to the full network per trajectory,
not each neuron.

Stored `isi_cv_std` is the across-trajectory sample standard deviation. The
original main CV plot used `std/sqrt(total_realizations)`, not a 95% bootstrap
interval. The later coupling-control analysis has a different bootstrap
procedure: 20000 paired-network bootstrap resamples, with pointwise percentile
95% intervals. The same resampled networks are used across conditions and noise
points; the eligibility rule is reapplied when selecting bootstrap minima.

## Full Experiment Commands

Run from this repository, with the installed virtual environment active.
No command below submits jobs automatically except the explicit sbatch example.

Main candidate: run each network 0--29, then merge:

```sh
python reproduce.py run-network --experiment candidate39_main --network 0
python reproduce.py merge --experiment candidate39_main --min-spikes 5
```

Repeat the first command with each network index before merging. The wrapper
refuses to merge missing networks, inconsistent parameters, or incomplete or
duplicated records. For other jobs use `python reproduce.py list` and specify
`--family` and `--experiment`. All main/sigma2 jobs use 30 networks; supplement
jobs use 20 networks and their own original seeds.

Original all-main-suite array: task IDs 0--689, 23 experiments x 30 networks:

```sh
python morris_lecar_simplicial_paper_array_chunked.py 0 690 --output-root results/main_suite
python merge_morris_lecar_simplicial_chunks.py --output-root results/main_suite --min-spikes 5
```

The first command is one task only. Repeat all 690 IDs before the merge.
Unlike the new wrapper, the original merger does not enforce complete coverage.
Its full-suite output includes `array_jobs.json` and `merged_summary.json`.

Additional one-task examples (repeat the indicated ID range):

| Calculation | Example command | IDs |
| --- | --- | --- |
| Structural/alpha scans | `python morris_lecar_simplicial_prl_supplement_array.py 0 260 --output-root results/supplement` | 0--259 |
| sigma2-Dv phase diagram | `python morris_lecar_simplicial_sigma2_noise_phase_array.py 0 510 --output-root results/phase` | 0--509 |
| Matched coupling control | `python morris_lecar_simplicial_matched_coupling_array.py 0 30 --output-root results/matched` | 0--29 |
| Phase duration and synchrony | `python morris_lecar_simplicial_mechanism_supplement_array.py 0 30 --output-root results/mechanism` | 0--29 |
| dt=0.01 check | `python morris_lecar_simplicial_dt_convergence_array.py 0 30 --output-root results/dt_check` | 0--29 |

Analysis tools expose their path arguments with `--help`. Root-level plotting
scripts retain historical numbering and layout; `figures/render.py` renders the
revised figures. For example:

```sh
python merge_prl_supplement_chunks.py --output-root results/supplement --min-spikes 5
python plot_sigma2_noise_phase_diagram.py --help
python plot_mechanism_supplement.py --help
python merge_plot_matched_coupling_control.py --help
python morris_lecar_hopf_distance_analysis.py --output-dir results/bifurcation
```

## SLURM

Create `logs/` before submitting, activate the environment on the cluster, and
adapt the resource limits to your site. The supplied 24-hour/4-GB values are a
generic template, not a measured runtime guarantee. Add your own allocation,
partition, and QoS flags to sbatch; none are embedded in the public code:

```sh
mkdir -p logs
sbatch slurm/run_main.sh
```

## Figure and Data Boundaries

Current figure numbering is used below; older script names may differ.

| Figure | Numerical source / shipped reference |
| --- | --- |
| 2 | `plot_figure1_dynamics_revised.py`, `plot_alpha_noise_cv_panel_like_reference.py`; short illustrations are separate from production statistics |
| 3(a-d) | main production driver; `reference_data/main_cv.csv` |
| 3(e-f) | voltage-correlation estimator variants; recovered coordinates in `figure3.csv`, matching raw output not identified |
| 4 | matched-coupling driver; `checks/analyze_comment2.py`; `matched_*.csv` summaries |
| 5 | sigma2 jobs; `reference_data/optimal_sigma2.csv` |
| 6 | phase driver; `reference_data/phase_diagram.npz` |
| 7--8 | supplement driver; `optimal_structure.csv`, `optimal_alpha.csv` |
| 9--11 | mechanism driver; `activation_excursion.csv`, `variance_fractions.csv`, `spike_correlations.csv` |

The two voltage-correlation modules and the debiasing analysis implement distinct
estimators. Figure 3(e,f) was recovered from PDF vector paths because an exact
matching source table had not been found. Those coordinates are bundled solely
for redrawing, with the provenance flag in `figure3_metadata.json` and a warning
from the renderer. They are not raw simulation output. The original extraction
procedure is retained in `checks/prepare_figure3.py`.

See CODE_MAP.md for the complete result-to-code map and required archive inputs.
The portable equivalence check uses the saved parameters and metadata for all
30 original networks. It runs both full-duration diagnostic cases and preserves
the original 1e-6 voltage-error warning; algebraic equivalence is not inferred
from a floating-point tolerance. The low-activity checks do not establish
timestep convergence or prove that the entire low-noise peak is an artifact.

## Packaging Validation

The local preparation environment was Python 3.13.2, NumPy 2.1.3, SciPy 1.15.2,
and Matplotlib 3.10.0rc1. These are packaging/test versions, not a reconstruction
of every historical cluster environment. `requirements.txt` specifies install
ranges; not every version combination has been tested.

Unit tests check source hashes, noise normalization, simplex-to-edge projection,
matched coupling, paired zero-simplex behavior, and saved production parameters.
The smoke run exercises simulation and aggregation at a clearly reduced scale.
These tests do not establish full-ensemble convergence or validate every estimator.
