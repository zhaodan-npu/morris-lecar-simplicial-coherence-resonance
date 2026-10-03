# Coherence Resonance in Simplicial Morris-Lecar Networks

Simulation and analysis code for Morris-Lecar networks driven by alpha-stable
Levy noise. The network is constructed by filling selected triangles of an
Erdos-Renyi graph. The linear simplex coupling used here has an exact weighted
pairwise representation; its weights depend on filled-triangle multiplicities.

## Installation

Use Python 3.11 or later. Create and activate a virtual environment, then run:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Publication figures use Times New Roman. Install that font before rendering;
the plotting routines check its availability instead of silently substituting it.

## Reproduce Figures

```sh
python figures/render.py --figure all
python figures/render.py --figure 7
```

These commands use the bundled reference inputs and export separate PDF and
600-dpi PNG panels under `results/figures/`. They do not rerun the ensemble.
The entry point uses the revised plotting routines, including the corrected
CV notation and external legends. Root-level `plot_*.py` files retain the
earlier analysis layouts and historical figure numbering.

Figure 3(e,f) is an exception to numerical reproducibility: the bundled curves
were recovered from PDF vector coordinates. The matching raw correlation-time
table has not been identified. Redrawing those panels does not validate their
underlying estimator. This distinction is recorded in the figure metadata.

## Run Simulations and Checks

```sh
python reproduce.py list
python reproduce.py parameters
python reproduce.py run-network --experiment candidate39_main --network 0
python checks/run.py equivalence
```

The production command runs one network with the original parameters. Complete
the 30 network indices before merging the main ensemble. The equivalence check
audits all 30 original network structures and compares two independent coupling
implementations on the first network at two specified noise conditions.

For a short installation check only: `python reproduce.py smoke`.

## Repository Structure

| Location | Contents |
| --- | --- |
| Root numerical modules | Model, full simulation drivers, mergers and original analyses |
| `figures/` | Revised figure routines and the portable rendering entry point |
| `checks/` | Coupling equivalence, matched-control statistics and low-activity diagnostics |
| `parameters/` | Parameter records from saved production runs |
| `reference_data/` | Figure inputs, numerical summaries and original network metadata |
| `tests/` | Model, parameter, import and input-integrity tests |
| `slurm/` | Cluster-independent submission template |

[CODE_MAP.md](CODE_MAP.md) maps each result to its simulation, analysis and
figure inputs. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) specifies the model,
parameters, estimators, seeds, commands and remaining limitations.
`SOURCE_MANIFEST.json` records file hashes and source provenance.
[VALIDATION.md](VALIDATION.md) records the executed checks and their scope.

The numerical source files are preserved without changing the model, parameter
grids or estimators. Portable entry points adapt file locations and output
directories. Full trajectory tables are not bundled, and the complete
production ensemble has not been rerun during repository validation.

## Publication Status

This is a private, pre-publication repository. The article citation and reuse
license have not yet been assigned. Manuscript drafts, reviewer correspondence
and institutional access details are not part of this distribution.
