# Morris-Lecar Simplicial Networks: Coherence Resonance

Numerical code for studying coherence resonance in stochastic Morris-Lecar
networks with pairwise and simplex-derived linear coupling under alpha-stable
Levy noise. This repository concerns the coherence-resonance study only.

This is a pre-publication code snapshot. Repository ownership, the reuse license,
and the manuscript citation must be confirmed before the public release.

## Install and Test

Use Python 3.11 or newer in a virtual environment:

```sh
python -m venv .venv
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python reproduce.py smoke
```

Activate the virtual environment before the second command. The smoke test uses
small, explicitly labeled demonstration parameters. It verifies execution only;
it is not a replacement for the manuscript's production ensemble.

## Entry Points

```sh
python reproduce.py list
python reproduce.py parameters
python reproduce.py run-network --family main --experiment candidate39_main --network 0
```

The final command runs one full-size production network and can be expensive.
For the main experiment, run network indices 0--29. The production defaults are
not shortened by this entry point. Every run records its actual parameters.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for equations, ensemble semantics,
noise grids, full experiment commands, and known reproduction boundaries.

## Contents

- `morris_lecar_paired_alpha_compare_simplicial.py`: topology construction,
  stochastic integration, paired noise, spike detection, and pooled-ISI CV.
- `morris_lecar_simplicial_*`: production, robustness, phase-diagram, mechanism,
  timestep, and coupling-strength control drivers.
- `morris_lecar_voltage_coherence*.py`: voltage correlation/spectral estimators.
- `morris_lecar_hopf_distance_analysis.py`: deterministic working-point analysis.
- `merge_*.py`, `aggregate_*.py`, `analyze_*.py`, `plot_*.py`: original analysis
  utilities; their dataset arguments and historical output names are retained.
- `parameters/`: parameter files copied from actual saved production runs.
- `reference_data/`: selected numerical outputs, not the full trajectory archive.
- `SOURCE_MANIFEST.json`: SHA-256 provenance for the unmodified source snapshot.
- `slurm/run_main.sh`: a generic array example with no institutional credentials.

The original numerical source files are copied without changing their equations,
random-seed rules, clipping, grid, or estimators. Convenience wrappers and tests
are additional files. Historical standalone defaults in the low-level model are
not the manuscript parameters; use the production entry points above.

## Reproduction Scope

The code executes the documented simulations and analysis pipelines. A full
stochastic rerun has not been performed as part of packaging. The distribution
does not include all raw trajectories or claim a one-command, pixel-identical
regeneration of every final manuscript panel. In particular, the exact final
correlation-time panel's numerical provenance needs reconciliation; the
available estimator variants are documented separately.

No manuscript PDF, reviewer correspondence, cluster credentials, SSH keys, or
subsequent-paper research is included. Do not put access tokens into this repository.

## Citation and License

The final article citation and a tagged release URL will be added before submission
of the code-availability statement. No reuse license has been granted in this
staging snapshot; the authors must approve the license before publication.
