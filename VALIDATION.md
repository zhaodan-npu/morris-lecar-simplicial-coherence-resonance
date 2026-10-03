# Repository Validation

Validated on 2026-10-03 with Python 3.13.2, NumPy 2.1.3, SciPy 1.15.2,
Matplotlib 3.10.0rc1, NetworkX 3.5, Numba 0.61.0rc2 and SymPy 1.14.0.
These describe the validation environment, not every historical cluster run.

| Check | Outcome |
| --- | --- |
| `python -m unittest discover -s tests -v` | 11 tests passed: model conventions, saved parameters, internal imports, network reconstruction and file integrity |
| `python reproduce.py smoke --output results/release_validation` | 32 trajectory records; reduced installation test only |
| `python figures/render.py --figure all` | 40 numbered panels and 3 construction panels exported; data and layout assertions passed |
| `python checks/run.py equivalence` | All 30 integer operators agree exactly; independent full-duration integrations retain identical detected spike steps for all 100 nodes in both cases; 4 diagnostic panels exported |
| `python checks/run.py matched --archive /path/to/archive` | Reanalysed 518400 trajectory records; six regenerated CSV tables are byte-identical to the bundled reference tables |
| `python checks/run.py peak --archive /path/to/archive` | Reanalysed 518400 Gaussian records across three datasets, plus count conditioning of 172800 original records; four regenerated CSV tables are byte-identical to the bundled references |
| Source audit | Copied numerical/plotting modules agree with their source SHA-256 hashes; Python syntax and credential/private-path scans passed |

The equivalence check retains a warning for the Gaussian case: its maximum
accumulated voltage difference is approximately 6.052e-6 mV, above the original
1e-6 diagnostic threshold. The non-Gaussian case gives approximately 3.933e-9
mV. This is not a claim of bitwise voltage equality or timestep convergence.

Figure 3(e,f) was redrawn from recovered PDF coordinates. Its matching raw
correlation-time table remains unresolved. The full production ensembles were
not rerun, and the complete trajectory archive is not distributed here.
The `matched` and `peak` validations used that separate local archive.

The manuscript-PDF extraction utilities are included for provenance but were
not rerun in this repository validation. Their additional dependency is pypdf.
