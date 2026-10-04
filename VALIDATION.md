# Repository Validation

Validated on 2026-10-03 with Python 3.13.2, NumPy 2.1.3, SciPy 1.15.2,
Matplotlib 3.10.0rc1, NetworkX 3.5, Numba 0.61.0rc2 and SymPy 1.14.0.
These describe the validation environment, not every historical cluster run.

| Check | Outcome |
| --- | --- |
| `python -m unittest discover -s tests -v` | 13 tests passed: model conventions, saved parameters, internal imports, network reconstruction, file integrity and optimum-guide selection |
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

Before 2026-10-04, Figure 3(e,f) was redrawn from recovered figure coordinates
because its source table had not been located. This was resolved by the
long-record archive recovery described below. The full production ensembles
were not rerun, and the complete trajectory archive is not distributed here.
The `matched` and `peak` validations used that separate local archive.

The manuscript-PDF extraction utilities are included for provenance but were
not rerun in this repository validation. Their additional dependency is pypdf.

Figure-annotation update: Figure 3(a,b) uses CV* and Dv* for the optimum
annotations. Figure 4(a,b) has three color-matched vertical dotted lines per
panel, marking the original eligible-grid optima. Existing open markers still
identify incomplete CV samples; no new optimum-marker overlays are added.
Independent PDF checks confirmed unchanged curve values and interval endpoints
and verified all six new guide positions against the saved control results.

## Source-Only Simulation Check (2026-10-04)

All 14 unit tests passed after adding `test_simulation_without_reference_inputs`.
This test copies only the numerical modules and command-line entry point into
an isolated temporary directory. No reference tables, figure inputs or
manuscript files are copied. A subprocess runs `python reproduce.py smoke`,
integrates the stochastic equations, and writes 32 trajectory records and an
aggregated statistics table. Both coupling conditions are present.

This establishes that the main simulation path can calculate new output from
the equations without bundled comparison data. It is a reduced installation
test, not a repeat of the production ensemble, a convergence test, or a
resolution of the correlation-time source mapping described above. The
production source modules and parameter grids were not changed.

## Correlation-Time Source Recovery (2026-10-04)

The long-record production archive was found on the cluster and retrieved.
Its 2400 combined network-level ACF files cover exactly 80 conditions and
30 networks per condition. The original aggregation validates all records'
model parameters, durations and lag grids. An additional audit checks the
12 sampled node indices, all paired trial seeds and the topology counts
against the original network construction. Six numerical source files,
including the model dependencies, match the cluster copies byte for byte.

Both statistical stages were rerun locally over the complete ACF archive:

- The uncorrected summary CSV is byte-identical to the archived version.
- All corrected integral values and both jackknife-error columns are identical
  to the archived values. Only unused envelope-fit diagnostics differ at
  floating-point precision; the complete diagnostic CSV is not byte-identical.
- The 80 corrected means and all error-bar endpoints match the archived
  Figure 3(e,f) to less than 1e-8 s. The residual is graphic-coordinate precision,
  not a difference between numerical estimators.

The active Figure 3 inputs now come exclusively from numerical CSV tables.
Its old coordinate-derived input is no longer used by the repository renderer.
`correlation_recovery_audit.json` records hashes, field-level differences,
sampling checks and the resolved source mapping. The full 4.15 GB ACF archive
is retained locally, not uploaded to the compact repository.

The original SDE ensemble was not reintegrated during this recovery. The
SLURM template and commands reproduce its full duration, sampling and parameter
grid; the plotting source replacement does not change the reported results.

The updated suite passes all 19 tests. The repository command
`python checks/run.py correlation --archive /path/to/network_acfs` was also
run end to end over all 2400 files: both statistical stages completed, all
240 generated Figure 3 input rows equal the shipped numerical rows, and the
two correlation panels were exported successfully. `figures/render.py --figure 3`
exports all six panels with the new numerical inputs. The source figure's six
PDFs pass the vector-value, embedded-font and optimum-guide checks; the changed
panels (e,f) were rendered and visually inspected. The SLURM template passes
`bash -n`; no new cluster job was submitted during this recovery.
