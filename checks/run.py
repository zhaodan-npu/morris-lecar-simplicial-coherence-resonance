"""Run coupling-equivalence checks or reanalyse a full production archive."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "checks"), str(ROOT / "figures")]


def equivalence(output):
    import verify_equivalence as check
    from render import verify_inputs
    verify_inputs()
    archive = output / "inputs"
    main = archive / "main"
    metadata = json.loads((ROOT / "reference_data" / "network_metadata.json").read_text())
    for index, record in enumerate(metadata):
        chunk = main / "chunks" / f"network_{index:03d}"
        chunk.mkdir(parents=True)
        shutil.copy2(ROOT / "parameters" / "main.json", chunk / "parameters.json")
        (chunk / "network_summary.json").write_text(json.dumps([record]), encoding="utf-8")
    for name in ("morris_lecar_paired_alpha_compare_simplicial.py", "morris_lecar_simplicial_paper_array_chunked.py"):
        shutil.copy2(ROOT / name, archive / name)
    check.ROOT, check.MAIN, check.OUT = archive, main, output / "analysis"
    check.main()
    import plot_equivalence as plot
    plot.HERE, plot.OUT, plot.FIG = output, check.OUT, output / "figures"
    plot.main()


def archive_analysis(kind, archive, output):
    if not archive.is_dir():
        raise FileNotFoundError(archive)
    main = archive / "paper_suite_true_simplicial_array_prl_N100_Dv_refined" / "candidate39_main"
    if kind == "matched":
        import analyze_comment2 as check
        check.ROOT, check.MAIN = archive, main
        check.MATCH = archive / "paper_suite_matched_coupling_control_N100_Dv_refined"
        check.HERE, check.OUT, check.FIG = output, output / "analysis", output / "figures"
        check.OUT.mkdir()
        params, grid, cvs, spikes, matching, provenance = check.read_raw()
        check.analyze(params, grid, cvs, spikes, matching, provenance)
        import redraw_comment2_cv as reader
        import plot_figure4 as figure
        reader.OUT = check.OUT
        figure.HERE = check.FIG
        figure.HERE.mkdir()
        figure.style()
        rows = reader.read_data()
        for letter, alpha in (("a", 1.5), ("b", 2.)):
            figure.draw(letter, alpha, rows)
    else:
        import analyze_peak as check
        check.ROOT = archive
        check.SOURCES = {
            "original": main,
            "longer": main.parent / "candidate39_longer_time",
            "smaller_dt": archive / "paper_suite_candidate39_dt001_N100_Dv_refined" / "candidate39_dt_0.010",
        }
        check.OUT, check.FIG = output / "analysis", output / "figures"
        check.main()
        import count_sensitivity as counts
        counts.ROOT, counts.SOURCE, counts.OUT = archive, main / "chunks", check.OUT
        counts.main()


def correlation_analysis(archive, output, main_csv):
    for script, subdir in (("aggregate_true_correlation_time.py", "summary"),
                           ("analyze_true_correlation_time_debiased.py", "debiased")):
        subprocess.run([sys.executable, str(ROOT / script), str(archive),
                        str(output / subdir)], check=True)
    from prepare_figure3_numerical import build_rows, prepare, read_rows
    correlation_csv = output / "debiased" / "true_correlation_time_debiased.csv"
    # Keep numerical inputs together so the provenance remains relative and portable.
    local_main = output / "main_cv.csv"
    shutil.copy2(main_csv, local_main)
    prepare(local_main, correlation_csv, output / "figure3_inputs", output)
    rows, panels = build_rows(read_rows(local_main), read_rows(correlation_csv))
    import plot_figure3 as plot
    plot.HERE = output / "figures"
    plot.HERE.mkdir()
    plot.apply_style()
    for spec in panels:
        if spec["metric"] == "tcor":
            plot.draw(spec, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=("equivalence", "matched", "peak", "correlation"))
    parser.add_argument("--archive", type=Path, help="Production archive with the directory names listed in CODE_MAP.md.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--main-csv", type=Path, default=ROOT / "reference_data" / "main_cv.csv",
                        help="Main CV summary used to locate the CV-optimal guide lines for correlation panels.")
    args = parser.parse_args()
    if args.check != "equivalence" and args.archive is None:
        parser.error("This analysis requires --archive; full trajectory/ACF archives are not bundled.")
    output = (args.output or ROOT / "results" / args.check).resolve()
    if output.exists():
        parser.error("Output already exists; select a new --output directory.")
    output.mkdir(parents=True)
    if args.check == "equivalence":
        equivalence(output)
    elif args.check == "correlation":
        correlation_analysis(args.archive.resolve(), output, args.main_csv.resolve())
    else:
        archive_analysis(args.check, args.archive.resolve(), output)


if __name__ == "__main__":
    main()
