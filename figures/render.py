"""Render the revised paper panels from the bundled, hash-checked inputs."""

import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "figures"), str(ROOT / "checks")]
import numpy as np

DATA = ROOT / "reference_data"
TABLES = {3: "figure3.csv", 4: "matched_curves.csv", 5: "optimal_sigma2.csv",
          7: "optimal_structure.csv", 8: "optimal_alpha.csv", 9: "activation_excursion.csv",
          10: "variance_fractions.csv", 11: "spike_correlations.csv"}


def verify_inputs():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    for entry in manifest["files"]:
        if entry["file"].startswith("reference_data/"):
            path = ROOT / entry["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Reference input changed: {entry['file']}")


def read_table(name):
    def value(text):
        if text in ("True", "False"):
            return text == "True"
        try:
            return float(text)
        except ValueError:
            return text
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return [{key: value(item) for key, item in row.items()} for row in csv.DictReader(handle)]


def render(number, output):
    output.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module(f"plot_figure{number}")
    module.HERE = output
    style = getattr(module, "style", None) or getattr(module, "apply_style")
    style()
    if number == 2:
        metadata = json.loads((DATA / "figure2_metadata.json").read_text())
        records = []
        for record in metadata["cases"]:
            case = module.original.Case(**record["case"])
            with np.load(DATA / f"figure2_{case.letter}.npz", allow_pickle=False) as arrays:
                item = dict(arrays)
            item.update(record["scalars"], case=case)
            records.append(module.draw(item))
        return records
    if number == 6:
        with np.load(DATA / "phase_diagram.npz", allow_pickle=False) as arrays:
            return [module.draw(spec, arrays[f"{spec[0]}_values"], arrays["sigma"],
                                arrays["dv"], arrays[f"{spec[0]}_ridge_dv"], {})
                    for spec in module.SPECS]
    rows = read_table(TABLES[number])
    if number == 3:
        metadata = json.loads((DATA / "figure3_metadata.json").read_text())
        print("Figure 3(e,f): recovered PDF coordinates, not identified raw simulation output.")
        return [module.draw(spec, rows) for spec in metadata["panels"]]
    if number == 4:
        return [module.draw(letter, alpha, rows) for letter, alpha in (("a", 1.5), ("b", 2.))]
    if number == 5:
        return [module.draw(letter, rows) for letter in "ab"]
    if number == 8:
        return [module.draw(letter, metric, rows) for letter, metric in (("a", "cv"), ("b", "Dv"))]
    if number == 11:
        module.plt.rcParams["mathtext.fallback"] = "stix"
    return [module.draw(spec, rows) for spec in module.SPECS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=["all", "construction"] + [str(n) for n in range(2, 12)], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "figures")
    args = parser.parse_args()
    verify_inputs()
    args.output = args.output.resolve()
    if args.figure in ("all", "construction"):
        import draw_construction_schematic as schematic
        schematic.ROOT = ROOT
        schematic.PARAMS = ROOT / "parameters" / "main.json"
        schematic.FIG = args.output / "construction"
        schematic.OUT = schematic.FIG / "analysis"
        schematic.main()
    selected = range(2, 12) if args.figure == "all" else ([] if args.figure == "construction" else [int(args.figure)])
    for number in selected:
        destination = args.output / f"figure{number}"
        records = render(number, destination)
        (destination / "render_report.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Figure {number}: {len(records)} panels -> {destination}", flush=True)
    verify_inputs()


if __name__ == "__main__":
    main()
