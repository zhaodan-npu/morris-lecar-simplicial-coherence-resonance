import ast
import csv
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "figures"), str(ROOT / "checks")]
import render


class ReleaseTests(unittest.TestCase):
    def test_internal_imports_present(self):
        for directory in (ROOT, ROOT / "figures", ROOT / "checks"):
            for path in directory.glob("*.py"):
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                    names = [node.module] if isinstance(node, ast.ImportFrom) else (
                        [n.name for n in node.names] if isinstance(node, ast.Import) else [])
                    for name in names:
                        if name and name.startswith(("morris_lecar", "plot_", "merge_", "analyze_", "aggregate_")):
                            self.assertIsNotNone(importlib.util.find_spec(name), (path.name, name))

    def test_reference_integrity(self):
        render.verify_inputs()

    def test_all_plot_inputs_present(self):
        for name in render.TABLES.values():
            self.assertTrue(render.read_table(name), name)
        with np.load(ROOT / "reference_data" / "phase_diagram.npz", allow_pickle=False) as data:
            self.assertEqual(data["sigma"].size, 17)
            self.assertEqual(data["dv"].size, 20)
        metadata = json.loads((ROOT / "reference_data" / "figure2_metadata.json").read_text())
        self.assertEqual(len(metadata["cases"]), 4)
        self.assertTrue(json.loads((ROOT / "reference_data" / "figure3_metadata.json").read_text())["tcor_csv_not_available"])

    def test_network_metadata(self):
        import morris_lecar_paired_alpha_compare_simplicial as base
        p = base.PairedSimplicialParams(**json.loads((ROOT / "parameters" / "main.json").read_text()))
        records = json.loads((ROOT / "reference_data" / "network_metadata.json").read_text())
        self.assertEqual(len(records), 30)
        for index, row in enumerate(records):
            net = base.build_simplicial_network(p, np.random.default_rng(p.base_seed + base.stable_seed("paper_network", index)))
            for key in ("edge_count", "clique_count", "simplex_count"):
                self.assertEqual(getattr(net, key), row[key])

    def test_figure2_production_averages(self):
        with (ROOT / "reference_data" / "main_cv.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        metadata = json.loads((ROOT / "reference_data" / "figure2_metadata.json").read_text())
        for record in metadata["cases"]:
            case = record["case"]
            row = next(r for r in rows if float(r["alpha"]) == case["alpha"] and
                       float(r["noise_intensity_dv"]) == case["dv"] and r["label"] == "higher_order")
            self.assertAlmostEqual(float(row["isi_cv_mean"]), record["scalars"]["ensemble_cv"])
            self.assertAlmostEqual(float(row["spike_count_total_mean"]), record["scalars"]["ensemble_spikes"])

    def test_figure4_optima_match_saved_results(self):
        from plot_figure4 import minimum_row
        rows = render.read_table("matched_curves.csv")
        expected = json.loads((ROOT / "reference_data" / "matched_summary.json").read_text())["best"]
        for record in expected:
            group = [r for r in rows if r["alpha"] == record["alpha"] and r["label"] == record["label"]]
            best = minimum_row(group)
            self.assertEqual(best["Dv"], record["Dv_star"])
            self.assertAlmostEqual(best["CV_mean"], record["CV_star"], places=12)

    def test_figure4_minimum_excludes_ineligible_and_nonfinite_values(self):
        from plot_figure4 import minimum_row
        good = {"eligible_for_minimum": True, "CV_mean": .3}
        invalid = [{"eligible_for_minimum": False, "CV_mean": .01},
                   {"eligible_for_minimum": True, "CV_mean": float("nan")}]
        self.assertIs(minimum_row(invalid + [good]), good)
        with self.assertRaises(ValueError):
            minimum_row(invalid)


if __name__ == "__main__":
    unittest.main()
