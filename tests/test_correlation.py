import json
from pathlib import Path
import sys
import unittest

import numpy as np

import morris_lecar_voltage_coherence_true as solver
from analyze_true_correlation_time_debiased import (
    debiased_amplitudes, jackknife_se, mean_and_se_from_sums,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))
from prepare_figure3_numerical import build_rows, index_rows, read_rows


class CorrelationTests(unittest.TestCase):
    def test_acf_matches_direct_connected_covariance(self):
        traces = np.array([[0., 1., 3., -1., 2., 0.], [3., 0., 1., 5., 0., 2.]])
        centered = traces - traces.mean(axis=1, keepdims=True)
        expected = np.array([[np.mean(row[:len(row)-lag] * row[lag:]) / np.mean(row**2)
                              for lag in range(4)] for row in centered])
        np.testing.assert_allclose(solver.normalized_acf_batch(traces, 3), expected,
                                   rtol=1e-13, atol=1e-13)

    def test_floor_and_jackknife_definitions(self):
        stack = np.array([[1., .5, -.2], [1., .1, -.1], [1., -.2, .05]])
        mean, se = mean_and_se_from_sums(stack.sum(axis=0), (stack**2).sum(axis=0), 3)
        np.testing.assert_allclose(mean, stack.mean(axis=0))
        np.testing.assert_allclose(se, stack.std(axis=0, ddof=1) / np.sqrt(3))
        floor, corrected, _ = debiased_amplitudes(mean, se)
        np.testing.assert_allclose(floor, np.sqrt(2/np.pi) * se)
        np.testing.assert_allclose(corrected, np.maximum(np.abs(mean)-floor, 0.))
        estimates = np.array([2., 3., 5.])
        self.assertAlmostEqual(jackknife_se(estimates),
                               np.sqrt(2/3 * np.sum((estimates-estimates.mean())**2)))

    def test_all_figure3_values_come_from_numerical_tables(self):
        data = ROOT / "reference_data"
        expected, _ = build_rows(read_rows(data / "main_cv.csv"),
                                 read_rows(data / "correlation_time_debiased.csv"))
        actual = read_rows(data / "figure3.csv")
        self.assertEqual(len(actual), 240)
        for calculated, saved in zip(expected, actual):
            for name in ("alpha", "D_v", "mean", "lower", "upper", "display_divisor"):
                self.assertEqual(float(saved[name]), calculated[name])
            self.assertEqual(saved["source_kind"], calculated["source_kind"])
            self.assertEqual(saved["error_kind"], calculated["error_kind"])
        recovery = json.loads((data / "correlation_recovery_audit.json").read_text())
        self.assertTrue(recovery["resolved"])
        self.assertLess(max(recovery["max_difference_from_archived_plot_seconds"].values()), 1e-8)
        self.assertTrue(recovery["reanalysis"]["raw_summary"]["byte_identical"])
        differences = recovery["reanalysis"]["floor_corrected"]["maximum_absolute_differences"]
        self.assertEqual(differences["floor_db_50000_ms"], 0.)
        self.assertEqual(differences["floor_db_jackknife_se_ms"], 0.)

    def test_reject_incomplete_duplicate_or_wrong_protocol(self):
        rows = read_rows(ROOT / "reference_data" / "correlation_time_debiased.csv")
        with self.assertRaises(ValueError):
            index_rows(rows[:-1], correlation=True)
        with self.assertRaises(ValueError):
            index_rows(rows + [rows[0]], correlation=True)
        wrong = [dict(r) for r in rows]
        wrong[0]["observation_ms"] = "12000"
        with self.assertRaises(ValueError):
            index_rows(wrong, correlation=True)

    def test_long_record_protocol(self):
        protocol = json.loads((ROOT / "parameters" / "correlation_time.json").read_text())
        actual = solver.exact_paper_params(12).__dict__.copy()
        actual.update(t_total=120000., t_warmup=3000.)
        self.assertEqual(json.loads(json.dumps(actual)), protocol["model"])
        self.assertEqual(protocol["trace_stride"] * actual["dt"], protocol["sample_dt_ms"])
        self.assertEqual(protocol["sampled_nodes"], 12)
        self.assertEqual(protocol["max_lag_ms"], 50000.)


if __name__ == "__main__":
    unittest.main()
