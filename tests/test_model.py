import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

import morris_lecar_paired_alpha_compare_simplicial as base
import morris_lecar_simplicial_paper_suite as suite
from morris_lecar_simplicial_matched_coupling_array import matched_pairwise_network

ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def test_source_snapshot(self):
        manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
        for entry in manifest["files"]:
            self.assertEqual(hashlib.sha256((ROOT / entry["file"]).read_bytes()).hexdigest(),
                             entry["sha256"], entry["file"])

    def test_noise_conversion(self):
        for alpha in (1.2, 1.5, 2.):
            for intensity in (0., .04, .3, 9.):
                scale = base.noise_scale_from_voltage_noise_intensity(intensity, alpha, 20.)
                self.assertAlmostEqual(base.voltage_noise_intensity(scale, alpha, 20.), intensity)

    def test_gaussian_normalization(self):
        actual = base.sample_alpha_stable(2., 0., (100,), np.random.default_rng(41))
        expected = np.sqrt(2.) * np.random.default_rng(41).standard_normal(100)
        np.testing.assert_array_equal(actual, expected)

    def test_simplex_projection_and_matching(self):
        p = base.PairedSimplicialParams(n_nodes=8, p_edge=1., p_fill_simplex=.5,
                                      sigma_1=.002, sigma_2=.12)
        net = base.build_simplicial_network(p, np.random.default_rng(19))
        np.testing.assert_allclose(net.l2.sum(axis=1), 0., atol=1e-12)
        voltage = np.linspace(-70., 20., p.n_nodes)
        explicit = np.zeros(p.n_nodes)
        for i, j, k in net.simplices2:
            self.assertTrue(net.a1[i, j] and net.a1[i, k] and net.a1[j, k])
            explicit[i] += .5 * (voltage[j] + voltage[k]) - voltage[i]
            explicit[j] += .5 * (voltage[i] + voltage[k]) - voltage[j]
            explicit[k] += .5 * (voltage[i] + voltage[j]) - voltage[k]
        np.testing.assert_allclose(net.l2 @ voltage, explicit, atol=1e-12)
        matched_p, matched_net, info = matched_pairwise_network(p, net)
        self.assertEqual(matched_p.sigma_2, 0.)
        self.assertAlmostEqual(-np.diag(matched_net.pairwise_operator).mean(), info["g_eff_target"])

    def test_paired_zero_simplex(self):
        p = base.PairedSimplicialParams(n_nodes=5, n_trials_per_network=2,
                                      sigma_2=0., t_total=2., t_warmup=.5)
        net = base.build_simplicial_network(p, np.random.default_rng(4))
        args = dict(alpha=1.5, noise_scale=10., init_seed=42, noise_seed=43)
        a = base.simulate_batch(p, net, higher_order_enabled=False, **args)
        b = base.simulate_batch(p, net, higher_order_enabled=True, **args)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_production_parameters_match_saved_run(self):
        actual = json.loads(json.dumps(suite.candidate39_params(suite.PaperSuiteConfig()).__dict__))
        saved = json.loads((ROOT / "parameters" / "main.json").read_text())
        self.assertEqual(actual, saved)
        self.assertEqual(saved["n_networks"] * saved["n_trials_per_network"] ** 2, 4320)


if __name__ == "__main__":
    unittest.main()
