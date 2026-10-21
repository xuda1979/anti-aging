import unittest
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cellclock.empirical import parse_sample, train_transform
from cellclock.synthetic import generate, benefit, supported_pair, InteractionRidge


class Invariants(unittest.TestCase):
    def test_no_intermediate_or_failed_samples_in_recovery_pilot(self):
        self.assertIsNone(parse_sample("O1_transient_reprogramming_intermediate_13days_exp2"))
        self.assertIsNone(parse_sample("O1_failed_to_transiently_reprogram_13days_exp1"))
        self.assertEqual(parse_sample("O2_transiently_reprogrammed_13days_exp1")["treated"], 1)

    def test_test_variance_cannot_change_selected_features(self):
        train = np.array([[1., 2., 3.], [2., 2., 7.], [3., 2., 5.]])
        test_a, test_b = np.array([[2., 2., 5.]]), np.array([[2., 1e9, 5.]])
        a, _, i = train_transform(train, test_a, max_features=2)
        b, _, j = train_transform(train, test_b, max_features=2)
        np.testing.assert_array_equal(i, j)
        np.testing.assert_array_equal(a, b)
        self.assertNotIn(1, i)

    def test_clock_decoy_has_no_functional_effect(self):
        u, z = np.zeros((2, 12)), np.zeros((2, 2))
        u[1, 2] = 1
        for scenario in ["additive", "synergy", "redundant", "stress_interaction", "no_causal_effect"]:
            self.assertEqual(benefit(u, z, scenario)[0], benefit(u, z, scenario)[1])

    def test_unseen_pair_triggers_abstention(self):
        d = generate(4, 900, "unseen_synergy", training=True)
        self.assertFalse(supported_pair(d["u"], (0, 1)))
        self.assertEqual((d["u"][:, 0] * d["u"][:, 1]).sum(), 0)

    def test_seed_reproduces_data_and_model(self):
        a, b = generate(8, 200, "additive", True), generate(8, 200, "additive", True)
        np.testing.assert_array_equal(a["function"], b["function"])
        m1 = InteractionRidge().fit(a["x"], a["function"])
        m2 = InteractionRidge().fit(b["x"], b["function"])
        np.testing.assert_allclose(m1.predict(a["x"]), m2.predict(b["x"]))

    def test_synergy_cannot_be_seen_in_singles(self):
        u, z = np.zeros((3, 12)), np.zeros((3, 2))
        u[0, 0], u[1, 1], u[2, :2] = 1, 1, 1
        np.testing.assert_allclose(benefit(u, z, "synergy"), [0, 0, 1.8])


class OptionalDeepModel(unittest.TestCase):
    def test_missing_modalities_and_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed; optional architecture not executed")
        from cellclock.deep_model import DynamicRegulatoryModel, masked_feature_loss
        torch.manual_seed(7)
        model = DynamicRegulatoryModel(5, [[0, 1, 2], [1, 2, 3]], width=8, layers=2)
        x = torch.randn(3, 5, 4)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, :, 1] = False
        x[:, :, 1] = float("nan")
        out = model(x, mask, torch.zeros(3, 5, 2), torch.ones(3))
        self.assertTrue(torch.isfinite(out["functional"]).all())
        loss = masked_feature_loss(out["next_features"], x, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(model.unit_embedding.grad).all())


if __name__ == "__main__":
    unittest.main()
