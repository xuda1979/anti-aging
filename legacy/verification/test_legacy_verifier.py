"""Independent scoring identities and corruption detection for recovered data."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("legacy_verify", HERE / "verify_legacy.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class LegacyVerifierTests(unittest.TestCase):
    def test_auc_and_balanced_accuracy_known_fixture(self):
        result = verifier.auc_and_balanced_accuracy([0, 1, 1, 0], [.1, .9, .4, .8])
        self.assertEqual(result, {"auroc": .75, "balanced_accuracy": .5})

    def test_ties_and_nonfinite_scores(self):
        self.assertEqual(verifier.auc_and_balanced_accuracy([0, 1], [.5, .5]), {"auroc": .5, "balanced_accuracy": .5})
        with self.assertRaises(AssertionError):
            verifier.auc_and_balanced_accuracy([0, 1], [.1, float("nan")])

    def test_saved_probability_record_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / "CellClockAI"
            shutil.copytree(HERE.parent / "CellClockAI", copied)
            path = copied / "results/public_predictions.csv"
            table = pd.read_csv(path)
            i = table.index[table.treated == 0][0]
            table.loc[i, "leave_donor_out_probability"] = 1.0
            table.to_csv(path, index=False)
            result = verifier.verify_results(copied)
            self.assertEqual(result["verdict"], "FAIL")
            self.assertIn("Independent numeric result differs", result["failure"]["message"])


if __name__ == "__main__":
    unittest.main()
