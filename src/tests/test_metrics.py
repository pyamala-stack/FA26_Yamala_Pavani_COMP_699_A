import unittest
from src.metrics import calculate_metrics


class MetricsTests(unittest.TestCase):
    def test_perfect_three_class_predictions(self):
        y = [0, 1, 2, 0, 1, 2]
        probabilities = [
            [0.90, 0.05, 0.05], [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90], [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10], [0.10, 0.10, 0.80]
        ]
        result = calculate_metrics(y, y, probabilities, 3)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["f1_macro"], 1.0)
        self.assertAlmostEqual(result["roc_auc_ovr_macro"], 1.0)

    def test_confusion_matrix_shape(self):
        result = calculate_metrics(
            [0, 1, 2], [0, 2, 2],
            [[.8,.1,.1], [.1,.2,.7], [.1,.1,.8]], 3
        )
        self.assertEqual(len(result["confusion_matrix"]), 3)
        self.assertTrue(all(len(row) == 3 for row in result["confusion_matrix"]))


if __name__ == "__main__":
    unittest.main()
