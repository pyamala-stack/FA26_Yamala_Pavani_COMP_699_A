import unittest
from src.error_analysis import analyze_prediction_rows


class ErrorAnalysisTests(unittest.TestCase):
    def test_classwise_counts_and_errors(self):
        rows = [
            {"filepath": "a.png", "actual": "Lung_Opacity", "predicted": "Lung_Opacity"},
            {"filepath": "b.png", "actual": "Lung_Opacity", "predicted": "Normal"},
            {"filepath": "c.png", "actual": "Normal", "predicted": "Normal"},
            {"filepath": "d.png", "actual": "Viral Pneumonia", "predicted": "Normal"},
        ]
        result = analyze_prediction_rows(rows)
        self.assertEqual(result["total_images"], 4)
        self.assertEqual(result["correct_predictions"], 2)
        self.assertEqual(result["incorrect_predictions"], 2)
        self.assertEqual(result["class_statistics"]["Lung_Opacity"]["incorrect"], 1)
        self.assertEqual(result["class_statistics"]["Normal"]["error_rate"], 0.0)
        self.assertEqual(len(result["misclassified_rows"]), 2)

    def test_empty_rows_raise_error(self):
        with self.assertRaises(ValueError):
            analyze_prediction_rows([])


if __name__ == "__main__":
    unittest.main()
