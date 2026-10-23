
import json
import os
import unittest

class TestMLPipeline(unittest.TestCase):

    def test_model_file_exists(self):
        self.assertTrue(
            os.path.exists("student_result_model.pkl"),
            "Trained model file is missing"
        )

    def test_metrics_file_exists(self):
        self.assertTrue(
            os.path.exists("metrics.json"),
            "Metrics file is missing"
        )

    def test_metrics_are_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertIsInstance(metrics, dict)
        self.assertGreater(len(metrics), 0)

if __name__ == "__main__":
    unittest.main()
