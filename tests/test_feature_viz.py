import unittest
import pandas as pd
from ml_helpers import plot_histogram

class TestFeatureViz(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "numbers": [1, 2, 3, 4, 5]
        })

    def test_plot_histogram_runs(self):
        # We just want to check that function runs without error
        try:
            plot_histogram(self.df['numbers'])
        except Exception as e:
            self.fail(f"plot_histogram raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()
