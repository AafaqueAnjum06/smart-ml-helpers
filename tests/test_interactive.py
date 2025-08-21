import unittest
import pandas as pd
from ml_helpers import interactive_plot

class TestInteractive(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "a": [1, 2, 3, 4],
            "b": [10, 20, 30, 40]
        })

    def test_interactive_plot_runs(self):
        # Function should run without error (actual widget display not tested)
        try:
            interactive_plot(self.df)
        except Exception as e:
            self.fail(f"interactive_plot raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()
