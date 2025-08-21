import unittest
from ml_helpers import plot_confusion
from sklearn.metrics import confusion_matrix

class TestModelViz(unittest.TestCase):

    def test_plot_confusion_runs(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]

        # Check function runs without error
        try:
            plot_confusion(y_true, y_pred)
        except Exception as e:
            self.fail(f"plot_confusion raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()
