import unittest
import pandas as pd
from ml_helpers import clean_data

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame with missing values
        self.df = pd.DataFrame({
            "numeric_col": [1, 2, None, 4],
            "categorical_col": ["A", None, "B", "C"]
        })

    def test_clean_data(self):
        cleaned_df = clean_data(self.df)
        
        # No missing values should remain
        self.assertFalse(cleaned_df.isna().any().any())

        # Numeric column missing value filled with mean
        expected_numeric_mean = (1 + 2 + 4) / 3
        self.assertEqual(cleaned_df.loc[2, "numeric_col"], expected_numeric_mean)

        # Categorical column missing value filled with "Unknown"
        self.assertEqual(cleaned_df.loc[1, "categorical_col"], "Unknown")

if __name__ == "__main__":
    unittest.main()
