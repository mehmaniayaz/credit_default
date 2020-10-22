import unittest
from src.setup import *


class TestSetUp(unittest.TestCase):
    def test_generate_cat_num_dataframe(self):
        df = generate_cat_num_dataframe(n_samples=100, n_features=10, n_cats=2)
        #generated dataframe will in fact have 11 features with the added binary target column (numerical)
        self.assertEqual(len(df._get_numeric_data().columns), 10+1-2)
        self.assertEqual(len(df), 100)
        self.assertEqual(len(df.columns), 11)
