import unittest
from src.setup import *


class TestSetUp(unittest.TestCase):
    def test_generate_cat_num_dataframe(self):
        df = generate_cat_num_dataframe(n_samples=100, n_features=10, n_cats=2)
        # generated dataframe will in fact have 11 features with the added binary target column (numerical)
        self.assertEqual(len(df._get_numeric_data().columns), 10 + 1 - 2)
        self.assertEqual(len(df), 100)
        self.assertEqual(len(df.columns), 11)

    def test_encode_string_list(self):
        x = ['cat', 'dog', 'book', 'pencil', 'dog', 'book']
        y = [1, 2, 0, 3, 2, 0]
        y_pred = encode_string_list(x)
        self.assertListEqual(y, y_pred)

    def test_encode_df_cat_columns(self):
        df = generate_cat_num_dataframe(n_samples=100, n_features=10, n_cats=2)
        cat_feature_list = [x for x in df.columns if x not in df._get_numeric_data().columns]
        df_encoded = encode_df_cat_columns(df)
        test_df = np.unique(df[cat_feature_list[0]])
        test_df_encoded = np.unique(df_encoded[cat_feature_list[0]])
        self.assertEqual(len(test_df), len(test_df_encoded))
