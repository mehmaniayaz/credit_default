import unittest
from pandas._testing import assert_frame_equal
from src.models import *
from src.setup import *


class TestSetUp(unittest.TestCase):
    def test_alphabetize_ordinals(self):
        df = pd.DataFrame(columns={'ordinal_feature', 'numerical_feature'})
        df['numerical_feature'] = np.random.choice(range(10), size=(5))
        df['ordinal_feature'] = ['carrot', 'cake', 'apple', 'pony', 'banana']
        encoded_arr_true = ['1__carrot', '0__cake', '4__apple',
                            '3__pony', '2__banana']
        ordinal_list = ['cake', 'carrot', 'banana', 'pony', 'apple']
        temp = alphabetize_ordinals(df['ordinal_feature'], ordinal_list)
        encoded_arr_predicted = list(np.array(temp).ravel())

        self.assertTrue(encoded_arr_true == encoded_arr_predicted)

    def test_transform_features(self):
        df = generate_cat_num_dataframe(n_samples=100, n_features=10, n_cats=2)
        cat_columns = [x for x in df.columns if 'cat' in x]
        order_list = ['lbl_3', 'lbl_0', 'lbl_4', 'lbl_1', 'lbl_2']
        dict_features = {cat_columns[0]: 'c', cat_columns[1]: 'o'}
        df['cat_feature_8'] = alphabetize_ordinals(df[cat_columns[0]], order_list)
        n_unique_cats = len(np.unique(df[cat_columns[1]]))-1
        df_transformed = transform_features(df, dict_features)
        n_unique_dummies = len([x for x in df_transformed.columns if cat_columns[1] in x])
        self.assertTrue(n_unique_cats,n_unique_dummies)

