import unittest
from pandas._testing import assert_frame_equal
from src.models import *


class TestSetUp(unittest.TestCase):
    def test_alphabetize_ordinals(self):
        df = pd.DataFrame(columns={'ordinal_feature', 'numerical_feature'})
        df['numerical_feature'] = np.random.choice(range(10), size=(5))
        df['ordinal_feature'] = ['carrot', 'cake', 'apple', 'pony', 'banana']
        encoded_arr_true = ['1__carrot', '0__cake', '4__apple',
                            '3__pony', '2__banana']
        ordinal_list = ['cake', 'carrot', 'banana', 'pony', 'apple']
        temp = alphabetize_ordinals(df['ordinal_feature'], ordinal_list)
        encoded_arr_predicted  = list(np.array(temp).ravel())

        self.assertTrue(encoded_arr_true == encoded_arr_predicted)
