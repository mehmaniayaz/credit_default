import unittest
from pandas._testing import assert_frame_equal
from src.models import *


class TestSetUp(unittest.TestCase):
    def test_alphabetize_ordinals(self):
        df = pd.DataFrame(columns={'ordinal_feature', 'numerical_feature'})
        df['numerical_feature'] = np.random.choice(range(10), size=(5))
        df['ordinal_feature'] = ['carrot', 'cake', 'apple', 'pony', 'banana']
        encoded_arr_true = [1, 0, 4, 3, 2]
        ordinal_list = ['cake', 'carrot', 'banana', 'pony', 'apple']
        encoded_arr_predicted = np.array(alphabetize_ordinals(df, ordinal_list))
        self.assertTrue((encoded_arr_true == encoded_arr_predicted).all())
