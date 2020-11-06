import unittest
from pandas._testing import assert_frame_equal
from src.models import *


class TestSetUp(unittest.TestCase):
    def test_alphabetize_ordinals(self):
        df = pd.DataFrame(columns={'ordinal_feature', 'numerical_feature'})
        df['numerical_feature'] = np.random.choice(range(10), size=(5))
        df['ordinal_feature'] = ['carrot', 'cake', 'apple', 'Pony', 'BanAna']
        encoded_arr_true = [3, 2, 0, 5, 1]
        encoded_arr_predicted = np.array(alphabetize_ordinals(df))
        self.assertTrue((encoded_arr_true == encoded_arr_predicted).all())
