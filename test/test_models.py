import unittest
from pandas._testing import assert_frame_equal
from src.models import *


class TestSetUp(unittest.TestCase):
    def test_alphabetize_ordinals(self):
        alpha_arr = alphabetize_ordinals(df,list_ordinals)
        self.assertTrue(alpha_arr,test_arr)