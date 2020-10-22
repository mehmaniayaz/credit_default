import unittest
from src.setup import *
from src.artifacts import *
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import os
from pathlib import Path

class TestArtifactsGeneration(unittest.TestCase):
    def setUp(self) -> None:
        os.mkdir(Path('./tmpt'))
        self.df = generate_cat_num_dataframe(n_samples=100, n_features=10, n_cats=2)
        self.cat_feature_list = [x for x in self.df.columns if x not in self.df._get_numeric_data().columns]

    def test_generate_mca(self):
        generate_mca(df = self.df,path=Path('./tmpt'))

    def tearDown(self) -> None:
        pass