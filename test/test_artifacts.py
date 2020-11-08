import unittest
from src.setup import *
from src.artifacts import *
import os
from pathlib import Path
import shutil
import matplotlib.pyplot as plt


class TestArtifactsGeneration(unittest.TestCase):
    def setUp(self) -> None:
        if os.path.isdir(Path('./tmpt')):
            shutil.rmtree(Path('./tmpt'))
        os.mkdir(Path('./tmpt'))
        self.df = generate_cat_num_dataframe(n_samples=100, n_features=10, n_cats=2)
        self.cat_feature_list = [x for x in self.df.columns if x not in self.df._get_numeric_data().columns]

    def test_generate_mca(self):
        generate_mca(df=self.df, save_path=Path('./tmpt'))

    def test_generate_cat_cat_jitter_crossplot(self):
        x = self.df[self.cat_feature_list[0]].to_numpy()
        y = self.df[self.cat_feature_list[1]].to_numpy()
        generate_cat_cat_jitter_crossplot(x, y)
        plt.close('all')

    def test_generate_cat_jitter_pairplot(self):
        generate_cat_jitter_pairplot(self.df)

    def tearDown(self) -> None:
        shutil.rmtree(Path('./tmpt'))
