import unittest
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

class TestArtifactsGeneration(unittest.TestCase):
    def setUp(self) -> None:
        X, y = make_classification(n_samples=100, n_features=5, n_redundant=2);
        df = pd.DataFrame(X, columns=['feature_' + str(x) for x in range(np.shape(X)[1])])
        df['target'] = y



    def tearDown(self) -> None:

if __name__=='__main__':
    unittest.main()
