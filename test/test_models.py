import unittest
from pandas._testing import assert_frame_equal
from src.models import *
from src.setup import *
import shutil


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
        n_unique_cats = len(np.unique(df[cat_columns[1]])) - 1
        df_transformed = transform_features(df, dict_features)
        n_unique_dummies = len([x for x in df_transformed.columns if cat_columns[1] in x])
        self.assertTrue(n_unique_cats, n_unique_dummies)

    def test_train_model(self):
        if os.path.isdir('../results/TEST'):
            shutil.rmtree('../results/TEST')

        # model_info should entail information about model
        df = generate_cat_num_dataframe(n_samples=100, n_features=10, n_cats=3)
        cat_columns = [x for x in df.columns if 'cat' in x]
        target_list = ['target']
        feature_list = [x for x in df.columns if x not in target_list]

        dict_features = {}
        for feature in feature_list:
            if feature in cat_columns:
                dict_features[feature] = 'c'
            else:
                dict_features[feature] = 'n'

        dict_targets = {}
        for feature in target_list:
            dict_targets[feature] = 'c'

        df = transform_features(df, Merg(dict_features, dict_targets))

        target_list = [x for x in df.columns if list(dict_targets.keys())[0] in x]
        feature_list = [x for x in df.columns if x not in target_list]
        model_info = {
            'model_name': 'TEST',
            'features': feature_list,
            'targets': target_list,
            'random_state': 42,
            'test_size': 0.3,
            'parameters': {'n_estimators': [3, 6, 10],
                           'max_depth': [2, 5, 10],
                           'min_samples_split': [2, 4, 6]},
            'plot_confusion_matrix': True,
            'learning_curve': True,
            'feature_importance': True
        }
        train_model(df, model_info)
        shutil.rmtree('../results/TEST')
