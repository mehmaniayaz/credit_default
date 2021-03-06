import matplotlib.pyplot as plt
import prince
from pathlib import Path
import seaborn as sns
from src.setup import encode_string_list, encode_df_cat_columns
import numpy as np


def generate_mca(df, save_path=None):
    """
    :param df: dataframe entered that contains categorical variables but can also contain numerical ones
    :param save_path: path for saving the figure
    :return:
    """
    cat_feature_list = [x for x in df.columns if x not in df._get_numeric_data().columns]
    df_cc_cat = df[cat_feature_list]
    mca = prince.MCA(
        n_components=2,
        n_iter=3,
        copy=True,
        check_input=True,
        engine='auto',
        random_state=42
    )

    mca.fit(df_cc_cat.astype('category'))


    mca.plot_coordinates(
        X=df_cc_cat.astype('category'),
        ax=None,
        figsize=(7, 7),
        show_row_points=False,
        row_points_size=30,
        show_row_labels=False,
        show_column_points=True,
        column_points_size=40,
        show_column_labels=True,
        legend_n_cols=2
    )
    if save_path is not None:
        plt.savefig(save_path / Path('mca.png'))
        plt.close('all')


def generate_cat_cat_jitter_crossplot(x, y, **kwargs):
    """
    :Summary: Create a scatterplot of two categorical variables. Each categorical string variable will first be
        encoded and then jittered for clarity.

    :param x: list of variables entered for x axis
    :param y: list of variables entered for y axis
    :param kwargs: obligagtory dictionary meant to pass into pairgrid
    :return: None
    """
    x = encode_string_list(x)
    y = encode_string_list(y)

    randomized = lambda x: x + (np.random.rand() - 0.5) / 3
    x = map(randomized, x)
    y = map(randomized, y)
    sns.scatterplot(x, y, **kwargs)


def generate_cat_jitter_pairplot(df):
    """
    :Summary: Generate a pairplot for a dataframe with categorical features whose dummy variables are jittered for clarity.

    :param df: dataframe to be visualized
    :return: None
    """
    cat_feature_list = [x for x in df.columns if x not in df._get_numeric_data().columns]
    df, dict_rename = encode_df_cat_columns(df)
    df[cat_feature_list] = df[cat_feature_list].applymap(lambda x: x + (np.random.rand() - 0.5) / 3)

    ax = sns.pairplot(data=df, hue='target')
    ax.savefig(Path('./tmpt'))
    plt.close('all')
