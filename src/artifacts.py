import matplotlib.pyplot as plt
import prince
from pathlib import Path


def generate_mca(df, save_path):
    """
    :param df:
    :param save_path:
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

    mca.fit(df_cc_cat)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1)
    ax = mca.plot_coordinates(
        X=df_cc_cat,
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
    fig.savefig(save_path+Path('mca.png'))

