import numpy as np
import pandas as pd
from sklearn.utils import check_array


def print_df_overview(df):
    print(pd.concat([df.head(5), df.sample(5), df.tail(5)]))


def mean_absolute_percentage_error(y_true, y_pred):
    """TODO describe function

    :param y_true:
    :param y_pred:
    :returns:

    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def cagr(start, end, nb_years):
    """TODO describe function

    :param end:
    :param start:
    :param nb_years:
    :returns:

    """

    return (((end / start) ** (1 / nb_years)) - 1) * 100
