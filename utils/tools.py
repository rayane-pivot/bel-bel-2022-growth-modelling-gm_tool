import numpy as np
from sklearn.utils import check_array


def mean_absolute_percentage_error(y_true, y_pred):
    """TODO describe function

    :param y_true:
    :param y_pred:
    :returns:

    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
