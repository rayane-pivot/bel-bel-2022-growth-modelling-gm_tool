import os


def get_data_path(data_file):
    """TODO describe function

    :param data_file:
    :returns:

    """
    return os.path.join("data/", data_file)


def get_assets_path(data_file):
    """TODO describe function

    :param data_file:
    :returns:

    """
    return os.path.join("assets/", data_file)


def get_raw_data_file():
    """TODO describe function

    :returns:

    """
    raw_data_file = "test_data.xlsx"
    return get_data_path(raw_data_file)


def get_brands_data_file():
    """TODO describe function

    :returns:

    """
    brands_file = "brands.json"
    return get_data_path(brands_file)


def replace_nan_by_zeros(df):
    """TODO describe function

    :param df:
    :returns:

    """
    df = df.fillna(0)
    return df
