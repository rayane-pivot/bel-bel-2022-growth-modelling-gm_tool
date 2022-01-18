import os

def get_data_path(data_file):
    return os.path.join("data/", data_file)

def replace_nan_by_zeros(df):
    df = df.fillna(0)
    return df
