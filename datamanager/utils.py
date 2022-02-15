import os

def get_data_path(data_file):
    return os.path.join("data/", data_file)

def get_assets_path(data_file):
    return os.path.join("assets/", data_file)

def get_raw_data_file():
    raw_data_file = "test_data.xlsx"
    return get_data_path(raw_data_file)

def get_brands_data_file():
    brands_file = "brands.json"
    return get_data_path(brands_file)

def replace_nan_by_zeros(df):
    df = df.fillna(0)
    return df
