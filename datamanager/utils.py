import os

def get_data_path(data_file):
    return os.path.join("data/", data_file)

def get_static_path(static_type, static_file):
    return os.path.join("static/", static_type, static_file)

def get_asset_path(asset_file):
    return os.path.join("assets/", asset_file)

