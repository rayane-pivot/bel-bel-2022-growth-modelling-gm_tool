import pandas as pd

import utils

raw_data_file = "test_data.xlsx"

df_raw = pd.read_excel(utils.get_data_path(raw_data_file), engine="openpyxl")


def get_sorted_df(category="ALL", subcategory="ALL", brand="ALL"):
    df = pd.read_excel(utils.get_data_path(raw_data_file), engine="openpyxl")

    if category != "ALL":
        df = df[df["Category"] == category]
    if subcategory != "ALL":
        df = df[df["SubCategory"] == subcategory]
    if brand != "ALL":
        df = df[df["Brand"] == brand]

    return df
