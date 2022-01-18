import utils
import pandas as pd
import json

config_file = "config.json"
with open(utils.get_data_path(config_file)) as json_file:
    BEL_BRANDS = json.load(json_file)

raw_data_file = "test_data.xlsx"
df_raw = pd.read_excel(utils.get_data_path(raw_data_file), engine="openpyxl")

MARKET_CATEGORIES = df_raw["Category"].unique()
MARKET_BRANDS = df_raw["Brand"].unique()
COMPETITION_BRANDS = [brand for brand in MARKET_BRANDS if brand not in BEL_BRANDS["core"] + BEL_BRANDS["local"]]


def get_bel_brands():
    return BEL_BRANDS

def get_market_categories():
    return MARKET_CATEGORIES

def get_market_brands():
    return MARKET_BRANDS

def get_competition_brands():
    return COMPETITION_BRANDS
