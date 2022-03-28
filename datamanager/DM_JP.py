import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import *

from datamanager.DataManager import DataManager

class DM_JP(DataManager):
    """DataManager for JP data"""

    _country = "JP"

    def ad_hoc_JP(self, json_sell_out_params):
        def date_to_format(df):
            df.Date = pd.to_datetime(df.Date+' 0', format="%Y W%U %w")
            return df
        
        def sub_cat_to_cat(df):
            df["Category"] = df["Sub Category"]
            
            df = df.replace({
                "Category": {
                    "COOKING OTHERS" : "COOKING OTHERS",
                    "COOKING_CREAM" : "COOKING CREAM",
                    "MOZZARELLA" : "COOKING MOZZARELLA",
                    "POWDER" : "COOKING POWDER",
                    "SHREDDED" : "COOKING SHREDDED",
                    "SLICE" : "COOKING SLICE",
                    "BABY" : "SNK BABY",
                    "CAMEMBERT" : "SNK CAMEMBERT",
                    "CARTON" : "SNK CARTON",
                    "CREAM PORTION" : "SNK CREAM PORTION",
                    "PORTION" : "SNK PORTION",
                    "ROUND" : "SNK ROUND",
                    "SNACKING_NATURAL" : "SNK NATURAL",
                    "SNACKING_PROCESS" : "SNK PROCESS",
                    "STRING" : "SNK STRING",
                    "SWEET" : "SNK SWEET"
                    }
                })
            return df

        df = super().fill_df(json_sell_out_params, self._country)
        df = (
            df
            .set_index(["CATEGORY", "SUB CATEGORY", "BRAND", "Feature", "Channel"])
            .stack(dropna=False)
            .reset_index()
            .rename(columns={
                "CATEGORY":"Category",
                "SUB CATEGORY":"Sub Category",
                "BRAND":"Brand",
                "level_5":"Date"
                })
            .set_index(["Category", "Sub Category", "Brand", "Date", "Feature", "Channel"])
            .unstack("Feature")
            .droplevel(0, axis=1)
            .reset_index()
            .rename(columns={
                "Avg Price per Pack (JPY)" : "Price per pack",
                "Avg Price per Volume (K JPY)": "Price per volume",
                "Sales Value (K JPY)": "Sales in value",
                "Sales Volume (kgs)": "Sales in volume",
                "TDP SKU Gross Weighted Distrib": "Distribution",
            })
            .rename_axis(None, axis=1)
            .pipe(date_to_format)
            .pipe(sub_cat_to_cat)
            )
        self._df = df

    def fill_df_bel(self, json_sell_out_params):
        """build df_bel"""
        pass

   