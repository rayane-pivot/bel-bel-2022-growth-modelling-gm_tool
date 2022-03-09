import pandas as pd

from model.Model import Model


class M_USA(Model):
    """extended class Model for Belgique"""

    def compute_growth(self, df, year, category):
        if year == "2018":
            return "NA"
        if year == "2019":
            return self.cagr_in_tons(
                df,
                brand="all",
                category=category,
                date_min={"Min": "2018-01-01", "Max": "2019-01-01"},
                date_max={"Min": "2019-01-01", "Max": "2020-01-01"},
            )
        elif year == "2020":
            return self.cagr_in_tons(
                df,
                brand="all",
                category=category,
                date_min={"Min": "2019-01-01", "Max": "2020-01-01"},
                date_max={"Min": "2020-01-01", "Max": "2021-01-01"},
            )
        elif year == "2021":
            return self.cagr_in_tons(
                df,
                brand="all",
                category=category,
                date_min={"Min": "2020-01-01", "Max": "2021-01-01"},
                date_max={"Min": "2021-01-01", "Max": "2022-01-01"},
            )
        else:
            return 0
