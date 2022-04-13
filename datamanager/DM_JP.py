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
        df["Price per volume"] = df["Price per volume"] * 1_000
        df["Sales in value"] = df["Sales in value"] * 1_000
        self._df = df

    def fill_df_bel(self, json_sell_out_params):
        df = self._df.copy()
        df.Date = pd.to_datetime(df.Date).dt.strftime("%Y-%m-%d")
        bel_brands = json_sell_out_params.get(self._country).get("bel_brands")
        df_bel = (
            df[df.Brand.isin(bel_brands)]
            .groupby(["Date", "Brand"], as_index=False)[
                [
                    "Price per volume",
                    "Sales in volume",
                    "Sales in value",
                    "Distribution",
                ]
            ]
            .agg(
                {
                    "Price per volume": "mean",
                    "Sales in volume": "sum",
                    "Sales in value": "sum",
                    "Distribution": "mean"
                }
            )
        )
        
        df_inno = self.fill_inno(json_sell_out_params)

        df_bel = pd.merge(df_bel, df_inno, on=["Brand", "Date"])

        df_finance = self.fill_finance(json_sell_out_params)

        df_bel = pd.merge(df_bel, df_finance, on=["Brand", "Date"])
        self._df_bel = df_bel


    def fill_inno(self, json_sell_out_params)->pd.DataFrame:
        path = json_sell_out_params.get(self._country).get("dict_path").get("PATH_INNO").get("Total Country")
        df_inno = pd.read_excel(path)
        df_inno_out = (
            df_inno
            .drop(columns=["Product"])
            .pipe(self.rename_date_columns)
            .pipe(self.compute_inno, json_sell_out_params, country=self._country)
        )
        return df_inno_out
    
    def rename_date_columns(self, df :pd.DataFrame)->pd.DataFrame:
        df = df.rename(
                columns={
                    x: dt.datetime.strftime(
                        dt.datetime.strptime(x, "%Y.%m.%d"), "%Y-%m-%d"
                    )
                    for x in df.columns if "20" in x
                }
            )
        return df

    def compute_inno(self, df :pd.DataFrame, json_sell_out_params, country)->pd.DataFrame:
        date_begining = json_sell_out_params.get(country).get("Inno").get("date_beg")
        innovation_duration = json_sell_out_params.get(country).get("Inno").get("innovation_duration")

        # Compute from innovation dataframe
        df_concat = pd.DataFrame()
        delta = dt.timedelta(weeks=innovation_duration)
        #delta = dt.timedelta(months=innovation_duration)
        # for each brand
        for brand, group in df.groupby("Brand"):
            # init df
            df_merge = pd.DataFrame(index=group.columns.values[:-1])
            group = group.drop("Brand", axis=1)
            # Find date of first sale for each product
            for col in group.T.columns:
                try:
                    first_sale = group.T[col][pd.notna(group.T[col])].index.values[0]
                except Exception:
                    first_sale = date_begining
                if first_sale == date_begining:
                    pass
                else:
                    # get data for 2 years window of innovation
                    
                    date_end = (
                        dt.datetime.strptime(first_sale, "%Y-%m-%d") + delta
                    ).strftime("%Y-%m-%d")
                    
                    df_merge = pd.concat(
                        [group.T[[col]].loc[first_sale:date_end], df_merge], axis=1
                    )
            # beautiful peace of code here, ask ahmed for details
            df_innovation = pd.DataFrame(
                df_merge.reset_index()
                .sort_values(by="index")
                .set_index("index")
                .sum(axis=1)
            ).rename(columns={0: "Rate of Innovation"})
            date_begining_to_delta = (
                dt.datetime.strptime(date_begining, "%Y-%m-%d") + delta
            ).strftime("%Y-%m-%d")
            df_innovation.loc[date_begining:date_begining_to_delta] = 0.0
            # divide innovations by total sales
            df_innovation = df_innovation.div(group.T.sum(axis=1), axis=0)
            df_innovation.loc[:, "Brand"] = brand
            df_innovation = df_innovation.reset_index().rename(
                columns={"index": "Date"}
            )
            df_innovation = df_innovation[df_innovation["Date"] != "Brand"]
            df_concat = pd.concat([df_concat, df_innovation])
        
        df_concat["Date"] = (pd.to_datetime(df_concat["Date"]) - dt.timedelta(days=1)).dt.strftime("%Y-%m-%d")
        return df_concat[["Brand", "Date", "Rate of Innovation"]].reset_index(drop=True)

    def fill_finance(self, json_sell_out_params)->pd.DataFrame:
        path = (
            json_sell_out_params.get(self._country)
            .get("dict_path")
            .get("PATH_FINANCE")
            .get("Total Country")
        )
        df_finance = pd.read_excel(path)
        columns = ["Brand", "Date", "Advertising", "MVC", "Promotion"]

        df_finance_out = (
            df_finance
            .copy()
            .set_index(["Product", "Source"])
            .stack(dropna=False)
            .reset_index()
            .set_index(["Product", "level_2", "Source"])
            .unstack("Source")
            .reset_index()
            .droplevel(level=0, axis=1)
        )

        df_finance_out.columns = columns

        df_finance_final = (
            df_finance_out
            .copy()
            .pipe(self.abs_a_and_p)
            .pipe(self.discretize_dates, json_sell_out_params)
        )
        return df_finance_final[["Brand", "Date", "Advertising", "Promotion", "A&P", "MVC"]]


    def discretize_dates(self, df :pd.DataFrame, json_sell_out_params)->pd.DataFrame:
        date_min = json_sell_out_params.get(self._country).get("dates_finance").get("Min")
        date_max = json_sell_out_params.get(self._country).get("dates_finance").get("Max")
        
        df.Date = pd.to_datetime(df.Date, format="%Y.%m")
        
        df = df.fillna(0.0)
        df["Year"] = df.Date.dt.year
        df["Month"] = df.Date.dt.month
        df = df.drop(columns=["Date"])
        # Months to week
        df["number of weeks"] = df.apply(
            lambda x: self.count_num_sundays_in_month(x.Year, x.Month), axis=1
        )
        df["A&P"] = df.apply(
            lambda x: (x.Advertising + x.Promotion) / x["number of weeks"],
            axis=1,
        )
        
        df["MVC"] = df.apply(
            lambda x: x["MVC"] / x["number of weeks"], axis=1
        )
        # Duplicate for n weeks
        full_idx = pd.date_range(start=date_min, end=date_max, freq="W-Sun")
        df_test = pd.DataFrame(index=full_idx)
        df_test["Year"] = df_test.index.year
        df_test["Month"] = df_test.index.month
        df_concat = pd.DataFrame()
        for brand in df.Brand.unique():
            df_concat = pd.concat(
                [
                    df_concat,
                    pd.merge(
                        df[df.Brand == brand],
                        df_test.reset_index(),
                        on=["Year", "Month"],
                    ).rename(columns={"index": "Date"}),
                ]
            )
        # Change date type to str
        df_concat["Date"] = df_concat.Date.dt.strftime("%Y-%m-%d")
        
        return df_concat

    def abs_a_and_p(self, df :pd.DataFrame)->pd.DataFrame:
        df["Advertising"] = df["Advertising"].abs() * 1000
        df["Promotion"] = df["Promotion"].abs() * 1000
        df["MVC"] = df["MVC"] * 1000
        return df

        
   