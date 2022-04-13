import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import *

from datamanager.DataManager import DataManager

class DM_UK(DataManager):
    """DataManager for UK data"""

    _country = "UK"

    def ad_hoc_UK(self, json_sell_out_params):
        df = super().fill_df(json_sell_out_params, self._country)
        
        sales_date_format = json_sell_out_params.get("UK").get("sales_date_format")
        date_format = json_sell_out_params.get("UK").get("date_format")
        
        df.Date = pd.to_datetime(df.Date, format=sales_date_format).dt.strftime(date_format)
        
        self._df = df

    def fill_df_bel(self, json_sell_out_params):
        """build df_bel"""
        assert not self._df.empty, "df is empty, call ad_hoc_USA() or load() first"
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
                    "Sales volume with promo"
                ]
            ]
            .agg(
                {
                    "Price per volume": "mean",
                    "Sales in volume": "sum",
                    "Sales in value": "sum",
                    "Sales volume with promo":"sum",
                }
            )
        )
        
        df_bel_brands = df[df.Brand.isin(bel_brands)]
        idx = df_bel_brands.groupby(["Date", "Brand"])['Sales in volume'].transform(max) == df_bel_brands["Sales in volume"]
        df_bel_brands = df_bel_brands.loc[idx, ["Date", "Brand", "Distribution"]]

        df_bel = pd.merge(df_bel, df_bel_brands, on=["Date", "Brand"], how="left")

        df_bel["Promo Cost"] = df_bel["Sales volume with promo"] / df_bel["Sales in volume"]
        print(f'<fill_df_bel> shape of df_bel : {df_bel.shape}')
        
        df_finance = self.fill_finance_UK(json_sell_out_params)
        df_finance = self.compute_Finance(df_finance, json_sell_out_params)
        print(f'<fill_df_bel> shape of df_finance : {df_finance.shape}')
        df_bel = pd.merge(df_bel, df_finance, on=["Date", "Brand"], how="left")
        print(f'<fill_df_bel> shape of df_bel : {df_bel.shape}')

        df_inno = self.fill_Inno_UK(json_sell_out_params)
        df_inno = self.compute_Inno(df_inno, json_sell_out_params)
        print(f'<fill_df_bel> shape of df_inno : {df_inno.shape}')
        df_bel = pd.merge(df_bel, df_inno, on=["Date", "Brand"], how="left")
        print(f'<fill_df_bel> shape of df_bel : {df_bel.shape}')

        self._df_bel = df_bel

    def fill_finance_UK(self, json_sell_out_params):
        path = (
                json_sell_out_params.get(self._country)
                .get("dict_path")
                .get("PATH_FINANCE")
                .get("Total Country")
            )
        header = (
                json_sell_out_params.get(self._country).get("Finance").get("header")
            )
        sheet_names = json_sell_out_params.get(self._country).get("Finance").get("sheet_names")
        df_finance = pd.read_excel(path, header=header, sheet_name=sheet_names)
        df_finance = pd.concat(df_finance.values(), axis=0)
        df_finance = df_finance.set_index(["BRAND", "Source", "SKUS"]).stack(dropna=False).reset_index()
        df_finance = df_finance.set_index(["BRAND", "SKUS", "level_3", "Source"]).unstack("Source").reset_index()

        df_finance = df_finance.reindex(columns=sorted(df_finance.columns, key=lambda x: x[::-1]))
        df_finance.columns = ['{}_{}'.format(t, v) for v,t in df_finance.columns]
        # df_finance["_Product"] = df_finance["_Product"].apply(lambda x:code_to_brand(x))
        # df_finance = df_finance[df_finance._Product.isin(ap_codes.values())]

        df_finance = df_finance.rename(columns={
            "_BRAND" : "Brand",
            "_SKUS" : "SKU",
            "_level_3" : "Date",
            "MVC_0" : "MVC",
            "Advertising_0" : "Advertising",
            "Promotion_0" : "Promotion"
        }).reset_index(drop=True)
        df_finance.Date = pd.to_datetime(df_finance.Date, format="%b %Y").dt.strftime("%Y-%m")
        return df_finance
    
    def compute_Finance(
        self,
        df_finance,
        json_sell_out_params
    ):
        """Many things done here, gl debugging it
        """
        date_min = json_sell_out_params.get(self._country).get("dates_finance").get("Min")
        date_max = json_sell_out_params.get(self._country).get("dates_finance").get("Max")
        
        # ABS for Advertising and Promotion
        df_finance["Advertising"] = df_finance["Advertising"].abs()
        df_finance["Promotion"] = df_finance["Promotion"].abs()
        
        df_finance = df_finance.fillna(0.0)
        df_finance["Year"] = pd.to_datetime(df_finance.Date, format="%Y-%m").dt.year
        df_finance["Month"] = pd.to_datetime(df_finance.Date, format="%Y-%m").dt.month
        df_finance = df_finance.drop(columns=["Date"])
        # Months to week
        df_finance["number of weeks"] = df_finance.apply(
            lambda x: self.count_num_sundays_in_month(x.Year, x.Month), axis=1
        )
        df_finance["A&P"] = df_finance.apply(
            lambda x: (x.Advertising + x.Promotion) / x["number of weeks"],
            axis=1,
        )
        
        df_finance["MVC"] = df_finance.apply(
            lambda x: x["MVC"] / x["number of weeks"] * 1000, axis=1
        )
        # Duplicate for n weeks
        full_idx = pd.date_range(start=date_min, end=date_max, freq="W-Sat")
        df_test = pd.DataFrame(index=full_idx)
        df_test["Year"] = df_test.index.year
        df_test["Month"] = df_test.index.month
        df_concat = pd.DataFrame()
        for brand in df_finance.Brand.unique():
            df_concat = pd.concat(
                [
                    df_concat,
                    pd.merge(
                        df_finance[df_finance.Brand == brand],
                        df_test.reset_index(),
                        on=["Year", "Month"],
                    ).rename(columns={"index": "Date"}),
                ]
            )
        # Change date type to str
        df_concat["Date"] = df_concat.Date.dt.strftime("%Y-%m-%d")
        
        df_concat = df_concat.groupby(["Brand", "Date"]).agg(
            {
                "Advertising" : "sum", 
                "Promotion" : "sum", 
                "A&P" : "sum", 
                "MVC" : "sum",
            }
        ).reset_index()

        return df_concat[
            ["Brand", "Date", "Advertising", "Promotion", "A&P", "MVC"]
        ]

    def fill_Inno_UK(
        self,
        json_sell_out_params
        ):
        """Read Innovation file, rename columns

        """
        path = json_sell_out_params.get(self._country).get("dict_path").get("PATH_INNO").get("Total Country")
        header = json_sell_out_params.get(self._country).get("Inno").get("header")
        brand_column_name = json_sell_out_params.get(self._country).get("Inno").get("brand_column_name")
        columns_to_remove = json_sell_out_params.get(self._country).get("Inno").get("columns_to_remove")
        date_format = json_sell_out_params.get(self._country).get("Inno").get("date_format")
        week_name = json_sell_out_params.get(self._country).get("Inno").get("week_name")
        print(f"<fill_Inno> Loading data from file {path}")
        # Load innovation file and some formating
        df_ino = pd.read_excel(path, header=header)
        # display(df_ino)
        # rename Brands
        df_ino = df_ino.rename(columns={brand_column_name: "Brand"})
        # Convert columns names to date format
        cols = [x for x in df_ino.columns if week_name in x]
        df_ino = df_ino.rename(
            columns={
                x: dt.datetime.strftime(
                    dt.datetime.strptime(x, date_format), "%Y-%m-%d"
                )
                for x in cols
            }
        )
        # remove unwanted columns
        df_ino = df_ino.drop(
            columns=[x for x in df_ino.columns if x in columns_to_remove]
        )
        return df_ino
    
    def compute_Inno(self, df, json_sell_out_params):
        #terrible terrible code, call ahmed for help
        """compute Innovation
        :returns: df_innovation

        """
        date_begining = json_sell_out_params.get(self._country).get("Inno").get("date_beg")
        innovation_duration = json_sell_out_params.get(self._country).get("Inno").get("innovation_duration")

        # Compute from innovation dataframe
        df_concat = pd.DataFrame()
        delta = dt.timedelta(weeks=innovation_duration)
        #delta = dt.timedelta(months=innovation_duration)
        # for each brand
        for brand, group in df.groupby(["Brand"]):
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

        return df_concat[["Brand", "Date", "Rate of Innovation"]].reset_index(drop=True)
