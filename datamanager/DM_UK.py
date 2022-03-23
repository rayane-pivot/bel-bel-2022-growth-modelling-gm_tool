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
                    "Sales volume with promo",
                    "Distribution",
                ]
            ]
            .agg(
                {
                    "Price per volume": "mean",
                    "Sales in volume": "sum",
                    "Sales in value": "sum",
                    "Sales volume with promo":"sum",
                    "Distribution": "mean",
                }
            )
        )
        df_bel["Promo Cost"] = df_bel["Sales volume with promo"] / df_bel["Sales in volume"]
        print(f'<fill_df_bel> shape of df_bel : {df_bel.shape}')

        df_inno = self.fill_Inno_UK(json_sell_out_params)
        df_inno = self.compute_Inno(df_inno, json_sell_out_params)
        print(f'<fill_df_bel> shape of df_inno : {df_inno.shape}')
        print(df_bel.head())
        print(df_inno.head())
        df_bel = pd.merge(df_bel, df_inno, on=["Date", "Brand"], how="outer")
        print(f'<fill_df_bel> shape of df_bel : {df_bel.shape}')

        self._df_bel = df_bel

    def fill_finance_UK(self, json_sell_out_params):
        #ONLY FOR MVC
        def code_to_brand(r):
            for code, brand in ap_codes.items():
                if code in r:
                    return brand
            return ""
        path = (
                json_sell_out_params.get(self._country)
                .get("dict_path")
                .get("PATH_FINANCE")
                .get("Total Country")
            )
        header = (
                json_sell_out_params.get(self._country).get("Finance").get("header")
            )
        # ap_codes = json_sell_out_params.get(self._country).get("Finance").get("finance_codes_to_brands")
        sheet_names = json_sell_out_params.get(self._country).get("Finance").get("sheet_names")
        print(f'{path= }')
        print(f'{header= }')
        # print(f'{ap_codes= }')
        df_finance = pd.read_excel(path, header=header, sheet_name=sheet_names)
        df_finance = pd.concat(df_finance.values(), axis=0)
        # display(df_finance)
        df_finance = df_finance.set_index(["BRAND", "Source", "SKUS"]).stack(dropna=False).reset_index()
        #print(df_finance.head())
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
            "Advertising_0" : "Net Sales",
            "Promotion_0" : "Sales Margin"
        }).reset_index(drop=True)
        df_finance.Date = pd.to_datetime(df_finance.Date, format="%b %Y").dt.strftime("%Y-%m")
        return df_finance

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
