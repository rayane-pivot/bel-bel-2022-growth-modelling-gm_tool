import numpy as np
import pandas as pd
import datetime as dt
import functools
from dateutil.relativedelta import *

from datamanager.DataManager import DataManager

class DM_BEL(DataManager):
    """DataManager for Belgium data"""

    _country = "BEL"

    def ad_hoc_BEL(self, json_sell_out_params):
        path = json_sell_out_params.get("BEL").get("dict_path").get("PATH_SALES").get("Total Country")
        print(f"<compute_Sales> open file {path}")
        df = pd.read_excel(path, header=[3, 4], sheet_name="WSP_Sheet1")
        
        index_features = [
            'Unnamed: 0_level_0', 
            'Unnamed: 1_level_0', 
            'Unnamed: 2_level_0',
            'Unnamed: 3_level_0'
        ]
        features = {
            'Sales Eq.':'Sales in volume', 
            'Sales Eq. (any promotion)':'Sales volume with promo', 
            'Sales Value':'Sales in value', 
            'Sales Value (any promotion)':'Sales value with promo', 
            'Price pr L/Kg LC':'Price per volume', 
            'No Promo Price pr L/Kg LC':'Price without promo',
            'Promo Price pr L/Kg LC':'Price with promo',
            'Weight. Distribution (w)':'Distribution',
        }

        df_merge = pd.DataFrame()
        for feature in features:
            df_feature = (
                df
                .loc[:, index_features+[feature]]
                .droplevel(0, axis=1)
                .pipe(self.sales_stack_dates)
                .pipe(self.sales_format_dates)
                .rename(columns={
                    "SEGMENTS" : "Category",
                    "SEGMENTS-2" : "Sub Category",
                    "SEGMENTS-3" : "Sub Sub Category",
                    "DOB" : "Brand",
                    0 : features[feature],
                })
                .drop(columns=["level_4"])
                .replace("ERR", np.nan)
            )
            if df_merge.empty:
                df_merge = df_feature
            else : 
                df_merge = pd.merge(df_merge, df_feature, on=["Category", "Sub Category", "Sub Sub Category", "Brand", "Date"])
        """ 
        Changes :
        ON YOUR BREAD TO SUB CATEGORY
            Spread to Sub Sub Category
        SPECIFIC NEEDS TO SUB CATEGORY
        PURE PLEASURE TO SUB CATEGORY
        """
        df_merge.loc[df_merge[(df_merge['Category'] == 'ON YOUR BREAD') & (df_merge['Sub Category']=='SLICES')].index, 'Category'] = 'OYB SLICES'
        df_merge.loc[df_merge[(df_merge['Category'] == 'ON YOUR BREAD') & (df_merge['Sub Category']=='SPREADS') & (df_merge['Sub Sub Category']=='OYB SPREADS FRESH-GOAT')].index, 'Category'] = 'OYB SPREADS FRESH-GOAT'
        df_merge.loc[df_merge[(df_merge['Category'] == 'ON YOUR BREAD') & (df_merge['Sub Category']=='SPREADS') & (df_merge['Sub Sub Category']=='OYB SPREADS MELTED')].index, 'Category'] = 'OYB SPREADS MELTED'        

        df_merge.loc[df_merge[(df_merge['Category'] == 'SPECIFIC NEEDS') & (df_merge['Sub Category']=='VEGAN')].index, 'Category'] = 'SPEC. NEEDS VEGAN'
        df_merge.loc[df_merge[(df_merge['Category'] == 'SPECIFIC NEEDS') & (df_merge['Sub Category']=='ORGANIC')].index, 'Category'] = 'SPEC. NEEDS ORGANIC'
        df_merge.loc[df_merge[(df_merge['Category'] == 'SPECIFIC NEEDS') & (df_merge['Sub Category']=='LACTOSE FREE')].index, 'Category'] = 'SPEC. NEEDS LACTOSE FREE'
        
        df_merge.loc[df_merge[(df_merge['Category'] == 'PURE PLEASURE') & (df_merge['Sub Category']=='HARD CHEESE')].index, 'Category'] = 'PP HARD CHEESE'
        df_merge.loc[df_merge[(df_merge['Category'] == 'PURE PLEASURE') & (df_merge['Sub Category']=='SOFT CHEESE')].index, 'Category'] = 'PP SOFT CHEESE'
        df_merge.loc[df_merge[(df_merge['Category'] == 'PURE PLEASURE') & (df_merge['Sub Category']=='PURE PLEASURE OTHERS')].index, 'Category'] = 'PP BLUE CHEESE'
        df_merge.loc[df_merge[(df_merge['Category'] == 'PURE PLEASURE') & (df_merge['Sub Category']=='BLUE CHEESE')].index, 'Category'] = 'PP BLUE CHEESE'
        
        self._df = df_merge


    def fill_df_bel(self, json_sell_out_params):
        df = self._df.copy()
        # df.Date = pd.to_datetime(df.Date).dt.strftime("%Y-%m-%d")
        bel_brands = json_sell_out_params.get(self._country).get("bel_brands")
        # df_bel = (
        #     df[df.Brand.isin(bel_brands)]
        #     .groupby(["Date", "Brand"], as_index=False)[
        #         [
        #             "Price per volume",
        #             "Sales in volume",
        #             "Sales in value",
        #             "Distribution",
        #             "Sales volume with promo",
        #         ]
        #     ]
        #     .agg(
        #         {
        #             "Price per volume": "mean",
        #             "Sales in volume": "sum",
        #             "Sales in value": "sum",
        #             "Distribution": "mean",
        #             "Sales volume with promo":"sum",
        #         }
        #     )
        # )
        df_bel = (
            df
            [df.Brand.isin(bel_brands)]
            .groupby(["Date", "Brand"], as_index=False)
            .apply(lambda group: pd.Series({
                    "Price per volume": (group["Price per volume"] * group["Sales in volume"]).sum()/group["Sales in volume"].sum(),
                    "Sales in volume": group["Sales in volume"].sum(),
                    "Sales in value": group["Sales in value"].sum(),
                    "Sales volume with promo":group["Sales volume with promo"].sum(),
                    "Distribution": (group["Distribution"] * group["Sales in volume"]).sum()/group["Sales in volume"].sum(),
            }))
        )
        
        df_bel["Promo Cost"] = df_bel["Sales volume with promo"] / df_bel["Sales in volume"]

        df_inno = self.compute_inno(json_sell_out_params)
        df_finance = self.compute_Finance(json_sell_out_params)
        
        df_bel = pd.merge(df_bel, df_inno, on=["Brand", "Date"], how="left").fillna(0.0)
        df_bel = pd.merge(df_bel, df_finance, on=["Brand", "Date"], how="left")

        self._df_bel = df_bel

    def sales_stack_dates(self, r):
        ### ad_hoc BEL
        return (
            r.set_index(["SEGMENTS", "SEGMENTS-2", "SEGMENTS-3", "DOB"])
            .stack()
            .reset_index()
        )

    def sales_format_dates(self, r):
        ### ad_hoc BEL
        r["Date"] = (
            (
                pd.to_datetime(r["level_4"] + "1", format="W %Y %W%w")
                + pd.to_timedelta(-1,unit='d')
            )
            .dt.strftime("%Y-%m-%d")
        )
        return r

    @staticmethod
    def inno_stack_dates(r):
        ### Rate of Innovation    
        return (
            r.set_index(["SEGMENTS", "SEGMENTS-2", "SIZE", "PRODUCT", "DOB"])
            .stack()
            .reset_index()
        )

    @staticmethod
    def inno_format_dates(r):
        ### Rate of Innovation
        r["Date"] = pd.to_datetime(r["level_5"] + "0", format="W %Y %W%w")
        return r
    
    @staticmethod
    def rate_of_inno(r):
        ### Rate of Innovation
        FIRST_DATE = pd.to_datetime("2019-02-17", format="%Y-%m-%d")
        LAST_DATE = pd.to_datetime("2022-02-27", format="%Y-%m-%d")
        df_inno_concat = pd.DataFrame()
        delta_inno_years = dt.timedelta(weeks=104)
        idx = pd.date_range(FIRST_DATE, LAST_DATE, freq="W")
        for _, group in r.groupby(["SEGMENTS", "SEGMENTS-2", "PRODUCT"]):
            group = group.set_index("Date")
            first_sale = group.index.min()
            if first_sale != FIRST_DATE:
                df_inno = group.loc[first_sale:first_sale+delta_inno_years, :]
                df_inno_concat = pd.concat([df_inno_concat, df_inno])
        
        df_inno_sum = df_inno_concat.groupby("Date")[0].sum()
        df_total_sum = r.groupby("Date")[0].sum()
        
        rate_of_inno = (df_inno_sum / df_total_sum).to_frame()
        rate_of_inno = rate_of_inno.reindex(idx, fill_value=0.0).fillna(0.0)
        rate_of_inno.loc[FIRST_DATE:FIRST_DATE+delta_inno_years] = 0.0
        rate_of_inno.index.name = "Date"
        rate_of_inno["Brand"] = r["DOB"].iloc[0]
        rate_of_inno = rate_of_inno.rename(columns={0:"Rate of Innovation"})
        return rate_of_inno


    def compute_inno(self, json_sell_out_params):
        def compose2(f, g):
            return lambda *a, **kw: g(f(*a, **kw))
        def compose(*fs):
            return functools.reduce(compose2, fs)

        path = json_sell_out_params.get(self._country).get("dict_path").get("PATH_INNO").get("Total Country")
        print(f"<compute_Inno> open file {path}")
        df = pd.read_excel(path, header=[3, 4], sheet_name="WSP_Sheet3")
        df_inno = (
            df
            .droplevel(0, axis=1)
            .groupby("DOB", as_index=False)
            .apply(compose(self.inno_stack_dates, self.inno_format_dates, self.rate_of_inno))
            .reset_index()
            .drop(columns=["level_0"])
            .assign(Date = lambda x:x["Date"].dt.strftime("%Y-%m-%d"))
        )
        return df_inno

    def compute_Finance(self, json_sell_out_params):
        path = json_sell_out_params.get(self._country).get("dict_path").get("PATH_FINANCE").get("Total Country")
        print(f"<compute_Finance> open file {path}")
        df = pd.read_excel(path, header=17, sheet_name="EPM DATABASE BEB V3")
        
        code_brands = {
            "000LA - THE LAUGHING COW":"VACHE QUI RIT",
            "000AC - APERICUBE":"APERICUBE",
            "000MD - MAREDSOUS":"MAREDSOUS",
            "000KE - KIRI":"KIRI",
            "000BA - BABYBEL":"BABYBEL",
            "000MB - MINI BABYBEL":"BABYBEL",
            "003NH - NURISHH":"NURISHH",
            "000BQ - BOURSIN":"BOURSIN",
        }

        def sum_ap(r):
            r["A&P"] = r[["Advertising", "Promotion"]].sum(axis=1, skipna=True)
            return r
        date_index = pd.date_range(start='2018-01-01', end='2022-01-01', freq='W-Sun')
        df_finance = (
            df
            # Filter country
            [df["MANAGERIAL EPM"] == "BELGIUM"]
            # Format variables
            .assign(Brand = lambda x: x["CODE EPM"].map(code_brands, na_action='ignore'))
            # Group by date brands in case multiple entree for one date brand
            .groupby(["Brand", "YEAR EPM"]).agg(sum).reset_index()
            .assign(MVC = lambda x: abs(x["MVC - Margin on variable costs"]) * 1_000)
            .assign(Advertising = lambda x: abs(x["R4100 - ADVERTISING"]) * 1_000)
            .assign(Promotion = lambda x: abs(x["R4200 - PROMOTION - CONSUMERS"]) * 1_000)
            .assign(Date = lambda x: pd.to_datetime(x["YEAR EPM"].str.rsplit(pat="-", expand=True)[0], format="%Y.%m "))
            # Keep columns of interest
            [["Date", "Brand", "MVC", "Advertising", "Promotion"]]
            # Resample Dates from Monthly to Weekly with first value copy
            .groupby('Brand', as_index=False)
            .apply(lambda x:
                # x.set_index('Date').resample('W', closed="left").ffill()
                x.set_index('Date').reindex(date_index, method="ffill")
            ).reset_index()
            .assign(Date = lambda x: x["level_1"])
            # Divide by number of weeks
            .groupby(["Brand", pd.Grouper(key="Date", freq="M")])
            .apply(lambda x:x.assign(
                Advertising=lambda y:y.Advertising/x.shape[0],
                Promotion=lambda y:y.Promotion/x.shape[0],
                MVC=lambda y:y.MVC/x.shape[0],
            ))
            # Date to string
            .assign(Date = lambda x: x.Date.dt.strftime("%Y-%m-%d"))
            # Add A&P column
            .pipe(sum_ap)
            # Clean columns
            .drop(columns=["level_0", "level_1"])
        )
        return df_finance

        
   