import datetime as dt

import numpy as np
import pandas as pd

from datamanager.DataManager import DataManager

class DM_KSA(DataManager):
    """DataManager for KSA data"""

    _country = "KSA"
    _df_channels = dict()
    _df_bel_channels = dict()

    def ad_hoc_KSA(self, json_sell_out_params):
        df = super().fill_df(json_sell_out_params, self._country)
        cat_cols = json_sell_out_params.get("KSA").get("sales_cat_cols")
        date_cols = [col for col in df.columns if col not in cat_cols]
        cat_cols_no_feature = [col for col in cat_cols if col != "Feature"]
        
        df_concat = pd.DataFrame()
        for market, group in df.groupby(["MARKET"]):
            dict_features = dict()
            for feature, feature_group in group.groupby(["Feature"]):
                f_group = feature_group.copy().set_index(cat_cols)
                f_group = f_group[date_cols].stack(dropna=False).reset_index()
                f_group = f_group.rename(columns={"level_11":f"Date", 0:feature})
                df_market = group.groupby(cat_cols_no_feature).count().reset_index()[cat_cols_no_feature].copy().reset_index(drop=True)
                df_market = df_market.merge(f_group, on=cat_cols_no_feature, how="left")
                dict_features[feature]=df_market[[feature, "Date"]]
            for key, value in dict_features.items():
                df_market[key] = value[key]
            df_concat = pd.concat([df_concat, df_market])

        df_concat = df_concat.rename(columns=json_sell_out_params.get(self._country).get("sales_renaming_columns_dict").get("CATEGORICALS"))
        df_concat = df_concat.rename(columns=json_sell_out_params.get(self._country).get("sales_renaming_columns_dict").get("KPIS"))
        
        df_concat = df_concat.reset_index(drop=True)

        sales_date_format=json_sell_out_params.get(self._country).get("sales_date_format")
        date_format = json_sell_out_params.get("KSA").get("date_format")
        df_concat.Date = df_concat.Date.apply(lambda x:dt.datetime.strptime(x, sales_date_format)).dt.strftime(date_format)
        
        df_concat.loc[df_concat[(df_concat['Category'] == 'PORTIONS') & (df_concat['Sub Category']=='TRIANGLE PORTION')].index, 'Category'] = 'PORTIONS TRIANGLE'
        df_concat.loc[df_concat[(df_concat['Category'] == 'PORTIONS') & (df_concat['Sub Category']=='BAGS')].index, 'Category'] = 'PORTIONS OTHER'
        df_concat.loc[df_concat[(df_concat['Category'] == 'PORTIONS') & (df_concat['Sub Category']=='SLICES')].index, 'Category'] = 'PORTIONS OTHER'
        df_concat.loc[df_concat[(df_concat['Category'] == 'PORTIONS') & (df_concat['Sub Category']=='TRAY')].index, 'Category'] = 'PORTIONS OTHER'
        df_concat.loc[df_concat[(df_concat['Category'] == 'PORTIONS') & (df_concat['Sub Category']=='SQUARE PORTION')].index, 'Category'] = 'PORTIONS SQUARE'
        
        

        df_concat.loc[df_concat[(df_concat['Category'] == 'JAR') & (df_concat['Flavour']=='BLU')].index, 'Category'] = 'JAR BLU'
        df_concat.loc[df_concat[(df_concat['Category'] == 'JAR') & (df_concat['Flavour']=='GOLD')].index, 'Category'] = 'JAR GOLD'
        df_concat.loc[df_concat[(df_concat['Category'] == 'JAR') & (df_concat['Flavour'].isin(['PARMESAN &ROMANO CHEESE', 
                                                                                                    'PARMESAN CHEESE', 
                                                                                                    'ROMANO CHEESE',
                                                                                                    'PARMESAN&ROMANO&ASIAGO CHEESE', 
                                                                                                    'NOT AVAILABLE',
                                                                                                    'PECORINO/PECORINO ROMANO CHEESE', 
                                                                                                    'OTHER FLAVOUR']))].index, 'Category'] = 'JAR OTHER'
        
        df_new_price = pd.DataFrame()
        df_concat = df_concat.replace(to_replace='ERR', value=None)
        for i, group in df_concat.groupby(['Channel', 'Category', 'Sub Category', 'Brand', 'Promo']):
            def fn(r):
                s = r.iloc[0, :]
                if sum(r["Sales in volume"]) != 0:
                    s["Price per volume"] = sum(r["Sales in volume"] * r["Price per volume"])/sum(r["Sales in volume"])
                else :
                    s["Price per volume"] = None
                s["Sales in volume"] = r["Sales in volume"].sum()
                s["Sales in value"] = r["Sales in value"].sum()
                s["Distribution"] = r["Distribution"].mean()
                return s
            df_bcs = group.groupby("Date").apply(lambda x: fn(x))
            df_new_price = pd.concat([df_new_price, df_bcs])
        
        df_modern = df_new_price[df_new_price["Channel"]=="MODERN TRADE"]
        df_trad = df_new_price[df_new_price["Channel"]=="TRADITIONAL TRADE"]
        print(df_modern.shape)
        print(df_trad.shape)
        self.add_df_channel(key='modern', df=df_modern)
        self.add_df_channel(key='trad', df=df_trad)
        
        self._df = df_new_price


    def fill_df_bel(self, json_sell_out_params):
        """build df_bel

        :param json_sell_out_params: json params dict
        :returns: None

        """
        for channel, df in self.get_df_channels().items():
            json_sell_out_params = json_sell_out_params.copy()
            df = df.copy()
            df.Date = pd.to_datetime(df.Date)
            df.Date = df.Date.dt.strftime("%Y-%m")
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
                        "Distribution": "mean",
                    }
                )
            )

            PATH_FINANCE = (
                json_sell_out_params.get(self._country)
                .get("dict_path")
                .get("PATH_FINANCE")
                .get("Total Country")
            )
            AP_CODES = json_sell_out_params.get(self._country).get("A&P_codes")
            FINANCE_COLS = json_sell_out_params.get(self._country).get("A&P_columns")
            FINANCE_RENAMING_COLS = json_sell_out_params.get(self._country).get(
                "finance_renaming_columns"
            )
            DATE_MIN = (
                json_sell_out_params.get(self._country).get("dates_finance").get("Min")
            )
            DATE_MAX = (
                json_sell_out_params.get(self._country).get("dates_finance").get("Max")
            )
            FINANCE_HEADER = (
                json_sell_out_params.get(self._country).get("Finance").get("header")
            )
            COUNTRY_NAME = (
                json_sell_out_params.get(self._country)
                .get("Finance")
                .get("country_name")
            )
            df_finance = self.fill_Finance(
                path=PATH_FINANCE,
                finance_cols=FINANCE_COLS,
                finance_renaming_columns=FINANCE_RENAMING_COLS,
                header=FINANCE_HEADER,
            )
            df_finance = self.compute_Finance(
                df_finance, AP_CODES, DATE_MIN, DATE_MAX, country_name=COUNTRY_NAME
            )
            df_finance = df_finance.replace("THE LAUGHING COW", "LA VACHE QUI RIT")
            df_bel = pd.merge(df_bel, df_finance, on=["Brand", "Date"], how="left")

            PATH_INNO = (
                json_sell_out_params.get(self._country)
                .get("dict_path")
                .get("PATH_INNO")
                .get("Total Country")
            )
            DATE_BEG = (
                json_sell_out_params.get(self._country).get("Inno").get("date_beg")
            )
            INNO_HEADER = (
                json_sell_out_params.get(self._country).get("Inno").get("header")
            )
            INNOVATION_DURATION = (
                json_sell_out_params.get(self._country)
                .get("Inno")
                .get("innovation_duration")
            )
            INNO_BRAND_COL_NAME = (
                json_sell_out_params.get(self._country)
                .get("Inno")
                .get("brand_column_name")
            )
            INNO_WEEK_NAME = (
                json_sell_out_params.get(self._country).get("Inno").get("week_name")
            )
            INNO_COLS_TO_REMOVE = (
                json_sell_out_params.get(self._country)
                .get("Inno")
                .get("columns_to_remove")
            )
            INNO_DATE_FORMAT = (
                json_sell_out_params.get(self._country).get("Inno").get("date_format")
            )
            df_inno = self.fill_Inno(
                path=PATH_INNO,
                header=INNO_HEADER,
                brand_column_name=INNO_BRAND_COL_NAME,
                week_name=INNO_WEEK_NAME,
                columns_to_remove=INNO_COLS_TO_REMOVE,
                date_format=INNO_DATE_FORMAT,
            )
            df_inno = self.compute_Inno(
                df=df_inno,
                date_begining=DATE_BEG,
                innovation_duration=INNOVATION_DURATION,
            )
            df_bel = pd.merge(df_bel, df_inno, on=["Brand", "Date"], how="left")

            self.add_df_bel_channel(key=channel, df=df_bel)
        pass

    def get_df_channels(self):
        """get dict of df with channels as keys

        :returns: df_channels

        """
        return self._df_channels

    def get_df_by_channel(self, channel):
        """get df from df_channels with channel as key

        :param channel:
        :returns: df

        """
        assert channel in self.get_df_channels().keys(), f"{channel} not in df_channels"
        return self.get_df_channels().get(channel)

    def add_df_channel(self, key, df):
        """add df in df_channels at key

        :param key:
        :param df:
        :returns:

        """
        self._df_channels[key] = df

    def get_df_bel_channels(self):
        """

        :returns:

        """
        return self._df_bel_channels

    def get_df_bel_by_channel(self, channel):
        """

        :param channel:
        :returns:

        """
        assert (
            channel in self.get_df_bel_channels().keys()
        ), f"{channel} not in df_channels"
        return self.get_df_bel_channels().get(channel)

    def add_df_bel_channel(self, key, df):
        """

        :param key:
        :param df:
        :returns:

        """
        self._df_bel_channels[key] = df