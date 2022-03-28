import datetime as dt

import numpy as np
import pandas as pd
import re
from datamanager.DataManager import DataManager
from dateutil.relativedelta import *
class DM_KSA(DataManager):
    """DataManager for KSA data"""

    _country = "KSA"
    _df_channels = dict()
    _df_bel_channels = dict()

    def ad_hoc_KSA(self, json_sell_out_params):
        def weighted_price(r):
            s = r.iloc[0, :]
            if sum(r["Sales in volume"]) != 0:
                s["Price per volume"] = sum(r["Sales in volume"] * r["Price per volume"])/sum(r["Sales in volume"])
            else :
                s["Price per volume"] = np.nan
            s["Sales in volume"] = r["Sales in volume"].sum()
            s["Sales in value"] = r["Sales in value"].sum()
            s["Distribution"] = r["Distribution"].mean()
            return s

        def to_promo(r):
            if r.shape[0] == 1:
                return r
            #else there is promo and no promo:
            s = r[r.Promo=="NO PROMO"]
            s.loc[:, ["Sales in volume", "Sales in value", "Price per volume"]] = r.loc[:, ["Sales in volume", "Sales in value", "Price per volume"]].sum().values
            s.loc[:, ["Sales volume with promo", "Sales value with promo", "Price with promo"]] = r[r.Promo=="PROMO"].loc[:, ["Sales in volume", "Sales in value", "Price per volume"]].values
            return s

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
        df_concat.loc[df_concat[(df_concat['Category'] == 'PORTIONS') & (df_concat['Sub Category']=='STICK')].index, 'Category'] = 'PORTIONS OTHER'
        df_concat.loc[df_concat[(df_concat['Category'] == 'PORTIONS') & (df_concat['Sub Category']=='SQUARE PORTION')].index, 'Category'] = 'PORTIONS SQUARE'
        
        

        df_concat.loc[df_concat[(df_concat['Category'] == 'JAR') & (df_concat['Flavour']=='BLU')].index, 'Category'] = 'JAR BLU'
        df_concat.loc[df_concat[(df_concat['Category'] == 'JAR') & (df_concat['Flavour']=='GOLD')].index, 'Category'] = 'JAR GOLD'
        df_concat.loc[df_concat[(df_concat['Category'] == 'JAR') & (df_concat['Flavour'].isin(['PARMESAN &ROMANO CHEESE', 
                                                                                                    'PARMESAN CHEESE', 
                                                                                                    'ROMANO CHEESE',
                                                                                                    'PARMESAN&ROMANO&ASIAGO CHEESE', 
                                                                                                    'NOT AVAILABLE',
                                                                                                    'PECORINO/PECORINO ROMANO CHEESE', 
                                                                                                    'OTHER FLAVOUR', 'FETTA CHEESE']))].index, 'Category'] = 'JAR OTHER'
        
        df_concat.loc[df_concat[df_concat['Brand'] == 'LAVACHEQUIRIT'].index, 'Brand'] = 'LA VACHE QUI RIT'
        df_new_price = pd.DataFrame()
        df_concat = df_concat.replace(to_replace='ERR', value=np.nan)
        for _, group in df_concat.groupby(['Channel', 'Category', 'Sub Category', 'Brand', 'Promo']):
            df_bcs = group.groupby("Date").apply(lambda x: weighted_price(x))
            df_new_price = pd.concat([df_new_price, df_bcs])
        
        df_new_price = df_new_price.reset_index(drop=True)
        df_concat_promo = pd.DataFrame()
        df_new_price[["Sales volume with promo", "Sales value with promo", "Price with promo"]] = np.nan
        for _, group in df_new_price.groupby(["Channel", "Category", "Sub Category", "Brand"]):
            df_promo = group.groupby("Date").apply(lambda x:to_promo(x)).reset_index(drop=True)
            df_concat_promo = pd.concat([df_concat_promo, df_promo])
        
        df_modern = df_concat_promo[df_concat_promo["Channel"]=="MODERN TRADE"]
        df_trad = df_concat_promo[df_concat_promo["Channel"]=="TRADITIONAL TRADE"]
    
        print(f"<ad_hoc_KSA> shape of total df : {df_concat_promo.shape}")
        print(f"<ad_hoc_KSA> shape of modern : {df_modern.shape}")
        print(f"<ad_hoc_KSA> shape of traditional : {df_trad.shape}")
        self.add_df_channel(key='modern', df=df_modern)
        self.add_df_channel(key='trad', df=df_trad)
        
        self._df = df_concat_promo


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
            
            #df_finance = self.fill_Finance(json_sell_out_params, country=self._country)
            df_finance = self.fill_finance_KSA(json_sell_out_params=json_sell_out_params)
            df_finance = self.compute_finance(df_finance)
            print(f'<fill_df_bel> shape of df_finance : {df_finance.shape}')
            # df_finance = df_finance.replace("THE LAUGHING COW", "LA VACHE QUI RIT")
            df_bel = pd.merge(df_bel, df_finance, on=["Brand", "Date"], how="left")
            print(f'<fill_df_bel> shape of df_bel : {df_bel.shape}')
            
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
            df_inno = self.fill_Inno_KSA(
                path=PATH_INNO,
                header=INNO_HEADER,
                brand_column_name=INNO_BRAND_COL_NAME,
                week_name=INNO_WEEK_NAME,
                columns_to_remove=INNO_COLS_TO_REMOVE,
                date_format=INNO_DATE_FORMAT,
            )
            print(f'<fill_df_bel> shape of df_inno : {df_inno.shape}')
            df_inno = self.compute_Inno(
                df=df_inno,
                date_begining=DATE_BEG,
                innovation_duration=INNOVATION_DURATION,
            )
            print(f'<fill_df_bel> shape of df_inno : {df_inno.shape}')
            df_bel = pd.merge(df_bel, df_inno, on=["Brand", "Date"], how="left")
            
            print(f'<fill_df_bel> shape of df_bel : {df_bel.shape}')
            self.add_df_bel_channel(key=channel, df=df_bel)
    
    # FINANCE
    def fill_finance_KSA(self, json_sell_out_params):
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
        ap_codes = json_sell_out_params.get(self._country).get("Finance").get("finance_codes_to_brands")
        df_finance = pd.read_excel(path, header=header)
        df_finance = df_finance.set_index(["Source", "Product"]).stack(dropna=False).reset_index()
        df_finance = df_finance.set_index(["Product", "level_2", "Source"]).unstack("Source").reset_index()
        df_finance = df_finance.reindex(columns=sorted(df_finance.columns, key=lambda x: x[::-1]))
        df_finance.columns = ['{}_{}'.format(t, v) for v,t in df_finance.columns]
        df_finance["_Product"] = df_finance["_Product"].apply(lambda x:code_to_brand(x))
        df_finance = df_finance[df_finance._Product.isin(ap_codes.values())]
        df_finance = df_finance.rename(columns={
            "_Product" : "Brand",
            "_level_2" : "Date",
            "MVC - Margin on variable costs_0" : "MVC",
            "R4100 - ADVERTISING_0" : "Advertising",
            "R4200 excluding R4220_0" : "Promotion",
        }).reset_index(drop=True)
        # df_finance.Date = pd.to_datetime(dt.datetime.strptime(df_finance.Date, "%Y-%m")).dt.strftime("")
        return df_finance

    def compute_finance(self, df_finance):
        df_finance = df_finance.replace(['    '], np.nan)
        df_finance = df_finance.fillna(0.0)
        
        df_finance["Advertising"] = df_finance["Advertising"].abs()
        df_finance["Promotion"] = df_finance["Promotion"].abs()

        df_finance["A&P"] = df_finance.apply(
            lambda x: (x.Advertising + x.Promotion) * 1000,
            axis=1,
        )
        df_finance["MVC"] = df_finance.apply(
            lambda x: x["MVC"] * 1000, axis=1
        )

        df_finance = df_finance.groupby(["Brand", "Date"]).agg(
            {
                "Advertising" : "sum", 
                "Promotion" : "sum", 
                "A&P" : "sum", 
                "MVC" : "sum"
            }
        ).reset_index()

        return df_finance[
            ["Brand", "Date", "Advertising", "Promotion", "A&P", "MVC"]
        ]

    def fill_Inno_KSA(
        self,
        path: str,
        header: list,
        brand_column_name: str,
        week_name: str,
        columns_to_remove: list,
        date_format: str,
        ):
        """Read Innovation file, rename columns

        :param path: path to Innovation file
        :param header: line in excel at which the headers are 
        :param brand_column_name: column for brands in Innovation file
        :param week_name: name of weeks in columns of Innovation file
        :param columns_to_remove: useless columns in innovation file
        :param date_format: date format in innovation columns
        :returns: df_inno

        """
        print(f"<fill_Inno> Loading data from file {path}")
        # Load innovation file and some formating
        df_ino = pd.read_excel(path, header=header, sheet_name="WSP_Sheet1")
        # display(df_ino)
        # rename Brands
        df_ino = df_ino.rename(columns={brand_column_name: "Brand"})
        # Convert columns names to date format
        cols = [x for x in df_ino.columns if week_name in x]
        df_ino = df_ino.rename(
            columns={
                x: dt.datetime.strftime(
                    dt.datetime.strptime(self.date_to_re_KSA(x, date_format), date_format), "%Y-%m"
                )
                for x in cols
            }
        )
        # remove unwanted columns
        df_ino = df_ino.drop(
            columns=[x for x in df_ino.columns if x in columns_to_remove]
        )
        return df_ino

    def date_to_re_KSA(self, string:str, date_format:str):
        look_up_table = {
            "d":"[0-9]{2}",
            "y":"[0-9]{2}",
            "Y":"[0-9]{4}",
            "m":"[0-9]{2}",
            "b":"[A-Za-z]{3}"
        }
        reg = r".*("
        char = date_format[2]
        for i in range(0, len(date_format), 3):
            reg+=look_up_table.get(date_format[i+1])
            reg+=char
        reg+='*)'
        return re.findall(reg, string)[0]
    
    
    def compute_Inno(self, df, date_begining: str, innovation_duration: int):
        #terrible terrible code, call ahmed for help
        """compute Innovation

        :param df:
        :param date_begining:
        :param innovation_duration:
        :returns: df_innovation

        """
        # Compute from innovation dataframe
        df_concat = pd.DataFrame()
        delta = relativedelta(months=+innovation_duration)
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
                        dt.datetime.strptime(first_sale, "%Y-%m") + delta
                    ).strftime("%Y-%m")
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
                dt.datetime.strptime(date_begining, "%Y-%m") + delta
            ).strftime("%Y-%m")
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