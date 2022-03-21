import datetime as dt
from itertools import count

import pandas as pd

from datamanager.DataManager import DataManager


class DM_FR(DataManager):
    """DataManager for French data"""

    _country = "FR"
    _df_channels = dict()
    _df_bel_channels = dict()

    def ad_hoc_FR(self, json_sell_out_params):
        """Call fill_df from parent class
        format dataframe for use in Model

        :param json_sell_out_params: json params dict
        :returns: None

        """
        df = super().fill_df(json_sell_out_params, self._country)
        """
        Spécialité : Must have (compétition à valider pour Port Salut et Cousteron)
        Frais à tartiner : Must have
        Enfant : Must have
        Tranche à froid : Must have
        """

        df.Date = df.Date.apply(
            lambda x: dt.datetime.strptime(x[-10:], "%d-%m-%Y").strftime("%Y-%m-%d")
        )

        #AD HOC SPECIALITE
        df.loc[df[(df['Sub Category'].isin(['AOR REGIONAL'])) & (~df['Category'].isin(['ALTERNATIVE VEGETALE']))].index, 'Category'] = 'SPECIALITE AOR REGIONAL'
        df.loc[df[(df['Sub Category'].isin(['SPECIALITE A GOUT DOUX'])) & (~df['Category'].isin(['ALTERNATIVE VEGETALE']))].index, 'Category'] = 'SPECIALITE A GOUT DOUX'

        #AD HOC FRAIS A TARTINER
        df.loc[df[(df['Sub Category'].isin(['ARO'])) & (~df['Category'].isin(['ALTERNATIVE VEGETALE']))].index, 'Category'] = 'FRAIS A TARTINER ARO'
        df.loc[df[(df['Sub Category'].isin(['NATURE'])) & (~df['Category'].isin(['ALTERNATIVE VEGETALE']))].index, 'Category'] = 'FRAIS A TARTINER NATURE'

        #AD HOC ENFANT
        df.loc[df[(df['Sub Category'].isin(['A TARTINER'])) & (~df['Category'].isin(['ALTERNATIVE VEGETALE']))].index, 'Category'] = 'ENFANT A TARTINER'
        df.loc[df[(df['Sub Category'].isin(['NOMADE'])) & (~df['Category'].isin(['ALTERNATIVE VEGETALE']))].index, 'Category'] = 'ENFANT NOMADE'

        #AD HOC TRANCHE A FROID
        df.loc[df[(df['Sub Category'].isin(['A GOUT'])) & (~df['Category'].isin(['ALTERNATIVE VEGETALE']))].index, 'Category'] = 'TRANCHE A FROID A GOUT'
        df.loc[df[(df['Sub Category'].isin(['CHEVRE&BREBIS'])) & (~df['Category'].isin(['ALTERNATIVE VEGETALE']))].index, 'Category'] = 'TRANCHE A FROID CHEVRE&BREBIS'
        df.loc[df[(df['Sub Category'].isin(['ORIGINAL'])) & (~df['Category'].isin(['ALTERNATIVE VEGETALE']))].index, 'Category'] = 'TRANCHE A FROID ORIGINAL'

        df = df[df.Date < json_sell_out_params.get(self._country).get("date_max")]
        for channel, group in df.groupby("Channel", as_index=False):
            self.add_df_channel(key=channel, df=group)
        self._df = df

    def fill_df_bel(self, json_sell_out_params):
        """Build df_bel

        :param json_sell_out_params: json params dict
        :returns: None

        """
        assert not self._df.empty, "df is empty, call ad_hoc_USA() or load() first"

        for channel, df in self.get_df_channels().items():
            json_sell_out_params = json_sell_out_params.copy()
            df = df.copy()
            df.Date = pd.to_datetime(df.Date)
            df.Date = df.Date.dt.strftime("%Y-%m-%d")
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
                json_sell_out_params=json_sell_out_params,
                country=self._country
            )
            # df_finance = self.fill_Finance(
            #     path=PATH_FINANCE,
            #     finance_cols=FINANCE_COLS,
            #     finance_renaming_columns=FINANCE_RENAMING_COLS,
            #     header=FINANCE_HEADER,
            # )
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

    def compute_Finance(
        self, df_finance, aandp_codes, date_min, date_max, country_name: str
    ):
        """Many things done here, gl debugging it

        :param df_finance:
        :param aandp_codes:
        :param date_min:
        :param date_max:
        :param country_name:
        :returns:

        """
        # Compute from Finance dataframe
        df_finance = df_finance[df_finance["Country"] == country_name]
        # Filter brands for the study
        df_finance = df_finance[df_finance["Brand"].isin(aandp_codes)]
        # ABS for Advertising and Promotion
        df_finance["Advertising"] = df_finance["Advertising"].abs()
        df_finance["Promotion"] = df_finance["Promotion"].abs()
        # Get Brand from code
        df_finance["Brand"] = df_finance["Brand"].apply(
            lambda x: x.split(sep="-")[-1].strip()
        )
        #### ADHOC FOR PRICES and BABYBEL
        # df_finance["Brand"] = df_finance["Brand"].apply(
        #     lambda x: "BABYBEL" if x == "MINI BABYBEL" else x
        # )
        df_finance = df_finance.fillna(0.0)
        # Months to week
        df_finance["number of weeks"] = df_finance.apply(
            lambda x: self.count_num_sundays_in_month(x.Year, x.Month), axis=1
        )
        df_finance["A&P"] = df_finance.apply(
            lambda x: (x.Advertising + x.Promotion) / x["number of weeks"] * 1000,
            axis=1,
        )
        df_finance["Sell-in"] = df_finance.apply(
            lambda x: x["Sell-in"] / x["number of weeks"] * 1000, axis=1
        )
        df_finance["MVC"] = df_finance.apply(
            lambda x: x["MVC"] / x["number of weeks"] * 1000, axis=1
        )
        # Duplicate for n weeks
        full_idx = pd.date_range(start=date_min, end=date_max, freq="W")
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
        df_concat["Date"] = df_concat["Date"].apply(
            lambda x: dt.datetime.strftime(x, "%Y-%m-%d")
        )
        return df_concat[
            ["Brand", "Date", "Sell-in", "Advertising", "Promotion", "A&P", "MVC"]
        ]

    def compute_Inno(self, df, date_begining: str, innovation_duration: int):
        """Compute Innovation

        :param df:
        :param date_begining:
        :param innovation_duration:
        :returns:

        """
        # Compute from innovation dataframe
        df_concat = pd.DataFrame()
        delta = dt.timedelta(weeks=innovation_duration)
        # for each brand
        for brand, group in df.groupby(["Brand"]):
            # init df
            df_merge = pd.DataFrame(index=group.columns.values[:-1])
            group = group.drop("Brand", axis=1)
            # Find date of first sale for each product
            for col in group.T.columns:
                first_sale = group.T[col][pd.notna(group.T[col])].index.values[0]
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
            df_innovation = df_innovation.div(group.sum(axis=0), axis=0)
            df_innovation.loc[:, "Brand"] = brand
            df_innovation = df_innovation.reset_index().rename(
                columns={"index": "Date"}
            )
            df_innovation = df_innovation[df_innovation["Date"] != "Brand"]
            df_concat = pd.concat([df_concat, df_innovation])

        return df_concat[["Brand", "Date", "Rate of Innovation"]]

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
