import datetime as dt
from itertools import count

import pandas as pd

from datamanager.DataManager import DataManager


class DM_GER(DataManager):
    """DataManager for French data"""

    _country = "GER"
    _df_channels = dict()
    _df_bel_channels = dict()

    def ad_hoc_GER(self, json_sell_out_params):
        """Call fill_df from parent class
        format dataframe for use in Model

        :param json_sell_out_params: json params dict
        :returns: None

        """
        df = super().fill_df(json_sell_out_params, self._country)

        sales_date_format=json_sell_out_params.get(self._country).get("sales_date_format")
        date_format = json_sell_out_params.get(self._country).get("date_format")

        df.Date = df.Date.apply(
            lambda x: dt.datetime.strptime(x[-7:] + ' 0', sales_date_format).strftime(date_format)
        )

        #AD HOC SNACK
        df.loc[df[df['Attribute'] == 'SNACK'].index, 'Category'] = 'SNACK'
        
        #AD HOC HOT CHEESE
        df.loc[df[df['Attribute'] == 'HOT CHEESE'].index, 'Category'] = 'HOT CHEESE'
        
        #AD HOC VEGAN
        df.loc[df[df['Attribute'] == 'VEGAN HOT CHEESE'].index, 'Category'] = 'VEGAN HOT CHEESE'
        df.loc[df[df['Attribute'] == 'VEGAN WITHOUT HOT CHEESE'].index, 'Category'] = 'VEGAN WITHOUT HOT CHEESE'

        #AD HOC ADLER
        df['Brand'] = df['Brand'].replace("ADLER EDELCREME", "ADLER")

        df["Sales in value"] = df["Sales in value"] * 1000
        df["Sales value with promo"] = df["Sales value with promo"] * 1000
        df["Sales in volume"] = df["Sales in volume"] * 1000
        df["Sales volume with promo"] = df["Sales volume with promo"] * 1000
        df = df[df.Date < json_sell_out_params.get(self._country).get("date_max")]
        for channel, group in df.groupby("Channel", as_index=False):
            self.add_df_channel(key=channel, df=group)
        
        self._df = df

    def fill_df_bel(self, json_sell_out_params):
        """Build df_bel

        :param json_sell_out_params: json params dict
        :returns: None

        """
        assert not self._df.empty, "df is empty, call ad_hoc_GER() or load() first"

        sales_date_format=json_sell_out_params.get(self._country).get("sales_date_format")
        date_format = json_sell_out_params.get(self._country).get("date_format")

        for channel, df in self.get_df_channels().items():
            json_sell_out_params = json_sell_out_params.copy()
            df = df.copy()
            df.Date = pd.to_datetime(df.Date)
            df.Date = df.Date.dt.strftime(date_format)
            bel_brands = json_sell_out_params.get(self._country).get("bel_brands")
            df_bel = (
                df[df.Brand.isin(bel_brands)]
                .groupby(["Date", "Brand"], as_index=False)[
                    [
                        "Price per volume",
                        "Sales in volume",
                        "Sales in value",
                        "Sales volume with promo",
                        "Weighted Distribution",
                        "Distribution",
                    ]
                ]
                .agg(
                    {
                        "Price per volume": "mean",
                        "Sales in volume": "sum",
                        "Sales in value": "sum",
                        "Sales volume with promo":"sum",
                        "Weighted Distribution": "mean",
                        "Distribution": "mean",
                    }
                )
            )

            df_bel["Promo Cost"] = df_bel["Sales volume with promo"] / df_bel["Sales in volume"]

            
            AP_CODES = json_sell_out_params.get(self._country).get("A&P_codes")
            DATE_MIN = (
                json_sell_out_params.get(self._country).get("dates_finance").get("Min")
            )
            DATE_MAX = (
                json_sell_out_params.get(self._country).get("dates_finance").get("Max")
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
            df_finance = self.compute_Finance(
                df_finance, AP_CODES, DATE_MIN, DATE_MAX, country_name=COUNTRY_NAME
            )
            df_finance = df_finance.replace("THE LAUGHING COW", "LA VACHE QUI RIT")
            df_bel = pd.merge(df_bel, df_finance, on=["Brand", "Date"], how="left")

            DATE_BEG = (
                json_sell_out_params.get(self._country).get("Inno").get("date_beg")
            )
            
            INNOVATION_DURATION = (
                json_sell_out_params.get(self._country)
                .get("Inno")
                .get("innovation_duration")
            )
            
            df_inno = self.fill_Inno_GER(
                json_sell_out_params
            )
            print(f'<fill_df_bel> shape of df_inno : {df_inno.shape}')
            df_inno = self.compute_Inno(
                df=df_inno,
                date_begining=DATE_BEG,
                innovation_duration=INNOVATION_DURATION,
            )
            df_bel = pd.merge(df_bel, df_inno, on=["Brand", "Date"], how="left")
            self.add_df_bel_channel(key=channel, df=df_bel)
            #self._df_bel = df_bel

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
        df_finance["Brand"] = df_finance["Brand"].apply(
            lambda x: "PRICES" if x == "PRICE'S" else x
        )
        df_finance["Brand"] = df_finance["Brand"].apply(
            lambda x: "BABYBEL" if x == "MINI BABYBEL" else x
        )
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

        df_concat = df_concat.groupby(["Brand", "Date"]).agg(
            {
                "Advertising" : "sum", 
                "Promotion" : "sum", 
                "A&P" : "sum", 
                "MVC" : "sum",
                "Sell-in" : "sum"
            }
        ).reset_index()

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
    
    def fill_Inno_GER(
        self,
        json_sell_out_params
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
        path = (
            json_sell_out_params.get(self._country)
            .get("dict_path")
            .get("PATH_INNO")
            .get("Total Country")
        )
        header = (
            json_sell_out_params.get(self._country).get("Inno").get("header")
        )
        brand_column_name = (
            json_sell_out_params.get(self._country)
            .get("Inno")
            .get("brand_column_name")
        )
        week_name = (
            json_sell_out_params.get(self._country).get("Inno").get("week_name")
        )
        columns_to_remove = (
            json_sell_out_params.get(self._country)
            .get("Inno")
            .get("columns_to_remove")
        )
        date_format_re = (
            json_sell_out_params.get(self._country).get("Inno").get("date_format_re")
        )
        date_format = (
            json_sell_out_params.get(self._country).get("Inno").get("date_format")
        )
        print(f"<fill_Inno> Loading data from file {path}")
        # Load innovation file and some formating
        df_ino = pd.read_excel(path, header=header)
        # rename Brands
        df_ino = df_ino.rename(columns={brand_column_name: "Brand"})
        # Remove 'all categories'
        try:
            df_ino = df_ino[~df_ino["Product"].str.contains("ALL CATEGORIES")]
        except KeyError:
            pass
        # Convert columns names to date format
        cols = [x for x in df_ino.columns if week_name in x]
        df_ino = df_ino.rename(
            columns={
                x: dt.datetime.strftime(
                    dt.datetime.strptime(self.date_to_re(x, date_format_re)+' 0', date_format), "%Y-%m-%d"
                )
                for x in cols
            }
        )
        # remove unwanted columns
        df_ino = df_ino.drop(
            columns=[x for x in df_ino.columns if x in columns_to_remove]
        )
        return df_ino
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
