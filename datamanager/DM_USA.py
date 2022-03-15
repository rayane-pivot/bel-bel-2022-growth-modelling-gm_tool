import datetime as dt

import numpy as np
import pandas as pd

from datamanager.DataManager import DataManager

# from DataManager import DataManager


class DM_USA(DataManager):
    """DataManager for US data"""

    _country = "USA"

    def ad_hoc_USA(self, json_sell_out_params):
        """Call fill_df from parent class
        format dataframe for use in Model

        :param json_sell_out_params: json params dict
        :returns: None

        """
        df = super().fill_df(json_sell_out_params, self._country)

        # DATES FORMATIING
        df = df[~df["Date"].str.contains("OK")]
        df["Date"] = df["Date"].apply(
            lambda x: dt.datetime.strptime(x.split()[-1], "%m-%d-%y").strftime(
                "%Y-%m-%d"
            )
        )

        # AD HOC PLANT BASED
        df.loc[
            df[
                df["Market"].isin(["PLANT BASED CHEESE", "PLANT BASED CREAM CHEESE"])
            ].index,
            "Category",
        ] = "PLANT BASED"
        # df.loc[df[df['Market'].isin(['PLANT BASED CHEESE', 'PLANT BASED CREAM CHEESE'])].index, 'Sub Category'] = 'PLANT BASED'

        # AD HOC CREAM CHEESE
        df.loc[
            df[
                df["Sub Category"].isin(
                    [
                        "FLAVORED TUBS",
                        "FLAVORED WHIPPED TUBS",
                        "PLAIN TUBS",
                        "PLAIN WHIPPED TUBS",
                    ]
                )
            ].index,
            "Sub Category",
        ] = "CREAM CHEESE TUBS"

        df.loc[
            df[
                df["Sub Category"].isin(
                    [
                        "SINGLE SERVE FLAVORED CREAM CHEESE",
                        "SINGLE SERVE PLAIN CREAM CHEESE",
                    ]
                )
            ].index,
            "Sub Category",
        ] = "SINGLE SERVE"

        df.loc[
            df[df["Market"].isin(["CREAM CHEESE"])].index, "Category"
        ] = "CREAM CHEESE"

        # AD HOC GOURMET
        df.loc[
            df[
                (df["Sub Category"].isin(["GOURMET BLOCK / WEDGE / ROUND"]))
                & (~df["Category"].isin(["PLANT BASED"]))
            ].index,
            "Category",
        ] = "GOURMET BLOCK / WEDGE / ROUND"
        df.loc[
            df[
                (df["Sub Category"].isin(["GOURMET CRUMBLED"]))
                & (~df["Category"].isin(["PLANT BASED"]))
            ].index,
            "Category",
        ] = "GOURMET CRUMBLED"
        df.loc[
            df[
                (df["Sub Category"].isin(["GOURMET FRESH ITALIAN"]))
                & (~df["Category"].isin(["PLANT BASED"]))
            ].index,
            "Category",
        ] = "GOURMET FRESH ITALIAN"
        df.loc[
            df[
                (df["Sub Category"].isin(["GOURMET SHREDDED / GRATED"]))
                & (~df["Category"].isin(["PLANT BASED"]))
            ].index,
            "Category",
        ] = "GOURMET SHREDDED / GRATED"
        df.loc[
            df[
                (df["Sub Category"].isin(["GOURMET SPREADS"]))
                & (~df["Category"].isin(["PLANT BASED"]))
            ].index,
            "Category",
        ] = "GOURMET SPREADS"

        # AD HOC DROP HYBRID FROM MARKET
        df = df[~df["Market"].isin(["HYBRID CHEESE"])]

        # DROP USELESS COLUMNS
        df = df.drop(columns=["Product", "Channel"])

        # ADD PERIODS
        df_periods = (
            pd.DataFrame(np.sort(df.Date.unique()))
            .reset_index()
            .rename(columns={"index": "Period", 0: "Date"})
        )
        df_periods.Period = df_periods.Period + 1
        df = pd.merge(df, df_periods, on="Date", how="left")

        # DROP DUPLICATES
        df = df[
            ~df.round(3).duplicated(
                subset=[
                    "Date",
                    "Brand",
                    "Sales in value",
                    "Sales value with promo",
                    "Sales in volume",
                    "Sales volume with promo",
                    "Price without promo",
                    "Price with promo",
                    "Price per volume",
                    "Distribution",
                ],
                keep="last",
            )
        ]

        df = df.reset_index(drop=True)
        self._df = df

    def fill_df_bel(self, json_sell_out_params):
        """build df_bel

        :param json_sell_out_params: json params dict
        :returns: None

        """
        assert not self._df.empty, "df is empty, call ad_hoc_USA() or load() first"
        df = self._df.copy()
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
        AP_CODES = json_sell_out_params.get(self._country).get("A&P_codes")
        DATE_MIN = (
            json_sell_out_params.get(self._country).get("dates_finance").get("Min")
        )
        DATE_MAX = (
            json_sell_out_params.get(self._country).get("dates_finance").get("Max")
        )
        df_finance = self.fill_Finance(json_sell_out_params, country=self._country)
        df_finance = self.compute_Finance(df_finance, AP_CODES, DATE_MIN, DATE_MAX)
        df_bel = pd.merge(df_bel, df_finance, on=["Brand", "Date"], how="left")

        PATH_INNO = (
            json_sell_out_params.get(self._country)
            .get("dict_path")
            .get("PATH_INNO")
            .get("Total Country")
        )
        DATE_BEG = json_sell_out_params.get(self._country).get("Inno").get("date_beg")
        INNO_HEADER = json_sell_out_params.get(self._country).get("Inno").get("header")
        INNOVATION_DURATION = (
            json_sell_out_params.get(self._country)
            .get("Inno")
            .get("innovation_duration")
        )
        INNO_BRAND_COL_NAME = (
            json_sell_out_params.get(self._country).get("Inno").get("brand_column_name")
        )
        INNO_WEEK_NAME = (
            json_sell_out_params.get(self._country).get("Inno").get("week_name")
        )
        INNO_COLS_TO_REMOVE = (
            json_sell_out_params.get(self._country).get("Inno").get("columns_to_remove")
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
            df=df_inno, date_begining=DATE_BEG, innovation_duration=INNOVATION_DURATION
        )
        df_bel = pd.merge(df_bel, df_inno, on=["Brand", "Date"], how="left")

        PATH_PROMOCOST = (
            json_sell_out_params.get(self._country)
            .get("dict_path")
            .get("PATH_PROMO_COST")
            .get("Total Country")
        )
        RENAMING_BRANDS = (
            json_sell_out_params.get(self._country)
            .get("Promo Cost")
            .get("renaming_brands")
        )
        FEATURES = (
            json_sell_out_params.get(self._country).get("Promo Cost").get("features")
        )
        df_promocost = self.compute_Promo_Cost(
            path=PATH_PROMOCOST, renaming_brands=RENAMING_BRANDS, features=FEATURES
        )
        df_bel = pd.merge(df_bel, df_promocost, on=["Brand", "Date"], how="left")

        PATH_HH = (
            json_sell_out_params.get(self._country)
            .get("dict_path")
            .get("PATH_HH_INDEX")
            .get("Total Country")
        )
        HH_HEADER = (
            json_sell_out_params.get(self._country).get("HH Index").get("header")
        )
        df_hh = self.compute_HH_Index(path=PATH_HH, header=HH_HEADER)
        df_bel = pd.merge(df_bel, df_hh, on=["Brand", "Date"], how="left")

        self._df_bel = df_bel

    def compute_Finance(self, df_finance, aandp_codes, date_min, date_max):
        """compute df_finance

        :param df_finance:
        :param aandp_codes:
        :param date_min:
        :param date_max:
        :returns: df_finance

        """
        # Compute from Finance dataframe
        df_finance = df_finance[df_finance["Country"] == self._country]
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
        return df_concat[
            ["Brand", "Date", "Sell-in", "Advertising", "Promotion", "A&P", "MVC"]
        ]

    def compute_Inno(self, df, date_begining: str, innovation_duration: int):
        """compute Innovation

        :param df:
        :param date_begining:
        :param innovation_duration:
        :returns: df_innovation

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
            df_innovation = df_innovation.div(group.T.sum(axis=1), axis=0)
            df_innovation.loc[:, "Brand"] = brand
            df_innovation = df_innovation.reset_index().rename(
                columns={"index": "Date"}
            )
            df_innovation = df_innovation[df_innovation["Date"] != "Brand"]
            df_concat = pd.concat([df_concat, df_innovation])

        return df_concat[["Brand", "Date", "Rate of Innovation"]]

    def compute_Promo_Cost(self, path: str, renaming_brands: dict, features: list):
        """Compute Cost of Promotions

        :param path:
        :param renaming_brands:
        :param features:
        :returns: df_promocost

        """
        # for any question, ask ahmed@pivotandco.com
        df_promo = pd.read_excel(path, engine="openpyxl")
        renaming_brands = renaming_brands
        df_promo_bel = df_promo[df_promo.PH3.isin(renaming_brands.keys())][features]
        df_promo_refri = df_promo[df_promo.PH3 == "Refrigerated Spreads"]
        # swap PH3 and PH4 columns (KAUKAUNA and MERKTS)
        df_promo_refri = df_promo_refri[df_promo_refri.PH4 != "Owls Nest"].rename(
            columns={"PH3": "PH4", "PH4": "PH3"}
        )[features]
        # concat to have all brands on same columns
        df_promo_res = pd.concat([df_promo_bel, df_promo_refri])
        # rename brands
        df_promo_res["PH3"] = df_promo_res["PH3"].map(renaming_brands)
        # sum promo columns
        df_promo_res["Cost"] = df_promo_res[
            ["09.Promo_OOI_USA", "06.OI_Promo_USA"]
        ].sum(axis=1)
        # We want 52 weeks, so we merge week 53 with week 52
        df_promo_res["Week"] = df_promo_res["Week"].replace({53: 52})
        # group by ...
        df_promo_res = (
            df_promo_res.groupby(["PH3", "Year", "Week"]).agg(np.sum).reset_index()
        )
        # NURISHH is a special case
        df_promo_nurishh = df_promo_res[df_promo_res.PH3 == "NURISHH"]
        df_promo_res = df_promo_res[df_promo_res.PH3 != "NURISHH"]
        df_promo_res["Date"] = (
            list(
                pd.date_range(
                    start="2019-01-01", end="2021-12-31", freq="W-SUN"
                ).strftime("%Y-%m-%d")
            )
            * df_promo_res.PH3.nunique()
        )
        df_promo_nurishh["Date"] = df_promo_res["Date"][
            -df_promo_nurishh.shape[0] :
        ].values
        df_promo_res = pd.concat([df_promo_res, df_promo_nurishh])
        df_promo_res = df_promo_res.rename(
            columns={"PH3": "Brand", "Cost": "Promo Cost"}
        )[["Brand", "Promo Cost", "Date"]]
        return df_promo_res

    def compute_HH_Index(self, path: str, header: list):
        """Compute Household penetration index

        :param path:
        :param header:
        :returns: df_hh

        """
        # cut excel file to ndarremove half the columns, keep HH index
        df_h = pd.read_excel(path, engine="openpyxl", header=header).iloc[:, 1:210]
        # remove ALL BRANDS
        df_hh = df_h[df_h["Unnamed: 1"] != "Total All Products"].fillna(0)
        # group by brand
        df_hh = df_hh.groupby("Unnamed: 1").agg(np.mean)
        # create dates
        df_hh.columns = pd.date_range(
            start="2018-01-07", end="2021-12-31", freq="W-SUN"
        ).strftime("%Y-%m-%d")
        df_hh = (
            pd.DataFrame(df_hh.stack())
            .reset_index()
            .rename(columns={"Unnamed: 1": "Brand", "level_1": "Date", 0: "HH"})
        )
        # rename brand
        df_hh["Brand"] = df_hh["Brand"].apply(lambda x: x.strip())
        return df_hh