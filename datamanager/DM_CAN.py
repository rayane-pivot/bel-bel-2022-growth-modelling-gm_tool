import datetime as dt

import numpy as np
import pandas as pd

from datamanager.DataManager import DataManager

# from DataManager import DataManager


class DM_CAN(DataManager):
    """DataManager for CAN data"""

    _country = "CAN"

    def ad_hoc_CAN(self, json_sell_out_params):
        df = super().fill_df(json_sell_out_params, self._country)
        
        # AD HOC PLANT BASED
        df.loc[
            df[
                df["FB CHEESE TYP"].str.contains("PLANT BASED")
            ].index,
            "FB CHEESE TYP",
        ] = "PLANT BASED"
        date_cols = [col for col in df.columns if json_sell_out_params.get(self._country).get("sales_col_week_name") in col]
        feature_cols = json_sell_out_params.get(self._country).get("sales_col_names")
        df_concat = pd.DataFrame()
        k=0
        for i , group in df.groupby(feature_cols):

            group = group.set_index("Feature")
            df_dates = group[date_cols].T.reset_index()
            df_dates = df_dates.rename(columns={'index':'Date'})
            group = group[feature_cols].reset_index(drop=True)
            df_dates = pd.concat([pd.DataFrame(group.iloc[0]).T, df_dates], axis=1)
            df_dates.loc[:, feature_cols] = group.values[0]
            try:
                df_concat = pd.concat([df_concat, df_dates])
            except Exception:
                k=k+1
                continue
        cat_names = json_sell_out_params.get(self._country).get("sales_renaming_columns_dict").get("CATEGORICALS")
        kpi_names = json_sell_out_params.get(self._country).get("sales_renaming_columns_dict").get("KPIS")
        
        df_concat = df_concat.rename(columns=dict(cat_names, **kpi_names))
        df_concat = df_concat.reset_index(drop=True)

        sales_date_format=json_sell_out_params.get(self._country).get("sales_date_format")
        date_format = json_sell_out_params.get(self._country).get("date_format")
        df_concat.Date = df_concat.Date.apply(lambda x:dt.datetime.strptime(''.join(x.strip()[3:]), sales_date_format).strftime(date_format))

        ###FILTER FOR PLATTER AND SPECIALTY ONLY
        # df_concat = df_concat[df_concat['Category'].isin(['PLATTER', 'SPECIALTY'])]
        # df_concat.loc[df_concat[df_concat['Category'] == 'PLATTER'].index, 'Category'] = df_concat.loc[df_concat[df_concat['Category'] == 'PLATTER'].index, 'Sub Category']
        # df_concat.loc[df_concat[df_concat['Category'] == 'SPECIALTY'].index, 'Category'] = df_concat.loc[df_concat[df_concat['Category'] == 'SPECIALTY'].index, 'Sub Category']

        ###FILTER FOR PLATTER ONLY
        #df_concat = df_concat[df_concat['Category'] == 'PLATTER']
        #df_concat.loc[df_concat[df_concat['Category'] == 'PLATTER'].index, 'Category'] = df_concat.loc[df_concat[df_concat['Category'] == 'PLATTER'].index, 'Sub Category']
        #df_concat.loc[df_concat[df_concat['Category'] == 'SPECIALTY'].index, 'Category'] = df_concat.loc[df_concat[df_concat['Category'] == 'SPECIALTY'].index, 'Sub Category']
        self._df = df_concat


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
        print(f'<fill_df_bel> shape of df_bel : {df_bel.shape}')
        print(f'<fill_df_bel> Dates : {df_bel.Date.min()}, {df_bel.Date.max()}')
        df_finance = self.fill_Finance(json_sell_out_params, country=self._country)
        df_finance = self.compute_Finance(df_finance, json_sell_out_params)
        print(f'<fill_df_bel> shape of df_finance : {df_finance.shape}')
        print(f'<fill_df_bel> Dates : {df_finance.Date.min()}, {df_finance.Date.max()}')
        df_bel = pd.merge(df_bel, df_finance, on=["Brand", "Date"], how="left").fillna(0.0)
        print(f'<fill_df_bel> shape of df_bel : {df_bel.shape}')
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
        print(f'<fill_df_bel> shape of df_inno : {df_inno.shape}')
        
        df_bel = pd.merge(df_bel, df_inno, on=["Brand", "Date"], how="left")

        df_promocost = self.compute_Promo_Cost(
            json_sell_out_params=json_sell_out_params
        )
        print(f'<fill_df_bel> shape of df_promocost : {df_promocost.shape}')
        print(f'<fill_df_bel> Dates : {df_promocost.Date.min()}, {df_promocost.Date.max()}')
        df_bel = pd.merge(df_bel, df_promocost, on=["Brand", "Date"], how="left").fillna(0.0)
        print(f'<fill_df_bel> shape of df_bel : {df_bel.shape}')
        self._df_bel = df_bel

    def compute_Finance(self, df_finance, json_sell_out_params):
        """compute df_finance

        :param df_finance:
        :param aandp_codes:
        :param date_min:
        :param date_max:
        :returns: df_finance

        """
        def fn(r):
            return any(word in r['Brand Code'] for word in aandp_codes)
        def code_to_brand(r):
            for code, brand in finance_codes_to_brands.items():
                if code in r:
                    return brand
            return ""
        date_min = (
            json_sell_out_params.get(self._country).get("dates_finance").get("Min")
        )
        date_max = (
            json_sell_out_params.get(self._country).get("dates_finance").get("Max")
        )
        aandp_codes = json_sell_out_params.get(self._country).get("A&P_codes")
        finance_codes_to_brands = json_sell_out_params.get(self._country).get("Finance").get("finance_codes_to_brands")
        # Filter brands for the study
        df_finance = df_finance[df_finance.apply(lambda x:fn(x), axis=1)]
        df_finance["Brand"] = df_finance["Brand Code"].apply(lambda x:code_to_brand(x))
        df_finance.Spendings = df_finance.Spendings.abs()
        df_finance = df_finance.groupby(['Brand', 'Year', 'Month']).agg('sum').reset_index().rename(columns={"Spendings":"A&P"})
        df_finance["number of weeks"] = df_finance.apply(lambda x: self.count_num_sundays_in_month(x.Year, x.Month), axis=1)
        df_finance["A&P"] = df_finance.apply(lambda x: x['A&P'] / x["number of weeks"] * 1000,axis=1)
        full_idx = pd.date_range(start=date_min, end=date_max, freq="7D")
        df_test = pd.DataFrame(index=full_idx)
        df_test["Year"] = df_test.index.year
        df_test["Month"] = df_test.index.month
       
        df_concat = pd.DataFrame()
        for brand in df_finance["Brand"].unique():
            df_concat = pd.concat(
                [
                    df_concat,
                    pd.merge(
                        df_finance[df_finance["Brand"] == brand],
                        df_test.reset_index(),
                        on=["Year", "Month"],
                        how="right"
                    ).rename(columns={"index": "Date"}),
                ]
            )
        df_concat["Date"] = df_concat["Date"].apply(
                lambda x: dt.datetime.strftime(x, "%Y-%m-%d")
            )
        
        return df_concat[["Brand", "Date", "A&P"]]

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
            # divide innovations by total sales
            df_innovation = df_innovation.div(group.T.sum(axis=1), axis=0)
            df_innovation.loc[date_begining:date_begining_to_delta] = 0.0
            df_innovation.loc[:, "Brand"] = brand
            df_innovation = df_innovation.reset_index().rename(
                columns={"index": "Date"}
            )
            df_innovation = df_innovation[df_innovation["Date"] != "Brand"]
            df_concat = pd.concat([df_concat, df_innovation])

        return df_concat[["Brand", "Date", "Rate of Innovation"]].fillna(0.0)
    
    def compute_Promo_Cost(self, json_sell_out_params):
        """Compute Cost of Promotions

        :param path:
        :param renaming_brands:
        :param features:
        :returns: df_promocost

        """

        path = (
            json_sell_out_params.get(self._country)
            .get("dict_path")
            .get("PATH_PROMO_COST")
            .get("Total Country")
        )
        renaming_brands = (
            json_sell_out_params.get(self._country)
            .get("Promo Cost")
            .get("renaming_brands")
        )
        features = (
            json_sell_out_params.get(self._country).get("Promo Cost").get("features")
        )
        date_min = (
            json_sell_out_params.get(self._country).get("Promo Cost").get("dates_promo").get("Min")
        )
        date_max = (
            json_sell_out_params.get(self._country).get("Promo Cost").get("dates_promo").get("Max")
        )
        # for any question, ask ahmed@pivotandco.com
        df_promo = pd.read_excel(path, engine="openpyxl")
        renaming_brands = renaming_brands
        df_promo_bel = df_promo[df_promo.PH3.isin(renaming_brands.keys())][features]
        # rename brands
        df_promo_bel["PH3"] = df_promo_bel["PH3"].map(renaming_brands)
        #Rename Promo cost column to Cost
        df_promo_res = df_promo_bel.rename(columns={'06.OI_Promo_Sell_In_CA':'Cost'})
        df_promo_res.Cost = df_promo_res.Cost
        # We want 52 weeks, so we merge week 53 with week 52
        df_promo_res["Week"] = df_promo_res["Week"].replace({53: 52})
        # group by ...
        df_promo_res = (
            df_promo_res.groupby(["PH3", "Year", "Month", "Week"]).agg(np.sum).reset_index()
        )
        # # NURISHH is a special case
        # df_promo_nurishh = df_promo_res[df_promo_res.PH3 == "NURISHH"]
        # df_promo_res = df_promo_res[df_promo_res.PH3 != "NURISHH"]
        # df_promo_res["Date"] = (
        #     list(
        #         pd.date_range(
        #             start=date_min, end=date_max, freq="7D"
        #         ).strftime("%Y-%m-%d")
        #     )
        #     * df_promo_res.PH3.nunique()
        # )
        # df_promo_nurishh["Date"] = df_promo_res["Date"][
        #     -df_promo_nurishh.shape[0] :
        # ].values
        # df_promo_res = pd.concat([df_promo_res, df_promo_nurishh])
        
        full_idx = pd.date_range(start=date_min, end=date_max, freq="7D")
        df_test = pd.DataFrame(index=full_idx)
        df_test["Year"] = df_test.index.year
        df_test["Month"] = df_test.index.month
        df_test["Week"] = df_test.index.isocalendar().week
        df_test["Week"] = df_test["Week"].apply(lambda x:int(x))
        df_concat = pd.DataFrame()
        for brand in df_promo_res["PH3"].unique():
            df_concat = pd.concat(
                [
                    df_concat,
                    pd.merge(
                        df_promo_res[df_promo_res["PH3"] == brand],
                        df_test.reset_index(),
                        on=["Year", "Month", "Week"],
                        how="right"
                    ).rename(columns={"index": "Date"}),
                ]
            )
        df_concat["Date"] = df_concat["Date"].apply(
                lambda x: dt.datetime.strftime(x, "%Y-%m-%d")
            )
        
        df_concat = df_concat.rename(
            columns={"PH3": "Brand", "Cost": "Promo Cost"}
        )[["Brand", "Promo Cost", "Date"]]
        return df_concat