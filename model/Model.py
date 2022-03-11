import json
import math
from tokenize import Number

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, minmax_scale


class Model:
    def brand_pos(self, df, year_min: int):
        """Brand Positioning Matrix
        pivot df to return Brand/Category/Sales in volume

        :param df:
        :param year_min:
        :returns:

        """
        df = df.copy()
        df.Date = pd.to_datetime(df.Date)
        df = df[df.Date.dt.year >= year_min]
        brand_positioning_matrix = pd.pivot_table(
            df,
            values="Sales in volume",
            columns="Category",
            index="Brand",
            aggfunc="sum",
        )

        brand_positioning_matrix.columns.name = None
        brand_positioning_matrix.index.name = "Brand"
        return brand_positioning_matrix.div(1000)

    def cat_yearly_sales(self, df, on: str):
        """Brand Positioning Matrix
        compute sales in volume per year

        :param df:
        :param on:
        :returns:

        """

        df = df.copy()
        df.Date = pd.to_datetime(df.Date)

        df_grp = (
            df.groupby([on, pd.Grouper(key="Date", freq="Y")])["Sales in volume"]
            .agg("sum")
            .reset_index()
        )
        df_grp.Date = df_grp.Date.dt.year
        table = pd.pivot_table(df_grp, columns="Category", index="Date", values="Sales in volume")
        table=table.div(1000)

        return table

    def growth(self, df, on: str, year1: int, year2: int):
        """Brand Positioning Matrix
        compute growth and cagr for brands and categories

        :param df:
        :param on:
        :param year1:
        :param year2:
        :returns:

        """

        def BPM_cagr(x):
            """local function for groupby.apply
            compute compund annual growth rate

            :param x:
            :returns:

            """
            y1 = (
                x[x.Date.dt.year == year1]["Sales in volume"].values[0]
                if len(x[x.Date.dt.year == year1]["Sales in volume"].values) > 0
                else None
            )
            y2 = (
                x[x.Date.dt.year == year2]["Sales in volume"].values[0]
                if len(x[x.Date.dt.year == year2]["Sales in volume"].values) > 0
                else None
            )

            if (y1 is None) or (y2 is None) or (y1==0):
                return 0.0
            return (math.pow((y2 / y1), (1 / (year2 - year1 + 1))) - 1) * 100

        def apply_growth(x):
            """local function for groupby.apply

            :param x:
            :returns: sales in volume year2 - year1

            """
            y1 = (
                x[x.Date.dt.year == year1]["Sales in volume"].values[0]
                if len(x[x.Date.dt.year == year1]["Sales in volume"].values) > 0
                else 0.0
            )
            y2 = (
                x[x.Date.dt.year == year2]["Sales in volume"].values[0]
                if len(x[x.Date.dt.year == year2]["Sales in volume"].values) > 0
                else 0.0
            )
            return y2 - y1

        df = df.copy()
        df.Date = pd.to_datetime(df.Date)

        df_grp = (
            df.groupby([on, pd.Grouper(key="Date", freq="Y")])["Sales in volume"]
            .agg("sum")
            .reset_index()
        )
        growth = (
            df_grp.groupby(on)
            .apply(apply_growth)
            .div(1000)
            .reset_index()
            .rename(columns={0: "GROWTH"})
        )
        growth["CAGR"] = df_grp.groupby(on).apply(BPM_cagr).reset_index().loc[:, 0]
        
        return growth.set_index(on)

    def compute_brand_positioning_matrix(
        self, df, year_min: int, year1: int, year2: int
    ):
        """Compute brand positioning matrix with Sales in volume
        from year_min.
        Cagr and growth are from year1 to year2

        :param df:
        :param year_min:
        :param year1:
        :param year2:
        :returns: brand_positioning_matrix

        """
        # Compute brand positioning matrix, cagr and growth
        brand_positioning_matrix = self.brand_pos(df, year_min=year_min)
        growth_brand = self.growth(df, on="Brand", year1=year1, year2=year2)
        growth_category = self.growth(df, on="Category", year1=year1, year2=year2)

        cat_yearly_sales = self.cat_yearly_sales(df, on="Category")

        # Concat brand positioning matrix, cagr and growth
        brand_positioning_matrix = pd.concat(
            [brand_positioning_matrix, growth_category.T]
        )
        brand_positioning_matrix = pd.concat(
            [brand_positioning_matrix, cat_yearly_sales]
        )
        brand_positioning_matrix[growth_brand.columns] = growth_brand[
            growth_brand.columns
        ]
        return brand_positioning_matrix

    def from_df_bel(self, df_bel, year: int):
        """Attack_init_state
        compute columns from df_bel

        :param df_bel:
        :param year:
        :returns:

        """

        def ap(x):
            """local function for groupby apply

            :param x:
            :returns: sum of A&P for year

            """
            if "A&P" in x:
                return x[x.Date.dt.year == year]["A&P"].agg("sum")

        def price(x):
            """local function for groupby apply

            :param x:
            :returns: average of Price per volume for year

            """
            if "Price per volume" in x.columns:
                return x[x.Date.dt.year == year]["Price per volume"].agg("mean")

        def promo(x):
            """local function for groupby apply

            :param x:
            :returns: sum of Promo Cost for year

            """
            if "Promo Cost" in x.columns:
                return x[x.Date.dt.year == year]["Promo Cost"].agg("sum")

        def size(x):
            """local function for groupby apply

            :param x:
            :returns: sum of Sales in volume for year

            """
            if "Sales in volume" in x.columns:
                return x[x.Date.dt.year == year]["Sales in volume"].agg("sum")

        df_bel = df_bel.copy()
        df_bel.Date = pd.to_datetime(df_bel.Date)
        df_grp = (
            df_bel.groupby(["Brand", pd.Grouper(key="Date", freq="Y")])
            .apply(
                lambda r: pd.Series(
                    {"A&P 2021": ap(r), "Price 2021": price(r), "Promo 2021": promo(r), "Size 2021": size(r)}
                )
            )
            .reset_index()
        )
        return df_grp.loc[df_grp[df_grp.Date.dt.year == year].index, :]

    def compute_attack_init_state(self, df, df_bel, json_sell_out_params, country: str):
        """compute attack init state

        :param df:
        :param df_bel:
        :param json_sell_out_params:
        :param country:
        :returns: df_attack_init_state

        """
        year = json_sell_out_params.get(country).get("attack_init_state").get("year")
        bel_brands = json_sell_out_params.get(country).get("bel_brands")

        df_temp = self.from_df_bel(df_bel, year=year)

        df = df.copy()
        df.Date = pd.to_datetime(df.Date)
        distrib = (
            df[df.Brand.isin(bel_brands)]
            .groupby(["Brand", pd.Grouper(key="Date", freq="Y")])["Distribution"]
            .mean()
            .reset_index()
        )
        distrib = distrib.loc[distrib[distrib.Date.dt.year == year].index, :]

        df_attack_init_state = pd.merge(df_temp, distrib, on=["Brand", "Date"])
        df_attack_init_state = df_attack_init_state.drop(columns=["Date"])
        return df_attack_init_state

    def compute_brand_scorecard(self, df, df_bel, json_sell_out_params, country):
        """Compute Brand Scorecard

        :param df:
        :param df_bel:
        :param json_sell_out_params:
        :param country:
        :returns: returns brand scorecard

        """
        # Columns :
        # Brand Sales 2018	Brand Sales 2019	Brand Sales 2020	Brand Sales 2021
        # Brand Market Share 2018	Brand Market Share 2019	Brand Market Share 2020	Brand Market Share 2021
        # Average Price (moyenne 2018-2021)
        # Brand Awareness
        # Brand A&P 2018	Brand A&P 2019	Brand A&P 2020	Brand A&P 2021
        # Volume sold on Promo (avg 2018-21)
        date_min = (
            json_sell_out_params.get(country).get("brand scorecard").get("date_min")
        )
        print(df.columns)
        df_sales = self.compute_sales(df_bel, date_min=date_min)
        df_sales = self.compute_share(df, df_sales, date_min=date_min)
        df_price = self.compute_price(df_bel, date_min=date_min)
        df_sales["Average Price (moyenne 2018-2021)"] = df_price
        df_sales["Brand Awareness"] = self.compute_awareness(
            json_sell_out_params=json_sell_out_params, country=country
        )
        df_AP = self.compute_AP(df_bel, date_min=date_min)
        df_sales[df_AP.columns] = df_AP
        df_volume_promo = self.compute_volume_promo(
            df, json_sell_out_params.get(country).get("bel_brands"), date_min=date_min
        )
        df_sales["Volume sold on Promo (avg 2018-21)"] = df_volume_promo
        return df_sales

    def compute_awareness(self, json_sell_out_params, country):
        """Brand Scorecard

        :param json_sell_out_params:
        :param country:
        :returns: df_awareness

        """
        aw = json_sell_out_params.get(country).get("brand scorecard").get("awareness")
        return pd.DataFrame.from_dict(aw, orient="index", columns=["Brand Awareness"])

    def compute_sales(self, df, date_min: str):
        """Brand Scorecard

        :param df:
        :param date_min:
        :returns: df_sales

        """
        df_temp = df.copy()
        df_temp.Date = pd.to_datetime(df_temp.Date)
        df_grp = (
            df_temp.groupby(["Brand", pd.Grouper(key="Date", freq="Y")])[
                "Sales in volume"
            ]
            .agg("sum")
            .reset_index()
        )
        df_grp = df_grp[df_grp.Date > date_min]
        table_sales = pd.pivot_table(
            df_grp, values="Sales in volume", index="Brand", columns="Date"
        )

        table_sales.index.name = None
        table_sales.columns.name = "Brand"
        table_sales = table_sales.rename(
            columns={x: f"Brand Sales {x.year}" for x in table_sales.columns}
        )
        table_sales = table_sales.div(1000)
        return table_sales

    def compute_share(self, df, df_sales, date_min: str):
        """Brand Scorecard

        :param df:
        :param df_sales:
        :param date_min:
        :returns: df_share

        """
        df_temp = df.copy()
        df_temp.Date = pd.to_datetime(df_temp.Date)
        df_grp = (
            df_temp.groupby(pd.Grouper(key="Date", freq="Y"))["Sales in volume"]
            .agg("sum")
            .reset_index()
        )
        df_grp = df_grp[df_grp.Date > date_min]
        sales_table = pd.pivot_table(df_grp, columns="Date", values="Sales in volume")
        sales_table = sales_table.div(1000)
        df_sales[[f"Brand Market Share {x.year}" for x in sales_table.columns]] = (
            df_sales[[f"Brand Sales {x.year}" for x in sales_table.columns]]
            / sales_table.iloc[0].values
            * 100
        )
        return df_sales

    def compute_price(self, df_bel, date_min: str):
        """Brand Scorecard

        :param df_bel:
        :param date_min:
        :returns: df_price

        """
        df_temp = df_bel.copy()
        df_temp.Date = pd.to_datetime(df_temp.Date)
        df_grp = (
            df_temp.groupby(["Brand", pd.Grouper(key="Date", freq="Y")])[
                "Price per volume"
            ]
            .agg("mean")
            .reset_index()
        )
        df_grp = df_grp[df_grp.Date > date_min]
        return df_grp.groupby("Brand")["Price per volume"].agg("mean")

    def compute_AP(self, df_bel, date_min: str):
        """Brand Scorecard

        :param df_bel:
        :param date_min:
        :returns:

        """
        df_temp = df_bel.copy()
        df_temp.Date = pd.to_datetime(df_temp.Date)
        df_grp = (
            df_temp.groupby(["Brand", pd.Grouper(key="Date", freq="Y")])["A&P"]
            .agg("sum")
            .reset_index()
        )
        df_grp = df_grp[df_grp.Date > date_min]
        AP_table = pd.pivot_table(df_grp, values="A&P", columns="Date", index="Brand")

        AP_table.index.name = None
        AP_table.columns.name = "Brand"
        AP_table = AP_table.rename(
            columns={x: f"Brand A&P {x.year}" for x in AP_table.columns}
        )
        return AP_table

    def compute_volume_promo(self, df, bel_brands, date_min: str):
        """TODO describe function

        :param df:
        :param bel_brands:
        :param date_min:
        :returns:

        """

        def promo_share(x):
            """TODO describe function

            :param x:
            :returns:

            """
            return x["Sales volume with promo"].mean() / x["Sales in volume"].mean()
        
        df_temp = df.copy()
        df_temp.Date = pd.to_datetime(df_temp.Date)
        df_grp = (
            df_temp[df_temp.Brand.isin(bel_brands)]
            .groupby(["Brand", pd.Grouper(key="Date", freq="Y")])[
                ["Sales volume with promo", "Sales in volume"]
            ]
            .agg("sum", "sum")
            .reset_index()
        )
        df_grp = df_grp[df_grp.Date > date_min]
        return df_grp.groupby("Brand").apply(promo_share)

    def cagr(self, group):
        """TODO describe function

        :param group:
        :returns:

        """
        s = group.reset_index().sort_values(by="Date")
        return s[0].values[-1] - s[0].values[0]

    def compute_cagr(self, df, by: str, year_min: int):
        """TODO describe function

        :param df:
        :param by:
        :param year_min:
        :returns:

        """
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df.Date.dt.year >= year_min]
        df = (
            df.set_index("Date")
            .groupby([by, pd.Grouper(freq="Y")])
            .agg({"Sales in volume": "sum"})
        )
        df_cagr = pd.DataFrame(df.stack().groupby(by).agg(self.cagr))
        return df_cagr.reset_index().rename(columns={0: "CAGR"})

    def compute_awareness(self, json_sell_out_params, country):
        """TODO describe function

        :param json_sell_out_params:
        :param country:
        :returns:

        """
        aw = json_sell_out_params.get(country).get("brand scorecard").get("awareness")
        return pd.DataFrame.from_dict(aw, orient="index", columns=["Awareness"])

    def compute_Market_Matrix(self, df, year_min: int, frag_n_brands: int):
        """TODO describe function

        :param df:
        :param year_min:
        :param frag_n_brands:
        :returns:

        """

        def compute_size(df):
            """TODO describe function

            :param df:
            :returns:

            """
            return (
                df.groupby("Category", as_index=False)["Sales in volume"]
                .sum()
                .rename(columns={"Sales in volume": "Size"})
            )

        def fragmentation_index(series, n):
            """TODO describe function

            :param series:
            :param n:
            :returns:

            """
            size = series.sum()
            return (
                series.reset_index()
                .sort_values("Sales in volume", ascending=False)[:n]["Sales in volume"]
                .apply(lambda x: x / size)
                .agg("prod")
            )

        def compute_fragmentation_index(df, frag_n_brands: int):
            """TODO describe function

            :param df:
            :param frag_n_brands:
            :returns:

            """
            df_FI = (
                df.groupby(["Category", "Brand"])["Sales in volume"]
                .agg(sum)
                .groupby("Category")
                .agg(fragmentation_index, frag_n_brands)
            )
            return (
                pd.DataFrame(df_FI)
                .reset_index()
                .rename(columns={"Sales in volume": "FI"})
            )

        def compute_promotion_intensity(df):
            """TODO describe function

            :param df:
            :returns:

            """
            size = df.groupby("Category")["Sales in volume"].sum()
            promo = df.groupby("Category")["Sales volume with promo"].sum()
            df_PI = promo / size
            return pd.DataFrame(df_PI).reset_index().rename(columns={0: "PI"})

        df_size = compute_size(df)
        df_cagr = self.compute_cagr(df, by="Category", year_min=year_min)
        df_FI = compute_fragmentation_index(df, frag_n_brands=frag_n_brands)
        df_PI = compute_promotion_intensity(df)

        df_category = pd.merge(df_size, df_cagr, on="Category")
        df_category = pd.merge(df_category, df_FI, on="Category")
        df_category = pd.merge(df_category, df_PI, on="Category")
        return df_category.set_index("Category")

    def compute_Brand_Matrix(
        self,
        df_bel,
        json_sell_out_params,
        country: str,
        year_min: int,
        inno_year_min: int,
    ):
        """TODO describe function

        :param df_bel:
        :param json_sell_out_params:
        :param country:
        :param year_min:
        :param inno_year_min:
        :returns:

        """

        def compute_inno(df, how: str, inno_year_min: int):
            """TODO describe function

            :param df:
            :param how:
            :param inno_year_min:
            :returns:

            """
            return (
                df[df.Date.dt.year >= inno_year_min]
                .groupby("Brand")["Rate of Innovation"]
                .agg(how)
            )

        def handle_brand(df, brand):
            """TODO describe function

            :param df:
            :param brand:
            :returns:

            """
            # Fill brand with Os
            date_min = df.Date.min()
            date_max = df.Date.max()
            df_brand = df[df.Brand == brand]
            daterange = pd.DataFrame(
                pd.date_range(start=date_min, end=date_max, freq="W")
            ).rename(columns={0: "Date"})
            df_sub = pd.merge(daterange, df_brand, on="Date", how="left")
            df_sub["Brand"] = brand
            df_sub = df_sub.fillna(0.0)

            df = df[df.Brand != brand]
            df = pd.concat([df, df_sub]).reset_index(drop=True)
            return df

        date_count = df_bel.Date.nunique()
        for brand, group in df_bel.groupby("Brand"):
            if group.Date.nunique() < date_count:
                df_bel = handle_brand(df_bel, brand)

        df_cagr = self.compute_cagr(df_bel, by="Brand", year_min=year_min)
        df_inno = compute_inno(df_bel, how="mean", inno_year_min=inno_year_min)

        df_brand = pd.merge(df_cagr, df_inno, on="Brand")
        df_aw = (
            self.compute_awareness(
                json_sell_out_params=json_sell_out_params, country=country
            )
            .reset_index()
            .rename(columns={"index": "Brand"})
        )
        # df_aw = df_aw.rename(columns={'Brand Awareness':'Awareness'})
        df_brand = pd.merge(df_brand, df_aw, on="Brand")
        return df_brand.set_index("Brand")

    def compute_Market_Brand(self, df, df_bel):
        """TODO describe function

        :param df:
        :param df_bel:
        :returns:

        """

        def compute_corr_table(df, df_bel, method: str):
            """TODO describe function

            :param df:
            :param df_bel:
            :param method:
            :returns:

            """
            brand_table = pd.pivot_table(
                data=df_bel, values="Sales in volume", index="Date", columns="Brand"
            ).reset_index(drop=True)
            category_table = pd.pivot_table(
                data=df, values="Sales in volume", index="Date", columns="Category"
            ).reset_index(drop=True)
            brand_dict = dict()
            for brand in brand_table.columns:
                cat_dict = dict()
                for cat in category_table.columns:
                    cat_dict[cat] = brand_table[brand].corr(
                        category_table[cat], method=method
                    )
                brand_dict[brand] = cat_dict
            return pd.DataFrame(brand_dict)

        def compute_price(df, df_bel):
            """Price = 1 - abs(prix moyen de la marque - prix moyen du marché)/prix moyen du marché

            :param df:
            :param df_bel:
            :returns:

            """
            #
            mean_price_brands = (
                df_bel.groupby("Brand", as_index=False)["Price per volume"]
                .agg("mean")
                .fillna(0.0)
            )
            mean_price_categories = (
                df.groupby("Category", as_index=False)["Price per volume"]
                .agg("mean")
                .fillna(0.0)
            )
            brand_dict = dict()
            for brand in mean_price_brands.Brand.unique():
                cat_dict = dict()
                for cat in mean_price_categories.Category.unique():
                    cat_dict[cat] = (
                        1
                        / abs(
                            mean_price_brands[mean_price_brands.Brand == brand][
                                "Price per volume"
                            ].values
                            - mean_price_categories[
                                mean_price_categories.Category == cat
                            ]["Price per volume"].values
                        )[0]
                    )
                brand_dict[brand] = cat_dict
            return pd.DataFrame(brand_dict)

        def compute_expertise(df, df_bel):
            """TODO describe function

            :param df:
            :param df_bel:
            :returns:

            """
            # Expertise = Questionnaire Bel (Match between Brand on Market) = un truc random
            brands = [
                "BABYBEL",
                "BOURSIN",
                "KAUKAUNA",
                "MERKTS",
                "NURISHH",
                "PRICES",
                "THE LAUGHING COW",
            ]  # df_bel.Brand.unique()
            # categories = df.Category.unique()
            categories = [
                "CLASSIC SPREADS",
                "CREAM CHEESE",
                "ENTERTAINING TRAYS",
                "EVERYDAY BLOCKS",
                "EVERYDAY SHREDDED & GRATED",
                "GOURMET BLOCK / WEDGE / ROUND",
                "GOURMET CRUMBLED",
                "GOURMET FRESH ITALIAN",
                "GOURMET SHREDDED / GRATED",
                "GOURMET SPREADS",
                "PIMENTO",
                "PLANT BASED",
                "RICOTTA AND FARMERS",
                "SLICES",
                "SNACK",
                "SNACKING COMBOS",
            ]
            # return pd.DataFrame(np.random.randint(0, 100, size=(len(categories), len(brands))), columns=brands, index=categories)
            return pd.DataFrame(
                np.zeros((len(categories), len(brands))),
                columns=brands,
                index=categories,
            )

        df_corr = compute_corr_table(df, df_bel, method="pearson")
        df_price = compute_price(df, df_bel)
        df_expertise = compute_expertise(df, df_bel)

        return df_corr, df_price, df_expertise

    def scale(self, df_category, df_brand, df_corr, df_price, df_expertise):
        """TODO describe function

        :param df_category:
        :param df_brand:
        :param df_corr:
        :param df_price:
        :param df_expertise:
        :returns:

        """

        def scale_category(df_category):
            """TODO describe function

            :param df_category:
            :returns:

            """
            Size_scaler = MinMaxScaler(feature_range=(0, 100))
            CAGR_scaler = MinMaxScaler(feature_range=(0, 100))
            FI_scaler = MinMaxScaler(feature_range=(0, 100))
            PI_scaler = MinMaxScaler(feature_range=(0, 100))

            df_category["Size"] = Size_scaler.fit_transform(df_category[["Size"]])
            df_category["CAGR"] = CAGR_scaler.fit_transform(df_category[["CAGR"]])
            df_category["FI"] = FI_scaler.fit_transform(df_category[["FI"]])
            df_category["PI"] = PI_scaler.fit_transform(df_category[["PI"]])

            df_category = df_category.fillna(0)

            df_category["Total"] = df_category[["Size", "CAGR", "FI", "PI"]].sum(axis=1)
            return df_category

        def scale_brand(df_brand):
            """TODO describe function

            :param df_brand:
            :returns:

            """
            CAGR_scaler = MinMaxScaler(feature_range=(0, 100))
            Inno_scaler = MinMaxScaler(feature_range=(0, 100))
            Awereness_scaler = MinMaxScaler(feature_range=(0, 100))

            df_brand["CAGR"] = CAGR_scaler.fit_transform(df_brand[["CAGR"]])
            df_brand["Rate of Innovation"] = Inno_scaler.fit_transform(
                df_brand[["Rate of Innovation"]]
            )
            df_brand["Awareness"] = Awereness_scaler.fit_transform(
                df_brand[["Awareness"]]
            )

            df_brand = df_brand.fillna(0)
            df_brand["Total"] = df_brand[
                ["CAGR", "Rate of Innovation", "Awareness"]
            ].sum(axis=1)
            return df_brand

        def scale_market_brand(df_corr, df_price, df_expertise):
            """TODO describe function

            :param df_corr:
            :param df_price:
            :param df_expertise:
            :returns:

            """
            df_corr_scaled = pd.DataFrame(
                minmax_scale(df_corr.T).T * 100,
                index=df_corr.index.values,
                columns=df_corr.columns.values,
            )
            df_price_scaled = pd.DataFrame(
                minmax_scale(df_price.T).T * 100,
                index=df_price.index.values,
                columns=df_price.columns.values,
            )
            df_expertise_scaled = pd.DataFrame(
                minmax_scale(df_expertise.T).T * 100,
                index=df_expertise.index.values,
                columns=df_expertise.columns.values,
            )
            return df_corr_scaled + df_price_scaled + df_expertise_scaled

        def compute_ctw(
            df_category_scaled,
            df_brand_scaled,
            df_market_brand_scaled,
            coefs={"category": 16, "brand": 16, "fit": 6},
        ):
            """TODO describe function

            :param df_category_scaled:
            :param df_brand_scaled:
            :param df_market_brand_scaled:
            :param coefs:
            :param "brand":
            :param "fit":
            :returns:

            """
            indexes = df_category_scaled.index.values
            nrows = df_category_scaled.shape[0]
            columns = df_brand_scaled.index.values
            ncolumns = df_brand_scaled.shape[0]

            df_category_scaled = df_category_scaled[["Total"]]
            for i in range(ncolumns - 1):
                df_category_scaled.loc[:, i] = df_category_scaled["Total"]
            df_category_scaled.columns = columns

            df_brand_scaled = df_brand_scaled.T.iloc[-1:, :]
            df_brand_scaled.columns.name = ""
            df_brand_scaled = pd.DataFrame().append(
                [df_brand_scaled.reset_index().drop("index", axis=1)] * nrows,
                ignore_index=True,
            )
            df_brand_scaled.index = indexes

            return (
                (df_category_scaled / coefs["category"])
                + (df_brand_scaled / coefs["brand"])
                + (df_market_brand_scaled / coefs["fit"])
            )

        df_brand_scaled = scale_brand(df_brand)
        df_category_scaled = scale_category(df_category)
        df_market_brand_scaled = scale_market_brand(df_corr, df_price, df_expertise)

        capacity_to_win = compute_ctw(
            df_category_scaled=df_category_scaled,
            df_brand_scaled=df_brand_scaled,
            df_market_brand_scaled=df_market_brand_scaled,
        )

        return (
            df_brand_scaled,
            df_category_scaled,
            df_market_brand_scaled,
            capacity_to_win,
        )

    def compute_Capacity_to_Win(self, df, df_bel, json_sell_out_params, country: str):
        """TODO describe function

        :param df:
        :param df_bel:
        :param json_sell_out_params:
        :param country:
        :returns:

        """
        df = df.copy()
        df_bel = df_bel.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df_bel["Date"] = pd.to_datetime(df_bel["Date"])

        YEAR_MIN = (
            json_sell_out_params.get(country).get("Capacity to Win").get("year_min")
        )
        FRAG_N_BRANDS = (
            json_sell_out_params.get(country)
            .get("Capacity to Win")
            .get("fragmentation n_brands")
        )
        market_matrix = self.compute_Market_Matrix(
            df, year_min=YEAR_MIN, frag_n_brands=FRAG_N_BRANDS
        )
        # display(market_matrix)

        INNO_YEAR_MIN = (
            json_sell_out_params.get(country)
            .get("Capacity to Win")
            .get("inno_year_min")
        )
        brand_matrix = self.compute_Brand_Matrix(
            df_bel,
            json_sell_out_params=json_sell_out_params,
            country=country,
            year_min=YEAR_MIN,
            inno_year_min=INNO_YEAR_MIN,
        )
        # display(brand_matrix)

        df_corr, df_price, df_expertise = self.compute_Market_Brand(df, df_bel)
        # display(df_corr)
        # display(df_price)
        # display(df_expertise)

        (
            df_brand_scaled,
            df_category_scaled,
            df_market_brand_scaled,
            capacity_to_win,
        ) = self.scale(
            df_category=market_matrix,
            df_brand=brand_matrix,
            df_corr=df_corr,
            df_price=df_price,
            df_expertise=df_expertise,
        )
        # display(capacity_to_win)

        return (
            df_brand_scaled,
            df_category_scaled,
            df_market_brand_scaled,
            capacity_to_win,
        )

    def compute_market_passeport(self, df_full):
        """TODO describe function

        :param df_full:
        :returns:

        """
        brands_to_remove = [
            "ALL BRANDS"
        ]  # , "ZZ OTHERS", "SS.MARQUE", "UNBRANDED", "PL"

        indicators = [
            "Category",
            "Size",
            "Count of Competitors",
            "L1 Brand",
            "L1 Sales",
            "L1 Share",
            "L2 Brand",
            "L2 Sales",
            "L2 Share",
            "L3 Brand",
            "L3 Sales",
            "L3 Share",
            "Growth",
            "Count of brands",
            "Promotion intensity",
            "Bel brand sales",
            "Bel brands market share",
            "Average Price",
        ]

        df_full = df_full[~df_full["Brand"].isin(brands_to_remove)]

        y_dfs = []
        keys = []
        for year, interval in self.dict_dates.items():
            if year != "HISTORY":
                df_concat = pd.DataFrame(columns=indicators)
                df = self.filter_data(
                    df_full,
                    category="all",
                    brand="all",
                    date_min=interval["Min"],
                    date_max=interval["Max"],
                )
                for category, group in df.groupby("Category"):
                    dict_cat = dict()
                    dict_cat["Category"] = category
                    dict_cat["Size"] = group["Sales in volume"].sum() / 1000
                    dict_cat["Count of Competitors"] = group[
                        ~group["Brand"].isin(self.bel_brands)
                    ]["Brand"].nunique()

                    leaders = (
                        group.groupby("Brand")["Sales in volume"]
                        .sum()
                        .sort_values(ascending=False)[:3]
                    )
                    list_of_leaders = [
                        {"BRAND": i, "SALES": j, "SHARE": 0} for i, j in leaders.items()
                    ]
                    for leader in list_of_leaders:
                        leader["SHARE"] = (
                            leader["SALES"] / group["Sales in volume"].sum() * 100
                        )
                    # list_of_leaders = [str(l) for l in list_of_leaders]
                    dict_cat["L1 Brand"] = list_of_leaders[0]["BRAND"]
                    dict_cat["L2 Brand"] = list_of_leaders[1]["BRAND"]
                    dict_cat["L3 Brand"] = list_of_leaders[2]["BRAND"]

                    dict_cat["L1 Sales"] = list_of_leaders[0]["SALES"] / 1000
                    dict_cat["L2 Sales"] = list_of_leaders[1]["SALES"] / 1000
                    dict_cat["L3 Sales"] = list_of_leaders[2]["SALES"] / 1000

                    dict_cat["L1 Share"] = list_of_leaders[0]["SHARE"]
                    dict_cat["L2 Share"] = list_of_leaders[1]["SHARE"]
                    dict_cat["L3 Share"] = list_of_leaders[2]["SHARE"]

                    # dict_cat['Leaders'] = ''.join(list_of_leaders)
                    # dict_cat['Leaders'] = np.array2string(group.groupby('Brand')['Sales in value'].sum().sort_values(ascending=False)[:3].index.array)
                    dict_cat["Growth"] = self.compute_growth(df_full, year, category)
                    dict_cat["Growth"] = (
                        dict_cat["Growth"]
                        if isinstance(dict_cat["Growth"], str)
                        else dict_cat["Growth"] / 1000
                    )
                    dict_cat["Count of brands"] = group["Brand"].nunique()
                    dict_cat["Promotion intensity"] = (
                        group["Sales volume with promo"].sum()
                        / group["Sales in volume"].sum()
                    )
                    dict_cat["Bel brand sales"] = (
                        group[group["Brand"].isin(self.bel_brands)][
                            "Sales in volume"
                        ].sum()
                        / 1000
                    )
                    dict_cat["Bel brands market share"] = (
                        group[group["Brand"].isin(self.bel_brands)][
                            "Sales in volume"
                        ].sum()
                        / group["Sales in volume"].sum()
                        * 100
                    )
                    dict_cat["Average Price"] = group["Price per volume"].agg("mean")

                    df_cat = pd.DataFrame(data=dict_cat, index=[0])
                    df_concat = pd.concat([df_concat, df_cat])
                y_dfs.append(df_concat.set_index("Category").T)
                keys.append(year)
        out = pd.concat(y_dfs, axis=0, keys=keys, names=["Years", "Drivers"])
        return out
