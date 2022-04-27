import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, minmax_scale

class Capacity_To_Win():
    def compute_volume_promo(self, df, bel_brands, date_min: str):
        def promo_share(x):
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
        s = group.reset_index().sort_values(by="Date")
        return s[0].values[-1] - s[0].values[0]

    def compute_cagr(self, df, by: str, year_min: int):
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
        aw = json_sell_out_params.get(country).get("brand scorecard").get("awareness")
        return pd.DataFrame.from_dict(aw, orient="index", columns=["Awareness"])

    def compute_Market_Matrix(self, df, year_min: int, frag_n_brands: int):
        def compute_size(df):
            return (
                df.groupby("Category", as_index=False)["Sales in volume"]
                .sum()
                .rename(columns={"Sales in volume": "Size"})
            )

        def fragmentation_index(series, n):
            size = series.sum()
            return (
                series.reset_index()
                .sort_values("Sales in volume", ascending=False)[:n]["Sales in volume"]
                .apply(lambda x: x / size)
                .agg("prod")
            )

        def compute_fragmentation_index(df, frag_n_brands: int):
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
            size = df.groupby("Category")["Sales in volume"].sum()
            promo = df.groupby("Category")["Sales volume with promo"].sum()
            df_PI = promo / size
            return pd.DataFrame(df_PI).reset_index().rename(columns={0: "PI"})

        df_size = compute_size(df)
        df_cagr = self.compute_cagr(df, by="Category", year_min=year_min)
        df_FI = compute_fragmentation_index(df, frag_n_brands=frag_n_brands)
        df_category = pd.merge(df_size, df_cagr, on="Category")
        df_category = pd.merge(df_category, df_FI, on="Category")
        if "Sales volume with promo" in df.columns:
            df_PI = compute_promotion_intensity(df)
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
        def compute_inno(df, how: str, inno_year_min: int):
            return (
                df[df.Date.dt.year >= inno_year_min]
                .groupby("Brand")["Rate of Innovation"]
                .agg(how)
            )

        def handle_brand(df, brand):
            # Fill brand with Os
            print(f"<handle_brand> {brand}")
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
        def compute_corr_table(df, df_bel, method: str):
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
            """
            #
            mean_price_brands = (
                df_bel.groupby("Brand", as_index=False)["Price per volume"]
                .agg("mean")
                .fillna(0.0)
            )
            mean_price_categories = (
                df
                [pd.to_numeric(df['Price per volume'], errors='coerce').notnull()]
                .fillna(0.0)
                .groupby("Category")["Price per volume"]
                .agg("mean")
                .reset_index()
            )

            brand_dict = dict()
            for brand in mean_price_brands.Brand.unique():
                cat_dict = dict()
                for cat in mean_price_categories.Category.unique():
                    cat_dict[cat] = (
                        1
                        / abs(
                            mean_price_brands[mean_price_brands.Brand == brand]["Price per volume"].values
                            - 
                            mean_price_categories[mean_price_categories.Category == cat]["Price per volume"].values
                        )[0]
                    )
                brand_dict[brand] = cat_dict
            return pd.DataFrame(brand_dict)

        def compute_expertise(df, df_bel):
            # Expertise = Questionnaire Bel (Match between Brand on Market) = un truc random
            df_expertise = pd.read_excel("data/CAN/questionnaire.xlsx").set_index("Category")
            #brands = df_bel.Brand.unique()
            #categories = df.Category.unique()
            #return pd.DataFrame(np.random.randint(0, 100, size=(len(categories), len(brands))), columns=brands, index=categories)
            #return pd.DataFrame(
            #   np.zeros((len(categories), len(brands))),
            #   columns=brands,
            #   index=categories,
            #)
            return df_expertise

        df_corr = compute_corr_table(df, df_bel, method="pearson")
        df_price = compute_price(df, df_bel)
        df_expertise = compute_expertise(df, df_bel)

        return df_corr, df_price, df_expertise

    def scale(self, df_category, df_brand, df_corr, df_price, df_expertise):
        def scale_category(df_category):
            Size_scaler = MinMaxScaler(feature_range=(0, 100))
            CAGR_scaler = MinMaxScaler(feature_range=(0, 100))
            FI_scaler = MinMaxScaler(feature_range=(0, 100))
            PI_scaler = MinMaxScaler(feature_range=(0, 100))

            df_category["Size"] = Size_scaler.fit_transform(df_category[["Size"]])
            df_category["CAGR"] = CAGR_scaler.fit_transform(df_category[["CAGR"]])
            df_category["FI"] = FI_scaler.fit_transform(df_category[["FI"]])
            ### very ad hoc, fix this
            if "PI" in df_category.columns:
                df_category["PI"] = PI_scaler.fit_transform(df_category[["PI"]])

            df_category = df_category.fillna(0)
            
            ### very ad hoc, fix this
            if "PI" in df_category.columns:
                df_category["Total"] = df_category[["Size", "CAGR", "FI", "PI"]].sum(axis=1)
            else :
                df_category["Total"] = df_category[["Size", "CAGR", "FI"]].sum(axis=1)
            return df_category

        def scale_brand(df_brand):
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
            # coefs={"category": 16, "brand": 12, "fit": 6},
        ):
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
                (df_category_scaled)
                + (df_brand_scaled)
                + (df_market_brand_scaled)
            )
            # return (
            #     (df_category_scaled / coefs["category"])
            #     + (df_brand_scaled / coefs["brand"])
            #     + (df_market_brand_scaled / coefs["fit"])
            # )

        df_brand_scaled = scale_brand(df_brand)

        ###FILTER DF
        
        #df_category = df_category[df_category.index.isin(["SPCLT AO SPCLT", "SPCLT CRUMBLED", "SPCLT FETA", "SPCLT FONDUE", "SPCLT FRESH MOZZ", "SPCLT FRESH TUB", "SPCLT GRILLING", "SPCLT PARMESAN TYPE", "SPCLT RACLETTE"])]
        df_category_scaled = scale_category(df_category)

        #df_corr = df_corr[df_corr.index.isin(["SPCLT AO SPCLT", "SPCLT CRUMBLED", "SPCLT FETA", "SPCLT FONDUE", "SPCLT FRESH MOZZ", "SPCLT FRESH TUB", "SPCLT GRILLING", "SPCLT PARMESAN TYPE", "SPCLT RACLETTE"])]
        #df_price = df_price[df_price.index.isin(["SPCLT AO SPCLT", "SPCLT CRUMBLED", "SPCLT FETA", "SPCLT FONDUE", "SPCLT FRESH MOZZ", "SPCLT FRESH TUB", "SPCLT GRILLING", "SPCLT PARMESAN TYPE", "SPCLT RACLETTE"])]
        #df_expertise = df_expertise[df_expertise.index.isin(["SPCLT AO SPCLT", "SPCLT CRUMBLED", "SPCLT FETA", "SPCLT FONDUE", "SPCLT FRESH MOZZ", "SPCLT FRESH TUB", "SPCLT GRILLING", "SPCLT PARMESAN TYPE", "SPCLT RACLETTE"])]
        df_market_brand_scaled = scale_market_brand(df_corr, df_price, df_expertise)
        
        # coefs={"category": 16, "brand": 12, "fit": 6},
        df_brand_scaled = df_brand_scaled.div(16)
        df_category_scaled = df_category_scaled.div(12)
        df_market_brand_scaled = df_market_brand_scaled.div(6)

        capacity_to_win = compute_ctw(
            df_category_scaled=df_category_scaled,
            df_brand_scaled=df_brand_scaled,
            df_market_brand_scaled=df_market_brand_scaled,
        )

        # Etendre
        df_brand_scaled = df_brand_scaled[["Total"]]
        for category in df_category_scaled.index:
            df_brand_scaled[category] = df_brand_scaled.loc[:, "Total"]
        #df_brand_scaled.drop("Total")

        # Transpose + Etendre
        df_category_scaled = df_category_scaled[["Total"]]
        for brand in df_brand_scaled.index:
            df_category_scaled[brand] = df_category_scaled.loc[:, "Total"]
        #df_category_scaled.drop("Total")
        df_category_scaled = df_category_scaled.T

        return (
            df_brand_scaled,
            df_category_scaled,
            df_market_brand_scaled.T,
            capacity_to_win.T,
        )

    def compute_Capacity_to_Win(self, df, df_bel, json_sell_out_params, country: str):
        df = df.copy()
        df_bel = df_bel.copy()
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        df_bel["Date"] = pd.to_datetime(df_bel["Date"], format="%Y-%m-%d")

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

        df_corr, df_price, df_expertise = self.compute_Market_Brand(df, df_bel)

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

        return (
            df_brand_scaled,
            df_category_scaled,
            df_market_brand_scaled,
            capacity_to_win,
        )