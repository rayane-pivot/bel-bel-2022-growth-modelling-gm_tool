import json
import math
from tokenize import Number

import numpy as np
import pandas as pd


class Model:
    def __init__(self):
        pass

    def brand_pos(self, df, year_min:int, values:str):
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
            values=values,
            columns="Category",
            index="Brand",
            aggfunc="sum",
        )

        brand_positioning_matrix.columns.name = None
        brand_positioning_matrix.index.name = "Brand"
        return brand_positioning_matrix.div(1000)

    def cat_yearly_sales(self, df, on:str, values:str):
        """Brand Positioning Matrix
        """

        df = df.copy()
        df.Date = pd.to_datetime(df.Date)

        df_grp = (
            df.groupby([on, pd.Grouper(key="Date", freq="Y")])[values]
            .agg("sum")
            .reset_index()
        )
        df_grp.Date = df_grp.Date.dt.year
        table = pd.pivot_table(df_grp, columns=on, index="Date", values=values)
        table=table.div(1000)

        ###Price
        df_grp = (
            df
            [pd.to_numeric(df['Price per volume'], errors='coerce').notnull()]
            .groupby([on, pd.Grouper(key="Date", freq="Y")])["Price per volume"]
            .agg(["mean", "median"])
            .reset_index()
        )
        df_grp.Date = df_grp.Date.dt.year
        mean_price_table = pd.pivot_table(df_grp, columns=on, index="Date", values="mean")
        mean_price_table = mean_price_table.rename(index={date : f"{date} mean Price" for date in mean_price_table.index.values})
        median_price_table = pd.pivot_table(df_grp, columns=on, index="Date", values="median")
        median_price_table = median_price_table.rename(index={date : f"{date} median Price" for date in median_price_table.index.values})
        cat_yearly_sales = pd.concat([table, mean_price_table, median_price_table])

        return cat_yearly_sales

    def growth(self, df, on: str, year1: int, year2: int, values:str):
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
                x[x.Date.dt.year == year1][values].values[0]
                if len(x[x.Date.dt.year == year1][values].values) > 0
                else None
            )
            y2 = (
                x[x.Date.dt.year == year2][values].values[0]
                if len(x[x.Date.dt.year == year2][values].values) > 0
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
                x[x.Date.dt.year == year1][values].values[0]
                if len(x[x.Date.dt.year == year1][values].values) > 0
                else 0.0
            )
            y2 = (
                x[x.Date.dt.year == year2][values].values[0]
                if len(x[x.Date.dt.year == year2][values].values) > 0
                else 0.0
            )
            return y2 - y1

        df = df.copy()
        df.Date = pd.to_datetime(df.Date)

        df_grp = (
            df.groupby([on, pd.Grouper(key="Date", freq="Y")])[values]
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
        values = "Sales in volume"
        brand_positioning_matrix = self.brand_pos(df, year_min=year_min, values=values)
        growth_brand = self.growth(df, on="Brand", year1=year1, year2=year2, values=values)
        growth_category = self.growth(df, on="Category", year1=year1, year2=year2, values=values)

        cat_yearly_sales = self.cat_yearly_sales(df, on="Category", values=values)

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
        
        def median_price(x):
            """local function for groupby apply

            :param x:
            :returns: average of Price per volume for year

            """
            if "Price per volume" in x.columns:
                return x[x.Date.dt.year == year]["Price per volume"].agg("median")
        
        def promo(x):
            """local function for groupby apply

            :param x:
            :returns: sum of Promo Cost for year

            """
            if "Promo Cost" in x.columns:
                return x[x.Date.dt.year == year]["Promo Cost"].agg("mean")

        def size(x):
            """local function for groupby apply

            :param x:
            :returns: sum of Sales in volume for year

            """
            if "Sales in volume" in x.columns:
                return x[x.Date.dt.year == year]["Sales in volume"].agg("sum")/1000

        def avg_distrib(x):
            if "Distribution" in x.columns:
                return x[x.Date.dt.year == year]["Distribution"].agg("mean")
        
        def median_distrib(x):
            if "Distribution" in x.columns:
                return x[x.Date.dt.year == year]["Distribution"].agg("median")
        
        df_bel = df_bel.copy()
        df_bel.Date = pd.to_datetime(df_bel.Date)
        df_grp = (
            df_bel.groupby(["Brand", pd.Grouper(key="Date", freq="Y")])
            .apply(
                lambda r: pd.Series(
                    {
                        "A&P 2021": ap(r), 
                        "Avg Price 2021": price(r), 
                        "Median Price 2021": median_price(r), 
                        "Promo 2021": promo(r), 
                        "Size 2021": size(r), 
                        "Avg Distribution 2021": avg_distrib(r), 
                        "Median Distribution 2021": median_distrib(r)
                    }
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
        # distrib = (
        #     df[df.Brand.isin(bel_brands)]
        #     .groupby(["Brand", pd.Grouper(key="Date", freq="Y")])["Distribution"]
        #     .mean()
        #     .reset_index()
        # )
        # distrib = distrib.loc[distrib[distrib.Date.dt.year == year].index, :]
        # print(distrib)
        # distrib = distrib.rename(columns={'Distribution':'Avg Distribution'})

        # new_distrib = (
        #     df[df.Brand.isin(bel_brands)]
        #     .groupby(["Brand", pd.Grouper(key="Date", freq="Y")])["Distribution"]
        #     .median()
        #     .reset_index()
        # )
        # new_distrib = new_distrib.loc[new_distrib[new_distrib.Date.dt.year == year].index, :]
        # new_distrib = new_distrib.rename(columns={'Distribution':'Median Distribution'})
        if "Sales volume with promo" in df.columns:
            new_promo = (
                df[df.Brand.isin(bel_brands)]
                .groupby(["Brand", pd.Grouper(key="Date", freq="Y")])["Sales in volume", "Sales volume with promo"]
                .sum()
                .reset_index()
            )
            new_promo = new_promo.loc[new_promo[new_promo.Date.dt.year == year].index, :]
            new_promo["New Promo"] = new_promo["Sales volume with promo"] / new_promo["Sales in volume"]
            df_attack_init_state = pd.merge(df_temp, new_promo, on=["Brand", "Date"])
        else :
            df_attack_init_state = df_temp
        # df_attack_init_state = pd.merge(df_temp, distrib, on=["Brand", "Date"])
        # df_attack_init_state = pd.merge(df_attack_init_state, new_distrib, on=["Brand", "Date"])
        
        
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
