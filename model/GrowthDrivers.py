import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBRegressor


class GrowthDrivers:
    def __init__(self, data_manager, bel_markets):
        self.dm = data_manager
        self.bel_markets = bel_markets

    def _growth_drivers(
        self,
        df_bel,
        df_markets,
        df_competition_brands,
        brands_name,
        brands_markets,
        controllable_features=False,
        add_market=False,
        add_competition=False,
    ):
        dict_res = {"xgb_feature_importance": {}, "permutation_importance": {}}
        competition_feats_cat = ["Competition price", "Competition sales"]
        compet_feats, ordered_drivers = [], []
        for brand in tqdm(brands_name, ascii=True, desc="Brands"):
            # Competition features
            feats_cat = [
                feat + "_" + cat
                for cat in brands_markets[brand]
                for feat in competition_feats_cat
            ]
            df_tmp = df_bel[df_bel.Brand == brand]
            if not controllable_features:
                if add_competition:
                    df_tmp = pd.merge(
                        df_tmp, df_competition_brands[["Date"] + feats_cat], on="Date"
                    )

                if add_market:
                    df_tmp = pd.merge(
                        df_tmp, df_markets[["Date"] + brands_markets[brand]], on="Date"
                    )

            if len(df_tmp) == 0:
                continue

            # Add competition features if df_tmp is not empty (eg: NURISHH for 2019, 2020, etc...)
            compet_feats += feats_cat

            for key in dict_res:
                dict_res[key][brand] = {}

            df_tmp_X = df_tmp.drop(["Date", "Brand", "Sales in volume"], axis=1)
            X = df_tmp_X.values
            y = df_tmp["Sales in volume"]

            # Giving 0.05 to train set, cause it's not used in anycase
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.05, random_state=42
            )
            xgb_model = XGBRegressor()
            xgb_model.fit(X_train, y_train)

            # XGB Feature importance
            feature_importances = xgb_model.feature_importances_
            sorted_idx_feat_imp = feature_importances.argsort()

            # Doing permutation importance on Train set
            perm_importance = permutation_importance(
                xgb_model, X_train, y_train, n_repeats=30
            )
            sorted_idx_perm_imp = perm_importance.importances_mean.argsort()
            for idx_fi, idx_pi in zip(sorted_idx_feat_imp, sorted_idx_perm_imp):
                dict_res["xgb_feature_importance"][brand][df_tmp_X.columns[idx_fi]] = (
                    feature_importances[idx_fi] * 100
                )
                dict_res["permutation_importance"][brand][
                    df_tmp_X.columns[idx_pi]
                ] = perm_importance["importances_mean"][idx_pi]

        # Set compet feats, cause many brands are in the same markets.
        # np.delete, in order to remove 'Sales in volume', and [2:], to remove Date and Brand
        if controllable_features:
            ordered_drivers = list(df_bel.columns[2:-1])
        else:
            bel_markets = list(
                set(
                    [
                        elem
                        for brand in dict_res["xgb_feature_importance"]
                        for elem in brands_markets[brand]
                    ]
                )
            )

            if add_market:
                ordered_drivers += bel_markets

            if add_competition:
                ordered_drivers += list(set(compet_feats))

            ordered_drivers += list(df_bel.columns[2:-1])

        for dict_imp in dict_res:
            dict_res[dict_imp] = pd.DataFrame(dict_res[dict_imp]).loc[ordered_drivers]

        return dict_res

    def compute_growth_drivers_past(
        self,
        df,
        df_bel,
        brands_name,
        years=[2018, 2019, 2020, 2021],
        list_controllable_features=[
            "A&P",
            "Price per volume",
            "Promo cost",
            "Rate of Innovation",
            "Distribution",
        ],
        controllable_features=False,
        add_market=False,
        add_competition=False,
    ):
        """ """

        brands_markets = {
            brand: df[df.Brand == brand].Category.unique().tolist()
            for brand in brands_name
        }
        # Get markets dataframe
        df_markets = self.dm.get_df_markets(df)
        # Computing competition price and sales for bel brands
        competition_features = [
            "Date",
            "Brand",
            "Category",
            "Distribution",
            "Price per volume",
            "Sales in volume",
        ]

        df_competition_brands = self.dm.get_df_competition_brands(
            df, competition_features, brands_name, self.bel_markets
        )

        df_competition_brands["Price per volume"] = df_competition_brands[
            "Price per volume"
        ].apply(float)

        df_competition_brands = (
            df_competition_brands.groupby(["Date", "Category"])
            .agg({"Price per volume": np.mean, "Sales in volume": np.sum})
            .unstack()
            .rename(
                columns={
                    "Price per volume": "Competition price",
                    "Sales in volume": "Competition sales",
                }
            )
        )
        df_competition_brands.columns = [
            "_".join(elem) for elem in df_competition_brands.columns
        ]
        df_competition_brands = df_competition_brands.reset_index()
        dict_df_gd = {year: {} for year in years}

        for year in tqdm(years, ascii=True, desc="Years"):
            df_bel_year = df_bel[df_bel.Date.astype("datetime64[ns]").dt.year == year]
            df_compet_year = df_competition_brands[
                df_competition_brands.Date.astype("datetime64[ns]").dt.year == year
            ]
            df_markets_year = df_markets[
                df_markets.Date.astype("datetime64[ns]").dt.year == year
            ]

            dict_df_gd[year] = self._growth_drivers(
                df_bel_year,
                df_markets_year,
                df_compet_year,
                brands_name,
                brands_markets,
                controllable_features=controllable_features,
                add_market=add_market,
                add_competition=add_competition,
            )

        dict_df_gd["all"] = self._growth_drivers(
            df_bel,
            df_markets,
            df_competition_brands,
            brands_name,
            brands_markets,
            controllable_features=controllable_features,
            add_market=add_market,
            add_competition=add_competition,
        )

        dict_df_gd_scaled = {k: {} for k in dict_df_gd}
        for year in dict_df_gd:
            for imp in dict_df_gd[year]:
                # Convert first dict_df_gd to dataframes
                dict_df_gd[year][imp] = pd.DataFrame(dict_df_gd[year][imp])
                # For Permutation importance specifically, replace negative values by 0
                dict_df_gd[year][imp][dict_df_gd[year][imp] < 0] = 0

                # Rescale the controllable_features to 100
                dict_df_gd_scaled[year][imp] = (
                    dict_df_gd[year][imp]
                    .loc[list_controllable_features]
                    .apply(lambda x: (x * 100) / x.sum())
                )

        return dict_df_gd, dict_df_gd_scaled
