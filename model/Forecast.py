import itertools
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import orbit
import pandas as pd
from fbprophet import Prophet
from orbit.models import DLT
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils.adstock import run_adstock_loop
from utils.tools import cagr, mean_absolute_percentage_error, print_df_overview


class Forecast:
    def __init__(self, country, data_manager, df, df_bel, periods, freq, spath):
        self.country = country
        self.dm = data_manager
        self.df = df
        self.df_bel = df_bel
        self.periods = periods
        self.freq = freq
        self.spath = spath
        self.brands_cat, self.bel_brands_cat = self.get_brands_cat()
        self.bel_markets_list = self.get_bel_markets()

    def get_brands_cat(self):
        brands_cat = {
            brand: self.df[self.df.Brand == brand].Category.unique().tolist()
            for brand in self.df.Brand.unique()
        }
        bel_brands_cat = {
            brand: brands_cat[brand] for brand in self.df_bel.Brand.unique()
        }

        return brands_cat, bel_brands_cat

    def get_bel_markets(self):

        bel_markets = list(set(list(itertools.chain(*self.bel_brands_cat.values()))))

        return bel_markets

    def get_competition(self, brands_name):
        """TODO describe function

        :param brands_name:
        :returns:

        """

        competition_features = [
            "Date",
            "Brand",
            "Category",
            "Distribution",
            "Price per volume",
            "Sales in volume",
        ]

        df_compet = self.dm.get_df_competition_brands(
            self.df, competition_features, brands_name, self.bel_markets_list
        )

        df_compet = (
            df_compet.groupby(["Date", "Category"])
            .agg({"Price per volume": np.mean, "Sales in volume": np.sum})
            .unstack()["Sales in volume"]
            .reset_index()
        )
        df_compet.index.name = ""

        return df_compet

    def forecasting_features(
        self, df, features_name, logistic=False, plot=False, splot_suffix=None
    ):
        """Get long term forecasts for features giving the dataframe and periods

        :param df:
        :param features_name:
        :param regressors:
        :param plot:

        """
        # Select features except the target
        list_df_features = []
        mape_errors = {}

        controllable_features = [
            "A&P",
            "Price per volume",
            "Distribution",
            "Promo Cost",
            "Rate of Innovation",
        ]
        for feature in tqdm(features_name, ascii=True, desc="Features"):
            # Preprocessing Inputs of Prophet
            df_frcast = df[["Date", feature]]
            df_frcast.columns = ["ds", "y"]
            df_frcast.ds = pd.to_datetime(df_frcast.ds)

            # Yearly seasonality variable
            ys = False if feature in controllable_features else "auto"
            if logistic:
                df_frcast["cap"] = df[feature].max() * 3
                df_frcast["floor"] = df[feature].mean() / 2

                if (df_frcast["cap"] <= df_frcast["floor"]).any():
                    df_frcast["cap"] = df_frcast.floor + 10

                # Model
                # yearly_seasonality=20
                model = Prophet(
                    growth="logistic",
                    # seasonality_prior_scale=2.5,
                    # holidays_prior_scale=20,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    yearly_seasonality=ys,
                )
            else:
                model = Prophet(
                    growth="linear",
                    daily_seasonality=False,
                    weekly_seasonality=False,
                    yearly_seasonality=ys,
                )

            model.fit(df_frcast)

            # Compute mape error of future in train
            y_train_hat = model.predict()["yhat"]
            mape = mean_absolute_percentage_error(
                df_frcast.y.values.reshape(-1, 1), y_train_hat.values.reshape(-1, 1)
            )
            mape_errors[feature] = mape

            # Periods is in months
            future = model.make_future_dataframe(periods=self.periods, freq=self.freq)

            if logistic:
                future["cap"] = df_frcast["cap"][0]
                future["floor"] = df_frcast["floor"][0]

            fcst = model.predict(future[future.ds > df_frcast.ds.iloc[-1]])
            fcst["yhat"] = fcst.yhat.apply(lambda x: 0 if x < 0 else x)

            # Plot forecast
            if plot or splot_suffix:
                fig = model.plot(fcst)
                ax = fig.gca()
                ax.set_title(feature)

                if plot:
                    plt.show()

                if splot_suffix:
                    # Plot Forecast path
                    print(f"   Saving plots of {feature} ...")
                    pf_path = os.path.join(self.spath, "Plots", splot_suffix)
                    if not os.path.exists(pf_path):
                        os.makedirs(pf_path)

                    plt.savefig(os.path.join(pf_path, feature + ".png"))

            df_frcast.rename(columns={"y": feature}, inplace=True)
            fcst.rename(columns={"yhat": feature}, inplace=True)

            # Adding result to list
            if logistic:
                list_df_features.append(
                    pd.concat([df_frcast.iloc[:, :-2], fcst[["ds", feature]]])
                )
            else:
                list_df_features.append(pd.concat([df_frcast, fcst[["ds", feature]]]))

        # Merging the dataframes of each feature
        df_features = list_df_features[0][["ds"]]
        for dff in list_df_features:
            df_features = pd.merge(df_features, dff, on="ds")

        df_features.rename(columns={"ds": "Date"}, inplace=True)

        df_mape_errors = pd.DataFrame(mape_errors, index=["MAPE"])

        return df_features, df_mape_errors

    def forecasting_one_brand(
        self,
        df,
        brand_name,
        features_name,
        logistic=False,
        plot=False,
        splot_suffix=None,
    ):
        """Forecasting giving brand_name for giving number of periods and

        :param df:
        :param brand_name:
        :param features_name:

        """
        # Get DataFrame for brand_name
        df = (
            df.groupby(["Brand", "Date"])
            .agg("sum")
            .loc[brand_name]
            .reset_index()
            .sort_values("Date")
        )

        # Adhoc for KSA distrib qui clamse
        # Ahhoc Germany HARD Discount, cut from 31/10/21 for ["ADLER", "BABYBEL", "KIRI"]
        # Total channel, cut from 31/10/21 for ["ADLER", "BABYBEL", "BONBEL"]
        if brand_name in ["ADLER", "BABYBEL", "KIRI"]:
            self.periods = 157 + 9
            df = df.iloc[:-9, :]

        # Adhoc Germany - Total Channel
        # if brand_name == "LA VACHE QUI RIT":
        #     self.periods = 157 + 47
        #     df = df.iloc[:-47, :]

        # Predict long-term forecast for features
        df_features, features_mape_errors = self.forecasting_features(
            df,
            features_name,
            logistic=False,
            plot=plot,
            splot_suffix=splot_suffix + "/" + brand_name,
        )

        # Preprocessing dataframe for prophet
        df["Date"] = pd.to_datetime(df.Date)
        df = df[["Date", *features_name, "Sales in volume"]]
        df.rename(columns={"Date": "ds", "Sales in volume": "y"}, inplace=True)

        df["cap"] = df["y"].max() * 3
        df["floor"] = df["y"].min() / 1.5
        if (df["cap"] <= df["floor"]).any():
            df["cap"] = df.floor + 10

        # Instanciating Prophet Model
        # yearly_seasonality=20
        if logistic:
            model = Prophet(
                growth="logistic", daily_seasonality=False, weekly_seasonality=False
            )
            # model = Prophet(
            #     growth="logistic",
            #     seasonality_prior_scale=2.5,
            #     holidays_prior_scale=20,
            #     yearly_seasonality="auto",
            # )
        else:
            model = Prophet(
                growth="linear", daily_seasonality=False, weekly_seasonality=False
            )

        # Adding regressors
        for feature in features_name:
            model.add_regressor(feature)

        # Fitting model
        model.fit(df)

        # Compute mape error of future in train
        y_train_hat = model.predict()["yhat"]
        mape = mean_absolute_percentage_error(
            df.y.values.reshape(-1, 1), y_train_hat.values.reshape(-1, 1)
        )

        future = model.make_future_dataframe(periods=self.periods, freq=self.freq)

        for feature in features_name:
            future[feature] = df_features[feature]

        future["cap"] = df["cap"][0]
        future["floor"] = df["floor"][0]

        # print(future)
        # future = future.fillna(0, axis=0)

        # Predicting sales forecasts
        fcst = model.predict(future[future.ds > df.ds.iloc[-1]])
        fcst["yhat"] = fcst.yhat.apply(lambda x: 0 if x < 0 else x)

        # Plot forecast
        if plot or splot_suffix:
            fig = model.plot(fcst)
            ax = fig.gca()
            ax.set_title(brand_name)

            if plot:
                plt.show()

            if splot_suffix:
                # Plot Forecast path
                print(f"   Saving plots of {brand_name} ...")
                pf_path = os.path.join(self.spath, "Plots", splot_suffix, brand_name)
                if not os.path.exists(pf_path):
                    os.makedirs(pf_path)
                plt.savefig(os.path.join(pf_path, brand_name + ".png"))

        fcst.rename(columns={"yhat": "y"}, inplace=True)
        df_brand_fcst = pd.concat([df[["ds", "y"]], fcst[["ds", "y"]]])
        df_brand_fcst.rename(columns={"ds": "Date", "y": brand_name}, inplace=True)

        return df_brand_fcst, future, mape

    def forecasting_brands_with_regressors(
        self,
        df,
        brands_name,
        features_name,
        logistic=False,
        plot=False,
        splot_suffix=None,
        add_market=True,
        add_competition=True,
    ):
        """Forecasting a list of brands using the given features and periods

        :param df:
        :param brands_name:
        :param features_name:
        :param logistic:
        :param plot:
        :param splot_suffix: save plot suffix

        """
        dict_feats_futures = {}
        list_df_brands = []
        mape_errors = {}

        if add_market:
            df_markets_or_compet = self.dm.get_df_markets(self.df)

        if add_competition:
            # Computing competition price and sales for bel brands
            df_markets_or_compet = self.get_competition(brands_name)

        df = pd.merge(df, df_markets_or_compet, on="Date")

        # Compute forecast for each brand
        for brand in tqdm(
            brands_name,
            ascii=True,
            desc="Brands",
        ):
            brand_fcst, future, mape = self.forecasting_one_brand(
                df,
                brand,
                features_name + self.brands_cat[brand],
                logistic=logistic,
                plot=plot,
                splot_suffix=splot_suffix,
            )
            list_df_brands.append(brand_fcst)
            dict_feats_futures[brand] = future
            mape_errors[brand] = mape

        # Merge results of each brand
        df_brands_fcst = list_df_brands[0].Date
        for df_tmp in list_df_brands:
            df_brands_fcst = pd.merge(df_brands_fcst, df_tmp, on="Date", how="outer")

        df_brands_fcst["Date"] = df_brands_fcst.Date.apply(
            lambda x: x.strftime("%Y-%m-%d")
        )

        df_mape_errors = pd.DataFrame(mape_errors, index=["MAPE"])
        return (df_brands_fcst, dict_feats_futures, df_mape_errors)

    def forecasting_categories(
        self, df, categories_name, logistic=False, plot=False, splot_suffix=None
    ):
        """Forecasting list of markets for the given periods

        :param df:
        :param categories_name:

        """
        df_categories = self.dm.get_df_markets(df)
        df_features, df_features_mape_errors = self.forecasting_features(
            df_categories,
            categories_name,
            logistic=logistic,
            plot=plot,
            splot_suffix=splot_suffix,
        )

        df_features["Date"] = df_features.Date.apply(lambda x: x.strftime("%Y-%m-%d"))

        return df_features, df_features_mape_errors

    def forecasting_brands_no_regressors(
        self, df_bel, logistic=False, plot=False, splot_suffix=None
    ):
        """TODO describe function

        :param df_bel:
        :returns:

        """
        df_bel_brands = (
            df_bel.groupby(["Date", "Brand"])
            .agg("sum")["Sales in volume"]
            .unstack()
            .reset_index()
        )
        df_bel_brands.columns.name = ""
        df_bel_brands = df_bel_brands.fillna(0)
        df_bel_brands_fcst, df_brands_mape_errors = self.forecasting_features(
            df_bel_brands,
            list(df_bel_brands.columns[1:]),
            logistic=logistic,
            plot=plot,
            splot_suffix=splot_suffix,
        )

        df_bel_brands_fcst["Date"] = df_bel_brands_fcst.Date.apply(
            lambda x: x.strftime("%Y-%m-%d")
        )

        return df_bel_brands_fcst, df_brands_mape_errors

    def create_scenario_data_per_year(self, df_bel, features, scenario):
        """
        :scenario: tuple
        """

        df_bel["Date"] = pd.to_datetime(df_bel.Date, format="%Y-%m-%d")
        df_bel = df_bel.sort_values(by="Date")
        last_year = df_bel.Date.iloc[-1].year

        # Start generating one year by weeks, and remove the first cause exists in bel
        periods = 13 if self.freq == "M" else 53
        # periods = 156
        df_bel_future = (
            pd.date_range(start=df_bel.Date.iloc[-1], periods=periods, freq=self.freq)[
                1:
            ]
            .to_frame()
            .reset_index(drop=True)
            .rename(columns={0: "Date"})
        )

        # for k, v in zip(features, scenario):
        #     df_bel_future[k] = (df_bel[k] + (df_bel[k] * v / 100)).values

        for k, v in zip(features, scenario):
            df_bel_last_year = df_bel[df_bel.Date.dt.year == last_year]

            if k in ["Price per volume", "Distribution"]:
                init_state = df_bel_last_year[k].iloc[-1]
                df_bel_future[k] = np.linspace(
                    init_state, init_state + (init_state * v / 100), periods - 1
                )
            else:
                df_bel_future[k] = (
                    df_bel_last_year[k] + df_bel_last_year[k] * v / 100
                ).values

        return df_bel_future

    def compute_scenarii(
        self,
        df,
        dscenarii,
        priors_sigmas,
        model_name="orbit",
        ntimes=3,
        logistic=False,
        add_market=False,
        add_competition=False,
        adstock=False,
    ):
        """

        :param df: df_bel
        :param dscenarii:
        :param ntimes:
        :param logistic:
        :param add_market:
        :param add_competition:

        """

        # Adstock initialization
        L, P, D = np.arange(2, 5), np.arange(2, 5), np.arange(1, 4)
        adstock_features = ["A&P"]
        adstock_params = {}
        for feat in adstock_features:
            adstock_params[feat] = {"L": L, "P": P, "D": D}

        df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
        df.fillna(0, inplace=True)

        brands_name = df.Brand.unique()
        scenarii = [*product(*list(dscenarii.values()))]
        print(
            "\nComputing {} scenarii\n==========================".format(len(scenarii))
        )
        features = list(dscenarii.keys())

        last_year = df.Date.iloc[-1].year
        last_years = df.Date.dt.year.apply(str).unique().tolist()
        percent_columns_names = ["% " + k for k in features]
        results_columns_name = (
            ["ds"] + percent_columns_names + features + ["Sales in volume"]
        )
        years = ["2022", "2023", "2024"]

        gen_features_name = lambda k: [k + y for y in last_years + years]
        resumed_results_columns_name = (
            percent_columns_names
            + [
                elem
                for subset in [
                    gen_features_name(f"{feat} ")
                    for feat in features + ["Sales volume"]
                ]
                for elem in subset
            ]
            + ["3Y forecasts sales"]
        )

        df_total_results = pd.DataFrame(columns=["Brand"] + results_columns_name)
        df_total_resumed_results = pd.DataFrame(
            columns=["Brand"] + resumed_results_columns_name
        )

        suffix = ""
        if add_market:
            suffix = "_markets"
            df_markets_or_compet = self.dm.get_df_markets(self.df)

        if add_competition:
            suffix = "_competition"
            # Computing competition price and sales for bel brands
            df_markets_or_compet = self.get_competition(brands_name)

        brands_cat = {brand: [] for brand in brands_name}
        if add_market or add_competition:
            brands_cat = self.brands_cat
            df_markets_or_compet["Date"] = pd.to_datetime(
                df_markets_or_compet.Date, format="%Y-%m-%d"
            )
            df = pd.merge(df, df_markets_or_compet, on="Date")

        # pbar = tqdm(list(set(brands_name) - set(["NURISHH"])), ascii=True)
        # pbar = tqdm(brands_name, ascii=True)
        pbar = tqdm(["BOURSIN"], ascii=True)
        for brand in pbar:
            all_features = features + brands_cat[brand]
            pbar.set_description(f"Brands|{brand}")
            print(
                "\nFeatures taking into account for this brand scenario computation...\n"
                "====================================================================\n",
                all_features,
            )
            # Results with all the forecasts

            df_results = pd.DataFrame(columns=results_columns_name)
            list_results = [df_results]

            # Resumed results, raw of scenario and sum of future forecasts
            df_resumed_results = pd.DataFrame(columns=resumed_results_columns_name)
            list_resumed_results = [df_resumed_results]

            df_bel_brand = df[df.Brand == brand]
            df_fcst = (
                df_bel_brand[all_features + ["Date", "Sales in volume"]]
                .rename(columns={"Date": "ds", "Sales in volume": "y"})
                .copy()
            )

            df_fcst_std = pd.DataFrame()
            if model_name == "orbit":
                if logistic:
                    # model = DLT(
                    #     response_col="y",
                    #     date_col="ds",
                    #     regressor_col=all_features,
                    #     regressor_sign=[priors_sigmas[feat][0] for feat in features]
                    #     + ["="] * len(brands_cat[brand]),
                    #     regressor_beta_prior=[
                    #         priors_sigmas[feat][1] for feat in features
                    #     ]
                    #     + [0.25] * len(brands_cat[brand]),
                    #     regressor_sigma_prior=[
                    #         priors_sigmas[feat][2] for feat in features
                    #     ]
                    #     + [1] * len(brands_cat[brand]),
                    #     global_trend_option="logistic",
                    #     estimator="stan-mcmc",
                    #     seasonality=52,
                    #     num_warmup=1000,
                    #     num_sample=1000,
                    #     seed=2022,
                    # )
                    model = DLT(
                        response_col="y",
                        date_col="ds",
                        regressor_col=all_features,
                        global_trend_option="logistic",
                        estimator="stan-mcmc",
                        seasonality=52,
                        num_warmup=1000,
                        seed=1000,
                    )
                else:
                    # model = DLT(
                    #     response_col="y",
                    #     date_col="ds",
                    #     regressor_col=all_features,
                    #     regressor_sign=[priors_sigmas[feat][0] for feat in features]
                    #     + ["="] * len(brands_cat[brand]),
                    #     estimator="stan-mcmc",
                    #     seasonality=52,
                    #     num_warmup=1000,
                    #     num_sample=1000,
                    #     seed=2022,
                    # )
                    model = DLT(
                        response_col="y",
                        date_col="ds",
                        regressor_col=all_features,
                        num_warmup=1000,
                        seed=1000,
                    )

            else:
                # Prophet
                # A model for each brand
                df_fcst_std["cap"] = df_fcst.y.max() * 5
                df_fcst_std["floor"] = df_fcst.y.min() / 3

                if logistic:
                    model = Prophet(
                        growth="logistic",
                        daily_seasonality=False,
                        weekly_seasonality=False,
                    )
                else:
                    model = Prophet(
                        growth="linear",
                        daily_seasonality=False,
                        weekly_seasonality=False,
                    )

                for feature in all_features:
                    model.add_regressor(feature)

            # Fitting model
            df_fcst = df_fcst.sort_values(by="ds")
            df_fcst = df_fcst.drop_duplicates()
            df_fcst = df_fcst.reset_index(drop=True)

            # Adstock
            df_fcst.loc[:, adstock_features] = run_adstock_loop(
                df_fcst, adstock_features, "y", adstock_params
            )[adstock_features]

            # Apply standard scaler to features
            model_std = StandardScaler()
            df_fcst_std.loc[:, ["ds", "y"]] = df_fcst[["ds", "y"]]
            df_fcst_std.loc[:, all_features] = model_std.fit_transform(
                df_fcst[all_features]
            )

            model.fit(df_fcst_std)
            # print(model.get_regression_coefs())
            last_year_sales = df_bel_brand[
                df_bel_brand.Date.dt.year == df_bel_brand.Date.iloc[-1].year
            ]["Sales in volume"].sum()

            # Categories/Competition Categories name
            if add_market or add_competition:
                df_features, df_features_mape_errors = self.forecasting_features(
                    df,
                    brands_cat[brand],
                    logistic=logistic,
                )

            for scenario in tqdm(scenarii, ascii=True, desc="Scenarios"):
                # Loop for generalising number of years, or any periods
                # df[df.Brand == brand]
                df_scenario = self.create_scenario_data_per_year(
                    df_bel_brand, features, scenario
                )
                for _ in range(0, ntimes - 1):
                    df_tmp = self.create_scenario_data_per_year(
                        df_scenario, features, scenario
                    )
                    df_scenario = pd.concat([df_scenario, df_tmp])

                # Add markets/competitions regressors to df_scenario
                if add_market or add_competition:
                    df_scenario = pd.merge(df_features, df_scenario, on="Date")

                # Renaming Date to ds
                df_scenario = df_scenario.rename(columns={"Date": "ds"})

                # print("\n DataFrame Scenario\n=====================")
                # print_df_overview(df_scenario)
                df_scenario_std = pd.DataFrame()
                df_scenario_std["ds"] = df_scenario["ds"]
                df_scenario_std.loc[:, all_features] = model_std.transform(
                    df_scenario[all_features]
                )

                if model_name == "orbit":
                    fcst = model.predict(df_scenario_std)
                    fcst = fcst.rename(columns={"prediction": "yhat"})
                else:
                    # Prophet
                    # Predicting model on a scenario
                    df_scenario_std["cap"] = df_fcst_std["cap"].iloc[0]
                    df_scenario_std["floor"] = df_fcst_std["floor"].iloc[0]

                    fcst = model.predict(df_scenario_std)
                    fcst["yhat"] = fcst.yhat.apply(lambda x: 0 if x < 0 else x)

                df_scenario["Sales in volume"] = fcst["yhat"]

                for k, v in zip(features, scenario):
                    df_scenario["% " + k] = v

                # Lambda functions
                gen_resumed_results = lambda df, feature, years, method: [
                    df[df.ds.dt.year == int(y)].describe()[feature][method]
                    for y in years
                ]
                gen_sum = lambda df, feature, years: [
                    df[df.ds.dt.year == int(y)][feature].sum() for y in years
                ]
                gen_last = lambda df, feature, years: [
                    df[df.ds.dt.year == int(y)][feature].iloc[-1] for y in years
                ]

                df_bb = df_bel_brand.rename(columns={"Date": "ds"})
                df_to_add = pd.DataFrame(
                    [
                        list(scenario)
                        + [
                            elem
                            for subset in [
                                gen_sum(df_bb, feat, last_years)
                                + gen_sum(df_scenario, feat, years)
                                if feat in ["A&P", "Promo Cost"]
                                else gen_last(df_bb, feat, last_years)
                                + gen_last(df_scenario, feat, years)
                                for feat in features
                            ]
                            for elem in subset
                        ]
                        + gen_sum(df_bb, "Sales in volume", last_years)
                        + gen_sum(fcst, "yhat", years)
                        + [df_scenario["Sales in volume"].sum()]
                    ],
                    columns=resumed_results_columns_name,
                )
                list_results.append(df_scenario[results_columns_name])
                list_resumed_results.append(df_to_add)

                # print_df_overview(df_scenario[results_columns_name])
                # print(df_to_add)

            df_results = pd.concat(list_results, ignore_index=True)
            df_resumed_results = pd.concat(list_resumed_results, ignore_index=True)
            df_results.rename(columns={"ds": "Date"}, inplace=True)
            df_resumed_results.rename(columns={"ds": "Date"}, inplace=True)
            save_scenarios_path = os.path.join(self.spath, self.country)

            if not os.path.exists(save_scenarios_path):
                os.makedirs(save_scenarios_path)

            df_results.round(1).to_excel(
                save_scenarios_path
                + f"/{self.country.lower()}_{brand}_scenarii{suffix}.xlsx",
                index=False,
            )

            df_resumed_results.round(1).to_excel(
                save_scenarios_path
                + f"/{self.country.lower()}_{brand}_resumed_scenarii{suffix}.xlsx",
                index=False,
            )

            df_results["Brand"] = brand
            df_resumed_results["Brand"] = brand

            df_total_results = pd.concat([df_total_results, df_results])
            df_total_resumed_results = pd.concat(
                [df_total_resumed_results, df_resumed_results]
            )

        return df_total_results, df_total_resumed_results
