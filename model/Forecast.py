import os
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
from fbprophet import Prophet
from tqdm import tqdm
from utils.tools import mean_absolute_percentage_error


class Forecast:
    def __init__(self, country, data_manager, periods, freq, spath):
        self.dm = data_manager
        self.periods = periods
        self.freq = freq
        self.spath = os.path.join(spath.replace("data", "results"))

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

        for feature in tqdm(features_name, ascii=True, desc="Features"):

            # Preprocessing Inputs of Prophet
            df_frcast = df[["Date", feature]]
            df_frcast.columns = ["ds", "y"]
            df_frcast.ds = pd.to_datetime(df_frcast.ds)

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
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                )
            else:
                model = Prophet(
                    growth="linear",
                    daily_seasonality=False,
                    weekly_seasonality=False,
                    yearly_seasonality=True,
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
            model = Prophet()
            # model = Prophet(
            #     growth="logistic",
            #     seasonality_prior_scale=2.5,
            #     holidays_prior_scale=20,
            #     yearly_seasonality="auto",
            # )
        else:
            model = Prophet()

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
        # Compute forecast for each brand
        for brand in tqdm(brands_name, ascii=True, desc="Brands"):
            brand_fcst, future, mape = self.forecasting_one_brand(
                df,
                brand,
                features_name,
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

    def create_scenario_data(self, df_bel, features, scenario, periods, freq):
        """
        :scenario: tuple
        """

        df_bel["Date"] = pd.to_datetime(df_bel.Date, format="%Y-%m-%d")
        df_bel = df_bel.sort_values(by="Date")
        last_year = df_bel.Date.iloc[-1].year
        # Start generating one year by weeks, and remove the first cause exists in bel
        df_bel_future = (
            pd.date_range(start=df_bel.Date.iloc[-1], periods=53, freq="W")[1:]
            .to_frame()
            .reset_index(drop=True)
            .rename(columns={0: "Date"})
        )

        for k, v in zip(features, scenario):
            mean_last_year = df_bel[df_bel.Date.dt.year == last_year][k].mean()
            df_bel_future[k] = mean_last_year + (mean_last_year * v / 100)

        return df_bel_future

    def compute_scenarii(self, df_bel, dscenarii, ntimes=3):
        """ """

        df_bel["Date"] = pd.to_datetime(df_bel.Date, format="%Y-%m-%d")
        df_bel.fillna(0, inplace=True)

        scenarii = [*product(*list(dscenarii.values()))]
        print(
            "\nComputing {} scenarii\n==========================".format(len(scenarii))
        )
        features = list(dscenarii.keys())

        last_year = df_bel.Date.iloc[-1].year
        percent_columns_names = ["% " + k for k in features]
        results_columns_name = (
            ["ds"] + percent_columns_names + features + ["Sales in volume"]
        )
        years = ["2022", "2023", "2024"]
        resumed_results_columns_name = percent_columns_names + [
            str(last_year) + "sales",
            *years,
            "3Y forecasts sales",
        ]

        df_total_results = pd.DataFrame(columns=["Brand"] + results_columns_name)
        df_total_resumed_results = pd.DataFrame(
            columns=["Brand"] + resumed_results_columns_name
        )

        for brand in tqdm(df_bel.Brand.unique(), ascii=True, desc="Brands"):
            # Results with all the forecasts
            df_results = pd.DataFrame(columns=results_columns_name)
            list_results = [df_results]
            # Resumed results, raw of scenario and sum of future forecasts
            df_resumed_results = pd.DataFrame(columns=resumed_results_columns_name)
            list_resumed_results = [df_resumed_results]

            df_bel_brand = df_bel[df_bel.Brand == brand][
                features + ["Date", "Sales in volume"]
            ]
            df_prophet = df_bel_brand.rename(
                columns={"Date": "ds", "Sales in volume": "y"}
            )
            df_prophet["cap"] = df_prophet.y.max() * 7
            df_prophet["floor"] = df_prophet.y.min() / 7

            # A model for each brand
            model = Prophet(growth="logistic")
            for feature in features:
                model.add_regressor(feature)

            # Fitting model
            model.fit(df_prophet)

            last_year_sales = df_bel_brand[
                df_bel_brand.Date.dt.year == df_bel_brand.Date.iloc[-1].year
            ]["Sales in volume"].sum()
            for scenario in tqdm(scenarii, ascii=True, desc="Scenarios"):
                # Loop for generalising number of years, or any periods
                df_scenario = self.create_scenario_data(
                    df_bel_brand, features, scenario, self.periods, self.freq
                )
                for _ in range(0, ntimes - 1):
                    df_tmp = self.create_scenario_data(
                        df_scenario, features, scenario, self.periods, self.freq
                    )
                    df_scenario = pd.concat([df_scenario, df_tmp])

                # Predicting model on a scenario
                df_scenario["cap"] = df_prophet["cap"].iloc[0]
                df_scenario["floor"] = df_prophet["floor"].iloc[0]
                df_scenario = df_scenario.rename(columns={"Date": "ds"})

                fcst = model.predict(df_scenario)

                df_scenario["Sales in volume"] = fcst["yhat"]
                for k, v in zip(features, scenario):
                    df_scenario["% " + k] = v

                list_results.append(df_scenario[results_columns_name])
                list_resumed_results.append(
                    pd.DataFrame(
                        [
                            list(scenario)
                            + [
                                last_year_sales,
                                *[
                                    fcst[fcst.ds.dt.year == int(y)]["yhat"].sum()
                                    for y in years
                                ],
                                fcst["yhat"].sum(),
                            ]
                        ],
                        columns=resumed_results_columns_name,
                    )
                )

            df_results = pd.concat(list_results, ignore_index=True)
            df_resumed_results = pd.concat(list_resumed_results, ignore_index=True)
            df_results.rename(columns={"ds": "Date"}, inplace=True)
            df_resumed_results.rename(columns={"ds": "Date"}, inplace=True)
            save_scenarios_path = os.path.join(self.spath, self.country)

            if not os.path.exists(save_scenarios_path):
                os.makedirs(save_scenarios_path)

            df_results.round(1).to_excel(
                save_scenarios_path + f"/{self.country.lower()}_{brand}_scenarii.xlsx",
                index=False,
            )

            df_resumed_results.round(1).to_excel(
                save_scenarios_path
                + f"/{self.country.lower()}_{brand}_resumed_scenarii.xlsx",
                index=False,
            )

            df_results["Brand"] = brand
            df_resumed_results["Brand"] = brand

            df_total_results = pd.concat([df_total_results, df_results])
            df_total_resumed_results = pd.concat(
                [df_total_resumed_results, df_resumed_results]
            )

        return df_total_results, df_total_resumed_results
