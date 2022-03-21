import argparse
import datetime as dt
import itertools
import os
import pprint

import pandas as pd
from openpyxl import load_workbook

import utils
from datamanager.DataManager import DataManager
from model.Forecast import Forecast
from model.GrowthDrivers import GrowthDrivers
from utils.trends import compute_trends

PATH_TO_PARAMS = "assets/params.json"
PATH_TO_OUTPUTS = "view/"

pd.set_option("display.width", 1000)
pd.set_option("max.columns", 1000)


def print_df_overview(df):
    print(pd.concat([df.head(5), df.sample(5), df.tail(5)]))


def print_df_bel(df):
    print("DataFrame \n==========")
    print_df_overview(df)
    print("\nDescription\n============")
    print(df.describe())

    brands = df.Brand.unique()
    print("\nBrands\n======")
    print(brands)

    return brands


def print_df(df):
    print("DataFrame \n==========")
    print_df_overview(df)
    print("\nDescription\n============")
    print(df.describe())

    print("\nBrands\n======")
    print(df.Brand.unique())

    print("\nDistinct categories\n===================")
    print(df.Category.unique())

    print("\nBrands => Categories \n=======================")
    brands_cat = {
        brand: df[df.Brand == brand].Category.unique().tolist()
        for brand in df.Brand.unique()
    }
    print(brands_cat)

    print("\nDistinct Sub Categories\n=======================")
    print(df["Sub Category"].unique())

    print("\nSub Categories of each category\n=============================")

    cat_subcats = {
        cat: df[df.Category == cat]["Sub Category"].unique().tolist()
        for cat in df.Category.unique()
    }
    print(cat_subcats)

    return brands_cat, cat_subcats


def save_results_and_errors(df, df_error, spath):
    """TODO describe function

    :param df:
    :param df_error:
    :param spath:
    :returns:

    """

    with pd.ExcelWriter(spath) as writer:
        df.round(1).to_excel(
            writer,
            index=False,
        )
        df_error.round(1).to_excel(writer, header=False, startrow=len(df) + 2)


def main(args):
    if args.weeks:
        tdelta = dt.timedelta(weeks=1)
        freq = "W"
    if args.months:
        tdelta = dt.timedelta(weeks=5)
        freq = "M"

    # Data Manager
    dm = DataManager()

    # Forecast
    forecast = Forecast(
        args.country, dm, periods=args.periods, freq=freq, spath=args.path
    )

    print(f"Country : {args.geo} \n=======")

    # Load data
    if args.path:
        df = pd.read_excel(args.path + "df.xlsx")
        df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
        df["Date"] = df.Date.apply(lambda x: x.strftime("%Y-%m-%d"))

        # Ad-hoc CAN, replace "ERR" by 0
        if args.country == "CAN":
            df[["Price per volume", "Price with promo"]] = df.replace("ERR", 0)[
                ["Price per volume", "Price with promo"]
            ].applymap(float)

        df_bel = pd.read_excel(args.path + "df_bel.xlsx")
        df_bel["Date"] = pd.to_datetime(df_bel.Date, format="%Y-%m-%d")
        df_bel["Date"] = df_bel.Date.apply(lambda x: x.strftime("%Y-%m-%d"))

        if args.verbose:
            print("\nGeneral\n========")
            brands_cat, cat_subcats = print_df(df)

            print("\nBel \n====")
            bel_brands = print_df_bel(df_bel)

            print("\n Bel Brands => Categories\n========================")
            bel_brands_cat = {brand: brands_cat[brand] for brand in bel_brands}
            pprint.pprint(bel_brands_cat)

            print("\nSub Categories of each category\n=============================")
            bel_markets = list(set(list(itertools.chain(*bel_brands_cat.values()))))
            pprint.pprint({cat: cat_subcats[cat] for cat in bel_markets})

    if args.compute_trends:
        # Compute trends
        timeframes = list(df_bel.Date.unique())

        timeframes.insert(
            0,
            (dt.datetime.strptime(timeframes[0], "%Y-%m-%d") - tdelta).strftime(
                "%Y-%m-%d"
            ),
        )
        timeframes.append(
            (dt.datetime.strptime(timeframes[-1], "%Y-%m-%d") + tdelta).strftime(
                "%Y-%m-%d"
            )
        )

        df_trends = compute_trends(
            args.geo, bel_brands, start_date=timeframes[0], end_date=timeframes[-1]
        )
        df_trends = df_trends.reset_index().rename(columns={"date": "Date"})

        if args.months:
            s = (
                (df_trends["Date"] + pd.offsets.MonthEnd(0)) - df_trends["Date"]
            ).dt.days
            Ndays = 7  # This many from the end
            df_trends = (
                df_trends[s.lt(Ndays)]
                .groupby(df_trends["Date"].dt.to_period("M"))
                .sum()
            ).reset_index()
            df_trends["Date"] = df_trends.Date.dt.to_timestamp("s").dt.strftime(
                "%Y-%m-%d"
            )

        # Adhoc for CA cause trends dates and df dates do not correspond
        if args.geo == "CA":
            tdelta_day = dt.timedelta(days=1)
            df_trends["Date"] = df_trends.Date.apply(lambda x: x - tdelta_day)

        if args.verbose:
            print("\nCompute Trends\n==============")
            print_df_overview(df_trends)

        df_bel = df_bel.fillna(0)
        df_bel["Trends"] = df_bel.apply(
            lambda x: df_trends[df_trends.Date == x.Date][x.Brand].values[0], axis=1
        )
        df_bel.fillna(0, inplace=True)
        df_bel.head()

        if args.verbose:
            print("\nNew Bel\n=======")
            print_df_overview(df_bel)

    if args.forecast_categories:
        categories_name = list(df.Category.unique()) + ["TOTAL CHEESE"]
        df_fcst_categories, df_categories_mape_errors = forecast.forecasting_categories(
            df,
            categories_name,
            logistic=args.logistic,
            plot=args.plot_forecast_categories,
            splot_suffix=args.save_plot_forecast_categories,
        )

        if args.verbose:
            print("\nForecast categories\n===================")
            print_df_overview(df_fcst_categories)

            print("\nMAPE\n=====")
            print(df_categories_mape_errors)

        if args.save_forecast_categories:
            print("\n    Saving categories forecasts ....")
            fc_path = os.path.join(
                args.path.replace("data", "results"),
                f"{args.country.lower()}_categories_forecasts.xlsx",
            )

            save_results_and_errors(
                df_fcst_categories, df_categories_mape_errors, fc_path
            )

    if args.forecast_brands_with_regressors:
        (
            df_brands_fcst_res,
            feats_futures,
            df_brands_mape_errors,
        ) = forecast.forecasting_brands_with_regressors(
            df_bel,
            bel_brands,
            args.forecast_brands_with_regressors,
            logistic=args.logistic,
            plot=args.plot_forecast_brands,
            splot_suffix=args.save_plot_forecast_brands_with_regressors,
        )

        if args.verbose:
            print("\nForecast Brands with regressors\n==============================")
            print_df_overview(df_brands_fcst_res)

            print("\nMAPE\n=====")
            print(df_brands_mape_errors)

        if args.save_forecast_brands_with_regressors:
            print("\n    Saving brands with regressors forecasts ...")
            fbwr_path = os.path.join(
                args.path.replace("data", "results"),
                f"{args.country.lower()}_brands_forecasts_with_regressors.xlsx",
            )

            save_results_and_errors(
                df_brands_fcst_res, df_brands_mape_errors, fbwr_path
            )

    if args.forecast_brands_no_regressors:
        (
            df_brands_fcst_res,
            df_brands_mape_errors,
        ) = forecast.forecasting_brands_no_regressors(
            df_bel,
            logistic=args.logistic,
            plot=args.plot_forecast_brands,
            splot_suffix=args.save_plot_forecast_brands_no_regressors,
        )

        if args.verbose:
            print("\nForecast Brands no regressors\n==============================")
            print_df_overview(df_brands_fcst_res)

            print("\nMAPE\n=====")
            print(df_brands_mape_errors)

        if args.save_forecast_brands_no_regressors:
            print("\n    Saving brands no regressors forecasts ...")
            fbnr_path = os.path.join(
                args.path.replace("data", "results"),
                f"{args.country.lower()}_brands_forecasts_no_regressors.xlsx",
            )

            save_results_and_errors(
                df_brands_fcst_res, df_brands_mape_errors, fbnr_path
            )

    if args.growth_drivers_past:

        gd = GrowthDrivers(dm, bel_markets)

        dict_df_gd, dict_df_gd_scaled = gd.compute_growth_drivers_past(
            df,
            df_bel[args.growth_drivers_past],
            bel_brands,
            years=args.gd_years,
            list_controllable_features=args.list_controllable_features,
            controllable_features=args.controllable_features,
        )

        if args.verbose:
            print("\n Growth Drivers Past All years \n==============================")
            print(dict_df_gd["all"]["xgb_feature_importance"])

            print(
                "\n Growth Drivers Past All years rescaled \n======================================="
            )
            print(dict_df_gd_scaled["all"]["xgb_feature_importance"])

        if args.save_growth_drivers_past:
            fgd_path = os.path.join(
                args.path.replace("data", "results"),
                f"{args.country.lower()}_growth_drivers_past",
            )

            for year in dict_df_gd:
                print("\n    Saving growth drivers past ...")
                dict_df_gd[year]["xgb_feature_importance"].round(1).to_excel(
                    fgd_path + "_{}.xlsx".format(year)
                )

                print(
                    "\n    Saving growth drivers past rescaled controllable features ..."
                )
                dict_df_gd_scaled[year]["xgb_feature_importance"].round(1).to_excel(
                    fgd_path + "_{}_controllable_features.xlsx".format(year)
                )

    if args.scenarios:
        dscenarii = {
            "A&P": args.scenario_a_and_p,
            "Price per volume": args.scenario_price_per_volume,
            "Promo Cost": args.scenario_promo_cost,
            "Distribution": args.scenario_distribution,
        }

        # Ntimes number of years ahead we are predicting
        df_total_results, df_total_resumed_results = forecast.compute_scenarii(
            df_bel, dscenarii, ntimes=3
        )

        if args.verbose:
            print("\nScenarios\n==========")
            print(dscenarii)
            print("\n Total results\n=============")
            print_df_overview(df_total_results)
            print("\n Total resumed results\n=====================")
            print_df_overview(df_total_resumed_results)

        if args.save_scenarios:
            s_path = os.path.join(
                args.path.replace("data", "results"), f"{args.country.lower()}"
            )

            df_total_results.round(1).to_excel(
                s_path + "_all_brands_scenarii.xlsx",
                index=False,
            )
            df_total_resumed_results.round(1).to_excel(
                s_path + "_all_brands_resumed_scenarii.xlsx",
                index=False,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to df_bel.xlxs, and df.xlxs")
    parser.add_argument("--periods", type=int, help="Number of periods to predict")
    parser.add_argument(
        "--country", type=str, help="Country code for which we are computing insights"
    )
    parser.add_argument(
        "-fc", "--forecast-categories", help="Forecast markets", action="store_true"
    )
    parser.add_argument(
        "--logistic", help="Using logistic growth of Prophet", action="store_true"
    )
    parser.add_argument(
        "-sfc",
        "--save-forecast-categories",
        help="Save category forecasts in an excel file",
        action="store_true",
    )
    parser.add_argument(
        "-pfc",
        "--plot-forecast-categories",
        help="Plot category forecasts",
        action="store_true",
    )
    parser.add_argument(
        "-spfc",
        "--save-plot-forecast-categories",
        type=str,
        default="",
        help="Save plots of categories forecasts, takes the suffix, where to save",
    )
    parser.add_argument(
        "-fbnr",
        "--forecast-brands-no-regressors",
        help="Forecast brands without using regressors",
        action="store_true",
    )
    parser.add_argument(
        "-sfbnr",
        "--save-forecast-brands-no-regressors",
        help="Save brand forecast without regressors in an excel file",
        action="store_true",
    )
    parser.add_argument(
        "-spfbnr",
        "--save-plot-forecast-brands-no-regressors",
        type=str,
        default="",
        help="Save plots of brands forecasts without regressors, takes the suffix where to save",
    )
    parser.add_argument(
        "-fbwr",
        "--forecast-brands-with-regressors",
        nargs="+",
        help="Forecast brands using regressors",
    )
    parser.add_argument(
        "-sfbwr",
        "--save-forecast-brands-with-regressors",
        help="Save brand forecast with regressors in an excel file",
        action="store_true",
    )
    parser.add_argument(
        "-pfb",
        "--plot-forecast-brands",
        help="Plot brands forecasts",
        action="store_true",
    )
    parser.add_argument(
        "-spfbwr",
        "--save-plot-forecast-brands-with-regressors",
        type=str,
        default="",
        help="Save plots of brands forecasts with regressors, takes the suffix where to save",
    )
    parser.add_argument(
        "-gdp",
        "--growth-drivers-past",
        nargs="+",
        help="Compute Growth drivers past given the features",
    )
    parser.add_argument(
        "-sgdp",
        "--save-growth-drivers-past",
        help="Save growth drivers past in an excel file",
        action="store_true",
    )
    parser.add_argument(
        "--gd-years",
        nargs="+",
        type=int,
        help="Years for which to compute growth drivers",
    )
    parser.add_argument(
        "-lcf",
        "--list-controllable-features",
        nargs="+",
        help="List of controllable features",
    )
    parser.add_argument(
        "--controllable_features",
        help="Compute growth drivers with only controllable features",
        action="store_true",
    )
    parser.add_argument(
        "--scenarios", help="Compute giving scenarios", action="store_true"
    )
    parser.add_argument(
        "-ss", "--save-scenarios", help="Save scenarios results", action="store_true"
    )
    parser.add_argument(
        "-saap",
        "--scenario-a-and-p",
        type=int,
        nargs="+",
        default=[-5, -2, 0, 2, 5],
        help="Scenarios for A&P",
    )
    parser.add_argument(
        "-sppv",
        "--scenario-price-per-volume",
        type=int,
        nargs="+",
        default=[0, 1, 3, 5, 7],
        help="Scenarios for Price per volume",
    )
    parser.add_argument(
        "-spc",
        "--scenario-promo-cost",
        type=int,
        nargs="+",
        default=[-6, -3, 0, 3, 10],
        help="Scenarios for promo cost",
    )
    parser.add_argument(
        "-sdist",
        "--scenario-distribution",
        type=int,
        nargs="+",
        default=[-10, -5, 0, 5, 10],
        help="Scenarios for Distribution",
    )
    parser.add_argument("--geo", type=str, help="Geo for computing trends")
    parser.add_argument(
        "--compute-trends", help="Computing trends giving the geo", action="store_true"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Print samples of dataframes and results",
        action="store_true",
    )
    parser.add_argument(
        "-w", "--weeks", help="Time series are in weeks", action="store_true"
    )
    parser.add_argument(
        "-m", "--months", help="Time series are in months", action="store_true"
    )

    parser.add_argument(
        "--compute-auguste", help="Compute Auguste parts", action="store_true"
    )
    args = parser.parse_args()
    main(args)
