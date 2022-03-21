import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd

from datamanager.DM_FR import DM_FR
from model.M_FR import M_FR

PATH_TO_PARAMS = "assets/params.json"
PATH_TO_OUTPUTS = "view/"

pd.set_option("display.width", 1000)
pd.set_option("max.columns", 1000)


def print_df_bel(df):
    print("DataFrame \n==========")
    print(pd.concat([df.head(5), df.sample(5), df.tail(5)]))

    brands = df.Brand.unique()
    print("\nBrands\n======")
    print(brands)

    return brands


def print_df(df):
    print("DataFrame \n==========")
    print(pd.concat([df.head(5), df.sample(5), df.tail(5)]))

    print("\nBrands\n======")
    print(df.Brand.unique())

    print("\nDistinct categories\n===================")
    print(df.Category.unique())

    print("\nBrands => Categories \n=======================")
    df_brands_cat = {
        brand: df[df.Brand == brand].Category.unique().tolist()
        for brand in df.Brand.unique()
    }
    print(df_brands_cat)

    print("\nDistinct Sub Categories\n=======================")
    print(df["Sub Category"].unique())

    print("\nSub Categories of each category\n=============================")
    print(
        {
            cat: df[df.Category == cat]["Sub Category"].unique().tolist()
            for cat in df.Category.unique()
        }
    )


def main(args):

    print(f"Country : {args.geo} \n=======")

    # Load data
    if args.path:
        df = pd.read_excel(args.path + "df.xlsx")
        df_bel = pd.read_excel(args.path + "df_bel.xlsx")

        if args.verbose:
            print("\nBel \n====")
            print_dataframe(df_bel)

            print("\nGeneral\n========")
            print_dataframe(df)

    return
    # Compute trends
    geo = "FR"
    tdelta = dt.timedelta(weeks=1)
    timeframes = list(df_bel.Date.unique())
    timeframes.insert(
        0,
        (dt.datetime.strptime(timeframes[0], "%Y-%m-%d") - tdelta).strftime("%Y-%m-%d"),
    )

    with open(PATH_TO_PARAMS, "r") as f:
        json_sell_out_params = json.load(f)

    data_manager = DM_FR()
    data_manager.ad_hoc_FR(json_sell_out_params)
    data_manager.fill_df_bel(json_sell_out_params)

    for channel, df in data_manager.get_df_channels().items():
        df.to_excel(f"view/France/FRANCE_NEW/FR_{channel}_df_1403.xlsx", index=False)
    
    for channel, df_bel in data_manager.get_df_bel_channels().items():
        df_bel.to_excel(f"view/France/FRANCE_NEW/FR_{channel}_df_bel_1403.xlsx", index=False)
    
    model = M_FR()

    # print(model.filter_data(data_manager.df))
    year1 = json_sell_out_params.get("FR").get("brand_positioning_matrix").get("year1")
    year2 = json_sell_out_params.get("FR").get("brand_positioning_matrix").get("year2")
    year_min = (
        json_sell_out_params.get("FR").get("brand_positioning_matrix").get("year_min")
    )

    for channel, df in data_manager.get_df_channels().items():
        brand_positioning_matrix = model.compute_brand_positioning_matrix(df,
                                                                      year_min=year_min,
                                                                      year1=year1,
                                                                      year2=year2)

        # brand_positioning_matrix.to_excel(f'view/France/FRANCE_NEW/FR_{channel}_brand_positioning_matrix_1403.xlsx')

    # for channel, df in data_manager.get_df_channels().items():
    #     brand_scorecard = model.compute_brand_scorecard(df,
    #                                                     data_manager.get_df_bel_by_channel(channel=channel),
    #                                                     json_sell_out_params=json_sell_out_params,
    #                                                     country='FR')

    #     brand_scorecard.to_excel(f'view/France/FRANCE_NEW/FR_{channel}_brand_scorecard_1403.xlsx')

    for channel in data_manager.get_df_channels().keys():
        attack_init_state = model.compute_attack_init_state(
            df=data_manager.get_df_by_channel(channel),
            df_bel=data_manager.get_df_bel_by_channel(channel),
            json_sell_out_params=json_sell_out_params,
            country="FR",
        )

        # attack_init_state.to_excel(
        #     f"view/France/FRANCE_NEW/FR_{channel}_attack_init_state_1403.xlsx", index=False
        # )

    # df_brand_scaled, df_category_scaled, df_market_brand_scaled, capacity_to_win = model.compute_Capacity_to_Win(
    #     df = data_manager.get_df(),
    #     df_bel = data_manager.get_df_bel(),
    #     json_sell_out_params = json_sell_out_params,
    #     country = country)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to df_bel.xlxs, and df.xlxs")
    parser.add_argument(
        "--country", type=str, help="Country code for which we are computing insights"
    )
    parser.add_argument(
        "-fm", "--forecast-markets", help="Forecast markets", action="store_true"
    )
    parser.add_argument(
        "-fbnr",
        "--forecast-brands-no-regressors",
        help="Forecast brands without using regressors",
        action="store_true",
    )
    parser.add_argument(
        "-fbwr",
        "--forecast-brands-with-regressors",
        help="Forecast brands using regressors",
        action="store_true",
    )
    parser.add_argument(
        "-gdp", "--growth-drivers-past", help="Growth drivers past", action="store_true"
    )
    parser.add_argument("--scenarios", type=list, help="Compute giving scenarios")
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
        "--compute-auguste", help="Compute Auguste parts", action="store_true"
    )
    args = parser.parse_args()
    main(args)
