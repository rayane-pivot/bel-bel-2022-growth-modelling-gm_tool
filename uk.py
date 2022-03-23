import json

import pandas as pd
import datetime as dt

from datamanager.DM_UK import DM_UK
from model.Model import Model

PATH_TO_PARAMS = "assets/params.json"

def main() -> None:
    with open(PATH_TO_PARAMS, "r") as f:
        json_sell_out_params = json.load(f)

    country = "UK"
    date = dt.datetime.now()

    data_manager = DM_UK()
    data_manager.ad_hoc_UK(json_sell_out_params=json_sell_out_params)
    data_manager.fill_df_bel(json_sell_out_params=json_sell_out_params)

    # data_manager.get_df().to_excel(f"view/UK/{country}_df_{date.strftime('%d%m')}.xlsx", index=False)
    # data_manager.get_df_bel().to_excel(f"view/UK/{country}_df_bel_{date.strftime('%d%m')}.xlsx", index=False)

    model = Model()

    # print(model.filter_data(data_manager.df))
    year1 = (
        json_sell_out_params.get(country).get("brand_positioning_matrix").get("year1")
    )
    year2 = (
        json_sell_out_params.get(country).get("brand_positioning_matrix").get("year2")
    )
    year_min = (
        json_sell_out_params.get(country)
        .get("brand_positioning_matrix")
        .get("year_min")
    )

    brand_positioning_matrix = model.compute_brand_positioning_matrix(
        data_manager.get_df(), year_min=year_min, year1=year1, year2=year2
    )
    # brand_positioning_matrix.to_excel(f"view/UK/{country}_brand_positioning_matrix_{date.strftime('%d%m')}.xlsx")

if __name__ == "__main__":
    main()