import sys
import json
import datetime as dt
import pandas as pd

from datamanager.DM_BEL import DM_BEL
from model.Model import Model
from model.Capacity_To_win import Capacity_To_Win

PATH_TO_PARAMS = "assets/params.json"

def main() -> None:
    with open(PATH_TO_PARAMS, "r") as f:
        json_sell_out_params = json.load(f)

    country = "BEL"
    date = dt.datetime.now()

    data_manager = DM_BEL()
    data_manager.ad_hoc_BEL(json_sell_out_params=json_sell_out_params)
    data_manager.fill_df_bel(json_sell_out_params=json_sell_out_params)

    # data_manager.get_df().to_excel(f"view/BEL/{country}_df_{date.strftime('%d%m')}.xlsx", index=False)
    # data_manager.get_df_bel().to_excel(f"view/BEL/{country}_df_bel_{date.strftime('%d%m')}.xlsx", index=False)

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
    # brand_positioning_matrix.to_excel(f"view/BEL/{country}_brand_positioning_matrix_{date.strftime('%d%m')}.xlsx")

    attack_init_state = model.compute_attack_init_state(
        df=data_manager.get_df(),
        df_bel=data_manager.get_df_bel(),
        json_sell_out_params=json_sell_out_params,
        country=country,
        )

    attack_init_state.to_excel(
        f"view/{country}/{country}_attack_init_state_{date.strftime('%d%m')}.xlsx", index=False
    )
    
    capacity_to_win = Capacity_To_Win()
    (
        df_brand_scaled,
        df_category_scaled,
        df_market_brand_scaled,
        capacity_to_win,
    ) = capacity_to_win.compute_Capacity_to_Win(
        df=data_manager.get_df(),
        df_bel=data_manager.get_df_bel(),
        json_sell_out_params=json_sell_out_params,
        country=country,
    )

    # with pd.ExcelWriter(f"view/{country}/capacity_to_win_{date.strftime('%d%m')}.xlsx") as writer:
    #     df_brand_scaled.to_excel(writer, sheet_name='Brand_Score')
    #     df_category_scaled.to_excel(writer, sheet_name='Market_Score')
    #     df_market_brand_scaled.to_excel(writer, sheet_name='Market_Brand_Score')
    #     capacity_to_win.to_excel(writer, sheet_name='CTW')


if __name__ == "__main__":
    main()