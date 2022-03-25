import json

import pandas as pd
import datetime as dt

from datamanager.DM_GER import DM_GER
from model.Model import Model

PATH_TO_PARAMS = "assets/params.json"
PATH_TO_OUTPUTS = "view/GER/"


def main():
    with open(PATH_TO_PARAMS, "r") as f:
        json_sell_out_params = json.load(f)
    
    country = "GER"
    date = dt.datetime.now()

    data_manager = DM_GER()
    data_manager.ad_hoc_GER(json_sell_out_params)
    data_manager.fill_df_bel(json_sell_out_params)
    
    # for channel, df in data_manager.get_df_channels().items():
    #     df.to_excel(f"view/GER/{country}_{channel}_df_{date.strftime('%d%m')}.xlsx", index=False)
    
    for channel, df_bel in data_manager.get_df_bel_channels().items():
        df_bel.to_excel(f"view/GER/{country}_{channel}_df_bel_{date.strftime('%d%m')}.xlsx", index=False)
    
    
    model = Model()
    year1 = json_sell_out_params.get("GER").get("brand_positioning_matrix").get("year1")
    year2 = json_sell_out_params.get("GER").get("brand_positioning_matrix").get("year2")
    year_min = (
        json_sell_out_params.get("GER").get("brand_positioning_matrix").get("year_min")
    )

    for channel, df in data_manager.get_df_channels().items():
        brand_positioning_matrix = model.compute_brand_positioning_matrix(df,
                                                                      year_min=year_min,
                                                                      year1=year1,
                                                                      year2=year2)

        # brand_positioning_matrix.to_excel(f"view/GER/{country}_{channel}_brand_positioning_matrix_{date.strftime('%d%m')}.xlsx")
    
    for channel in data_manager.get_df_channels().keys():
        attack_init_state = model.compute_attack_init_state(
            df=data_manager.get_df_by_channel(channel),
            df_bel=data_manager.get_df_bel_by_channel(channel),
            json_sell_out_params=json_sell_out_params,
            country="GER",
            )

        attack_init_state.to_excel(
            f"view/GER/{country}_{channel}_attack_init_state_{date.strftime('%d%m')}.xlsx", index=False
        )
if __name__ == "__main__":
    main()
