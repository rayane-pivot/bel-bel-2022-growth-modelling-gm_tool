import json

import pandas as pd

from datamanager.DM_KSA import DM_KSA
from model.Model import Model

PATH_TO_PARAMS = "assets/params.json"
PATH_TO_OUTPUTS = "view/USA/"



def main():
    with open(PATH_TO_PARAMS, "r") as f:
        json_sell_out_params = json.load(f)
    country = "KSA"
    data_manager = DM_KSA()
    data_manager.ad_hoc_KSA(json_sell_out_params)
    data_manager.fill_df_bel(json_sell_out_params)
    
    # for channel, df in data_manager.get_df_channels().items():
    #     df.to_excel(f"view/KSA/KSA_{channel}_df_1603.xlsx", index=False)
    
    # for channel, df_bel in data_manager.get_df_bel_channels().items():
    #     df_bel.to_excel(f"view/KSA/KSA_{channel}_df_bel_1603.xlsx", index=False)
    
    
    model = Model()
    year1 = json_sell_out_params.get("KSA").get("brand_positioning_matrix").get("year1")
    year2 = json_sell_out_params.get("KSA").get("brand_positioning_matrix").get("year2")
    year_min = (
        json_sell_out_params.get("KSA").get("brand_positioning_matrix").get("year_min")
    )

    for channel, df in data_manager.get_df_channels().items():
        brand_positioning_matrix = model.compute_brand_positioning_matrix(df,
                                                                      year_min=year_min,
                                                                      year1=year1,
                                                                      year2=year2)

        brand_positioning_matrix.to_excel(f'view/KSA/KSA_{channel}_brand_positioning_matrix_1703.xlsx')
    
    for channel in data_manager.get_df_channels().keys():
        attack_init_state = model.compute_attack_init_state(
            df=data_manager.get_df_by_channel(channel),
            df_bel=data_manager.get_df_bel_by_channel(channel),
            json_sell_out_params=json_sell_out_params,
            country="KSA",
            )

        # attack_init_state.to_excel(
        #     f"view/KSA/KSA_{channel}_attack_init_state_1603.xlsx", index=False
        # )
if __name__ == "__main__":
    main()
