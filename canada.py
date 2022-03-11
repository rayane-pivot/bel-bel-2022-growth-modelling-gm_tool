import json

import pandas as pd

from datamanager.DM_CAN import DM_CAN
from model.M_CAN import M_CAN

PATH_TO_PARAMS = "assets/params.json"
PATH_TO_OUTPUTS = "view/CAN/"


def main():
    with open(PATH_TO_PARAMS, "r") as f:
        json_sell_out_params = json.load(f)
    
    country = "CAN"

    data_manager = DM_CAN()
    data_manager.ad_hoc_CAN(json_sell_out_params)
    data_manager.fill_df_bel(json_sell_out_params)

    #data_manager.get_df().to_excel("view/CAN/CAN_df_postprocessing.xlsx", index=False)
    # data_manager.get_df_bel().to_excel("view/CAN/CAN_df_bel.xlsx", index=False)

    model = M_CAN()

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

    # brand_positioning_matrix.to_excel(PATH_TO_OUTPUTS + 'CAN_brand_positioning_matrix_1103.xlsx')
    
    attack_init_state = model.compute_attack_init_state(
        df=data_manager.get_df(),
        df_bel=data_manager.get_df_bel(),
        json_sell_out_params=json_sell_out_params,
        country=country,
    )

    attack_init_state.to_excel(f'view/CAN/CAN_attack_init_state_1103.xlsx', index=False)


if __name__ == "__main__":
    main()
