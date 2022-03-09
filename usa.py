import json

import pandas as pd

from datamanager.DM_USA import DM_USA
from model.M_USA import M_USA

PATH_TO_PARAMS = "assets/params.json"
PATH_TO_OUTPUTS = "view/"


def main():
    with open(PATH_TO_PARAMS, "r") as f:
        json_sell_out_params = json.load(f)
    country = "USA"
    data_manager = DM_USA()
    data_manager.ad_hoc_USA(json_sell_out_params)
    data_manager.fill_df_bel(json_sell_out_params)

    # data_manager.load('view/USA_df_post_processing.xlsx')
    # print(data_manager.get_df().shape)
    # print(data_manager.get_df_bel().shape)

    # data_manager.get_df().to_excel('view/USA_df_postprocessing_0303.xlsx', index=False)
    # data_manager.get_df_bel().to_excel('view/USA_df_bel_0303.xlsx')

    model = M_USA()

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
    # brand_positioning_matrix.to_excel(PATH_TO_OUTPUTS + 'USA_brand_positioning_matrix_0303.xlsx')

    brand_scorecard = model.compute_brand_scorecard(
        data_manager.get_df(),
        data_manager.get_df_bel(),
        json_sell_out_params=json_sell_out_params,
        country=country,
    )

    # brand_scorecard.to_excel('view/USA_brand_scorecard_0303.xlsx')

    attack_init_state = model.compute_attack_init_state(
        df=data_manager.get_df(),
        df_bel=data_manager.get_df_bel(),
        json_sell_out_params=json_sell_out_params,
        country=country,
    )

    # attack_init_state.to_excel(f'view/USA_attack_init_state_0703.xlsx', index=False)

    (
        df_brand_scaled,
        df_category_scaled,
        df_market_brand_scaled,
        capacity_to_win,
    ) = model.compute_Capacity_to_Win(
        df=data_manager.get_df(),
        df_bel=data_manager.get_df_bel(),
        json_sell_out_params=json_sell_out_params,
        country=country,
    )

    # with pd.ExcelWriter("view/USA/capacity_to_win_0803_2.xlsx") as writer:
    #     df_brand_scaled.to_excel(writer, sheet_name='Brand_Score')
    #     df_category_scaled.to_excel(writer, sheet_name='Market_Score')
    #     df_market_brand_scaled.to_excel(writer, sheet_name='Market_Brand_Score')
    #     capacity_to_win.to_excel(writer, sheet_name='CTW')


if __name__ == "__main__":
    main()
