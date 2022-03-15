import json

from datamanager.DM_FR import DM_FR
from model.M_FR import M_FR

PATH_TO_PARAMS = "assets/params.json"
PATH_TO_OUTPUTS = "view/"


def main():
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
    main()
