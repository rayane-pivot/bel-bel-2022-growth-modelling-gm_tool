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
    #data_manager.fill_df_bel(json_sell_out_params)

    # data_manager.load('view/USA_df_post_processing.xlsx')
    print(data_manager.get_df().shape)
    print(data_manager.get_df().head())
    # print(data_manager.get_df_bel().shape)


    for channel, df in data_manager.get_df_channels().items():
        df.to_excel(f"view/KSA/KSA_{channel}_df_1503.xlsx", index=False)

    data_manager.get_df().to_excel('view/KSA/KSA_df_1503.xlsx', index=False)
    # data_manager.get_df_bel().to_excel('view/USA_df_bel_0303.xlsx')
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

        brand_positioning_matrix.to_excel(f'view/KSA/KSA_{channel}_brand_positioning_matrix_1503.xlsx')
if __name__ == "__main__":
    main()
