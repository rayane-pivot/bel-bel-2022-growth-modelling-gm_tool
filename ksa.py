import json

import pandas as pd

from datamanager.DM_KSA import DM_KSA

PATH_TO_PARAMS = "assets/params.json"
PATH_TO_OUTPUTS = "view/USA/"


def main():
    with open(PATH_TO_PARAMS, "r") as f:
        json_sell_out_params = json.load(f)
    country = "KSA"
    data_manager = DM_KSA()
    data_manager.ad_hoc_KSA(json_sell_out_params)
    data_manager.fill_df_bel(json_sell_out_params)

    # data_manager.load('view/USA_df_post_processing.xlsx')
    print(data_manager.get_df().shape)
    # print(data_manager.get_df_bel().shape)

    # data_manager.get_df().to_excel('view/USA_df_postprocessing_0303.xlsx', index=False)
    # data_manager.get_df_bel().to_excel('view/USA_df_bel_0303.xlsx')

if __name__ == "__main__":
    main()
