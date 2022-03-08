import json

import pandas as pd

from datamanager.DM_USA import DM_CAN

PATH_TO_PARAMS = "assets/params.json"
PATH_TO_OUTPUTS = "view/"


def main():
    with open(PATH_TO_PARAMS, "r") as f:
        json_sell_out_params = json.load(f)
    country = "CAN"
    data_manager = DM_CAN()
    # data_manager.ad_hoc_USA(json_sell_out_params)
    # data_manager.fill_df_bel(json_sell_out_params)

if __name__ == "__main__":
    main()
