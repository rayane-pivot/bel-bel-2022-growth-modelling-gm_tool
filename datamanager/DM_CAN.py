import datetime as dt

import numpy as np
import pandas as pd

from datamanager.DataManager import DataManager

# from DataManager import DataManager


class DM_CAN(DataManager):
    """DataManager for CAN data"""

    _country = "CAN"

    def ad_hoc_CAN(self, json_sell_out_params):
        df = super().fill_df(json_sell_out_params, self._country)
