import datetime as dt

import numpy as np
import pandas as pd

from datamanager.DataManager import DataManager

# from DataManager import DataManager


class DM_CAN(DataManager):
    """DataManager for CAN data"""

    _country = "CAN"
