import pandas as pd
import numpy as np
import datetime as dt

from datamanager.DataManager import DataManager
#from DataManager import DataManager

class DM_CAN(DataManager):
    """ DataManager for CAN data"""
    _country = 'CAN'