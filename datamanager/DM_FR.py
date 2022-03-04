from datamanager.DataManager import DataManager
import datetime
import pandas as pd

class DM_FR(DataManager):
    """ DataManager for French data"""
    _country = 'FR'
    _df_channels = dict()

    def ad_hoc_FR(self, json_sell_out_params):
        df = super().fill_df(json_sell_out_params, self._country)
        df.Date = df.Date.apply(lambda x:datetime.datetime.strptime(x[-10:], '%d-%m-%Y').strftime('%Y-%m-%d'))
        for channel, group in df.groupby('Channel', as_index=False):
            self.add_df_channel(key=channel, df=group)
        #Here self._df should sum of df.groupby('Channel)
        self._df = df
        

    def get_df_channels(self):
        return self._df_channels
    
    def add_df_channel(self, key, df):
        self._df_channels[key] = df