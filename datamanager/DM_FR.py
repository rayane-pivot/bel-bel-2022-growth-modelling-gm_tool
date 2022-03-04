from datamanager.DataManager import DataManager
import datetime
import pandas as pd

class DM_FR(DataManager):
    """ DataManager for French data"""
    _country = 'FR'
    _df_channels = dict()

    def ad_hoc_FR(self, json_sell_out_params):
        df = super().fill_df(json_sell_out_params, self._country)
        print("VIRER 2022, on s arrete a decembre 2021")
        print("shift les sub categories vers categories, care about Plant Based")
        print("faire DFBEL")

        """TODO: TOUS LES MUST HAVE : Sub Category shift -> Category, CARE ABOUT PLANT BASED
        Ajouter la catégorie Plant Based (IRI à alternative végétale)
        Sous-catégorie :
        Classique : Nice to have
        Spécialité : Must have (compétition à valider pour Port Salut et Cousteron)
        Bloc : Nice to have
        Chèvre : Nice to have
        Frais à tartiner : Must have
        Enfant : Must have
        Salade : Nice to have
        Tranche à froid : Must have
        Apéritif : Nice to have
        Brebis : Nice to have
        Dessert : Nice to have
        """

        df.Date = df.Date.apply(lambda x:datetime.datetime.strptime(x[-10:], '%d-%m-%Y').strftime('%Y-%m-%d'))
        for channel, group in df.groupby('Channel', as_index=False):
            self.add_df_channel(key=channel, df=group)
        #Here self._df should sum of df.groupby('Channel)
        self._df = df
    
    def fill_df_bel(self, json_sell_out_params):
        assert not self._df.empty, 'df is empty, call ad_hoc_USA() or load() first'
        df = self._df.copy()
        df.Date = pd.to_datetime(df.Date)
        df.Date = df.Date.dt.strftime('%Y-%m-%d')
        bel_brands = json_sell_out_params.get(self._country).get('bel_brands')
        df_bel=df[df.Brand.isin(bel_brands)].groupby(['Date', 'Brand'], as_index=False)[['Price per volume', 'Sales in volume', 'Sales in value', 'Distribution']].agg({'Price per volume':'mean', 'Sales in volume':'sum', 'Sales in value':'sum', 'Distribution':'mean'})

        #TODO: CODE A&P PAREIL QUE POUR USA normalement, a checker
        #TODO: CARE code for france in FRANCE COMMERCIAL, ignore FRANCE OVERSEES et l'autre
        PATH_FINANCE = json_sell_out_params.get(self._country).get('dict_path').get('PATH_FINANCE').get('Total Country')
        AP_CODES = json_sell_out_params.get(self._country).get('A&P_codes')
        FINANCE_COLS = json_sell_out_params.get(self._country).get('A&P_columns')
        FINANCE_RENAMING_COLS = json_sell_out_params.get(self._country).get('finance_renaming_columns')
        DATE_MIN = json_sell_out_params.get(self._country).get('dates_finance').get('Min')
        DATE_MAX = json_sell_out_params.get(self._country).get('dates_finance').get('Max')
        FINANCE_HEADER = json_sell_out_params.get(self._country).get('Finance').get('header')
        df_finance = self.fill_Finance(path=PATH_FINANCE, finance_cols=FINANCE_COLS, finance_renaming_columns=FINANCE_RENAMING_COLS, header=FINANCE_HEADER)
        df_finance = self.compute_Finance(df_finance, AP_CODES, DATE_MIN, DATE_MAX)
        df_bel = pd.merge(df_bel, df_finance, on=['Brand', 'Date'], how='left')

        #TODO: 2 ans pour les innovations pareil.
        PATH_INNO = json_sell_out_params.get(self._country).get('dict_path').get('PATH_INNO').get('Total Country')
        DATE_BEG = json_sell_out_params.get(self._country).get('Inno').get('date_beg')
        INNO_HEADER = json_sell_out_params.get(self._country).get('Inno').get('header')
        INNOVATION_DURATION = json_sell_out_params.get(self._country).get('Inno').get('innovation_duration')
        df_inno = self.fill_Inno(path=PATH_INNO, header=INNO_HEADER)
        df_inno = self.compute_Inno(df=df_inno, date_begining=DATE_BEG, innovation_duration=INNOVATION_DURATION)
        df_bel = pd.merge(df_bel, df_inno, on=['Brand', 'Date'], how='left')

        self._df_bel = df_bel

    def get_df_channels(self):
        return self._df_channels

    def get_df_by_channel(self, channel):
        assert channel in self.get_df_by_channels.keys(), f'{channel} not in df_channels'
        return self.get_df_channels.get(channel)
    
    def add_df_channel(self, key, df):
        self._df_channels[key] = df