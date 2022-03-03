import pandas as pd
import json
from pydoc import locate
import calendar
import datetime as dt

class DataManager():
    """ABSTRACT class for sales data"""
    """" Class that handles data processing from csv to usable dataframes """

    _df = pd.DataFrame()
    _df_bel = pd.DataFrame()

    def open_excel(self, json_sell_out_params, country):
        headers = json_sell_out_params.get(country).get('header_lines')
        sheet_name = json_sell_out_params.get(country).get('sheet_name')
        
        dict_distrib = dict()
        dict_path = json_sell_out_params.get(country).get('dict_path').get('PATH_SALES')
        for distrib, PATH in dict_path.items():
            dict_temp = pd.read_excel(PATH, header=headers, sheet_name=sheet_name)
            dict_distrib[distrib] = dict_temp
        return dict_distrib

    def fill_df(self, json_sell_out_params, country):
        dict_distrib = self.open_excel(json_sell_out_params, country)
        df_concat = pd.DataFrame()
        for distrib, dict_df in dict_distrib.items():
            df = pd.concat(dict_df.values(), axis=0)
            if json_sell_out_params.get(country).get('levels'):
                df.columns = df.columns.droplevel(0)
            df.columns = json_sell_out_params.get(country).get('sales_renaming_columns')
            df['Channel'] = distrib
            df_concat = pd.concat([df_concat, df])
        return df_concat.reset_index(drop=True)

    def fill_Finance(self, path:str, finance_cols:list, finance_renaming_columns:list, header:list):
        #Load finance file and some formating
        df_finance = pd.read_excel(path, header=header)
        df_finance = df_finance[finance_cols]
        #Rename columns
        df_finance.columns = finance_renaming_columns
        #Handle dates
        df_finance['Month'] = df_finance['Year'].apply(lambda x:int(x[5:8]))
        df_finance['Year'] = df_finance['Year'].apply(lambda x:int(x[:4]))
        return df_finance
    
    def fill_Inno(self, path:str, header:list):
        #Load innovation file and some formating
        df_ino = pd.read_excel(path, header=header)
        #rename Brands
        df_ino = df_ino.rename(columns = {"MAJOR BRAND_PRIBEL  [ MAJOR BRAND_PRIBEL ]":'Brand'})
        #Remove 'all categories'
        df_ino = df_ino[~df_ino['Product'].str.contains('ALL CATEGORIES')]
        #Convert columns names to date format
        cols = [x for x in df_ino.columns if 'Week' in x]
        df_ino=df_ino.rename(columns={x:dt.datetime.strftime(dt.datetime.strptime(x.split()[-1], '%m-%d-%y'), '%Y-%m-%d') for x in cols})
        #remove unwanted columns
        df_ino = df_ino.drop(columns=[x for x in df_ino.columns if x in ['Product', 'Dollar Sales 2018-2021 OK']])
        return df_ino

    def load(self, path):
        self._df = pd.read_excel(path)

    def load_df_bel(self, path):
        self._df_bel = pd.read_excel(path)

    def fill_df_bel_old(self):
        df_bel = self.df[self.df['Brand'].isin(self.bel_brands)]
        self.df_bel = df_bel

    def count_num_sundays_in_month(self, year, month):
            day_to_count = calendar.SUNDAY
            matrix = calendar.monthcalendar(year, month)
            num_days = sum(1 for x in matrix if x[day_to_count] != 0)
            return num_days
    
    def assert_dataframe(self):
        """HERE ASSERT DF COLUMNS NAMES AND TYPES"""
        """READ IN JSON FILE, colmuns, types, etc"""
        """assert shape for Category-Sub Category-Brand"""
        with open('assets/data_params.json', 'r') as f:
            params = json.load(f)
        for col in params["columns"]:
            assert col["column"] in self.df.columns, f'Column not found: {col["column"]}'
            var_type = locate(col['type'])
            assert isinstance(self.df[col["column"]][0], var_type), f'Column {col["column"]} of type {type(self.df[col["column"]][0])} is not of type {var_type}' 
        print('columns and types are correct')
    
    def get_df(self):
        assert not self._df.empty, 'df is empty, call ad_hoc_COUNTRY() or load_df() first'
        return self._df

    def get_df_bel(self):
        assert not self._df_bel.empty, 'df_bel is empty, call fill_df_bel() or load_df_bel() first'
        return self._df_bel

    def compute_Finance_old(self, json_params, country):
        #Compute A&P
        df_AP = pd.read_excel(json_params.get(self._country).get('dict_path').get('PATH_FINANCE').get('Total Country'), header=17)
        #Filter by country
        df_AP = df_AP[df_AP['MANAGERIAL EPM'] == country]
        #Filter brands for the study
        df_AP = df_AP[df_AP['CODE EPM'].isin(self.aandp_codes)]
        df_final = df_AP[['YEAR EPM', 
                          'CODE EPM', 
                          'R4100 - ADVERTISING', 
                          'R4200 - PROMOTION - CONSUMERS', 
                          'R1000 - NET SALES', 
                          'MVC - Margin on variable costs']]
        #Rename columns
        df_final = df_final.rename(columns={
            'YEAR EPM':'Year', 
            'CODE EPM':'Brand', 
            'R4100 - ADVERTISING':'Advertising', 
            'R4200 - PROMOTION - CONSUMERS':'Promotion', 
            'R1000 - NET SALES':'Sell-in', 
            'MVC - Margin on variable costs':'MVC'
            })
        #ABS for Advertising and Promotion
        df_final['Advertising'] = df_final['Advertising'].abs()
        df_final['Promotion'] = df_final['Promotion'].abs()
        #Get Brand from code
        df_final['Brand'] = df_final['Brand'].apply(lambda x:x.split(sep='-')[-1].strip())
        #### ADHOC FOR PRICES and BABYBEL
        df_final['Brand'] = df_final['Brand'].apply(lambda x: 'PRICES' if x=="PRICE'S" else x)
        df_final['Brand'] = df_final['Brand'].apply(lambda x: 'BABYBEL' if x=="MINI BABYBEL" else x)
        #Handle dates
        df_final['Month'] = df_final['Year'].apply(lambda x:int(x[5:8]))
        df_final['Year'] = df_final['Year'].apply(lambda x:int(x[:4]))
        df_final = df_final.fillna(0.0)
        #Months to week
        df_final['number of weeks'] = df_final.apply(lambda x:self.count_num_sundays_in_month(x.Year, x.Month), axis=1)
        df_final['A&P'] = df_final.apply(lambda x: (x.Advertising + x.Promotion) / x['number of weeks'] * 1000, axis=1)
        df_final['Sell-in'] = df_final.apply(lambda x: x['Sell-in'] / x['number of weeks'] * 1000, axis=1)
        df_final['MVC'] = df_final.apply(lambda x: x['MVC'] / x['number of weeks'] * 1000, axis=1)
        #Duplicate for n weeks
        full_idx = pd.date_range(start='2017-12-31', end='2021-12-26', freq='W')
        df_test = pd.DataFrame(index=full_idx)
        df_test['Year'] = df_test.index.year
        df_test['Month'] = df_test.index.month
        df_concat = pd.DataFrame()
        for brand in df_final.Brand.unique():
            df_concat = pd.concat([df_concat, pd.merge(df_final[df_final.Brand==brand], df_test.reset_index(), on=['Year', 'Month']).rename(columns={'index':'Date'})])
        #Change date type to str
        df_concat['Date'] = df_concat['Date'].apply(lambda x : dt.datetime.strftime(x, "%Y-%m-%d"))
        return df_concat[['Brand', 'Date', 'A&P', 'Sell-in', 'MVC']]

    


    
