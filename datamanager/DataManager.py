import pandas as pd
import json
from pydoc import locate
import calendar
import datetime as dt

class DataManager():
    """ABSTRACT class for sales data"""
    """" Class that handles data processing from csv to usable dataframes """
    PATH = ""
    PATH_TO_DATES = ""

    PATH_INNO = ''
    PATH_AandP = ''
    PATH_PROMO_COST = ''

    df = pd.DataFrame()
    df_bel = pd.DataFrame()
    bel_brands = []
    aandp_codes = []
    
    def __init__(self, bel_brands, aandp_codes):
        """TODO describe function

        :param bel_brands: 
        :param aandp_codes: 
        :returns: 

        """
        self.bel_brands = bel_brands
        self.aandp_codes = aandp_codes
        self.open_excel()
        self.fill_df_bel()
        self.assert_dataframe()

    def open_excel(self, path=None):
        # set self.df as ['Category', 'Sub Category', 'Brand', 'Date', 'Sales in value',
        #    'Sales in volume', 'Distribution', 'Price per Volume',
        #    'Price without Promo', 'Sales value with Promo',
        #    'Sales volume with Promo']
        #do not use

        self.df = pd.read_excel(self.PATH, sheet_name="total belgium", engine="openpyxl")
        dates = pd.read_excel(self.PATH_TO_DATES, sheet_name="dates", engine="openpyxl")["dates"].values

    def fill_df_bel(self):
        df_bel = self.df[self.df['Brand'].isin(self.bel_brands)]
        self.df_bel = df_bel

    def count_num_sundays_in_month(self, year, month):
            day_to_count = calendar.SUNDAY
            matrix = calendar.monthcalendar(year, month)
            num_days = sum(1 for x in matrix if x[day_to_count] != 0)
            return num_days
    
    def compute_AandP(self, country, path):
        #Compute A&P
        df_AP = pd.read_excel(path, header=17)
        #Filter by country
        df_AP = df_AP[df_AP['MANAGERIAL EPM'] == country]
        #Filter brands for the study
        df_AP = df_AP[df_AP['CODE EPM'].isin(self.aandp_codes)]
        df_final = df_AP[['YEAR EPM', 'CODE EPM', 'R4100 - ADVERTISING', 'R4200 - PROMOTION - CONSUMERS']]
        #Rename columns
        df_final = df_final.rename(columns={'YEAR EPM':'Year', 'CODE EPM':'Brand', 'R4100 - ADVERTISING':'Advertising', 'R4200 - PROMOTION - CONSUMERS':'Promotion'})
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
        #Compute A&P per week
        df_final['number of weeks'] = df_final.apply(lambda x:self.count_num_sundays_in_month(x.Year, x.Month), axis=1)
        df_final['A&P'] = df_final.apply(lambda x: (x.Advertising + x.Promotion) / x['number of weeks'] * 1000, axis=1)
        full_idx = pd.date_range(start='2017-12-31', end='2021-12-26', freq='W')
        df_test = pd.DataFrame(index=full_idx)
        df_test['Year'] = df_test.index.year
        df_test['Month'] = df_test.index.month
        df_concat = pd.DataFrame()
        for brand in df_final.Brand.unique():
            df_concat = pd.concat([df_concat, pd.merge(df_final[df_final.Brand==brand], df_test.reset_index(), on=['Year', 'Month']).rename(columns={'index':'Date'})])
        #Change date type to str
        df_concat['Date'] = df_concat['Date'].apply(lambda x : dt.datetime.strftime(x, "%Y-%m-%d"))
        return df_concat[['Brand', 'Date', 'A&P']]

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
    


    
