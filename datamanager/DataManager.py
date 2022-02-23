import pandas as pd
import json
from pydoc import locate

class DataManager():
    """ABSTRACT class for sales data"""
    """" Class that handles data processing from csv to usable dataframes """
    PATH = ""
    PATH_TO_DATES = ""
    df = pd.DataFrame()
    
    def __init__(self):
        self.open_excel()
        self.assert_dataframe()

    def open_excel(self, path=None):
        # set self.df as ['Category', 'Sub Category', 'Brand', 'Date', 'Sales in value',
        #    'Sales in volume', 'Distribution', 'Price per Volume',
        #    'Price without Promo', 'Sales value with Promo',
        #    'Sales volume with Promo']
        #do not use

        self.df = pd.read_excel(self.PATH, sheet_name="total belgium", engine="openpyxl")
        dates = pd.read_excel(self.PATH_TO_DATES, sheet_name="dates", engine="openpyxl")["dates"].values

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
    


    
