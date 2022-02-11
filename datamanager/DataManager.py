import pandas as pd


class DataManager():
    """ABSTRACT class for sales data"""
    """" Class that handles data processing from csv to usable dataframes """
    PATH = ""
    PATH_TO_DATES = ""
    df = pd.DataFrame()
    
    def open_excel(self, path=None):
        # set self.df as ['Category', 'Sub Category', 'Brand', 'Date', 'Sales in value',
        #    'Sales in volume', 'ACV Weighted Distribution', 'Price per Volume',
        #    'Price without Promo', 'Sales value with Promo',
        #    'Sales volume with Promo']
        #do not use

        self.df = pd.read_excel(self.PATH, sheet_name="total belgium", engine="openpyxl")
        dates = pd.read_excel(self.PATH_TO_DATES, sheet_name="dates", engine="openpyxl")["dates"].values


    
