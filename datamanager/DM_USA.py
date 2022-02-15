import pandas as pd
import datamanager.utils as utils
import json

from datamanager.DataManager import DataManager
#from DataManager import DataManager

class DM_USA(DataManager):
    """ DataManager for US data"""

    input_xlsx = "/Users/augustecousin/Documents/bel_gm_tool/gm_tool/data/Growth Modelling - USA - 2018-2021 - Sell-Out Data (IRI).xlsx"    
    #PATH = utils.get_data_path(input_xlsx)
    PATH = input_xlsx
    def open_excel(self):
        
        columns = ["Date", "Category", "Brand", "Sales in value", "Sales in volume", "Distribution", 
                   "Price per volume without promo", "Price per volume with promo", "Price per volume"]
        df = pd.DataFrame(columns=columns)
        
        for i in range(1, 3):
            data = pd.read_excel(self.PATH, sheet_name=str(i), engine="openpyxl").iloc[8:, :]
            data.columns = [column for column in columns if column != "Category"]
            category = data["Brand"].iloc[0]
            category = [category] * data.shape[0]
            data.insert(1, "Category", category)    
            df = pd.concat([df, data], ignore_index=True)        
        
        df = df[~df["Date"].str.contains("2018-2021")]
        df = df[~df["Brand"].str.contains("All Categories")]
        df["Date"] = pd.to_datetime(df["Date"].apply(lambda x: x[-8:]))
        periods = df.groupby("Date", as_index=False).any().reset_index()[["index", "Date"]]
        periods["index"] = periods["index"] + 1
        periods = periods.rename(columns={"index": "Period"})

        df = df.merge(periods, on="Date", how="inner")        
        
        self.df = df
