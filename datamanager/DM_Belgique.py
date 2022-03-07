import pandas as pd
import datetime as dt
from datamanager.DataManager import DataManager


class DM_Belgique(DataManager):
    """ DataManager for Belgium data"""
    PATH = "data/Belgium_Data.xlsx"
    PATH_TO_DATES = "data/raw_data_minimal.xlsx"
    
    def open_excel(self, path=None):
        if path:
            self.df = pd.read_csv(path)
            return
        # self.PATH=path
        df_preproc = pd.read_excel(self.PATH, sheet_name="total belgium", engine="openpyxl")
        dates = pd.read_excel(self.PATH_TO_DATES, sheet_name="dates", engine="openpyxl")["dates"].values
        
        df_final = pd.DataFrame()

        for index, row in df_preproc.iloc[2:, 1:].iterrows():
            dict_temp = dict()
            dict_temp["Category"] = [row["Unnamed: 1"]] * 40
            dict_temp["Sub Category"] = [row["Unnamed: 2"]] * 40
            dict_temp["Brand"] = [row["Unnamed: 3"]] * 40
            dict_temp["Date"] = dates
            dict_temp["Period"] = [i for i in range(1, 41, 1)]
            dict_temp["Sales in value"] = row.iloc[3:43].values
            dict_temp["Sales in volume"] = row.iloc[43:83].values
            dict_temp["Distribution"] = row.iloc[83:123].values
            dict_temp["Price per volume"] = row.iloc[123:163].values
            dict_temp["Price without promo"] = row.iloc[163:203].values
            dict_temp["Sales value with promo"] = row.iloc[203:243].values
            dict_temp["Sales volume with promo"] = row.iloc[243:].values
            
            df_final = pd.concat([df_final, pd.DataFrame.from_dict(dict_temp)], ignore_index=True)
            
        df_final["Sub Category"] = df_final["Sub Category"].fillna("ALL SUB CATEGORIES") 
        df_final["Brand"] = df_final["Brand"].fillna("ALL BRANDS") 
        df_final['Date'] = df_final['Date'].apply(lambda x: dt.datetime.strftime(x, "%Y-%m-%d"))
        self.df = df_final

    
