import pandas as pd
import datamanager.utils as utils
import json

from datamanager.DataManager import DataManager
#from DataManager import DataManager

class DM_USA(DataManager):
    """ DataManager for US data"""

    input_xlsx = "data/Growth Modelling - USA - 2018-2021 - Sell-Out Data (IRI).xlsx"    
    #PATH = utils.get_data_path(input_xlsx)
    PATH = input_xlsx
    def open_excel(self):

        def replace_strip(brand_name, subcategory_name):
            '''Function to remove the subcategory name in the brand name'''
            return brand_name.replace(subcategory_name, '').strip()
        
        # Initial columns of the dataframe
        columns = ["Date", "Category", "Brand", "Sales in value", "Sales in volume", "Distribution", 
                "Price per volume without promo", "Price per volume with promo", "Price per volume"]

        # Subcategories for each category in the dataset
        subcategories = {"CLASSIC SPREADS": ["CLASSIC SPREADS"], 
                        "CREAM CHEESE BLOCKS": ["CREAM CHEESE BLOCKS"], 
                        "CREAM CHEESE TUBS": ["FLAVORED TUBS", "FLAVORED WHIPPED TUBS", "PLAIN TUBS", "PLAIN WHIPPED TUBS"], 
                        "ENTERTAINING TRAYS": ["ENTERTAINING TRAYS"], 
                        "EVERYDAY BLOCKS": ["EVERYDAY BLOCKS"], 
                        "EVERYDAY SHREDDED & GRATED": ["EVERYDAY GRATED", "EVERYDAY SHREDS"], 
                        "GOURMET": ["GOURMET BLOCK / WEDGE / ROUND", "GOURMET CRUMBLED", "GOURMET FRESH ITALIAN", "GOURMET SPREADS"], 
                        "PIMENTO": ["PIMENTO"], 
                        "RICOTTA AND FARMERS": ["RICOTTA AND FARMERS"], 
                        'SINGLE SERVE': ['SINGLE SERVE FLAVORED CREAM CHEESE', 'SINGLE SERVE PLAIN CREAM CHEESE'],
                        'SLICES': ['EVERYDAY SLICES', 'PREMIUM SLICES'],
                        'SNACK': ['ALL OTHER SNACK CHEESE', 'SNACKING BAR / ROUND', 'SNACKING CRACKER CUTS', 
                                'SNACKING CUBE', 'SNACKING SPREADS', 'SNACKING STRING & STICK'],
                        'SNACKING COMBOS': ['ALL OTHER COMBOS', 'CHEESE COMBO WITH FRESH PRODUCE', 
                                            'CHEESE COMBO WITH MEAT PROTEIN', 'CHEESE DIPPER COMBOS', 'CHEESE SNACKING COMBO']}

        # Initialization of the final dataframe with processed data
        df = pd.DataFrame(columns=columns)

        # Loop over all sheets of raw excel data (13 categories)
        for i in range(1, 14):
            print(f'page {i} / 13')
            # Read sheet of the category from row 8 (row 1 to 7 are not useful)
            data = pd.read_excel(self.PATH, sheet_name=str(i), engine="openpyxl").iloc[8:, :]
            # Assign names to columns (except category which does not exist as row in the initial dataset and is added later)
            data.columns = [column for column in columns if column != "Category"]
            # Get the category name
            category = data["Brand"].iloc[0]
            # Insert the category column = list of category name x shape of the dataset
            data.insert(1, "Category", [category] * data.shape[0])    
            # Change the name of the first line for brand to ALL BRANDS as it is the sum/mean for all brands in the category 
            data["Brand"] = data["Brand"].apply(lambda x: x if x != category else "ALL BRANDS")
            # Get the subcategory name for each row and add a corresponding columns
            data["Sub Category"] = data["Brand"].apply(lambda x: [subcategory for subcategory in subcategories[category] if subcategory in x])
            data["Sub Category"] = data["Sub Category"].apply(lambda x: x[0] if len(x) > 0 else "ALL SUB CATEGORIES")
            # Remove subcategory name in brand name 
            data["Brand"] = data.apply(lambda x: replace_strip(x["Brand"], x["Sub Category"]), axis=1)
            # Add processed dataset to the final dataframe
            df = pd.concat([df, data], ignore_index=True)
            
        df = df[~df["Date"].str.contains("2018-2021")]
        df = df[~df["Brand"].str.contains("All Categories")]
        df["Date"] = pd.to_datetime(df["Date"].apply(lambda x: x[-8:]))
        periods = df.groupby("Date", as_index=False).any().reset_index()[["index", "Date"]]
        periods["index"] = periods["index"] + 1
        periods = periods.rename(columns={"index": "Period"})

        df = df.merge(periods, on="Date", how="inner")

        self.df = df

    def find_leaders(self):
        """this function is just a stash for code"""
        df_leaders['year'] = pd.DatetimeIndex(df_leaders['Date']).year
        df_leaders = df_leaders[df_leaders.Brand!='ALL BRANDS']

        df_concat = pd.DataFrame(columns=['Brand', 'Sales in volume', 'SHARE'])
        for name, group in df_leaders.groupby(['year']):
            if name==2017:
                continue
            #display(group)
            leaders = group.groupby('Brand', as_index=False)['Sales in volume'].agg(sum).sort_values(by='Sales in volume', ascending=False).iloc[:4].reset_index(drop=True)
            leaders['SHARE']=leaders['Sales in volume']/group['Sales in volume'].sum()*100
            leaders['Sales in volume'] = leaders['Sales in volume'].apply(lambda x:x/100000)
            leaders['year']=int(name)
            df_concat = pd.concat([df_concat, leaders], ignore_index=True)
            #print(group['Sales in volume'].sum().sort_values(ascending=False))

        #display(df_concat)

        #df_concat.to_excel('assets/cheese_market_leaders_USA.xlsx')