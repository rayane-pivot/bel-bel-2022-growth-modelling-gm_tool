import pandas as pd
import numpy as np
import datetime as dt

from datamanager.DataManager import DataManager
#from DataManager import DataManager

class DM_USA(DataManager):
    """ DataManager for US data"""

    input_xlsx = "data/Growth Modelling - USA - 2018-2021 - Sell-Out Data (IRI).xlsx"
    #PATH = utils.get_data_path(input_xlsx)
    PATH = input_xlsx

    PATH_INNO = 'data/Growth Modelling - USA - 2018-2021 - Rate Of Innovation (IRI).xlsx'
    PATH_FINANCE = 'data/Growth Modelling - USA - 2018-2021 - Finance Data (SAP - Flux Rio).xlsx'
    PATH_PROMO_COST = 'data/Growth Modelling - USA - 2019-2021 - Promo Cost (Xtel).xlsx'
    PATH_HH_INDEX = "data/Growth Modelling - USA - Consumer Panel -US HH penetration GM.xlsx"

    def open_excel(self):
        df_concat = pd.DataFrame()
        for page in range(1, 15):
            print(f'page {page}/14')
            df_sell = pd.read_excel(self.input_xlsx, header=7, sheet_name=str(page))
            if page==14:
                df_sell["SEGMENT_PRIBEL  [ SEGMENT_PRIBEL ]"]='PLANT BASED'
                df_sell["SUB SEGMENT_PRIBEL  [ SUB SEGMENT_PRIBEL ]"]='PLANT BASED'
            df_concat = pd.concat([df_concat, df_sell])

        df_concat = df_concat.rename(columns={
            'Time':'Date', 
            "SEGMENT_PRIBEL  [ SEGMENT_PRIBEL ]":'Category',
            "SUB SEGMENT_PRIBEL  [ SUB SEGMENT_PRIBEL ]":'Sub Category',
            "MAJOR BRAND_PRIBEL  [ MAJOR BRAND_PRIBEL ]":'Brand',
            'Dollar Sales':'Sales in value',
            'Volume Sales' : 'Sales in volume',
            'Total Points of Distribution' : 'Distribution',
            'Price per Volume' : 'Price per volume',
            'Price per Volume No Merch' : 'Price without promo',
            'Incremental Dollars' : 'Sales value with promo',
            'Incremental Volume' : 'Sales volume with promo'
        })
        df_concat = df_concat[~df_concat['Date'].str.contains('OK')]
        df_concat['Date'] = df_concat['Date'].apply(lambda x:dt.datetime.strptime(x.split()[-1], '%m-%d-%y').strftime('%Y-%m-%d'))

        df_concat = df_concat[['Date', 'Category', 'Sub Category', 'Brand', 'Sales in value', 'Sales in volume', 'Distribution', 'Price per volume', 'Price without promo', 'Sales value with promo', 'Sales volume with promo']]
        df_concat = df_concat.reset_index(drop=True)
        df_periods = pd.DataFrame(np.sort(df_concat.Date.unique())).reset_index().rename(columns={'index':'Period', 0:'Date'})
        df_periods.Period = df_periods.Period + 1
        df_concat = pd.merge(df_concat, df_periods, on='Date', how='left')
        self.df = df_concat

    def fill_df_bel(self):
        df_bel = self.df[self.df['Brand'].isin(self.bel_brands)]
    
        df_AP = self.compute_Finance('USA', self.PATH_FINANCE)
        df_bel = pd.merge(df_bel, df_AP, on=['Brand', 'Date'], how='left')
    
        df_inno = self.compute_Inno(self.PATH_INNO)
        df_bel = pd.merge(df_bel, df_inno, on=['Brand', 'Date'], how='left')
    
        df_promocost = self.compute_Promo_Cost(self.PATH_PROMO_COST)
        df_bel = pd.merge(df_bel, df_promocost, on=['Brand', 'Date'], how='left')
    
        df_hh = self.compute_HH_Index(self.PATH_HH_INDEX)
        df_bel = pd.merge(df_bel, df_hh, on=['Brand', 'Date'], how='left')
    
        self.df_bel = df_bel
        #df_bel.to_excel('assets/df_bel.xlsx')

    def compute_Inno(self, path, date_begining='2017-12-31'):
        #load excel file
        df_ino = pd.read_excel(path, header=7)
        #rename Brands
        df_ino = df_ino.rename(columns = {"MAJOR BRAND_PRIBEL  [ MAJOR BRAND_PRIBEL ]":'Brand'})
        #Remove 'all categories'
        df_ino = df_ino[~df_ino['Product'].str.contains('ALL CATEGORIES')]
        #Convert columns names to date format
        cols = [x for x in df_ino.columns if 'Week' in x]
        df_ino=df_ino.rename(columns={x:dt.datetime.strftime(dt.datetime.strptime(x.split()[-1], '%m-%d-%y'), '%Y-%m-%d') for x in cols})
        #remove unwanted columns
        df_ino = df_ino.drop(columns=['Product', 'Dollar Sales 2018-2021 OK'])
        #Set concat dataframe
        df_concat = pd.DataFrame()
        #for each brand
        for brand, group in df_ino.groupby(['Brand']):
            #init df
            df_merge = pd.DataFrame(index=group.columns.values[:-1])
            group = group.drop('Brand', axis=1)
            #Find date of first sale for each product
            for col in group.T.columns:
                first_sale = group.T[col][pd.notna(group.T[col])].index.values[0]
                if first_sale == date_begining:
                    pass
                else:
                    delta = dt.timedelta(weeks=104)
                    date_end = (dt.datetime.strptime(first_sale, '%Y-%m-%d') + delta).strftime('%Y-%m-%d')                    
                    df_merge = pd.concat([group.T[[col]].loc[first_sale:date_end], df_merge],axis=1)
                    
            df_innovation = pd.DataFrame(df_merge.reset_index().sort_values(by='index').set_index('index').sum(axis=1)).rename(columns={0:'Rate of Innovation'})
            df_innovation.loc['2015-01-01':'2020-01-01'] = 0.0
            df_innovation = df_innovation.div(group.T.sum(axis=1), axis=0)
            df_innovation['Brand'] = brand
            df_innovation = df_innovation.reset_index().rename(columns={'index':'Date'})
            df_innovation = df_innovation[df_innovation['Date']!='Brand']
            df_concat = pd.concat([df_concat, df_innovation])
        
        return df_concat[['Brand', 'Date', 'Rate of Innovation']]

    def compute_Promo_Cost(self, path):
        df_promo = pd.read_excel(path, engine="openpyxl")
        renaming_brands = {'BOURSIN':'BOURSIN', 
            'Laughing Cow': 'THE LAUGHING COW', 
            'Mini Babybel': 'BABYBEL', 
            'Prices': 'PRICES',
            'Nurishh': 'NURISHH',
            'Kaukauna': 'KAUKAUNA',
            'Merkts': 'MERKTS'}
        features = ['PH3', 'Year', 'Month', 'Week', '09.Promo_OOI_USA', '06.OI_Promo_USA']
        df_promo_bel = df_promo[df_promo.PH3.isin(renaming_brands.keys())][features]
        df_promo_refri = df_promo[df_promo.PH3 == 'Refrigerated Spreads']
        #swap PH3 and PH4 columns (KAUKAUNA and MERKTS)
        df_promo_refri = df_promo_refri[df_promo_refri.PH4 != 'Owls Nest'].rename(columns={'PH3':'PH4', 'PH4':'PH3'})[features]
        #concat to have all brands on same columns
        df_promo_res = pd.concat([df_promo_bel, df_promo_refri])
        #rename brands
        df_promo_res['PH3'] = df_promo_res['PH3'].map(renaming_brands)
        #sum promo columns
        df_promo_res['Cost'] = df_promo_res[['09.Promo_OOI_USA', '06.OI_Promo_USA']].sum(axis=1)
        #We want 52 weeks, so we merge week 53 with week 52
        df_promo_res['Week'] = df_promo_res['Week'].replace({53: 52})
        #group by ...
        df_promo_res = df_promo_res.groupby(['PH3', 'Year', 'Week']).agg(np.sum).reset_index()
        #NURISHH is a special case
        df_promo_nurishh = df_promo_res[df_promo_res.PH3 == 'NURISHH']
        df_promo_res = df_promo_res[df_promo_res.PH3 != 'NURISHH']
        df_promo_res['Date'] = list(pd.date_range(start="2019-01-01", end="2021-12-31", freq='W-SUN').strftime('%Y-%m-%d')) * df_promo_res.PH3.nunique()
        df_promo_nurishh['Date'] = df_promo_res['Date'][-df_promo_nurishh.shape[0]:].values
        df_promo_res = pd.concat([df_promo_res, df_promo_nurishh])
        df_promo_res = df_promo_res.rename(columns={'PH3': 'Brand', 'Cost':'Promo Cost'})[['Brand', 'Promo Cost', 'Date']]

        return df_promo_res

    def compute_HH_Index(self, path):
        #cut excel file to remove half the columns, keep HH index
        df_h = pd.read_excel(path, engine="openpyxl", header=8).iloc[:, 1:210]
        #remove ALL BRANDS
        df_hh = df_h[df_h["Unnamed: 1"] != "Total All Products"].fillna(0)
        #group by brand
        df_hh = df_hh.groupby("Unnamed: 1").agg(np.mean)
        #create dates
        df_hh.columns = pd.date_range(start="2018-01-07", end="2021-12-31", freq='W-SUN').strftime('%Y-%m-%d')
        df_hh = pd.DataFrame(df_hh.stack()).reset_index().rename(columns={'Unnamed: 1': 'Brand', 'level_1': 'Date', 0: 'HH'})
        #rename brand
        df_hh['Brand'] = df_hh['Brand'].apply(lambda x: x.strip())
        return df_hh
        
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

    def open_excel_old(self):

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