import pandas as pd
import numpy as np
import math


class Model():

    def __init__(self, bel_brands, dict_dates):
        self.bel_brands=bel_brands
        self.dict_dates=dict_dates

    def compute_brand_positioning_matrix(self, df, Y_BEG, Y_END):
        brands_to_remove = ["ALL BRANDS"]#, "ZZ OTHERS", "SS.MARQUE", "UNBRANDED", "PL"

        df_2Y = df[(df['Date'] >= Y_BEG['Min']) & (df['Date'] < Y_END['Max'])][~df['Brand'].isin(brands_to_remove)]
        
        #build brand positioning matrix
        brand_pos_matrix = pd.pivot_table(df_2Y, values='Sales in volume', index=['Brand'], columns=['Category'], aggfunc=np.sum, fill_value=0)

        #brand_pos_matrix = brand_pos_matrix.apply(lambda x: 100*x/x.sum(), axis=1)
        brand_pos_matrix = brand_pos_matrix.round(2)
        
        #compute cagr by category and brand, Sales in value and Sales in volume
        cagr_brand_value, cagr_category_value, cagr_brand_volume, cagr_category_volume = self.compute_cagr_arrays(df_2Y, Y_BEG=Y_BEG, Y_END=Y_END, NbYears=2)
        df_cagr_brand_value = pd.DataFrame(data=cagr_brand_value, columns=['Brand', 'CAGR_value']).set_index('Brand').astype(float).round(2)
        df_cagr_category_value = pd.DataFrame(data=cagr_category_value, columns=['Category', 'CAGR_value']).set_index('Category').astype(float).round(2)
        df_cagr_brand_volume = pd.DataFrame(data=cagr_brand_volume, columns=['Brand', 'Past growth volume']).set_index('Brand').astype(float).round(2)
        df_cagr_category_volume = pd.DataFrame(data=cagr_category_volume, columns=['Category', 'Past growth volume']).set_index('Category').astype(float).round(2)
        
        #append cagr to the matrix
        brand_pos_matrix['CAGR_value']=df_cagr_brand_value
        brand_pos_matrix['Past growth volume']=df_cagr_brand_volume
        brand_pos_matrix = pd.concat([brand_pos_matrix, df_cagr_category_value.T, df_cagr_category_volume.T])
        #brand_pos_matrix = brand_pos_matrix[brand_pos_matrix['Brand'].isin(self.bel_brands)]
        return brand_pos_matrix
    

    def compute_market_passeport(self, df_full):
        brands_to_remove = ["ALL BRANDS"]#, "ZZ OTHERS", "SS.MARQUE", "UNBRANDED", "PL"
        
        indicators = ['Category', 'Size', 'Count of Competitors', 'L1 Brand', 'L1 Sales', 'L1 Share', 
                    'L2 Brand', 'L2 Sales', 'L2 Share', 'L3 Brand', 'L3 Sales', 'L3 Share', 'Growth', 
                    'Count of brands', 'Promotion intensity', 'Bel brand sales', 'Bel brands market share', 
                    'Average Price']
        
        df_full = df_full[~df_full['Brand'].isin(brands_to_remove)]
        
        y_dfs = []
        keys = []
        for year, interval in self.dict_dates.items():
            if year != 'HISTORY':
                df_concat = pd.DataFrame(columns=indicators)
                df = self.filter_data(df_full, category="all", brand="all", date_min=interval['Min'], date_max=interval['Max'])
                for category, group in df.groupby('Category'):
                    dict_cat = dict()
                    dict_cat['Category'] = category
                    dict_cat['Size'] = group['Sales in volume'].sum()
                    dict_cat['Count of Competitors'] = group[~group['Brand'].isin(self.bel_brands)]['Brand'].nunique()
                    
                    leaders = group.groupby('Brand')['Sales in volume'].sum().sort_values(ascending=False)[:3]
                    list_of_leaders = [{'BRAND':i, 'SALES':j, 'SHARE':0} for i, j in leaders.items()]
                    for leader in list_of_leaders:
                        leader['SHARE'] = leader['SALES']/group['Sales in volume'].sum()*100
                    #list_of_leaders = [str(l) for l in list_of_leaders]
                    dict_cat['L1 Brand'] = list_of_leaders[0]['BRAND']
                    dict_cat['L2 Brand'] = list_of_leaders[1]['BRAND']
                    dict_cat['L3 Brand'] = list_of_leaders[2]['BRAND']
                    
                    dict_cat['L1 Sales'] = list_of_leaders[0]['SALES']
                    dict_cat['L2 Sales'] = list_of_leaders[1]['SALES']
                    dict_cat['L3 Sales'] = list_of_leaders[2]['SALES']
                    
                    dict_cat['L1 Share'] = list_of_leaders[0]['SHARE']
                    dict_cat['L2 Share'] = list_of_leaders[1]['SHARE']
                    dict_cat['L3 Share'] = list_of_leaders[2]['SHARE']
                    
                    #dict_cat['Leaders'] = ''.join(list_of_leaders)
                    #dict_cat['Leaders'] = np.array2string(group.groupby('Brand')['Sales in value'].sum().sort_values(ascending=False)[:3].index.array)
                    dict_cat['Growth'] = self.compute_growth(df_full, year, category)
                    dict_cat['Count of brands'] = group['Brand'].nunique()
                    dict_cat['Promotion intensity'] = group['Sales volume with Promo'].sum() / group['Sales in volume'].sum()
                    dict_cat['Bel brand sales'] = group[group['Brand'].isin(self.bel_brands)]['Sales in volume'].sum()
                    dict_cat['Bel brands market share'] = group[group['Brand'].isin(self.bel_brands)]['Sales in volume'].sum() / group['Sales in volume'].sum() * 100
                    dict_cat['Average Price'] = group['Price per Volume'].agg('average')

                    df_cat = pd.DataFrame(data=dict_cat, index=[0])
                    df_concat = pd.concat([df_concat, df_cat]) 
                y_dfs.append(df_concat.set_index('Category').T)
                keys.append(year)
        out = pd.concat(y_dfs, axis=0, keys=keys, names=['Years', 'Drivers'])
        return out  
    
    def filter_data(self, df, category="all", subcategory="all", brand="all", 
             date_min="all", date_max="all"):
        if category != "all":
            df = df[df["Category"] == category]
        if subcategory != "all":
            df = df[df["Sub Category"] == subcategory]
        if brand != "all":
            df = df[df["Brand"] == brand]
        if date_min != "all":
            df = df[df["Date"] >= date_min]
        if date_max != "all":
            df = df[df["Date"] < date_max]
        return df

    def cagr(self, df, category, brand, date_min, date_max, NbYears):
        #CAGR = END/BEG ^(1/NbYears) -1 * 100

        BEG = self.filter_data(df, category=category, brand=brand, date_min=date_min['Min'], date_max=date_min['Max'])
        BEG = BEG['Sales in value'].sum()
        
        END = self.filter_data(df, category=category, brand=brand, date_min=date_max['Min'], date_max=date_max['Max'])
        END = END['Sales in value'].sum()
        
        #NbYears=3
        if BEG != 0:
            CAGR =  (math.pow((END/BEG), (1/NbYears)) -1) * 100
        else:
            CAGR = 0.0

        return CAGR

    def cagr_in_tons(self, df, category, brand, date_min, date_max):
        #CAGR = END - BEG
        BEG = self.filter_data(df, category=category, brand=brand, date_min=date_min['Min'], date_max=date_min['Max'])
        #BEG = filter_data(df, [category], None, dict_dates[Y_BEG]['Min'], dict_dates[Y_BEG]['Max'])
        BEG = BEG['Sales in volume'].sum()
        
        END = self.filter_data(df, category=category, brand=brand, date_min=date_max['Min'], date_max=date_max['Max'])
        END = END['Sales in volume'].sum()
        
        return END - BEG

    def compute_cagr_arrays(self, df, Y_BEG, Y_END, NbYears):
        cagr_brand_value = np.empty((0,2))
        cagr_category_value = np.empty((0,2))
        
        cagr_brand_volume = np.empty((0,2))
        cagr_category_volume = np.empty((0,2))
        
        for brand in df['Brand'].unique():
            cagr_brand_value = np.append(
                cagr_brand_value, 
                [[brand, self.cagr(df, brand=brand, category='all', date_min=Y_BEG, date_max=Y_END, NbYears=NbYears)]],
                axis=0)
            cagr_brand_volume = np.append(cagr_brand_volume, 
                [[brand, self.cagr_in_tons(df, brand=brand, category='all', date_min=Y_BEG, date_max=Y_END)]],
                axis=0)
        for category in df['Category'].unique():
            cagr_category_value = np.append(cagr_category_value, 
                [[category, self.cagr(df, brand='all', category=category, date_min=Y_BEG, date_max=Y_END, NbYears=NbYears)]], 
                axis=0)
            cagr_category_volume = np.append(cagr_category_volume, 
                [[category, self.cagr_in_tons(df, brand='all', category=category, date_min=Y_BEG, date_max=Y_END)]], 
                axis=0)
            
        return cagr_brand_value, cagr_category_value, cagr_brand_volume, cagr_category_volume

     
    def compute_growth(self, df, year, category):
        if year == '2019':
            return 'NA'
            #return cagr_in_tons(df, brand='all', category=category, date_min={'Min':'2018-01-01', 'Max':'2019-01-01'}, date_max={'Min':'2019-01-01', 'Max':'2020-01-01'})
        elif year=='2020':
            return self.cagr_in_tons(df, brand='all', category=category, date_min={'Min':'2019-01-01', 'Max':'2020-01-01'}, date_max={'Min':'2020-01-01', 'Max':'2021-01-01'})
        elif year=='2021':
            return 'NA'
            #return cagr_in_tons(df, brand='all', category=category, date_min={'Min':'2020-01-01', 'Max':'2021-01-01'}, date_max={'Min':'2021-01-01', 'Max':'2022-01-01'})
        else:
            return 0
   

    

    