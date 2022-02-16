from model.Model import Model
import pandas as pd

class M_USA(Model):
    """extended class Model for Belgique"""
    
    def compute_market_passeport(self, df_full):
        brands_to_remove = ["ALL BRANDS"]#, "ZZ OTHERS", "SS.MARQUE", "UNBRANDED", "PL"
        
        indicators = ['Category', 'Size', 'Count of Competitors', 'L1 Brand', 'L1 Sales', 'L1 Share', 
                    'L2 Brand', 'L2 Sales', 'L2 Share', 'L3 Brand', 'L3 Sales', 'L3 Share', 'Growth', 
                    'Count of brands', 'Bel brand sales', 'Bel brands market share', 
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
                    #dict_cat['Promotion intensity'] = group['Sales volume with Promo'].sum() / group['Sales in volume'].sum()
                    dict_cat['Bel brand sales'] = group[group['Brand'].isin(self.bel_brands)]['Sales in volume'].sum()
                    dict_cat['Bel brands market share'] = group[group['Brand'].isin(self.bel_brands)]['Sales in volume'].sum() / group['Sales in volume'].sum() * 100
                    dict_cat['Average Price'] = group['Price per volume'].agg('average')

                    df_cat = pd.DataFrame(data=dict_cat, index=[0])
                    df_concat = pd.concat([df_concat, df_cat]) 
                y_dfs.append(df_concat.set_index('Category').T)
                keys.append(year)
        out = pd.concat(y_dfs, axis=0, keys=keys, names=['Years', 'Drivers'])
        return out  

    def compute_growth(self, df, year, category):
        if year == '2018':
            return 'NA'
        if year == '2019':
            return self.cagr_in_tons(df, brand='all', category=category, date_min={'Min':'2018-01-01', 'Max':'2019-01-01'}, date_max={'Min':'2019-01-01', 'Max':'2020-01-01'})
        elif year=='2020':
            return self.cagr_in_tons(df, brand='all', category=category, date_min={'Min':'2019-01-01', 'Max':'2020-01-01'}, date_max={'Min':'2020-01-01', 'Max':'2021-01-01'})
        elif year=='2021':
            return self.cagr_in_tons(df, brand='all', category=category, date_min={'Min':'2020-01-01', 'Max':'2021-01-01'}, date_max={'Min':'2021-01-01', 'Max':'2022-01-01'})
        else:
            return 0