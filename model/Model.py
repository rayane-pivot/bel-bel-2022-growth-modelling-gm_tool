from tokenize import Number
import pandas as pd
import numpy as np
import math


class Model():

    def brand_pos(self, df, year_min:int):
        df = df.copy()
        df.Date = pd.to_datetime(df.Date)
        df = df[df.Date.dt.year>=year_min]
        brand_positioning_matrix = pd.pivot_table(df, values='Sales in volume', columns='Category', index='Brand', aggfunc='sum')

        brand_positioning_matrix.columns.name=None
        brand_positioning_matrix.index.name='Brand'
        return brand_positioning_matrix.div(1000)

    def growth(self, df, on:str, year1:int, year2:int):
        def cagr(x):
            y1 = x[x.Date.dt.year == year1]['Sales in volume'].values[0] if len(x[x.Date.dt.year == year1]['Sales in volume'].values)>0 else None
            y2 = x[x.Date.dt.year == year2]['Sales in volume'].values[0] if len(x[x.Date.dt.year == year2]['Sales in volume'].values)>0 else None
            if (y1 is None) or (y2 is None):
                return None
            return (math.pow((y2 / y1), (1/(year2-year1+1))) -1)*100
        def apply_growth(x):
            y1 = x[x.Date.dt.year == year1]['Sales in volume'].values[0] if len(x[x.Date.dt.year == year1]['Sales in volume'].values)>0 else 0.0
            y2 = x[x.Date.dt.year == year2]['Sales in volume'].values[0] if len(x[x.Date.dt.year == year2]['Sales in volume'].values)>0 else 0.0
            return  y2 - y1

        df = df.copy()
        df.Date = pd.to_datetime(df.Date)
        
        df_grp = df.groupby([on, pd.Grouper(key="Date", freq='Y')])['Sales in volume'].agg('sum').reset_index()
        growth = df_grp.groupby(on).apply(apply_growth).div(1000).reset_index().rename(columns={0:'GROWTH'})

        growth['CAGR'] = df_grp.groupby(on).apply(cagr).reset_index()[0]

        return growth.set_index(on)

    def compute_brand_positioning_matrix(self, df, year_min:int, year1:int, year2:int):
        #Compute brand positioning matrix, cagr and growth
        brand_positioning_matrix = self.brand_pos(df, year_min=year_min)
        growth_brand = self.growth(df, on='Brand', year1=year1, year2=year2)
        growth_category = self.growth(df, on='Category', year1=year1, year2=year2)

        #Concat brand positioning matrix, cagr and growth
        brand_positioning_matrix = pd.concat([brand_positioning_matrix, growth_category.T])
        brand_positioning_matrix[growth_brand.columns] = growth_brand[growth_brand.columns]
        return brand_positioning_matrix


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
                    dict_cat['Size'] = group['Sales in volume'].sum()/1000
                    dict_cat['Count of Competitors'] = group[~group['Brand'].isin(self.bel_brands)]['Brand'].nunique()
                    
                    leaders = group.groupby('Brand')['Sales in volume'].sum().sort_values(ascending=False)[:3]
                    list_of_leaders = [{'BRAND':i, 'SALES':j, 'SHARE':0} for i, j in leaders.items()]
                    for leader in list_of_leaders:
                        leader['SHARE'] = leader['SALES']/group['Sales in volume'].sum()*100
                    #list_of_leaders = [str(l) for l in list_of_leaders]
                    dict_cat['L1 Brand'] = list_of_leaders[0]['BRAND']
                    dict_cat['L2 Brand'] = list_of_leaders[1]['BRAND']
                    dict_cat['L3 Brand'] = list_of_leaders[2]['BRAND']
                    
                    dict_cat['L1 Sales'] = list_of_leaders[0]['SALES']/1000
                    dict_cat['L2 Sales'] = list_of_leaders[1]['SALES']/1000
                    dict_cat['L3 Sales'] = list_of_leaders[2]['SALES']/1000
                    
                    dict_cat['L1 Share'] = list_of_leaders[0]['SHARE']
                    dict_cat['L2 Share'] = list_of_leaders[1]['SHARE']
                    dict_cat['L3 Share'] = list_of_leaders[2]['SHARE']
                    
                    #dict_cat['Leaders'] = ''.join(list_of_leaders)
                    #dict_cat['Leaders'] = np.array2string(group.groupby('Brand')['Sales in value'].sum().sort_values(ascending=False)[:3].index.array)
                    dict_cat['Growth'] = self.compute_growth(df_full, year, category)
                    dict_cat['Growth'] = dict_cat['Growth'] if isinstance(dict_cat['Growth'], str) else dict_cat['Growth']/1000
                    dict_cat['Count of brands'] = group['Brand'].nunique()
                    dict_cat['Promotion intensity'] = group['Sales volume with promo'].sum() / group['Sales in volume'].sum()
                    dict_cat['Bel brand sales'] = group[group['Brand'].isin(self.bel_brands)]['Sales in volume'].sum()/1000
                    dict_cat['Bel brands market share'] = group[group['Brand'].isin(self.bel_brands)]['Sales in volume'].sum() / group['Sales in volume'].sum() * 100
                    dict_cat['Average Price'] = group['Price per volume'].agg('mean')

                    df_cat = pd.DataFrame(data=dict_cat, index=[0])
                    df_concat = pd.concat([df_concat, df_cat]) 
                y_dfs.append(df_concat.set_index('Category').T)
                keys.append(year)
        out = pd.concat(y_dfs, axis=0, keys=keys, names=['Years', 'Drivers'])
        return out  
    
    def compute_brand_scorecard(self, df, df_bel, json_sell_out_params, country):
        # Columns : 
        # Brand Sales 2018	Brand Sales 2019	Brand Sales 2020	Brand Sales 2021	
        # Brand Market Share 2018	Brand Market Share 2019	Brand Market Share 2020	Brand Market Share 2021	
        # Average Price (moyenne 2018-2021)	
        # Brand Awareness	
        # Brand A&P 2018	Brand A&P 2019	Brand A&P 2020	Brand A&P 2021	
        # Volume sold on Promo (avg 2018-21)
        date_min=json_sell_out_params.get(country).get("brand scorecard").get("date_min")
        df_sales = self.compute_sales(df_bel, date_min=date_min)
        df_sales = self.compute_share(df, df_sales, date_min=date_min)
        df_price = self.compute_price(df_bel, date_min=date_min)
        df_sales['Average Price (moyenne 2018-2021)'] = df_price
        df_sales['Brand Awareness'] = self.compute_awareness(json_sell_out_params=json_sell_out_params, country=country)
        df_AP = self.compute_AP(df_bel, date_min=date_min)
        df_sales[df_AP.columns] = df_AP
        df_volume_promo = self.compute_volume_promo(df, json_sell_out_params.get(country).get('bel_brands'), date_min=date_min)
        df_sales['Volume sold on Promo (avg 2018-21)']=df_volume_promo
        return df_sales
    
    def compute_awareness(self, json_sell_out_params, country):
        aw = json_sell_out_params.get(country).get("brand scorecard").get("awareness")
        return pd.DataFrame.from_dict(aw, orient='index', columns=['Brand Awareness'])

    def compute_sales(self, df, date_min:str):
        df_temp = df.copy()
        df_temp.Date = pd.to_datetime(df_temp.Date)
        df_grp = df_temp.groupby(['Brand', pd.Grouper(key="Date", freq='Y')])['Sales in volume'].agg('sum').reset_index()
        df_grp = df_grp[df_grp.Date > date_min]
        table_sales = pd.pivot_table(df_grp, values='Sales in volume', index='Brand', columns='Date')

        table_sales.index.name=None
        table_sales.columns.name = 'Brand'
        table_sales = table_sales.rename(columns={x : f'Brand Sales {x.year}' for x in table_sales.columns})
        table_sales = table_sales.div(1000)
        return table_sales

    def compute_share(self, df, df_sales, date_min:str):
        df_temp = df.copy()
        df_temp.Date = pd.to_datetime(df_temp.Date)
        df_grp = df_temp.groupby(pd.Grouper(key="Date", freq='Y'))['Sales in volume'].agg('sum').reset_index()
        df_grp = df_grp[df_grp.Date > date_min]
        sales_table = pd.pivot_table(df_grp, columns="Date", values='Sales in volume')
        sales_table = sales_table.div(1000)
        df_sales[[f'Brand Market Share {x.year}' for x in sales_table.columns]] = df_sales[[f'Brand Sales {x.year}' for x in sales_table.columns]] / sales_table.iloc[0].values * 100
        return df_sales

    def compute_price(self, df_bel, date_min:str):
        df_temp = df_bel.copy()
        df_temp.Date = pd.to_datetime(df_temp.Date)
        df_grp = df_temp.groupby(['Brand', pd.Grouper(key="Date", freq='Y')])['Price per volume'].agg('mean').reset_index()
        df_grp = df_grp[df_grp.Date > date_min]
        return df_grp.groupby('Brand')['Price per volume'].agg('mean')

    def compute_AP(self, df_bel, date_min:str):
        df_temp = df_bel.copy()
        df_temp.Date = pd.to_datetime(df_temp.Date)
        df_grp = df_temp.groupby(['Brand', pd.Grouper(key="Date", freq='Y')])['A&P'].agg('sum').reset_index()
        df_grp = df_grp[df_grp.Date > date_min]
        AP_table = pd.pivot_table(df_grp, values='A&P', columns='Date', index='Brand')

        AP_table.index.name=None
        AP_table.columns.name = 'Brand'
        AP_table = AP_table.rename(columns={x : f'Brand A&P {x.year}' for x in AP_table.columns})
        return AP_table
    
    def compute_volume_promo(self, df, bel_brands, date_min:str):
        def promo_share(x):
            return x['Sales volume with promo'].mean()/x['Sales in volume'].mean()
        df_temp = df.copy()
        df_temp.Date = pd.to_datetime(df_temp.Date)
        df_grp = df_temp[df_temp.Brand.isin(bel_brands)].groupby(['Brand', pd.Grouper(key="Date", freq='Y')])[['Sales volume with promo', 'Sales in volume']].agg('sum', 'sum').reset_index()
        df_grp = df_grp[df_grp.Date > date_min]
        return df_grp.groupby('Brand').apply(promo_share)