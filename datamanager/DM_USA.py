import pandas as pd
import numpy as np
import datetime as dt

from datamanager.DataManager import DataManager
#from DataManager import DataManager

class DM_USA(DataManager):
    """ DataManager for US data"""
    _country = 'USA'

    def ad_hoc_USA(self, json_sell_out_params):
        df = super().fill_df(json_sell_out_params, self._country)
        
        #DATES FORMATIING
        df = df[~df['Date'].str.contains('OK')]
        df['Date'] = df['Date'].apply(lambda x:dt.datetime.strptime(x.split()[-1], '%m-%d-%y').strftime('%Y-%m-%d'))
        
        #AD HOC PLANT BASED
        df.loc[df[df['Market'].isin(['PLANT BASED CHEESE', 'PLANT BASED CREAM CHEESE'])].index, 'Category'] = 'PLANT BASED'
        # df.loc[df[df['Market'].isin(['PLANT BASED CHEESE', 'PLANT BASED CREAM CHEESE'])].index, 'Sub Category'] = 'PLANT BASED'
        
        #AD HOC CREAM CHEESE
        df.loc[df[df['Sub Category'].isin(['FLAVORED TUBS',
                                        'FLAVORED WHIPPED TUBS', 
                                        'PLAIN TUBS', 
                                        'PLAIN WHIPPED TUBS'])].index, 'Category'] = 'CREAM CHEESE TUBS'
        
        df.loc[df[df['Sub Category'].isin(['SINGLE SERVE FLAVORED CREAM CHEESE',
                                        'SINGLE SERVE PLAIN CREAM CHEESE'
                                        ])].index, 'Category'] = 'SINGLE SERVE'
        
        df.loc[df[df['Market'].isin(['CREAM CHEESE'])].index, 'Category'] = 'CREAM CHEESE'
        
        #AD HOC GOURMET
        df.loc[df[(df['Sub Category'].isin(['GOURMET BLOCK / WEDGE / ROUND'])) & (~df['Category'].isin(['PLANT BASED']))].index, 'Category'] = 'GOURMET BLOCK / WEDGE / ROUND'
        df.loc[df[(df['Sub Category'].isin(['GOURMET CRUMBLED'])) & (~df['Category'].isin(['PLANT BASED']))].index, 'Category'] = 'GOURMET CRUMBLED'
        df.loc[df[(df['Sub Category'].isin(['GOURMET FRESH ITALIAN'])) & (~df['Category'].isin(['PLANT BASED']))].index, 'Category'] = 'GOURMET FRESH ITALIAN'
        df.loc[df[(df['Sub Category'].isin(['GOURMET SHREDDED / GRATED'])) & (~df['Category'].isin(['PLANT BASED']))].index, 'Category'] = 'GOURMET SHREDDED / GRATED'
        df.loc[df[(df['Sub Category'].isin(['GOURMET SPREADS'])) & (~df['Category'].isin(['PLANT BASED']))].index, 'Category'] = 'GOURMET SPREADS'
        
        #AD HOC DROP HYBRID FROM MARKET
        df = df[~df['Market'].isin(['HYBRID CHEESE'])]
        
        #DROP USELESS COLUMNS
        df = df.drop(columns=['Product', 'Channel'])

        #ADD PERIODS
        df_periods = pd.DataFrame(np.sort(df.Date.unique())).reset_index().rename(columns={'index':'Period', 0:'Date'})
        df_periods.Period = df_periods.Period + 1
        df = pd.merge(df, df_periods, on='Date', how='left')
        
        #Remove PLANT BASED DUPLICATES
        # for brand in df[df.Category=='PLANT BASED'].Brand.unique():
        #     df = df[~((df.Brand==brand) & (df.Category != "PLANT BASED"))]
        
        #DROP DUPLICATES    
        df = df[~df.round(3).duplicated(subset=['Date', 'Brand',
            'Sales in value', 'Sales value with promo', 'Sales in volume',
            'Sales volume with promo', 'Price without promo',
            'Price with promo', 'Price per volume', 'Distribution'], keep='last')]
        
        df = df.reset_index(drop=True)
        self._df = df
    
    def fill_df_bel(self, json_sell_out_params):
        assert not self._df.empty, 'df is empty, call ad_hoc_USA() or load() first'
        df = self._df.copy()
        df.Date = pd.to_datetime(df.Date)
        df.Date = df.Date.dt.strftime('%Y-%m-%d')
        bel_brands = json_sell_out_params.get(self._country).get('bel_brands')
        df_bel=df[df.Brand.isin(bel_brands)].groupby(['Date', 'Brand'], as_index=False)[['Price per volume', 'Sales in volume', 'Sales in value', 'Distribution']].agg({'Price per volume':'mean', 'Sales in volume':'sum', 'Sales in value':'sum', 'Distribution':'mean'})

        PATH_FINANCE = json_sell_out_params.get(self._country).get('dict_path').get('PATH_FINANCE').get('Total Country')
        AP_CODES = json_sell_out_params.get(self._country).get('A&P_codes')
        FINANCE_COLS = json_sell_out_params.get(self._country).get('A&P_columns')
        FINANCE_RENAMING_COLS = json_sell_out_params.get(self._country).get('finance_renaming_columns')
        DATE_MIN = json_sell_out_params.get(self._country).get('dates_finance').get('Min')
        DATE_MAX = json_sell_out_params.get(self._country).get('dates_finance').get('Max')
        FINANCE_HEADER = json_sell_out_params.get(self._country).get('Finance').get('header')
        df_finance = self.fill_Finance(path=PATH_FINANCE, finance_cols=FINANCE_COLS, finance_renaming_columns=FINANCE_RENAMING_COLS, header=FINANCE_HEADER)
        df_finance = self.compute_Finance(df_finance, AP_CODES, DATE_MIN, DATE_MAX)
        df_bel = pd.merge(df_bel, df_finance, on=['Brand', 'Date'], how='left')

        PATH_INNO = json_sell_out_params.get(self._country).get('dict_path').get('PATH_INNO').get('Total Country')
        DATE_BEG = json_sell_out_params.get(self._country).get('Inno').get('date_beg')
        INNO_HEADER = json_sell_out_params.get(self._country).get('Inno').get('header')
        INNOVATION_DURATION = json_sell_out_params.get(self._country).get('Inno').get('innovation_duration')
        df_inno = self.fill_Inno(path=PATH_INNO, header=INNO_HEADER)
        df_inno = self.compute_Inno(df=df_inno, date_begining=DATE_BEG, innovation_duration=INNOVATION_DURATION)
        df_bel = pd.merge(df_bel, df_inno, on=['Brand', 'Date'], how='left')

        PATH_PROMOCOST = json_sell_out_params.get(self._country).get('dict_path').get('PATH_PROMO_COST').get('Total Country')
        RENAMING_BRANDS = json_sell_out_params.get(self._country).get('Promo Cost').get('renaming_brands')
        FEATURES = json_sell_out_params.get(self._country).get('Promo Cost').get('features')
        df_promocost = self.compute_Promo_Cost(path=PATH_PROMOCOST, renaming_brands=RENAMING_BRANDS, features=FEATURES)
        df_bel = pd.merge(df_bel, df_promocost, on=['Brand', 'Date'], how='left')

        PATH_HH = json_sell_out_params.get(self._country).get('dict_path').get('PATH_HH_INDEX').get('Total Country')
        HH_HEADER = json_sell_out_params.get(self._country).get('HH Index').get('header')
        df_hh = self.compute_HH_Index(path=PATH_HH, header=HH_HEADER)
        df_bel = pd.merge(df_bel, df_hh, on=['Brand', 'Date'], how='left')
        
        self._df_bel = df_bel

    def compute_Finance(self, df_finance, aandp_codes, date_min, date_max):
        #Compute from Finance dataframe
        df_finance = df_finance[df_finance['Country'] == self._country]
        #Filter brands for the study
        df_finance = df_finance[df_finance['Brand'].isin(aandp_codes)]
        #ABS for Advertising and Promotion
        df_finance['Advertising'] = df_finance['Advertising'].abs()
        df_finance['Promotion'] = df_finance['Promotion'].abs()
        #Get Brand from code
        df_finance['Brand'] = df_finance['Brand'].apply(lambda x:x.split(sep='-')[-1].strip())
        #### ADHOC FOR PRICES and BABYBEL
        df_finance['Brand'] = df_finance['Brand'].apply(lambda x: 'PRICES' if x=="PRICE'S" else x)
        df_finance['Brand'] = df_finance['Brand'].apply(lambda x: 'BABYBEL' if x=="MINI BABYBEL" else x)
        df_finance = df_finance.fillna(0.0)
        #Months to week
        df_finance['number of weeks'] = df_finance.apply(lambda x:self.count_num_sundays_in_month(x.Year, x.Month), axis=1)
        df_finance['A&P'] = df_finance.apply(lambda x: (x.Advertising + x.Promotion) / x['number of weeks'] * 1000, axis=1)
        df_finance['Sell-in'] = df_finance.apply(lambda x: x['Sell-in'] / x['number of weeks'] * 1000, axis=1)
        df_finance['MVC'] = df_finance.apply(lambda x: x['MVC'] / x['number of weeks'] * 1000, axis=1)
        #Duplicate for n weeks
        full_idx = pd.date_range(start=date_min, end=date_max, freq='W')
        df_test = pd.DataFrame(index=full_idx)
        df_test['Year'] = df_test.index.year
        df_test['Month'] = df_test.index.month
        df_concat = pd.DataFrame()
        for brand in df_finance.Brand.unique():
            df_concat = pd.concat([df_concat, pd.merge(df_finance[df_finance.Brand==brand], df_test.reset_index(), on=['Year', 'Month']).rename(columns={'index':'Date'})])
        #Change date type to str
        df_concat['Date'] = df_concat['Date'].apply(lambda x : dt.datetime.strftime(x, "%Y-%m-%d"))
        return df_concat[['Brand', 'Date', 'Sell-in', 'Advertising', 'Promotion', 'A&P', 'MVC']]


    def compute_Inno(self, df, date_begining:str, innovation_duration:int):
        #Compute from innovation dataframe
        df_concat = pd.DataFrame()
        delta = dt.timedelta(weeks=innovation_duration)
        #for each brand
        for brand, group in df.groupby(['Brand']):
            #init df
            df_merge = pd.DataFrame(index=group.columns.values[:-1])
            group = group.drop('Brand', axis=1)
            #Find date of first sale for each product
            for col in group.T.columns:
                first_sale = group.T[col][pd.notna(group.T[col])].index.values[0]
                if first_sale == date_begining:
                    pass
                else:
                    #get data for 2 years window of innovation
                    date_end = (dt.datetime.strptime(first_sale, '%Y-%m-%d') + delta).strftime('%Y-%m-%d')                    
                    df_merge = pd.concat([group.T[[col]].loc[first_sale:date_end], df_merge],axis=1)
            #beautiful peace of code here, ask ahmed for details        
            df_innovation = pd.DataFrame(df_merge.reset_index().sort_values(by='index').set_index('index').sum(axis=1)).rename(columns={0:'Rate of Innovation'})
            date_begining_to_delta = (dt.datetime.strptime(first_sale, '%Y-%m-%d') + delta).strftime('%Y-%m-%d')                    
            df_innovation.loc[first_sale:date_begining_to_delta] = 0.0
            #divide innovations by total sales
            df_innovation = df_innovation.div(group.T.sum(axis=1), axis=0)
            df_innovation.loc[:, 'Brand'] = brand
            df_innovation = df_innovation.reset_index().rename(columns={'index':'Date'})
            df_innovation = df_innovation[df_innovation['Date']!='Brand']
            df_concat = pd.concat([df_concat, df_innovation])
        
        return df_concat[['Brand', 'Date', 'Rate of Innovation']]

    
    def compute_Promo_Cost(self, path:str, renaming_brands:dict, features:list):
        #for any question, ask ahmed@pivotandco.com
        df_promo = pd.read_excel(path, engine="openpyxl")
        renaming_brands = renaming_brands
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

    def compute_HH_Index(self, path:str, header:list):
        #cut excel file to ndarremove half the columns, keep HH index
        df_h = pd.read_excel(path, engine="openpyxl", header=header).iloc[:, 1:210]
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
            

    def fill_df_bel_old(self, json_params):
        assert self._df is not None, 'df is empty, call ad_hoc_USA() or load() first'
        df = self.get_df()
        df_bel=df[df.Brand.isin(json_params.get(self._country).get('bel_brands'))].groupby(['Date', 'Brand'], as_index=False)['Price per volume', 'Sales in volume', 'Sales in value', 'Distribution'].agg({'Price per volume':'mean', 'Sales in volume':'sum', 'Sales in value':'sum', 'Distribution':'mean'})
        
        df_AP = self.compute_Finance(json_params, self._country)
        df_bel = pd.merge(df_bel, df_AP, on=['Brand', 'Date'], how='left')
    
        df_inno = self.compute_Inno(json_params.get(self._country).get('dict_path').get('PATH_INNO').get('Total Country'))
        df_bel = pd.merge(df_bel, df_inno, on=['Brand', 'Date'], how='left')
    
        df_promocost = self.compute_Promo_Cost(json_params.get(self._country).get('dict_path').get('PATH_PROMO_COST').get('Total Country'))
        df_bel = pd.merge(df_bel, df_promocost, on=['Brand', 'Date'], how='left')
    
        df_hh = self.compute_HH_Index(json_params.get(self._country).get('dict_path').get('PATH_HH_INDEX').get('Total Country'))
        df_bel = pd.merge(df_bel, df_hh, on=['Brand', 'Date'], how='left')
    
        self._df_bel = df_bel
        #df_bel.to_excel('assets/df_bel.xlsx')
    
    def compute_mean_Price_sum_Volume(self):
        df_temp = self.get_df().groupby(['Date', 'Brand'], as_index=False)['Price per volume', 'Sales in volume'].agg(['mean', 'sum'])


    def compute_Inno_old(self, path, date_begining='2017-12-31'):
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

    def compute_Promo_Cost_old(self, path):
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

    def compute_HH_Index_old(self, path):
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
            

        #display(df_concat)

        #df_concat.to_excel('assets/cheese_market_leaders_USA.xlsx')