import json
import datetime as dt
import argparse
import warnings
import time
import random 

import pandas as pd
import numpy as np
from prophet import Prophet
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_absolute_percentage_error

from datamanager.DataManager import DataManager

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

def build_train_set(df):
    df_temp = df.copy()
    df_temp = df_temp[pd.to_numeric(df_temp['Price per volume'], errors='coerce').notnull()]
    df_temp = df_temp[["Category", "Sub Category", "Brand", "Distribution", "Price per volume", "Sales in volume"]]

    df_n_cat = df_temp.groupby(['Brand'])['Category'].agg(Number_of_Categories = "nunique").reset_index().rename(columns={'Number_of_Categories':'Number of Categories'})
    df_n_sub_cat = df_temp.groupby(['Brand', 'Category'])['Sub Category'].agg(Sub_Categories='nunique').reset_index().rename(columns={'Sub_Categories':'Number of Sub Categories'})

    df_temp = pd.merge(df_temp, df_n_cat, on="Brand", how="left")
    df_temp = pd.merge(df_temp, df_n_sub_cat, on=["Brand", 'Category'], how="left")

    df_sub = df_temp.groupby(['Category', 'Sub Category', 'Brand'], as_index=False)['Number of Categories', 'Number of Sub Categories'].max()

    """compute high level metrics for time series of Distribution and Price per volume : min, max, mean, std"""
    Distribution = df_temp.groupby(['Category', 'Sub Category', 'Brand'], as_index=False)['Distribution'].agg(
        {'min Distribution':'min', 'max Distribution':'max', 'mean Distribution':'mean', 'std Distribution':'std'}
    )

    Price = df_temp.groupby(['Category', 'Sub Category', 'Brand'], as_index=False)['Price per volume'].agg(
        {'min Price':'min', 'max Price':'max', 'mean Price':'mean', 'std Price':'std'}
    )

    """merge high level metrics into df_sub"""
    df_sub = pd.merge(df_sub, Distribution, on=['Category', 'Sub Category', 'Brand'])

    df_sub = pd.merge(df_sub, Price, on=['Category', 'Sub Category', 'Brand'])

    df_sub = df_sub.fillna(0.0)
    
    return df_sub

def generate_targets(data_manager:DataManager, channel:str=None) -> dict:
    targets = []
    
    if channel:
        df = data_manager.get_df_by_channel(channel)
        df_bel = data_manager.get_df_bel_by_channel(channel)
    else:
        df = data_manager.get_df()
        df_bel = data_manager.get_df_bel()
    
    print(f"<generate_targets> generating {df.Category.nunique() * df_bel.Brand.nunique() = } targets")
    #bel_brands = np.append(["BOURSIN"], df_bel.Brand.unique())
    bel_brands = df_bel.Brand.unique()
    for category in df.Category.unique():
        #for brand in df_bel.Brand.unique():
        for brand in bel_brands:
            n_categories = df[df.Brand==brand].Category.nunique()
            n_sub_categories = df[df.Brand==brand]["Sub Category"].nunique()
            sub_cat = df[df.Category==category].groupby('Sub Category')['Sales in volume'].agg('sum').sort_values(ascending=False).index[0]
            distribution = df[df.Category==category].Distribution.mean()
            price = df[pd.to_numeric(df['Price per volume'], errors='coerce').notnull()][df.Category==category]['Price per volume'].agg("mean")

            own_distribution = df[df.Brand==brand].Distribution.mean()
            own_price = df[df.Brand==brand]['Price per volume'].agg("mean")

            if 'Price without promo' in df.columns:
                price_no_promo = df[df.Category==category]['Price without promo'].median()
            else :
                price_no_promo = df[pd.to_numeric(df['Price per volume'], errors='coerce').notnull()][df.Category==category]['Price per volume'].mean()
            target ={
                'Brand': brand,
                'Category': category,
                'Sub Category': sub_cat,
                'Number of Categories': n_categories,
                'Number of Sub Categories': n_sub_categories,
                'Date': "2022-01-01",
                'Period': 1,
                'Distribution': (2*distribution + own_distribution)/3,
                'Price per volume': (2*price + own_price)/3,
                'Price per volume without promo': price_no_promo,
            }
            targets.append(target)

    return pd.DataFrame(targets).to_dict(orient='index')

def build_profile(neighbors, df_sub, df, le_brand, le_cat, le_sub_cat):
    df_temp = df_sub.iloc[neighbors]
    df_temp['Category'] = le_cat.inverse_transform(df_temp['Category'])
    df_temp['Sub Category'] = le_sub_cat.inverse_transform(df_temp['Sub Category'])
    df_temp['Brand'] = le_brand.inverse_transform(df_temp['Brand'])
    #df_merge = pd.merge(df_temp[['Category', 'Sub Category']], df, on=['Category', 'Sub Category'], how='left')
    df_merge = pd.merge(df_temp[['Category', 'Sub Category', 'Brand']], df, on=['Category', 'Sub Category', 'Brand'], how='left')
    profile = df_merge.groupby('Date')['Sales in volume'].agg('mean')
    return profile

def forecast_profile(profile, target, periods=157, freq='M', plot=False):
    """ Forecasting profile using Prophet logistic growth giving the profile, 
    the targets for plotting useful information, and finally the periods and 
    freq of projection.

    """
    model_err = Prophet(growth='logistic', daily_seasonality=False, weekly_seasonality=False)
    
    # Adding Cap and Floor prediction 
    profile['cap'] = profile.y.max() * 2
    profile['floor'] = profile.y.mean() / 2
    model_err.fit(profile)

    err_forecast = model_err.predict()
    profile_err = profile.copy()
    
    profile_err.ds = pd.to_datetime(profile_err.ds)
    #profile_err = profile_err.rename(columns={"Date": "ds"})
    error_df = pd.merge(profile_err, err_forecast, on='ds')
    mape =  mean_absolute_percentage_error(error_df['y'], error_df['yhat'])
    
    model = Prophet(growth='logistic', daily_seasonality=False, weekly_seasonality=False)
    model.fit(profile)

    future = model.make_future_dataframe(periods=periods, freq=freq)
    future['cap'] = profile['cap'][0]
    future['floor'] = profile['floor'][0]
    
    # Predicting sales forecasts
    fcst = model.predict(future[future.ds > profile.ds.iloc[-1]])
    # Plot forecast
    # if plot:
    #     fig = model.plot(fcst)
    #     ax = fig.gca()
    #     ax.set_title("{} => {}".format(target['Brand'], target['Category']))
        
    # dataset of columns ['ds', 'y'] to return, with correct historic values
    fcst = pd.concat([profile[['ds', 'y']], fcst[['ds', 'yhat']].rename(
        columns={'yhat': 'y'})])
    fcst['ds'] = pd.to_datetime(fcst['ds'], format='%Y-%m-%d')

    return fcst, mape

def attack_new_markets(
    data_manager:DataManager, 
    targets:dict, 
    json_sell_out_params:dict, 
    country:str, 
    channel:str=None, 
    periods:int=157) -> dict:

    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import LabelEncoder
    
    if channel:
        df = data_manager.get_df_by_channel(channel)
    else:
        df = data_manager.get_df()

    df_sub = build_train_set(df)
    df_sub_no_encoding = df_sub.copy()

    le_cat = LabelEncoder()
    df_sub['Category'] = le_cat.fit_transform(df_sub['Category'])
    le_sub_cat = LabelEncoder()
    df_sub['Sub Category'] = le_sub_cat.fit_transform(df_sub['Sub Category'])
    le_brand = LabelEncoder()
    df_sub['Brand'] = le_brand.fit_transform(df_sub['Brand'])

    neigh = NearestNeighbors()
    neigh.fit(df_sub)
    for key in targets:
        print(f"<attack_new_market> predict : {targets.get(key).get('Brand')} {targets.get(key).get('Category')}")
        target = targets.get(key).copy()
        category = target.get('Category')
        
        p = {
                'Category':le_cat.transform([category])[0],
                'Sub Category':le_sub_cat.transform([target.get('Sub Category')])[0],
                'Brand':le_brand.transform([target.get('Brand')])[0],
                'Number of Categories':target.get('Number of Categories'),
                'Number of Sub Categories':target.get('Number of Sub Categories'),
                'min Distribution':target.get('Distribution') - (target.get('Distribution')*1/20),
                'max Distribution':target.get('Distribution') + (target.get('Distribution')*1/20),
                'mean Distribution':target.get('Distribution'),
                'std Distribution':(target.get('Distribution')*1/100),
                'min Price':target.get('Price per volume') - (target.get('Price per volume')*1/20),
                'max Price':target.get('Price per volume') + (target.get('Price per volume')*1/20),
                'mean Price':target.get('Price per volume'),
                'std Price':(target.get('Price per volume')*1/100),
        }

        nb_samples_per_cat = len(df_sub_no_encoding[df_sub_no_encoding.Category == category])
        ten_per_cent = int(nb_samples_per_cat * 0.1)
        k = ten_per_cent if ten_per_cent >= 100 else 10
        
        p = pd.DataFrame(p, index=[0]).fillna(0.0)

        neighbors = neigh.kneighbors(p, k, return_distance=False)
        #neighbors = random.choices(neighbors[0], k=80)
        #p["Brand"] = b["Brand"]
        profile = build_profile(neighbors[0], df_sub, df, le_brand, le_cat, le_sub_cat)                         
        # Transform series to dataframe
        profile = profile.to_frame().reset_index().rename(columns={'Date': 'ds', 'Sales in volume': 'y'})
        # Forecast with profile
        forecasts, mape = forecast_profile(profile, target, periods=periods, freq='W', plot=False)
        
        targets[key]['3Y'] = forecasts[forecasts.ds > profile.ds.iloc[-1]].y.sum()
        targets[key]['mape'] = mape
    
    pred = [
        {
            'Brand': targets.get(x).get('Brand'), 
            'Category': targets.get(x).get('Category'), 
            '3Y': targets.get(x).get('3Y'),
            'mape':targets.get(x).get('mape'),
        } for x in targets
    ]
        
    bel_brands = json_sell_out_params.get(country).get('bel_brands')
    y_true=pd.pivot_table(df[df.Brand.isin(bel_brands)], values='Sales in volume', columns='Category', index='Brand', aggfunc='sum')
    y_true = y_true.div(1000)
    df_pred = pd.DataFrame(pred)

    df_pred["3Y"] = df_pred["3Y"].apply(lambda x:x/1000)
    df_pred["mape"] = df_pred["mape"]

    table_preds = []
    for key in ["3Y", "mape"]:
        y_pred = df_pred.pivot_table(values=key, columns='Category', index='Brand')
        y_pred[~y_true.isna()] = None
        table_preds.append(y_pred)

    return table_preds

def main(args) -> None:
    ### Run with : python ACF.py --country GER --channel "Total Country"
    date = dt.datetime.now()
    country = args.country
    channel = args.channel
    periods = args.periods
    
    with open("assets/params.json", "r") as f:
        json_sell_out_params = json.load(f)
    
    ### Load data
    data_manager_name = f"DM_{country}"

    DM = __import__(f"datamanager.{data_manager_name}", globals(), locals(), fromlist=[data_manager_name], level=0)
    data_manager = eval(f"DM.{data_manager_name}()")
    
    eval(f"data_manager.ad_hoc_{country}(json_sell_out_params)")
    data_manager.fill_df_bel(json_sell_out_params)

    ### Generate targets
    targets = generate_targets(
        data_manager=data_manager,
        channel=channel
    )

    ### Run simulations
    (sales_pred, mape_pred) = attack_new_markets(
        data_manager=data_manager,
        targets=targets,
        json_sell_out_params=json_sell_out_params,
        country=country,
        channel=channel,
        periods=periods
    )

    ### SAVE predictions
    sales_pred.to_excel(f"view/{country}/{country}_pred_sales_{str(periods)}_weeks_{date.strftime('%d%m')}.xlsx")
    mape_pred.to_excel(f"view/{country}/{country}_pred_mape_{str(periods)}_weeks_{date.strftime('%d%m')}.xlsx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", type=str, help="Country")
    parser.add_argument("--channel", type=str, help="Channel")
    parser.add_argument("--periods", type=int, help="Num of Periods")
    args = parser.parse_args()
    
    start_time = time.time()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))