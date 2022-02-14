import sys
#sys.path.insert(0,"..")
sys.path.insert(0,"/Users/augustecousin/Documents/bel_gm_tool/gm_tool/")
import json

from datamanager.DM_Belgique import DM_Belgique
from model.M_Belgique import M_Belgique

PATH_TO_BRANDS = "/Users/augustecousin/Documents/bel_gm_tool/gm_tool/assets/brands.json"
PATH_TO_DATES = "/Users/augustecousin/Documents/bel_gm_tool/gm_tool/assets/dates.json"
PATH_TO_OUTPUTS = "/Users/augustecousin/Documents/bel_gm_tool/gm_tool/view/"

def main():
    with open(PATH_TO_BRANDS, 'r') as f:
        bel_brands = json.load(f)['BELGIUM']
    
    with open(PATH_TO_DATES, 'r') as f:
        dict_dates = json.load(f)['BELGIUM']

    data_manager=DM_Belgique()
    data_manager.open_excel()

    model = M_Belgique(bel_brands, dict_dates)
    
    #print(model.filter_data(data_manager.df))
    
    brand_positioning_matrix = model.compute_brand_positioning_matrix(data_manager.df, Y_BEG=dict_dates['2019'], Y_END=dict_dates['2020'])
    #brand_positioning_matrix.to_excel(PATH_TO_OUTPUTS + 'brand_positioning_matrix.xlsx')

    market_passeport = model.compute_market_passeport(data_manager.df)
    #market_passeport.to_excel(PATH_TO_OUTPUTS + 'market_passeport.xlsx')


if __name__ == '__main__':
    main()