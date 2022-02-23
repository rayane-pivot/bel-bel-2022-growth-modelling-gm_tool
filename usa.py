import sys
#sys.path.insert(0,"../")

import json

from datamanager import utils
from datamanager.DM_USA import DM_USA
from model.M_USA import M_USA

PATH_TO_BRANDS = utils.get_assets_path("brands.json")

PATH_TO_BRANDS = "assets/brands.json"
PATH_TO_AandP_CODES = "assets/AandP_codes.json"
PATH_TO_DATES = "assets/dates.json"
PATH_TO_OUTPUTS = "view/"

def main():
    with open(PATH_TO_AandP_CODES, 'r') as f:
        aandp_codes = json.load(f)['USA']
    
    with open(PATH_TO_BRANDS, 'r') as f:
        bel_brands = json.load(f)['USA']
    
    with open(PATH_TO_DATES, 'r') as f:
        dict_dates = json.load(f)['USA']
    
    data_manager=DM_USA(bel_brands, aandp_codes)
    #data_manager.open_excel()
    #data_manager.assert_dataframe()

    model = M_USA(bel_brands, dict_dates)
    
    #print(model.filter_data(data_manager.df))
    
    #brand_positioning_matrix = model.compute_brand_positioning_matrix(data_manager.df, Y_BEG=dict_dates['2018'], Y_END=dict_dates['2021'])
    #brand_positioning_matrix.to_excel(PATH_TO_OUTPUTS + 'USA_brand_positioning_matrix.xlsx')

    #market_passeport = model.compute_market_passeport(data_manager.df)
    #market_passeport.to_excel(PATH_TO_OUTPUTS + 'USA_market_passeport.xlsx')


if __name__ == '__main__':
    main()