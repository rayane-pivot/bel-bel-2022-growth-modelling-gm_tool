import sys
#sys.path.insert(0,"../")
from datamanager import utils

from datamanager.DM_USA import DM_USA
from model.Model import Model

PATH_TO_BRANDS = utils.get_assets_path("brands.json")
PATH_TO_OUTPUTS = "view/"

def main():
    data_manager=DM_USA()
    data_manager.open_excel()

    model = Model()
    
    #print(model.filter_data(data_manager.df))
    
    brand_positioning_matrix = model.compute_brand_positioning_matrix(data_manager.df)
    #brand_positioning_matrix.to_excel(PATH_TO_OUTPUTS + 'brand_positioning_matrix.xlsx')

    market_passeport = model.compute_market_passeport(data_manager.df)
    #market_passeport.to_excel(PATH_TO_OUTPUTS + 'market_passeport.xlsx')


if __name__ == '__main__':
    main()