import sys
sys.path.insert(0,"..")

from datamanager.DM_Belgique import DM_Belgique
from model.M_Belgique import M_Belgique

PATH_TO_OUTPUTS = ""

def main():
    data_manager=DM_Belgique()
    data_manager.open_excel()

    model = M_Belgique()
    
    #print(model.filter_data(data_manager.df))
    
    brand_positioning_matrix = model.compute_brand_positioning_matrix(data_manager.df)
    #brand_positioning_matrix.to_excel(PATH_TO_OUTPUTS + 'brand_positioning_matrix.xlsx')

    market_passeport = model.compute_market_passeport(data_manager.df)
    #market_passeport.to_excel(PATH_TO_OUTPUTS + 'market_passeport.xlsx')


if __name__ == '__main__':
    main()