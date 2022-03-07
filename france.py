import json
from datamanager.DM_FR import DM_FR
from model.M_FR import M_FR

PATH_TO_PARAMS = 'assets/params.json'
PATH_TO_OUTPUTS = "view/"

def main():
    with open(PATH_TO_PARAMS, 'r') as f:
        json_sell_out_params = json.load(f)
    
    data_manager=DM_FR()
    data_manager.ad_hoc_FR(json_sell_out_params)
    #data_manager.fill_df_bel(json_sell_out_params)
    
    #data_manager.load('view/USA_df_post_processing.xlsx')
    #print(data_manager.get_df().shape)
    #print(data_manager.get_df_bel().shape)

    #data_manager.get_df().to_excel('view/USA_df_postprocessing_0303.xlsx', index=False)
    #data_manager.get_df_bel().to_excel('view/USA_df_bel_0303.xlsx')
    
    model = M_FR()
    
    #print(model.filter_data(data_manager.df))
    year1 = json_sell_out_params.get('FR').get('brand_positioning_matrix').get("year1")
    year2 = json_sell_out_params.get('FR').get('brand_positioning_matrix').get("year2")
    year_min= json_sell_out_params.get('FR').get('brand_positioning_matrix').get("year_min")

    for channel, df in data_manager.get_df_channels().items():
        brand_positioning_matrix = model.compute_brand_positioning_matrix(df, 
                                                                      year_min=year_min, 
                                                                      year1=year1, 
                                                                      year2=year2)
        
        brand_positioning_matrix.to_excel(f'view/France/FR_{channel}_brand_positioning_matrix_0703.xlsx')

    # brand_scorecard = model.compute_brand_scorecard(data_manager.get_df(), 
    #                                                 data_manager.get_df_bel(), 
    #                                                 json_sell_out_params=json_sell_out_params, 
    #                                                 country='USA')
    
    #brand_scorecard.to_excel('view/USA_brand_scorecard_0303.xlsx')


if __name__ == '__main__':
    main()