from utils.utils import check_region_data, std_interval_correction, get_std_from_data, get_last_data_points, get_max_value, collect_and_correct
import numpy as np
import pandas as pd
from datetime import date, timedelta
from utils.vis import visualize_region
from epiweeks import Week
from scipy import stats
from itertools import compress
import pdb
import pickle
import os
from utils.flu_prediction_list import *

regions_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
            'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
            'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE',
            'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
            'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
            'VA', 'WA', 'WV', 'WI', 'WY', 'X']


def parse(region,ew,target_name,suffix,daily,write_submission,visualize,data_ew=None,res_path='./results-flu/',sub_path='./submissions-flu/'):
    """
        @param write_submission: bool
        @param visualize: bool
        @param data_ew: int, this is needed to use the most recent data file
                    if None, it takes the values of ew
    """
    if data_ew is None:
        data_ew=ew  
    k_ahead=4
    datafile='./data/covid-hospitalization-all-state-merged_vEW'+str(data_ew)+'.csv'
    if not check_region_data(datafile,region,target_name,ew): # checks if region is there in our dataset
        return 0    

    # max_val = get_max_value(datafile,region,target_name,ew)
    scale_data = get_std_from_data(datafile,region,target_name,ew)
    last_data_vals = get_last_data_points(datafile,region,target_name,ew)
    print(region)
    point_preds = []
    lower_bounds_preds = []
    upper_bounds_preds = []

    """ collect predictions """
    # get last values
    medians = last_data_vals.ravel().tolist()
    # extend list with collected predictions
    from utils.utils import get_AR_predictions_from_pkl
    medians.extend(get_AR_predictions_from_pkl(res_path,region,'flu'))

    """ smooth predictions using last observations in time series """
    window_width = 3
    cumsum_vec = np.cumsum(np.insert(medians, 0, 0)) 
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    medians = ma_vec[-4:]
    # medians = medians[-4:]

    """ scale data is tunable """
    scale_data = scale_data/2

    for nextk in range(1,k_ahead+1):
        quantile_cuts = [0.01, 0.025] + list(np.arange(0.05, 0.95+0.05, 0.05,dtype=float)) + [0.975, 0.99]
        predictions = std_interval_correction(medians[nextk-1],scale_data,nextk)
        quantiles = np.quantile(predictions, quantile_cuts)
        df = pd.read_csv(datafile, header=0)
        df = df[(df['region']==region)]
        # add to list
        lower_bounds_preds.append(quantiles[1])
        upper_bounds_preds.append(quantiles[-2])
        point_preds.append(np.mean(predictions))
        
        suffix_=suffix
        if write_submission:
            # >>>>>> 
            team='GT'
            model='FluFNP'
            # date=Week.fromstring('2020'+str(ew)).enddate() + timedelta(days=2)
            date=ew.enddate() + timedelta(days=2)
            datex=date
            date=date.strftime("%Y-%m-%d") 
            
            sub_file=sub_path+date+'-'+team+'-'+model+'.csv'
            if not os.path.exists(sub_file):
                f = open(sub_file,'w+')
                f.write('forecast_date,target,target_end_date,location,type,quantile,value\n')
                f.close()

            f = open(sub_file,'a+')
            # first target is inmmediate saturday
            target_end_date = datex + timedelta(days=5) + timedelta(days=7)*(nextk-1)  # good for weekly pred
            location_fips=df.loc[:,'fips'].iloc[-1]
            location_fips=str(location_fips)
            if len(location_fips)==1:
                location_fips = '0'+location_fips 

            if region in hosp_remove:
                suffix_=suffix+'_rm'
                continue
            f.write(str(datex)+','+str(nextk)+' wk ahead inc '+target_name+','+str(target_end_date)+','+location_fips+','+'point'+','+'NA'+','+"{:.2f}".format(np.mean(predictions))+'\n')
            for q_c, q_v in zip(quantile_cuts, quantiles):
                f.write(str(datex)+','+str(nextk)+' wk ahead inc '+target_name+','+str(target_end_date)+','+location_fips+','+'quantile'+','+"{:.4f}".format(q_c)+','+"{:.4f}".format(q_v)+'\n')

    if visualize:
        print('==='+target_name+' '+region+'===')
        figpath=f'./figures-flu/ew{ew}/processed-AR/'
        if not os.path.exists(figpath):
            os.makedirs(figpath)
        visualize_region(target_name,region,point_preds,datafile,'inc',lower_bounds_preds.copy(),upper_bounds_preds.copy(),ew,suffix_,daily,figpath)

    
if __name__ == "__main__":
    
    PLOT=True
    # WRITE_SUBMISSION_FILE=False
    WRITE_SUBMISSION_FILE=True
    ew= int(str(Week.thisweek(system="CDC") - 1)[-2:])
    print(ew)
    ew=Week.fromstring('2022'+str(ew))

    states = pd.read_csv("./data/states.csv", header=0, squeeze=True).iloc[:,1].unique()
    regions = np.concatenate((['X'], states),axis=0)
    regions = list(regions)
    
    target_name='flu hosp'
    daily=False
    temp_regions = regions

    suffix='M1_10_vEW'+str(ew)
    print(suffix)

    for region in temp_regions:
        parse(region,ew,target_name,suffix,daily,WRITE_SUBMISSION_FILE,PLOT)

    quit()
