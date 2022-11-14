from torch import exp
from utils.utils import check_region_data, std_interval_correction, get_std_from_data, get_last_data_points, get_max_value, get_cumsum_region, collect_and_correct
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
import time
from utils.covid_prediction_list import *

regions_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
            'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
            'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE',
            'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
            'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
            'VA', 'WA', 'WV', 'WI', 'WY', 'X']

def parse(region,ew,target_name,suffix,daily,write_submission,visualize,data_ew=None,res_path='./results-covid/',sub_path='./submissions-covid/'):
    """
        @param write_submission: bool
        @param visualize: bool
        @param data_ew: int, this is needed to use the most recent data file
                    if None, it takes the values of ew
    """
    if data_ew is None:
        data_ew=ew  
    if daily:
        k_ahead=30 # changed from 28 to 30 on nov14 as requested by CDC (only needed for training)
        datafile='./data/covid-hospitalization-daily-all-state-merged_vEW'+str(data_ew)+'.csv'
    else:
        k_ahead=4
        datafile='./data/covid-hospitalization-all-state-merged_vEW'+str(data_ew)+'.csv'
    # pdb.set_trace()
    if not check_region_data(datafile,region,target_name,ew): # checks if region is there in our dataset
        return 0    

    prev_cum = get_cumsum_region(datafile,region,target_name,ew)
     # max_val = get_max_value(datafile,region,target_name,ew)
    std_data = get_std_from_data(datafile,region,target_name,ew)
    last_data_vals = get_last_data_points(datafile,region,target_name,ew)
    print(region)
    point_preds = []
    lower_bounds_preds = []
    upper_bounds_preds = []

    """ collect predictions """
    # get last values
    medians = last_data_vals.ravel().tolist()
    # extend list with collected predictions
    # medians.extend(collect_and_correct(k_ahead,res_path,'covid',region,death_replace_last3,death_replace_last2,death_replace_last1,mode='downwards'))
    medians.extend(collect_and_correct(k_ahead,res_path,'covid',region,death_replace_last3,death_replace_last2,death_replace_last1,mode='upwards'))

    """ smooth predictions using last observations in time series """
    window_width = 3
    cumsum_vec = np.cumsum(np.insert(medians, 0, 0)) 
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    medians = ma_vec[-4:]

    """ scale data is tunable """
    std_data = std_data/9

    if region in REDUCE_UNCERTAINTY_REGIONS:
        std_data = std_data/2

    for next in range(1,k_ahead+1):
        quantile_cuts = [0.01, 0.025] + list(np.arange(0.05, 0.95+0.05, 0.05,dtype=float)) + [0.975, 0.99]
        predictions = std_interval_correction(medians[next-1],std_data,next)
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
            model='DeepCOVID'
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
            target_end_date = datex + timedelta(days=5) + timedelta(days=7)*(next-1)  # good for weekly pred
            location_fips=df.loc[:,'fips'].iloc[-1]
            location_fips=str(location_fips)
            if len(location_fips)==1:
                location_fips = '0'+location_fips 

            if target_name=='death':
                if region in death_remove:
                    suffix_=suffix+'_rm'
                    continue
                f.write(str(datex)+','+str(next)+' wk ahead inc '+target_name+','+str(target_end_date)+','+location_fips+','+'point'+','+'NA'+','+"{:.2f}".format(np.mean(predictions))+'\n')
                for q_c, q_v in zip(quantile_cuts, quantiles):
                    f.write(str(datex)+','+str(next)+' wk ahead inc '+target_name+','+str(target_end_date)+','+location_fips+','+'quantile'+','+"{:.4f}".format(q_c)+','+"{:.4f}".format(q_v)+'\n')
                # cumulative here
                current_cum = prev_cum
                # print('region',current_cum)
                f.write(str(datex)+','+str(next)+' wk ahead cum '+target_name+','+str(target_end_date)+','+location_fips+','+'point'+','+'NA'+','+"{:.4f}".format(np.mean(predictions)+current_cum)+'\n')
                for q_c, q_v in zip(quantile_cuts, quantiles):
                    f.write(str(datex)+','+str(next)+' wk ahead cum '+target_name+','+str(target_end_date)+','+location_fips+','+'quantile'+','+"{:.4f}".format(q_c)+','+"{:.4f}".format(q_v+current_cum)+'\n')
                prev_cum = current_cum + np.mean(predictions)  # add mean
            elif target_name=='hosp':
                if region in hosp_remove:
                    suffix_=suffix+'_rm'
                    continue

                if daily:
                    target_end_date = datex + next*timedelta(days=1)
                    f.write(str(datex)+','+str(next)+' day ahead inc '+target_name+','+str(target_end_date)+','+location_fips+','+'point'+','+'NA'+','+"{:.4f}".format(np.mean(predictions))+'\n')
                    for q_c, q_v in zip(quantile_cuts, quantiles):
                        f.write(str(datex)+','+str(next)+' day ahead inc '+target_name+','+str(target_end_date)+','+location_fips+','+'quantile'+','+"{:.4f}".format(q_c)+','+"{:.4f}".format(q_v)+'\n')
                else: # weekly
                    # target_end_date = forecast_date + (next-1)*timedelta(days=7)  # for daily pred
                    target_end_date = datex + (next-1)*timedelta(days=7)  # for daily pred
                    current_cum = prev_cum 
                    point = np.mean(predictions) / 7.
                    quantiles = quantiles / 7.
                    for i in range(1,8):
                        target_end_date += timedelta(days=1)
                        f.write(str(datex)+','+str((next-1)*7+i)+' day ahead inc '+target_name+','+str(target_end_date)+','+location_fips+','+'point'+','+'NA'+','+"{:.4f}".format(point)+'\n')
                        for q_c, q_v in zip(quantile_cuts, quantiles):
                            f.write(str(datex)+','+str((next-1)*7+i)+' day ahead inc '+target_name+','+str(target_end_date)+','+location_fips+','+'quantile'+','+"{:.4f}".format(q_c)+','+"{:.4f}".format(q_v)+'\n')

    if visualize:
        figpath=f'./figures-covid/ew{ew}/processed/'
        if not os.path.exists(figpath):
            os.makedirs(figpath)
        if target_name=='death':
            print('==='+target_name+' '+region+'===')
            visualize_region(target_name,region,point_preds,datafile,'inc',lower_bounds_preds.copy(),upper_bounds_preds.copy(),ew,suffix_,daily,figpath)
            if region=='X':
                visualize_region(target_name,region,point_preds,datafile,'cum',lower_bounds_preds.copy(),upper_bounds_preds.copy(),ew,suffix_,daily,figpath)
        if target_name=='hosp':
            print('==='+target_name+' '+region+'===')
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
    
    target_name='death'
    daily=False
    temp_regions = regions

    suffix='M1_10_vEW'+str(ew)
    print(suffix)

    team='GT'
    model='DeepCOVID'
    # date=Week.fromstring('2020'+str(ew)).enddate() + timedelta(days=2)
    date=ew.enddate() + timedelta(days=2)
    datex=date
    date=date.strftime("%Y-%m-%d") 
    sub_path='./submissions-covid/'
    sub_file=sub_path+date+'-'+team+'-'+model+'.csv'
    if os.path.exists(sub_file):
        os.remove(sub_file)

    for region in temp_regions:
        parse(region,ew,target_name,suffix,daily,WRITE_SUBMISSION_FILE,PLOT)
    target_name='hosp'
    suffix='M1_daily_5_vEW'+str(ew)
    temp_regions = regions
    daily=True
    # for region in temp_regions:
    #     parse(region,ew,target_name,suffix,daily,True,PLOT)
    quit()
