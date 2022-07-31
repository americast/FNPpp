
from utils.utils import check_region_data
import numpy as np
import pandas as pd
from datetime import date, timedelta
from utils.vis import visualize_region
import time
from epiweeks import Week
from scipy import stats
from itertools import compress
import pdb
import pickle
import os

# ew202201 remove
death_remove = []
hosp_remove = [] 
death_replace_4th_with_3th = []
increase_death_interval_high = []
increase_death_interval_low = []

regions_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
            'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
            'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE',
            'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
            'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
            'VA', 'WA', 'WV', 'WI', 'WY', 'X']



# get cumulative
def get_cumsum_region(datafile,region,target_name,ew):
    df = pd.read_csv(datafile, header=0)
    df = df[(df['region']==region)]
    if target_name=='death':
        cum = df.loc[:,'cdc_flu_hosp'].sum()
    elif target_name=='hosp':
        cum = None
        # raise Exception('not implemented')
        # cum = df.loc[:,'hospitalizedIncrease'].sum()
    else:
        print('error', region,target_name)
        time.sleep(2)
    return cum

def get_predictions_from_pkl(next,res_path,region,week_current=None):
    """ reads from pkl, returns predictions for a region as a list"""
    if week_current is None:
        week_current = int(str(Week.thisweek(system="CDC") - 1)[-2:])
        week_current = str(week_current)
    if len(week_current)==1:
        week_current = '0'+week_current
    path=res_path+'flu_deploy_week_' + week_current +'_feats_' + str(next) + '_predictions.pkl' # normal

    if not os.path.exists(path):
        print(path)
        return None
    predictions = []
    
    with open(path, 'rb') as f:
        data_pickle = pickle.load(f)

    idx = regions_list.index(region)
    predictions = data_pickle[:,idx]
    return predictions

def parse(region,ew,target_name,suffix,daily,write_submission,visualize,data_ew=None,res_path='./results-flu/',sub_path='./submissions-flu/'):
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

    # prev_cum = get_cumsum_region(datafile,region,target_name,ew)
    print(region)
    point_preds = []
    lower_bounds_preds = []
    upper_bounds_preds = []
    for nextk in range(1,k_ahead+1):
        predictions = get_predictions_from_pkl(nextk,res_path,region,str(ew.week))
        
        if predictions is None:
            continue
        quantile_cuts = [0.01, 0.025] + list(np.arange(0.05, 0.95+0.05, 0.05,dtype=float)) + [0.975, 0.99]
        new_predictions = []
        for pred in predictions:
            if pred < 0:
                pred = 0
            new_predictions.append(pred)
        predictions = new_predictions

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
            model='FluFNP-raw'
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

        
    suffix_=suffix
    if visualize:
        figpath=f'./figures-flu/ew{ew}/raw/'
        if not os.path.exists(figpath):
            os.makedirs(figpath)
        print('==='+target_name+' '+region+'===')
        # pdb.set_trace()
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

    # for region in temp_regions:
    #     parse(region,ew,target_name,suffix,daily,WRITE_SUBMISSION_FILE,PLOT)

    """ to generate past ones """

    # for ew in ['01','02','03','04','05','06','07','08','09']:
    for ew in ['10','11','12']:
        ew=Week.fromstring('2022'+str(ew))
        for region in temp_regions:
            parse(region,ew,target_name,suffix,daily,WRITE_SUBMISSION_FILE,PLOT)

    quit()
