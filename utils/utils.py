import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
device = torch.device("cpu")
dtype = torch.float

regions_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
            'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
            'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE',
            'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
            'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
            'VA', 'WA', 'WV', 'WI', 'WY', 'X']

hosp_state_error=[]
def check_region_data(datapath,region,target_name,ew):
    df = pd.read_csv(datapath, header=0)
    df = df[(df['region']==region)]
    if df.size == 0:
        print('region ', region, ' is missing!')
        return False
    if target_name == 'hosp':  # next starts at 1
        y = df.loc[:,'cdc_hospitalized'].to_numpy()
        if region in hosp_state_error:
            print('region ', region, ' is in list of hosp error!')
            return False
    elif target_name == 'death':
        y = df.loc[:,'death_jhu_incidence'].to_numpy()
    elif target_name == 'flu hosp':
        y = df.loc[:,'cdc_flu_hosp'].to_numpy()
    if y.sum()==0.:
        print('region ', region, ' is all zeros part!')
        return False
    return True


from epiweeks import Week, Year
# NOTE: if you change it, change also the one in ./data/data_utils.py
def get_epiweeks_list(start_ew,end_ew):
    """
        returns list of epiweeks objects between start_ew and end_ew (inclusive)
        this is useful for iterating through these weeks
    """
    iter_weeks = list(Year(2020).iterweeks()) + list(Year(2021).iterweeks())
    idx_start = iter_weeks.index(start_ew)
    idx_end = iter_weeks.index(end_ew)
    return iter_weeks[idx_start:idx_end+1]


def get_ground_truth(region, ew, target_name, ground_truth_file, daily):
    """
        Returns the ground truth values to compare with predictions for particular week, region
        @param ground_truth_file: ground truth file for different targets from the most recent epiweek
    """

    overlap = False
    df = pd.read_csv(ground_truth_file, header=0)
    df = df[df['region']==region]
    def convert(x):
        return int(str(x)[-2:])
    df['epiweek'] = df.loc[:, 'epiweek'].apply(convert)
    ew = convert(ew)
    end_week = df['epiweek'].max()
    if end_week - ew != 0:
        overlap = True
        df_overlap = df[(df.loc[:,'epiweek'] <= end_week) & (df.loc[:, 'epiweek'] > ew)].copy()
        if target_name =='hosp':
            # ground_truth = df_overlap.loc[:,'hospitalizedIncrease'].to_numpy()
            ground_truth = df_overlap.loc[:,'cdc_hospitalized'].to_numpy() # changed for nov23
        elif target_name == 'death' or target_name=='cum_death':
            ground_truth = df_overlap.loc[:,'death_jhu_incidence'].to_numpy()
    if daily:
        ground_truth = ground_truth[:min(28, (end_week-ew)*7)]
    else:
        ground_truth = ground_truth[:min(4, end_week-ew)]

    return ground_truth



def convert_regions_to_fips(regions):
    """
        One-to-one mapping between regions and fips 
        @params regions: list of regions
        @return list of fips numbers, same size as regions
        NOTE: repeatead in vis.py
    """
    df = pd.read_csv('./data/us-state-ansi-fips.csv')
    fips = df.set_index('stusps').loc[regions,'st']
    # print(regions,fips)
    return fips.to_list()


def convert_fips_to_regions(fips):
    """      
        One-to-one mapping between fips and regions
        @params fips: list of fips numbers
        @return list of regions, same size as fips
    """
    df = pd.read_csv('./data/us-state-ansi-fips.csv')
    regions = df.set_index('st').loc[fips,'stusps']
    return regions.to_list()


def multiplier_interval_correction(predictions,MULT):
    median = np.median(predictions)
    new_predictions = []
    for pred in predictions:
        if pred < median:
            deviation = median - pred
            deviation = deviation*MULT
            pred = median - deviation
        if pred > median:
            deviation = pred - median
            deviation = deviation*MULT
            pred = median + deviation
        if pred < 0:
            pred = 0
        new_predictions.append(pred)
    return new_predictions
# predictions = multiplier_interval_correction(predictions,MULT)

def std_interval_correction(median,scale_data,nextk):
    if nextk == 1:
        scale_data *= 1.
    elif nextk == 2:
        scale_data *= 1.2
    elif nextk == 3:
        scale_data *= 1.5
    elif nextk == 4:
        scale_data *= 1.8
    new_predictions = median + np.random.randn(2000)*scale_data
    return np.maximum(new_predictions, 0)

def get_max_value(datafile,region,target_name,ew):
    df = pd.read_csv(datafile, header=0)
    df = df[(df['region']==region)]
    if target_name=='death':
        val = df.loc[:,'death_jhu_incidence'].max()
    elif target_name=='hosp':
        val = df.loc[:,'cdc_hospitalized'].max()
        print('max val',val)
    elif target_name=='flu hosp':
        val = df.loc[:,'cdc_flu_hosp'].max()
    else:
        print('error', region,target_name)
        time.sleep(2)
    return val

def get_std_from_data(datafile,region,target_name,ew):
    df = pd.read_csv(datafile, header=0)
    df = df[(df['region']==region)]
    if target_name=='death':
        val = df.loc[:,'death_jhu_incidence'].std()
    elif target_name=='hosp':
        val = df.loc[:,'cdc_hospitalized'].std()
        print('max val',val)
    elif target_name=='flu hosp':
        val = df.loc[:,'cdc_flu_hosp'].std()
    else:
        print('error', region,target_name)
        time.sleep(2)
    return val

def get_last_data_points(datafile,region,target_name,ew,k_last=4):
    df = pd.read_csv(datafile, header=0)
    df = df[(df['region']==region)]
    if target_name=='death':
        val = df.loc[:,'death_jhu_incidence'].to_numpy()[-k_last:]
    elif target_name=='hosp':
        val = df.loc[:,'cdc_hospitalized'].to_numpy()[-k_last:]
    elif target_name=='flu hosp':
        val = df.loc[:,'cdc_flu_hosp'].to_numpy()[-k_last:]
    else:
        print('error', region,target_name)
        time.sleep(2)
    return val

# get cumulative
def get_cumsum_region(datafile,region,target_name,ew):
    df = pd.read_csv(datafile, header=0)
    df = df[(df['region']==region)]
    if target_name=='death':
        cum = df.loc[:,'death_jhu_incidence'].sum()
    elif target_name=='hosp':
        cum = None
        # raise Exception('not implemented')
        # cum = df.loc[:,'hospitalizedIncrease'].sum()
    else:
        print('error', region,target_name)
        time.sleep(2)
    return cum

import pickle

def get_predictions_from_pkl(next,res_path,region,target='covid'):
    """ reads from pkl, returns predictions for a region as a list"""
    week_current = int(str(Week.thisweek(system="CDC") - 1)[-2:])
    week_current = str(week_current)
    if len(week_current)==1:
        week_current = '0'+week_current
    if target=='covid':
        path=res_path+'mort_deploy_week_' + week_current +'_' + str(next) + '_predictions.pkl' # normal
    elif target=='flu':
        path=res_path+'flu_deploy_week_' + week_current +'_feats_' + str(next) + '_predictions.pkl'# b2f

    if not os.path.exists(path):
        print(path)
        return None
    predictions = []
    
    with open(path, 'rb') as f:
        data_pickle = pickle.load(f)

    idx = regions_list.index(region)
    predictions = data_pickle[:,idx]
    return predictions

def get_AR_predictions_from_pkl(res_path,region,target='covid'):
    """ reads from pkl, returns predictions for a region as a list"""
    week_current = int(str(Week.thisweek(system="CDC") - 1)[-2:])
    week_current = str(week_current)
    if len(week_current)==1:
        week_current = '0'+week_current
    if target=='covid':
        path=res_path+'mort_deploy_week_' + week_current +'_predictions.pkl' 
    elif target=='flu':
        path=res_path+'flu_deploy_week_' + week_current +'_feats_predictions.pkl'

    if not os.path.exists(path):
        print(path)
        return None
    predictions = []
    
    with open(path, 'rb') as f:
        data_pickle = pickle.load(f)
    
    idx = regions_list.index(region)
    predictions = data_pickle[:,idx,:]
    predictions = np.median(predictions,0)
    return predictions

def collect_and_correct(k_ahead,res_path,target,region,death_replace_last3,death_replace_last2,death_replace_last1,mode):
    medians = []
    for nextk in range(1,k_ahead+1):
        predictions = get_predictions_from_pkl(nextk,res_path,region,target)
        if mode=='downwards':
            if nextk in [2,3,4] and region in death_replace_last3:
                predictions = get_predictions_from_pkl(1,res_path,region,target)
                if nextk == 2:
                    predictions = [pred*0.8 for pred in predictions]
                elif nextk == 3:
                    predictions = [pred*0.7 for pred in predictions]
                else:
                    predictions = [pred*0.6 for pred in predictions]
            if nextk in [3,4] and region in death_replace_last2:
                predictions = get_predictions_from_pkl(2,res_path,region,target)
                if nextk == 3:
                    predictions = [pred*0.8 for pred in predictions]
                else:
                    predictions = [pred*0.6 for pred in predictions]
            elif nextk in [4] and region in death_replace_last1:
                    predictions = get_predictions_from_pkl(3,res_path,region,target)
                    # be lower than previous
                    predictions = [pred*0.6 for pred in predictions]
        elif mode=='upwards':
            if nextk in [2,3,4] and region in death_replace_last3:
                predictions = get_predictions_from_pkl(1,res_path,region,target)
                if nextk == 2:
                    predictions = [pred*1.1 for pred in predictions]
                elif nextk == 3:
                    predictions = [pred*1.2 for pred in predictions]
                else:
                    predictions = [pred*1.3 for pred in predictions]
            if nextk in [3,4] and region in death_replace_last2:
                predictions = get_predictions_from_pkl(2,res_path,region,target)
                if nextk == 3:
                    predictions = [pred*1.1 for pred in predictions]
                else:
                    predictions = [pred*1.2 for pred in predictions]
            elif nextk in [4] and region in death_replace_last1:
                    predictions = get_predictions_from_pkl(3,res_path,region,target)
                    # be lower than previous
                    predictions = [pred*1.1 for pred in predictions]
        else:
            raise Exception('mode incorrect')
        if predictions is None:
            continue
        pred_median = np.median(predictions)
        medians.append(pred_median)
    return medians