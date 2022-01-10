import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
device = torch.device("cpu")
dtype = torch.float

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

def std_interval_correction(median,scale_data):
    # median = np.median(predictions)
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