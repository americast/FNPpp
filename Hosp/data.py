import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from epiweeks import Week, Year
import pandas as pd


DAY_WEEK_MULTIPLIER = 7
SMOOTH_WINDOW = 21
WEEKS_AHEAD = 4
PAD_VALUE = -9
EW_START_DATA = '202020'
SMOOTH_MOVING_WINDOW = True

datapath = './data/covid-hospitalization-daily-all-state-merged_vEW202133.csv'
datapath_weekly = './data/covid-hospitalization-all-state-merged_vEW202133.csv'

min_sequence_length = 20

all_hhs_regions = ['AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC',
            'FL', 'GA', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
            'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'NE',
            'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
            'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT',
            'VA', 'WA', 'WV', 'WI', 'X']


macro_features=[
    'retail_and_recreation_percent_change_from_baseline',
    'grocery_and_pharmacy_percent_change_from_baseline',
    'parks_percent_change_from_baseline',
    'transit_stations_percent_change_from_baseline',
    'workplaces_percent_change_from_baseline',
    'residential_percent_change_from_baseline',
    'apple_mobility',
    'cdc_hospitalized',
    'covidnet',
    'fb_survey_cli', # Not available now
    'death_jhu_incidence',
    'positiveIncr',
    'negativeIncr',
    ] 


def convert_to_epiweek(x):
    return Week.fromstring(str(x))


def get_epiweeks_list(start_ew,end_ew):
    """
        returns list of epiweeks objects between start_ew and end_ew (inclusive)
        this is useful for iterating through these weeks
    """
    iter_weeks = list(Year(2020).iterweeks()) + list(Year(2021).iterweeks())
    idx_start = iter_weeks.index(start_ew)
    idx_end = iter_weeks.index(end_ew)
    return iter_weeks[idx_start:idx_end+1]



def load_df(region,ew_start_data,ew_end_data,temporal='daily'):
    """ load and clean data"""
    if temporal=='daily':
        df = pd.read_csv(datapath,low_memory=False)
    elif temporal=='weekly':
        df = pd.read_csv(datapath_weekly,low_memory=False)
    df = df[(df["region"] == region)]
    df['epiweek'] = df.loc[:, 'epiweek'].apply(convert_to_epiweek)
    # subset data using init parameters
    df = df[(df["epiweek"] <= ew_end_data) & (df["epiweek"] >= ew_start_data)]
    df = df.fillna(method="ffill")
    df = df.fillna(method="backfill")
    df = df.fillna(0)
    if SMOOTH_MOVING_WINDOW:
        def moving_average(x, w):
            return np.convolve(x, np.ones(w)/w, mode='full')[:-w+1]
        # smooth
        # df.loc[:,'positiveIncr'] = moving_average(df.loc[:,'positiveIncr'].values,SMOOTH_WINDOW)
        df.loc[:,'cdc_hospitalized'] = moving_average(df.loc[:,'cdc_hospitalized'].values,SMOOTH_WINDOW)
    return df


def get_state_train_data(region,pred_week,ew_start_data=EW_START_DATA,temporal='daily'):
    """ get processed dataframe of data + target as array """
    # import data
    region = str.upper(region)
    pred_week=convert_to_epiweek(pred_week) 
    ew_start_data=convert_to_epiweek(ew_start_data)
    df = load_df(region,ew_start_data,pred_week,temporal)
    # select targets
    # targets = df.loc[:,['positiveIncr','death_jhu_incidence']].values
    targets = df.loc[:,['cdc_hospitalized']].values
    # now subset based on input ew_start_data
    df = df[macro_features]
    return df, targets



def create_window_seqs(X, y, min_sequence_length, pad_value=PAD_VALUE):
    """
        Creates windows of fixed size with appended zeros
        @param X: features
        @param y: targets, in synchrony with features (i.e. x[t] and y[t] correspond to the same time)
    """
    # convert to small sequences for training, starting with length 10
    seqs, mask_seqs, targets, mask_ys = [], [], [], []

    # starts at sequence_length and goes until the end
    for idx in range(min_sequence_length, X.shape[0]+1, 1):
        # Sequences
        seqs.append(torch.from_numpy(X[:idx,:]))
        mask_seqs.append(torch.ones(idx))

        # Targets
        y_val = y[idx:idx+WEEKS_AHEAD*DAY_WEEK_MULTIPLIER]
        y_ = np.ones((WEEKS_AHEAD*DAY_WEEK_MULTIPLIER,y_val.shape[1])) * pad_value
        y_[:y_val.shape[0],:] = y_val
        mask_y = torch.zeros(WEEKS_AHEAD*DAY_WEEK_MULTIPLIER)  # this ensures that 
        mask_y[:len(y_val)] = 1
        targets.append(torch.from_numpy(y_))
        mask_ys.append(mask_y)

    seqs = pad_sequence(seqs,batch_first=True,padding_value=0).type(torch.float)
    mask_seqs = pad_sequence(mask_seqs,batch_first=True,padding_value=0).type(torch.float)
    ys = pad_sequence(targets,batch_first=True,padding_value=pad_value).type(torch.float)
    mask_ys = pad_sequence(mask_ys,batch_first=True,padding_value=0).type(torch.float)

    return seqs, mask_seqs, ys, mask_ys


class SeqData(torch.utils.data.Dataset):
    def __init__(self, region, X, mask_X, y, mask_y, time_seq):
        self.region = region
        self.X = X
        self.mask_X = mask_X
        self.y = y
        self.mask_y = mask_y
        self.time = time_seq

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            self.region[idx],
            self.X[idx, :, :],
            self.mask_X[idx],
            self.y[idx],
            self.mask_y[idx],
            self.time[idx]
        )
