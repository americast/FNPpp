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
# ew202141 remove
death_remove = ['TX','AZ','CT','CO']
death_replace_4th_with_3th = ['X','AL','AR','GA','ID','IN','LA','NV','OR','SC','TX','WV','WA']
hosp_remove = ['AR','LA']
# ew202142 remove
death_remove = ['X','NY']
death_replace_4th_with_3th = ['MI','MN','AZ','CT','ID','NJ','NM','NY','OH','OK','PA','VT']
hosp_remove = ['X','FL','MO','MS','NV','OK','AL','KY','GA','IA','ID','IL','IN','NC','SC','TN','TX','WA','WI']
# ew202143 remove
death_remove = ['X','WA']
hosp_remove = ["X","FL","LA","MO","MS","OK","SC","TX","AL","AR"]
death_replace_4th_with_3th = []
# ew202144 remove
death_remove = ['NY','IL']
hosp_remove = ['X','OR','OK','SC','FL','GA','NV','NC','MS','MO','LA','IN','KY','AL','NC','AR','TX','VA']
death_replace_4th_with_3th = ['NH','NM']
# ew202145 remove
death_remove = ['NY','OH','PA']
hosp_remove = ['X','FL','KY','LA','SC','AL','AR','ID','MO','NC','OR','PA','SC']
death_replace_4th_with_3th = ['NC']
# ew202146 remove
death_remove = []
hosp_remove = ['LA']
death_replace_4th_with_3th = ['SD']
# ew202147 remove
death_remove = ['MA','NE','NY']  # track 'IL','IN'
hosp_remove = ['AL','FL','LA','MS']
death_replace_4th_with_3th = []
# ew202148 remove
death_remove = ['OH'] #NJ
hosp_remove = ['X','AL','FL','GA','LA','MI','MS','SC','TX','CA','ID','KY'] #
death_replace_4th_with_3th = []
# ew202149 remove
death_remove = ['X','CT','DC','IL','MA','MI','MO','NJ','NY','OH','ME']
hosp_remove = []
death_replace_4th_with_3th = ['GA','IL','ME','NJ','NY','RI','VT']
# ew202150 remove
death_remove = ['CO','NJ']
hosp_remove = [] # all removed 
death_replace_4th_with_3th = []
increase_death_interval = ['MI','MN','NM','ME','NH','WI','RI']
# ew202151 remove
death_remove = ['X','WI','VA','OH','GA','FL','RI','AZ','NJ','NY','PA']#,'IA','SC','WI','OH','CT','WI']
hosp_remove = []  # all removed 
death_replace_4th_with_3th = []
increase_death_interval = ['DC','MI','MN','NM','ME','NH','DE','IL','LA']
# ew202152 remove
death_remove = ['AK','CA','GA','DC','OR','MN','AL','AR','AZ','CO','VA','IA','ND','KS','KY','IN','IA','TX','WY','SD','TN','NE','MI','MS','VT','OH','MT','UT','MD']
# to fix: 'X',NJ,NY,WA
hosp_remove = []  # all removed 
death_replace_4th_with_3th = ['ND','NV','MO','X','CT','NJ']
increase_death_interval_high = ['CO','FL','GA','IA','IL','MO','MS','NC','OH','OR','PA','RI','UT','WA','LA','DE','NH','SC']
increase_death_interval_low = ['X','NJ','NY']
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
        cum = df.loc[:,'death_jhu_incidence'].sum()
    elif target_name=='hosp':
        cum = None
        # raise Exception('not implemented')
        # cum = df.loc[:,'hospitalizedIncrease'].sum()
    else:
        print('error', region,target_name)
        time.sleep(2)
    return cum

def get_max_value(datafile,region,target_name,ew):
    df = pd.read_csv(datafile, header=0)
    df = df[(df['region']==region)]
    if target_name=='death':
        val = df.loc[:,'death_jhu_incidence'].max()
    elif target_name=='hosp':
        val = df.loc[:,'cdc_hospitalized'].max()
        print('max val',val)
    else:
        print('error', region,target_name)
        time.sleep(2)
    return val

def get_predictions_from_pkl(next,res_path,region):
    """ reads from pkl, returns predictions for a region as a list"""
    week_current = int(str(Week.thisweek(system="CDC") - 1)[-2:])
    if(daily):
        path=res_path+ 'deploy_week_' + str(week_current) +'_' + str(next) + '_predictions.pkl'
    else:
        path=res_path+'mort_deploy_week_' + str(week_current) +'_' + str(next) + '_predictions.pkl' # normal
        # path=res_path+'mort_deploy_week_' + str(week_current) +'_' + str(next) + '_predictions_refined.pkl'# b2f

    if not os.path.exists(path):
        print(path)
        return None
    predictions = []
    
    with open(path, 'rb') as f:
        data_pickle = pickle.load(f)

    idx = regions_list.index(region)
    predictions = data_pickle[:,idx]
    return predictions

def parse(region,ew,target_name,suffix,daily,write_submission,visualize,data_ew=None,res_path='./results/',sub_path='./submissions/'):
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
    max_val = get_max_value(datafile,region,target_name,ew)
    print(region,prev_cum)
    point_preds = []
    lower_bounds_preds = []
    upper_bounds_preds = []
    for next in range(1,k_ahead+1):

        predictions = get_predictions_from_pkl(next,res_path,region)
        
        if target_name=='death':
            # MULT = 1.2
            MULT = 2
            if region in increase_death_interval_high:
                MULT = 6
            elif region in increase_death_interval_low:
                MULT = 4
            LIMIT=3 # for outliers
            if next==4:
                if region in death_replace_4th_with_3th:
                    predictions = get_predictions_from_pkl(3,res_path,region)
                    # subtract one (just to make them different)
                    predictions = [pred*1.2 for pred in predictions]
        elif target_name=='hosp':
            # MULT = 3.5
            # MULT = 3  # changed to 2 on 11/15
            # update on 11/29
            # if next < 13:
            #     MULT = 3.5
            # else:
            MULT = 1
            LIMIT= 1.3 # for outliers

        if predictions is None:
            continue
        quantile_cuts = [0.01, 0.025] + list(np.arange(0.05, 0.95+0.05, 0.05,dtype=float)) + [0.975, 0.99]
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
        predictions = new_predictions

        # filter outliers
        z_scores = stats.zscore(predictions)
        abs_z_scores = np.abs(z_scores)
        # fil = (abs_z_scores < LIMIT) & (predictions < max_val) # filter too big, TODO: too small & (predictions > min_val)
        fil = (abs_z_scores < LIMIT)
        predictions=list(compress(predictions, fil))

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
        if target_name=='death':
            print('==='+target_name+' '+region+'===')
            visualize_region(target_name,region,point_preds,datafile,'inc',lower_bounds_preds.copy(),upper_bounds_preds.copy(),ew,suffix_,daily)
            if region=='X':
                visualize_region(target_name,region,point_preds,datafile,'cum',lower_bounds_preds.copy(),upper_bounds_preds.copy(),ew,suffix_,daily)
        if target_name=='hosp':
            print('==='+target_name+' '+region+'===')
            visualize_region(target_name,region,point_preds,datafile,'inc',lower_bounds_preds.copy(),upper_bounds_preds.copy(),ew,suffix_,daily)

    
if __name__ == "__main__":
    
    PLOT=True
    # WRITE_SUBMISSION_FILE=False
    WRITE_SUBMISSION_FILE=True
    ew= int(str(Week.thisweek(system="CDC") - 1)[-2:])
    print(ew)
    ew=Week.fromstring('2021'+str(ew))

    states = pd.read_csv("./data/states.csv", header=0, squeeze=True).iloc[:,1].unique()
    regions = np.concatenate((['X'], states),axis=0)
    regions = list(regions)
    
    target_name='death'
    daily=False
    temp_regions = regions

    suffix='M1_10_vEW'+str(ew)
    print(suffix)

    for region in temp_regions:
        parse(region,ew,target_name,suffix,daily,WRITE_SUBMISSION_FILE,PLOT)
    target_name='hosp'
    suffix='M1_daily_5_vEW'+str(ew)
    temp_regions = regions
    daily=True
    # for region in temp_regions:
    #     parse(region,ew,target_name,suffix,daily,True,PLOT)
    quit()
