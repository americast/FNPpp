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
# ew50 remove
death_remove = []
hosp_remove = []

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
    if target_name=='death' or 'cum_death':
        cum = df.loc[:,'death_jhu_incidence'].sum()
    elif target_name=='hosp':
        cum = df.loc[:,'hospitalizedIncrease'].sum()
    else:
        print('error', region,target_name)
        time.sleep(2)
    return cum

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
    print(region,prev_cum)
    point_preds = []
    lower_bounds_preds = []
    upper_bounds_preds = []
    for next in range(1,k_ahead+1):
        import os
        # NOTE: change here to read from pkl
        
#         path=res_path+'EW'+str(ew)+'/'+target_name+'_'+region+'_next'+str(next)+suffix+'.csv'
      
    
#         Yet to test these edits      
        if(daily):
            path=res_path+ 'deploy_week_' + str(next) + '_predictions.pkl'
        else:
            path=res_path+'mort_model_week_' + str(next) + '_predictions.pkl'
        
        
        if not os.path.exists(path):
            print(path)
            continue
        predictions = []
        
        
        with open(path, 'rb') as f:
            data_pickle = pickle.load(f)

        idx = regions_list.index(region)
        predictions = data_pickle[:,idx]
        # pdb.set_trace()
        
        
#         with open(path, 'r') as f:
#             for line in f:
#                 # print(line, end='')
#                 pred = float(line)
#                 predictions.append(pred)
            
        # filter outliers
        # z_scores = stats.zscore(predictions)
        # abs_z_scores = np.abs(z_scores)
        # fil = (abs_z_scores < LIMIT)
        # predictions=list(compress(predictions, fil))
        # print(np.mean(predictions))
        
        quantile_cuts = [0.01, 0.025] + list(np.arange(0.05, 0.95+0.05, 0.05,dtype=float)) + [0.975, 0.99]
        # import pdb
        # pdb.set_trace()
        if target_name=='death':
            MULT = 15
        elif target_name=='hosp':
            MULT = 30
        
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
        parse(region,ew,target_name,suffix,daily,True,PLOT)
    target_name='hosp'
    suffix='M1_daily_5_vEW'+str(ew)
    temp_regions = regions
    daily=True
    for region in temp_regions:
        parse(region,ew,target_name,suffix,daily,True,PLOT)
    quit()
