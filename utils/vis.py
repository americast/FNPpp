
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
import numpy as np
from epiweeks import Year, Week
import os

# set the start week for the visualization
ew_vis_start='202210'

def visualize_region(target_name,region,predictions,datafile,opt,_min=None,_max=None,ew=19,suffix='',daily=False,fig_path='./figures/',show_rmse=False):
    """
        @param target_name: 'death' or 'hosp'
        @param predictions: [next1, next2, ...]
        @param datafile: e.g. "./data/merged-data3.csv"
        @param opt: 'inc' or 'cum'   NOTE: it converts inc to cumulatives
        @param _min: percentile for 5% - e.g. [min1,min2] for next2 pred  
        @param _max: percentile for 95% - e.g. [max1,max2] for next2 pred 
        @param daily: boolean

            
        ##### IMPORTANT !!!!!!!!!!!!!!!!!!!!
        @param ew: the week before the start of prediction, like ew =19, so the first week of prediction is 20, but the ground truth maybe more than 20
        NOTE:
        @param ew: making it epiweek obj
    """
    
    overlap= False
    df = pd.read_csv(datafile, header=0)
    df = df[df['region']==region]
    def convert(x):
        return Week.fromstring(str(x), system="CDC")
    df['epiweek'] = df.loc[:,'epiweek'].apply(convert)
    ## The end week of the groud turth
    end_week = df['epiweek'].max()
    ## The length of predictions
    len_pred = len(predictions)

    ## determine whether rmse is needed
    ## then establish the variables in need
    ## overlap_pred is the instance prediction from the model overlapped with ground truth in the same time interval
    ## overlap_inc is the instance ground truth in the same time inveral with overlap_pred
    # if end_week - ew !=0:
    if end_week != ew:
        overlap= True
        #overlap_pred = predictions[:(end_week-ew)]
        df_overlap = df[(df.loc[:,'epiweek'] <= (end_week)) & (df.loc[:,'epiweek'] > ew)].copy() 
        if target_name =='hosp':
            # overlap_inc = df_overlap.loc[:,'hospitalizedIncrease'].to_numpy()
            overlap_inc = df_overlap.loc[:,'cdc_hospitalized'].to_numpy()
        elif target_name == 'death' or target_name=='cum_death':
            overlap_inc = df_overlap.loc[:,'death_jhu_incidence'].to_numpy()

    cum_25 = df[(df.loc[:,'epiweek'] <= Week.fromstring('202040'))].loc[:,'death_jhu_incidence'].sum()
    df = df[(df.loc[:,'epiweek'] <= ew) & (df.loc[:,'epiweek'] >= Week.fromstring(ew_vis_start))]   ## data only from 10 to ew
    epiweeks = list(df.loc[:,'epiweek'].to_numpy())
    # days=list(df.loc[:,'date'].to_numpy())
    days=list(df.loc[:,'epiweek'].to_numpy())
    days=list(range(1,len(days)+1))
    if target_name == 'hosp':  # next starts at 1
        # inc = df.loc[:,'hospitalizedIncrease'].to_numpy()
        inc = df.loc[:,'cdc_hospitalized'].to_numpy()
        title_txt = 'Hospitalizations'
    elif target_name == 'death' or target_name=='cum_death':
        # inc = df.loc[:,'deathIncrease'].to_numpy()
        inc = df.loc[:,'death_jhu_incidence'].to_numpy()
        title_txt = 'Mortality'
    elif target_name == 'flu hosp':
        inc = df.loc[:,'cdc_flu_hosp'].to_numpy()
        title_txt = 'Flu Hosp'

    if opt=='inc':
        y=inc
        if overlap ==True:
            y_overlap = overlap_inc
        label='Incidence'
    elif opt=='cum':
        """
            Hack to fix cumulative: add what is was before ew202025
        """
        cum = [cum_25]
        
        for i in range(len(inc)-1):
            cum.append(inc[i+1]+cum[-1])
        
        y=cum
        if overlap==True:
            overlap_inc[0] += y[-1]
            y_overlap  = np.cumsum(overlap_inc, dtype=np.float64)
        label='Cumulative'
        _min[0] = _min[0]+cum[-1]
        _min[1:] = [_min[i]+y[-1]+sum(predictions[:i]) for i in range(1,len(_min))]
        _max[0] = _max[0]+y[-1]
        _max[1:] = [_max[i]+y[-1]+sum(predictions[:i]) for i in range(1,len(_max))]
        predictions[0] = predictions[0] + cum[-1]
        predictions[1:] = [sum(predictions[:i+1]) for i in range(1,len_pred)]
        

    ## weeks of predictions: like from 10 to 18 is the range of ground truth, 19 to 21 is the range of predictions

    ## overlap_pred is the instance prediction from the model overlapped with ground truth in the same time interval
    if overlap==True:
        overlap_pred = predictions[:(end_week-ew)]
        overlap_pred_weeks = list(range(ew+1, end_week+1))
    pred_weeks = [epiweeks[-1]+w for w in range(1,1+len_pred)]
    pred_days = [days[-1]+w for w in range(1,1+len_pred)]

    ## Calculate the RMSE
    if overlap ==True:  
        RMSE = []
        for index in range(1,len(overlap_pred)+1):
            RMSE.append(np.sqrt(mean_squared_error([overlap_pred[index-1]], [y_overlap[index-1]])))
        y_overlap = y_overlap.tolist()
        red_x= [epiweeks[-1]] + overlap_pred_weeks
        red_y = [y[-1]] + y_overlap

    if daily:
        plt.plot(days,y,'b',label='Ground truth data from JHU',linestyle='-')
        plt.plot([days[-1]]+pred_days,[y[-1]]+predictions,linestyle='-', marker='o', markersize=1, linewidth=1, color='r',label='Associated predictions')
    else:
        epiweeks = [str(e)[-2:] for e in epiweeks]
        pred_weeks = [str(e)[-2:] for e in pred_weeks]
        ## Plot ground truth data first
        plt.plot(epiweeks,y,'b',label='Ground truth data from JHU',linestyle='-')
        ## The predictions starts from the last week of ground truth
        plt.plot([epiweeks[-1]]+pred_weeks,[y[-1]]+predictions,linestyle='--', marker='o', color='r',label='Associated predictions')
    ## Plot the overlap data
    
    if overlap==True:
        plt.plot(red_x,red_y,linestyle='-', color='m', marker= "^",label=' Associated Ground Truth to Compare')

        if show_rmse:
            ##Plot RMSE 
            y_max = np.max(predictions)
            tick = y_max/10
            for  index,value in enumerate(RMSE): 
                plt.text( 0.85* end_week, 0.5*y_max - tick*index, 'rmse'+ str(index+1)+ ': '+str(round(value,2)), size=12)

    if daily:
        plt.xlabel('day')
    else:
        plt.xlabel('epidemic week')
    plt.ylabel(label+' '+target_name+' counts')

    #print(y)
    if _min is not None and _max is not None:  

        _min.insert(0,y[-1])
        _max.insert(0,y[-1])
        if daily:
            plt.fill_between([days[-1]]+pred_days, _min, _max, alpha = 0.25, label='95% Confidence Interval')
        else:
            plt.fill_between([epiweeks[-1]]+pred_weeks, _min, _max, alpha = 0.25, label='95% Confidence Interval')
        # plt.fill_between(pred_weeks, _min, _max, alpha = 0.25, label='95% Confidence Interval')
    # plt.xscale([0,])
    plt.legend(loc='upper left')
    plt.gca().set_ylim(bottom=0)
    if opt=='inc':
        if region=='X':
            plt.title('US Incidence '+title_txt)  
        else:
            plt.title(region+' '+label)  
        plt.savefig(fig_path+region+'_'+target_name+'_'+'ew'+str(ew)+suffix+'.png')
        print('inc predictions >>>>>',predictions)
    else:
        if region=='X':
            plt.title('US Cumulative '+title_txt)  
        else:
            plt.title(region+' '+label)   
        plt.savefig(fig_path+region+'_cum'+target_name+'_'+'ew'+str(ew)+suffix+'.png')
        print('cum predictions >>>>>',predictions)
    
    plt.close()


