#!/usr/bin/env python
# coding: utf-8

# In[34]:


'''
data sources:
1. google mobility: https://www.google.com/covid19/mobility/
2. apple mobilty: https://www.apple.com/covid19/mobility
3. Covid ExposureIndex dex: https://github.com/COVIDExposureIndices/COVIDExposureIndices
4. kinsa: kinsa_pull.ipynb 
5. fb-google survey: deplhi api
6. hospitalization: https://covidtracking.com/api
6a. positiveIncrease: https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series
6b. negaiveResults, totalRes: https://healthdata.gov/dataset/COVID-19-Diagnostic-Laboratory-Testing-PCR-Testing/j8mb-icvb
7. jhu deaths: https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series
8. iqvia: Alex from Jimeng
9. excess deaths: https://www.cdc.gov/nchs/nvss/vsrr/covid19/excess_deaths.htm
10. Emergency visits (less priority: region level): 
#https://www.cdc.gov/coronavirus/2019-ncov/covid-data/covidview/07242020/covid-like-illness.html
https://www.cdc.gov/coronavirus/2019-ncov/covid-data/covidview/10092020/outpatient-emergency-visits.html
11. covidnet: original cdc, processed data to use given by ALEX
12. CDC hospitalized: https://healthdata.gov/dataset/covid-19-reported-patient-impact-and-hospital-capacity-state-timeseries
13. vaccine doses: https://github.com/govex/COVID-19/blob/master/data_tables/vaccine_data/us_data/time_series/vaccine_data_us_timeline.csv
'''


# In[112]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import requests
import pandas as pd
import numpy as np
import math
import glob
import copy
from datetime import datetime
from datetime import timedelta
from datetime import date

from epiweeks import Week, Year
from delphi_epidata import Epidata
from dateutil.relativedelta import relativedelta


# In[36]:


def unit_test(data_dic,cols,epiweek_date,state_index,outfile):
    state_names=list(state_index.keys())
    all_cols=['date','region']+cols
    out_data=pd.DataFrame(columns=all_cols)
    for st in range(len(state_names)):
        temp_data=pd.DataFrame(columns=all_cols)
        temp_data['date']=epiweek_date
        temp_data['region']=[state_names[st]]*len(epiweek_date)
        for c in cols:
            temp_data[c]=data_dic[c][st][:]
        out_data=out_data.append(temp_data,ignore_index=True)
    out_data=out_data[all_cols]
    print(out_data.shape)
    out_data.to_csv(outfile,index=False)
    print('output file written')


# In[37]:


def read_survey_epidata(col_name,epidata,state_index,state_names,epiweek_date):
    week_cases=np.full((len(state_names),len(epiweek_date)), np.nan)
    total_sample=0
    for ix in range(len(epidata)):
        row=epidata[ix]
        name=row['geo_value'].upper()
        w_idx=find_date_index(epiweek_date,str(row['time_value']))
        if name == 'US' and w_idx !=-1:
            week_cases[0][w_idx] = row['value']
        elif name in state_names and w_idx!=-1:
            state_id=state_index[name]
            week_cases[state_id][w_idx]=row['value']
    return week_cases


# In[38]:


def read_delphi_vaccine(state_index,epiweek_date,start_week,end_week):
    vacc11=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc21=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc31=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','state',[20210208, Epidata.range(20210208, end_week)],'*')
    vacc41=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','state',[20210308, Epidata.range(20210308, end_week)],'*')
    vacc51=Epidata.covidcast('fb-survey','smoothed_wspent_time_1d','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')
    
    vacc12=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210301, Epidata.range(20210301, 20210501)],'*')
    vacc13=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210501, Epidata.range(20210501, 20210701)],'*')
    vacc14=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210701, Epidata.range(20210701, 20210901)],'*')
    vacc15=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc16=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20211101, Epidata.range(20211101, 20220101)],'*')
    vacc17=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20220101, Epidata.range(20211101, end_week)],'*')
    
    vacc22=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210301, Epidata.range(20210301, 20210601)],'*')
    vacc23=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210601, Epidata.range(20210601, 20210901)],'*')
    vacc24=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc25=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20211101, Epidata.range(20211101, 20220101)],'*')
    vacc26=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20220101, Epidata.range(20220101, end_week)],'*')
    #vacc32=Epidata.covidcast('fb-survey','smoothed_wearing_mask','day','state',[20210301, Epidata.range(20210301, end_week)],'*')
    vacc42=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_5d','day','state',[20210301, Epidata.range(20210301, end_week)],'*')
    vacc52=Epidata.covidcast('fb-survey','smoothed_wspent_time_1d','day','state',[20210301, Epidata.range(20210301, end_week)],'*')
    
    vacc1=vacc11['epidata']+vacc12['epidata']+vacc13['epidata']+vacc14['epidata']+vacc15['epidata']+vacc16['epidata']
    vacc2=vacc21['epidata']+vacc22['epidata']+vacc23['epidata']+vacc24['epidata']+vacc25['epidata']
    vacc3=vacc31['epidata']#+vacc32['epidata']
    vacc4=vacc41['epidata']+vacc42['epidata']
    vacc5=vacc51['epidata']+vacc52['epidata']
    
    print('smoothed_wcovid_vaccinated',vacc15['result'], vacc15['message'], len(vacc15['epidata']))
    print('smoothed_wcovid_vaccinated',vacc16['result'], vacc16['message'], len(vacc16['epidata']))
    print('smoothed_wtested_positive_14d',vacc24['result'], vacc24['message'], len(vacc24['epidata']))
    print('smoothed_wtested_positive_14d',vacc25['result'], vacc25['message'], len(vacc25['epidata']))
    print('smoothed_wwearing_mask',vacc31['result'], vacc31['message'], len(vacc31['epidata']))
    print('smoothed_wtravel_outside_state_5d',vacc42['result'], vacc42['message'], len(vacc42['epidata']))
    print('smoothed_wspent_time_1d',vacc52['result'], vacc52['message'], len(vacc52['epidata']))
    #print(vacc5['epidata'][0]['value'])
    
    state_names=list(state_index.keys())
    cols=['smoothed_wcovid_vaccinated','smoothed_wtested_positive_14d','smoothed_wwearing_mask',
          'smoothed_wtravel_outside_state_5d','smoothed_wspent_time_1d']
    week_cases={}
    for c in cols:
        week_cases[c]=np.zeros((len(state_names),len(epiweek_date)))
    
    week_cases[cols[0]]=read_survey_epidata(cols[0],vacc1,state_index,state_names,epiweek_date)
    week_cases[cols[1]]=read_survey_epidata(cols[1],vacc2,state_index,state_names,epiweek_date)
    week_cases[cols[2]]=read_survey_epidata(cols[2],vacc3,state_index,state_names,epiweek_date)
    week_cases[cols[3]]=read_survey_epidata(cols[3],vacc4,state_index,state_names,epiweek_date)
    week_cases[cols[4]]=read_survey_epidata(cols[4],vacc5,state_index,state_names,epiweek_date)
    
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test_date/vaccine_survey.csv")
    
    return week_cases,cols


# In[67]:


def read_fb(state_index,epiweek_date,start_week,end_week,start_week2):
    vacc = []
    vacc.append([])
    vacc.append([])
    vacc.append([])
    vacc.append([])
    vacc[0].append(Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[start_week, Epidata.range(start_week, 20210301)],'*'))
    vacc[0].append(Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[start_week, Epidata.range(start_week, 20210301)],'*'))
    vacc[1].append(Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[start_week, Epidata.range(start_week, 20210301)],'*'))
    vacc[1].append(Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[start_week, Epidata.range(start_week, 20210301)],'*'))
    vacc[2].append(Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','state',[20210208, Epidata.range(20210208, 20210301)],'*'))
    vacc[2].append(Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','nation',[20210208, Epidata.range(20210208, 20210301)],'*'))
    start_date = date(2021, 3, 1)
    while start_date < date.today() + relativedelta(months=-2):
        start = (int) (start_date.strftime("%Y%m%d"))
        start_date = start_date + relativedelta(months=+2)
        end = (int) (start_date.strftime("%Y%m%d"))
        vacc[0].append(Epidata.covidcast('fb-survey', 'smoothed_wcovid_vaccinated', 'day', 'state', [start, Epidata.range(start, end)], '*'))
        vacc[0].append(Epidata.covidcast('fb-survey', 'smoothed_wcovid_vaccinated', 'day', 'nation', [start, Epidata.range(start, end)], '*'))
        vacc[1].append(Epidata.covidcast('fb-survey', 'smoothed_wtested_positive_14d', 'day', 'state', [start, Epidata.range(start, end)], '*'))
        vacc[1].append(Epidata.covidcast('fb-survey', 'smoothed_wtested_positive_14d', 'day', 'nation', [start, Epidata.range(start, end)], '*'))
        vacc[2].append(Epidata.covidcast('fb-survey', 'smoothed_wwearing_mask_7d', 'day', 'state', [start, Epidata.range(start, end)], '*'))
        vacc[2].append(Epidata.covidcast('fb-survey', 'smoothed_wwearing_mask_7d', 'day', 'nation', [start, Epidata.range(start, end)], '*'))
        vacc[3].append(Epidata.covidcast('fb-survey', 'smoothed_wspent_time_indoors_1d', 'day', 'state', [start, Epidata.range(start, end)], '*'))
        vacc[3].append(Epidata.covidcast('fb-survey', 'smoothed_wspent_time_indoors_1d', 'day', 'nation', [start, Epidata.range(start, end)], '*'))
        print(start)
        print(end)
    vacc[0].append(Epidata.covidcast('fb-survey', 'smoothed_wcovid_vaccinated', 'day', 'state', [end, Epidata.range(end, end_week)], '*'))
    vacc[0].append(Epidata.covidcast('fb-survey', 'smoothed_wcovid_vaccinated', 'day', 'nation', [end, Epidata.range(end, end_week)], '*'))
    vacc[1].append(Epidata.covidcast('fb-survey', 'smoothed_wtested_positive_14d', 'day', 'state', [end, Epidata.range(end, end_week)], '*'))
    vacc[1].append(Epidata.covidcast('fb-survey', 'smoothed_wtested_positive_14d', 'day', 'nation', [end, Epidata.range(end, end_week)], '*'))
    vacc[2].append(Epidata.covidcast('fb-survey', 'smoothed_wwearing_mask_7d', 'day', 'state', [end, Epidata.range(end, end_week)], '*'))
    vacc[2].append(Epidata.covidcast('fb-survey', 'smoothed_wwearing_mask_7d', 'day', 'nation', [end, Epidata.range(end, end_week)], '*'))
    vacc[3].append(Epidata.covidcast('fb-survey', 'smoothed_wspent_time_indoors_1d', 'day', 'state', [end, Epidata.range(end, end_week)], '*'))
    vacc[3].append(Epidata.covidcast('fb-survey', 'smoothed_wspent_time_indoors_1d', 'day', 'nation', [end, Epidata.range(end, end_week)], '*'))
    
    vacc_final = []
    vacc_final.append(vacc[0][0]['epidata'])
    vacc_final.append(vacc[1][0]['epidata'])
    vacc_final.append(vacc[2][0]['epidata'])
    vacc_final.append(vacc[3][0]['epidata'])

    for i in range(len(vacc_final)):
        for j in range(1, len(vacc[i])): 
            try:
                vacc_final[i] += vacc[i][j]['epidata']
            except KeyError:
                print('key error epidata - vaccine')
                pass
    
    fb_survey = []
    fb_survey.append([])
    fb_survey.append([])
    fb_survey[0].append(Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [start_week2, Epidata.range(start_week2, 20200601)], '*'))
    fb_survey[0].append(Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation', [start_week2, Epidata.range(start_week2, 20200601)], '*'))
    fb_survey[0].append(Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20200601, Epidata.range(20200601, 20200701)], '*'))
    fb_survey[0].append(Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation', [20200601, Epidata.range(20200601, 20200701)], '*'))
    fb_survey[1].append(Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [start_week2, Epidata.range(start_week2, 20200601)], '*'))
    fb_survey[1].append(Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation', [start_week2, Epidata.range(start_week2, 20200601)], '*'))
    fb_survey[1].append(Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20200601, Epidata.range(20200601, 20200701)], '*'))
    fb_survey[1].append(Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation', [20200601, Epidata.range(20200601, 20200701)], '*'))
    start_date = date(2020, 7, 1)
    while start_date < date.today() + relativedelta(months=-2):
        start = (int) (start_date.strftime("%Y%m%d"))
        start_date = start_date + relativedelta(months=+2)
        end = (int) (start_date.strftime("%Y%m%d"))
        fb_survey[0].append(Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [start, Epidata.range(start, end)], '*'))
        fb_survey[0].append(Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation', [start, Epidata.range(start, end)], '*'))
        fb_survey[1].append(Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [start, Epidata.range(start, end)], '*'))
        fb_survey[1].append(Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation', [start, Epidata.range(start, end)], '*'))
        print(start)
        print(end)
    fb_survey[0].append(Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state',[end, Epidata.range(end, end_week)], '*'))
    fb_survey[0].append(Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation',[end, Epidata.range(end, end_week)], '*'))
    fb_survey[1].append(Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state',[end, Epidata.range(end, end_week)], '*'))
    fb_survey[1].append(Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation',[end, Epidata.range(end, end_week)], '*'))
    
    fb_survey_final = []
    fb_survey_final.append(fb_survey[0][0]['epidata'])
    fb_survey_final.append(fb_survey[1][0]['epidata'])
    for i in range(len(fb_survey_final)):
        for j in range(1, len(fb_survey[i])): 
            try:
                fb_survey_final[i] += fb_survey[i][j]['epidata']
            except KeyError:
                print('key error epidata - wcli and wili')
                pass
    
    print(len(vacc[0]))
    print(len(vacc[1]))
    print(len(vacc[2]))
    print(len(vacc[3]))
    print(len(fb_survey[0]))
    print(len(fb_survey[1]))
    
    #'''
    
    state_names=list(state_index.keys())
    cols=['smoothed_wcovid_vaccinated','smoothed_wtested_positive_14d','smoothed_wwearing_mask_7d'
          ,'smoothed_wspent_time_indoors_1d','fb_survey_wcli','fb_survey_wili']
    week_cases={}
    for c in cols:
        week_cases[c]=np.zeros((len(state_names),len(epiweek_date)))
    #'''
    week_cases[cols[0]]=read_survey_epidata(cols[0],vacc_final[0],state_index,state_names,epiweek_date)
    week_cases[cols[1]]=read_survey_epidata(cols[1],vacc_final[1],state_index,state_names,epiweek_date)
    week_cases[cols[2]]=read_survey_epidata(cols[2],vacc_final[2],state_index,state_names,epiweek_date)
    week_cases[cols[3]]=read_survey_epidata(cols[3],vacc_final[3],state_index,state_names,epiweek_date)
    week_cases[cols[4]]=read_survey_epidata(cols[4],fb_survey_final[0],state_index,state_names,epiweek_date)
    week_cases[cols[5]]=read_survey_epidata(cols[5],fb_survey_final[1],state_index,state_names,epiweek_date)
    
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/vaccine_survey.csv")
    #'''
    return week_cases,cols
    


# In[40]:


def read_delphi_vaccine_test_two(state_index,epiweek_date,start_week,end_week):
    #'''
    # The 
    vacc11=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc21=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc31=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','state',[20210208, Epidata.range(20210208, 20210301)],'*')
    ##vacc41=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_5d','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')
    ##vacc51=Epidata.covidcast('fb-survey','smoothed_wspent_time_1d','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')
    
    vacc11_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc21_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc31_us=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','nation',[20210208, Epidata.range(20210208, 20210301)],'*')
    ##vacc41_us=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_5d','day','nation',[start_week, Epidata.range(start_week, 20210301)],'*')
    ##vacc51_us=Epidata.covidcast('fb-survey','smoothed_wspent_time_1d','day','nation',[start_week, Epidata.range(start_week, 20210301)],'*')
    
    vacc12=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210301, Epidata.range(20210301, 20210501)],'*')
    vacc13=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210501, Epidata.range(20210501, 20210701)],'*')
    vacc14=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210701, Epidata.range(20210701, 20210901)],'*')
    vacc15=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc16=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20211101, Epidata.range(20211101, 20220101)],'*')
    vacc17=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20220101, Epidata.range(20220101, 20220301)],'*')
    vacc18=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20220301, Epidata.range(20220301, end_week)],'*')
    
    vacc12_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20210301, Epidata.range(20210301, 20210501)],'*')
    vacc13_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20210501, Epidata.range(20210501, 20210701)],'*')
    vacc14_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20210701, Epidata.range(20210701, 20210901)],'*')
    vacc15_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc16_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20211101, Epidata.range(20211101, 20220101)],'*')
    vacc17_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20220101, Epidata.range(20220101, 20220301)],'*')
    vacc18_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20220301, Epidata.range(20220101, end_week)],'*')
    
    
    vacc22=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210301, Epidata.range(20210301, 20210501)],'*')
    vacc23=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210501, Epidata.range(20210501, 20210701)],'*')
    vacc24=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210701, Epidata.range(20210701, 20210901)],'*')
    vacc25=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc26=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20211101, Epidata.range(20211101, 20220101)],'*')
    vacc27=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20220101, Epidata.range(20220101, 20220301)],'*')
    vacc28=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20220301, Epidata.range(20220301, end_week)],'*')
    
    vacc22_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20210301, Epidata.range(20210301, 20210501)],'*')
    vacc23_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20210501, Epidata.range(20210501, 20210701)],'*')
    vacc24_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20210701, Epidata.range(20210701, 20210901)],'*')
    vacc25_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc26_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20211101, Epidata.range(20211101, 20220101)],'*')
    vacc27_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20220101, Epidata.range(20220101, 20220301)],'*')
    vacc28_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20220301, Epidata.range(20220301, end_week)],'*')
    
    
    vacc32=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','state',[20210301, Epidata.range(20210301, 20210501)],'*')
    vacc33=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','state',[20210501, Epidata.range(20210501, 20210701)],'*')
    vacc34=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','state',[20210701, Epidata.range(20210701, 20210901)],'*')
    vacc35=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','state',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc36=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','state',[20211101, Epidata.range(20211101, 20220101)],'*')
    vacc37=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','state',[20220101, Epidata.range(20220101, 20220301)],'*')
    vacc38=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','state',[20220301, Epidata.range(20220301, end_week)],'*')
    
    vacc32_us=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','nation',[20210301, Epidata.range(20210301, 20210501)],'*')
    vacc33_us=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','nation',[20210501, Epidata.range(20210501, 20210701)],'*')
    vacc34_us=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','nation',[20210701, Epidata.range(20210701, 20210901)],'*')
    vacc35_us=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','nation',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc36_us=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','nation',[20211101, Epidata.range(20211101, 20220101)],'*')
    vacc37_us=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','nation',[20220101, Epidata.range(20220101, 20220301)],'*')
    vacc38_us=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','nation',[20220301, Epidata.range(20220301, end_week)],'*')
    
    
    vacc41=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','state',[20210302, Epidata.range(20210302, 20210501)],'*')
    vacc42=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','state',[20210501, Epidata.range(20210501, 20210701)],'*')
    vacc43=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','state',[20210701, Epidata.range(20210701, 20210901)],'*')
    vacc44=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','state',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc45=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','state',[20211101, Epidata.range(20211101, 20220101)],'*')
    vacc46=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','state',[20220101, Epidata.range(20220101, 20220301)],'*')
    #vacc47=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','state',[20220301, Epidata.range(20220301, end_week)],'*')
    
    vacc41_us=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','nation',[20210302, Epidata.range(20210302, 20210501)],'*')
    vacc42_us=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','nation',[20210501, Epidata.range(20210501, 20210701)],'*')
    vacc43_us=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','nation',[20210701, Epidata.range(20210701, 20210901)],'*')
    vacc44_us=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','nation',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc45_us=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','nation',[20211101, Epidata.range(20211101, 20220101)],'*')
    vacc46_us=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','nation',[20220101, Epidata.range(20220101, 20220301)],'*')
    #vacc47_us=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','nation',[20220301, Epidata.range(20220301, end_week)],'*')
    
    vacc51=Epidata.covidcast('fb-survey','smoothed_wspent_time_indoors_1d','day','state',[20210302, Epidata.range(20210302, 20210501)],'*')
    vacc52=Epidata.covidcast('fb-survey','smoothed_wspent_time_indoors_1d','day','state',[20210501, Epidata.range(20210501, 20210701)],'*')
    vacc53=Epidata.covidcast('fb-survey','smoothed_wspent_time_indoors_1d','day','state',[20210701, Epidata.range(20210701, 20210901)],'*')
    vacc54=Epidata.covidcast('fb-survey','smoothed_wspent_time_indoors_1d','day','state',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc55=Epidata.covidcast('fb-survey','smoothed_wspent_time_indoors_1d','day','state',[20211101, Epidata.range(20211101, 20220101)],'*')
    vacc56=Epidata.covidcast('fb-survey','smoothed_wspent_time_indoors_1d','day','state',[20220101, Epidata.range(20220101, 20220301)],'*')
    vacc57=Epidata.covidcast('fb-survey','smoothed_wspent_time_indoors_1d','day','state',[20220301, Epidata.range(20220301, end_week)],'*')
    
    vacc51_us=Epidata.covidcast('fb-survey','smoothed_wspent_time_indoors_1d','day','nation',[20210302, Epidata.range(20210302, 20210501)],'*')
    vacc52_us=Epidata.covidcast('fb-survey','smoothed_wspent_time_indoors_1d','day','nation',[20210501, Epidata.range(20210501, 20210701)],'*')
    vacc53_us=Epidata.covidcast('fb-survey','smoothed_wspent_time_indoors_1d','day','nation',[20210701, Epidata.range(20210701, 20210901)],'*')
    vacc54_us=Epidata.covidcast('fb-survey','smoothed_wspent_time_indoors_1d','day','nation',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc55_us=Epidata.covidcast('fb-survey','smoothed_wspent_time_indoors_1d','day','nation',[20211101, Epidata.range(20211101, 20220101)],'*')
    vacc56_us=Epidata.covidcast('fb-survey','smoothed_wspent_time_indoors_1d','day','nation',[20220101, Epidata.range(20220101, 20220301)],'*')
    vacc57_us=Epidata.covidcast('fb-survey','smoothed_wspent_time_indoors_1d','day','nation',[20220301, Epidata.range(20220301, end_week)],'*')
    
    #vacc32=Epidata.covidcast('fb-survey','smoothed_wearing_mask','day','state',[20210301, Epidata.range(20210301, end_week)],'*')
    ##vacc42=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_5d','day','state',[20210301, Epidata.range(20210301, end_week)],'*')
    ##vacc52=Epidata.covidcast('fb-survey','smoothed_wspent_time_1d','day','state',[20210301, Epidata.range(20210301, end_week)],'*')
    
    #vacc42_us=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_5d','day','nation',[20210301, Epidata.range(20210301, end_week)],'*')
    #vacc52_us=Epidata.covidcast('fb-survey','smoothed_wspent_time_1d','day','nation',[20210301, Epidata.range(20210301, end_week)],'*')
    
    vacc1=vacc11['epidata']+vacc12['epidata']+vacc13['epidata']+vacc14['epidata']+vacc15['epidata']+vacc16['epidata']+vacc17['epidata']+vacc18['epidata']
    vacc2=vacc21['epidata']+vacc22['epidata']+vacc23['epidata']+vacc24['epidata']+vacc25['epidata']+vacc26['epidata']+vacc27['epidata']+vacc28['epidata']
    vacc3=vacc31['epidata']+vacc32['epidata']+vacc33['epidata']+vacc34['epidata']+vacc35['epidata']+vacc36['epidata']+vacc37['epidata']+vacc38['epidata']
    vacc4=vacc41['epidata']+vacc42['epidata']+vacc43['epidata']+vacc44['epidata']+vacc45['epidata']+vacc46['epidata']#+vacc47['epidata']
    vacc5=vacc51['epidata']+vacc52['epidata']+vacc53['epidata']+vacc54['epidata']+vacc55['epidata']+vacc56['epidata']+vacc57['epidata']
    
    vacc1_us=vacc11_us['epidata']+vacc12_us['epidata']+vacc13_us['epidata']+vacc14_us['epidata']+vacc15_us['epidata']+vacc16_us['epidata']+vacc17_us['epidata']+vacc18_us['epidata']
    vacc2_us=vacc21_us['epidata']+vacc22_us['epidata']+vacc23_us['epidata']+vacc24_us['epidata']+vacc25_us['epidata']+vacc26_us['epidata']+vacc27_us['epidata']+vacc28_us['epidata']
    vacc3_us=vacc31_us['epidata']+vacc32_us['epidata']+vacc33_us['epidata']+vacc34_us['epidata']+vacc35_us['epidata']+vacc36_us['epidata']+vacc37_us['epidata']+vacc38_us['epidata']
    vacc4_us=vacc41_us['epidata']+vacc42_us['epidata']+vacc43_us['epidata']+vacc44_us['epidata']+vacc45_us['epidata']+vacc46_us['epidata']#+vacc47_us['epidata']
    vacc5_us=vacc51_us['epidata']+vacc52_us['epidata']+vacc53_us['epidata']+vacc54_us['epidata']+vacc55_us['epidata']+vacc56_us['epidata']+vacc57_us['epidata']
    
    vacc1_final = vacc1 + vacc1_us
    vacc2_final = vacc2 + vacc2_us
    vacc3_final = vacc3 + vacc3_us
    vacc4_final = vacc4 + vacc4_us
    vacc5_final = vacc5 + vacc5_us
    
    print('smoothed_wcovid_vaccinated',vacc18['result'], vacc18['message'], len(vacc18['epidata']))
    print('smoothed_wtested_positive_14d',vacc28['result'], vacc28['message'], len(vacc28['epidata']))
    print('smoothed_wwearing_mask',vacc38['result'], vacc38['message'], len(vacc38['epidata']))
    print('smoothed_wtravel_outside_state_5d',vacc46['result'], vacc46['message'], len(vacc46['epidata']))
    print('smoothed_wspent_time_1d',vacc57['result'], vacc57['message'], len(vacc57['epidata']))
    #'''
    
    state_names=list(state_index.keys())
    cols=['smoothed_wcovid_vaccinated','smoothed_wtested_positive_14d','smoothed_wwearing_mask_7d',
          'smoothed_wtravel_outside_state_7d','smoothed_wspent_time_indoors_1d']
    week_cases={}
    for c in cols:
        week_cases[c]=np.zeros((len(state_names),len(epiweek_date)))
    #'''
    week_cases[cols[0]]=read_survey_epidata(cols[0],vacc1_final,state_index,state_names,epiweek_date)
    week_cases[cols[1]]=read_survey_epidata(cols[1],vacc2_final,state_index,state_names,epiweek_date)
    week_cases[cols[2]]=read_survey_epidata(cols[2],vacc3_final,state_index,state_names,epiweek_date)
    week_cases[cols[3]]=read_survey_epidata(cols[3],vacc4_final,state_index,state_names,epiweek_date)
    week_cases[cols[4]]=read_survey_epidata(cols[4],vacc5_final,state_index,state_names,epiweek_date)
    
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/vaccine_survey.csv")
    #'''
    return week_cases,cols


# In[41]:


def read_delphi_vaccine_test(state_index,epiweek_date,start_week,end_week):
    #'''
    # The 
    vacc11=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc21=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc31=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','state',[20210208, Epidata.range(20210208, end_week)],'*')
    vacc41=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','state',[20210308, Epidata.range(20210308, end_week)],'*')
    vacc51=Epidata.covidcast('fb-survey','smoothed_wspent_time_indoors_1d','day','state',[20210302, Epidata.range(20210302, end_week)],'*')
    
    vacc11_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc21_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc31_us=Epidata.covidcast('fb-survey','smoothed_wwearing_mask_7d','day','nation',[20210208, Epidata.range(20210208, end_week)],'*')
    vacc41_us=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_7d','day','nation',[20210308, Epidata.range(20210308, end_week)],'*')
    vacc51_us=Epidata.covidcast('fb-survey','smoothed_wspent_time_indoors_1d','day','nation',[20210302, Epidata.range(20210302, end_week)],'*')
    
    vacc12=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210301, Epidata.range(20210301, 20210501)],'*')
    vacc13=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210501, Epidata.range(20210501, 20210701)],'*')
    vacc14=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210701, Epidata.range(20210701, 20210901)],'*')
    vacc15=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc16=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20211101, Epidata.range(20211101, 20220101)],'*')
    vacc17=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20220101, Epidata.range(20220101, end_week)],'*')
    
    vacc12_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20210301, Epidata.range(20210301, 20210501)],'*')
    vacc13_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20210501, Epidata.range(20210501, 20210701)],'*')
    vacc14_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20210701, Epidata.range(20210701, 20210901)],'*')
    vacc15_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc16_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20211101, Epidata.range(20211101, 20220101)],'*')
    vacc17_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20220101, Epidata.range(20220101, end_week)],'*')
    
    vacc22=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210301, Epidata.range(20210301, 20210601)],'*')
    vacc23=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210601, Epidata.range(20210601, 20210901)],'*')
    vacc24=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210601, Epidata.range(20210901, 20211101)],'*')
    vacc25=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210601, Epidata.range(20211101, 20220101)],'*')
    vacc26=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20220101, Epidata.range(20220101, end_week)],'*')
    
    vacc22_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20210301, Epidata.range(20210301, 20210601)],'*')
    vacc23_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20210601, Epidata.range(20210601, 20210901)],'*')
    vacc24_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20210601, Epidata.range(20210901, 20211101)],'*')
    vacc25_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20210601, Epidata.range(20211101, 20220101)],'*')
    vacc26_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20220101, Epidata.range(20220101, end_week)],'*')
    
    #vacc32=Epidata.covidcast('fb-survey','smoothed_wearing_mask','day','state',[20210301, Epidata.range(20210301, end_week)],'*')
    """vacc42=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_5d','day','state',[20210301, Epidata.range(20210301, end_week)],'*')
    vacc52=Epidata.covidcast('fb-survey','smoothed_wspent_time_1d','day','state',[20210301, Epidata.range(20210301, end_week)],'*')
    
    vacc42_us=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_5d','day','nation',[20210301, Epidata.range(20210301, end_week)],'*')
    vacc52_us=Epidata.covidcast('fb-survey','smoothed_wspent_time_1d','day','nation',[20210301, Epidata.range(20210301, end_week)],'*')"""
    
    vacc1=vacc11['epidata']+vacc12['epidata']+vacc13['epidata']+vacc14['epidata']+vacc15['epidata']+vacc16['epidata']+vacc17['epidata']
    vacc2=vacc21['epidata']+vacc22['epidata']+vacc23['epidata']+vacc24['epidata']+vacc25['epidata']+vacc26['epidata']
    vacc3=vacc31['epidata']#+vacc32['epidata']
    vacc4=vacc41['epidata']#+vacc42['epidata']
    vacc5=vacc51['epidata']#+vacc52['epidata']
    
    vacc1_us=vacc11_us['epidata']+vacc12_us['epidata']+vacc13_us['epidata']+vacc14_us['epidata']+vacc15_us['epidata']+vacc16_us['epidata']+vacc17_us['epidata']
    vacc2_us=vacc21_us['epidata']+vacc22_us['epidata']+vacc23_us['epidata']+vacc24_us['epidata']+vacc25_us['epidata']+vacc26_us['epidata']
    vacc3_us=vacc31_us['epidata']#+vacc32['epidata']
    vacc4_us=vacc41_us['epidata']#+vacc42_us['epidata']
    vacc5_us=vacc51_us['epidata']#+vacc52_us['epidata']
    
    vacc1_final = vacc1 + vacc1_us
    vacc2_final = vacc2 + vacc2_us
    vacc3_final = vacc3 + vacc3_us
    vacc4_final = vacc4 + vacc4_us
    vacc5_final = vacc5 + vacc5_us
    
    """
    print('smoothed_wcovid_vaccinated',vacc17['result'], vacc17['message'], len(vacc17['epidata']))
    print('smoothed_wtested_positive_14d',vacc26['result'], vacc26['message'], len(vacc26['epidata']))
    print('smoothed_wwearing_mask',vacc31['result'], vacc31['message'], len(vacc31['epidata']))
    print('smoothed_wtravel_outside_state_5d',vacc42['result'], vacc42['message'], len(vacc42['epidata']))
    print('smoothed_wspent_time_1d',vacc52['result'], vacc52['message'], len(vacc52['epidata']))"""
    #'''
    
    state_names=list(state_index.keys())
    cols=['smoothed_wcovid_vaccinated','smoothed_wtested_positive_14d','smoothed_wwearing_mask',
          'smoothed_wtravel_outside_state_5d','smoothed_wspent_time_1d']
    week_cases={}
    for c in cols:
        week_cases[c]=np.full((len(state_names),len(epiweek_date)), np.nan)
    #'''
    week_cases[cols[0]]=read_survey_epidata(cols[0],vacc1_final,state_index,state_names,epiweek_date)
    week_cases[cols[1]]=read_survey_epidata(cols[1],vacc2_final,state_index,state_names,epiweek_date)
    week_cases[cols[2]]=read_survey_epidata(cols[2],vacc3_final,state_index,state_names,epiweek_date)
    week_cases[cols[3]]=read_survey_epidata(cols[3],vacc4_final,state_index,state_names,epiweek_date)
    week_cases[cols[4]]=read_survey_epidata(cols[4],vacc5_final,state_index,state_names,epiweek_date)
    
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/vaccine_survey.csv")
    #'''
    return week_cases,cols


# In[42]:


def read_delphi(state_index, epiweek_date, start_week, end_week): 
    #'''
    def read_delphi_vaccine_test(state_index,epiweek_date,start_week,end_week):
        return 
    # The 
    vacc11=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc21=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc31=Epidata.covidcast('fb-survey','smoothed_wwearing_mask','day','state',[start_week, Epidata.range(start_week, end_week)],'*')
    vacc41=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_5d','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc51=Epidata.covidcast('fb-survey','smoothed_wspent_time_1d','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')
    
    vacc11_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc21_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc31_us=Epidata.covidcast('fb-survey','smoothed_wwearing_mask','day','nation',[start_week, Epidata.range(start_week, end_week)],'*')
    vacc41_us=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_5d','day','nation',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc51_us=Epidata.covidcast('fb-survey','smoothed_wspent_time_1d','day','nation',[start_week, Epidata.range(start_week, 20210301)],'*')
    
    vacc12=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210301, Epidata.range(20210301, 20210501)],'*')
    vacc13=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210501, Epidata.range(20210501, 20210701)],'*')
    vacc14=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210701, Epidata.range(20210701, 20210901)],'*')
    vacc15=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc16=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20211101, Epidata.range(20211101, 20220101)],'*')
    vacc17=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20220101, Epidata.range(20220101, end_week)],'*')
    
    vacc12_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20210301, Epidata.range(20210301, 20210501)],'*')
    vacc13_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20210501, Epidata.range(20210501, 20210701)],'*')
    vacc14_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20210701, Epidata.range(20210701, 20210901)],'*')
    vacc15_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc16_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20211101, Epidata.range(20211101, 20220101)],'*')
    vacc17_us=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','nation',[20220101, Epidata.range(20220101, end_week)],'*')
    
    vacc22=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210301, Epidata.range(20210301, 20210601)],'*')
    vacc23=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210601, Epidata.range(20210601, 20210901)],'*')
    vacc24=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210601, Epidata.range(20210901, 20211101)],'*')
    vacc25=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210601, Epidata.range(20211101, 20220101)],'*')
    vacc26=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20220101, Epidata.range(20220101, end_week)],'*')
    
    vacc22_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20210301, Epidata.range(20210301, 20210601)],'*')
    vacc23_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20210601, Epidata.range(20210601, 20210901)],'*')
    vacc24_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20210601, Epidata.range(20210901, 20211101)],'*')
    vacc25_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20210601, Epidata.range(20211101, 20220101)],'*')
    vacc26_us=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','nation',[20220101, Epidata.range(20220101, end_week)],'*')
    
    #vacc32=Epidata.covidcast('fb-survey','smoothed_wearing_mask','day','state',[20210301, Epidata.range(20210301, end_week)],'*')
    vacc42=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_5d','day','state',[20210301, Epidata.range(20210301, end_week)],'*')
    vacc52=Epidata.covidcast('fb-survey','smoothed_wspent_time_1d','day','state',[20210301, Epidata.range(20210301, end_week)],'*')
    
    vacc42_us=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_5d','day','nation',[20210301, Epidata.range(20210301, end_week)],'*')
    vacc52_us=Epidata.covidcast('fb-survey','smoothed_wspent_time_1d','day','nation',[20210301, Epidata.range(20210301, end_week)],'*')
    
    vacc1=vacc11['epidata']+vacc12['epidata']+vacc13['epidata']+vacc14['epidata']+vacc15['epidata']+vacc16['epidata']+vacc17['epidata']
    vacc2=vacc21['epidata']+vacc22['epidata']+vacc23['epidata']+vacc24['epidata']+vacc25['epidata']+vacc26['epidata']
    vacc3=vacc31['epidata']#+vacc32['epidata']
    vacc4=vacc41['epidata']+vacc42['epidata']
    vacc5=vacc51['epidata']+vacc52['epidata']
    
    vacc1_us=vacc11_us['epidata']+vacc12_us['epidata']+vacc13_us['epidata']+vacc14_us['epidata']+vacc15_us['epidata']+vacc16_us['epidata']+vacc17_us['epidata']
    vacc2_us=vacc21_us['epidata']+vacc22_us['epidata']+vacc23_us['epidata']+vacc24_us['epidata']+vacc25_us['epidata']+vacc26_us['epidata']
    vacc3_us=vacc31_us['epidata']#+vacc32['epidata']
    vacc4_us=vacc41_us['epidata']+vacc42_us['epidata']
    vacc5_us=vacc51_us['epidata']+vacc52_us['epidata']
    
    vacc1_final = vacc1 + vacc1_us
    vacc2_final = vacc2 + vacc2_us
    vacc3_final = vacc3 + vacc3_us
    vacc4_final = vacc4 + vacc4_us
    vacc5_final = vacc5 + vacc5_us
    
    print('smoothed_wcovid_vaccinated',vacc17['result'], vacc17['message'], len(vacc17['epidata']))
    print('smoothed_wtested_positive_14d',vacc26['result'], vacc26['message'], len(vacc26['epidata']))
    print('smoothed_wwearing_mask',vacc31['result'], vacc31['message'], len(vacc31['epidata']))
    print('smoothed_wtravel_outside_state_5d',vacc42['result'], vacc42['message'], len(vacc42['epidata']))
    print('smoothed_wspent_time_1d',vacc52['result'], vacc52['message'], len(vacc52['epidata']))
    #'''
    
    state_names=list(state_index.keys())
    cols=['smoothed_wcovid_vaccinated','smoothed_wtested_positive_14d','smoothed_wwearing_mask',
          'smoothed_wtravel_outside_state_5d','smoothed_wspent_time_1d']
    week_cases={}
    for c in cols:
        week_cases[c]=np.full((len(state_names),len(epiweek_date)), np.nan)
    #'''
    week_cases[cols[0]]=read_survey_epidata(cols[0],vacc1_final,state_index,state_names,epiweek_date)
    week_cases[cols[1]]=read_survey_epidata(cols[1],vacc2_final,state_index,state_names,epiweek_date)
    week_cases[cols[2]]=read_survey_epidata(cols[2],vacc3_final,state_index,state_names,epiweek_date)
    week_cases[cols[3]]=read_survey_epidata(cols[3],vacc4_final,state_index,state_names,epiweek_date)
    week_cases[cols[4]]=read_survey_epidata(cols[4],vacc5_final,state_index,state_names,epiweek_date)
    
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/vaccine_survey.csv")
    #'''
    return week_cases,cols


# In[43]:


def read_delphi_fb_google_survey(state_index,epiweek_date,start_week,end_week):
    #fb_res_cli = Epidata.covidcast('fb-survey', 'raw_cli', 'day', 'state', [start_week, Epidata.range(start_week, end_week)], '*')
    fb_res_cli1 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [start_week, Epidata.range(start_week, 20200601)], '*')
    fb_res_cli2 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20200601, Epidata.range(20200601, 20200701)], '*')
    fb_res_cli3 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20200701, Epidata.range(20200701, 20200901)], '*')
    fb_res_cli4 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20200901, Epidata.range(20200901, 20201101)], '*')
    fb_res_cli5 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20201101, Epidata.range(20201101, 20210101)], '*')
    fb_res_cli6 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20210101, Epidata.range(20210101, 20210301)], '*')
    fb_res_cli7 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20210301, Epidata.range(20210301, 20210501)], '*')
    fb_res_cli8 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20210501, Epidata.range(20210501, 20210701)], '*')
    fb_res_cli9 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20210701, Epidata.range(20210701, 20210901)], '*')
    fb_res_cli10 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20210901, Epidata.range(20210901, 20211101)], '*')
    fb_res_cli11 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20211101, Epidata.range(20211101, 20220101)], '*')
    fb_res_cli12 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20220101, Epidata.range(20220101, end_week)], '*')

    fb_cli1=fb_res_cli1['epidata']+fb_res_cli2['epidata']
    fb_cli2=fb_cli1+fb_res_cli3['epidata']
    fb_cli3=fb_cli2+fb_res_cli4['epidata']
    #fb_res_cli=fb_cli3+fb_res_cli5['epidata']
    fb_cli4=fb_cli3+fb_res_cli5['epidata'] 
    fb_cli5=fb_cli4+fb_res_cli6['epidata'] 
    fb_cli6=fb_cli5+fb_res_cli7['epidata'] 
    fb_cli7=fb_cli6+fb_res_cli8['epidata'] 
    fb_cli8=fb_cli7+fb_res_cli9['epidata']
    fb_cli9=fb_cli8+fb_res_cli10['epidata']
    fb_cli10=fb_cli8+fb_res_cli10['epidata']
    fb_res_cli=fb_cli10+fb_res_cli12['epidata']

    
    #google_res_cli = Epidata.covidcast('google-survey', 'raw_cli', 'day', 'state', [start_week, Epidata.range(start_week, end_week)], '*')
    
    #fb_res_wli = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [start_week, Epidata.range(start_week, end_week)], '*')
    fb_res_wli1 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [start_week, Epidata.range(start_week, 20200601)], '*')
    fb_res_wli2 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20200601, Epidata.range(20200601, 20200701)], '*')
    fb_res_wli3 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20200701, Epidata.range(20200701, 20200901)], '*')
    fb_res_wli4 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20200901, Epidata.range(20200901, 20201101)], '*')
    fb_res_wli5 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20201101, Epidata.range(20201101, 20210101)], '*')
    fb_res_wli6 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20210101, Epidata.range(20210101, 20210301)], '*')
    fb_res_wli7 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20210301, Epidata.range(20210301, 20210501)], '*')
    fb_res_wli8 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20210501, Epidata.range(20210501, 20210701)], '*')
    fb_res_wli9 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20210701, Epidata.range(20210701, 20210901)], '*')
    fb_res_wli10 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20210901, Epidata.range(20210901, 20211101)], '*')
    fb_res_wli11 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20211101, Epidata.range(20211101, 20220101)], '*')
    fb_res_wli12 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20220101, Epidata.range(20211101, end_week)], '*')

    
    fb_wli1=fb_res_wli1['epidata']+fb_res_wli2['epidata']
    fb_wli2=fb_wli1+fb_res_wli3['epidata']
    fb_wli3=fb_wli2+fb_res_wli4['epidata']
    fb_wli4=fb_wli3+fb_res_wli5['epidata']
    fb_wli5=fb_wli4+fb_res_wli6['epidata']
    fb_wli6=fb_wli5+fb_res_wli7['epidata']
    fb_wli7=fb_wli6+fb_res_wli8['epidata']
    fb_wli8=fb_wli7+fb_res_wli9['epidata']
    fb_wli9=fb_wli8+fb_res_wli10['epidata']
    fb_wli10=fb_wli9+fb_res_wli10['epidata']
    fb_res_wli=fb_wli10+fb_res_wli12['epidata']
    
    
    #google_res_wli = Epidata.covidcast('google-survey', 'raw_wili', 'day', 'state', [start_week, Epidata.range(start_week, end_week)], '*')
    
    #print('fb_cli1',fb_res_cli1['result'], fb_res_cli1['message'], len(fb_res_cli1['epidata']))
    #print('fb_cli2',fb_res_cli2['result'], fb_res_cli2['message'], len(fb_res_cli2['epidata']))
    #print('fb_cli3',fb_res_cli3['result'], fb_res_cli3['message'], len(fb_res_cli3['epidata']))
    print('fb_cli4',fb_res_cli3['result'], fb_res_cli4['message'], len(fb_res_cli4['epidata']))
    print('fb_cli5',fb_res_cli5['result'], fb_res_cli5['message'], len(fb_res_cli5['epidata']))
    print('fb_cli6',fb_res_cli6['result'], fb_res_cli6['message'], len(fb_res_cli6['epidata']))
    print('fb_cli8',fb_res_cli8['result'], fb_res_cli8['message'], len(fb_res_cli8['epidata']))
    print('fb_cli9',fb_res_cli9['result'], fb_res_cli9['message'], len(fb_res_cli9['epidata']))
    print('fb_cli11',fb_res_cli10['result'], fb_res_cli10['message'], len(fb_res_cli11['epidata']))
    print('fb_cli12',fb_res_cli11['result'], fb_res_cli11['message'], len(fb_res_cli12['epidata']))

    
    #print(google_res_cli['result'], google_res_cli['message'], len(google_res_cli['epidata']))
    
    #print(fb_res_wli1['result'], fb_res_wli1['message'], len(fb_res_wli1['epidata']))
    #print(fb_res_wli2['result'], fb_res_wli2['message'], len(fb_res_wli2['epidata']))
    #print(fb_res_wli3['result'], fb_res_wli3['message'], len(fb_res_wli3['epidata']))
    print('fb_wili4',fb_res_wli4['result'], fb_res_wli4['message'], len(fb_res_wli4['epidata']))
    print('fb_wili5',fb_res_wli5['result'], fb_res_wli5['message'], len(fb_res_wli5['epidata']))
    print('fb_wili6',fb_res_wli6['result'], fb_res_wli6['message'], len(fb_res_wli6['epidata']))
    print('fb_wili8',fb_res_wli8['result'], fb_res_wli8['message'], len(fb_res_wli8['epidata']))
    print('fb_wili9',fb_res_wli9['result'], fb_res_wli9['message'], len(fb_res_wli9['epidata']))
    print('fb_wili11',fb_res_wli10['result'], fb_res_wli10['message'], len(fb_res_wli11['epidata']))
    print('fb_wili12',fb_res_wli11['result'], fb_res_wli11['message'], len(fb_res_wli12['epidata']))

    
    print('fb_cli len',len(fb_res_cli))
    print('fb_wli len',len(fb_res_wli))
    #print(google_res_wli['result'], google_res_wli['message'], len(google_res_wli['epidata']))
    
    state_names=list(state_index.keys())
    cols=['fb_survey_wcli','fb_survey_wili']
    week_cases={}
    for c in cols:
        week_cases[c]=np.full((len(state_names),len(epiweek_date)), np.nan)
    
    #week_cases[cols[0]]=read_survey_epidata(cols[0],fb_res_cli['epidata'],state_index,state_names,epiweek_date)
    week_cases[cols[0]]=read_survey_epidata(cols[0],fb_res_cli,state_index,state_names,epiweek_date)
    #week_cases[cols[1]]=read_survey_epidata(cols[1],google_res_cli['epidata'],state_index,state_names,epiweek_date)
    #week_cases[cols[2]]=read_survey_epidata(cols[2],fb_res_wli['epidata'],state_index,state_names,epiweek_date)
    week_cases[cols[1]]=read_survey_epidata(cols[1],fb_res_wli,state_index,state_names,epiweek_date)
    #week_cases[cols[3]]=read_survey_epidata(cols[3],google_res_wli['epidata'],state_index,state_names,epiweek_date)
    
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test_date/fb-google-survey.csv")
    
    return week_cases,cols


# In[44]:


def read_delphi_fb_google_survey_test(state_index,epiweek_date,start_week,end_week):
    fb_res_cli1 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [start_week, Epidata.range(start_week, 20200601)], '*')
    fb_res_cli2 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20200601, Epidata.range(20200601, 20200701)], '*')
    fb_res_cli3 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20200701, Epidata.range(20200701, 20200901)], '*')
    fb_res_cli4 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20200901, Epidata.range(20200901, 20201101)], '*')
    fb_res_cli5 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20201101, Epidata.range(20201101, 20210101)], '*')
    fb_res_cli6 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20210101, Epidata.range(20210101, 20210301)], '*')
    fb_res_cli7 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20210301, Epidata.range(20210301, 20210501)], '*')
    fb_res_cli8 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20210501, Epidata.range(20210501, 20210701)], '*')
    fb_res_cli9 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20210701, Epidata.range(20210701, 20210901)], '*')
    fb_res_cli10 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20210901, Epidata.range(20210901, 20211101)], '*')
    fb_res_cli11 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20211101, Epidata.range(20211101, 20220101)], '*')
    fb_res_cli12 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20220101, Epidata.range(20220101, 20220301)], '*')
    fb_res_cli13 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20220301, Epidata.range(20220301, end_week)], '*')

    fb_res_cli1_us = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation', [start_week, Epidata.range(start_week, 20200601)], '*')
    fb_res_cli2_us = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation', [20200601, Epidata.range(20200601, 20200701)], '*')
    fb_res_cli3_us = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation', [20200701, Epidata.range(20200701, 20200901)], '*')
    fb_res_cli4_us = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation', [20200901, Epidata.range(20200901, 20201101)], '*')
    fb_res_cli5_us = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation', [20201101, Epidata.range(20201101, 20210101)], '*')
    fb_res_cli6_us = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation', [20210101, Epidata.range(20210101, 20210301)], '*')
    fb_res_cli7_us = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation', [20210301, Epidata.range(20210301, 20210501)], '*')
    fb_res_cli8_us = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation', [20210501, Epidata.range(20210501, 20210701)], '*')
    fb_res_cli9_us = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation', [20210701, Epidata.range(20210701, 20210901)], '*')
    fb_res_cli10_us = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation', [20210901, Epidata.range(20210901, 20211101)], '*')
    fb_res_cli11_us = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation', [20211101, Epidata.range(20211101, 20220101)], '*')
    fb_res_cli12_us = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation', [20220101, Epidata.range(20220101, 20220301)], '*')
    fb_res_cli13_us = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'nation', [20220301, Epidata.range(20220301, end_week)], '*')

    #'''
    fb_cli1=fb_res_cli1['epidata']+fb_res_cli2['epidata']
    fb_cli2=fb_cli1+fb_res_cli3['epidata']
    fb_cli3=fb_cli2+fb_res_cli4['epidata']
    #fb_res_cli=fb_cli3+fb_res_cli5['epidata']
    fb_cli4=fb_cli3+fb_res_cli5['epidata'] 
    fb_cli5=fb_cli4+fb_res_cli6['epidata']
    fb_cli6=fb_cli5+fb_res_cli7['epidata']
    fb_cli7=fb_cli6+fb_res_cli8['epidata']
    fb_cli8 = fb_cli7 + fb_res_cli9['epidata']
    fb_cli9=fb_cli8 + fb_res_cli10['epidata']
    fb_res_cli=fb_cli9+fb_res_cli11['epidata']+fb_res_cli12['epidata']+fb_res_cli13['epidata']
    
    fb_cli1_us = fb_res_cli1_us['epidata'] + fb_res_cli2_us['epidata']
    fb_cli2_us=fb_cli1_us+fb_res_cli3_us['epidata']
    fb_cli3_us=fb_cli2_us+fb_res_cli4_us['epidata']
    #fb_res_cli=fb_cli3+fb_res_cli5['epidata']
    fb_cli4_us=fb_cli3_us+fb_res_cli5_us['epidata'] 
    fb_cli5_us=fb_cli4_us+fb_res_cli6_us['epidata']
    fb_cli6_us=fb_cli5_us+fb_res_cli7_us['epidata']
    fb_cli7_us=fb_cli6_us+fb_res_cli8_us['epidata']
    fb_cli8_us = fb_cli7_us + fb_res_cli9_us['epidata']
    fb_cli9_us=fb_cli8_us + fb_res_cli10_us['epidata']
    fb_res_cli_us=fb_cli9_us+fb_res_cli11_us['epidata']+fb_res_cli12_us['epidata']+fb_res_cli13_us['epidata']
    
    fb_res_cli_final = fb_res_cli + fb_res_cli_us
    #print(fb_res1['epidata'][0])
    #'''
    #google_res_cli = Epidata.covidcast('google-survey', 'raw_cli', 'day', 'state', [start_week, Epidata.range(start_week, end_week)], '*')
    #google_res_cli_us = Epidata.covidcast('google-survey', 'raw_cli', 'day', 'nation', [start_week, Epidata.range(start_week, end_week)], '*')
    
    #fb_res_wli = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [start_week, Epidata.range(start_week, end_week)], '*')
    fb_res_wli1 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [start_week, Epidata.range(start_week, 20200601)], '*')
    fb_res_wli2 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20200601, Epidata.range(20200601, 20200701)], '*')
    fb_res_wli3 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20200701, Epidata.range(20200701, 20200901)], '*')
    fb_res_wli4 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20200901, Epidata.range(20200901, 20201101)], '*')
    fb_res_wli5 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20201101, Epidata.range(20201101, 20210101)], '*')
    fb_res_wli6 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20210101, Epidata.range(20210101, 20210301)], '*')
    fb_res_wli7 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20210301, Epidata.range(20210301, 20210501)], '*')
    fb_res_wli8 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20210501, Epidata.range(20210501, 20210701)], '*')
    fb_res_wli9 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20210701, Epidata.range(20210701, 20210901)], '*')
    fb_res_wli10 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20210901, Epidata.range(20210901, 20211101)], '*')
    fb_res_wli11 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20211101, Epidata.range(20211101, 20220101)], '*')
    fb_res_wli12 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20220101, Epidata.range(20220101, 20220301)], '*')
    fb_res_wli13 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20220301, Epidata.range(20220301, end_week)], '*')
    
    fb_res_wli1_us = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation', [start_week, Epidata.range(start_week, 20200601)], '*')
    fb_res_wli2_us = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation', [20200601, Epidata.range(20200601, 20200701)], '*')
    fb_res_wli3_us = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation', [20200701, Epidata.range(20200701, 20200901)], '*')
    fb_res_wli4_us = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation', [20200901, Epidata.range(20200901, 20201101)], '*')
    fb_res_wli5_us = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation', [20201101, Epidata.range(20201101, 20210101)], '*')
    fb_res_wli6_us = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation', [20210101, Epidata.range(20210101, 20210301)], '*')
    fb_res_wli7_us = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation', [20210301, Epidata.range(20210301, 20210501)], '*')
    fb_res_wli8_us = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation', [20210501, Epidata.range(20210501, 20210701)], '*')
    fb_res_wli9_us = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation', [20210701, Epidata.range(20210701, 20210901)], '*')
    fb_res_wli10_us = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation', [20210901, Epidata.range(20210901, 20211101)], '*')
    fb_res_wli11_us = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation', [20211101, Epidata.range(20211101, 20220101)], '*')
    fb_res_wli12_us = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation', [20220101, Epidata.range(20220101, 20220301)], '*')
    fb_res_wli13_us = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'nation', [20220301, Epidata.range(20220101, end_week)], '*')
    #'''
    fb_wli1=fb_res_wli1['epidata']+fb_res_wli2['epidata']
    fb_wli2=fb_wli1+fb_res_wli3['epidata']
    fb_wli3=fb_wli2+fb_res_wli4['epidata']
    fb_wli4=fb_wli3+fb_res_wli5['epidata']
    fb_wli5=fb_wli4+fb_res_wli6['epidata']
    fb_wli6=fb_wli5+fb_res_wli7['epidata']
    fb_wli7=fb_wli6+fb_res_wli8['epidata']
    fb_wli8=fb_wli7+fb_res_wli9['epidata']
    fb_wli9=fb_wli8+fb_res_wli10['epidata']
    #fb_res_wli=fb_wli3+fb_res_wli5['epidata']
    fb_res_wli=fb_wli9+fb_res_wli11['epidata']+fb_res_wli12['epidata']+fb_res_wli13['epidata']
    
    fb_wli1_us=fb_res_wli1_us['epidata']+fb_res_wli2_us['epidata']
    fb_wli2_us=fb_wli1_us+fb_res_wli3_us['epidata']
    fb_wli3_us=fb_wli2_us+fb_res_wli4_us['epidata']
    fb_wli4_us=fb_wli3_us+fb_res_wli5_us['epidata']
    fb_wli5_us=fb_wli4_us+fb_res_wli6_us['epidata']
    fb_wli6_us=fb_wli5_us+fb_res_wli7_us['epidata']
    fb_wli7_us=fb_wli6_us+fb_res_wli8_us['epidata']
    fb_wli8_us=fb_wli7_us+fb_res_wli9_us['epidata']
    fb_wli9_us=fb_wli8_us+fb_res_wli10_us['epidata']
    #fb_res_wli=fb_wli3+fb_res_wli5['epidata']
    fb_res_wli_us=fb_wli9_us+fb_res_wli11_us['epidata']+fb_res_wli12_us['epidata']+fb_res_wli13_us['epidata']
    
    fb_res_wli_final = fb_res_wli + fb_res_wli_us
    
    #google_res_wli = Epidata.covidcast('google-survey', 'raw_wili', 'day', 'state', [start_week, Epidata.range(start_week, end_week)], '*')
    
    #print('fb_cli1',fb_res_cli1['result'], fb_res_cli1['message'], len(fb_res_cli1['epidata']))
    #print('fb_cli2',fb_res_cli2['result'], fb_res_cli2['message'], len(fb_res_cli2['epidata']))
    #print('fb_cli3',fb_res_cli3['result'], fb_res_cli3['message'], len(fb_res_cli3['epidata']))
    print('fb_wcli4',fb_res_cli4['result'], fb_res_cli4['message'], len(fb_res_cli4['epidata']))
    print('fb_wcli5',fb_res_cli9['result'], fb_res_cli9['message'], len(fb_res_cli9['epidata']))
    print('fb_wcli6',fb_res_cli10['result'], fb_res_cli10['message'], len(fb_res_cli10['epidata']))
    print('fb_wcli8',fb_res_cli11['result'], fb_res_cli11['message'], len(fb_res_cli11['epidata']))
    print('fb_wcli9',fb_res_cli12['result'], fb_res_cli12['message'], len(fb_res_cli12['epidata']))
    
    #print('google_cli',google_res_cli['result'], google_res_cli['message'], len(google_res_cli['epidata']))
    
    #print(fb_res_wli['result'], fb_res_wli['message'], len(fb_res_wli['epidata']))
    
    #print('fb_wili1',fb_res_wli1['result'], fb_res_wli1['message'], len(fb_res_wli1['epidata']))
    #print('fb_wili2',fb_res_wli2['result'], fb_res_wli2['message'], len(fb_res_wli2['epidata']))
    #print('fb_wili3',fb_res_wli3['result'], fb_res_wli3['message'], len(fb_res_wli3['epidata']))
    print('fb_wili4',fb_res_wli4['result'], fb_res_wli4['message'], len(fb_res_wli4['epidata']))
    print('fb_wili5',fb_res_wli5['result'], fb_res_wli5['message'], len(fb_res_wli5['epidata']))
    print('fb_wili10',fb_res_wli10['result'], fb_res_wli10['message'], len(fb_res_wli10['epidata']))
    print('fb_wili11',fb_res_wli11['result'], fb_res_wli11['message'], len(fb_res_wli11['epidata']))
    print('fb_wili12',fb_res_wli12['result'], fb_res_wli12['message'], len(fb_res_wli12['epidata']))
    
    print('fb_wcli len',len(fb_res_cli))
    print('fb_wli len',len(fb_res_wli))
    #'''
    state_names=list(state_index.keys())
    cols=['fb_survey_wcli','fb_survey_wili']
    week_cases={}
    for c in cols:
        week_cases[c]=np.zeros((len(state_names),len(epiweek_date)))
    #'''
    #week_cases[cols[0]]=read_survey_epidata(cols[0],fb_res_cli['epidata'],state_index,state_names,epiweek_date)
    week_cases[cols[0]]=read_survey_epidata(cols[0],fb_res_cli_final,state_index,state_names,epiweek_date)
    #week_cases[cols[1]]=read_survey_epidata(cols[1],google_res_cli['epidata'],state_index,state_names,epiweek_date)
    #week_cases[cols[2]]=read_survey_epidata(cols[2],fb_res_wli['epidata'],state_index,state_names,epiweek_date)
    week_cases[cols[1]]=read_survey_epidata(cols[1],fb_res_wli_final,state_index,state_names,epiweek_date)
    #'''

    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/fb-google-survey.csv")
    return week_cases,cols
    


# In[45]:


def read_fips_code(state_names,fips_path,abbv_to_code=True):
    #if abbv_to_code=True; read fips with given state code (state_names)
    #if abbv_to_code=False; read state code with given state name (state_names)
    
    fips=pd.read_csv(fips_path+"other_data/fips_codes.csv",delimiter=',',dtype={'state_code':str})
    if abbv_to_code:
        fips=fips[['state','state_code']].drop_duplicates().reset_index()
    else:
        fips=fips[['state','state_name']].drop_duplicates().reset_index()
    
    state_fips={}
    for name in state_names:
        if abbv_to_code:
            if name=='X':
                row='US'
            else:
                row= str(fips.loc[fips['state'] == name,'state_code'].iloc[0])   
        else:
            if name=='X':
                row='X'
            else:
                row= fips.loc[fips['state_name'] == name,'state'].iloc[0]   
        
        state_fips[name]=row
    
    #print(state_fips)
    return state_fips

def find_date_index(dates,cur_date,date_string=1):
    if date_string==1:
        year,month,date=cur_date[:4],cur_date[4:6],cur_date[6:8]
    elif date_string==2:
        year,month,date=cur_date.split('-')
    else:
        month,date,year=cur_date.split('/')
    
    cdate=int(date)
    cmonth=int(month)
    year=int(year)
    
    if year==20 or year==21 or year==22:
        stryear='20'+str(year)
        year=int(stryear)
    
    for id in range(0,len(dates)):
        y,m,d=dates[id].split('-')
        wd,wm,wy=int(d),int(m),int(y)
        if wm==cmonth and wd==cdate and wy==year:
              return id
    #print('week index not found:'+cur_date)
    #print(cdate,cmonth,year)
    return -1

def get_epiweek_list(week_start,week_end,year):
    def convert(x):
        return Week.fromstring(str(x))
    
    wstart=convert(week_start)
    wend=convert(week_end)
    #print(wstart,wend)
    week_edate_list=[]
    week_list=[]
    for week in Year(year).iterweeks():
        date_time = week.enddate().strftime("%Y-%m-%d")
        #print(week,date_time)
        if week>=wstart and week<=wend:
            week_edate_list.append(date_time)
            week_list.append(str(week))
     
    return week_list,week_edate_list  


# In[46]:


def read_apple_mobility(inputdir,epiweek_date,state_index,dic_names_to_abbv):
    data=pd.read_csv(inputdir+"applemobilitytrends.csv",low_memory=False)
    state_names=list(dic_names_to_abbv.keys())
    dates=list(data.columns)
    dates=dates[6:]
    data = data.loc[data['region'].isin(state_names)]
    #data=data.fillna(0)
    #print(data.shape)
    #week_cases=np.zeros((len(state_names),len(epiweek_date)))
    week_cases=np.empty((len(state_names),len(epiweek_date)))
    week_cases[:][:]=np.nan
    for ix,row in data.iterrows():
        if row['transportation_type']=='driving':
            if row['region'] in state_names:
                state_id=state_index[dic_names_to_abbv[row['region']]] 
                for d in dates:
                    w_idx=find_date_index(epiweek_date,d,date_string=2)
                    if w_idx!=-1 and pd.isnull(row[d])==False:
                        if np.isnan(week_cases[state_id][w_idx]):
                            week_cases[state_id][w_idx]=0
                        week_cases[state_id][w_idx]=float(row[d])
    apple_dic={}
    apple_dic['apple_mobility']=week_cases    
    
    unit_test(apple_dic,['apple_mobility'],epiweek_date,state_index,"unit_test_date/apple.csv")
    
    return apple_dic,['apple_mobility']


# In[47]:


def read_mobility(inputdir,epiweek_date,end_day,MISSING_TOKEN=0):
    data=pd.read_csv(inputdir+"Global_Mobility_Report.csv",low_memory=False)
    data=data[data['country_region_code']=='US']
    data=data.drop(data[data['sub_region_1'] == 'Hawaii'].index)
    data= data[pd.isnull(data['sub_region_2'])]
    data=data.drop(columns=['sub_region_2'])

    state_names=data['sub_region_1'].drop_duplicates().values
    state_names[0]='X' #changing nan to US national X
    cols=data.columns
    cols=list(cols[8:])
    dic_names_to_abbv=read_fips_code(state_names,inputdir,abbv_to_code=False)
    
    state_index = {dic_names_to_abbv[state_names[i]]: i for i in range(len(state_names))} 
    #data[cols] = data[cols].fillna(MISSING_TOKEN)
    num_states=len(state_names)
    #print(cols)
    week_cases={}
    for c in cols:
        week_cases[c]=np.empty((num_states,len(epiweek_date)))
        week_cases[c][:][:]=np.nan
        '''
        week_cases[c]=np.zeros((num_states,len(epiweek_date)))
        if (len(epiweek_date)-end_day)!=0:
            total_len=len(epiweek_date)-end_day
            week_cases[c][:][-total_len:]=np.nan
        '''
    print(cols)
    #'''
    for ix,row in data.iterrows():
        if type(row['sub_region_1']) is float:
            state_id=0
        else:
            state_id=state_index[dic_names_to_abbv[row['sub_region_1']]]
        week_id=find_date_index(epiweek_date,str(row['date']),date_string=2)
        if week_id!=-1:
            for c in cols:
                if pd.isnull(row[c])==False:
                    if np.isnan(week_cases[c][state_id][week_id]):
                            week_cases[c][state_id][week_id]=0
                    week_cases[c][state_id][week_id]=row[c]
    
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test_date/google-mobility.csv")
    #'''
    
    return week_cases,cols,state_index,dic_names_to_abbv
    


# In[48]:


def read_vaccine_doses(inputdir,epiweek_date,state_index,dic_names_to_abbv):
    data=pd.read_csv(inputdir+"vaccine_data_us_state_timeline.csv")
    #data=data.fillna(0)
    #state_names=list(state_index.keys())
    state_names=list(dic_names_to_abbv.keys())
    #cols=['people_total','people_total_2nd_dose']
    cols=['Stage_One_Doses','Stage_Two_Doses']
    week_cases={}
    for c in cols:
        week_cases[c]=np.full((len(state_names),len(epiweek_date)), np.nan)
    
    
    for ix,row in data.iterrows():
        w_idx=find_date_index(epiweek_date,row['Date'],date_string=2)
        if row['Province_State'] in state_names and w_idx!=-1 and row['Vaccine_Type']=='All':
            #state_id=state_index[row['stabbr']] 
            state_id=state_index[dic_names_to_abbv[row['Province_State']]] 
            for c in cols:
                val=None
                if pd.isnull(row[c])==False:
                    val=float(row[c])
                elif w_idx>0:
                    val=week_cases[c][state_id][w_idx-1]               
                week_cases[c][state_id][w_idx]=val
                if np.isnan(week_cases[c][0][w_idx]): 
                    week_cases[c][0][w_idx] = 0
                week_cases[c][0][w_idx]+=val
    
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test_date/vaccine.csv")
    
    return week_cases,cols


# In[49]:


def read_dex(inputdir,epiweek_date,state_index,start_day,end_day):
    data=pd.read_csv(inputdir+"state_dex.csv",low_memory=False)
    cols=[]
    state_names=list(state_index.keys())
    for c in data.columns:
        if 'dex' in c:
            cols.append(c)
    
    week_cases={}
    for c in cols:
        week_cases[c]=np.empty((len(state_names),len(epiweek_date)))
        week_cases[c][:][:]=np.nan
        week_cases[c][0][:]=0
        '''
        week_cases[c]=np.zeros((len(state_names),len(epiweek_date)))
        if (len(epiweek_date)-end_day)!=0:
            total_len=len(epiweek_date)-end_day
            week_cases[c][:][-total_len:]=np.nan
        '''
    
    for ix,row in data.iterrows():
        w_idx=find_date_index(epiweek_date,row['date'],date_string=2)
        if row['state'] in state_names and w_idx!=-1:
            state_id=state_index[row['state']]
            for c in cols:
                if not math.isnan(row[c]):
                    if np.isnan(week_cases[c][state_id][w_idx]):
                        week_cases[c][state_id][w_idx]=0
                    week_cases[c][state_id][w_idx]=float(row[c])
                    week_cases[c][0][w_idx]+=float(row[c])
    
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test_date/dex.csv")
    
    return week_cases,cols


# In[50]:


def find_county_cases(observed,county_name,state_name,population,onlypopu=False):
    #print(state_name,county_name)
    try:
        row=population.loc[population['CTYNAME']==county_name]
        row=row.loc[row['STNAME']==state_name]
        popu=row['POPESTIMATE2019'].values
        if len(popu)==0:
            return -1
        #print(row['CTYNAME'].values,popu)
        if onlypopu:
            return popu[0]
        else:
            return (observed*popu[0])/100
    except KeyError:
        print('county not found')
        return -1
    
def read_kinsa(inputdir,epiweek_date,state_index,last_date,last_month,avg=1):
    '''
    data=pd.read_csv(inputdir+"ili_case_counts_by_state.csv",delimiter=',')
    data=data[['region_name','state','date','observed_ili']]

    county_names=data[['region_name']].drop_duplicates().reset_index()
    county_names=county_names['region_name'].values
    state_abbrv=pd.read_csv(inputdir+"state_abbrv.csv",delimiter=',')
    '''
    state_names=list(state_index.keys())
    week_cases_state=np.zeros((len(state_names),len(epiweek_date)))
    '''
    population_data=pd.read_csv(inputdir+"co-est2019-alldata2.csv",delimiter=',',encoding = "ISO-8859-1")
    population=population_data[['STNAME','CTYNAME','POPESTIMATE2019']]
    total=328239523
    
    county_index = {county_names[i]: i for i in range(len(county_names))} 
    
    for index,row in data.iterrows():
        state_id=state_index[row['state']]
        year,month,date=row['date'].split('-')
        #month,date,year=row['date'].split('/')
        if int(month)>last_month:
            continue
        if int(month)==last_month and int(date)>last_date:
            continue
        week_id=find_date_index(epiweek_date,row['date'],date_string=2)
        abbrv=state_abbrv.loc[state_abbrv['Code']==row['state']]
        abbrv=abbrv['State'].values
        observed=find_county_cases(row['observed_ili'],row['region_name'],abbrv[0],population)
        if week_id!=-1:
            week_cases_state[state_id][week_id]+=observed
            week_cases_state[0][week_id]+=observed
    
    #kinsa_dic={}
    #kinsa_dic['kinsa_cases']=week_cases_state
    #unit_test(kinsa_dic,['kinsa_cases'],epiweek_date,state_index,"unit_test_date/kinsa_rough.csv")
    print('data extrcation done, aggregating counties')
    week_cases_state[0][:]=(week_cases_state[0][:]*100)/(avg*total)
    for st_id in range(1,len(state_names)):
        st_name=state_abbrv.loc[state_abbrv['Code']==state_names[st_id]]
        st_name=st_name['State'].values
        popu=find_county_cases(week_cases_state[st_id][0],st_name[0],st_name[0],population,onlypopu=True)
        print(st_name[0],popu)
        week_cases_state[st_id][:]=(week_cases_state[st_id][:]*100)/(avg*popu)
    
    '''
    kinsa_dic={}
    kinsa_dic['kinsa_cases']=week_cases_state
    #unit_test(kinsa_dic,['kinsa_cases'],epiweek_date,state_index,"unit_test_date/kinsa.csv")
    return kinsa_dic,['kinsa_cases']


# In[51]:


def hosp_negative_total_data(inputdir,epiweek_date,state_index):
    data_hosp=pd.read_csv(inputdir+"COVID-19_PCR_Testing_Time_Series.csv",delimiter=',')
    state_names=list(state_index.keys())
    week_cases={}
    cols=['cdc_negativeIncr','cdc_positiveIncr','cdc_total_resultsIncr']
    for c in cols:
        week_cases[c]=np.full((len(state_names),len(epiweek_date)),np.nan)
    
    for index,row in data_hosp.iterrows():
        if row['overall_outcome']=='Negative':
            if row['state'] in state_index.keys():
                state_id=state_index[row['state']]
                date=row['date']
                week_id=find_date_index(epiweek_date,date.replace('/','-'),date_string=2)
                week_cases[cols[0]][state_id][week_id]=float(row['new_results_reported'])
                week_cases[cols[1]][state_id][week_id]=float(row['total_results_reported'])
                if np.isnan(week_cases[cols[0]][0][week_id]): 
                    week_cases[cols[0]][0][week_id] = 0
                    week_cases[cols[1]][0][week_id] = 0
                week_cases[cols[0]][0][week_id]+=float(row['new_results_reported'])
                week_cases[cols[1]][0][week_id]+=float(row['total_results_reported'])
    
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test_date/hosp_neg_ttl.csv") 
    
    return week_cases,cols


# In[52]:


def hosp_negative_total_data_test(inputdir,epiweek_date,state_index):
    data_hosp=pd.read_csv(inputdir+"COVID-19_PCR_Testing_Time_Series.csv",delimiter=',')
    state_names=list(state_index.keys())
    week_cases={}
    cols=['cdc_negativeIncr','cdc_positiveIncr', 'cdc_total_resultsIncr']
    for c in cols:
        week_cases[c]=np.full((len(state_names),len(epiweek_date)),np.nan)
    
    for index,row in data_hosp.iterrows():
        if row['state'] in state_index.keys():
            if row['overall_outcome']=='Negative':
                test_type = cols[0]
            elif row['overall_outcome'] == 'Positive': 
                test_type = cols[1]
            else:
                test_type = "NA"
            state_id=state_index[row['state']]
            date=row['date']
            week_id=find_date_index(epiweek_date,date.replace('/','-'),date_string=2)
            if test_type != "NA": 
#                 print(week_cases)[state_id][week_id]
                week_cases[test_type][state_id][week_id]=float(row['new_results_reported'])
                if np.isnan(week_cases[test_type][0][week_id]): 
                    week_cases[test_type][0][week_id] = 0
                week_cases[test_type][0][week_id]+=float(row['new_results_reported'])
            if np.isnan(week_cases[cols[2]][state_id][week_id]): 
                week_cases[cols[2]][state_id][week_id] = 0
            if np.isnan(week_cases[cols[2]][0][week_id]):
                week_cases[cols[2]][0][week_id] = 0
            week_cases[cols[2]][state_id][week_id]+=float(row['new_results_reported'])
            week_cases[cols[2]][0][week_id]+=float(row['new_results_reported'])
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/hosp_neg_ttl.csv") 
    
    return week_cases,cols


# In[53]:


def read_jhu_cases(inputdir,epiweek_date,state_index,dic_names_to_abbv):
    data=pd.read_csv(inputdir+"time_series_covid19_confirmed_US.csv")
    state_names=list(dic_names_to_abbv.keys())
    dates_list=list(data.columns)
    dates_list=dates_list[11:]
    #print(state_names)
    #print(dates_list)
    #not_added_rows=['Diamond Princess','Grand Princess','Northern Mariana Islands','Guam','Hawaii',
                   # 'Puerto Rico','Virgin Islands','American Samoa']
    not_added_rows=[]
    week_cases={}
    cols=['positiveIncr_cumulative','positiveIncr']
    for c in cols:
        week_cases[c]=np.full((len(state_names),len(epiweek_date)),np.nan)
    
    data2 = data.groupby(['Province_State']).sum()
    print(state_names)
    for ix, row in data2.iterrows():
        name=ix
        if name in state_names:
            state_id=state_index[dic_names_to_abbv[name]]
            for date in dates_list:
                w_idx=find_date_index(epiweek_date,date,date_string=3)
                if w_idx!=-1:
                    if np.isnan(week_cases[cols[0]][state_id][w_idx]): 
                        week_cases[cols[0]][state_id][w_idx] = 0
                    if np.isnan(week_cases[cols[0]][0][w_idx]):
                        week_cases[cols[0]][0][w_idx] = 0
                    #print(name,date,row[date])
                    week_cases[cols[0]][state_id][w_idx]+=int(row[date])
                    week_cases[cols[0]][0][w_idx]+=int(row[date])
        elif name not in not_added_rows: #if not state_names still adding them for national
            #print('state name not added:'+name)
            for date in dates_list:
                w_idx=find_date_index(epiweek_date,date,date_string=3)
                if w_idx!=-1:
                    if np.isnan(week_cases[cols[0]][0][w_idx]):
                        week_cases[cols[0]][0][w_idx] = 0
                    week_cases[cols[0]][0][w_idx]+=int(row[date])
    
    
    #count incidence for the national+states
    #count incidence for the national+states
    for state_id in range(len(state_names)):
        if state_id == 51: 
            continue
        index = 0
        while np.isnan(week_cases[cols[0]][state_id][index]):
            index += 1
        week_cases[cols[1]][state_id][index]=int(week_cases[cols[0]][state_id][index])
        for w_idx in range(index + 1,len(epiweek_date)):
            if np.isnan(week_cases[cols[0]][state_id][w_idx]): 
                continue
            week_cases[cols[1]][state_id][w_idx]=abs(int(week_cases[cols[0]][state_id][w_idx])-int(week_cases[cols[0]][state_id][w_idx-1]))         
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test_date/jhu-cases.csv")
    return week_cases,cols


# In[54]:


def hosp_cases_nat_state(data_national,data_state,epiweek_date,week_save,state_index,cols,state_error,epiweek_end):
    state_names=list(state_index.keys())
    week_cases={}
    
    for c in cols:
        week_cases[c]=np.zeros((len(state_names),len(epiweek_date)))
    
    #processing national cases
    nat_id=state_index['X']
    nat_hosp_incr=np.zeros(len(epiweek_date))
    for index,row in data_national.iterrows():
        week_id=find_date_index(week_save,str(row['date']),date_string=1)
        for c in range(len(cols)): #if next1,next2 consider then do cols-4
            week_cases[cols[c]][nat_id][week_id]+=row[cols[c]]
    
    #processing state cases
    for index,row in data_state.iterrows():
        state_id=state_index[row['states']]
        cur_date=int(row['date'])
        week_id=find_date_index(week_save,str(cur_date))
        if week_id!=-1:
            for c in range(len(cols)): #if next1,next2 consider then do cols-4
                if cols[c]=='hospitalizedIncrease':
                    if row['states'] in state_error:
                        week_cases[cols[c]][state_id][week_id]=max(0,row['hospitalizedCurrently'])
                    else:
                        week_cases[cols[c]][state_id][week_id]=max(0,row[cols[c]])
                        nat_hosp_incr[week_id]+=max(0,row[cols[c]])
                else:
                    week_cases[cols[c]][state_id][week_id]=max(0,row[cols[c]])
    
    #for states with no hospitalze cumulative using formula week[t]=hosp_cur[t]-(week[t-1]/2)
    c='hospitalizedIncrease'
    temp_data=pd.read_csv("unit_test/hospitalization.csv")
    
    for s in state_error:
        st_idx=state_index[s]
        region_hosp=temp_data.loc[temp_data['region']==s]
        hosp_weekly=region_hosp['hospitalizedIncrease'].values
        #print('hosp_weekly:',len(hosp_weekly))
        #print(hosp_weekly)
        hsp_incr=np.zeros(len(epiweek_date))
        hsp_incr[0]=week_cases[c][st_idx][0]
        nat_hosp_incr[0]+=week_cases[c][st_idx][0]
        for w in range(1,len(epiweek_date)):
            cur_date=epiweek_date[w]
            e_idx=get_epiweek_index(cur_date,'202001',epiweek_end) #change/edit to work it for year>2020
            #hsp_incr[w]=max(0,week_cases[c][st_idx][w]-float(hsp_incr[w-1])/2) #7 days before
            if e_idx!=-1:
                hsp_incr[w-6:w+1]=hosp_weekly[e_idx-1]
                nat_hosp_incr[w-6:w+1]+=(hosp_weekly[e_idx-1]/7)
            #else:
             #   hsp_incr[w]=hsp_incr[w-1]
            #else:
             #   hsp_incr[w]=max(0,week_cases[c][st_idx][w]-float(week_cases[c][st_idx][w-1])) #7 days not passed yet
        week_cases[c][st_idx]=hsp_incr/7
    
    c='recovered'
    rec_nat=np.zeros(len(epiweek_date))
    for s in state_names:
        st_idx=state_index[s]
        rec_incr=np.zeros(len(epiweek_date))
        rec_incr[0]=week_cases[c][st_idx][0]
        rec_nat[0]+=week_cases[c][st_idx][0]
        for w in range(1,len(epiweek_date)):
            rec_incr[w]=week_cases[c][st_idx][w]-week_cases[c][st_idx][w-1]
            rec_nat[w]+=max(rec_incr[w],0)

        week_cases[c][st_idx]=rec_incr
    week_cases[c][nat_id]=rec_nat
    np.savetxt("unit_test_date/hos_nat_aggregated.csv", nat_hosp_incr, fmt='%.4f')
    return week_cases

def read_hospitalization(input_national,input_state,epiweek_date,week_save,state_index,state_error,ew_end):
    nat_data_path="us-daily-hospitalizations.csv"
    state_data_path="states-daily-hospitalizations.csv"
    nat_data=pd.read_csv(input_national+nat_data_path,delimiter=',')
    state_data=pd.read_csv(input_state+state_data_path,delimiter=',')
    state_data=state_data.rename(columns={"state": "states"})
    
    columns=['date','states','hospitalizedCurrently','onVentilatorCurrently','positiveIncrease','negativeIncrease',
             'totalTestResultsIncrease','inIcuCurrently','recovered','deathIncrease','hospitalizedIncrease']
    rows_to_drop=['GU','AS','HI','PR', 'MP','VI']
    
    nat_data_filter=nat_data[columns]
    state_data_filter=state_data[columns]
    
    nat_data_filter=nat_data_filter.fillna(0)
    state_data_filter=state_data_filter.fillna(0)
    
    index_to_remove=[]
    for index, row in state_data_filter.iterrows():
        #print(row['Province/State'])
        if row['states'] in rows_to_drop:
            index_to_remove.append(index)

    state_data_filter=state_data_filter.drop(index=index_to_remove).reset_index()
    #print(state_data_filter['states'].drop_duplicates())
    #cols_for_hosp=['positiveIncrease','negativeIncrease','totalTestResultsIncrease','onVentilatorCurrently',
     #              'inIcuCurrently','deathIncrease','recovered','hospitalizedIncrease','h_next1','h_next2',
      #             'd_next1','d_next2']
    cols_for_hosp=['positiveIncrease','negativeIncrease','totalTestResultsIncrease','onVentilatorCurrently',
                   'inIcuCurrently','deathIncrease','recovered','hospitalizedIncrease']
    
    week_cases=hosp_cases_nat_state(nat_data_filter,state_data_filter,epiweek_date,week_save,state_index,cols_for_hosp,state_error,ew_end)
    
    unit_test(week_cases,cols_for_hosp,epiweek_date,state_index,"unit_test_date/hospitalization_date.csv")
    return week_cases,cols_for_hosp


# In[55]:


def read_jhu_death(inputdir,epiweek_date,state_index,dic_names_to_abbv):
    data=pd.read_csv(inputdir+"time_series_covid19_deaths_US.csv")
    state_names=list(dic_names_to_abbv.keys())
    dates_list=list(data.columns)
    dates_list=dates_list[12:]
    #print(state_names)
    #print(dates_list)
    #not_added_rows=['Diamond Princess','Grand Princess','Northern Mariana Islands','Guam','Hawaii',
                   # 'Puerto Rico','Virgin Islands','American Samoa']
    not_added_rows=[]
    week_cases={}
    cols=['death_jhu_cumulative','death_jhu_incidence']
    for c in cols:
        week_cases[c]=np.full((len(state_names),len(epiweek_date)), np.nan)
    
    data2 = data.groupby(['Province_State']).sum()
    for ix, row in data2.iterrows():
        name=ix
        if name in state_names:
            state_id=state_index[dic_names_to_abbv[name]]
            for date in dates_list:
                w_idx=find_date_index(epiweek_date,date,date_string=3)
                if w_idx!=-1:
                    #print(name,date,row[date])
                    if np.isnan(week_cases[cols[0]][state_id][w_idx]): 
                         week_cases[cols[0]][state_id][w_idx] = 0
                    if np.isnan(week_cases[cols[0]][0][w_idx]): 
                         week_cases[cols[0]][0][w_idx] = 0
                    week_cases[cols[0]][state_id][w_idx]+=int(row[date])
                    week_cases[cols[0]][0][w_idx]+=int(row[date])
        elif name not in not_added_rows: #if not state_names still adding them for national
            #print('state name not added:'+name)
            for date in dates_list:
                w_idx=find_date_index(epiweek_date,date,date_string=3)
                if w_idx!=-1:
                    if np.isnan(week_cases[cols[0]][0][w_idx]): 
                        week_cases[cols[0]][0][w_idx] = 0
                    week_cases[cols[0]][0][w_idx]+=int(row[date])
    
    
    #count incidence for the national+states
    for state_id in range(len(state_names)):
        if state_id == 51: 
            continue
        index = 0
        while np.isnan(week_cases[cols[0]][state_id][index]):
            index += 1
        week_cases[cols[1]][state_id][index]=int(week_cases[cols[0]][state_id][index])
        for w_idx in range(index+1,len(epiweek_date)):
            if np.isnan(week_cases[cols[0]][state_id][w_idx]): 
                continue
            week_cases[cols[1]][state_id][w_idx]=abs(int(week_cases[cols[0]][state_id][w_idx])-int(week_cases[cols[0]][state_id][w_idx-1]))
             
    
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test_date/jhu-death.csv")
    return week_cases,cols


# In[56]:


def read_covidnet(data_covidnet,week_len,start,end,step,weekly_rate,region): #region='X'
    covid=np.empty(week_len)
    covid[:]=np.nan
    data_f=data_covidnet.loc[data_covidnet['CATCHMENT']==region]
    #data_f=data[data['AGE CATEGORY']=='Overall'].reset_index()
    for index,row in data_f.iterrows():
        mmr_week=int(row['MMWR-WEEK'])
        year=int(row['MMWR-YEAR'])
        if row['AGE CATEGORY']=='Overall' and row['SEX']=='Overall' and row['RACE']=='Overall':
            if year==2020:
                if mmr_week>=start and mmr_week<=53:
                    if region=='Entire Network':
                        if row['NETWORK']=='COVID-NET':
                            covid[mmr_week-10+step]=float(row[weekly_rate])
                    else:
                        covid[mmr_week-10+step]=float(row[weekly_rate])
            elif year==2021:
                if mmr_week<=52:
                    if region=='Entire Network':
                        if row['NETWORK']=='COVID-NET':
                            covid[43+step+mmr_week]=float(row[weekly_rate])
                    else:
                        covid[43+step+mmr_week]=float(row[weekly_rate])
            elif year==2022:
                if mmr_week<=end:
                    if region=='Entire Network':
                        if row['NETWORK']=='COVID-NET':
                            covid[95+step+mmr_week]=float(row[weekly_rate])
                    else:
                        covid[95+step+mmr_week]=float(row[weekly_rate])
    return covid

def read_covidnet_data(inputdir,epiweek_date,state_index,dic_names_to_abbv,start_week,end_week,step,num_epiweek):
    #data_covidnet=pd.read_csv(inputdir+"COVID-NET_Processed.csv",delimiter=',')
    if date.today().weekday() != 6:
        week_num = (date.today()-timedelta(days=2)).strftime("%U")
        year_week_num = "2022" + week_num
    else:
        week_num = (date.today()-timedelta(days=1)).strftime("%U")
        year_week_num = "2022" + week_num
        
    file_name = "COVID-NET_v"+year_week_num + ".csv"
    data_covidnet=pd.read_csv(file_name,delimiter=',')
    columns=list(data_covidnet.columns)
    
    dic_names_to_abbv['Entire Network']='X'
    state_names=list(dic_names_to_abbv.keys())
    #state_names=list(state_index.keys())
    #print(columns)
    weekly_rate='WEEKLY RATE'
    if 'WEEKLY RATE' in columns:
        data_covidnet=data_covidnet[['CATCHMENT','NETWORK','MMWR-YEAR','MMWR-WEEK','AGE CATEGORY','SEX','RACE','WEEKLY RATE']]
    else:
        data_covidnet=data_covidnet[['CATCHMENT','NETWORK','MMWR-YEAR','MMWR-WEEK','AGE CATEGORY','SEX','RACE','WEEKLY RATE ']]
        weekly_rate='WEEKLY RATE '
    
    week_cases=np.zeros((len(state_names),len(epiweek_date)))
    #for st in state_index.keys():
    for state in state_names:
        if state=='United States' or state=='X':
            continue
        st=dic_names_to_abbv[state]
        #print(state,state_index[st])
        covidnet_week=read_covidnet(data_covidnet,num_epiweek,start_week,end_week,step,weekly_rate,region=state)
        for week in range(0,num_epiweek):
            start,end=map_epiweek_to_date(week+1,epiweek_date,num_epiweek,week_string=False,year=2022)
            week_cases[state_index[st]][start:end+1]=[covidnet_week[week]]*(end-start+1)
    
    covidnet_dic={}
    covidnet_dic['covidnet']=week_cases
    unit_test(covidnet_dic,['covidnet'],epiweek_date,state_index,"unit_test_date/covidnet.csv")
    return covidnet_dic,['covidnet']


# In[57]:


def get_epiweek_index(cur_date,start_week,end_week,year=2022):
    #print('epiweek_index',cur_date)
    week_num = date.today().strftime("%U")
    year_week_num = "2022" + week_num
    epiweek1,epiweek_date1=get_epiweek_list(start_week,'202053',2020)
    epiweek2,epiweek_date2=get_epiweek_list('202101','202152',2021)
    epiweek3,epiweek_date3=get_epiweek_list('202201',year_week_num,2022)
    #epiweek=epiweek1+epiweek2
    #epiweek_date=epiweek_date1+epiweek_date2
    for d in range(0,len(epiweek_date1)):
        if cur_date==epiweek_date1[d]:
            return int(epiweek1[d][4:])
    
    for d in range(0,len(epiweek_date2)):
        if cur_date==epiweek_date2[d]:
            return int(epiweek2[d][4:])+53
    for d in range(0, len(epiweek_date3)):
        if cur_date==epiweek_date3[d]:
            return int(epiweek3[d][4:])+105
    
    return -1
            
def map_epiweek_to_date(week,epiweek_date,num_epiweek,week_string=True,year=2022):
    if week_string:
        stryear=int(week[:4])
        week=int(week[4:])
        if stryear>2020:
            week+=53
    if year==2020:
        end_week=str(year)+str(num_epiweek)
    else:
        pos=num_epiweek%106+1
        end_week=str(year)+str(pos)
    #start_week=str(year)+'01'
    start_week='202001'
    epiweek1, week_dates1=get_epiweek_list('202001','202053',2020)
    epiweek2,week_dates2=[],[]
    epiweek2, week_dates2=get_epiweek_list('202101','202152',2021)
    epiweek3, week_dates3=get_epiweek_list('202201', end_week, year)
    
    epiweek=epiweek1+epiweek2+epiweek3
    week_dates=week_dates1+week_dates2+week_dates3
    if week-2<0:
        #week_date_start_str=str(year)+'-01-01' #2020-01-01, 2021-01-01
        week_date_start_str='2020-01-01' #2020-01-01, 2021-01-01
        week_date_start_obj = datetime.strptime(week_date_start_str, '%Y-%m-%d')
    else:
        week_date_start_str=week_dates[week-2]
        week_date_start_obj = datetime.strptime(week_date_start_str, '%Y-%m-%d')+timedelta(days=1)
    
    week_date_end_str=week_dates[week-1]
    start_idx=-1
    end_idx=-1
    
    week_date_end_obj = datetime.strptime(week_date_end_str, '%Y-%m-%d')
    
    for d in range(0,len(epiweek_date)):
        cur_date_obj=datetime.strptime(epiweek_date[d], '%Y-%m-%d')
        if cur_date_obj>=week_date_start_obj and cur_date_obj<week_date_end_obj and start_idx==-1:
            start_idx=d
        elif cur_date_obj==week_date_end_obj:
            end_idx=d
    
    return start_idx,end_idx 
        
    


# In[58]:


def read_iqvia(inputdir,epiweek_date,state_index,num_epiweek):
    data=pd.read_csv(inputdir+"iqvia_Processed.csv")
    state_names=list(state_index.keys())
    week_cases={}

    cols=list(data.columns)
    cols=cols[4:]
    print(cols)
    for c in cols:
        week_cases[c]=np.full([len(state_names),len(epiweek_date)],np.nan)
    
    for ix, row in data.iterrows():
        state_id=state_index[row['region']]
        week=str(row['epiweek'])
        start,end=map_epiweek_to_date(week,epiweek_date,num_epiweek)
        for c in cols:
            #week_cases[c][state_id][start_week-10:end_week-start_week]=group[c]
            week_cases[c][state_id][start:end+1]=float(row[c])
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test_date/iqvia.csv")
    return week_cases,cols


# In[59]:


def read_cdc_hosp(inputdir,epiweek_date,state_index,start_day,end_day):
    data=pd.read_csv(inputdir+"COVID-19_Reported_Timeseries.csv")
    #data=pd.read_csv(inputdir+"reported_hospital_timeseries.csv")
    
    #data=data.fillna(0)
    state_index['PR'] = 51
    state_index['VI'] = 52
    state_names=list(state_index.keys())
    week_cases={}
    #cols=list(data.columns)
    #cols=cols[2:]
    cols=['cdc_hospitalized']
    
    cols_to_check=['previous_day_admission_adult_covid_confirmed','previous_day_admission_pediatric_covid_confirmed']
    for c in cols:
        week_cases[c]=np.empty((len(state_names),len(epiweek_date)))
    
    week_cases[c][:][:]=np.nan
    week_cases[c][0][:]=0
    for index,row in data.iterrows():
        if row['state'] in state_index.keys():
            state_id=state_index[row['state']]
        else:
            continue
        date=str(row['date'])
        week_id=find_date_index(epiweek_date,date.replace('/','-'),date_string=2)
        if week_id!=-1:
            ttl=0
            flag_is_not_nan=False
            for c in cols_to_check: 
                if pd.isnull(row[c])==False:
                    ttl+=float(row[c])
                    flag_is_not_nan=True
            if flag_is_not_nan:
                week_cases[cols[0]][state_id][week_id]=ttl
                week_cases[cols[0]][0][week_id]+=ttl
                    
    
    '''
    missing_days=len(epiweek_date)-end_day
    for s in range(len(state_names)):
        for c in cols:
            week_cases[c][s][-missing_days:]=np.nan #considering data 2 weeks lag
    '''
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test_date/cdc_hosp.csv") 
    del state_index['PR']
    del state_index['VI']
    return week_cases,cols


# In[60]:


def read_excess_death(inputdir,epiweek_date,state_index,dic_names_to_abbv,start_week,end_week,num_epiweek,this_month,this_year=2020):
    data=pd.read_csv(inputdir+"Excess_Deaths_COVID-19.csv")
    #cols_death=['Observed Number','Excess Higher Estimate']
    cols_death=['Observed Number','Excess Estimate'] 
#     out_cols = ['Observed Numberv2','Excess Estimatev2']
    cols=['Week Ending Date','State']+cols_death
    data=data[cols]
    data[cols]=data[cols].fillna(0)
    state_names=list(dic_names_to_abbv.keys())
    week_cases={}
    for c in cols_death:
        week_cases[c]=np.zeros((len(state_names),len(epiweek_date)))
    
    for ix, row in data.iterrows():
        week=row['Week Ending Date']
        y,m,d=week.split('-')
        cm,cd,cy=int(m),int(d),int(y)
        name=row['State']
        if name in state_names and cy>=this_year:# and cm<=this_month:
            cur_date=y+'-'+m+'-'+d
            week=get_epiweek_index(cur_date,start_week,end_week)
            start,end=map_epiweek_to_date(week,epiweek_date,num_epiweek,week_string=False,year=2022)
            state_id=state_index[dic_names_to_abbv[name]]
            #print(week,start,end)
            for c in cols_death:
                week_cases[c][state_id][start:end+1]+=int(row[c])
                week_cases[c][0][start:end+1]=week_cases[c][0][start:end+1]+int(row[c])
     
    ##ADD NAN VALUES TO THE LAST 1 WEEK
    for s in range(len(state_names)):
        for c in cols_death:
            week_cases[c][s]/=3
            week_cases[c][s][-14:]=np.nan
    
    
    unit_test(week_cases,cols_death,epiweek_date,state_index,"unit_test_date/excess-death.csv")
    return week_cases,cols_death


# In[61]:


def read_emergency(inputdir,epiweek_date,state_index,start_week,end_week):
    #data=pd.read_csv(inputdir+"emergency-visits.csv")
    #data=pd.read_csv(inputdir+"covid-like-illness.csv")
    data=pd.read_csv(inputdir+"covid-like-illness-v202040.csv")
    cols=['Number of Facilities Reporting','CLI Percent of Total Visits']
    data=data[data['week']>=start_week]
    state_names=list(state_index.keys())
    week_cases={}
    for c in cols:
        week_cases[c]=np.full([len(state_names),len(epiweek_date)],np.nan)
    
    
    region_map={'X':0,'Region 1':1,'Region 2':2, 'Region 3':3,'Region 4':4,'Region 5':5,'Region 6':6,
               'Region 7':7,'Region 8':8,'Region 9':9,'Region 10':10}

    file = open(inputdir+"other_data/hhs_regions_abbv.txt", 'r') 
    Lines = file.readlines() 
    state_hhs_map={}
    #### mapping each state to a hhs region ###
    state_hhs_map[0]=0 # setting national value
    for line in Lines:
        regions=line.strip().split(',')
        reg_id=int(regions[0])
        for j in range(1,len(regions)):
            if state_index.get(regions[j],-1)!=-1:
                state_id=state_index[regions[j]]
                state_hhs_map[state_id]=reg_id
            else:
                print('state not found '+regions[j])
    #print(state_hhs_map)
    num_epiweeks=end_week-start_week+1
    for ix,row in data.iterrows():
        reg_id=region_map[row['region']]
        week=str(row['Week'])
        year=int(week[:4])
        w_idx=int(week[4:])
        for st in range(len(state_names)):
            if state_hhs_map[st]==reg_id:
                for c in cols:
                    reporting=row[c]
                    if c=='Number of Facilities Reporting':
                        reporting=int(reporting.replace(',',''))
                    #print(state_names[st],w_idx,reporting)
                    start,end=map_epiweek_to_date(week,epiweek_date,num_epiweeks)
                    week_cases[c][st][start:end+1]=reporting
    
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test_date/emergency.csv")
    return week_cases,cols
    


# In[70]:


def merge_data_state(mobility,apple,cdc_hosp,vacc,covidnet,excess,jhu,survey,jhu_case,hosp_new_res,
                     cols_m,cols_a,cols_cdc,cols_vacc,cols_net,cols_excess,
                     cols_jhu,cols_survey,cols_jhu_case,cols_hosp_new_res,
                     state_fips,epiweek,epiweek_date,region_names,outputdir,outfilename):
    
    cols_common=['date','epiweek','region','fips']
    #all_cols=cols_common+cols_m+cols_a+cols_cdc+cols_d+cols_k+cols_q+cols_net+cols_hosp+cols_excess+cols_jhu+cols_survey+cols_v
    all_cols=cols_common+cols_m+cols_a+cols_cdc+cols_vacc+cols_net+cols_excess+cols_jhu+cols_survey+cols_jhu_case+cols_hosp_new_res
    #all_cols=cols_common+cols_m+cols_a+cols_d+cols_k+cols_net+cols_hosp+cols_excess+cols_jhu+cols_survey+cols_v
    print(all_cols)
    final_data=pd.DataFrame(columns=all_cols)
    for reg in range(len(region_names)):
        temp_data=pd.DataFrame(columns=all_cols)
        temp_data['date']=epiweek_date
        temp_data['epiweek']=epiweek
        temp_data['region']=[region_names[reg]]*len(epiweek_date)
        temp_data['fips']=[state_fips[region_names[reg]]]*len(epiweek_date)
        for c in cols_m:
            temp_data[c]=mobility[c][reg][:]
        for c in cols_a:
            temp_data[c]=apple[c][reg][:]
        for c in cols_cdc:
            if reg == 0: 
                cdc_hosp[c][reg][0:96] = np.nan
                print(cdc_hosp[c][reg][0:96])
            temp_data[c]=cdc_hosp[c][reg][:]   
        #for c in cols_d:
        #    temp_data[c]=dex[c][reg][:]
        for c in cols_vacc:
            temp_data[c]=vacc[c][reg][:]
        #for c in cols_vac_delphi:
        #    temp_data[c]=vac_delphi[c][reg][:]
        #for c in cols_k:
        #    temp_data[c]=kinsa[c][reg][:]
        #for c in cols_q:
         #   temp_data[c]=iqvia[c][reg][:]
        for c in cols_net:
            temp_data[c]=covidnet[c][reg][:]
        #for c in cols_hosp:
        #    temp_data[c]=hosp[c][reg][:]
        for c in cols_excess:
            temp_data[c]=excess[c][reg][:]
        for c in cols_jhu:
            temp_data[c]=jhu[c][reg][:]
        for c in cols_survey:
            temp_data[c]=survey[c][reg][:]
        #for c in cols_v:
            #temp_data[c]=em_visit[c][reg][:]
        for c in cols_jhu_case:
            temp_data[c]=jhu_case[c][reg][:]
        for c in cols_hosp_new_res:
            temp_data[c]=hosp_new_res[c][reg][:]
        
        temp_data=temp_data[all_cols]
        final_data=final_data.append(temp_data,ignore_index=True)
    
    final_data=final_data[all_cols]
    print(final_data.shape)
    final_data.to_csv(outputdir+outfilename,index=False)
    
    print('FINISHED....')


# In[63]:


def merge_data_state_test(mobility,cols_m,state_fips,epiweek,epiweek_date,region_names,outputdir,outfilename):
    cols_common=['date','epiweek','region','fips']
    #all_cols=cols_common+cols_m+cols_a+cols_cdc+cols_d+cols_k+cols_q+cols_net+cols_hosp+cols_excess+cols_jhu+cols_survey+cols_v
    all_cols=cols_common+cols_m
    #all_cols=cols_common+cols_m+cols_a+cols_d+cols_k+cols_net+cols_hosp+cols_excess+cols_jhu+cols_survey+cols_v
    print(all_cols)
    final_data=pd.DataFrame(columns=all_cols)
    for reg in range(len(region_names)):
        temp_data=pd.DataFrame(columns=all_cols)
        temp_data['date']=epiweek_date
        temp_data['epiweek']=epiweek
        temp_data['region']=[region_names[reg]]*len(epiweek_date)
        temp_data['fips']=[state_fips[region_names[reg]]]*len(epiweek_date)
        for c in cols_m:
            temp_data[c]=mobility[c][reg][:]
        """    
        for c in cols_a:
            temp_data[c]=apple[c][reg][:]
        for c in cols_cdc:
            temp_data[c]=cdc_hosp[c][reg][:]
        for c in cols_d:
            temp_data[c]=dex[c][reg][:]
        for c in cols_vacc:
            temp_data[c]=vacc[c][reg][:]
        for c in cols_vac_delphi:
            temp_data[c]=vac_delphi[c][reg][:]
        for c in cols_k:
            temp_data[c]=kinsa[c][reg][:]
        #for c in cols_q:
         #   temp_data[c]=iqvia[c][reg][:]
        for c in cols_net:
            temp_data[c]=covidnet[c][reg][:]
        for c in cols_hosp:
            temp_data[c]=hosp[c][reg][:]
        for c in cols_excess:
            temp_data[c]=excess[c][reg][:]
        for c in cols_jhu:
            temp_data[c]=jhu[c][reg][:]
        for c in cols_survey:
            temp_data[c]=survey[c][reg][:]
        #for c in cols_v:
            #temp_data[c]=em_visit[c][reg][:]
        for c in cols_jhu_case:
            temp_data[c]=jhu_case[c][reg][:]
        for c in cols_hosp_new_res:
            temp_data[c]=hosp_new_res[c][reg][:]
        """
        temp_data=temp_data[all_cols]
        final_data=final_data.append(temp_data,ignore_index=True)
    
    final_data=final_data[all_cols]
    print(final_data.shape)
    final_data.to_csv(outputdir+outfilename,index=False)
    
    print('FINISHED....')


# In[71]:



data_path="./"
kinsa_path= "./"

if date.today().weekday() != 6:
    week_num = (date.today()-timedelta(2)).strftime("%U")
    year_week_num = "2022"  + week_num
    print(year_week_num)
    week_end_date = (date.today() +timedelta((5-date.today().weekday()) % 7 )).strftime('%Y-%m-%d')
    week_end_date = (date.today() - timedelta(2)).strftime('%Y-%m-%d')
    week_end_string = (date.today() + timedelta((5-date.today().weekday()) % 7 )).strftime('%Y%m%d')
    week_end_string = (date.today() - timedelta(2)).strftime('%Y%m%d')
else:
    week_num = (date.today()-timedelta(days=1)).strftime("%U")
    year_week_num = "2022" + week_num
    print(year_week_num)
    week_end_date = (date.today() -timedelta((date.today().weekday()-5) % 7 )).strftime('%Y-%m-%d')
    week_end_string = (date.today() - timedelta((date.today().weekday()-5) % 7 )).strftime('%Y%m%d')

epiweek_date_obj=pd.date_range(start='2020-01-01', end=week_end_date) #update date as this epiweek yyyy-mm-dd
epiweek_date=[date_obj.strftime('%Y-%m-%d') for date_obj in epiweek_date_obj]
week_save=[date_obj.strftime('%Y-%m-%d') for date_obj in epiweek_date_obj]

num_epiweek=105+int(week_num) #total epiweeks for 2022: 53+52+current_epiweek#update total num of epiweeks

epiweek_list1,epiweek_date_list1=get_epiweek_list('202001','202053',2020) #update end epiweek as current epiweek
epiweek_list2,epiweek_date_list2=get_epiweek_list('202101','202152',2021) #update end epiweek as current epiweek
epiweek_list3,epiweek_date_list3=get_epiweek_list('202201',year_week_num,2022)

epiweek_list=epiweek_list1+epiweek_list2+epiweek_list3
epiweek_date_list=epiweek_date_list1+epiweek_date_list2+epiweek_date_list3

epiweek=np.array([0 for i in range(len(epiweek_date))])
#print(len(epiweek))
for week in range(1,num_epiweek+1):
    st,ed=map_epiweek_to_date(week,epiweek_date,num_epiweek,week_string=False)
    #print(week,st,ed)
    epiweek[st:ed+1]=epiweek_list[week-1]

print(len(epiweek_date))
#print(epiweek)
#print(epiweek_date)
#print(week_save)

print('google mobility')
end_day=len(epiweek_date)-4 #usually m=ewdate-4, since current data does not have value for this week
print(epiweek_date[end_day])
mobility_state,cols_m,state_index,dic_names_to_abbv=read_mobility(data_path,epiweek_date,end_day)
dic_names_to_abbv['United States']='X'


print('JHU-cases')
jhu_cases,cols_jhu_case=read_jhu_cases(data_path,epiweek_date,state_index,dic_names_to_abbv)


print('hosp-neg-total result')
hosp_new_res,cols_hosp_new_res=hosp_negative_total_data_test(data_path,epiweek_date,state_index)


print('FB-GOOGLE')
start_week=20210101
start_week2 = 20200307
end_week=int(week_end_string) #yyyymmdd
survey, cols_survey = read_fb(state_index,epiweek_date,start_week,end_week,start_week2) 
#survey,cols_survey=read_delphi_fb_google_survey_test(state_index,epiweek_date,start_week,end_week)
"""
print('delphi vaccine survey')
start_week=20210101
start_week2 = 20200307
end_week=int(week_end_string) #yyyymmdd

vacc_delphi,cols_vacc_delphi=read_delphi_vaccine_test_two(state_index,epiweek_date,start_week,end_week)"""

print('vaccine doses')
vacc,cols_vacc=read_vaccine_doses(data_path,epiweek_date,state_index,dic_names_to_abbv)

print('CDC hospitalization')
start_day=3
end_day=len(epiweek_date)-2 #change the value 14 if its is more/less
#print(epiweek_date[start_day],epiweek_date[end_day])
cdc_hosp,cols_cdc=read_cdc_hosp(data_path,epiweek_date,state_index,start_day,end_day)

print('apple mobility')
apple_mobility,cols_a=read_apple_mobility(data_path,epiweek_date,state_index,dic_names_to_abbv)

print('JHU death')
jhu_death,cols_jhu=read_jhu_death(data_path,epiweek_date,state_index,dic_names_to_abbv)

"""
#print('iqvia')
#change start, end week based on data and epiweek_date. current data has value since epiweek 10-17
#num_epiweek=34#total epiweek-1
#iqvia_state,cols_q=read_iqvia(data_path,epiweek_date,state_index,num_epiweek)"""

print('covidnet')
step=9 #if epiweek starts from 10 then step=0, if epiweek starts from 1 step=9 (10-epiweek_start)
start_week_n=10
end_week_n=int(week_num) #change to current epiweek (1,...) since next week
num_epiweek=105+int(week_num) #total epiweeks
data_covidnet,cols_net= read_covidnet_data(data_path,epiweek_date,state_index,dic_names_to_abbv,start_week_n,end_week_n,step,num_epiweek)

print('excess death')
start_week_e='202001'
end_week_e= year_week_num #current epiweek
this_year=2020 #NEVER change this year, keep it always 2020, as all data started from 2020
this_month=1 #latest epiweek month in excess-death data
num_epiweek=105+int(week_num) #total epiweeks
excess_death,cols_excess=read_excess_death(data_path,epiweek_date,state_index,dic_names_to_abbv,start_week_e,end_week_e,num_epiweek,this_month,this_year)

"""
####DONOT CHANGE THESE 3, they stopped update
print('covid exposure index:dex')
start_day=20
end_day=len(epiweek_date)-42 #1 week 1 day lag
print(epiweek_date[start_day],epiweek_date[end_day])
dex,cols_d=read_dex(data_path,epiweek_date,state_index,start_day,end_day)

print('hospitalization')
last_date='2021-03-07' #yyyy-mm-dd: change to original epiweek date if pulled after Sat, else keep it as yesterday date
#states needs hosp current 2020-12-26 
state_error=['CA','DC','TX','IL','LA','PA','MI','MO','NC','NV','DE'] #'NJ', 'WA', 'NE'
ew_end='202110' #the last epiweek in string
data_hosp,cols_hosp=read_hospitalization(data_path,data_path,epiweek_date,week_save,state_index,state_error,ew_end)
"""

"""
print('emergency-visits')
start_week_v=202001
end_week_v=202105 #current epiweek
em_visit,cols_v=read_emergency(data_path,epiweek_date,state_index,start_week_v,end_week_v)"""

"""
print('kinsa')
last_date=10 #Aug 8
last_month=10
kinsa_state,cols_k=read_kinsa(kinsa_path,epiweek_date,state_index,last_date,last_month)"""

state_names=list(state_index.keys())
state_fips=read_fips_code(state_names,data_path)

#'''
#cdc_hosp={}
#cols_cdc=""

print('merging all state..')
#Change outputdir and outfilename with the path and name of out file
outputdir=data_path #"/Users/anikat/Downloads/covid-hospitalization-data/"
outfile="covid-hospitalization-daily-all-state-merged_vEW"+ year_week_num+".csv"
merge_data_state(mobility_state,apple_mobility,cdc_hosp,vacc,data_covidnet,excess_death,
                 jhu_death,survey,jhu_cases,hosp_new_res,
                 cols_m,cols_a,cols_cdc,cols_vacc,cols_net,
                 cols_excess,cols_jhu,cols_survey,cols_jhu_case,cols_hosp_new_res,
                 state_fips,epiweek,week_save,state_names,outputdir,outfile)
"""merge_data_state(mobility_state,apple_mobility,cdc_hosp,vacc,vacc_delphi,dex,kinsa_state,data_covidnet,data_hosp,excess_death,
                 jhu_death,survey,jhu_cases,hosp_new_res,
                 cols_m,cols_a,cols_cdc,cols_vacc,cols_vacc_delphi,cols_d,cols_k,cols_net,
                 cols_hosp,cols_excess,cols_jhu,cols_survey,cols_jhu_case,cols_hosp_new_res,
                 state_fips,epiweek,week_save,state_names,outputdir,outfile)"""

#


# In[158]:


def read_cdc_hosp_weekly(inputdir,epiweek_date,state_index,end_week):
    data=pd.read_csv(inputdir+"COVID-19_Reported_Timeseries.csv")
    #data=pd.read_csv(inputdir+"reported_hospital_timeseries.csv")
    
    #data=data.fillna(0)
    state_index['PR'] = 51
    state_index['VI'] = 52
    state_names=list(state_index.keys())
    flu_week_cases={}
    #cols_d=list(data.columns)
    #print(cols_d)
    #cols=cols[2:]
    flu_cols=['cdc_flu_hosp']
    
    flu_cols_to_check=['previous_day_admission_influenza_confirmed']
    for c in flu_cols:
        flu_week_cases[c]=np.empty((len(state_names),len(epiweek_date)))
    flu_week_cases[c][:][:]=np.nan
    flu_week_cases[c][0][:]=0
    
    # shift one up
    data['datetime'] = pd.to_datetime(data['date'])
    data = data.sort_values(by=['datetime'])
    data['previous_day_admission_influenza_confirmed'] = data.groupby('state')['previous_day_admission_influenza_confirmed'].shift(-1)
    data.reset_index()
    
    # data['previous_day_admission_influenza_confirmed'] = data.groupby('state')['previous_day_admission_influenza_confirmed'].shift(-1)
    # data['previous_day_admission_influenza_confirmed'] = data['previous_day_admission_influenza_confirmed'].shift(-1)

    for index,row in data.iterrows():
        if row['state'] in state_index.keys():
            state_id=state_index[row['state']]
        else:
            continue
        date=str(row['date'])
        week_id=find_week_index(epiweek_date,date.replace('/','-'),date_string=False)
        # flu_date = str(datetime.strptime(date.replace('/','-'), '%Y-%m-%d') - timedelta(days=1))
        # flu_date = flu_date[:10]
        # flu_week_id = find_week_index(epiweek_date, flu_date, date_string=False)
        if week_id!=-1:
            # same for flu hosp
            ttl=0
            flag_is_not_nan=False
            for c in flu_cols_to_check: 
                if pd.isnull(row[c])==False:
                    ttl+=float(row[c])
                    flag_is_not_nan=True
            if flag_is_not_nan:
                if np.isnan(flu_week_cases[flu_cols[0]][state_id][week_id]):
                    flu_week_cases[flu_cols[0]][state_id][week_id]=0
                flu_week_cases[flu_cols[0]][state_id][week_id]+=ttl
                flu_week_cases[flu_cols[0]][0][week_id]+=ttl
            else:
                flu_week_cases[flu_cols[0]][state_id][week_id]+=0
                
            
                    
    '''
    week_cases[c][0]=np.zeros(len(epiweek_date))
    for s in range(1,len(state_names)):
        for c in cols:
            week_cases[c][0]+=week_cases[c][s]
            #week_cases[c][s][-1]=np.nan #considering data 2 weeks lag
            #week_cases[c][s][-2]=np.nan
    '''
    #unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/cdc_hosp.csv") 
    del state_index['PR']
    del state_index['VI']
    return flu_week_cases,flu_cols


# In[157]:


def find_week_index(week,cur_date,date_string=True,strsplit='-'):
    if date_string:
        year,month,date=cur_date[:4],cur_date[4:6],cur_date[6:8]
    else:
        if strsplit=='-':
            year,month,date=cur_date.split(strsplit)
        elif strsplit=='/':
            month,date,year=cur_date.split(strsplit)
    
    cdate=int(date)
    cmonth=int(month)
    year=int(year)
    
    if year==20 or year==21 or year == 22:
        stryear='20'+str(year)
        year=int(stryear)
        
    for id in range(0,len(week)):
        y,m,d=week[id].split('-')
        wd,wm,wy=int(d),int(m),int(y)
        if wm==cmonth:
          #if wm==12:
           #   print(wm,wd,wy,id+1,len(week))
          if cdate>wd and wy==year and (id+1)<len(week):
              yn,mn,dn=week[id+1].split('-')
              wmn=int(mn)
              if wmn==(cmonth%12)+1:
                  return id+1
          elif cdate<=wd and wy==year:
              return id
    print('week index not found:'+cur_date)
    print(cdate,cmonth,year)
    return -1


# In[163]:


# generate weekly from daily 
daily_csv = pd.read_csv(outputdir+outfile)
mean_df = daily_csv.groupby(by=['epiweek','region','fips'],as_index=False, sort=False).mean()
sum_df = daily_csv.groupby(by=['epiweek','region','fips'],as_index=False, sort=False).sum(min_count=1)
last_df = daily_csv.groupby(by=['epiweek','region','fips'],as_index=False, sort=False).last()

weekly_df = pd.DataFrame()
mean_key = ['retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline', 'apple_mobility','smoothed_wcovid_vaccinated', 'smoothed_wtested_positive_14d',
       'smoothed_wwearing_mask_7d','smoothed_wspent_time_indoors_1d','fb_survey_wcli', 'fb_survey_wili']

sum_key = ['cdc_hospitalized', 'death_jhu_incidence',
       'positiveIncr', 'cdc_negativeIncr', 'cdc_positiveIncr',
       'cdc_total_resultsIncr']

last_key = ['Stage_One_Doses', 'Stage_Two_Doses','covidnet', 'Observed Number','Excess Estimate',
            'death_jhu_cumulative','positiveIncr_cumulative']
for ix, row in sum_df.iterrows(): 
    add_dict = {}
    add_dict['epiweek'] = sum_df['epiweek'][ix]
    add_dict['region'] = sum_df['region'][ix]
    add_dict['fips'] = sum_df['fips'][ix]
    for i in mean_key: 
        add_dict[i] = mean_df[i][ix]
    for i in sum_key: 
        add_dict[i] = sum_df[i][ix]
    for i in last_key: 
        add_dict[i] = last_df[i][ix]
    weekly_df = weekly_df.append(add_dict, ignore_index = True)

# import pdb
# pdb.set_trace()
weekly_df = weekly_df.reindex(columns=['epiweek','region','fips',
       'retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline', 'apple_mobility',
       'cdc_hospitalized', 'Stage_One_Doses', 'Stage_Two_Doses',
       'smoothed_wcovid_vaccinated', 'smoothed_wtested_positive_14d',
       'smoothed_wwearing_mask_7d',
       'smoothed_wspent_time_indoors_1d', 'covidnet', 'Observed Number',
       'Excess Estimate', 'death_jhu_cumulative', 'death_jhu_incidence',
       'fb_survey_wcli', 'fb_survey_wili', 'positiveIncr_cumulative',
       'positiveIncr', 'cdc_negativeIncr', 'cdc_positiveIncr',
       'cdc_total_resultsIncr'])
weekly_df['epiweek'] = weekly_df['epiweek'].astype('int')


# In[164]:


epiweek1,epiweek_date1=get_epiweek_list('202001','202053',2020)
epiweek2,epiweek_date2=get_epiweek_list('202101','202152',2021)
epiweek3,epiweek_date3=get_epiweek_list('202201',year_week_num,2022)

epiweek=epiweek1+epiweek2+epiweek3
epiweek_date=epiweek_date1+epiweek_date2+epiweek_date3
week_save=copy.deepcopy(epiweek_date)
end_week=int(week_num) #2021
flu_hosp,cols_flu=read_cdc_hosp_weekly(data_path,epiweek_date,state_index,end_week)

# cols_common=['date','epiweek','region','fips']
cols_common=['epiweek','region','fips']
all_cols = cols_common+cols_flu
final_data=pd.DataFrame(columns=all_cols)
for reg in range(len(state_names)):
    temp_data=pd.DataFrame(columns=all_cols)
    temp_data['epiweek']=epiweek
    temp_data['region']=[state_names[reg]]*len(epiweek_date)
    temp_data['fips']=[state_fips[state_names[reg]]]*len(epiweek_date)
    for c in cols_flu:
        if reg == 0: 
            flu_hosp[c][reg][0:41] = np.nan
        temp_data[c]=flu_hosp[c][reg][:]
    temp_data=temp_data[all_cols]
    final_data=final_data.append(temp_data,ignore_index=True)
final_data=final_data[all_cols]


# In[165]:


outputdir=data_path
outfile="covid-hospitalization-all-state-merged_vEW"+ year_week_num+".csv"
weekly_df = weekly_df.join(final_data['cdc_flu_hosp'])
weekly_df = weekly_df.reindex(columns=['epiweek','region','fips',
       'retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline', 'apple_mobility',
       'cdc_hospitalized', 'cdc_flu_hosp','Stage_One_Doses', 'Stage_Two_Doses',
       'smoothed_wcovid_vaccinated', 'smoothed_wtested_positive_14d',
       'smoothed_wwearing_mask_7d',
       'smoothed_wspent_time_indoors_1d', 'covidnet', 'Observed Number',
       'Excess Estimate', 'death_jhu_cumulative', 'death_jhu_incidence',
       'fb_survey_wcli', 'fb_survey_wili', 'positiveIncr_cumulative',
       'positiveIncr', 'cdc_negativeIncr', 'cdc_positiveIncr',
       'cdc_total_resultsIncr'])
weekly_df['epiweek'] = weekly_df['epiweek'].astype('int')
weekly_df.to_csv(outputdir+outfile,index=False)
