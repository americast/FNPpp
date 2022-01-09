#!/usr/bin/env python
# coding: utf-8

# In[31]:


'''
data sources:
1. google mobility: https://www.google.com/covid19/mobility/
2. apple mobilty: https://covid19.apple.com/mobility
6a. positiveIncrease: https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series
6b. negaiveResults, totalRes: https://healthdata.gov/dataset/COVID-19-Diagnostic-Laboratory-Testing-PCR-Testing/j8mb-icvb
7. jhu deaths: https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series
9. excess deaths: https://www.cdc.gov/nchs/nvss/vsrr/covid19/excess_deaths.htm
13. vaccine doses: https://github.com/govex/COVID-19/blob/master/data_tables/vaccine_data/us_data/time_series/vaccine_data_us_timeline.csv
5. fb-google survey: deplhi api

DATA PROVIDED BY ALEX
11. covidnet: original cdc, processed data to use given by ALEX [https://gis.cdc.gov/grasp/covidnet/COVID19_3.html]
12. CDC hospitalized: https://healthdata.gov/dataset/covid-19-reported-patient-impact-and-hospital-capacity-state-timeseries

DATA OBSOLETE-- NOT COLLECTING ANYMORE
3. Covid ExposureIndex dex: https://github.com/COVIDExposureIndices/COVIDExposureIndices
4. kinsa: kinsa_pull.ipynb
6. hospitalization: https://covidtracking.com/api
8. iqvia: Alex from Jimeng
10. Emergency visits (less priority: region level):
#https://www.cdc.gov/coronavirus/2019-ncov/covid-data/covidview/09042020/covid-like-illness.html
https://www.cdc.gov/coronavirus/2019-ncov/covid-data/covidview/10092020/outpatient-emergency-visits.html
'''


# In[60]:


import requests
import pandas as pd
import numpy as np
import math
import glob
from datetime import datetime
from datetime import timedelta
import datetime

from epiweeks import Week, Year
from delphi_epidata import Epidata


# In[33]:


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


# In[34]:


def read_survey_epidata(col_name,epidata,state_index,state_names,epiweek_date):
    week_cases=np.zeros((len(state_names),len(epiweek_date)))
    total_sample=0
    for ix in range(len(epidata)):
        row=epidata[ix]
        name=row['geo_value'].upper()
        w_idx=find_week_index(epiweek_date,str(row['time_value']))
        if name in state_names and w_idx!=-1:
            state_id=state_index[name]
            week_cases[state_id][w_idx]+=row['value']
            week_cases[0][w_idx]+=(row['value']/100)*row['sample_size']
            total_sample+=row['sample_size']

    #national
    week_cases[0]=week_cases[0]*100/total_sample

    return week_cases


# In[70]:


def read_delphi_vaccine(state_index,epiweek_date,start_week,end_week):
    #'''
    vacc11=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc21=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc31=Epidata.covidcast('fb-survey','smoothed_wwearing_mask','day','state',[start_week, Epidata.range(start_week, end_week)],'*')
    vacc41=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_5d','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')
    vacc51=Epidata.covidcast('fb-survey','smoothed_wspent_time_1d','day','state',[start_week, Epidata.range(start_week, 20210301)],'*')

    vacc12=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210301, Epidata.range(20210301, 20210501)],'*')
    vacc13=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210501, Epidata.range(20210501, 20210701)],'*')
    vacc14=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210701, Epidata.range(20210701, 20210901)],'*')
    vacc15=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210901, Epidata.range(20210901, 20211101)],'*')
    vacc16=Epidata.covidcast('fb-survey','smoothed_wcovid_vaccinated','day','state',[20210901, Epidata.range(20211101, end_week)],'*')

    vacc22=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210301, Epidata.range(20210301, 20210601)],'*')
    vacc23=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210601, Epidata.range(20210601, 20210901)],'*')
    vacc24=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210601, Epidata.range(20210901, 20211101)],'*')
    vacc25=Epidata.covidcast('fb-survey','smoothed_wtested_positive_14d','day','state',[20210601, Epidata.range(20211101, end_week)],'*')

    #vacc32=Epidata.covidcast('fb-survey','smoothed_wearing_mask','day','state',[20210301, Epidata.range(20210301, end_week)],'*')
    vacc42=Epidata.covidcast('fb-survey','smoothed_wtravel_outside_state_5d','day','state',[20210301, Epidata.range(20210301, end_week)],'*')
    vacc52=Epidata.covidcast('fb-survey','smoothed_wspent_time_1d','day','state',[20210301, Epidata.range(20210301, end_week)],'*')

    vacc1=vacc11['epidata']+vacc12['epidata']+vacc13['epidata']+vacc14['epidata']+vacc15['epidata']+vacc16['epidata']
    vacc2=vacc21['epidata']+vacc22['epidata']+vacc23['epidata']+vacc24['epidata']+vacc25['epidata']
    vacc3=vacc31['epidata']#+vacc32['epidata']
    vacc4=vacc41['epidata']+vacc42['epidata']
    vacc5=vacc51['epidata']+vacc52['epidata']

    print('smoothed_wcovid_vaccinated',vacc16['result'], vacc16['message'], len(vacc16['epidata']))
    print('smoothed_wtested_positive_14d',vacc25['result'], vacc25['message'], len(vacc25['epidata']))
    print('smoothed_wwearing_mask',vacc31['result'], vacc31['message'], len(vacc31['epidata']))
    print('smoothed_wtravel_outside_state_5d',vacc42['result'], vacc42['message'], len(vacc42['epidata']))
    print('smoothed_wspent_time_1d',vacc52['result'], vacc52['message'], len(vacc52['epidata']))
    #'''

    state_names=list(state_index.keys())
    cols=['smoothed_wcovid_vaccinated','smoothed_wtested_positive_14d','smoothed_wwearing_mask',
          'smoothed_wtravel_outside_state_5d','smoothed_wspent_time_1d']
    week_cases={}
    for c in cols:
        week_cases[c]=np.zeros((len(state_names),len(epiweek_date)))
    #'''
    week_cases[cols[0]]=read_survey_epidata(cols[0],vacc1,state_index,state_names,epiweek_date)
    week_cases[cols[1]]=read_survey_epidata(cols[1],vacc2,state_index,state_names,epiweek_date)
    week_cases[cols[2]]=read_survey_epidata(cols[2],vacc3,state_index,state_names,epiweek_date)
    week_cases[cols[3]]=read_survey_epidata(cols[3],vacc4,state_index,state_names,epiweek_date)
    week_cases[cols[4]]=read_survey_epidata(cols[4],vacc5,state_index,state_names,epiweek_date)

    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/vaccine_survey.csv")
    #'''
    return week_cases,cols


# In[36]:


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
    fb_res_cli11 = Epidata.covidcast('fb-survey', 'raw_wcli', 'day', 'state', [20211101, Epidata.range(20211101, end_week)], '*')

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
    fb_res_cli=fb_cli9+fb_res_cli11['epidata']
    #print(fb_res1['epidata'][0])
    #'''
    google_res_cli = Epidata.covidcast('google-survey', 'raw_cli', 'day', 'state', [start_week, Epidata.range(start_week, end_week)], '*')

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
    fb_res_wli11 = Epidata.covidcast('fb-survey', 'raw_wili', 'day', 'state', [20211101, Epidata.range(20211101, end_week)], '*')

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
    fb_res_wli=fb_wli9+fb_res_wli11['epidata']

    #google_res_wli = Epidata.covidcast('google-survey', 'raw_wili', 'day', 'state', [start_week, Epidata.range(start_week, end_week)], '*')

    #print('fb_cli1',fb_res_cli1['result'], fb_res_cli1['message'], len(fb_res_cli1['epidata']))
    #print('fb_cli2',fb_res_cli2['result'], fb_res_cli2['message'], len(fb_res_cli2['epidata']))
    #print('fb_cli3',fb_res_cli3['result'], fb_res_cli3['message'], len(fb_res_cli3['epidata']))
    print('fb_wcli4',fb_res_cli4['result'], fb_res_cli4['message'], len(fb_res_cli4['epidata']))
    print('fb_wcli5',fb_res_cli5['result'], fb_res_cli5['message'], len(fb_res_cli5['epidata']))
    print('fb_wcli6',fb_res_cli6['result'], fb_res_cli6['message'], len(fb_res_cli6['epidata']))
    print('fb_wcli8',fb_res_cli8['result'], fb_res_cli8['message'], len(fb_res_cli8['epidata']))
    print('fb_wcli9',fb_res_cli9['result'], fb_res_cli9['message'], len(fb_res_cli9['epidata']))

    print('google_cli',google_res_cli['result'], google_res_cli['message'], len(google_res_cli['epidata']))

    #print(fb_res_wli['result'], fb_res_wli['message'], len(fb_res_wli['epidata']))

    #print('fb_wili1',fb_res_wli1['result'], fb_res_wli1['message'], len(fb_res_wli1['epidata']))
    #print('fb_wili2',fb_res_wli2['result'], fb_res_wli2['message'], len(fb_res_wli2['epidata']))
    #print('fb_wili3',fb_res_wli3['result'], fb_res_wli3['message'], len(fb_res_wli3['epidata']))
    print('fb_wili4',fb_res_wli4['result'], fb_res_wli4['message'], len(fb_res_wli4['epidata']))
    print('fb_wili5',fb_res_wli5['result'], fb_res_wli5['message'], len(fb_res_wli5['epidata']))
    print('fb_wili6',fb_res_wli6['result'], fb_res_wli6['message'], len(fb_res_wli6['epidata']))
    print('fb_wili8',fb_res_wli7['result'], fb_res_wli8['message'], len(fb_res_wli8['epidata']))
    print('fb_wili9',fb_res_wli8['result'], fb_res_wli9['message'], len(fb_res_wli9['epidata']))

    print('fb_wcli len',len(fb_res_cli))
    print('fb_wli len',len(fb_res_wli))
    #'''
    state_names=list(state_index.keys())
    cols=['fb_survey_wcli','google_survey_cli','fb_survey_wili']
    week_cases={}
    for c in cols:
        week_cases[c]=np.zeros((len(state_names),len(epiweek_date)))
    #'''
    #week_cases[cols[0]]=read_survey_epidata(cols[0],fb_res_cli['epidata'],state_index,state_names,epiweek_date)
    week_cases[cols[0]]=read_survey_epidata(cols[0],fb_res_cli,state_index,state_names,epiweek_date)
    week_cases[cols[1]]=read_survey_epidata(cols[1],google_res_cli['epidata'],state_index,state_names,epiweek_date)
    #week_cases[cols[2]]=read_survey_epidata(cols[2],fb_res_wli['epidata'],state_index,state_names,epiweek_date)
    week_cases[cols[2]]=read_survey_epidata(cols[2],fb_res_wli,state_index,state_names,epiweek_date)
    #'''

    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/fb-google-survey.csv")
    return week_cases,cols



# In[37]:


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
'''
def read_covidnet(data_covidnet,week_len,start,end,step,weekly_rate,region):
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
                if mmr_week<=end:
                    if region=='Entire Network':
                        if row['NETWORK']=='COVID-NET':
                            covid[43+step+mmr_week]=float(row[weekly_rate])
                    else:
                        covid[43+step+mmr_week]=float(row[weekly_rate])
    return covid
'''
def get_current_week(week,cur_date,last_date,strsplit='/'):
    if strsplit=='/':
        month,date,year=cur_date.split('/')
    elif strsplit=='-':
        year,month,date=cur_date.split('-')
    cdate=int(date)
    cmonth=int(month)
    year=int(year)

    #print(cdate,cmonth,year)
    if year==20 or year==21 or year == 22:
        stryear='20'+str(year)
        year=int(stryear)

    for id in range(0,len(week)):
        y,m,d=week[id].split('-')
        wd,wm,wy=int(d),int(m),int(y)
        if wm==cmonth and wd==cdate and wy==year:
              return id
    if strsplit=='/':
        m,d,y=last_date.split('/')
    elif strsplit=='-':
        y,m,d=last_date.split('-')

    wd,wm,wy=int(d),int(m),int(y)
    if wm==cmonth and cdate==wd and wy==year:
        return len(week)-1
    #print('week index not found:'+cur_date)
    return -1

def find_same_week(week,cur_date,last_date,date_string=True):
    #date,month,year=cur_date.split('-')
    tmp_week=week
    #tmp_week[-1]=last_date
    if date_string:
        year,month,date=cur_date[:4],cur_date[4:6],cur_date[6:8]
        ly,lm,ld=last_date[:4],last_date[4:6],last_date[6:8]
    else:
        year,month,date=cur_date.split('-')
        ly,lm,ld=last_date.split('-')

    cdate=int(date)
    cmonth=int(month)
    year=int(year)

    if year==20 or year==21 or year == 22:
        stryear='20'+str(year)
        year=int(stryear)

    for id in range(0,len(tmp_week)):
        y,m,d=tmp_week[id].split('-')
        wd,wm,wy=int(d),int(m),int(y)
        if wm==cmonth and cdate==wd and wy==year:
            return id

    ldd,lmm,lyy=int(ld),int(lm),int(ly)
    if lmm==cmonth and cdate==ldd and lyy==year:
        return len(week)-1
    #print('week index not found:'+cur_date)
    return -1

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


# In[38]:


def read_apple_mobility_per_date(inputdir,epiweek_date,state_index,dic_names_to_abbv,start_week,end_week):
    data=pd.read_csv(inputdir+"applemobilitytrends.csv",low_memory=False)
    state_names=list(dic_names_to_abbv.keys())
    dates=list(data.columns)
    dates=dates[6:]
    data = data.loc[data['region'].isin(state_names)]
    data=data.fillna(0)
    #print(data.shape)
    date_dic={dates[i]: i for i in range(len(dates))}
    week_cases=np.zeros((len(state_names),len(dates)))
    for ix,row in data.iterrows():
        if row['transportation_type']=='driving':
            if row['region'] in state_names:
                state_id=state_index[dic_names_to_abbv[row['region']]]
                for d in dates:
                    #w_idx=find_week_index(epiweek_date,d,date_string=False)
                    #if w_idx!=-1:
                    week_cases[state_id][date_dic[d]]+=float(row[d])
    apple_dic={}
    apple_dic['mobility']=week_cases

    unit_test(apple_dic,['mobility'],dates,state_index,"unit_test/mobility-sampled-data.csv")

    return apple_dic,['apple_mobility']


# In[39]:


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


def read_mobility(inputdir,epiweek_date,end_week,MISSING_TOKEN=0):
    data=pd.read_csv(inputdir+"Global_Mobility_Report.csv",low_memory=False)
    data=data[data['country_region_code']=='US']
    data=data.drop(data[data['sub_region_1'] == 'Hawaii'].index)
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
        if (len(epiweek_date)-end_week)!=0:
            week_cases[c][:][end_week]=np.nan
        '''
    print(cols)
    #'''
    for ix,row in data.iterrows():
        if type(row['sub_region_1']) is float:
            state_id=0
        else:
            state_id=state_index[dic_names_to_abbv[row['sub_region_1']]]
        week_id=find_week_index(epiweek_date,str(row['date']),date_string=False)
        if week_id!=-1:
            for c in cols:
                if pd.isnull(row[c])==False:
                    if np.isnan(week_cases[c][state_id][week_id]):
                            week_cases[c][state_id][week_id]=0
                    week_cases[c][state_id][week_id]+=row[c]

    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/google-mobility.csv")
    #'''

    return week_cases,cols,state_index,dic_names_to_abbv



# In[40]:


def read_vaccine_doses(inputdir,epiweek_date,state_index,dic_names_to_abbv,last_date):
    data=pd.read_csv(inputdir+"vaccine_data_us_state_timeline.csv",low_memory=False)
    #data=data.fillna(0)
    #state_names=list(state_index.keys())
    state_names=list(dic_names_to_abbv.keys())
    #cols=['people_total','people_total_2nd_dose']
    cols=['Stage_One_Doses','Stage_Two_Doses']
    week_cases={}
    for c in cols:
        week_cases[c]=np.zeros((len(state_names),len(epiweek_date)))

    for ix,row in data.iterrows():
        #print(len(epiweek_date),row['Date'],last_date)
        w_idx=get_current_week(epiweek_date,row['Date'],last_date,strsplit='-')
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
                week_cases[c][0][w_idx]+=val

    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/vaccine.csv")

    return week_cases,cols



# In[41]:


def read_apple_mobility(inputdir,epiweek_date,state_index,dic_names_to_abbv,start_week,end_week):
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
                    w_idx=find_week_index(epiweek_date,d,date_string=False)
                    if w_idx!=-1 and pd.isnull(row[d])==False:
                        if np.isnan(week_cases[state_id][w_idx]):
                            week_cases[state_id][w_idx]=0
                        week_cases[state_id][w_idx]+=float(row[d])
    apple_dic={}
    apple_dic['apple_mobility']=week_cases

    unit_test(apple_dic,['apple_mobility'],epiweek_date,state_index,"unit_test/apple.csv")

    return apple_dic,['apple_mobility']


# In[42]:


def read_dex(inputdir,epiweek_date,state_index,start_week,end_week):
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
        if (end_week-len(epiweek_date))!=0:
            week_cases[c][:][-1]=np.nan
        '''

    for ix,row in data.iterrows():
        w_idx=find_week_index(epiweek_date,row['date'],date_string=False)
        if row['state'] in state_names and w_idx!=-1:
            state_id=state_index[row['state']]
            for c in cols:
                if not math.isnan(row[c]):
                    if np.isnan(week_cases[c][state_id][w_idx]):
                        week_cases[c][state_id][w_idx]=0
                    week_cases[c][state_id][w_idx]+=float(row[c])
                    week_cases[c][0][w_idx]+=float(row[c])

    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/dex.csv")

    return week_cases,cols


# In[43]:


def read_emergency(inputdir,epiweek_date,state_index,start_week,end_week):
    #data=pd.read_csv(inputdir+"emergency-visits.csv")
    #data=pd.read_csv(inputdir+"covid-like-illness.csv")
    data=pd.read_csv(inputdir+"covid-like-illness-v202040.csv")
    cols=['Number of Facilities Reporting','CLI Percent of Total Visits']
    print(data.columns)
    data=data[data['Week']>=start_week]
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
    print(state_index)
    list1 = state_hhs_map.keys()
    list2 = sorted(list1)
    print(list2)
    for ix,row in data.iterrows():
        reg_id=region_map[row['region']]
        week=str(row['Week'])
        w_idx=int(week[4:])-1
        year=int(week[:4])
        if year>2020:
            w_idx+=53*(year-2020)
        for st in range(len(state_names)):
            if state_hhs_map[st]==reg_id:
                for c in cols:
                    reporting=row[c]
                    if c=='Number of Facilities Reporting':
                        reporting=int(reporting.replace(',',''))
                    #print(state_names[st],w_idx,reporting)
                    week_cases[c][st][w_idx]=reporting

    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/emergency.csv")
    return week_cases,cols



# In[44]:


def read_kinsa(inputdir,epiweek_date,state_index,start_week,end_week,isregion=False):
    if isregion:
        data=pd.read_csv(inputdir+"kinsa_observedili_region_avg.csv")
    else:
        data=pd.read_csv(inputdir+"kinsa_observedili_state_avg.csv")
    state_names=list(state_index.keys())
    week_cases=np.full([len(state_names),len(epiweek_date)],np.nan)
    grouped_kinsa= data.groupby(['region'])

    for name, group in grouped_kinsa:
        state_id=state_index[name]
        #print(name)
        #print(group)
        week_cases[state_id][:end_week]=group['cases'][start_week-1:end_week]
    kinsa_dic={}
    kinsa_dic['kinsa_cases']=week_cases
    unit_test(kinsa_dic,['kinsa_cases'],epiweek_date,state_index,"unit_test/kinsa.csv")
    return kinsa_dic,['kinsa_cases']


# In[45]:


def read_iqvia(inputdir,epiweek_date,state_index,start_week,end_week):
    data=pd.read_csv(inputdir+"iqvia_Processed.csv")
    state_names=list(state_index.keys())
    week_cases={}
    grouped_iqvia= data.groupby(['region'])
    cols=list(data.columns)
    cols=cols[4:]
    print(cols)
    for c in cols:
        week_cases[c]=np.full([len(state_names),len(epiweek_date)],np.nan)

    for name, group in grouped_iqvia:
        state_id=state_index[name]
        for c in cols:
            #week_cases[c][state_id][start_week-10:end_week-start_week]=group[c]
            week_cases[c][state_id][:end_week]=group[c][start_week-1:end_week]
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/iqvia.csv")
    return week_cases,cols


# In[46]:


def read_cdc_hosp(inputdir,epiweek_date,state_index,end_week):
    data=pd.read_csv(inputdir+"COVID-19_Reported_Timeseries.csv")
    #data=pd.read_csv(inputdir+"reported_hospital_timeseries.csv")
    
    #data=data.fillna(0)
    state_names=list(state_index.keys())
    week_cases={}
    flu_week_cases={}
    #cols_d=list(data.columns)
    #print(cols_d)
    #cols=cols[2:]
    cols=['cdc_hospitalized']
    flu_cols=['cdc_flu_hosp']
    
    cols_to_check=['previous_day_admission_adult_covid_confirmed', 'previous_day_admission_pediatric_covid_confirmed']
    flu_cols_to_check=['previous_day_admission_influenza_confirmed']
    for c in cols:
        week_cases[c]=np.empty((len(state_names),len(epiweek_date)))
    for c in flu_cols:
        flu_week_cases[c]=np.empty((len(state_names),len(epiweek_date)))
    
    week_cases[c][:][:]=np.nan
    week_cases[c][0][:]=0
    flu_week_cases[c][:][:]=np.nan
    flu_week_cases[c][0][:]=0
    for index,row in data.iterrows():
        if row['state'] in state_index.keys():
            state_id=state_index[row['state']]
        else:
            continue
        date=str(row['date'])
        week_id=find_week_index(epiweek_date,date.replace('/','-'),date_string=False)
        if week_id!=-1:
            ttl=0
            flag_is_not_nan=False
            for c in cols_to_check: 
                if pd.isnull(row[c])==False:
                    ttl+=float(row[c])
                    flag_is_not_nan=True
            if flag_is_not_nan:
                if np.isnan(week_cases[cols[0]][state_id][week_id]):
                    week_cases[cols[0]][state_id][week_id]=0
                week_cases[cols[0]][state_id][week_id]+=ttl
                week_cases[cols[0]][0][week_id]+=ttl
            else:
                week_cases[cols[0]][state_id][week_id]=np.nan
            
            # same for flu hosp
            ttl=0
            flag_is_not_nan=False
            for c in flu_cols_to_check: 
                if pd.isnull(row[c])==False:
                    ttl+=float(row[c])
                    flag_is_not_nan=True
            if flag_is_not_nan:
                if np.isnan(flu_week_cases[cols[0]][state_id][week_id]):
                    flu_week_cases[cols[0]][state_id][week_id]=0
                flu_week_cases[cols[0]][state_id][week_id]+=ttl
                flu_week_cases[cols[0]][0][week_id]+=ttl
            else:
                flu_week_cases[cols[0]][state_id][week_id]=np.nan
                
            
                    
    '''
    week_cases[c][0]=np.zeros(len(epiweek_date))
    for s in range(1,len(state_names)):
        for c in cols:
            week_cases[c][0]+=week_cases[c][s]
            #week_cases[c][s][-1]=np.nan #considering data 2 weeks lag
            #week_cases[c][s][-2]=np.nan
    '''
    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/cdc_hosp.csv") 
    
    return week_cases,cols,flu_week_cases,flu_cols


# In[47]:


def read_covidnet(data_covidnet,week_len,start,end,step,weekly_rate,region): #region='X'
    print(region)
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
                if mmr_week<=end:
                    if region=='Entire Network':
                        if row['NETWORK']=='COVID-NET':
                            covid[43+step+mmr_week]=float(row[weekly_rate])
                    else:
                        covid[43+step+mmr_week]=float(row[weekly_rate])

    return covid

def read_covidnet_data(inputdir,epiweek_date,state_index,dic_names_to_abbv,start_week,end_week,step):
    #data_covidnet=pd.read_csv(inputdir+"COVID-NET_Processed.csv",delimiter=',')
    week_num = datetime.date.today().strftime("%U")
    year_week_num = "2022" + week_num
    file_name = "COVID-NET_v"+year_week_num + ".csv"
    data_covidnet=pd.read_csv(inputdir+file_name,delimiter=',')
    columns=list(data_covidnet.columns)
    dic_names_to_abbv['Entire Network']='X'
    state_names=list(dic_names_to_abbv.keys())
    #print(state_names)
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
        #print(st,state_index[st])
        week_cases[state_index[st]][:]=read_covidnet(data_covidnet,len(epiweek_date),start_week,end_week,step,weekly_rate,region=state)

    covidnet_dic={}
    covidnet_dic['covidnet']=week_cases
    unit_test(covidnet_dic,['covidnet'],epiweek_date,state_index,"unit_test/covidnet.csv")
    return covidnet_dic,['covidnet']


# In[48]:


def hosp_negative_total_data(inputdir,epiweek_date,state_index):
    data_hosp=pd.read_csv(inputdir+"COVID-19_PCR_Testing_Time_Series.csv",delimiter=',')
    state_names=list(state_index.keys())
    week_cases={}
    cols=['negativeIncr','total_resultsIncr']
    for c in cols:
        week_cases[c]=np.zeros((len(state_names),len(epiweek_date)))

    for index,row in data_hosp.iterrows():
        if row['overall_outcome']=='Negative':
            if row['state'] in state_index.keys():
                state_id=state_index[row['state']]
                date=row['date']
                week_id=find_week_index(epiweek_date,date.replace('/','-'),date_string=False)
#                 print(week_cases)[state_id][week_id]
                week_cases[cols[0]][state_id][week_id]+=float(row['new_results_reported'])
                week_cases[cols[1]][state_id][week_id]+=float(row['total_results_reported'])

                week_cases[cols[0]][0][week_id]+=float(row['new_results_reported'])
                week_cases[cols[1]][0][week_id]+=float(row['total_results_reported'])

    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/hosp_neg_ttl.csv")

    return week_cases,cols


# In[49]:


def impute_columns(orig_col,epiweek_date):
    d_len=len(epiweek_date)-3
    imputed_col=np.empty(len(epiweek_date))
    imputed_col[-2:]=orig_col[-2:]
    while d_len>=0:
        if orig_col[d_len]==0 or orig_col[d_len]==np.nan:
            if orig_col[d_len+2]!=0: #geometric mean a,b,c a=0,c!=0
                imputed_col[d_len]=np.square(orig_col[d_len+1])/orig_col[d_len+2]
            #else orig_col[d_len+1]!=0:: #arithmetic mean a=0, c==0
             #   imputed_col[d_len]=2*orig_col[d_len+1]-orig_col[d_len+2]
        d_len-=1
    return imputed_col


# In[50]:


def hosp_cases_nat_state(data_national,data_state,epiweek_date,week_save,state_index,cols,state_error,last_date):
    state_names=list(state_index.keys())
    week_cases={}
    for c in cols:
        week_cases[c]=np.zeros((len(state_names),len(epiweek_date)))

    #processing national cases
    nat_id=state_index['X']
    nat_hosp_incr=np.zeros(len(epiweek_date))
    for index,row in data_national.iterrows():
        week_id=find_week_index(week_save,str(row['date']))
        for c in range(len(cols)): #if next1,next2 consider then do cols-4
            week_cases[cols[c]][nat_id][week_id]+=row[cols[c]]

    #processing state cases
    for index,row in data_state.iterrows():
        state_id=state_index[row['states']]
        week_id=find_week_index(week_save,str(row['date']))
        if week_id!=-1:
            for c in range(len(cols)): #if next1,next2 consider then do cols-4
                if cols[c]=='hospitalizedIncrease':
                    if row['states'] in state_error:
                        week_id_for_hsp=find_same_week(epiweek_date,str(row['date']),last_date)
                        if week_id_for_hsp!=-1:
                            week_cases[cols[c]][state_id][week_id_for_hsp]=max(0,row['hospitalizedCurrently'])
                    else:
                        week_cases[cols[c]][state_id][week_id]+=max(0,row[cols[c]])
                        nat_hosp_incr[week_id]+=max(0,row[cols[c]])
                    #week_id_for_hsp=find_same_week(epiweek_date,str(row['date']),last_date)
                    #if week_id_for_hsp!=-1:
                     #   week_cases[cols[c]][state_id][week_id_for_hsp]=max(0,row['hospitalizedCurrently'])
                elif cols[c]=='recovered':
                    week_id_for_rec=find_same_week(epiweek_date,str(row['date']),last_date)
                    if week_id_for_rec!=-1:
                        week_cases[cols[c]][state_id][week_id_for_rec]=max(0,row[cols[c]])
                else:
                    week_cases[cols[c]][state_id][week_id]+=max(0,row[cols[c]])

    #for states with no hospitalze cumulative using formula week[t]=hosp_cur[t]-(week[t-1]/2)

    c='hospitalizedIncrease'
    for s in state_error:
    #for s in state_names:
        st_idx=state_index[s]
        hsp_incr=np.zeros(len(epiweek_date))
        hsp_incr[0]=week_cases[c][st_idx][0]
        nat_hosp_incr[0]+=week_cases[c][st_idx][0]
        for w in range(1,len(epiweek_date)):
            hsp_incr[w]=max(0,week_cases[c][st_idx][w]-float(hsp_incr[w-1]/2))
            nat_hosp_incr[w]+=hsp_incr[w]
            #if s=='CA':
             #   print(w,week_cases[c][st_idx][w],hsp_incr[w-1],hsp_incr[w])
        week_cases[c][st_idx]=hsp_incr

    c='recovered'
    rec_nat=np.zeros(len(epiweek_date))
    for s in state_names:
        st_idx=state_index[s]
        rec_incr=np.zeros(len(epiweek_date))
        rec_incr[0]=week_cases[c][st_idx][0]
        for w in range(1,len(epiweek_date)):
            rec_incr[w]=week_cases[c][st_idx][w]-week_cases[c][st_idx][w-1]
            rec_nat[w]+=max(rec_incr[w],0)
        week_cases[c][st_idx]=rec_incr

    week_cases[c][nat_id]=rec_nat

    '''
    #for data imputation
    for st in range(len(state_names)):
        for c in range(0,len(cols)):#if next1,next2 consider then do cols-4
            if c<3 or c>4: #avoiding inVentilation, inICU for imputation
                imputed_week=impute_columns(week_cases[cols[c]][st],epiweek_date)
                week_cases[cols[c]][st]=imputed_week
    '''
    '''
    #filling up next1,next2 values
    h_next1=np.empty((len(state_names),len(epiweek_date)))
    h_next2=np.empty((len(state_names),len(epiweek_date)))
    d_next1=np.empty((len(state_names),len(epiweek_date)))
    d_next2=np.empty((len(state_names),len(epiweek_date)))

    h_next1[:,:]=np.nan
    h_next2[:,:]=np.nan
    d_next1[:,:]=np.nan
    d_next2[:,:]=np.nan
    for st in range(len(state_names)):
        for widx in range(0,len(epiweek_date)-1):
            h_next1[st][widx]=week_cases['hospitalizedIncrease'][st][widx+1]
            d_next1[st][widx]=week_cases['deathIncrease'][st][widx+1]
            if widx+2<len(epiweek_date):
                h_next2[st][widx]=week_cases['hospitalizedIncrease'][st][widx+2]
                d_next2[st][widx]=week_cases['deathIncrease'][st][widx+2]
        week_cases['h_next1'][st]=h_next1[st]
        week_cases['h_next2'][st]=h_next2[st]
        week_cases['d_next1'][st]=d_next1[st]
        week_cases['d_next2'][st]=d_next2[st]
    '''
    '''
    out=pd.DataFrame(columns=['date','states']+cols)
    for st in state_index.keys():
        tmp_out=pd.DataFrame(columns=['date','states']+cols)
        tmp_out['date']=week_save
        tmp_out['states']=[st]*len(epiweek_date)
        for c in cols:
            tmp_out[c]=week_cases[c][state_index[st]]
        out=out.append(tmp_out,ignore_index=True)

    out.to_csv("/Users/anikat/Downloads/covid-hospitalization-data/hosp_check.csv",index=False)
    '''
    np.savetxt("unit_test/hos_nat_aggregated.csv", nat_hosp_incr, fmt='%.4f')
    return week_cases


# In[51]:


def read_hospitalization(input_national,input_state,epiweek_date,week_save,state_index,
                         last_date,state_error):
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

    week_cases=hosp_cases_nat_state(nat_data_filter,state_data_filter,epiweek_date,week_save,state_index,cols_for_hosp,state_error,last_date)

    unit_test(week_cases,cols_for_hosp,epiweek_date,state_index,"unit_test/hospitalization.csv")
    return week_cases,cols_for_hosp


# In[52]:


def read_excess_death(inputdir,epiweek_date,state_index,dic_names_to_abbv,start_week,end_week,this_month,last_date):
    data=pd.read_csv(inputdir+"Excess_Deaths_COVID-19.csv")
#     cols_death=['Observed Number','Excess Higher Estimate']
    cols_death = ['Observed Number','Excess Estimate']
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
        if name in state_names and cy>=2020: #and cm<=this_month:
            #print(name)
            #print(group)
            cur_date=y+'-'+m+'-'+d
            w_idx=find_same_week(epiweek_date,cur_date,last_date,date_string=False)
            if w_idx!=-1:
                state_id=state_index[dic_names_to_abbv[name]]
                for c in cols_death:
                    #if name=='Alabama':
                     #   print(name,cur_date,w_idx,row[c])
                    week_cases[c][state_id][w_idx]+=int(row[c])
                    week_cases[c][0][w_idx]+=int(row[c])


    ##ADD NAN VALUES TO THE LAST 2 WEEK
    for s in range(len(state_names)):
        for c in cols_death:
            week_cases[c][s]/=3
            week_cases[c][s][-1]=np.nan
            week_cases[c][s][-2]=np.nan


    unit_test(week_cases,cols_death,epiweek_date,state_index,"unit_test/excess-death-weekly.csv")
    return week_cases,cols_death


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
        week_cases[c]=np.zeros((len(state_names),len(epiweek_date)))

    for ix, row in data.iterrows():
        name=row['Province_State']
        if name in state_names:
            state_id=state_index[dic_names_to_abbv[name]]
            for date in dates_list:
                w_idx=get_current_week(epiweek_date,date,dates_list[-1])
                #if date==dates_list[-1]:
                 #   print(w_idx)
                if w_idx!=-1:
                    #print(name,date,row[date])
                    week_cases[cols[0]][state_id][w_idx]+=int(row[date])
                    week_cases[cols[0]][0][w_idx]+=int(row[date])
        elif name not in not_added_rows: #if not state_names still adding them for national
            #print('state name not added:'+name)
            for date in dates_list:
                w_idx=get_current_week(epiweek_date,date,dates_list[-1])
                if w_idx!=-1:
                    week_cases[cols[0]][0][w_idx]+=int(row[date])


    #count incidence for the national+states
    for state_id in range(len(state_names)):
        week_cases[cols[1]][state_id][0]=int(week_cases[cols[0]][state_id][0])
        for w_idx in range(1,len(epiweek_date)):
            week_cases[cols[1]][state_id][w_idx]=int(week_cases[cols[0]][state_id][w_idx])-int(week_cases[cols[0]][state_id][w_idx-1])


    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/jhu-cases.csv")
    return week_cases,cols


# In[54]:


def read_jhu_death(inputdir,epiweek_date,state_index,dic_names_to_abbv,start_week=None,end_week=None):
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
        week_cases[c]=np.zeros((len(state_names),len(epiweek_date)))

    for ix, row in data.iterrows():
        name=row['Province_State']
        if name in state_names:
            state_id=state_index[dic_names_to_abbv[name]]
            for date in dates_list:
                w_idx=get_current_week(epiweek_date,date,dates_list[-1])
                #if date==dates_list[-1]:
                 #   print(w_idx)
                if w_idx!=-1:
                    #print(name,date,row[date])
                    week_cases[cols[0]][state_id][w_idx]+=int(row[date])
                    week_cases[cols[0]][0][w_idx]+=int(row[date])
        elif name not in not_added_rows: #if not state_names still adding them for national
            #print('state name not added:'+name)
            for date in dates_list:
                w_idx=get_current_week(epiweek_date,date,dates_list[-1])
                if w_idx!=-1:
                    week_cases[cols[0]][0][w_idx]+=int(row[date])


    #count incidence for the national+states
    for state_id in range(len(state_names)):
        week_cases[cols[1]][state_id][0]=int(week_cases[cols[0]][state_id][0])
        for w_idx in range(1,len(epiweek_date)):
            week_cases[cols[1]][state_id][w_idx]=int(week_cases[cols[0]][state_id][w_idx])-int(week_cases[cols[0]][state_id][w_idx-1])


    unit_test(week_cases,cols,epiweek_date,state_index,"unit_test/jhu-death.csv")
    return week_cases,cols


# In[55]:


def region_level_cases(inputdir,week_state,state_index,epiweek_date,cols, is_covidnet=False):
    otherdatasir=inputdir+"other_data/"
    week_hhs={}
    num_region=11
    state_names=list(state_index.keys())
    for c in cols:
        week_hhs[c]=np.zeros((num_region,len(epiweek_date)))

    file = open(otherdatadir+"hhs_regions_abbv.txt", 'r')
    Lines = file.readlines()
    state_hhs_map={}
    state_pop=np.zeros(len(state_names))
    population=open(otherdatadir+"state_population.csv",'r')
    total=0
    popu_region=np.zeros(num_region)

    ######## Getting population. Need this for weighted count ###########################

    Lines_pop=population.readlines()
    for line in Lines_pop:
        state,popu,abbv=line.strip().split(',')
        if state=='ttl':
            state_pop[0]=float(popu)
        else:
            if state_index.get(abbv,-1)!=-1:
                state_id=state_index[abbv]
                state_pop[state_id]=float(popu)
            else:
                print('population state not found '+state)

    #### mapping each state to a hhs region ###
    state_hhs_map[0]=0 # setting national value
    popu_region[0]=1
    for line in Lines:
        regions=line.strip().split(',')
        reg_id=int(regions[0])
        for j in range(1,len(regions)):
            if state_index.get(regions[j],-1)!=-1:
                state_id=state_index[regions[j]]
                state_hhs_map[state_id]=reg_id
                popu_region[reg_id]+=state_pop[state_id]
            else:
                print('state not found '+regions[j])

    for key in state_hhs_map.keys():
        hhs_id=int(state_hhs_map[int(key)])
        for j in range(len(epiweek_date)):
            for c in cols:
                if is_covidnet and int(key)>0:
                    if math.isnan(week_state[c][int(key)][j])==False:
                        #print(int(key),week_state[c][int(key)][j])
                        week_hhs[c][hhs_id][j]+=(week_state[c][int(key)][j]*state_pop[int(key)])/100000
                else:
                    week_hhs[c][hhs_id][j]+=week_state[c][int(key)][j]
    if is_covidnet:
        for r in range(1,num_region):
            for c in cols:
                week_hhs[c][r][:]/=popu_region[r]

    return week_hhs


# In[56]:


def merge_data_region(mobility,kinsadir,iqvia,covidnet,hosp,cols_m,cols_k,cols_q,cols_net,cols_hosp,epiweek,
               epiweek_date,outputdir,otherdatadir,outfilename,state_index,kinsa_start,kinsa_end):
    r='Region '
    region_names=['X']
    for i in range(1,11):
        region_names.append(r+str(i))
    region_index={region_names[i]:i for i in range(len(region_names))}
    cols_common=['date','epiweek','region']
    all_cols=cols_common+cols_m+cols_k+cols_q+cols_net+cols_hosp

    mobility_hhs=region_level_cases(otherdatadir,mobility,state_index,epiweek_date,cols_m)
    kinsa_hhs,cols_kinsa=read_kinsa(kinsadir,epiweek_date,region_index,kinsa_start,kinsa_end,isregion=True)
    iqvia_hhs=region_level_cases(otherdatadir,iqvia,state_index,epiweek_date,cols_q)
    covidnet_hhs=region_level_cases(otherdatadir,covidnet,state_index,epiweek_date,cols_net,is_covidnet=True)
    hosp_hhs=region_level_cases(otherdatadir,hosp,state_index,epiweek_date,cols_hosp)

    #print(covidnet_hhs[cols_net[0]][1:])

    final_data=pd.DataFrame(columns=all_cols)
    for reg in range(len(region_names)):
        temp_data=pd.DataFrame(columns=all_cols)
        temp_data['date']=epiweek_date
        temp_data['epiweek']=epiweek
        temp_data['region']=[region_names[reg]]*len(epiweek_date)
        for c in cols_m:
            temp_data[c]=mobility_hhs[c][reg][:]
        for c in cols_kinsa:
            temp_data[c]=kinsa_hhs[c][reg][:]
        for c in cols_q:
            temp_data[c]=iqvia_hhs[c][reg][:]
        for c in cols_net:
            temp_data[c]=covidnet_hhs[c][reg][:]
        for c in cols_hosp:
            temp_data[c]=hosp_hhs[c][reg][:]

        temp_data=temp_data[all_cols]
        final_data=final_data.append(temp_data,ignore_index=True)
    final_data=final_data[all_cols]
    print(final_data.shape)
    final_data.to_csv(outputdir+outfilename,index=False)

    print('FINISHED....')



# In[57]:


def merge_data_state(mobility,apple,vacc,vac_delphi,cdc_hosp,flu_hosp,dex,kinsa,covidnet,hosp,excess,jhu,survey,em_visit,jhu_case,hosp_new_res,
                     cols_m,cols_a,cols_vacc,cols_vac_delphi,cols_cdc,cols_flu,cols_d,cols_k,cols_net,
                     cols_hosp,cols_excess,cols_jhu,cols_survey,cols_v,cols_jhu_case,cols_hosp_new_res,
                     state_fips,epiweek,epiweek_date,region_names,outputdir,outfilename):

    cols_common=['date','epiweek','region','fips']
    #all_cols=cols_common+cols_m+cols_a+cols_d+cols_k+cols_net+cols_hosp+cols_excess+cols_jhu+cols_survey+cols_v
    #all_cols=cols_common+cols_m+cols_a+cols_cdc+cols_d+cols_k+cols_q+cols_net+cols_hosp+cols_excess+cols_jhu+cols_survey+cols_v
    all_cols=cols_common+cols_m+cols_a+cols_cdc+cols_d+cols_vacc+cols_vac_delphi+cols_k+cols_net+cols_hosp+cols_excess+cols_jhu+cols_survey+cols_v+cols_jhu_case+cols_hosp_new_res

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
            temp_data[c]=cdc_hosp[c][reg][:]
        for c in cols_flu:
            temp_data[c]=flu_hosp[c][reg][:]
        for c in cols_d:
            temp_data[c]=dex[c][reg][:]
        for c in cols_vacc:
            temp_data[c]=vacc[c][reg][:]
        for c in cols_vacc_delphi:
            temp_data[c]=vacc_delphi[c][reg][:]
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
        for c in cols_v:
            temp_data[c]=em_visit[c][reg][:]
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



if __name__ == "__main__":

    # In[71]:


    '''
    ### WHAT DATA ##
    1.To get the hospitalization data we need the following datasets:
      a. mobility (filename:Global_Mobility_Report.csv)
      b.kinsa (filename:kinsa_observedili_state_avg.csv)
      c.iqvia (filename:iqvia_Processed.csv)
      d.covidnet (filename:COVID-NET_Processed.csv)
      e.covid-tracking (national, state)
          datasets are collected from site: https://covidtracking.com/api
            i. We collected US historical data (filename save: us-daily-hospitalizations.csv)
            ii. State historical data (filename save: states-daily-hospitalizations.csv)
      f. apple (applemobilitytrends.csv)
      g. dex (state-dex.csv)
      h. emergency visit (emeregency-vists.csv)
      i. jhu deaths: time_series_covid19_deaths_US.csv
      j. excess deaths: Excess_Deaths_COVID_19.csv

    ## WHERE DATA SHOULD BE KEPT AND WHAT NAME ###
    2. change all data directory as where it is kept
    3. DONT CHANGE FILENAME (keep in the same name as written above)
    4. There are some additional data like population, state-fips code etc required to process this file,
      keep all of them in same directory as data_path/other_data

    ### INPUT PARAMETERS TO PASS FOR PROCESSING EACH FILE ##
    5. At the beginning of every data to be processes also pass a input parameter as start_week/end_week or
       both based on dataset. This is for consistency as not all dataset contain all epiweeks value

    6. change get_epiweek_list(start_week,end_week) also. These are the weeks to show in the final output file

    7. for hospitalization only change last_date instead of start/end_week.
       last_date means the date which upto which data is available

    ##OUTPUT FILE NAME AND DIRECTORY##
    8. merged_data(..) is merging all the columns. Change the filename and output directory in the fields
       before calling the method

    '''
    import copy
    week_num = datetime.date.today().strftime("%U")
    year_week_num = "2022" + week_num
    week_end_date = (datetime.date.today() + datetime.timedelta((5-datetime.date.today().weekday()) % 7 )).strftime('%Y-%m-%d')
    week_end_string = (datetime.date.today() + datetime.timedelta((5-datetime.date.today().weekday()) % 7 )).strftime('%Y%m%d')
    ### INPUT FILE DIRECTORY #####
    # data_path="/Users/anikat/Downloads/covid-hospitalization-data/"
    data_path="./"
    # kinsa_path= "/Users/anikat/Downloads/kinsahealth/"
    kinsa_path = "./"
    ### INPUT: EPIWEEK START, END WEEK ####
    #Both the method args will be same. This is to keep last date in the ouput file to same as epiweek
    epiweek1,epiweek_date1=get_epiweek_list('202001','202053',2020)
    epiweek2,epiweek_date2=get_epiweek_list('202101','202152',2021)
    epiweek3,epiweek_date3=get_epiweek_list('202201',year_week_num,2022)




    epiweek=epiweek1+epiweek2+epiweek3
    epiweek_date=epiweek_date1+epiweek_date2+epiweek_date3
    week_save=copy.deepcopy(epiweek_date)
    ### ****NOTE: for new year 2021 add another epiweek list and append with previous**** #######

    print(epiweek_date)
    print(epiweek)
    # PROCESSING MOBILITY ####
    # INPUT: END_WEEK: the last epiweek this csv file contains  ####
    print('google mobility')
    start_week_m=1
    end_week_m=int(week_num) #2021, m=epiweek since data generally have value -4 days lag for current week
    mobility_state,cols_m,state_index,dic_names_to_abbv=read_mobility(data_path,epiweek_date,end_week_m)
    dic_names_to_abbv['United States']='X'
    #print(dic_names_to_abbv)
    print(len(state_index.keys()))
    #print(state_index)

    print('JHU-cases')
    jhu_cases,cols_jhu_case=read_jhu_cases(data_path,epiweek_date,state_index,dic_names_to_abbv)

    print('hosp-neg-total result')
    hosp_new_res,cols_hosp_new_res=hosp_negative_total_data(data_path,epiweek_date,state_index)

    print('Vaccine dose')
    last_date=week_end_date
    vacc,cols_vacc=read_vaccine_doses(data_path,epiweek_date,state_index,dic_names_to_abbv,last_date)

    print('delphi vaccine survey')
    start_week=20210101
    end_week=int(week_end_string)  #yyyymmdd

    vacc_delphi,cols_vacc_delphi=read_delphi_vaccine(state_index,epiweek_date,start_week,end_week)

    ### PROCESSING FB-GOOGLE ####
    ### INPUT: start_week, end_week; start_week does not matter as both survey starts from 20200401,
    ###  but always update end_week###
    print('FB-GOOGLE')
    start_week=20200307
    end_week=int(week_end_string) #yyyymmdd
    survey,cols_survey=read_delphi_fb_google_survey(state_index,epiweek_date,start_week,end_week)


    print('CDC hospitalization')
    end_week=int(week_num) #2021
    cdc_hosp,cols_cdc,flu_hosp,cols_flu=read_cdc_hosp(data_path,epiweek_date,state_index,end_week)

    ### PROCESSING APPLE MOBILITY per date for Berkely guys####
    #read_apple_mobility_per_date(data_path,epiweek_date,state_index,dic_names_to_abbv,start_week_m,end_week_m)

    ### PROCESSING APPLE MOBILITY ####
    ### INPUT: START_WEEK:1,END_WEEK: the last epiweek this csv file contains  ####
    print('apple mobility')
    end_week_m=int(week_num)
    apple_mobility,cols_a=read_apple_mobility(data_path,epiweek_date,state_index,dic_names_to_abbv,start_week_m,end_week_m)


    ### PROCESSING COVIDNET ####
    ### INPUT: START, END_WEEK: have 1 weeks lag data, same as IQVIA; STEP: TILL which index covidnet starts###
    print('covidnet')
    step=9 #if epiweek starts from 10 then step=0, if epiweek starts from 1 step=9 (10-epiweek_start)
    start_week_n=10
    end_week_n=int(week_num) #0 is just for week 53, change to 1, since next week change it to epiweek-1 if data pulled after Sat
    data_covidnet,cols_net= read_covidnet_data(data_path,epiweek_date,state_index,dic_names_to_abbv,start_week_n,end_week_n,step)


    ### PROCESSING excess death ####
    ### INPUT: start_week, end_week, this_month: number of current month###
    print('excess death')
    start_week_e=1
    end_week_e=int(week_num) #epiweek
    this_month=1
    last_date=week_end_date #yyyy-mm-dd change to original epiweek date if pulled after Sat, else keep it as today's date
    excess_death,cols_excess=read_excess_death(data_path,epiweek_date,state_index,dic_names_to_abbv,start_week_e,end_week_e,this_month,last_date)

    ### PROCESSING jhu death ####
    print('JHU')
    jhu_death,cols_jhu=read_jhu_death(data_path,epiweek_date,state_index,dic_names_to_abbv)


    ##DO NOT CHANGE THESE 3, they stopped update

    ### PROCESSING Covid exposure indices on income, race,etc. ####
    ### INPUT: END_WEEK: the last epiweek this csv file contains  ####
    print('covid exposure index:dex')
    end_week_dex=16
    dex,cols_d=read_dex(data_path,epiweek_date,state_index,start_week_m,end_week=end_week_dex)

    ### PROCESSING KINSA ####
    print('kinsa')
    start_week_k=1
    end_week_k=40 #if this data is 1 week lag, then epiweek-1, else epiweek
    kinsa_state,cols_k=read_kinsa(kinsa_path,epiweek_date,state_index,start_week_k,end_week_k)

    ## PROCESSING HOSPITALIZATION ####
    print('hospitalization')
    last_date='20210307' #yyymmdd:change to original epiweek date if pulled after Sat, else keep it as yesterday date
    #states needs hosp current
    state_error=['CA','DC','TX','IL','LA','PA','MI','MO','NC','NV','DE'] #'NJ', 'WA', 'NE'
    data_hosp,cols_hosp=read_hospitalization(data_path,data_path,epiweek_date,week_save,state_index,
                                          last_date,state_error)

    # ## PROCESSING IQVIA ####
    # ## INPUT: START, END_WEEK: have 2 weeks lag data###
    # print('iqvia')
    # change start, end week based on data and epiweek_date. current data has value since epiweek 10-17
    # start_week_q=1
    # end_week_q=33 #if this data is 1 week lag, then epiweek-1, else epiweek
    # iqvia_state,cols_q=read_iqvia(data_path,epiweek_date,state_index,start_week_q,end_week_q)

    ## PROCESSING Emergency-visits ####
    print('emergency-visits')
    start_week_v=202001
    end_week_v=202105
    em_visit,cols_v=read_emergency(data_path,epiweek_date,state_index,start_week_v,end_week_v)

    state_names=list(state_index.keys())
    state_fips=read_fips_code(state_names,data_path)


    # #'''
    # #cdc_hosp={}
    # #cols_cdc=""
    print('merging all state..')
    #Change outputdir and outfilename with the path and name of out file
    outputdir=data_path #"/Users/anikat/Downloads/covid-hospitalization-data/"
    outfile="covid-hospitalization-all-state-merged_vEW" + year_week_num+".csv"
    merge_data_state(mobility_state,apple_mobility,vacc,vacc_delphi,cdc_hosp,flu_hosp,dex,kinsa_state,data_covidnet,data_hosp,excess_death,
                     jhu_death,survey,em_visit,jhu_cases,hosp_new_res,
                     cols_m,cols_a,cols_vacc,cols_vacc_delphi,cols_cdc,cols_flu,cols_d,cols_k,cols_net,cols_hosp,cols_excess,
                     cols_jhu,cols_survey,cols_v,cols_jhu_case,cols_hosp_new_res,
                     state_fips,epiweek,week_save,state_names,outputdir,outfile)




