#!/usr/bin/env python
# coding: utf-8

# In[53]:


#import all necessary libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import requests
import pandas as pd
import numpy as np
import math
import glob
from datetime import datetime
from datetime import timedelta
from datetime import date

from epiweeks import Week, Year
from delphi_epidata import Epidata
from dateutil.relativedelta import relativedelta


# In[54]:


daily_csv = pd.read_csv("covid-hospitalization-daily-all-state-merged_vEW202218.csv")
daily_csv


# In[55]:


daily_csv.columns


# In[56]:


mean_df = daily_csv.groupby(by=['epiweek','region'],as_index=False, sort=False).mean()
mean_df.head(50)


# In[57]:


sum_df = daily_csv.groupby(by=['epiweek','region'],as_index=False, sort=False).sum(min_count=1)
sum_df


# In[58]:


last_df = daily_csv.groupby(by=['epiweek','region'],as_index=False, sort=False).last()
print(len(last_df))
last_df


# In[59]:


weekly_df = pd.DataFrame()
mean_key = ['retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline', 'apple_mobility','smoothed_wcovid_vaccinated', 'smoothed_wtested_positive_14d',
       'smoothed_wwearing_mask_7d', 'smoothed_wtravel_outside_state_7d',
       'smoothed_wspent_time_indoors_1d','fb_survey_wcli', 'fb_survey_wili']

sum_key = ['cdc_hospitalized', 'death_jhu_incidence',
       'positiveIncr', 'cdc_negativeIncr', 'cdc_positiveIncr',
       'cdc_total_resultsIncr']

last_key = ['Stage_One_Doses', 'Stage_Two_Doses','covidnet', 'Observed Number','Excess Estimate',
            'death_jhu_cumulative','positiveIncr_cumulative']
for ix, row in sum_df.iterrows(): 
    add_dict = {}
    add_dict['epiweek'] = sum_df['epiweek'][ix]
    print(sum_df['epiweek'][ix])
    add_dict['region'] = sum_df['region'][ix]
    for i in mean_key: 
        add_dict[i] = mean_df[i][ix]
    for i in sum_key: 
        add_dict[i] = sum_df[i][ix]
    for i in last_key: 
        add_dict[i] = last_df[i][ix]
    weekly_df = weekly_df.append(add_dict, ignore_index = True)


# In[61]:


weekly_df = weekly_df.reindex(columns=['epiweek', 'region',
       'retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline', 'apple_mobility',
       'cdc_hospitalized', 'Stage_One_Doses', 'Stage_Two_Doses',
       'smoothed_wcovid_vaccinated', 'smoothed_wtested_positive_14d',
       'smoothed_wwearing_mask_7d', 'smoothed_wtravel_outside_state_7d',
       'smoothed_wspent_time_indoors_1d', 'covidnet', 'Observed Number',
       'Excess Estimate', 'death_jhu_cumulative', 'death_jhu_incidence',
       'fb_survey_wcli', 'fb_survey_wili', 'positiveIncr_cumulative',
       'positiveIncr', 'cdc_negativeIncr', 'cdc_positiveIncr',
       'cdc_total_resultsIncr'])
weekly_df['epiweek'] = weekly_df['epiweek'].astype('int')
weekly_df.to_csv("dailytoweeklytest.csv",index=False)


# In[60]:


weekly_df.head(50)


# In[22]:


sum_df['epiweek'][100]


# In[12]:


weekly_df = pd.DataFrame()
d_dict = {'First Name': 'Vikram', 'Last Name': 'Aruchamy', 'Country': 'India'}
weekly_df = weekly_df.append(d_dict, ignore_index = True)
weekly_df


# In[21]:


daily_csv['epiweek'][100]


# In[33]:


flag_dict = {}
for ix, row in daily_csv.iterrows():
    epiweek_dict = {}
    curr_epi = row['epiweek']
    move = curr_epi
    counter = 0
    if move in flag_dict.keys(): 
        print("SAUCE")
        continue
    while move == curr_epi: 
        move = daily_csv['epiweek'][ix+counter]
        counter += 1 
    flag_dict[daily_csv['epiweek'][ix]] = True
    print(ix)
    print(move)
    print(flag_dict.keys())


# In[ ]:


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
print('CDC hospitalization')
end_week=int(week_num) #2021
cdc_hosp,cols_cdc,flu_hosp,cols_flu=read_cdc_hosp(data_path,epiweek_date,state_index,end_week)

