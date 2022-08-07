# import requests
# import pandas as pd
# import numpy as np
# import math
# import glob
# import copy
# from datetime import datetime
# from datetime import timedelta
# from datetime import date

# from epiweeks import Week, Year
# from delphi_epidata import Epidata
# from dateutil.relativedelta import relativedelta

from data_generation import *
import os
import pdb

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

print('merging all state..')
#Change outputdir and outfilename with the path and name of out file
outputdir=data_path #"/Users/anikat/Downloads/covid-hospitalization-data/"
# year_week_num = '202227' # for debuging
outfile="covid-hospitalization-daily-all-state-merged_vEW"+ year_week_num+".csv"

# pdb.set_trace()
if not os.path.exists(outputdir+outfile):
    raise Exception('daily file not found -- run daily script first')

# generate weekly from daily 
daily_csv = pd.read_csv(outputdir+outfile)
# changes US to 0 so it is an int type and converts the whole column to the int type
daily_csv['fips'] = daily_csv['fips'].replace('US', '0')
daily_csv = daily_csv.astype({'fips': 'int'})
mean_df = daily_csv.groupby(by=['epiweek','region','fips'],as_index=False, sort=False).mean()
sum_df = daily_csv.groupby(by=['epiweek','region','fips'],as_index=False, sort=False).sum(min_count=1)
last_df = daily_csv.groupby(by=['epiweek','region','fips'],as_index=False, sort=False).last()
# convert the 0 back into “US”
mean_df['fips'] = mean_df['fips'].replace(0, 'US')
sum_df['fips'] = sum_df['fips'].replace(0, 'US')
last_df['fips'] = last_df['fips'].replace(0, 'US')

weekly_df = pd.DataFrame()
mean_key = ['retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline']

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

weekly_df = weekly_df.reindex(columns=['epiweek','region','fips',
       'retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline',
       'cdc_hospitalized', 'Stage_One_Doses', 'Stage_Two_Doses',
       'covidnet', 'Observed Number',
       'Excess Estimate', 'death_jhu_cumulative', 'death_jhu_incidence',
       'positiveIncr_cumulative',
       'positiveIncr', 'cdc_negativeIncr', 'cdc_positiveIncr',
       'cdc_total_resultsIncr'])
weekly_df['epiweek'] = weekly_df['epiweek'].astype('int')


# In[164]:

"""
    Add CDC flu hosp data
"""
state_index = get_state_index(data_path)
state_names=list(state_index.keys())
state_fips=read_fips_code(state_names,data_path)

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
       'residential_percent_change_from_baseline', 
       'cdc_hospitalized', 'cdc_flu_hosp','Stage_One_Doses', 'Stage_Two_Doses',
       'covidnet', 'Observed Number',
       'Excess Estimate', 'death_jhu_cumulative', 'death_jhu_incidence',
       'positiveIncr_cumulative',
       'positiveIncr', 'cdc_negativeIncr', 'cdc_positiveIncr',
       'cdc_total_resultsIncr'])

weekly_df['epiweek'] = weekly_df['epiweek'].astype('int')
weekly_df.to_csv(outputdir+outfile,index=False)


# weekly_df.groupby(['region']).count()
