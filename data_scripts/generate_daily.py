"""
    Generates both daily and weekly
"""

from data_generation import *

''' TODO: check all files are in place, e.g. COVIDNet'''

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

print(dic_names_to_abbv)
print(state_index)
print('google symptoms')
symptom, cols_symptom = read_google_symptoms(data_path,epiweek_date,state_index,dic_names_to_abbv)
print(cols_symptom)


print('JHU-cases')
jhu_cases,cols_jhu_case=read_jhu_cases(data_path,epiweek_date,state_index,dic_names_to_abbv)


print('hosp-neg-total result')
hosp_new_res,cols_hosp_new_res=hosp_negative_total_data_test(data_path,epiweek_date,state_index)


print('FB-GOOGLE')
start_week=20210101
start_week2 = 20200307
end_week=int(week_end_string) #yyyymmdd
# print('--skip FB')
# survey, cols_survey = read_fb(state_index,epiweek_date,start_week,end_week,start_week2) 

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

# print('apple mobility')
# apple_mobility,cols_a=read_apple_mobility(data_path,epiweek_date,state_index,dic_names_to_abbv)

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
# merge and save
merge_data_state(mobility_state,cdc_hosp,vacc,data_covidnet,excess_death,
                 jhu_death,jhu_cases,hosp_new_res,symptom,
                 cols_m,cols_cdc,cols_vacc,cols_net,
                 cols_excess,cols_jhu,cols_jhu_case,cols_hosp_new_res,cols_symptom,
                 state_fips,epiweek,week_save,state_names,outputdir,outfile)

