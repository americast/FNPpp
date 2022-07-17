#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
import numpy as np
import math
import glob
from datetime import datetime
from datetime import timedelta
import os

from epiweeks import Week, Year
from delphi_epidata import Epidata


def get_datasets():
    # apple mobility needs to be downloaded manually, COVIDNET and CDC hospitalization sent by Alex
    # rename the file names
    for filename in os.listdir("."):
        if filename.startswith("applemobilitytrends-"):
            if os.path.exists("./applemobilitytrends.csv"):
                os.remove('applemobilitytrends.csv')
            os.rename(filename, 'applemobilitytrends.csv')
            print("Apple Mobility Data ... DONE")
        """elif filename.startswith("COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries"):
            #os.remove("COVID-19_Reported_Timeseries.csv")
            os.rename(filename, "COVID-19_Reported_Timeseries.csv")
            print("CDC Hospitalization Data ... DONE")"""

    # get google global mobility
    url = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
    response = requests.get(url).content
    if os.path.exists("Global_Mobility_Report.csv"):
        os.remove("Global_Mobility_Report.csv")
    mobility_csv = open("Global_Mobility_Report.csv", 'wb')
    mobility_csv.write(response)
    mobility_csv.close()
    print("Google Mobility Data ... DONE")

    # apple has to be manually retrieved since link changes daily

    #get CDC hospitalization data
    url = "https://healthdata.gov/api/views/g62h-syeh/rows.csv?accessType=DOWNLOAD"
    response = requests.get(url).content
    if os.path.exists("COVID-19_Reported_Timeseries.csv"):
        os.remove("COVID-19_Reported_Timeseries.csv")
    hospital_csv = open("COVID-19_Reported_Timeseries.csv",'wb')
    hospital_csv.write(response)
    hospital_csv.close()
    print("CDC Hospitalization Data ... DONE")

    # get jhu casecount data
    url = "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
    response = requests.get(url).content
    if os.path.exists("time_series_covid19_confirmed_US.csv"):
        os.remove("time_series_covid19_confirmed_US.csv")
    jhu_confirmed_csv = open("time_series_covid19_confirmed_US.csv", 'wb')
    jhu_confirmed_csv.write(response)
    jhu_confirmed_csv.close()
    print("JHU Case count data ... DONE")

    # get jhu deaths data
    url = "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"
    response = requests.get(url).content
    if os.path.exists("time_series_covid19_deaths_US.csv"):
        os.remove("time_series_covid19_deaths_US.csv")
    jhu_deaths_csv = open("time_series_covid19_deaths_US.csv", 'wb')
    jhu_deaths_csv.write(response)
    jhu_deaths_csv.close()
    print("JHU Deaths data ... DONE")

    # get pcr test data
    url = "https://healthdata.gov//api/views/j8mb-icvb/rows.csv?accessType=DOWNLOAD"
    response = requests.get(url).content
    if os.path.exists("COVID-19_PCR_Testing_Time_Series.csv"):
        os.remove("COVID-19_PCR_Testing_Time_Series.csv")
    pcr_test_csv = open("COVID-19_PCR_Testing_Time_Series.csv", 'wb')
    pcr_test_csv.write(response)
    pcr_test_csv.close()
    print("PCR Testing data ... DONE")

    # get excess death
    url = "https://data.cdc.gov/api/views/xkkf-xrst/rows.csv?accessType=DOWNLOAD&bom=true&format=true%20target="
    response = requests.get(url).content
    if os.path.exists("Excess_Deaths_COVID-19.csv"):
        os.remove("Excess_Deaths_COVID-19.csv")
    excess_death_csv = open("Excess_Deaths_COVID-19.csv", 'wb')
    excess_death_csv.write(response)
    excess_death_csv.close()
    print("Excess Death data ... DONE")

    # get vaccine data
    url = "https://github.com/govex/COVID-19/raw/master/data_tables/vaccine_data/us_data/time_series/vaccine_data_us_timeline.csv"
    response = requests.get(url).content
    if os.path.exists("vaccine_data_us_state_timeline.csv"):
        os.remove("vaccine_data_us_state_timeline.csv")
    vaccine_csv = open("vaccine_data_us_state_timeline.csv", 'wb')
    vaccine_csv.write(response)
    vaccine_csv.close()
    print("Vaccine data ... DONE")

if __name__ == "__main__":
    get_datasets()


# In[ ]:




