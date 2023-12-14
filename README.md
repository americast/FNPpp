# FNP++
Repo for our forecasting efforts towards FNP++

## Data generation

- manually download Google symptom search data (daily) for the US -- make sure to have all years starting from 2020: https://pair-code.github.io/covid19_symptom_dataset/?country=US
- manually download covidnet: https://gis.cdc.gov/grasp/COVIDNet/COVID19_3.html
- change name of covidnet file to `COVID-NET_vYYYYWW` where YYYYWW is the epiweek 
- remove text in header and tail of covidnet file
- run script https://github.com/AdityaLab/CDC-Forecasting/blob/main/generate_data.sh

## Model training

See https://github.com/AdityaLab/CDC-Forecasting/blob/main/Model_Training/README.md
