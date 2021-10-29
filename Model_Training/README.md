# Running the CAMul code

## To set up the environment : 

./scripts/setup.sh



## To preprocess the mort and hosp computations :

./scripts/mort_preprocess.sh 202142
./scripts/hosp_preprocess.sh 202142



## For hosp predictions (train and test):

python train_hosp.py -e 202142 -m deploy_week_42_1 -c True -d 1  —start_model hosp_deploy_week_41_1

-e 202142  :  epiweek number

-d 1  :  Represents the day which is trained/tested for (We need to repeat this comment from -d 1 to -d 30)

-m deploy_week_42_1. :  name of the model that it is saved to (change from deploy_week_<current_week>_1  to deploy_week_<current_week>_30)

—start_model hosp_deploy_week_41_1  :  Represents the initialized weights folder name (Change from deploy_week_<current_week-1>_1  to deploy_week_<current_week-1>_30 )

-c True  :  Cuda enabled




## For mort predictions (train and test):

python train_covid2.py -e 202142 -c True -m mort_deploy_week_42_1 -w 1 —start_model mort_deploy_week_41_1

-e 202142  :  epiweek number

-w 1  :  Represents the day which is trained/tested for (We need to repeat this comment from -w 1 to -w 4)

-m mort_deploy_week_42_1  :  name of the model that it is saved to (change from mort_deploy_week_<current_week>_1  to mort_deploy_week_<current_week>_4)

—start_model hosp_deploy_week_41_1  :  Represents the initialized weights folder name (Change from mort_deploy_week_<current_week-1>_1  to mort_deploy_week_<current_week-1>_30 )

-c True  :  Cuda enabled
