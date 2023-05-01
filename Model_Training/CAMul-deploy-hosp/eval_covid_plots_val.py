import pickle
import os
import pandas as pd
import numpy as np
from covid_extract.hosp_consts import states
from eval_metrics import rmse, nrmse, mape, crps_samples, get_pr
import pudb
from torch.utils.tensorboard import SummaryWriter
from optparse import OptionParser
from tqdm import tqdm
import pudb
import matplotlib.pyplot as plt
import sys

parser = OptionParser()
parser.add_option("-s", "--epiweek-start", dest="epiweek_start", default="202232", type="string")
parser.add_option("-e", "--epiweek-end", dest="epiweek_end", default="202306", type="string")
parser.add_option("-m", "--model_type", dest="model_type", default="normal", type="string") # normal, cnn, slidingwindow, preprocess, slidingwindow_cnn, rag, slidingwindow_rag
parser.add_option("--size", dest="window_size", default=10, type="int")
parser.add_option("--stride", dest="window_stride", default=10, type="int")
parser.add_option("--seed", dest="seed", default=0, type="int")
# parser.add_option("-f", "--files", dest="file_names", default="sliding_model_"+str(options.epiweek)+"_True_0.001_500_4", type="string")
# parser.add_option("-f", "--files", dest="file_names", default="sliding_model_"+str(options.epiweek)+"_True_0.001_500_4", type="string")
# parser.add_option("-s", "--state", dest="state", default="AR", type="string")
(options, args) = parser.parse_args()
plot_count = 0
states = [
    "DC", "MA", "FL", "GA", "IL", "NY", "NJ", "PA", "TX", "WA", "CA", "X"
]
all_yps = []
all_devs = []
all_yts = []

statewise_yps = [[] for state in states]
statewise_devs = [[] for state in states]
statewise_yts = [[] for state in states]

weekwise_data = [] # list of lists of lists with each element of weekwise_data being [ahead_wise_week_yps, aheadwise_week_devs, aheadwise_week_yts]

if "2023" in options.epiweek_end:
    epiweeks = list(range(int(options.epiweek_start), 202252)) + list(range(202301, int(options.epiweek_end)))
else:
    epiweeks = list(range(int(options.epiweek_start), int(options.epiweek_end)))

rmse_all, crps_all = [], []
for epiweek in tqdm(epiweeks):
    week_ahead = [1,2,3]
    plot_dict = {}
    heat_map_means = {}
    heat_map_stds = {}

    for state in states:
        plot_dict[state] = [None, None, None]  # [pred_mean, pred_stddev, label]

    # yp_this_week, y_this_week = [], []
    # counter = -1
    for ah in week_ahead:
        if "slidingwindow" in options.model_type or "preprocess" in options.model_type:
            if "autosize" in options.model_type:
                save_model = f"slidingwindow_disease_covid_epiweek_"+str(epiweek)+"_weekahead_"+str(ah)+"_autosize_"+str(options.model_type[-1])
            else:
                save_model = f"slidingwindow_disease_covid_epiweek_"+str(epiweek)+"_weekahead_"+str(ah)+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)

            if "preprocess" in options.model_type:
                save_model = f"slidingwindowpreprocessed_disease_covid_epiweek_"+str(epiweek)+"_weekahead_"+str(ah)+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)
            
            if "cnn" in options.model_type:
                save_model = "cnn_" + save_model
            elif "rag" in options.model_type:
                save_model = "rag_" + save_model
        else:
            save_model = "disease_covid_epiweek_"+str(epiweek)+"_weekahead_"+str(ah)
            if "cnn" in options.model_type:
                save_model = "cnn_"+save_model
            elif "rag" in options.model_type:
                save_model = "rag_"+save_model
            else:
                save_model = "normal_"+save_model
        
        if options.seed != 0:
            save_model = save_model + "_seed_"+str(options.seed)

        disease_here = "covid"
        if "cnn" in options.model_type:
            disease_here = disease_here + "_cnn"
        if "rag" in options.model_type:
            disease_here = disease_here + "_rag"
        
        file_to_load = save_model + "_predictions.pkl"
        if "slidingwindow" in options.model_type or "preprocess" in options.model_type:
            directory = "./"+disease_here+"_val_predictions_slidingwindow"
        else:
            directory = "./"+disease_here+"_val_predictions_normal"
        with open(directory+"/"+file_to_load, "rb") as f:
            data_pickle = pickle.load(f)

        yp = data_pickle[list(data_pickle.keys())[-1]]["pred"]
        y  = data_pickle[list(data_pickle.keys())[-1]]["gt"]

        # yp_this_week


        rmse_here = rmse(yp,y)
        crps_here = crps_samples(yp, y)

        rmse_all.append(rmse_here)
        crps_all.append(crps_here)

rmse_avg = np.mean(rmse_all)
crps_avg = np.mean(crps_all)

print("RMSE avg: "+str(rmse_avg))
print("CRPS avg: "+str(crps_avg))

