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
parser.add_option("--optionals", dest="optionals", type="string", default=" ")

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

if options.model_type == "all":
    model_types = ["slidingwindow", "slidingwindow", "slidingwindow_autosize_0", "slidingwindow_rag_autosize_0"]
    sizes = [1000, 128]
    strides = [1000, 1000]
else:
    model_types = [options.model_type]
    sizes = [options.window_size]
    strides = [options.window_stride]

statewise_yps = [[] for state in states]
statewise_devs = [[] for state in states]
statewise_yts = [[] for state in states]

weekwise_data = [] # list of lists of lists with each element of weekwise_data being [ahead_wise_week_yps, aheadwise_week_devs, aheadwise_week_yts]

if "2023" in options.epiweek_end:
    epiweeks = list(range(int(options.epiweek_start), 202252)) + list(range(202301, int(options.epiweek_end)))
else:
    epiweeks = list(range(int(options.epiweek_start), int(options.epiweek_end)))

rmse_all, crps_all, conf_all = [], [], []
for mt, model_type in enumerate(tqdm(model_types)):
    week_ahead = [1,2,3]
    plot_dict = {}
    heat_map_means = {}
    heat_map_stds = {}

    for state in states:
        plot_dict[state] = [None, None, None]  # [pred_mean, pred_stddev, label]

    yp_this_week, y_this_week, v_this_week = [], [], []
    # counter = -1
    # rmse_this_week, crps_this_week = [], []
    for ah in week_ahead:
        if "slidingwindow" in model_type or "preprocess" in model_type:
            if "autosize" in model_type:
                save_model = f"slidingwindow_disease_power_weekahead_"+str(ah)+"_autosize_"+str(model_type[-1])
            elif "smart-mode" in model_type:
                save_model = f"slidingwindow_disease_power_weekahead_"+str(ah)+"_smart-mode-"+str(model_type[-1])
            else:
                try:
                    save_model = f"slidingwindow_disease_power_weekahead_"+str(ah)+"_wsize_"+str(sizes[mt])+"_wstride_"+str(strides[mt])
                except: pu.db

            if "preprocess" in model_type:
                save_model = f"slidingwindowpreprocessed_disease_power_weekahead_"+str(ah)+"_wsize_"+str(sizes[mt])+"_wstride_"+str(strides[mt])
            
            if "cnn" in model_type:
                save_model = "cnn_" + save_model
            elif "rag" in model_type:
                save_model = "rag_" + save_model
            elif "nn" in model_type:
                if "nn-simple" in model_type:
                    save_model = "nn-simple_"+save_model
                elif "nn-bn" in model_type:
                    save_model = "nn-bn_"+save_model
                elif "nn-dot" in model_type:
                    save_model = "nn-dot_"+save_model
                elif "nn-bert" in model_type:
                    save_model = "nn-bert_"+save_model
        else:
            save_model = "disease_power_weekahead_"+str(ah)
            if "smart-mode" in model_type:
                save_model += "_smart-mode-"+str(model_type[-1])
            if "cnn" in model_type:
                save_model = "cnn_"+save_model
            elif "rag" in model_type:
                save_model = "rag_"+save_model
            elif "nn" in model_type:
                if "nn-simple" in model_type:
                    save_model = "nn-simple_"+save_model
                elif "nn-bn" in model_type:
                    save_model = "nn-bn_"+save_model
                elif "nn-dot" in model_type:
                    save_model = "nn-dot_"+save_model
                elif "nn-bert" in model_type:
                    save_model = "nn-bert_"+save_model
            else:
                save_model = "normal_"+save_model
        if options.seed != 0:
            save_model = save_model + "_seed_"+str(options.seed)

        disease_here = "power"
        if "cnn" in model_type:
            disease_here = disease_here + "_cnn"
        if "rag" in model_type:
            disease_here = disease_here + "_rag"
        elif "nn" in model_type:
            if "nn-simple" in model_type:
                disease_here = disease_here + "_nn-simple"
            elif "nn-bn" in model_type:
                disease_here = disease_here + "_nn-bn"
            elif "nn-dot" in model_type:
                disease_here = disease_here + "_nn-dot"
            elif "nn-bert" in model_type:
                disease_here = disease_here + "_nn-bert"
        
        if options.optionals != " ":
            file_to_load = save_model+"_optionals_"+options.optionals+"_predictions.pkl"
        else:
            file_to_load = save_model + "_predictions.pkl"
        if "slidingwindow" in model_type or "preprocess" in model_type:
            directory = "/localscratch/ssinha97/fnp_evaluations/"+disease_here+"_val_predictions_slidingwindow"
        else:
            directory = "/localscratch/ssinha97/fnp_evaluations/"+disease_here+"_val_predictions_normal"
        with open(directory+"/"+file_to_load, "rb") as f:
            data_pickle = pickle.load(f)

        rmse_min = np.inf
        # pu.db
        for key in data_pickle.keys():
            yp = data_pickle[list(data_pickle.keys())[key]]["pred"].tolist()
            y  = data_pickle[list(data_pickle.keys())[key]]["gt"].tolist()
            v  = data_pickle[list(data_pickle.keys())[key]]["vars"]
            rmse_here_inloop = rmse(np.array(yp)[:int(0.9*len(yp))], np.array(y)[:int(0.9*len(yp))])
            if rmse_here_inloop < rmse_min:
                yp_to_consider = yp[int(0.9*len(yp)):]
                y_to_consider = y[int(0.9*len(yp)):]
                v_to_consider = v[int(0.9*len(yp)):]
                rmse_min = rmse(np.array(yp_to_consider), np.array(y_to_consider))
        
        # pu.db
        # yp_this_week




        yp_this_week.extend(yp_to_consider)
        y_this_week.extend(y_to_consider)
        v_this_week.extend(v_to_consider)


    rmse_here = rmse(np.array(yp_this_week), np.array(y_this_week))
    crps_here = crps_samples(np.array(yp_this_week),  np.array(y_this_week))
    conf_score = get_pr(np.array(yp_this_week), np.array(v_this_week), np.array(y_this_week))[1]
    rmse_all.append(np.mean(rmse_here))
    crps_all.append(np.mean(crps_here))
    conf_all.append(np.mean(conf_score))

# pu.db
rmse_min = np.min(rmse_all, where=True)
poses_1 = np.argmin(rmse_all)
crps_min = np.min(crps_all, where=True)
poses_2 = np.argmin(crps_all)
conf_max = np.max(crps_all, where=True)
poses_3 = np.argmax(conf_all)

print("RMSE min: "+str(rmse_min))
print(model_types[poses_1])
if len(sizes) > poses_1:
    print("size: "+str(sizes[poses_1]))
if len(strides) > poses_1:
    print("stride: "+str(strides[poses_1]))
print("CRPS min: "+str(crps_min))
print(model_types[poses_2])
if len(sizes) > poses_2: 
    print("size: "+str(sizes[poses_2]))
if len(strides) > poses_2:
    print("stride: "+str(strides[poses_2]))

print("\n")
print("all_rmse")
print(rmse_all)

print("\nall_crps")
print(crps_all)

print("\nall_conf")
print(conf_all)

print("best model: ")
print(model_types[poses_1])
