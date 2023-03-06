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
parser.add_option("-e", "--epiweek", dest="epiweek", default="202252", type="string")
parser.add_option("-m", "--model_type", dest="model_type", default="normal", type="string") # normal, cnn, slidingwindow, preprocess, slidingwindow_cnn
parser.add_option("--size", dest="window_size", default=10, type="int")
parser.add_option("--stride", dest="window_stride", default=10, type="int")
# parser.add_option("-f", "--files", dest="file_names", default="sliding_model_202252_True_0.001_500_4", type="string")
# parser.add_option("-f", "--files", dest="file_names", default="sliding_model_202252_True_0.001_500_4", type="string")
# parser.add_option("-s", "--state", dest="state", default="AR", type="string")
(options, args) = parser.parse_args()

if options.model_type == "normal":
    file_initials = "flu_hosp_stable_predictions/normal_disease_flu_epiweek_202252_weekahead_"
    writer = SummaryWriter("runs/flu/flu_normal_epiweek"+str(options.epiweek))
elif "slidingwindow" in options.model_type and "cnn" in options.model_type:
        file_initials = "flu_cnn_hosp_stable_predictions_slidingwindow/cnn_slidingwindow_disease_flu_epiweek_202252_weekahead_"
        writer = SummaryWriter("runs/flu/flu_cnn_slidingwindowpreprocessed_epiweek"+str(options.epiweek)+"_windowsize_"+str(options.window_size)+"_stride_"+str(options.window_stride))        
elif "preprocess" in options.model_type:
        file_initials = "flu_hosp_stable_predictions_slidingwindow/slidingwindowpreprocessed_disease_flu_epiweek_202252_weekahead_"
        writer = SummaryWriter("runs/flu/flu_slidingwindowpreprocessed_epiweek"+str(options.epiweek)+"_windowsize_"+str(options.window_size)+"_stride_"+str(options.window_stride))
elif "cnn" in options.model_type:
        file_initials = "flu_cnn_hosp_stable_predictions/cnn_disease_flu_epiweek_202252_weekahead_"
        writer = SummaryWriter("runs/flu/flu_cnn_disease_flu_epiweek"+str(options.epiweek))
else:
        file_initials = "flu_hosp_stable_predictions_slidingwindow/slidingwindow_disease_flu_epiweek_202252_weekahead_"
        writer = SummaryWriter("runs/flu/flu_slidingwindow_epiweek"+str(options.epiweek)+"_windowsize_"+str(options.window_size)+"_stride_"+str(options.window_stride))


def get_metrics(file_name, epiweek_now, ahead):
    if not os.path.exists(file_name):
        print(f"File {file_name} does not exist")
        return None

    with open(file_name, "rb") as f:
        fl_data = pickle.load(f)
        predictions = fl_data[0][:, :, ahead-1]
        ground_truth = fl_data[1][:]

    mean_preds = predictions.mean(axis=0)
    std_preds = predictions.std(axis=0)

    metrics = {
        "rmse": rmse(mean_preds, ground_truth),
        "nrmse": nrmse(mean_preds, ground_truth),
        "mape": mape(mean_preds, ground_truth),
        "crps": crps_samples(predictions.T, ground_truth),
        "cs": get_pr(mean_preds, std_preds, ground_truth)[1],
    }
    return metrics


# sample_out = [True, False]
# lr = [0.001, 0.0001]
# epiweeks = list(range(202101, 202153))
# ahead = [1]

# metrics_df = pd.DataFrame(
#     columns=[
#         "sample_out",
#         "lr",
#         "ahead",
#         "epiweek",
#         "rmse",
#         "nrmse",
#         "mape",
#         "crps",
#         "cs",
#     ]
# )

# for sample in sample_out:
#     for lr_ in lr:
#         for week in epiweeks:
#             for ah in ahead:
#                 save_model = f"ar_model_{week}_{sample}_{lr_}"
#                 file_name = os.path.join(
#                     "hosp_stable_predictions", f"{save_model}_predictions.pkl"
#                 )
#                 print(f"Getting metrics for {file_name}")
#                 metrics = get_metrics(file_name, week, ah)

#                 if metrics is None:
#                     continue
#                 print(metrics)
#                 metrics_df = metrics_df.append(
#                     {
#                         "sample_out": sample,
#                         "lr": lr_,
#                         "ahead": ah,
#                         "epiweek": week,
#                         "rmse": metrics["rmse"],
#                         "nrmse": metrics["nrmse"],
#                         "mape": metrics["mape"],
#                         "crps": metrics["crps"],
#                         "cs": metrics["cs"],
#                     },
#                     ignore_index=True,
#                 )



states = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "DC",
    "FL",
    "GA",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "X",
]

week_ahead = [1,2,3]
plot_dict = {}
heat_map_means = {}
heat_map_stds = {}

for state in states:
    plot_dict[state] = [None, None, None]  # [pred_mean, pred_stddev, label]

# counter = -1
for ah in week_ahead:
    if "preprocess" in options.model_type or "slidingwindow" in options.model_type:
        with open(file_initials+str(ah)+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)+"_predictions.pkl", "rb") as f:
            data_pickle = pickle.load(f)
    else:
        with open(file_initials+str(ah)+"_predictions.pkl", "rb") as f:
            data_pickle = pickle.load(f)

    for st, state in enumerate(states):
        if ah == 1:
            plot_dict[state][0] = data_pickle[2][st].tolist()
            plot_dict[state][1] = [0 for x in range(len(plot_dict[state][0]))]
            plot_dict[state][2] = data_pickle[2][st].tolist()
        # if ah == 3:
        #     pu.db
        
        plot_dict[state][0].append(np.mean(data_pickle[0][:, st]))
        plot_dict[state][1].append(np.std(data_pickle[0][:, st]))
        # pu.db
        plot_dict[state][2].append(data_pickle[1][st])
        # plot_dict[state][3].append(data_pickle[3][st])
    
    heat_map_means[ah] = np.mean(data_pickle[3], axis=0)
    heat_map_stds[ah] = np.std(data_pickle[3], axis=0)

# for ah in week_ahead:
#     for state in states:
#         heat_map_means[ah].append(plot_dict[state][3][ah - 1])
#     heat_map_stds[ah] = np.std(heat_map_means[ah], axis=0)
#     pu.db

#plots
for st, state in enumerate(states):
    plt.figure(st)
    yp = np.array(plot_dict[state][0][-10:])
    dev = np.array(plot_dict[state][1][-10:]) * 1.95
    yt = np.array(plot_dict[state][2][-10:])
    plt.plot(yp, label="Predicted 95%", color="blue")
    plt.fill_between(np.arange(len(yp)), yp + dev, yp - dev, color="blue", alpha=0.2)
    plt.plot(yt, label="True Value", color="green")
    plt.legend()


    plot_name = f"plots_flu/"+state
    if options.model_type is not "normal":
        if "cnn" in options.model_type and "slidingwindow" in options.model_type:
            plot_name = plot_name+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)+"_cnn"
        elif "cnn" in options.model_type:
             plot_name = plot_name + "_cnn"
        elif "preprocess" in options.model_type:
            plot_name = plot_name+"_preprocessed"
        else:
            plot_name = f"plots_flu/"+state+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)
       
    plot_name = plot_name + ".png"
    # pu.db
    plt.savefig(plot_name)
        
    # if state == "FL":
    #     pu.db

    # pu.db
for ah in week_ahead:
    plt.figure(len(states) + ah)
    plt.imshow(heat_map_means[ah], cmap='viridis')
    plt.colorbar()
    if "preprocess" in options.model_type:
        plt.savefig(f"plots_"+file_initials.split("_")[0]+"/heatmap_"+str(ah)+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)+"_preprocessed.png")
    elif "cnn" in options.model_type and "slidingwindow" in options.model_type:
        plt.savefig(f"plots_"+file_initials.split("_")[0]+"/heatmap_"+str(ah)+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)+"_cnn.png")
    elif "slidingwindow" in options.model_type:
        plt.savefig(f"plots_"+file_initials.split("_")[0]+"/heatmap_"+str(ah)+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)+".png")
    elif "cnn" in options.model_type:
        plt.savefig(f"plots_"+file_initials.split("_")[0]+"/heatmap_"+str(ah)+"_cnn.png")
    else:
        plt.savefig(f"plots_"+file_initials.split("_")[0]+"/heatmap_"+str(ah)+".png")


sys.exit(0)




# pu.db

for ep in tqdm(data_dict.keys()):
    pred_here = []
    gt_here = []
    vars_here = []
    pred    = data_dict[ep]['pred']
    gt      = data_dict[ep]['gt']
    states  = data_dict[ep]['states']
    vars    = data_dict[ep]['vars']
    for i in range(len(states)):
        if states[i] == options.state:
            pred_here.append(pred[i])
            gt_here.append(gt[i])
            vars_here.append(vars[i])
    pred_here = np.array(pred_here)
    gt_here = np.array(gt_here)
    vars_here = np.array(vars_here)


    rmse_here   = rmse(pred_here, gt_here)
    nrmse_here  = nrmse(pred_here, gt_here)
    mape_here   = mape(pred_here, gt_here)
    crps_here   = crps_samples(pred_here, gt_here)
    cs_here     = get_pr(pred_here, vars_here, gt_here)[1]

    writer.add_scalar('Val/RMSE', rmse_here, ep)
    writer.add_scalar('Val/NRMSE', nrmse_here, ep)
    writer.add_scalar('Val/MAPE', mape_here, ep)
    writer.add_scalar('Val/CRPS', crps_here, ep)
    writer.add_scalar('Val/CS_HERE', cs_here, ep)
    
# Save as csv
# metrics_df.to_csv("ar_metrics_stable.csv", index=False)

# # Avergae over all epiweeks
# metrics_all_weeks = metrics_df.groupby(["sample_out", "lr", "ahead"]).mean().reset_index()
# metrics_all_weeks.to_csv("ar_metrics_stable_all_weeks.csv", index=False)

