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
parser.add_option("-m", "--model_type", dest="model_type", default="normal", type="string") # normal, cnn, slidingwindow, preprocess, slidingwindow_cnn, rag, slidingwindow_rag
parser.add_option("--size", dest="window_size", default=128, type="int")
parser.add_option("--stride", dest="window_stride", default=1000, type="int")
parser.add_option("--seed", dest="seed", default=0, type="int")

# parser.add_option("-f", "--files", dest="file_names", default="sliding_model_202252_True_0.001_500_4", type="string")
# parser.add_option("-f", "--files", dest="file_names", default="sliding_model_202252_True_0.001_500_4", type="string")
# parser.add_option("-s", "--state", dest="state", default="AR", type="string")
(options, args) = parser.parse_args()


disease_here = "traffic"
if "cnn" in options.model_type:
    disease_here = disease_here + "_cnn"
if "rag" in options.model_type:
    disease_here = disease_here + "_rag"
elif "nn" in options.model_type:
    if "nn-simple" in options.model_type:
        disease_here = disease_here + "_nn-simple"
    elif "nn-bn" in options.model_type:
        disease_here = disease_here + "_nn-bn"
    elif "nn-dot" in options.model_type:
        disease_here = disease_here + "_nn-dot"
    elif "nn-bert" in options.model_type:
        disease_here = disease_here + "_nn-bert"


if "slidingwindow" in options.model_type or "preprocess" in options.model_type:
    directory = "/localscratch/ssinha97/fnp_evaluations/"+disease_here+"_traffic_stable_predictions_slidingwindow"
else:
    directory = "/localscratch/ssinha97/fnp_evaluations/"+disease_here+"_traffic_stable_predictions"


# # if options.model_type == "normal":
# #     file_initials = "traffic_traffic_stable_predictions/normal_disease_traffic_epiweek_202252_weekahead_"
# #     writer = SummaryWriter("runs/traffic/traffic_normal_epiweek"+str(options.epiweek))
# if "slidingwindow" in options.model_type and "cnn" in options.model_type:
#         file_initials = "traffic_cnn_traffic_stable_predictions_slidingwindow/cnn_slidingwindow_disease_traffic_weekahead_"
#         # writer = SummaryWriter("runs/traffic/traffic_cnn_slidingwindowpreprocessed_epiweek"+str(options.epiweek)+"_windowsize_"+str(options.window_size)+"_stride_"+str(options.window_stride))
# elif "slidingwindow" in options.model_type and "rag" in options.model_type:
#         file_initials = "traffic_rag_traffic_stable_predictions_slidingwindow/rag_slidingwindow_disease_traffic_weekahead_"
#         # file_initials = "traffic_rag_hosp_stable_predictions_slidingwindow/rag_slidingwindow_disease_traffic_epiweek_202252_weekahead_"
#         # writer = SummaryWriter("runs/traffic/traffic_rag_slidingwindowpreprocessed_epiweek"+str(options.epiweek)+"_windowsize_"+str(options.window_size)+"_stride_"+str(options.window_stride))        
# elif "preprocess" in options.model_type:
#         file_initials = "traffic_hosp_stable_predictions_slidingwindow/slidingwindowpreprocessed_disease_traffic_epiweek_202252_weekahead_"
#         # writer = SummaryWriter("runs/traffic/traffic_slidingwindowpreprocessed_epiweek"+str(options.epiweek)+"_windowsize_"+str(options.window_size)+"_stride_"+str(options.window_stride))
# # elif "cnn" in options.model_type:
# #         file_initials = "traffic_cnn_hosp_stable_predictions/cnn_disease_traffic_epiweek_202252_weekahead_"
# #         writer = SummaryWriter("runs/traffic/traffic_cnn_disease_traffic_epiweek"+str(options.epiweek))
# # elif "rag" in options.model_type:
# #         file_initials = "traffic_rag_hosp_stable_predictions/rag_disease_traffic_epiweek_202252_weekahead_"
# #         writer = SummaryWriter("runs/traffic/traffic_rag_disease_traffic_epiweek"+str(options.epiweek))
# else:
#         file_initials = "traffic_traffic_stable_predictions_slidingwindow/slidingwindow_disease_traffic_weekahead_"
#         # writer = SummaryWriter("runs/traffic/traffic_slidingwindow_epiweek"+str(options.epiweek)+"_windowsize_"+str(options.window_size)+"_stride_"+str(options.window_stride))


# def get_metrics(file_name, epiweek_now, ahead):
#     if not os.path.exists(file_name):
#         print(f"File {file_name} does not exist")
#         return None

#     with open(file_name, "rb") as f:
#         fl_data = pickle.load(f)
#         predictions = fl_data[0][:, :, ahead-1]
#         ground_truth = fl_data[1][:]

#     mean_preds = predictions.mean(axis=0)
#     std_preds = predictions.std(axis=0)

#     metrics = {
#         "rmse": rmse(mean_preds, ground_truth),
#         "nrmse": nrmse(mean_preds, ground_truth),
#         "mape": mape(mean_preds, ground_truth),
#         "crps": crps_samples(predictions.T, ground_truth),
#         "cs": get_pr(mean_preds, std_preds, ground_truth)[1],
#     }
#     return metrics


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



# states = [
#     "AL",
#     "AK",
#     "AZ",
#     "AR",
#     "CA",
#     "CO",
#     "CT",
#     "DE",
#     "DC",
#     "FL",
#     "GA",
#     "ID",
#     "IL",
#     "IN",
#     "IA",
#     "KS",
#     "KY",
#     "LA",
#     "ME",
#     "MD",
#     "MA",
#     "MI",
#     "MN",
#     "MS",
#     "MO",
#     "MT",
#     "NE",
#     "NV",
#     "NH",
#     "NJ",
#     "NM",
#     "NY",
#     "NC",
#     "ND",
#     "OH",
#     "OK",
#     "OR",
#     "PA",
#     "RI",
#     "SC",
#     "SD",
#     "TN",
#     "TX",
#     "UT",
#     "VT",
#     "VA",
#     "WA",
#     "WV",
#     "WI",
#     "WY",
#     "X",
# ]

states = list(range(88))
week_ahead = [1,2,3]
plot_dict = {}
heat_map_means = {}
heat_map_stds = {}

for state in states:
    plot_dict[state] = [None, None, None]  # [pred_mean, pred_stddev, label]
all_weeks_results = []
# counter = -1
for ah in tqdm(week_ahead):
    
    if "slidingwindow" in options.model_type or "preprocess" in options.model_type:
        if "autosize" in options.model_type:
            save_model = f"slidingwindow_disease_traffic_weekahead_"+str(ah)+"_autosize_"+str(options.model_type[-1])
        else:
            try:
                save_model = f"slidingwindow_disease_traffic_weekahead_"+str(ah)+"_wsize_"+str(sizes[mt])+"_wstride_"+str(strides[mt])
            except: pu.db

        if "preprocess" in options.model_type:
            save_model = f"slidingwindowpreprocessed_disease_traffic_weekahead_"+str(ah)+"_wsize_"+str(sizes[mt])+"_wstride_"+str(strides[mt])
        
        if "cnn" in options.model_type:
            save_model = "cnn_" + save_model
        elif "rag" in options.model_type:
            save_model = "rag_" + save_model
        elif "nn" in options.model_type:
            if "nn-simple" in options.model_type:
                save_model = "nn-simple_"+save_model
            elif "nn-bn" in options.model_type:
                save_model = "nn-bn_"+save_model
            elif "nn-dot" in options.model_type:
                save_model = "nn-dot_"+save_model
            elif "nn-bert" in options.model_type:
                save_model = "nn-bert_"+save_model
    else:
        save_model = "disease_traffic_weekahead_"+str(ah)
        if "cnn" in options.model_type:
            save_model = "cnn_"+save_model
        elif "rag" in options.model_type:
            save_model = "rag_"+save_model
        elif "nn" in options.model_type:
            if "nn-simple" in options.model_type:
                save_model = "nn-simple_"+save_model
            elif "nn-bn" in options.model_type:
                save_model = "nn-bn_"+save_model
            elif "nn-dot" in options.model_type:
                save_model = "nn-dot_"+save_model
            elif "nn-bert" in options.model_type:
                save_model = "nn-bert_"+save_model
        else:
            save_model = "normal_"+save_model
    if options.seed != 0:
        save_model = save_model + "_seed_"+str(options.seed)
    
    file_to_load = save_model + "_predictions.pkl"
    # if "slidingwindow" in options.model_type or "preprocess" in options.model_type:
    #     directory = "/localscratch/ssinha97/fnp_evaluations/"+disease_here+"_predictions_slidingwindow"
    # else:
    #     directory = "/localscratch/ssinha97/fnp_evaluations/"+disease_here+"_predictions_normal"
    with open(directory+"/"+file_to_load, "rb") as f:
        data_pickle = pickle.load(f)
    # else:
    #     with open(file_initials+str(ah)+"_predictions.pkl", "rb") as f:
    #         data_pickle = pickle.load(f)

    y_preds     = np.mean(data_pickle[0], axis=0)
    std_devs    = np.std(data_pickle[0], axis=0)
    y_gts       = np.squeeze(data_pickle[1], axis=0)
    all_weeks_results.append([y_preds, std_devs, y_gts])

y_preds_all = np.concatenate([np.expand_dims(all_weeks_results[0][0], axis=-1), np.expand_dims(all_weeks_results[1][0], axis=-1), np.expand_dims(all_weeks_results[2][0], axis=-1)], axis=-1)
y_std_all =  np.concatenate([np.expand_dims(all_weeks_results[0][1], axis=-1), np.expand_dims(all_weeks_results[1][1], axis=-1), np.expand_dims(all_weeks_results[2][1], axis=-1)], axis=-1)
y_gts_all = np.concatenate([np.expand_dims(all_weeks_results[0][-1], axis=-1), np.expand_dims(all_weeks_results[1][-1], axis=-1), np.expand_dims(all_weeks_results[2][-1], axis=-1)], axis=-1)
all_rmse = []
all_crps = []
all_cs = []
for i in tqdm(range(len(y_preds_all))):
    all_rmse.append(rmse(y_preds_all[i], y_gts_all[i]))
    all_crps.append(crps_samples(y_preds_all[i], y_gts_all[i]))
    all_cs.append(get_pr(y_preds_all[i], y_std_all[i]**2, y_gts_all[i])[1])
print("\nRMSE average")
print(np.mean(all_rmse))
print("\nCRPS average")
print(np.mean(all_crps))
print("\nCS average")
print(np.mean(all_cs))
"""
for st, state in enumerate(states):
        if ah == 1:
            try:
                plot_dict[state][0] = data_pickle[2][st].tolist()
                plot_dict[state][1] = [0 for x in range(len(plot_dict[state][0]))]
                plot_dict[state][2] = data_pickle[2][st].tolist()
            except:
                pu.db
        # if ah == 3:
        #     pu.db
        
        plot_dict[state][0].append(np.mean(data_pickle[0][:, st]))
        plot_dict[state][1].append(np.std(data_pickle[0][:, st]))
        # pu.db
        try:
            plot_dict[state][2].append(data_pickle[1][st])
        except:
            pu.db
        # plot_dict[state][3].append(data_pickle[3][st])
    
heat_map_means[ah] = np.mean(data_pickle[3], axis=-1)
heat_map_stds[ah] = np.std(data_pickle[3], axis=-1)

# for ah in week_ahead:
#     for state in states:
#         heat_map_means[ah].append(plot_dict[state][3][ah - 1])
#     heat_map_stds[ah] = np.std(heat_map_means[ah], axis=0)
#     pu.db

#plots
all_yps = []
all_devs = []
all_yts = []
for st, state in enumerate(tqdm(states)):
    plt.figure(st)
    yp = np.array(plot_dict[state][0])
    dev = np.array(plot_dict[state][1]) * 1.95
    yt = np.array(plot_dict[state][2])
    plt.plot(yp, label="Predicted 95%", color="blue")
    plt.fill_between(np.arange(len(yp)), yp + dev, yp - dev, color="blue", alpha=0.2)
    plt.plot(yt, label="True Value", color="green")
    plt.legend()

    yp_here = yp[-len(week_ahead):]
    dev_here = dev[-len(week_ahead):]
    yt_here = yt[-len(week_ahead):]

    all_yps.extend(yp_here.tolist())
    all_devs.extend(dev_here.tolist())
    all_yts.extend(yt_here.tolist())

    rmse_here   = rmse(yp_here, yt_here)
    # nrmse_here  = nrmse(yp_here, yt_here)
    # mape_here   = mape(yp_here, yt_here)
    crps_here   = crps_samples(yp_here, yt_here)
    cs_here     = get_pr(yp_here, dev_here**2, yt_here)[1]

    txt = "RMSE: "+str(rmse_here)+"\nCRPS: "+str(crps_here)+"\nCS: "+str(cs_here)
    plt.figtext(0.5, 0.0, txt, wrap=True, va="top", ha="center", fontsize=12)



    plot_name = f"plots_traffic/"+str(state)
    if options.model_type is not "normal":
        if "cnn" in options.model_type and "slidingwindow" in options.model_type:
            plot_name = plot_name+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)+"_cnn"
        if "rag" in options.model_type and "slidingwindow" in options.model_type:
            plot_name = plot_name+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)+"_rag"
        elif "cnn" in options.model_type:
             plot_name = plot_name + "_cnn"
        elif "rag" in options.model_type:
             plot_name = plot_name + "_rag"
        elif "rag" in options.model_type:
             plot_name = plot_name + "_rag"
        elif "preprocess" in options.model_type:
            plot_name = plot_name+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)+"_preprocessed"
        else:
            plot_name = f"plots_traffic/"+str(state)+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)
       


    if "autosize" in options.model_type:
        plot_name = plot_name.split("_")[0]+"_autosize_"+str(options.model_type[-1])
        if "rag" in options.model_type:
            plot_name = plot_name+"_rag"
        if "cnn" in options.model_type:
            plot_name = plot_name+"_cnn"
            
    plot_name = plot_name + ".png"
    # pu.db
    plt.savefig(plot_name, bbox_inches = "tight")
        
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
    elif "rag" in options.model_type and "slidingwindow" in options.model_type:
        plt.savefig(f"plots_"+file_initials.split("_")[0]+"/heatmap_"+str(ah)+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)+"_rag.png")
    elif "slidingwindow" in options.model_type:
        plt.savefig(f"plots_"+file_initials.split("_")[0]+"/heatmap_"+str(ah)+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)+".png")
    elif "cnn" in options.model_type:
        plt.savefig(f"plots_"+file_initials.split("_")[0]+"/heatmap_"+str(ah)+"_cnn.png")
    elif "rag" in options.model_type:
        plt.savefig(f"plots_"+file_initials.split("_")[0]+"/heatmap_"+str(ah)+"_rag.png")
    else:
        plt.savefig(f"plots_"+file_initials.split("_")[0]+"/heatmap_"+str(ah)+".png")

all_rmse   = rmse(np.array(all_yps), np.array(all_yts))
# all_nrmse  = nrmse(np.array(all_yps), np.array(all_yts))
# all_mape   = mape(np.array(all_yps), np.array(all_yts))
all_crps   = crps_samples(np.array(all_yps), np.array(all_yts))
all_cs     = get_pr(np.array(all_yps), np.array(all_devs)**2, np.array(all_yts))[1]

if "preprocess" in options.model_type or "slidingwindow" in options.model_type:
    if "autosize" in options.model_type:
        txt = file_initials+"_autosize_"+str(options.model_type[-1])
    else:
        txt = file_initials+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)
else:
    txt = file_initials
if options.seed != 0:
    txt = txt + "_seed_"+str(options.seed)
txt = "\n\n"+txt+"\nRMSE: "+str(all_rmse)+"\nCRPS: "+str(all_crps)+"\nCS: "+str(all_cs)
print(txt)

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

"""