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

for epiweek in tqdm(epiweeks):

    if options.model_type == "normal":
        file_initials = "flu_hosp_stable_predictions/normal_disease_flu_epiweek_"+str(epiweek)+"_weekahead_"
        writer = SummaryWriter("runs/flu/flu_normal_epiweek"+str(epiweek))
    elif "slidingwindow" in options.model_type and "cnn" in options.model_type:
            file_initials = "flu_cnn_hosp_stable_predictions_slidingwindow/cnn_slidingwindow_disease_flu_epiweek_"+str(epiweek)+"_weekahead_"
            writer = SummaryWriter("runs/flu/flu_cnn_slidingwindowpreprocessed_epiweek"+str(epiweek)+"_windowsize_"+str(options.window_size)+"_stride_"+str(options.window_stride))
    elif "slidingwindow" in options.model_type and "rag" in options.model_type:
            file_initials = "flu_rag_hosp_stable_predictions_slidingwindow/rag_slidingwindow_disease_flu_epiweek_"+str(epiweek)+"_weekahead_"
            writer = SummaryWriter("runs/flu/flu_rag_slidingwindowpreprocessed_epiweek"+str(epiweek)+"_windowsize_"+str(options.window_size)+"_stride_"+str(options.window_stride))        
    elif "preprocess" in options.model_type:
            file_initials = "flu_hosp_stable_predictions_slidingwindow/slidingwindowpreprocessed_disease_flu_epiweek_"+str(epiweek)+"_weekahead_"
            writer = SummaryWriter("runs/flu/flu_slidingwindowpreprocessed_epiweek"+str(epiweek)+"_windowsize_"+str(options.window_size)+"_stride_"+str(options.window_stride))
    elif "cnn" in options.model_type:
            file_initials = "flu_cnn_hosp_stable_predictions/cnn_disease_flu_epiweek_"+str(epiweek)+"_weekahead_"
            writer = SummaryWriter("runs/flu/flu_cnn_disease_flu_epiweek"+str(epiweek))
    elif "rag" in options.model_type:
            file_initials = "flu_rag_hosp_stable_predictions/rag_disease_flu_epiweek_"+str(epiweek)+"_weekahead_"
            writer = SummaryWriter("runs/flu/flu_rag_disease_flu_epiweek"+str(epiweek))
    else:
            file_initials = "flu_hosp_stable_predictions_slidingwindow/slidingwindow_disease_flu_epiweek_"+str(epiweek)+"_weekahead_"
            writer = SummaryWriter("runs/flu/flu_slidingwindow_epiweek"+str(epiweek)+"_windowsize_"+str(options.window_size)+"_stride_"+str(options.window_stride))


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

    week_ahead = [1,2,3]
    plot_dict = {}
    heat_map_means = {}
    heat_map_stds = {}

    for state in states:
        plot_dict[state] = [None, None, None]  # [pred_mean, pred_stddev, label]

    # counter = -1
    for ah in week_ahead:
        if "preprocess" in options.model_type or "slidingwindow" in options.model_type:
            if "autosize" in options.model_type:
                file_to_load = file_initials+str(ah)+"_autosize_"+str(options.model_type[-1])
            else:
                file_to_load = file_initials+str(ah)+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)
        else:
            file_to_load = file_initials+str(ah)

        if options.seed != 0:
            file_to_load = file_to_load+"_seed_"+str(options.seed)

        file_to_load = file_to_load + "_predictions.pkl"
        try:
            with open(file_to_load, "rb") as f:
                data_pickle = pickle.load(f)
        except:
            try:
                with open("/localscratch/ssinha97/fnp_evaluations/"+file_to_load, "rb") as f:
                    data_pickle = pickle.load(f)
            except:
                with open("/nvmescratch/ssinha97/fnp_evaluations/"+file_to_load, "rb") as f:
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
        # this_ah_yp = []
        # this_ah_dev = []
        # this_ah_yt = []

    # for ah in week_ahead:
    #     for state in states:
    #         heat_map_means[ah].append(plot_dict[state][3][ah - 1])
    #     heat_map_stds[ah] = np.std(heat_map_means[ah], axis=0)
    #     pu.db
    ahead_wise_yps = [[] for ah in week_ahead]
    ahead_wise_devs = [[] for ah in week_ahead]
    ahead_wise_yts = [[] for ah in week_ahead]
    #plots

    for st, state in enumerate(states):
        plt.figure(plot_count)
        plot_count += 1
        yp = np.array(plot_dict[state][0][-10:])
        dev = np.array(plot_dict[state][1][-10:]) * 1.95
        yt = np.array(plot_dict[state][2][-10:])
        plt.plot(yp, label="Predicted 95%", color="blue")
        plt.fill_between(np.arange(len(yp)), yp + dev, yp - dev, color="blue", alpha=0.2)
        plt.plot(yt, label="True Value", color="green")
        plt.legend()

        yp_here = yp[-len(week_ahead):]
        dev_here = dev[-len(week_ahead):]
        yt_here = yt[-len(week_ahead):]

        statewise_yps[st].extend(yp[-len(week_ahead):])
        statewise_devs[st].extend(dev[-len(week_ahead):])
        statewise_yts[st].extend(yt[-len(week_ahead):])
        for ah in week_ahead:
            ahead_wise_yps[ah - 1].append(yp_here[ah - 1])
            ahead_wise_devs[ah - 1].append(dev_here[ah - 1])
            ahead_wise_yts[ah - 1].append(yt_here[ah - 1])
        all_yps.extend(yp_here.tolist())
        all_devs.extend(dev_here.tolist())
        all_yts.extend(yt_here.tolist())

        rmse_here   = rmse(yp_here, yt_here)
        # nrmse_here  = nrmse(yp_here, yt_here)
        # mape_here   = mape(yp_here, yt_here)
        crps_here   = crps_samples(yp_here, yt_here)
        cs_here     = get_pr(yp_here, dev_here**2, yt_here)[1]

        txt = str(state)+" "+str(epiweek)+"\nRMSE: "+str(rmse_here)+"\nCRPS: "+str(crps_here)+"\nCS: "+str(cs_here)
        plt.figtext(0.5, 0.0, txt, wrap=True, va="top", ha="center", fontsize=12)




        plot_name = "plots_flu/"+str(state)+"_"+str(epiweek)
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
                plot_name = f"plots_flu/"+str(state)+"_"+str(epiweek)+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)

        if "autosize" in options.model_type:
            plot_name = plot_name.split("_")[0]+str(epiweek)+"_autosize_"+str(options.model_type[-1])
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
        plt.figure(plot_count)
        plot_count += 1
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
    weekwise_data.append([ahead_wise_yps, ahead_wise_devs, ahead_wise_yts])

all_rmse   = rmse(np.array(all_yps), np.array(all_yts))
all_nrmse  = nrmse(np.array(all_yps), np.array(all_yts))
# all_mape   = mape(np.array(all_yps), np.array(all_yts))
all_crps   = crps_samples(np.array(all_yps), np.array(all_yts))
all_cs     = get_pr(np.array(all_yps), np.array(all_devs)**2, np.array(all_yts))[1]
txt_all = ""
if "preprocess" in options.model_type or "slidingwindow" in options.model_type:
    txt_here = file_initials+"_wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)
else:
    txt_here = file_initials

txt_here = "\n\nTotal\n"+txt_here+"\nRMSE: "+str(all_rmse)+"\nNRMSE: "+str(all_nrmse)+"\nCRPS: "+str(all_crps)+"\nCS: "+str(all_cs)+"\n"
print(txt_here)
txt_all = txt_all + txt_here
for st, state in enumerate(states):
    statewise_yp  = np.array(statewise_yps[st])
    statewise_dev = np.array(statewise_devs[st])
    statewise_yt  = np.array(statewise_yts[st])


    statewise_rmse   = rmse(statewise_yp, statewise_yt)
    statewise_nrmse  = nrmse(statewise_yp, statewise_yt)
    # statewise_mape   = mape(statewise_yp, statewise_yt)
    statewise_crps   = crps_samples(statewise_yp, statewise_yt)
    statewise_cs     = get_pr(statewise_yp, statewise_dev**2, statewise_yt)[1]

    txt_here = "\nState: "+str(state)+"\nRMSE: "+str(statewise_rmse)+"\nNRMSE: "+str(statewise_nrmse)+"\nCRPS: "+str(statewise_crps)+"\nCS: "+str(statewise_cs)+"\n\n"
    print(txt_here)
    txt_all = txt_all + txt_here

if "preprocess" in options.model_type or "slidingwindow" in options.model_type:
    if "autosize" in options.model_type:
        file_initials = file_initials+"_autosize_"+str(options.model_type[-1])
    else:
        file_initials = file_initials+"wsize_"+str(options.window_size)+"_wstride_"+str(options.window_stride)+"_"

if options.seed != 0:
    file_initials = file_initials + "_seed_"+str(options.seed)

epiweek_rmses = {}
epiweek_nrmses = {}
epiweek_crpss = {}
epiweek_css = {}

rmse_all_states = [[] for x in states]
nrmse_all_states = [[] for x in states]
crps_all_states = [[] for x in states]
cs_all_states = [[] for x in states]

for e, epiweek in enumerate(epiweeks):
    # print(e)
    this_week_data = weekwise_data[e]
    this_week_rmses = []
    this_week_nrmses = []
    this_week_crpss = []
    this_week_css = []

    
    # all_states_this_week_rmse = []
    # all_states_this_week_nrmse = []
    # all_states_this_week_crps = []
    # all_states_this_week_cs = []
    for st, state in enumerate(states):
        yp_this_state = [this_week_data[0][x - 1][st] for x in week_ahead]
        dev_this_state = [this_week_data[1][x - 1][st] for x in week_ahead]
        yt_this_state = [this_week_data[2][x - 1][st] for x in week_ahead]

        rmse_this_state   = rmse(np.array(yp_this_state), np.array(yt_this_state))
        nrmse_this_state  = nrmse(np.array(yp_this_state), np.array(yt_this_state))
        # mape_this_state   = mape(np.array(yp_this_state), np.array(yt_this_state))
        crps_this_state   = crps_samples(np.array(yp_this_state), np.array(yt_this_state))
        cs_this_state     = get_pr(np.array(yp_this_state), np.array(dev_this_state)**2, np.array(yt_this_state))[1]

        rmse_all_states[st].append(rmse_this_state)
        nrmse_all_states[st].append(nrmse_this_state)
        crps_all_states[st].append(crps_this_state)
        cs_all_states[st].append(cs_this_state)






    for ah in week_ahead:
        yp_here = this_week_data[0][ah-1]
        dev_here = this_week_data[1][ah-1]
        yt_here = this_week_data[2][ah-1]

        rmse_here   = rmse(np.array(yp_here), np.array(yt_here))
        # if rmse_here > 80:
        #     pu.db
        this_week_rmses.append(rmse_here)
        nrmse_here  = nrmse(np.array(yp_here), np.array(yt_here))
        this_week_nrmses.append(nrmse_here)
        # mape_here   = mape(np.array(yp_here), np.array(yt_here))
        crps_here   = crps_samples(np.array(yp_here), np.array(yt_here))
        this_week_crpss.append(crps_here)
        cs_here     = get_pr(np.array(yp_here), np.array(dev_here)**2, yt_here)[1]
        this_week_css.append(cs_here)
    epiweek_rmses[epiweek] = np.mean(this_week_rmses)
    epiweek_nrmses[epiweek] = np.mean(this_week_nrmses)
    epiweek_crpss[epiweek] = np.mean(this_week_crpss)
    epiweek_css[epiweek] = np.mean(this_week_css)

# txt += "\nepiweek_rmses: \n"+str(epiweek_rmses)
# txt += "\nepiweek_nrmses: \n"+str(epiweek_nrmses)
# txt += "\nepiweek_crpss: \n"+str(epiweek_crpss)
# txt += "\nepiweek_css: \n"+str(epiweek_css)

rmses_to_mean = []
nrmses_to_mean = []
crpss_to_mean = []
css_to_mean = []
for week in epiweek_rmses:
    rmses_to_mean.append(epiweek_rmses[week])
    nrmses_to_mean.append(epiweek_nrmses[week])
    crpss_to_mean.append(epiweek_crpss[week])
    css_to_mean.append(epiweek_css[week])

rmse_mean = np.mean(rmses_to_mean)
nrmse_mean = np.mean(nrmses_to_mean)
crps_mean = np.mean(crpss_to_mean)
cs_mean = np.mean(css_to_mean)

rmse_all_states_means = [np.mean(x) for x in rmse_all_states]
nrmse_all_states_means = [np.mean(x) for x in nrmse_all_states]
crps_all_states_means = [np.mean(x) for x in crps_all_states]
cs_all_states_means = [np.mean(x) for x in cs_all_states]

txt_all += "\n\nMean computed results: \n"
print("\n\nMean computed results: \n")
for st, state in enumerate(states):
    txt_all = txt_all + \
    "\n"+state+"\n"+ \
    "rmse_all_states_means: "+str(rmse_all_states_means[st])+"\n"+\
    "nrmse_all_states_means: "+str(nrmse_all_states_means[st])+"\n"+\
    "crps_all_states_means: "+str(crps_all_states_means[st])+"\n"+\
    "cs_all_states_means: "+str(cs_all_states_means[st])+"\n"+\
    "\n"


print("rmse_mean: "+str(np.mean(rmse_all_states_means)))
print("nrmse_mean: "+str(np.mean(nrmse_all_states_means)))
print("crps_mean: "+str(np.mean(crps_all_states_means)))
print("cs_mean: "+str(np.mean(cs_all_states_means)))

txt_all += "\nmean_rmses: \n"+str(np.mean(rmse_all_states_means))
txt_all += "\nmean_nrmses: \n"+str(np.mean(nrmse_all_states_means))
txt_all += "\nmean_crpss: \n"+str(np.mean(crps_all_states_means))
txt_all += "\nmean_css: \n"+str(np.mean(cs_all_states_means))


f = open("plots_flu/"+file_initials.split("/")[1]+"results.txt", "w")
f.write(txt_all)
f.close()
print("\nSaved as: "+"plots_flu/"+file_initials.split("/")[1]+"results.txt")

f = open("plots_flu/"+file_initials.split("/")[1]+"plainvalues.pkl","wb")
pickle.dump([all_yps, all_yts],f)
f.close()

sys.exit(0)
"""



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