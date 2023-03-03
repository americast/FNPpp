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

parser = OptionParser()
parser.add_option("-f", "--files", dest="file_names", default="sliding_model_202252_True_0.001_500_4", type="string")
parser.add_option("-s", "--state", dest="state", default="AR", type="string")
(options, args) = parser.parse_args()
if options.file_names.split("_")[0] == "sliding":
    slide_text = "slidingwindow"
else:
    slide_text = "normal"
# writer = SummaryWriter("runs/eval/covid_slidingwindow_"+str(epiweek_pres)+"_weekahead_"+str(options.day_ahead)+"_windowsize_"+str(options.window_size)+"_stride_"+str(options.window_stride))
writer = SummaryWriter("runs/eval/covid_"+slide_text+"_"+str(options.file_names))

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
with open("val_predictions_"+slide_text+"/"+options.file_names+"_predictions.pkl", "rb") as f:
    data_dict = pickle.load(f)
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

