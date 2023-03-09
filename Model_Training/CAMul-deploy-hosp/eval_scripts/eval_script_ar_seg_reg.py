import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from covid_extract.hosp_consts import states
from eval_metrics import rmse, nrmse, mape, crps_samples, get_pr


def get_metrics(file_name, epiweek_now, ahead):
    if not os.path.exists(file_name):
        print(f"File {file_name} does not exist")
        return None

    with open(file_name, "rb") as f:
        fl_data = pickle.load(f)
        predictions = fl_data[0][:, :, ahead - 1]
        ground_truth = fl_data[1][:, ahead - 1]

    mean_preds = predictions.mean(axis=0)
    std_preds = predictions.std(axis=0)

    metrics = {
        # "rmse": rmse(mean_preds, ground_truth),
        # "nrmse": nrmse(mean_preds, ground_truth),
        # "mape": mape(mean_preds, ground_truth),
        # "crps": crps_samples(predictions.T, ground_truth),
        # "cs": get_pr(mean_preds, std_preds, ground_truth)[1],
        "ground_truth": ground_truth,
        "mean_preds": mean_preds,
        "std_preds": std_preds,
    }
    return metrics


def plot_round(ground_truth, mean_preds, std_preds, title):
    for i in range(ground_truth.shape[0]):
        plt.clf()
        plt.title(f"{title} {states[i]}")
        plt.plot(ground_truth[i], label="Ground Truth")
        plt.plot(mean_preds[i], label="Mean Prediction")
        plt.fill_between(
            np.arange(ground_truth.shape[1]),
            mean_preds[i] - std_preds[i],
            mean_preds[i] + std_preds[i],
            alpha=0.3,
            label="Std Prediction",
        )
        plt.legend()
        os.makedirs(f"plots_seg_stable/{title}", exist_ok=True)
        plt.savefig(f"plots_seg_stable/{title}/{states[i]}.png")


sample_out = [True, False]
lr = [0.001, 0.0001]
epiweeks = list(range(202101, 202153))
ahead = [1, 2, 3, 4]

sample_out = [True, False]
lr = 0.001
segments = [2, 3, 4, 8]
adaptive = [True, False]

metrics_df = pd.DataFrame(
    columns=[
        "sample_out",
        "ahead",
        "epiweek",
        "segments",
        "adaptive",
        "rmse",
        "nrmse",
        "mape",
        "crps",
        "cs",
    ]
)
for sample in sample_out:
    for segment in segments:
        for adapt in adaptive:
            for ah in ahead:
                gt, mean, std = [], [], []
                for week in epiweeks:
                    save_model = f"ar_seg_model_{week}_{sample}_{segment}_{adapt}"
                    file_name = os.path.join(
                        "hosp_stable_predictions", f"{save_model}_predictions.pkl"
                    )
                    print(f"Getting metrics for {file_name}")
                    metrics = get_metrics(file_name, week, ah)

                    if metrics is None:
                        continue

                    gt.append(metrics["ground_truth"])
                    mean.append(metrics["mean_preds"])
                    std.append(metrics["std_preds"])
                    # print([metrics[m] for m in metrics_df.columns[-5:]])
                    # metrics_df = metrics_df.append(
                    #    {
                    #        "sample_out": sample,
                    #        "ahead": ah,
                    #        "epiweek": week,
                    #        "segments": segment,
                    #        "adaptive": adapt,
                    # "rmse": metrics["rmse"],
                    # "nrmse": metrics["nrmse"],
                    # "mape": metrics["mape"],
                    # "crps": metrics["crps"],
                    # "cs": metrics["cs"],
                    #    },
                    #    ignore_index=True,
                    # )
                gt = np.array(gt).T
                mean = np.array(mean).T
                std = np.array(std).T
                print(mean, std)
                plot_round(
                    gt, mean, std, f"ar_seg_stable_{sample}_{segment}_{adapt}/{ah}"
                )

with open("ar_metrics_stable.pkl", "wb") as f:
    pickle.dump(metrics_df, f)

# Save as csv
metrics_df.to_csv("ar_seg_metrics_stable.csv", index=False)

# Avergae over all epiweeks
metrics_all_weeks = (
    metrics_df.groupby(["sample_out", "ahead", "segments", "adaptive"])
    .mean()
    .reset_index()
)
metrics_all_weeks.to_csv("ar_seg_metrics_stable_all_weeks.csv", index=False)
