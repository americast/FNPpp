import pickle
import os
import pandas as pd
import numpy as np
from covid_extract.hosp_consts import states
from eval_metrics import rmse, nrmse, mape, crps_samples, get_pr


EPIWEEK_LATEST = 202240
LATEST_FILE = os.path.join(
    "data",
    "covid_data",
    f"covid-hospitalization-all-state-merged_vEW{EPIWEEK_LATEST}.csv",
)


def get_metrics(file_name, epiweek_now, ahead):
    if not os.path.exists(file_name):
        print(f"File {file_name} does not exist")
        return None

    with open(file_name, "rb") as f:
        predictions: np.ndarray = pickle.load(f)[:, :, ahead - 1]

    gt_df = pd.read_csv(LATEST_FILE)
    df = gt_df[["region", "epiweek", "cdc_hospitalized"]].copy()
    df = df[df["epiweek"] == epiweek_now + ahead]

    ground_truth_ = []
    for state in states:
        ground_truth_.append(
            gt_df[gt_df["region"] == state]["cdc_hospitalized"].iloc[-1]
        )

    ground_truth = np.array(ground_truth_)

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


sample_out = [True, False]
lr = [0.001, 0.0001]
epiweeks = list(range(202101, 202153))
ahead = [1, 2, 3, 4]

metrics_df = pd.DataFrame(
    columns=[
        "sample_out",
        "lr",
        "ahead",
        "epiweek",
        "rmse",
        "nrmse",
        "mape",
        "crps",
        "cs",
    ]
)

for sample in sample_out:
    for lr_ in lr:
        for week in epiweeks:
            for ah in ahead:
                save_model = f"ar_model_{week}_{sample}_{lr_}"
                file_name = os.path.join(
                    "hosp_stable_predictions", f"{save_model}_predictions.pkl"
                )
                print(f"Getting metrics for {file_name}")
                metrics = get_metrics(file_name, week, ah)

                if metrics is None:
                    continue
                print(metrics)
                metrics_df = metrics_df.append(
                    {
                        "sample_out": sample,
                        "lr": lr_,
                        "ahead": ah,
                        "epiweek": week,
                        "rmse": metrics["rmse"],
                        "nrmse": metrics["nrmse"],
                        "mape": metrics["mape"],
                        "crps": metrics["crps"],
                        "cs": metrics["cs"],
                    },
                    ignore_index=True,
                )

with open("ar_metrics.pkl", "wb") as f:
    pickle.dump(metrics_df, f)

# Save as csv
metrics_df.to_csv("ar_metrics.csv", index=False)
