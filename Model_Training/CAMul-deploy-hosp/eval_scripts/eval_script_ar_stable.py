import pickle
import os
import pandas as pd
import numpy as np
from covid_extract.hosp_consts import states
from eval_metrics import rmse, nrmse, mape, crps_samples, get_pr


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


sample_out = [True, False]
lr = [0.001, 0.0001]
epiweeks = list(range(202101, 202153))
ahead = [1]

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

with open("ar_metrics_stable.pkl", "wb") as f:
    pickle.dump(metrics_df, f)

# Save as csv
metrics_df.to_csv("ar_metrics_stable.csv", index=False)

# Avergae over all epiweeks
metrics_all_weeks = metrics_df.groupby(["sample_out", "lr", "ahead"]).mean().reset_index()
metrics_all_weeks.to_csv("ar_metrics_stable_all_weeks.csv", index=False)

