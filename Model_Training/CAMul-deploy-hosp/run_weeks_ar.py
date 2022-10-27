import subprocess

epiweeks = [202150, 202152, 202204, 202208, 202212, 202216] + list(
    range(202220, 202237, 4)
)

sample_out = [True, False]
lr = [0.001, 0.0001]

for sample in sample_out:
    for l in lr:
        for week in epiweeks:
            save_model = f"ar_model_{week}_{sample}_{l}"
            subprocess.run(
                [
                    "python",
                    "train_hosp_ar.py",
                    "--epiweek",
                    str(week),
                    "--lr",
                    str(l),
                    "--sample_out" if sample else "",
                    "--save",
                    save_model,
                ]
            )
