import subprocess

epiweeks = list(range(202101, 202153))

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

subprocess.run(
        [
            "bash",
            "./scripts/hosp_preprocess.sh",
            str("202240"),
        ]
    )

sample_out = [True, False]
lr = [0.001, 0.0001]
patience = [1000, 3000]
ahead = [1, 2, 3, 4]


for pat in patience:
    for sample in sample_out:
        for lr_ in lr:
            for week in epiweeks:
                for ah in ahead:
                    save_model = f"normal_model_{week}_{sample}_{lr_}_{pat}_{ah}"
                    print(f"Training {save_model}")
                    subprocess.run(
                        [
                            "python",
                            "train_hosp_revised.py",
                            "--epiweek",
                            str(week),
                            "--lr",
                            str(lr_),
                            "--save",
                            save_model,
                            "--epochs",
                            "5000",
                            "--patience",
                            str(pat),
                            "-d",
                            str(ah),
                        ]
                    )
