import subprocess

epiweeks = [202204, 202208, 202212, 202216] + list(range(202220, 202241, 4))

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

for epiweek in epiweeks:
    subprocess.run(
        [
            "bash",
            "./scripts/hosp_preprocess.sh",
            str(epiweek),
        ]
    )

sample_out = [True, False]
lr = [0.001, 0.0001]

for sample in sample_out:
    for l in lr:
        for week in epiweeks:
            save_model = f"ar_model_{week}_{sample}_{l}"
            print(f"Training {save_model}")
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
