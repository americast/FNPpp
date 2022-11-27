import subprocess

epiweeks = [202244, 202245, 202246]

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
for week in epiweeks:
    subprocess.run(
        [
            "bash",
            "./scripts/hosp_preprocess.sh",
            str(week),
        ]
    )


sample_out = [True]
lr = 0.001
segments = [2, 4, 8]
adaptive = [True, False]

for sample in sample_out:
    for segment in segments:
        for adapt in adaptive:
            for week in epiweeks:
                for state in states:
                    save_model = f"ar_seg_reg_real_{sample}_{segment}_{adapt}"
                    print(f"Training {save_model}")
                    subprocess.run(
                        [
                            "python",
                            "train_hosp_ar_seg_reg_real.py",
                            "--epiweek",
                            str(week),
                            "--region",
                            state,
                            "--lr",
                            str(lr),
                            "--sample_out" if sample else "",
                            "--save",
                            save_model,
                            "--cuda",
                            "--segments",
                            str(segment),
                            "--adaptive" if adapt else "",
                            "--num_workers",
                            "8",
                        ]
                    )
