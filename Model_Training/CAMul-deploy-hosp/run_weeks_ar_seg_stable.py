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
lr = 0.001
segments = [2, 3, 4, 8]
adaptive = [True, False]

for sample in sample_out:
    for segment in segments:
        for adapt in adaptive:
            for week in epiweeks:
                save_model = f"ar_seg_model_{week}_{sample}_{segment}_{adapt}"
                print(f"Training {save_model}")
                subprocess.run(
                    [
                        "python",
                        "train_hosp_ar_revised_seg.py",
                        "--epiweek",
                        str(week),
                        "--lr",
                        str(lr),
                        "--sample_out" if sample else "",
                        "--save",
                        save_model,
                        "--cuda",
                        "--segments",
                        str(segment),
                        "--adaptive" if adapt else "",
                    ]
                )
