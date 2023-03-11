import subprocess
import pudb
from tqdm import tqdm
from optparse import OptionParser
import sys
parser = OptionParser()
parser.add_option("--epiweek_start", dest="epiweek_start", default="202232", type="string")
parser.add_option("--epiweek_end", dest="epiweek_end", default="202310", type="string")
parser.add_option("-d", "--disease", dest="disease", default="flu", type="string")
parser.add_option("--epochs", dest="epochs", default=300, type="int")
parser.add_option("--cnn", dest="cnn", action="store_true", default=False)
parser.add_option("--rag", dest="rag", action="store_true", default=False)

# epiweeks = list(range(202101, 202153))
(options, args) = parser.parse_args()
if "2023" in options.epiweek_end:
    epiweeks = list(range(int(options.epiweek_start), 202252)) + list(range(202301, int(options.epiweek_end)))
else:
    epiweeks = list(range(int(options.epiweek_start), int(options.epiweek_end)))

# pu.db

if options.cnn and options.rag:
    print("Cannot have both cnn and rag")
    sys.exit(0)
# pu.db
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
            options.epiweek_end,
        ]
    )

# if options.epiweek is not None and int(options.epiweek)> 202153:
#     subprocess.run(
#             [
#                 "bash",
#                 "./scripts/hosp_preprocess.sh",
#                 options.epiweek,
#             ]
#         )
# else:
#     subprocess.run(
#             [
#                 "bash",
#                 "./scripts/hosp_preprocess.sh",
#                 str("202240"),
#             ]
#         )

# sample_out = [True, False]
sample_out = [True]
# lr = [0.001, 0.0001]
lr = [0.001]
# patience = [1000, 3000]
patience = [500]
ahead = [1,2,3]
# ahead = [4]
epiweeks = epiweeks[:-max(ahead)]

for pat in patience:
    for sample in sample_out:
        for lr_ in lr:
            for week in tqdm(epiweeks):
                for ah in ahead:
                    if options.cnn:
                        save_model = f"cnn_disease_{options.disease}_epiweek_{week}_weekahead_{ah}"
                        print(f"Training {save_model}")
                        subprocess.run(
                            [
                                "python",
                                "train_hosp_revised_refsetsupdated.py",
                                "--epiweek",
                                str(week),
                                "--lr",
                                str(lr_),
                                "--save",
                                save_model,
                                "--epochs",
                                str(options.epochs),
                                "--patience",
                                str(pat),
                                "-d",
                                str(ah),
                                "--tb",
                                "--disease",
                                "flu",
                                "--cnn",
                            ]
                        )
                    elif options.rag:
                        save_model = f"rag_disease_{options.disease}_epiweek_{week}_weekahead_{ah}"
                        print(f"Training {save_model}")
                        subprocess.run(
                            [
                                "python",
                                "train_hosp_revised_refsetsupdated.py",
                                "--epiweek",
                                str(week),
                                "--lr",
                                str(lr_),
                                "--save",
                                save_model,
                                "--epochs",
                                str(options.epochs),
                                "--patience",
                                str(pat),
                                "-d",
                                str(ah),
                                "--tb",
                                "--disease",
                                "flu",
                                "--rag",
                            ]
                        )
                    else:
                        save_model = f"normal_disease_{options.disease}_epiweek_{week}_weekahead_{ah}"
                        print(f"Training {save_model}")
                        subprocess.run(
                            [
                                "python",
                                "train_hosp_revised_refsetsupdated.py",
                                "--epiweek",
                                str(week),
                                "--lr",
                                str(lr_),
                                "--save",
                                save_model,
                                "--epochs",
                                str(options.epochs),
                                "--patience",
                                str(pat),
                                "-d",
                                str(ah),
                                "--tb",
                                "--disease",
                                "flu",
                            ]
                        )
