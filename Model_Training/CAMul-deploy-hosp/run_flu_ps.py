import subprocess
import pudb
from tqdm import tqdm
from optparse import OptionParser
import sys
parser = OptionParser()
parser.add_option("--epiweek_start", dest="epiweek_start", default="202132", type="string")
parser.add_option("--epiweek_end", dest="epiweek_end", default="202210", type="string")
parser.add_option("-d", "--disease", dest="disease", default="flu", type="string")
parser.add_option("--epochs", dest="epochs", default=500, type="int")
parser.add_option("--seed", dest="seed", default=0, type="int")
parser.add_option("--cnn", dest="cnn", action="store_true", default=False)
parser.add_option("--rag", dest="rag", action="store_true", default=False)
parser.add_option("--nn", dest="nn", default="none", type="choice", choices=["none", "simple", "bn", "dot", "bert"])
parser.add_option("--bert", dest="bert", action="store_true", default=False)
parser.add_option("--smart-mode", dest="smart_mode", default=0, type="int")
parser.add_option("--optionals", dest="optionals", default=" ", type="str")

# 1 divide ref sets as per new rule
# 2 use smoothing throughout (even for output)
# 3 use smoothing for all inputs
# 4 use smoothing for all inputs and divide ref sets as per new rule
# 5 use smoothing for only inputs
# 6 use smoothing for only ref sets
# 7 conv for inp
# epiweeks = list(range(202101, 202153))
(options, args) = parser.parse_args()
if "2022" in options.epiweek_end:
    epiweeks = list(range(int(options.epiweek_start), 202152)) + list(range(202201, int(options.epiweek_end)))
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

# subprocess.run(
#         [
#             "bash",
#             "./scripts/hosp_preprocess.sh",
#             options.epiweek_end,
#         ]
#     )

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
patience = [100]
ahead = [1,2,3]
# ahead = [4]
epiweeks = epiweeks[:-max(ahead)]

for pat in patience:
    for sample in sample_out:
        for lr_ in lr:
            for week in tqdm(epiweeks):
                for ah in ahead:
                    save_model = "ps_disease_"+str(options.disease)+"_epiweek_"+str(week)+"_weekahead_"+str(ah)
                    to_run = []
                    if options.smart_mode != 0:
                        save_model += "_smart-mode-"+str(options.smart_mode)
                        to_run = to_run +["--smart-mode", str(options.smart_mode)]
                    if options.cnn:
                        save_model = "cnn_"+save_model
                        to_run = to_run +["--cnn"]
                    elif options.rag:
                        save_model = "rag_"+save_model
                        to_run = to_run +["--rag"]
                    elif options.nn != "none":
                        save_model = "nn-"+options.nn + "_" + save_model
                        to_run = to_run + ["--nn", options.nn]
                    else:
                        save_model = "normal_"+save_model
                    
                    if options.bert:
                        save_model = "bert_" + save_model
                        to_run = to_run + ["--bert-emb"]

                    if options.seed != 0:
                        save_model = save_model + "_seed_"+str(options.seed)
                        to_run = to_run +["--seed", str(options.seed)]

                    to_run = [
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
                                str(options.disease),
                                "--optionals",
                                options.optionals
                        ] + to_run
                    print(f"Training {save_model}")
                    subprocess.run(
                        to_run
                    )
