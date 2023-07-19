import subprocess
import pudb
from tqdm import tqdm
from optparse import OptionParser
import sys
parser = OptionParser()
parser.add_option("-d", "--disease", dest="disease", default="traffic", type="string")
parser.add_option("--epochs", dest="epochs", default=300, type="int")
parser.add_option("--size", dest="window_size", type="int", default=128)
parser.add_option("--stride", dest="window_stride", type="int", default=1000)
parser.add_option("--preprocess", dest="preprocess", action="store_true", default=False)
parser.add_option("--cnn", dest="cnn", action="store_true", default=False)
parser.add_option("--rag", dest="rag", action="store_true", default=False)
parser.add_option("--nn", dest="nn", default="none", type="choice", choices=["none", "simple", "bn", "dot", "bert"])
parser.add_option("--auto-size-best-num", dest="auto_size_best_num", default=None, type="int")
parser.add_option("--seed", dest="seed", default=0, type="int")
parser.add_option("--smart-mode", dest="smart_mode", default=0, type="int")

# epiweeks = list(range(202101, 202153))
(options, args) = parser.parse_args()
# pu.db

if options.cnn and options.rag:
    print("Cannot have both cnn and rag")
    sys.exit(0)

# states = [
#     "AL",
#     "AK",
#     "AZ",
#     "AR",
#     "CA",
#     "CO",
#     "CT",
#     "DE",
#     "DC",
#     "FL",
#     "GA",
#     "ID",
#     "IL",
#     "IN",
#     "IA",
#     "KS",
#     "KY",
#     "LA",
#     "ME",
#     "MD",
#     "MA",
#     "MI",
#     "MN",
#     "MS",
#     "MO",
#     "MT",
#     "NE",
#     "NV",
#     "NH",
#     "NJ",
#     "NM",
#     "NY",
#     "NC",
#     "ND",
#     "OH",
#     "OK",
#     "OR",
#     "PA",
#     "RI",
#     "SC",
#     "SD",
#     "TN",
#     "TX",
#     "UT",
#     "VT",
#     "VA",
#     "WA",
#     "WV",
#     "WI",
#     "WY",
#     "X",
# ]

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


for pat in patience:
    for sample in sample_out:
        for lr_ in lr:
                for ah in ahead:
                    to_run = []
                    if options.auto_size_best_num is not None:
                        save_model = f"slidingwindow_disease_{options.disease}_weekahead_{ah}_autosize_{options.auto_size_best_num}"
                    else:
                        save_model = f"slidingwindow_disease_{options.disease}_weekahead_{ah}_wsize_{options.window_size}_wstride_{options.window_stride}"

                    if options.preprocess:
                        save_model = f"slidingwindowpreprocessed_disease_{options.disease}_weekahead_{ah}_wsize_{options.window_size}_wstride_{options.window_stride}"
                        to_run = ["--preprocess"] + to_run
                    
                    if options.smart_mode != 0:
                        save_model += "_smart-mode-"+str(options.smart_mode)
                        to_run = to_run +["--smart-mode", str(options.smart_mode)]
                    
                    if options.cnn:
                        save_model = "cnn_" + save_model
                        to_run = ["--cnn"] + to_run
                    elif options.rag:
                        save_model = "rag_" + save_model
                        to_run = ["--rag"] + to_run
                    elif options.nn != "none":
                        save_model = "nn-"+options.nn + "_" + save_model
                        to_run = to_run + ["--nn", options.nn]
                    if options.auto_size_best_num is not None:
                        to_run = ["--auto-size-best-num", str(options.auto_size_best_num)] + to_run
                    if options.seed != 0:
                        save_model = save_model + "_seed_"+str(options.seed)
                        to_run = to_run +["--seed", str(options.seed)]

                    print(f"Training {save_model}")
                    
                    to_run =  [
                                "python",
                                "train_traffic_revised_refsetsupdated.py",
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
                                "traffic",
                                "-W",
                                "--sliding-window-stride",
                                str(options.window_stride),
                                "--sliding-window-size",
                                str(options.window_size),
                            ] + to_run
                    
                    subprocess.run(
                        to_run
                    )
