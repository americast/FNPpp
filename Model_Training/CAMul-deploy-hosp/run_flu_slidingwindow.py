import subprocess
import pudb
from tqdm import tqdm
from optparse import OptionParser
import sys
parser = OptionParser()
parser.add_option("--epiweek_start", dest="epiweek_start", default="202232", type="string")
parser.add_option("--epiweek_end", dest="epiweek_end", default="202310", type="string")
parser.add_option("-d", "--disease", dest="disease", default="flu", type="string")
parser.add_option("--epochs", dest="epochs", default=500, type="int")
parser.add_option("--auto-size-best-num", dest="auto_size_best_num", default=None, type="int")
parser.add_option("--smart-mode", dest="smart_mode", default=0, type="int")
# 1 divide ref sets as per new rule
# 2 use smoothing throughout (even for output)
# 3 use smoothing for all inputs
# 4 use smoothing for all inputs and divide ref sets as per new rule
# 5 use smoothing for only inputs
# 6 use smoothing for only ref sets
parser.add_option("--size", dest="window_size", type="int", default=10)
parser.add_option("--stride", dest="window_stride", type="int", default=10)
parser.add_option("--preprocess", dest="preprocess", action="store_true", default=False)
parser.add_option("--cnn", dest="cnn", action="store_true", default=False)
parser.add_option("--rag", dest="rag", action="store_true", default=False)
parser.add_option("-p", "--use-pretrained", dest="use_pretrained", action="store_true", default=False)
parser.add_option("--seed", dest="seed", default=0, type="int")
parser.add_option("--bert", dest="bert", action="store_true", default=False)
parser.add_option("--nn", dest="nn", default="none", type="choice", choices=["none", "simple", "bn", "dot", "bert"])


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

# sample_out = [True, False]
sample_out = [True]
# lr = [0.001, 0.0001]
lr = [0.001]
# patience = [1000, 3000]
patience = [100]
ahead = [1,2,3]
# ahead = [4]


for pat in patience:
    for sample in sample_out:
        for lr_ in lr:
            for week in tqdm(epiweeks):
                for ah in ahead:
                    to_run = []
                    if options.auto_size_best_num is not None:
                        save_model = f"slidingwindow_disease_{options.disease}_epiweek_{week}_weekahead_{ah}_autosize_{options.auto_size_best_num}"
                    elif options.smart_mode != 0:
                        save_model = f"slidingwindow_disease_{options.disease}_epiweek_{week}_weekahead_{ah}_smart-mode-{options.smart_mode}"
                    else:
                        save_model = f"slidingwindow_disease_{options.disease}_epiweek_{week}_weekahead_{ah}_wsize_{options.window_size}_wstride_{options.window_stride}"

                    if options.preprocess:
                        save_model = f"slidingwindowpreprocessed_disease_{options.disease}_epiweek_{week}_weekahead_{ah}_wsize_{options.window_size}_wstride_{options.window_stride}"
                        to_run = ["--preprocess"] + to_run
                    
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
                    elif options.smart_mode != 0:
                        to_run = to_run +["--smart-mode", str(options.smart_mode)]
                    if options.seed != 0:
                        save_model = save_model + "_seed_"+str(options.seed)
                        to_run = to_run +["--seed", str(options.seed)]
                    if options.bert:
                        save_model = "bert_" + save_model
                        to_run = to_run + ["--bert-emb"]

                    print(f"Training {save_model}")
                    
                    to_run =  [
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
                                "-W",
                                "--sliding-window-stride",
                                str(options.window_stride),
                                "--sliding-window-size",
                                str(options.window_size),
                            ] + to_run
                    
                    if options.use_pretrained:
                        to_run = to_run + ["--start_model", "/localscratch/ssinha97/fnp_saved_models/fluhosp_models/normal_disease_flu_epiweek_"+str(week)+"_weekahead_"+str(ah)]
                    subprocess.run(
                        to_run
                    )
                    # sys.exit(0)
