import subprocess
import pudb
from tqdm import tqdm
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-d", "--disease", dest="disease", default="traffic", type="string")
parser.add_option("--epochs", dest="epochs", default=500, type="int")
parser.add_option("--cnn", dest="cnn", action="store_true", default=False)
# parser.add_option("--nndot", dest="nndot", action="store_true", default=False)
parser.add_option("--nn", dest="nn", default="none", type="choice", choices=["none", "simple", "bn", "dot", "bert"])
parser.add_option("--rag", dest="rag", action="store_true", default=False)
parser.add_option("--seed", dest="seed", default=0, type="int")
parser.add_option("--val", dest="val", action="store_true", default=False)
parser.add_option("--smart-mode", dest="smart_mode", default=0, type="int")

# Ours (smart mode 8)
# EpiFNP + conv (smart mode 5)
# EpiFNP + partitioning (smart mode 1)

# epiweeks = list(range(202101, 202153))
(options, args) = parser.parse_args()
# if options.epiweek is None:
#     epiweeks = list(range(202101, 202153,4))
# else:
#     epiweeks = [options.epiweek]
# pu.db
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
lr = [0.0001]
# patience = [1000, 3000]
patience = [100]
ahead = [1, 2, 3]
# ahead = [4]


for pat in patience:
    for sample in sample_out:
        for lr_ in lr:
                for ah in ahead:
                    save_model = f"disease_{options.disease}_weekahead_{ah}"
                    to_run = []
                    if options.cnn:
                        save_model = "cnn_" + save_model
                        to_run = to_run + ["--cnn"]
                    elif options.rag:
                        save_model = "rag_" + save_model
                        to_run = to_run + ["--rag"]
                    elif options.nn != "none":
                        save_model = "nn-"+options.nn + "_" + save_model
                        to_run = to_run + ["--nn", options.nn]
                    else:
                        save_model = "normal_" + save_model
                    if options.smart_mode != 0:
                        save_model += "_smart-mode-"+str(options.smart_mode)
                        to_run = to_run +["--smart-mode", str(options.smart_mode)]
                    if options.seed != 0:
                        save_model = save_model + "_seed_"+str(options.seed)
                        to_run = to_run + ["--seed", str(options.seed)]

                    print(f"Training {save_model}")
                    if options.val:
                        subprocess.run(
                            [
                                "python",
                                "val_traffic_revised_refsetsupdated.py",
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
                            ] + to_run
                        )
                    else:
                        subprocess.run(
                            [
                                "python",
                                "train_traffic.py",
                                "--save",
                                save_model,
                                "-w",
                                str(ah),
                            ] + to_run
                        )