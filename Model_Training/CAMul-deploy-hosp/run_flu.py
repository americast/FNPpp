import subprocess
import pudb
from tqdm import tqdm
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-e", "--epiweek", dest="epiweek", default="202252", type="string")
parser.add_option("-d", "--disease", dest="disease", default="flu", type="string")
parser.add_option("--epochs", dest="epochs", default=1500, type="int")
parser.add_option("--cnn", dest="cnn", action="store_true", default=False)
parser.add_option("--rag", dest="rag", action="store_true", default=False)

# epiweeks = list(range(202101, 202153))
(options, args) = parser.parse_args()
if options.epiweek is None:
    epiweeks = list(range(202101, 202153,4))
else:
    epiweeks = [options.epiweek]
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

if options.epiweek is not None and int(options.epiweek)> 202153:
    subprocess.run(
            [
                "bash",
                "./scripts/hosp_preprocess.sh",
                options.epiweek,
            ]
        )
else:
    subprocess.run(
            [
                "bash",
                "./scripts/hosp_preprocess.sh",
                str("202240"),
            ]
        )

# sample_out = [True, False]
sample_out = [True]
# lr = [0.001, 0.0001]
lr = [0.001]
# patience = [1000, 3000]
patience = [500]
ahead = [1, 2, 3]
# ahead = [4]


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
