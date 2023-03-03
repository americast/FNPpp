from curses import raw
import itertools
import numpy as np
import pickle
import os

from optparse import OptionParser
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
import torch
from covid_extract.hosp_consts import include_cols_flu as include_cols
from models.multimodels import (
    EmbedEncoder,
    GRUEncoder,
    EmbGCNEncoder,
    LatentEncoder,
    CorrEncoder,
    Decoder,
)
from models.fnpmodels import RegressionFNP2
from tqdm import tqdm
parser = OptionParser()
# parser.add_option("-p", "--epiweek_pres", dest="epiweek_pres", default="202240", type="string")
# parser.add_option("-e", "--epiweek", dest="epiweek", default="202140", type="string")
# parser.add_option("--epochs", dest="epochs", default=3500, type="int")
parser.add_option("--lr", dest="lr", default=1e-3, type="float")
parser.add_option("--patience", dest="patience", default=1000, type="int")
# parser.add_option("-d", "--day", dest="day_ahead", default=1, type="int")
parser.add_option("-d", "--splits", dest="splits", default=10, type="int")
parser.add_option("-s", "--seed", dest="seed", default=0, type="int")
parser.add_option("-b", "--batch", dest="batch_size", default=100, type="int")
# parser.add_option("-m", "--save", dest="save_model", default="default", type="string")
# parser.add_option("--start_model", dest="start_model", default="None", type="string")
parser.add_option("-c", "--cuda", dest="cuda", default=True, action="store_true")
# parser.add_option("--start", dest="start_day", default=-120, type="int")
parser.add_option("-t", "--tb", action="store_true", dest="tb", default=True)

# parser = OptionParser()
parser.add_option("-w", "--week", dest="week_ahead", type="int", default=0)
parser.add_option("-y", "--year", dest="year", type="int", default=2020)
parser.add_option("-n", "--num", dest="num", type="string")
parser.add_option("-e", "--epoch", dest="epochs", type="int", default="1500")
(options, args) = parser.parse_args()
# (options, args) = parser.parse_args()
# epiweek_pres = options.epiweek_pres
# epiweek = options.epiweek
# day_ahead = options.day_ahead
seed = options.seed
# save_model_name = options.save_model
# start_model = options.start_model
cuda = options.cuda
# start_day = options.start_day
batch_size = options.batch_size
lr = options.lr
epochs = options.epochs
patience = options.patience
# First do sequence alone
# Then add exo features
# Then TOD (as feature, as view)
# llely demographic features

# Initialize random seed
np.random.seed(seed)
torch.manual_seed(seed)

import random
random.seed(seed)

float_tensor = (
    torch.cuda.FloatTensor
    if (cuda and torch.cuda.is_available())
    else torch.FloatTensor
)
long_tensor = (
    torch.cuda.LongTensor if (cuda and torch.cuda.is_available()) else torch.LongTensor
)
device = "cuda" if (cuda and torch.cuda.is_available()) else "cpu"




with open("./data/household_power_consumption/household_power_consumption.txt", "r") as f:
    data = f.readlines()

data = [d.strip().split(";") for d in data][1:]

def get_month(ss: str):
    i = ss.find("/")
    return int(ss[i+1:ss[i+1:].find("/")+i + 1]) - 1
def get_time_of_day(ss: str):
    hour = int(ss[:2])
    if hour < 6:
        return 0
    elif hour < 12:
        return 1
    elif hour < 18:
        return 2
    else:
        return 3

tod = np.array([get_time_of_day(d[1]) for d in data], dtype=np.int32)
month = np.array([get_month(d[0]) for d in data], dtype=np.int32)
features = []
for d in data:
    f = []
    for x in d[2:]:
        try:
            f.append(float(x))
        except:
            f.append(0.0)
    features.append(f)
features = np.array(features)
target = features[:, 0]

total_time = len(data)
test_start = int(total_time * 0.8)

X, X_symp, Y, mt, reg = [], [], [], [], []

def sample_train(n_samples, window = 20):
    X, X_symp, Y, mt, reg = [], [], [], [], []
    start_seqs = np.random.randint(0, test_start, n_samples)
    for start_seq in start_seqs:
        X.append(target[start_seq:start_seq+window-options.week_ahead, np.newaxis])
        X_symp.append(features[start_seq:start_seq+window-options.week_ahead])
        if options.week_ahead == 0:
            Y.append(target[start_seq+window])
        else:
            Y.append(target[start_seq+window-options.week_ahead:start_seq+window])
        mt.append(month[start_seq+window])
        reg.append(tod[start_seq+window])
    X = np.array(X)
    X_symp = np.array(X_symp)
    Y = np.array(Y)
    mt = np.array(mt)
    reg = np.array(reg)
    return X, X_symp, Y, mt, reg

def sample_test(n_samples, window = 20):
    X, X_symp, Y, mt, reg = [], [], [], [], []
    start_seqs = np.random.randint(test_start, total_time, n_samples)
    for start_seq in start_seqs:
        X.append(target[start_seq:start_seq+window-options.week_ahead, np.newaxis])
        X_symp.append(features[start_seq:start_seq+window-options.week_ahead])
        if options.week_ahead == 0:
            Y.append(target[start_seq+window])
        else:
            Y.append(target[start_seq+window-options.week_ahead:start_seq+window])
        mt.append(month[start_seq+window])
        reg.append(tod[start_seq+window])
    X = np.array(X)
    X_symp = np.array(X_symp)
    Y = np.array(Y)
    mt = np.array(mt)
    reg = np.array(reg)
    return X, X_symp, Y, mt, reg




# Reference points
splits = options.splits
len_seq = test_start//splits
# seq_references = features[np.newaxis,:,0, np.newaxis]
seq_references = np.array([features[i: i+len_seq, 0, np.newaxis] for i in range(0, test_start, len_seq)])[:, :options.batch_size, :]
symp_references = np.array([features[i: i+len_seq] for i in range(0, test_start, len_seq)])[:, :options.batch_size, :]
# month_references = np.arange(12)
# reg_references = np.arange(4)
# pu.db
train_seqs, train_symp_seqs, train_y, mt, reg = sample_train(options.batch_size)
val_seqs, val_symp_seqs, val_y, mt_val, reg_val = sample_test(options.batch_size)

# pu.db


# Get dataset as numpy arrays

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

# raw_data = []
# for st in states:
#     with open(f"./data/hosp_data/saves/hosp_{st}_{epiweek_pres}.pkl", "rb") as fl:
#         raw_data.append(pickle.load(fl))

# def diff_epiweeks(epiweek1, epiweek2):
#     """
#     Compute difference in epiweeks
#     """
#     year1, week1 = int(epiweek1[:4]), int(epiweek1[4:])
#     year2, week2 = int(epiweek2[:4]), int(epiweek2[4:])
#     return (year1 - year2) * 52 + week1 - week2

# if diff_epiweeks(epiweek, epiweek_pres) > 0:
#     raw_data = np.array(raw_data)
# else:    
#     raw_data = np.array(raw_data)[:, :diff_epiweeks(epiweek, epiweek_pres) + day_ahead, :]  # states x days x features
# label_idx = include_cols.index("cdc_hospitalized")
# all_labels = raw_data[:, -1, label_idx]
# print(f"Diff epiweeks: {diff_epiweeks(epiweek, epiweek_pres)}")
# raw_data = raw_data[:, start_day:-day_ahead, :]

# raw_data_unnorm = raw_data.copy()

if options.tb:
    writer = SummaryWriter("runs/power/power_"+str(options.splits)+"_splits_weekahead_"+str(options.week_ahead))

# class ScalerFeat:
#     def __init__(self, raw_data):
#         self.means = np.mean(raw_data, axis=1)
#         self.vars = np.std(raw_data, axis=1) + 1e-8

#     def transform(self, data):
#         return (data - np.transpose(self.means[:, :, None], (0, 2, 1))) / np.transpose(
#             self.vars[:, :, None], (0, 2, 1)
#         )

#     def inverse_transform(self, data):
#         return data * np.transpose(self.vars[:, :, None], (0, 2, 1)) + np.transpose(
#             self.means[:, :, None], (0, 2, 1)
#         )

#     def transform_idx(self, data, idx=label_idx):
#         return (data - self.means[:, idx]) / self.vars[:, idx]

#     def inverse_transform_idx(self, data, idx=label_idx):
#         return data * self.vars[:, idx] + self.means[:, idx]


# scaler = ScalerFeat(raw_data)
# raw_data = scaler.transform(raw_data)


# Chunk dataset sequences


# def prefix_sequences(seq, day_ahead=day_ahead):
#     """
#     Prefix sequences with zeros
#     """
#     l = len(seq)
#     X, Y = np.zeros((l - day_ahead, l, seq.shape[-1])), np.zeros(l - day_ahead)
#     for i in range(l - day_ahead):
#         X[i, (l - i - 1) :, :] = seq[: i + 1, :]
#         Y[i] = seq[i + day_ahead, label_idx]
#     return X, Y


# X, Y = [], []
# for i, st in enumerate(states):
#     x, y = prefix_sequences(raw_data[i])
#     X.append(x)
#     Y.append(y)

# X_train, Y_train = np.concatenate(X), np.concatenate(Y)
# num_repeat = int(X_train.shape[0]/len(states))
# states_train_unflattened = [list(itertools.repeat(st, num_repeat)) for st in states]
# states_train = []
# for st_here in states_train_unflattened:
#     states_train.extend(st_here)
# Shuffle data
# perm = np.random.permutation(len(X_train))
# X_train, Y_train = X_train[perm], Y_train[perm]

# Reference sequences
# X_ref = raw_data[:, :, label_idx]

# Divide val and train
# frac = 0.1
# X_val, Y_val = X_train[: int(len(X_train) * frac)], Y_train[: int(len(X_train) * frac)], states_train[: int(len(X_train) * frac)]
# X_train, Y_train, states_train = (
#     X_train[int(len(X_train) * frac) :],
#     Y_train[int(len(X_train) * frac) :],
#     states_train[int(len(X_train) * frac) :],
# )


# Build model
feat_enc = GRUEncoder(in_size=7, out_dim=60,).to(device)
seq_enc = GRUEncoder(in_size=1, out_dim=60,).to(device)
fnp_enc = RegressionFNP2(
    dim_x=60,
    dim_y=1,
    dim_h=100,
    n_layers=3,
    num_M=batch_size,
    dim_u=60,
    dim_z=60,
    use_DAG=False,
    use_ref_labels=False,
    add_atten=False,
).to(device)


# def load_model(folder, file=save_model_name):
#     """
#     Load model
#     """
#     full_path = os.path.join(folder, file)
#     assert os.path.exists(full_path)
#     feat_enc.load_state_dict(torch.load(os.path.join(full_path, "feat_enc.pt")))
#     seq_enc.load_state_dict(torch.load(os.path.join(full_path, "seq_enc.pt")))
#     fnp_enc.load_state_dict(torch.load(os.path.join(full_path, "fnp_enc.pt")))


# def save_model(folder, file=save_model_name):
#     """
#     Save model
#     """
#     full_path = os.path.join(folder, file)
#     os.makedirs(full_path, exist_ok=True)
#     torch.save(feat_enc.state_dict(), os.path.join(full_path, "feat_enc.pt"))
#     torch.save(seq_enc.state_dict(), os.path.join(full_path, "seq_enc.pt"))
#     torch.save(fnp_enc.state_dict(), os.path.join(full_path, "fnp_enc.pt"))


# Build dataset
class SeqData(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y[:, None]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            float_tensor(self.X[idx, :, :]),
            float_tensor(self.Y[idx]),
        )

# Build dataset with state info
# class SeqDataWithStates(torch.utils.data.Dataset):
#     def __init__(self, X, Y, states):
#         self.X = X
#         self.Y = Y[:, None]
#         self.states = states

#     def __len__(self):
#         return self.X.shape[0]

#     def __getitem__(self, idx):
#         try:
#             return (
#                 float_tensor(self.X[idx, :, :]),
#                 float_tensor(self.Y[idx]),
#                 self.states[idx],
#             )
#         except:
#             pu.db

train_dataset = SeqData(train_symp_seqs, train_y)
val_dataset = SeqData(val_symp_seqs, val_y)
# val_dataset_with_states = SeqDataWithStates(X_val, Y_val, states_val)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True
)
# val_loader_with_states = torch.utils.data.DataLoader(
#     val_dataset_with_states, batch_size=batch_size, shuffle=True
# )
# if start_model != "None":
#     load_model("./hosp_models", file=start_model)
#     print("Loaded model from", start_model)

opt = torch.optim.Adam(
    list(seq_enc.parameters())
    + list(feat_enc.parameters())
    + list(fnp_enc.parameters()),
    lr=lr,
)


def train_step(data_loader, X, Y, X_ref):
    """
    Train step
    """
    feat_enc.train()
    seq_enc.train()
    fnp_enc.train()
    total_loss = 0.0
    train_err = 0.0
    YP = []
    T_target = []
    for i, (x, y) in enumerate(data_loader):
        opt.zero_grad()
        x_seq = seq_enc(float_tensor(X_ref))
        x_feat = feat_enc(x)
        # pu.db
        loss, yp, _ = fnp_enc(x_seq, float_tensor(X_ref), x_feat, y)
        yp = yp[X_ref.shape[0] :]
        loss.backward()
        opt.step()
        YP.append(yp.detach().cpu().numpy())
        T_target.append(y.detach().cpu().numpy())
        total_loss += loss.detach().cpu().numpy()
        train_err += torch.pow(yp - y, 2).mean().sqrt().detach().cpu().numpy()
    return (
        total_loss / (i + 1),
        train_err / (i + 1),
        np.array(YP).ravel(),
        np.array(T_target).ravel(),
    )


def val_step(data_loader, X, Y, X_ref, sample=True):
    """
    Validation step
    """
    with torch.set_grad_enabled(False):
        feat_enc.eval()
        seq_enc.eval()
        fnp_enc.eval()
        val_err = 0.0
        YP = []
        T_target = []
        all_vars = []
        for i, (x, y) in enumerate(data_loader):
            x_seq = seq_enc(float_tensor(X_ref))
            x_feat = feat_enc(x)
            yp, _, vars, _, _, _, _ = fnp_enc.predict(
                x_feat, x_seq, float_tensor(X_ref), sample
            )
            val_err += torch.pow(yp - y, 2).mean().sqrt().detach().cpu().numpy()
            YP.append(yp.detach().cpu().numpy())
            T_target.append(y.detach().cpu().numpy())
            all_vars.extend(vars.detach().cpu().numpy().tolist())
        YP = YP[0][:,0].tolist()
        T_target = T_target[0][:,0].tolist()
        all_vars = [x[0] for x in all_vars]
        return val_err / (i + 1), np.array(YP).ravel(), np.array(T_target).ravel(), all_vars


# def val_step_with_states(data_loader, X, Y, X_ref, sample=True):
#     """
#     Validation step
#     """
#     with torch.set_grad_enabled(False):
#         feat_enc.eval()
#         seq_enc.eval()
#         fnp_enc.eval()
#         val_err = 0.0
#         YP = []
#         T_target = []
#         states_here = []
#         all_vars = []
#         for i, (x, y, st) in enumerate(data_loader):
#             x_seq = seq_enc(float_tensor(X_ref).unsqueeze(2))
#             x_feat = feat_enc(x)
#             yp, _, vars, _, _, _, _ = fnp_enc.predict(
#                 x_feat, x_seq, float_tensor(X_ref), sample
#             )
#             val_err += torch.pow(yp - y, 2).mean().sqrt().detach().cpu().numpy()
#             YP.extend(yp.detach().cpu().numpy().tolist())
#             T_target.extend(y.detach().cpu().numpy().tolist())
#             all_vars.extend(vars.detach().cpu().numpy().tolist())
#             states_here.extend(st)
#         YP = [x[0] for x in YP]
#         T_target = [x[0] for x in T_target]
#         all_vars = [x[0] for x in all_vars]
#         return val_err / (i + 1), np.array(YP, dtype=object).ravel(), np.array(T_target).ravel(), states_here, all_vars


def test_step(X, X_ref, samples=1000):
    """
    Test step
    """
    with torch.set_grad_enabled(False):
        feat_enc.eval()
        seq_enc.eval()
        fnp_enc.eval()
        YP = []
        for i in range(samples):
            x_seq = seq_enc(float_tensor(X_ref).unsqueeze(2))
            x_feat = feat_enc(float_tensor(X))
            yp, _, vars, _, _, _, _ = fnp_enc.predict(
                x_feat, x_seq, float_tensor(X_ref), sample=False
            )
            YP.append(yp.detach().cpu().numpy())
        return np.array(YP)


min_val_err = np.inf
min_val_epoch = 0
all_results = {}
for ep in tqdm(range(epochs)):
    print(f"Epoch {ep+1}")
    train_loss, train_err, yp, yt = train_step(train_loader, None, None, seq_references)
    print(f"Train loss: {train_loss:.4f}, Train err: {train_err:.4f}")
    val_err, yp, yt, vars = val_step(val_loader, None, None, seq_references)
    print(f"Val err: {val_err:.4f}")
    all_results[ep] = {"pred": yp, "gt": yt, "vars": vars}

    if options.tb:
        writer.add_scalar('Train/RMSE', train_err, ep)
        writer.add_scalar('Train/loss', train_loss, ep)
        writer.add_scalar('Val/RMSE', val_err, ep)
    if val_err < min_val_err:
        min_val_err = val_err
        min_val_epoch = ep
        # save_model("./hosp_models")
        # print("Saved model")
    print()
    print()
    if ep > 100 and ep - min_val_epoch > patience:
        break

os.makedirs(f"./val_predictions_power", exist_ok=True)
with open(f"./val_predictions_power/power_"+str(options.splits)+"_splits_weekahead_"+str(options.week_ahead)+".pkl", "wb") as f:
    pickle.dump(all_results, f)

# Now we get results
# load_model("./hosp_models")
# X_test = raw_data
# Y_test = test_step(X_test, X_ref, samples=2000).squeeze()

# Y_test_unnorm = scaler.inverse_transform_idx(Y_test, label_idx)

# # Save predictions
# os.makedirs(f"./hosp_stable_predictions", exist_ok=True)
# with open(f"./hosp_stable_predictions/{save_model_name}_predictions.pkl", "wb") as f:
#     pickle.dump([Y_test_unnorm, all_labels], f)
