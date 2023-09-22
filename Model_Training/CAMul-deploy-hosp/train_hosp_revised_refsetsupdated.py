from curses import raw
import sys
import numpy as np
import pickle
import os
import pudb
from optparse import OptionParser
from torch.utils.tensorboard import SummaryWriter
import itertools
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
    Combine
)
from copy import deepcopy
from models.fnpmodels import RegressionFNP2
from tqdm import tqdm
from itertools import product
from transformers import BertModel, BertConfig, BertTokenizer, pipeline, get_scheduler, BertForSequenceClassification, InformerConfig, InformerModel
from scipy.signal import argrelextrema
from scipy.fft import fft
# sys.path.append("~/FEDformer")
parser = OptionParser()
parser.add_option("-p", "--epiweek_pres", dest="epiweek_pres", default="202310", type="string")
parser.add_option("-e", "--epiweek", dest="epiweek", default="202132", type="string")
parser.add_option("--epochs", dest="epochs", default=3500, type="int")
parser.add_option("--lr", dest="lr", default=6e-5, type="float")
parser.add_option("--patience", dest="patience", default=1000, type="int")
parser.add_option("-d", "--day", dest="day_ahead", default=1, type="int")
parser.add_option("-s", "--seed", dest="seed", default=0, type="int")
parser.add_option("-b", "--batch", dest="batch_size", default=256, type="int")
parser.add_option("-m", "--save", dest="save_model", default="default", type="string")
parser.add_option("--start_model", dest="start_model", default="None", type="string")
parser.add_option("-c", "--cuda", dest="cuda", default=True, action="store_true")
parser.add_option("--start", dest="start_day", default=-120, type="int")
parser.add_option("-t", "--tb", action="store_true", dest="tb", default=False)
parser.add_option("-W", "--use-sliding-window", dest="sliding_window", default=False, action="store_true")
parser.add_option("--auto-size-best-num", dest="auto_size_best_num", default=None, type="int")
parser.add_option("--sliding-window-size", dest="window_size", type="int", default=17)
parser.add_option("--sliding-window-stride", dest="window_stride", type="int", default=15)
parser.add_option("--disease", dest="disease", type="string", default="covid")
parser.add_option("--preprocess", dest="preprocess", action="store_true", default=False)
parser.add_option("--cnn", dest="cnn", action="store_true", default=False)
parser.add_option("--rag", dest="rag", action="store_true", default=False)
parser.add_option("--nn", dest="nn", default="none", type="choice", choices=["none", "simple", "bn", "dot", "bert"])
parser.add_option("--bert-emb", dest="bert_emb", action="store_true", default=False)
parser.add_option("--fed", dest="fed", action="store_true", default=False)
parser.add_option("--smart-mode", dest="smart_mode", default=0, type="int")
parser.add_option("--optionals", dest="optionals", default=" ", type="str")

(options, args) = parser.parse_args()
epiweek_pres = options.epiweek_pres
epiweek = options.epiweek
day_ahead = options.day_ahead
seed = options.seed
if options.optionals != " ":
    save_model_name = options.save_model+"_optionals_"+options.optionals
else:
    save_model_name = options.save_model
start_model = options.start_model
cuda = options.cuda
start_day = options.start_day
batch_size = options.batch_size
lr = options.lr
epochs = options.epochs
patience = options.patience
disease = options.disease
fed = options.fed

if options.rag and options.cnn:
    print("Cannot have cnn and rag together")
    sys.exit(0)
    
if options.cnn:
    disease = disease + "_cnn"
if options.rag:
    disease = disease + "_rag"
if options.nn != "none":
    disease = disease + "_nn-" + options.nn
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

# Get dataset as numpy arrays

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

states_to_consider = [
    "DC", "MA", "FL", "GA", "IL", "NY", "NJ", "PA", "TX", "WA", "CA", "X"
]

states_to_consider_indices = [states.index(x) for x in states_to_consider]
raw_data = []
for st in states:
    with open(f"./data/hosp_data/saves/hosp_{st}_{epiweek_pres}.pkl", "rb") as fl:
        raw_data.append(pickle.load(fl))
def diff_epiweeks(epiweek1, epiweek2):
    """
    Compute difference in epiweeks
    """
    year1, week1 = int(epiweek1[:4]), int(epiweek1[4:])
    year2, week2 = int(epiweek2[:4]), int(epiweek2[4:])
    return (year1 - year2) * 52 + week1 - week2
# pu.db
raw_data_weeks = list(range(-raw_data[0].shape[0], 0))
raw_data_weeks = [raw_data_weeks for x in range(51)]
if diff_epiweeks(epiweek, epiweek_pres) > 0:
    raw_data = np.array(raw_data)[:, :-5 + day_ahead, :]
    raw_data_weeks = np.array(raw_data_weeks)[:, :-5 + day_ahead]
else:    
    raw_data = np.array(raw_data)[:, :diff_epiweeks(epiweek, epiweek_pres) + day_ahead, :]  # states x days x featureslabel_idx = include_cols.index("cdc_hospitalized")
    raw_data_weeks = np.array(raw_data_weeks)[:, :diff_epiweeks(epiweek, epiweek_pres) + day_ahead]

if options.disease == "flu":
    label_idx = include_cols.index("cdc_flu_hosp")
else:
    label_idx = include_cols.index("cdc_hospitalized")


if options.smart_mode == 7:
    conv_here_1 = torch.nn.Conv1d(1,1,kernel_size=3,padding="valid").to(device)
    conv_here_1.weight.data = (torch.ones_like(conv_here_1.weight.data)/3).to(device)
    conv_here_1.bias.data = torch.zeros(1).to(device)
    conv_here_2 = torch.nn.Conv1d(1,1,kernel_size=5,padding="valid").to(device)
    conv_here_2.weight.data = (torch.ones_like(conv_here_2.weight.data)/5).to(device)
    conv_here_2.bias.data = torch.zeros(1).to(device)

if options.smart_mode >= 2 or "combine" in options.optionals or "fft" in options.optionals:
    raw_data_unavgd = deepcopy(raw_data)
    kernel_size = 3
    avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
    def moving_avg(x, kernel_size=kernel_size):
        # padding on the both ends of time series
        front = x[0:1].repeat((kernel_size - 1) // 2)
        end = x[-1:].repeat((kernel_size - 1) // 2)
        x = np.concatenate([front, x, end]).tolist()
        with torch.no_grad():
            # if options.smart_mode == 7:
            #     x = conv_here_1(torch.tensor([[x]]))
            # else:
                x = avg(torch.tensor([[x]]))
        x = x.detach().numpy().tolist()
        return np.array(x[0][0])
    for i, rw in enumerate(raw_data):
        rw_here = np.zeros_like(rw)
        for j in range(len(include_cols)):
            rw_here[:,j] = moving_avg(rw[:,j])
        raw_data[i] = rw_here
    raw_data_avg3 = deepcopy(raw_data)
    kernel_size = 5
    avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
    def moving_avg(x, kernel_size=kernel_size):
        # padding on the both ends of time series
        front = x[0:1].repeat((kernel_size - 1) // 2)
        end = x[-1:].repeat((kernel_size - 1) // 2)
        x = np.concatenate([front, x, end]).tolist()
        with torch.no_grad():
            # if options.smart_mode == 7:
            #     x = conv_here_2(torch.tensor([[x]]))
            # else:
                x = avg(torch.tensor([[x]]))
        x = x.detach().numpy().tolist()
        return np.array(x[0][0])
    for i, rw in enumerate(raw_data):
        rw_here = np.zeros_like(rw)
        for j in range(len(include_cols)):
            rw_here[:,j] = moving_avg(rw[:,j])
        raw_data[i] = rw_here
    raw_data_avg35 = deepcopy(raw_data)
if options.smart_mode >= 3:
    all_labels = raw_data_unavgd[:, -1, label_idx]
else:
    all_labels = raw_data[:, -1, label_idx]
print(f"Diff epiweeks: {diff_epiweeks(epiweek, epiweek_pres)}")
# pu.db
raw_data = raw_data[:, start_day:-day_ahead, :]
raw_data_weeks = raw_data_weeks[:, start_day:-day_ahead]
if options.smart_mode >= 3 and options.smart_mode != 7:
    raw_data_unavgd = raw_data_unavgd[:, start_day:-day_ahead, :]

raw_data_unnorm = raw_data.copy()

if options.tb:
    writer = SummaryWriter("runs/"+disease+"/"+options.save_model)
    # if options.sliding_window:
    #     if options.preprocess:
    #     else:
    #         writer = SummaryWriter("runs/"+disease+"/"+disease+"_slidingwindow_epiweek"+str(epiweek_pres)+"_weekahead_"+str(options.day_ahead)+"_windowsize_"+str(options.window_size)+"_stride_"+str(options.window_stride))
    # else:
    #     writer = SummaryWriter("runs/"+disease+"/"+disease+"_normal_epiweek"+str(epiweek_pres)+"_weekahead_"+str(options.day_ahead))


class ScalerFeatTorch:
    def __init__(self, raw_data):
        self.means = torch.mean(raw_data, axis=1)
        self.vars = torch.std(raw_data, axis=1) + 1e-8

    def transform(self, data):
        return (data - torch.permute(self.means[:, :, None], (0, 2, 1))) / torch.permute(
            self.vars[:, :, None], (0, 2, 1)
        )

    def inverse_transform(self, data):
        return data * torch.permute(self.vars[:, :, None], (0, 2, 1)) + torch.permute(
            self.means[:, :, None], (0, 2, 1)
        )

    def transform_idx(self, data, idx=label_idx):
        return (data - self.means[:, idx]) / self.vars[:, idx]

    def inverse_transform_idx(self, data, idx=label_idx):
        return data * self.vars[:, idx] + self.means[:, idx]
    
    def inverse_transform_idx_selected_states(self, data, idx=label_idx, state_indices=states_to_consider_indices):
        return data * self.vars[:, idx][states_to_consider_indices] + self.means[:, idx][states_to_consider_indices]



class ScalerFeat:
    def __init__(self, raw_data):
        self.means = np.mean(raw_data, axis=1)
        self.vars = np.std(raw_data, axis=1) + 1e-8

    def transform(self, data):
        return (data - np.transpose(self.means[:, :, None], (0, 2, 1))) / np.transpose(
            self.vars[:, :, None], (0, 2, 1)
        )

    def inverse_transform(self, data):
        return data * np.transpose(self.vars[:, :, None], (0, 2, 1)) + np.transpose(
            self.means[:, :, None], (0, 2, 1)
        )

    def transform_idx(self, data, idx=label_idx):
        return (data - self.means[:, idx]) / self.vars[:, idx]

    def inverse_transform_idx(self, data, idx=label_idx):
        return data * self.vars[:, idx] + self.means[:, idx]
    
    def inverse_transform_idx_selected_states(self, data, idx=label_idx, state_indices=states_to_consider_indices):
        return data * self.vars[:, idx][states_to_consider_indices] + self.means[:, idx][states_to_consider_indices]
# pu.db
if options.smart_mode == 5 or options.smart_mode == 8:
    scaler = ScalerFeat(raw_data_unavgd)
elif  options.smart_mode == 7:
    scaler = ScalerFeat(raw_data_unavgd[:, start_day:-day_ahead, :])
    scalertorch = ScalerFeatTorch(float_tensor(raw_data_unavgd[:, start_day:-day_ahead, :]))
else:
    scaler = ScalerFeat(raw_data)
raw_data = scaler.transform(raw_data)
if options.smart_mode == 3 or options.smart_mode == 4 or options.smart_mode == 5 or options.smart_mode == 8:
    raw_data_unavgd = scaler.transform(raw_data_unavgd)

# Chunk dataset sequences
def prefix_sequences_torch(seq_avg, seq_unavg, day_ahead=day_ahead):
    """
    Prefix sequences with zeros
    """
    l = len(seq_avg)
    # try:
    X, Y = torch.zeros((l - day_ahead, l, seq_avg.shape[-1])).to(device), torch.zeros(l - day_ahead).to(device)
    # except:
    #     pu.db
    for i in range(l - day_ahead):
        X[i, (l - i - 1) :, :] = seq_avg[: i + 1, :]
        Y[i] = seq_unavg[i + day_ahead, label_idx]
    return X, Y

def prefix_sequences(seq, day_ahead=day_ahead):
    """
    Prefix sequences with zeros
    """
    l = len(seq)
    # try:
    X, Y = np.zeros((l - day_ahead, l, seq.shape[-1])), np.zeros(l - day_ahead)
    # except:
    #     pu.db
    for i in range(l - day_ahead):
        X[i, (l - i - 1) :, :] = seq[: i + 1, :]
        Y[i] = seq[i + day_ahead, label_idx]
    return X, Y

def prefix_sequences_sm3(seq_avg, seq_unavg, day_ahead=day_ahead):
    """
    Prefix sequences with zeros
    """
    l = len(seq_avg)
    # try:
    X, X_smart, Y = np.zeros((l - day_ahead, l, seq_avg.shape[-1])), np.zeros((l - day_ahead, l, seq_avg.shape[-1])), np.zeros(l - day_ahead)
    # except:
    #     pu.db
    for i in range(l - day_ahead):
        X[i, (l - i - 1) :, :] = seq_unavg[: i + 1, :]
        X_smart[i, (l - i - 1) :, :] = seq_avg[: i + 1, :]
        Y[i] = seq_unavg[i + day_ahead, label_idx]
    return X, X_smart, Y

def prefix_sequences_weeks(seq, day_ahead=day_ahead):
    """
    Prefix sequences with zeros
    """
    l = len(seq)
    # try:
    X = np.zeros((l - day_ahead, l))
    # except:
    #     pu.db
    for i in range(l - day_ahead):
        X[i, (l - i - 1) :] = seq[: i + 1]
    return X

X, X_smart, Y = [], [], []
X_weeks = []
for i, st in enumerate(states):
    if st in states_to_consider:
        if options.smart_mode == 3 or options.smart_mode == 4 or options.smart_mode == 5 or options.smart_mode == 8  or "combine" in options.optionals or "fft" in options.optionals:
            x, x_smart, y = prefix_sequences_sm3(raw_data[i], raw_data_unavgd[i])
            X_smart.append(x_smart)
        else:
            if options.smart_mode == 6 or options.smart_mode == 7:
                x, y = prefix_sequences(raw_data_unavgd[:, start_day:-day_ahead, :][i])
            else:
                x, y = prefix_sequences(raw_data[i])
        x_weeks = prefix_sequences_weeks(raw_data_weeks[i])
        X.append(x)
        X_weeks.append(x_weeks)
        Y.append(y)

num_repeat = len(X[0])
states_unflattened = [list(itertools.repeat(st, num_repeat)) for st in states_to_consider]
# sts = []
# for st_here in states_unflattened:
#     sts.extend(st_here)

# Randomizing before statewise merging so as to make sure no bias in the ordering of train and val generated data
# np.random.seed(seed)
# torch.manual_seed(seed)
# random.seed(seed)
for i in range(len(X)):
    perm = np.random.permutation(len(X[i]))
    X[i] = X[i][perm]
    if len(X_smart) > 0:
        X_smart[i] = X_smart[i][perm]
    X_weeks[i] = X_weeks[i][perm]
    Y[i] = Y[i][perm]
    # states_unflattened[i] = np.array(states_unflattened[i])[perm].tolist()


# Divide val and train and test
frac_val = 0.7
frac_test = 0.9
if len(X_smart) > 0:
    X_train, X_train_smart, X_train_weeks, Y_train = np.concatenate([x[:int(len(X[0]) * frac_val)] for x in X]), np.concatenate([x[:int(len(X[0]) * frac_val)] for x in X_smart]), np.concatenate([x_weeks[:int(len(X_weeks[0]) * frac_val)] for x_weeks in X_weeks]), np.concatenate([y[:int(len(X[0]) * frac_val)] for y in Y])
else:
    X_train, X_train_weeks, Y_train = np.concatenate([x[:int(len(X[0]) * frac_val)] for x in X]), np.concatenate([x_weeks[:int(len(X_weeks[0]) * frac_val)] for x_weeks in X_weeks]), np.concatenate([y[:int(len(X[0]) * frac_val)] for y in Y])
states_train = []
for st_here in [x[:int(len(X[0]) * frac_val)] for x in states_unflattened]:
    states_train.extend(st_here)

if len(X_smart) > 0:
    X_val, X_val_smart, X_val_weeks, Y_val = np.concatenate([x[int(len(X[0]) * frac_val):int(len(X[0]) * frac_test)] for x in X]), np.concatenate([x[int(len(X[0]) * frac_val):int(len(X[0]) * frac_test)] for x in X_smart]), np.concatenate([x_weeks[int(len(X_weeks[0]) * frac_val):int(len(X_weeks[0]) * frac_test)] for x_weeks in X_weeks]), np.concatenate([y[int(len(X[0]) * frac_val):int(len(X[0]) * frac_test)] for y in Y])
else:
    X_val, X_val_weeks, Y_val = np.concatenate([x[int(len(X[0]) * frac_val):int(len(X[0]) * frac_test)] for x in X]), np.concatenate([x_weeks[int(len(X_weeks[0]) * frac_val):int(len(X_weeks[0]) * frac_test)] for x_weeks in X_weeks]), np.concatenate([y[int(len(X[0]) * frac_val):int(len(X[0]) * frac_test)] for y in Y])
states_val = []
for st_here in [x[int(len(X[0]) * frac_val):int(len(X[0]) * frac_test)] for x in states_unflattened]:
    states_val.extend(st_here)

X_test, X_test_weeks, Y_test = np.concatenate([x[int(len(X[0]) * frac_test):] for x in X]), np.concatenate([x_weeks[int(len(X_weeks[0]) * frac_test):] for x_weeks in X_weeks]), np.concatenate([y[int(len(X[0]) * frac_test):] for y in Y])
states_test = []
for st_here in [x[int(len(X[0]) * frac_test):] for x in states_unflattened]:
    states_test.extend(st_here)

# np.random.seed(seed)
# torch.manual_seed(seed)
# random.seed(seed)
# Shuffle data
perm = np.random.permutation(len(X_train))
if len(X_smart) > 0:
    X_train, X_train_smart, X_train_weeks, Y_train, states_train = X_train[perm], X_train_smart[perm], X_train_weeks[perm], Y_train[perm], np.array(states_train)[perm].tolist()
else:
    X_train, X_train_weeks, Y_train, states_train = X_train[perm], X_train_weeks[perm], Y_train[perm], np.array(states_train)[perm].tolist()

perm = np.random.permutation(len(X_val))
if len(X_smart) > 0:
    X_val, X_val_smart, X_val_weeks, Y_val, states_val = X_val[perm], X_val_smart[perm], X_val_weeks[perm], Y_val[perm], np.array(states_val)[perm].tolist()
else:
    X_val, X_val_weeks, Y_val, states_val = X_val[perm], X_val_weeks[perm], Y_val[perm], np.array(states_val)[perm].tolist()

perm = np.random.permutation(len(X_test))
X_test, X_test_weeks, Y_test, states_test = X_test[perm], X_test_weeks[perm], Y_test[perm], np.array(states_test)[perm].tolist()

# Reference sequences
if options.smart_mode == 5 or options.smart_mode == 8:
    X_ref = raw_data_unavgd[:, :, label_idx]
elif options.smart_mode == 7:
    X_ref = scaler.transform(raw_data_unavgd[:, start_day:-day_ahead, :])[:, :, label_idx]
else:
    X_ref = raw_data[:, :, label_idx]
# pu.db
# Divide val and train
# frac = 0.1
# X_val, Y_val, states_val = X_train[: int(len(X_train) * frac)], Y_train[: int(len(X_train) * frac)], states_train[: int(len(X_train) * frac)]
# X_train, Y_train, states_train = (
#     X_train[int(len(X_train) * frac) :],
#     Y_train[int(len(X_train) * frac) :],
#     states_train[int(len(X_train) * frac) :],
# )

def batched_compute_pcc(x, y):
    """ R computation
    :param  list  x: 1st list of random variables
    :param  list  y: 2nd list of random variables
    :return float r: correlation coefficient of X and Y
    """
    x = x.repeat(1,(y.shape[1]//x.shape[1])+1)[:,:y.shape[1]]
    mean_x, mean_y  = torch.mean(x, axis=-1), torch.mean(y, axis=-1)
    mean_x = mean_x.unsqueeze(1).repeat([1, x.shape[1]])
    mean_y = mean_y.unsqueeze(1).repeat([1, y.shape[1]])
    # mean_x, mean_y  = sum(x) / len(x), sum(y) / len(y)
    cov = torch.sum(torch.multiply(x - mean_x, y - mean_y), axis=-1)
    # cov   = sum([(a - mean_x) * (b - mean_y) for a, b in zip(x, y)])
    var_x = torch.sum(torch.pow(x - mean_x, 2), axis=-1)
    var_y = torch.sum(torch.pow(y - mean_y, 2), axis=-1)
    # var_x = sum([(a - mean_x) ** 2 for a in x])
    # var_y = sum([(b - mean_y) ** 2 for b in y])
    return (cov / (torch.sqrt(var_x)+1e-6)) / (torch.sqrt(var_y)+1e-6)


if options.smart_mode == 1 or options.smart_mode == 4 or options.smart_mode == 8:
    kernel_size = 3
    avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
    def moving_avg(x, kernel_size=kernel_size):
        # padding on the both ends of time series
        front = x[0:1].repeat((kernel_size - 1) // 2)
        end = x[-1:].repeat((kernel_size - 1) // 2)
        x = np.concatenate([front, x, end]).tolist()
        with torch.no_grad():
            x = avg(torch.tensor([[x]]))
        x = x.detach().numpy().tolist()
        return np.array(x[0][0])
    X_ref_average = np.zeros_like(X_ref)


    for i in range(51):
        X_ref_average[i] = moving_avg(X_ref[i], kernel_size=kernel_size)

    # if options.smart_mode == 2:
    #     pu.db
    # for i in tqdm(range(51)):
    #     state = states[i]
    #     plt.plot(X_ref_average[i, :])
    #     plt.savefig("flu ref "+str(state)+" avg "+str(kernel_size)+".png")
    #     plt.clf()

    # for i in tqdm(range(51)):
    #     state = states[i]
    #     plt.plot(X_ref[i, :])
    #     plt.savefig("flu ref here "+str(state)+".png")
    #     plt.clf()

    kernel_size_2 = 3
    avg = torch.nn.AvgPool1d(kernel_size=kernel_size_2, stride=1, padding=0)
    def moving_avg(x, kernel_size=kernel_size_2):
        # padding on the both ends of time series
        front = x[0:1].repeat((kernel_size - 1) // 2)
        end = x[-1:].repeat((kernel_size - 1) // 2)
        x = np.concatenate([front, x, end]).tolist()
        with torch.no_grad():
            x = avg(torch.tensor([[x]]))
        x = x.detach().numpy().tolist()
        return np.array(x[0][0])
    X_ref_average_2 = np.zeros_like(X_ref)

    for i in range(51):
        X_ref_average_2[i] = moving_avg(X_ref_average[i], kernel_size=kernel_size_2)

    # for i in tqdm(range(51)):
    #     state = states[i]
    #     plt.plot(X_ref_average_2[i, :])
    #     plt.savefig("flu ref "+str(state)+" avg "+str(kernel_size)+"_"+str(kernel_size_2)+".png")
    #     plt.clf()

# if options.smart_mode == 4:
#     X_ref_average_2 = X_ref

if options.sliding_window or options.smart_mode == 8 or options.smart_mode == 1 or "part" in options.optionals:
    if options.smart_mode == 1 or options.smart_mode == 4 or options.smart_mode == 8 or "part" in options.optionals:
        X_ref_orig_shape = X_ref.shape
        to_concat = []
        to_concat_weeks = []
        for i in range(len(states)):
            # pu.db
            # if options.smart_mode == 8:
            #     series_here = raw_data_unavgd[i]
            # else:
            series_here = raw_data[i]
            minimas =  argrelextrema(series_here, np.less)[0]
            derivative = np.diff(series_here)
            split_idxs= [np.where(derivative!=0)[0][0]]
            start_scan = split_idxs[0]
            # if states[i] == "ID":
            # pu.db
            for stop_scan in minimas.tolist()+[len(series_here)-1]:
                vals = series_here[start_scan:stop_scan]
                if len(vals) < 1:
                    continue
                max_here = np.max(vals)
                if max_here > 0:
                    split_idxs.append(start_scan)
                    split_idxs.append(stop_scan)
                start_scan = stop_scan
            print(states[i])
            # print("mean "+str(np.mean(series_here)))
            # print("std "+str(np.std(series_here)))
            # print("mean+std "+str(np.mean(series_here) + np.std(series_here)))
            if len(series_here) - 1 not in split_idxs:
                split_idxs.append(len(series_here) - 1)
            split_idxs_diffs = [100]+np.diff(split_idxs).tolist()
            split_idxs = np.array(split_idxs)[np.array(split_idxs_diffs) >= 10].tolist()
            # if states[i] == "AR":
            #     pu.db

            print(split_idxs)
            # if states[i] == "AR":
            #     pu.db
            print("\n")
            # plt.plot(X_ref_average_2[i, :])
            # for si in split_idxs:
            #     plt.axvline(si, color="red")
            # plt.savefig("flu ref "+str(states[i])+" avg "+str(kernel_size)+"_"+str(kernel_size_2)+" split.png")
            # plt.clf()
            # plt.plot(X_ref[i, :])
            # for si in split_idxs:
            #     plt.axvline(si, color="red")
            # plt.savefig("flu ref "+str(states[i])+" avg "+str(kernel_size)+"_"+str(kernel_size_2)+" split orig.png")
            # plt.clf()

            
            ci = 0
            for si in split_idxs:
                to_concat.append(raw_data_unavgd[i, ci:si, label_idx].tolist())
                to_concat_weeks.append(raw_data_weeks[i, ci:si].tolist())
                ci = si
            to_concat.append(raw_data_unavgd[i, :, label_idx].tolist())
            to_concat_weeks.append(raw_data_weeks[i, :].tolist())

        max_len = 0
        to_concat_orig = deepcopy(to_concat)
        for tc in to_concat:
            if len(tc) > max_len:
                max_len = len(tc)
        # pu.db
        for t, tc in enumerate(to_concat):
            to_concat[t] = np.expand_dims(np.array([-100 for x in range(max_len-len(tc))] + tc), axis=0)
            try:
                to_concat_weeks[t] = np.expand_dims(np.array([-100 for x in range(max_len-len(tc))] + to_concat_weeks[t]), axis=0)
            except: pu.db
        try:
            X_ref = np.concatenate(to_concat)
        except: pu.db
        # raw_data_weeks = np.concatenate(to_concat_weeks)




    elif options.auto_size_best_num is not None:
        lags = [x for x in range(4, 30)]
        idx_scores = [0 for x in range(len(lags))]
        for st_idx, st_here in zip(states_to_consider_indices, states_to_consider):
            series_here = X_ref[st_idx][30:-30]
            series_here = series_here - moving_avg(series_here)
            acs = []
            for lag in lags:
                ac_here = []
                for it in range(len(series_here)-1, lag-1, -1):
                    ac_here.append(series_here[it] * series_here[it - lag])
                acs.append(np.mean(np.array(ac_here)))
            sorted_indices = np.flip(np.argsort(acs)).tolist()
            for k, sindxs in enumerate(sorted_indices):
                idx_scores[sindxs] += len(lags) - k

        lags_needed_idxs = np.flip(np.argsort(idx_scores)).tolist()
        ilk = lags[lags_needed_idxs[options.auto_size_best_num]]
        ils = lags[lags_needed_idxs[options.auto_size_best_num]]
    else:
        ilk = options.window_size
        ils = options.window_stride
    # """
    if options.smart_mode == 0:
        to_concat = []
        for w in range(0, X_ref.shape[1] - ilk + 1, ils):
            to_concat.append(X_ref[:,w:w + ilk])
        
        X_ref_orig_shape = X_ref.shape
        X_ref = np.concatenate(to_concat)
    
    if options.preprocess:
        all_idxs = np.zeros((X_ref.shape[0], X_train.shape[0], X_ref.shape[1]+X_train.shape[1]))
        for idx in tqdm(product(range(X_ref.shape[0]), range(X_train.shape[0]))):
            all_idxs[idx[0], idx[1]] = np.concatenate((X_ref[idx[0]], X_train[idx[1]][:,label_idx]), axis = -1)
        all_idxs = np.reshape(all_idxs,(-1, all_idxs.shape[-1]))
        pccs = batched_compute_pcc(torch.tensor(all_idxs[:,:X_ref.shape[1]]), torch.tensor(all_idxs[:,X_ref.shape[1]:]))
        pccs = pccs.numpy().reshape((X_ref.shape[0], X_train.shape[0]))
        summed_pccs = np.sum(pccs, axis=-1)
        best_ref_idxs = np.argsort(summed_pccs)[:X_ref_orig_shape[0]]
        X_ref = X_ref[best_ref_idxs, :]
        # pu.db

# pu.db
# Build model
if options.bert_emb:
    # My method
    # feat_enc_linear = torch.nn.Linear(len(include_cols), 768).to(device)
    # seq_enc_linear = torch.nn.Linear(1, 768).to(device)
    
    configuration = InformerConfig(prediction_length=4, num_time_features=1, input_size=1, context_length=raw_data_weeks.shape[1], lags_sequence=[0])
    seq_enc = InformerModel(configuration).to(device)
    configuration = InformerConfig(prediction_length=3, num_time_features=1, input_size=len(include_cols), context_length=raw_data.shape[1], lags_sequence=[0])
    feat_enc = InformerModel(configuration).to(device)
else:
    # Original method
    feat_enc = GRUEncoder(in_size=len(include_cols), out_dim=60,).to(device)
    seq_enc = GRUEncoder(in_size=1, out_dim=60,).to(device)

if "fft" in options.optionals:
    feat_enc_fft = GRUEncoder(in_size=len(include_cols), out_dim=60,).to(device)
    seq_enc_fft = GRUEncoder(in_size=1, out_dim=60,).to(device)
    fnp_enc = RegressionFNP2(
        dim_x=120,
        dim_y=1,
        dim_h=100,
        size_ref=X_ref.shape[0],
        n_layers=3,
        num_M=batch_size,
        dim_u=60,
        dim_z=60,
        use_DAG=False,
        use_ref_labels=False,
        add_atten=False,
        cnn=options.cnn,
        rag=options.rag,
        nn_A=options.nn
    ).to(device)

# np.random.seed(seed)
# torch.manual_seed(seed)
# random.seed(seed)
elif options.smart_mode == 8:
    fnp_enc = RegressionFNP2(
        dim_x=60,
        dim_y=1,
        dim_h=100,
        size_ref=X_ref.shape[0],
        n_layers=3,
        num_M=batch_size,
        dim_u=60,
        dim_z=60,
        use_DAG=False,
        use_ref_labels=False,
        add_atten=False,
        cnn=options.cnn,
        rag=options.rag,
        nn_A="bn"
    ).to(device)

elif options.bert_emb:
    fnp_enc = RegressionFNP2(
        dim_x=64,
        dim_y=1,
        dim_h=100,
        size_ref=X_ref.shape[0],
        n_layers=3,
        num_M=batch_size,
        dim_u=64,
        dim_z=64,
        use_DAG=False,
        use_ref_labels=False,
        add_atten=False,
        cnn=options.cnn,
        rag=options.rag,
        nn_A=options.nn
    ).to(device)
else:
    fnp_enc = RegressionFNP2(
        dim_x=60,
        dim_y=1,
        dim_h=100,
        size_ref=X_ref.shape[0],
        n_layers=3,
        num_M=batch_size,
        dim_u=60,
        dim_z=60,
        use_DAG=False,
        use_ref_labels=False,
        add_atten=False,
        cnn=options.cnn,
        rag=options.rag,
        nn_A=options.nn
    ).to(device)


def load_model(folder, file=save_model_name):
    """
    Load model
    """
    full_path = os.path.join(folder, file)
    if not os.path.exists(full_path):
        full_path = "/localscratch/"+full_path[14:]
    assert os.path.exists(full_path)
    feat_enc.load_state_dict(torch.load(os.path.join(full_path, "feat_enc.pt")))
    seq_enc.load_state_dict(torch.load(os.path.join(full_path, "seq_enc.pt")))
    fnp_enc.load_state_dict(torch.load(os.path.join(full_path, "fnp_enc.pt")))


def save_model(folder, file=save_model_name):
    """
    Save model
    """
    full_path = os.path.join(folder, file)
    os.makedirs(full_path, exist_ok=True)
    torch.save(feat_enc.state_dict(), os.path.join(full_path, "feat_enc.pt"))
    torch.save(seq_enc.state_dict(), os.path.join(full_path, "seq_enc.pt"))
    torch.save(fnp_enc.state_dict(), os.path.join(full_path, "fnp_enc.pt"))

# avg3 = torch.nn.AvgPool1d(kernel_size=3, stride=1, padding=0)
def moving_wavg_3(x, kernel_size=3):
    # padding on the both ends of time series
    front = x[0:1].repeat((kernel_size - 1) // 2)
    end = x[-1:].repeat((kernel_size - 1) // 2)
    x = np.concatenate([front, x, end]).tolist()
    # with torch.no_grad():
    x = conv_here_1(float_tensor([[x]]))
    # x = x.detach().numpy().tolist()
    return x[0][0]

# avg5 = torch.nn.AvgPool1d(kernel_size=5, stride=1, padding=0)
def moving_wavg_5(x, kernel_size=5):
    # padding on the both ends of time series
    front = x[0:1].repeat((kernel_size - 1) // 2)
    end = x[-1:].repeat((kernel_size - 1) // 2)
    x = np.concatenate([front, x, end]).tolist()
    # with torch.no_grad():
    x = conv_here_2(float_tensor([[x]]))
    # x = x.detach().numpy().tolist()
    return x[0][0]

def moving_wavg_35(x, kernel_size_2=5, kernel_size_1=3):
    # padding on the both ends of time series
    front = x[0:1].repeat((kernel_size_1 + kernel_size_2 - 2) // 2)
    end = x[-1:].repeat((kernel_size_1 + kernel_size_2 - 2) // 2)
    x = np.concatenate([front, x, end]).tolist()
    # with torch.no_grad():
    x = conv_here_2(conv_here_1(float_tensor([[x]])))
    # x = x.detach().numpy().tolist()
    return x[0][0]

# hidden_size_combine = 238
# if day_ahead == 2:
#     hidden_size_combine = 236
# if day_ahead == 3:
#     hidden_size_combine = 234

if "combine" in options.optionals:
    combine = Combine(len(include_cols)).to(device)
if "fft" in options.optionals:
    combine_fft = Combine(len(include_cols)).to(device)


# Build dataset
class SeqDataWithConv(torch.utils.data.Dataset):
    def __init__(self, raw_data_here, tv):
        self.raw_data_here = raw_data_here
        self.tv = tv

    def __len__(self):
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # random.seed(seed)
        raw_data_conved_here = float_tensor(np.zeros_like(self.raw_data_here))
        for i, rw in enumerate(self.raw_data_here):
            rw_here = torch.zeros_like(torch.tensor(rw)).to(device)
            for j in range(len(include_cols)):
                rw_here[:,j] = moving_wavg_35(rw[:,j])
            raw_data_conved_here[i] = rw_here
        # for i, rw in enumerate(raw_data_conved_here):
        #     rw_here = torch.zeros_like(rw).to(device)
        #     for j in range(len(include_cols)):
        #         rw_here[:,j] = moving_wavg_5(rw[:,j])
        #     raw_data_conved_here[i] = rw_here
        # pu.db
        raw_data_conved_here = scalertorch.transform(raw_data_conved_here[:, start_day:-day_ahead, :]) 
        raw_data_unconved_here = scalertorch.transform(float_tensor(self.raw_data_here[:, start_day:-day_ahead, :]))
        # raw_data_unconved_here = scaler.transform(self.raw_data_here[:, start_day:-day_ahead, :])
        # raw_data_conved_here = scaler.transform(raw_data_conved_here[:, start_day:-day_ahead, :])
        X_conved, Y_conved = [], []
        X_ref_conved = raw_data_conved_here[:, :, label_idx]
        # raw_data_conved_here = scaler_conved.transform(raw_data_conved_here)
        # X_weeks = []
        for i, st in enumerate(states):
            if st in states_to_consider:
                x, y = prefix_sequences_torch(raw_data_conved_here[i], raw_data_unconved_here[i])
                X_conved.append(x)
                Y_conved.append(y)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # random.seed(seed)
        for i in range(len(X_conved)):
            perm = np.random.permutation(len(X_conved[i]))
            X_conved[i] = X_conved[i][perm]
            Y_conved[i] = Y_conved[i][perm]
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # random.seed(seed)
        # pu.db
        X_train_conved, Y_train_conved = torch.cat([x[:int(len(X_conved[0]) * frac_val)] for x in X_conved]),  torch.cat([y[:int(len(X_conved[0]) * frac_val)] for y in Y_conved])
        perm = np.random.permutation(len(X_train_conved))
        X_train_conved, Y_train_conved = X_train_conved[perm], Y_train_conved[perm].unsqueeze(axis=-1)
        X_val_conved, Y_val_conved = torch.cat([x[int(len(X_conved[0]) * frac_val):int(len(X_conved[0]) * frac_test)] for x in X_conved]),  torch.cat([y[int(len(X_conved[0]) * frac_val):int(len(X_conved[0]) * frac_test)] for y in Y_conved])
        perm = np.random.permutation(len(X_val_conved))
        X_val_conved, Y_val_conved = X_val_conved[perm], Y_val_conved[perm].unsqueeze(axis=-1)
        if self.tv=="train":
            return X_train_conved.shape[0]
        elif self.tv=="val":
            return X_val_conved.shape[0]

    def __getitem__(self, idx):
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # random.seed(seed)
        raw_data_conved_here = float_tensor(np.zeros_like(self.raw_data_here))
        for i, rw in enumerate(self.raw_data_here):
            rw_here = torch.zeros_like(torch.tensor(rw)).to(device)
            for j in range(len(include_cols)):
                rw_here[:,j] = moving_wavg_35(rw[:,j])
            raw_data_conved_here[i] = rw_here
        # for i, rw in enumerate(raw_data_conved_here):
        #     rw_here = torch.zeros_like(rw).to(device)
        #     for j in range(len(include_cols)):
        #         rw_here[:,j] = moving_wavg_5(rw[:,j])
        #     raw_data_conved_here[i] = rw_here
        # pu.db
        raw_data_conved_here = scalertorch.transform(raw_data_conved_here[:, start_day:-day_ahead, :]) 
        raw_data_unconved_here = scalertorch.transform(float_tensor(self.raw_data_here[:, start_day:-day_ahead, :]))
        X_conved, Y_conved = [], []
        X_ref_conved = raw_data_conved_here[:, :, label_idx]
        # raw_data_conved_here = scaler_conved.transform(raw_data_conved_here)
        X_weeks = []
        for i, st in enumerate(states):
            if st in states_to_consider:
                x, y = prefix_sequences_torch(raw_data_conved_here[i], raw_data_unconved_here[i])
                X_conved.append(x)
                Y_conved.append(y)
                x_weeks = prefix_sequences_weeks(raw_data_weeks[i])
                X_weeks.append(x_weeks)
        
        num_repeat = len(X_conved[0])
        states_unflattened = [list(itertools.repeat(st, num_repeat)) for st in states_to_consider]
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # random.seed(seed)
        for i in range(len(X_conved)):
            perm = np.random.permutation(len(X_conved[i]))
            X_conved[i] = X_conved[i][perm]
            Y_conved[i] = Y_conved[i][perm]
            X_weeks[i] = X_weeks[i][perm]
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # random.seed(seed)
        # pu.db
        X_train_conved, X_train_weeks, Y_train_conved = torch.cat([x[:int(len(X_conved[0]) * frac_val)] for x in X_conved]), np.concatenate([x[:int(len(X_conved[0]) * frac_val)] for x in X_weeks]), torch.cat([y[:int(len(X_conved[0]) * frac_val)] for y in Y_conved])
        states_train = []
        for st_here in [x[:int(len(X_conved[0]) * frac_val)] for x in states_unflattened]:
            states_train.extend(st_here)
        perm = np.random.permutation(len(X_train_conved))
        X_train_conved, X_train_weeks, Y_train_conved, states_train = X_train_conved[perm], X_train_weeks[perm], Y_train_conved[perm].unsqueeze(axis=-1), np.array(states_train)[perm].tolist()
        
        
        X_val_conved, X_val_weeks, Y_val_conved = torch.cat([x[int(len(X_conved[0]) * frac_val):int(len(X_conved[0]) * frac_test)] for x in X_conved]), np.concatenate([x[int(len(X_conved[0]) * frac_val):int(len(X_conved[0]) * frac_test)] for x in X_weeks]), torch.cat([y[int(len(X_conved[0]) * frac_val):int(len(X_conved[0]) * frac_test)] for y in Y_conved])
        states_val = []
        for st_here in [x[int(len(X_conved[0]) * frac_val):int(len(X_conved[0]) * frac_test)] for x in states_unflattened]:
            states_val.extend(st_here)
        perm = np.random.permutation(len(X_val_conved))
        X_val_conved, X_val_weeks, Y_val_conved, states_val = X_val_conved[perm], X_val_weeks[perm], Y_val_conved[perm].unsqueeze(axis=-1), np.array(states_val)[perm].tolist()
        
        if self.tv=="train":
            return (
                X_train_conved[idx, :, :],
                float_tensor(X_train_weeks[idx, :]),
                Y_train_conved[idx]
            )
        elif self.tv=="val":
            return (
                X_val_conved[idx, :, :],
                float_tensor(X_val_weeks[idx, :]),
                Y_val_conved[idx],
                states_val[idx],
            )


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

# Build dataset
class SeqDataWithWeeks(torch.utils.data.Dataset):
    def __init__(self, X, X_weeks, Y):
        self.X = X
        self.X_weeks = X_weeks
        self.Y = Y[:, None]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            float_tensor(self.X[idx, :, :]),
            float_tensor(self.X_weeks[idx, :]),
            float_tensor(self.Y[idx]),
        )

class SeqDataWithWeeksSmart(torch.utils.data.Dataset):
    def __init__(self, X, X_smart, X_weeks, Y):
        self.X = X
        self.X_smart = X_smart
        self.X_weeks = X_weeks
        self.Y = Y[:, None]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            float_tensor(self.X[idx, :, :]),
            float_tensor(self.X_smart[idx, :, :]),
            float_tensor(self.X_weeks[idx, :]),
            float_tensor(self.Y[idx]),
        )

# Build dataset with state info
class SeqDataWithStates(torch.utils.data.Dataset):
    def __init__(self, X, X_weeks, Y, states):
        self.X = X
        self.X_weeks = X_weeks
        self.Y = Y[:, None]
        self.states = states

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        try:
            return (
                float_tensor(self.X[idx, :, :]),
                float_tensor(self.X_weeks[idx, :]),
                float_tensor(self.Y[idx]),
                self.states[idx],
            )
        except:
            pu.db

class SeqDataWithStatesSmart(torch.utils.data.Dataset):
    def __init__(self, X, X_smart, X_weeks, Y, states):
        self.X = X
        self.X_smart = X_smart
        self.X_weeks = X_weeks
        self.Y = Y[:, None]
        self.states = states

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        try:
            return (
                float_tensor(self.X[idx, :, :]),
                float_tensor(self.X_smart[idx, :, :]),
                float_tensor(self.X_weeks[idx, :]),
                float_tensor(self.Y[idx]),
                self.states[idx],
            )
        except:
            pu.db
if options.smart_mode == 3 or options.smart_mode == 4 or options.smart_mode == 5 or options.smart_mode == 8 or "combine" in options.optionals or "fft" in options.optionals:
    train_dataset = SeqDataWithWeeksSmart(X_train, X_train_smart, X_train_weeks, Y_train)
    val_dataset_with_states = SeqDataWithStatesSmart(X_val, X_val_smart, X_val_weeks, Y_val, states_val)
else:
    train_dataset = SeqDataWithWeeks(X_train, X_train_weeks, Y_train)
    val_dataset_with_states = SeqDataWithStates(X_val, X_val_weeks, Y_val, states_val)
val_dataset = SeqDataWithWeeks(X_val, X_val_weeks, Y_val)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)
val_loader_with_states = torch.utils.data.DataLoader(
    val_dataset_with_states, batch_size=batch_size, shuffle=False
)

if options.smart_mode == 7:
    train_dataset_conved = SeqDataWithConv(raw_data_here=raw_data_unavgd, tv="train")
    val_dataset_conved = SeqDataWithConv(raw_data_here=raw_data_unavgd, tv="val")
    train_loader = torch.utils.data.DataLoader(
        train_dataset_conved, batch_size=batch_size, shuffle=False
    )
    val_loader_with_states = torch.utils.data.DataLoader(
        val_dataset_conved, batch_size=batch_size, shuffle=False
    )


if start_model != "None":
    load_model("/", file=start_model)
    print("Loaded model from", start_model)

    
if options.bert_emb:
    # My method
    opt_bert = torch.optim.AdamW(
        list(seq_enc.parameters())
        + list(feat_enc.parameters()),
        lr=6e-4,betas=(0.9, 0.95), weight_decay=1e-1
    )
    
    opt = torch.optim.Adam(
        list(fnp_enc.parameters()),
        lr=lr,
    )
elif options.rag:
    all_params = list(fnp_enc.named_parameters())
    fnp_enc_params_nobert = [list(fnp_enc.parameters())[i] for i, x in enumerate(all_params) if "bert" not in x[0]]
    fnp_enc_params_bert = [list(fnp_enc.parameters())[i] for i, x in enumerate(all_params) if "bert" in x[0]]

    opt_bert = torch.optim.AdamW(
        fnp_enc_params_bert,
        lr=lr,betas=(0.9, 0.95), weight_decay=1e-1
    )
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
      name="linear", optimizer=opt_bert, num_warmup_steps=0, num_training_steps=num_training_steps)
    
    opt = torch.optim.Adam(
        list(seq_enc.parameters())
        + list(feat_enc.parameters())
        + fnp_enc_params_nobert,
        lr=lr,
    )
else:
    if options.smart_mode == 7:
        # with conv_here
        opt_conv = torch.optim.Adam(
            list(conv_here_1.parameters())
            + list(conv_here_2.parameters()),
            lr=lr*0.0001,
        )
        opt = torch.optim.Adam(
            list(seq_enc.parameters())
            + list(feat_enc.parameters())
            + list(fnp_enc.parameters()),
            lr=lr,
        )
    elif options.smart_mode == 5 and "combine" in options.optionals:
        opt = torch.optim.Adam(
            list(combine.parameters())
            + list(seq_enc.parameters())
            + list(feat_enc.parameters())
            + list(fnp_enc.parameters()),
            lr=lr,
        )
    else:
        # Original method
        opt = torch.optim.Adam(
            list(seq_enc.parameters())
            + list(feat_enc.parameters())
            + list(fnp_enc.parameters()),
            lr=lr,
        )
if "fft" in options.optionals:
        if "combine" in options.optionals:
            opt_fft = torch.optim.Adam(
                list(combine_fft.parameters())
                + list(seq_enc_fft.parameters())
                + list(feat_enc_fft.parameters()),
                lr=lr,
            )
        else:
            opt_fft = torch.optim.Adam(
                list(seq_enc_fft.parameters())
                + list(feat_enc_fft.parameters()),
                lr=lr,
            )

if "fft" in options.optionals:
    X_ref_fft = fft(X_ref).real
    X_ref_all = np.concatenate([X_ref, X_ref_fft], axis=0)
else:
    X_ref_all = X_ref
kkk=0
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
    if options.smart_mode == 7:
        conv_here_1.train()
        conv_here_2.train()
        # opt.zero_grad()
        for i, (x, x_weeks, y) in enumerate(tqdm(data_loader)):
            # if kkk:
            #     pu.db
            
            # if i == 1:
            #     pu.db
            if options.bert_emb or options.rag:
                opt_bert.zero_grad()
            if options.bert_emb:
                # My method
                inp = float_tensor(X_ref).unsqueeze(2)
                mask = float_tensor(X_ref!=-100)
                x_seq = seq_enc(past_values=inp[:,:,-1], past_time_features=float_tensor(raw_data_weeks).unsqueeze(2), past_observed_mask=mask).encoder_last_hidden_state[:,-1,:]
                
                inp = float_tensor(x)
                mask = x != -100
                mask = mask.float()
                x_feat = feat_enc(past_values=inp,past_time_features=float_tensor(x_weeks).unsqueeze(2), past_observed_mask=mask).encoder_last_hidden_state[:,-1,:] # Final dimension is [128, 64]
                
            else:
                # Original method
                # to_concat_batch = 
                # np.random.seed(seed)
                # torch.manual_seed(seed)
                # random.seed(seed)
                x_seq = seq_enc(float_tensor(X_ref).unsqueeze(2)) # Converts [51,119 (,1)] to [51, 60]
                x_feat = feat_enc(float_tensor(x)) # Converts [128, 119, 5] to [128, 60]
                # pu.db
                loss, yp, _ = fnp_enc(x_seq, float_tensor(X_ref), x_feat, y)
                yp = yp[X_ref.shape[0] :]
                loss.backward()
                opt.step()
                opt_conv.step()
                if options.bert_emb or options.rag:
                    opt_bert.step()
                    # lr_scheduler.step()
                YP.append(yp.detach().cpu().numpy())
                T_target.append(y.detach().cpu().numpy())
                total_loss += loss.detach().cpu().numpy()
                train_err += torch.pow(yp - y, 2).mean().sqrt().detach().cpu().numpy()
            opt.zero_grad()
            opt_conv.zero_grad()
                # pu.db
    else:
        # pu.db
        if options.smart_mode == 3 or options.smart_mode == 4 or options.smart_mode == 5 or options.smart_mode == 8 or "combine" in options.optionals or "fft" in options.optionals:
            for i, (x, x_smart, x_weeks, y) in enumerate(data_loader):
                # if kkk:
                #     pu.db
                
                if options.bert_emb or options.rag:
                    opt_bert.zero_grad()
                if options.bert_emb:
                    # My method
                    inp = float_tensor(X_ref).unsqueeze(2)
                    mask = float_tensor(X_ref!=-100)
                    x_seq = seq_enc(past_values=inp[:,:,-1], past_time_features=float_tensor(raw_data_weeks).unsqueeze(2), past_observed_mask=mask).encoder_last_hidden_state[:,-1,:]
                    
                    inp = float_tensor(x)
                    mask = x != -100
                    mask = mask.float()
                    x_feat = feat_enc(past_values=inp,past_time_features=float_tensor(x_weeks).unsqueeze(2), past_observed_mask=mask).encoder_last_hidden_state[:,-1,:] # Final dimension is [128, 64]
                    
                else:
                    if "combine" in options.optionals:
                        x_here = combine(x,x_smart)
                    else:
                        x_here = x

                    x_seq = seq_enc(float_tensor(X_ref).unsqueeze(2))
                    x_feat = feat_enc(x_here) # Converts [128, 119, 5] to [128, 60]

                    if "fft" in options.optionals:
                        x_here_fft = torch.fft.fft(combine_fft(x,x_smart)).real
                        x_seq_fft = seq_enc_fft(float_tensor(X_ref_fft).unsqueeze(2))
                        x_feat_fft = feat_enc_fft(x_here_fft) # Converts [128, 119, 5] to [128, 60]
                        x_seq = torch.cat([x_seq, x_seq_fft], dim=-1)
                        x_feat = torch.cat([x_feat, x_feat_fft], dim=-1)
                        # loss_fft, yp_fft, _ = fnp_enc_fft(x_seq_fft, float_tensor(X_ref_fft), x_feat_fft, y)
                        # yp_fft = yp_fft[X_ref.shape[0] :]
                        # loss_fft.backward()
                        # opt_fft.step()

                loss, yp, _ = fnp_enc(x_seq, float_tensor(X_ref), x_feat, y)
                yp = yp[X_ref.shape[0] :]
                loss.backward()
                opt.step()
                if options.bert_emb or options.rag:
                    opt_bert.step()
                    # lr_scheduler.step()
                # if "fft" in options.optionals:
                #     yp = torch.mean(torch.cat([yp, yp_fft], dim=-1), dim=-1).unsqueeze(-1)
                YP.append(yp.detach().cpu().numpy())
                T_target.append(y.detach().cpu().numpy())
                total_loss += loss.detach().cpu().numpy()
                train_err += torch.pow(yp - y, 2).mean().sqrt().detach().cpu().numpy()
                opt.zero_grad()
                # pu.db

        else:
            for i, (x, x_weeks, y) in enumerate(data_loader):
                # if kkk:
                #     pu.db
                
                if options.bert_emb or options.rag:
                    opt_bert.zero_grad()
                if options.bert_emb:
                    # My method
                    inp = float_tensor(X_ref).unsqueeze(2)
                    mask = float_tensor(X_ref!=-100)
                    x_seq = seq_enc(past_values=inp[:,:,-1], past_time_features=float_tensor(raw_data_weeks).unsqueeze(2), past_observed_mask=mask).encoder_last_hidden_state[:,-1,:]
                    
                    inp = float_tensor(x)
                    mask = x != -100
                    mask = mask.float()
                    x_feat = feat_enc(past_values=inp,past_time_features=float_tensor(x_weeks).unsqueeze(2), past_observed_mask=mask).encoder_last_hidden_state[:,-1,:] # Final dimension is [128, 64]
                    
                else:
                    # Original method
                    # to_concat_batch = 
                    # np.random.seed(seed)
                    # torch.manual_seed(seed)
                    # random.seed(seed)
                    # if options.smart_mode == 8:
                    #     x_seq = seq_enc(float_tensor(np.ones_like(X_ref)).unsqueeze(2))
                    # else:
                    x_seq = seq_enc(float_tensor(X_ref).unsqueeze(2))
                    # x_seq = seq_enc(float_tensor(X_ref).unsqueeze(2)) # Converts [51,119 (,1)] to [51, 60]
                    # if options.smart_mode == 7:
                    #     try:
                    #         x_res = np.zeros_like(x.cpu().detach().numpy())
                    #     except:
                    #         pu.db
                    #     for a in range(x.shape[0]):
                    #         for b in range(x.shape[-1]):
                    #             x_res[a,:,b] = moving_wavg_5(moving_wavg_3(x[a,:,b].cpu().detach().numpy()))
                    #     scaler_here = ScalerFeat(x_res)
                    #     # pu.db
                    #     x = scaler_here.transform(x_res)
                    #     x[x_res==0] = x_res[x_res==0]
                    #     y = float_tensor(np.expand_dims(scaler_here.transform_idx(y[:,0].cpu().detach().numpy(), label_idx), axis=-1))
                    #     # pu.db
                    #     # x_res = np.zeros_like(x.cpu().detach().numpy())
                    #     # for a in range(x.shape[0]):
                    #     #     for b in range(x.shape[-1]):
                    #     #         x_res[a,:,b] = moving_avg_5(moving_avg_3(x[a,:,b].cpu().detach().numpy()))
                    #     # x_feat = feat_enc(float_tensor(x_res)) 
                    #     # x_res = []
                    #     # for ch in range(5):
                    #     #     x_res.append(conv_here_1(float_tensor(x[:,:,ch:ch+1]).permute([0,2,1])))
                    #     # # pu.db
                    #     # # x_res = [conv_here(float_tensor(x[:,:,ch:ch+1]).permute([0,2,1])) for ch in range(5)]
                    #     # x = torch.cat(x_res, axis = 1).permute([0,2,1])
                    #     # x_res = []
                    #     # for ch in range(5):
                    #     #     x_res.append(conv_here_2(float_tensor(x[:,:,ch:ch+1]).permute([0,2,1])))
                    #     # # pu.db
                    #     # # x_res = [conv_here(float_tensor(x[:,:,ch:ch+1]).permute([0,2,1])) for ch in range(5)]
                    #     # x = torch.cat(x_res, axis = 1).permute([0,2,1])

                    #     # m = torch.mean(x_res, axis=1)
                    #     # v = torch.var(x_res, axis=1)
                    #     # x = (x_res - m.unsqueeze(1))/v.unsqueeze(1)
                    #     x_feat = feat_enc(float_tensor(x) )
                    # else:
                    x_feat = feat_enc(float_tensor(x)) # Converts [128, 119, 5] to [128, 60]
                    # pu.db


                # pu.db
                loss, yp, _ = fnp_enc(x_seq, float_tensor(X_ref), x_feat, y)
                yp = yp[X_ref.shape[0] :]
                loss.backward()
                opt.step()
                if options.bert_emb or options.rag:
                    opt_bert.step()
                    # lr_scheduler.step()
                YP.append(yp.detach().cpu().numpy())
                T_target.append(y.detach().cpu().numpy())
                total_loss += loss.detach().cpu().numpy()
                train_err += torch.pow(yp - y, 2).mean().sqrt().detach().cpu().numpy()
                # pu.db
            opt.zero_grad()
    return (
        total_loss / (i + 1),
        train_err / (i + 1),
        np.array(YP).ravel(),
        np.array(T_target).ravel(),
    )

# pu.db
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
        all_As = []
        if options.smart_mode == 7:
            conv_here_1.eval()
            conv_here_2.eval()
            for i, (x, y) in enumerate(tqdm(data_loader)):
                # pu.db
                opt.zero_grad()
                if options.bert_emb or options.rag:
                    opt_bert.zero_grad()
                if options.bert_emb:
                    # My method
                    inp = float_tensor(X_ref).unsqueeze(2)
                    mask = float_tensor(X_ref!=-100)
                    x_seq = seq_enc(past_values=inp[:,:,-1], past_time_features=float_tensor(raw_data_weeks).unsqueeze(2), past_observed_mask=mask).encoder_last_hidden_state[:,-1,:]
                    
                    inp = float_tensor(x)
                    mask = x != -100
                    mask = mask.float()
                    x_feat = feat_enc(past_values=inp,past_time_features=float_tensor(x_weeks).unsqueeze(2), past_observed_mask=mask).encoder_last_hidden_state[:,-1,:] # Final dimension is [128, 64]
                    
                else:
                    # Original method
                    # to_concat_batch = 
                    # np.random.seed(seed)
                    # torch.manual_seed(seed)
                    # random.seed(seed)
                    x_seq = seq_enc(float_tensor(X_ref).unsqueeze(2)) # Converts [51,119 (,1)] to [51, 60]
                    x_feat = feat_enc(float_tensor(x)) # Converts [128, 119, 5] to [128, 60]
                    
                    yp, _, vars, _, _, _, A = fnp_enc.predict(
                        x_feat, x_seq, float_tensor(X_ref), sample
                    )
                    val_err += torch.pow(yp - y, 2).mean().sqrt().detach().cpu().numpy()
                    YP.append(yp.detach().cpu().numpy())
                    T_target.append(y.detach().cpu().numpy())

                    all_As = [] 
        else:
            for i, (x, x_weeks, y) in enumerate(data_loader):
                if options.bert_emb:
                    # My method
                    inp = float_tensor(X_ref).unsqueeze(2)
                    mask = float_tensor(torch.ones_like(inp))
                    x_seq = seq_enc(past_values=inp[:,:,-1], past_time_features=float_tensor(raw_data_weeks).unsqueeze(2), past_observed_mask=mask[:,:,-1]).encoder_last_hidden_state[:,-1,:]
                    
                    inp = float_tensor(x)
                    mask = float_tensor(torch.ones_like(inp))
                    x_feat = feat_enc(past_values=inp,past_time_features=float_tensor(x_weeks).unsqueeze(2), past_observed_mask=mask).encoder_last_hidden_state[:,-1,:] # Final dimension is [128, 64]
                else:
                    # np.random.seed(seed)
                    # torch.manual_seed(seed)
                    # random.seed(seed)
                    x_seq = seq_enc(float_tensor(X_ref).unsqueeze(2))
                    # if options.smart_mode == 7:
                    #     x_res = np.zeros_like(x.cpu().detach().numpy())
                    #     for a in range(x.shape[0]):
                    #         for b in range(x.shape[-1]):
                    #             x_res[a,:,b] = moving_wavg_5(moving_wavg_3(x[a,:,b].cpu().detach().numpy()))
                    #     scaler_here = ScalerFeat(x_res)
                    #     x = scaler_here.transform(x_res)
                    #     x[x_res==0] = x_res[x_res==0]
                    #     y = float_tensor(np.expand_dims(scaler_here.transform_idx(y[:,0].cpu().detach().numpy(), label_idx), axis=-1))
                    #     # x_res = []
                    #     # for ch in range(5):
                    #     #     x_res.append(conv_here_1(float_tensor(x[:,:,ch:ch+1]).permute([0,2,1])))
                    #     # # pu.db
                    #     # # x_res = [conv_here(float_tensor(x[:,:,ch:ch+1]).permute([0,2,1])) for ch in range(5)]
                    #     # x = torch.cat(x_res, axis = 1).permute([0,2,1])
                    #     # x_res = []
                    #     # for ch in range(5):
                    #     #     x_res.append(conv_here_2(float_tensor(x[:,:,ch:ch+1]).permute([0,2,1])))
                    #     # # pu.db
                    #     # # x_res = [conv_here(float_tensor(x[:,:,ch:ch+1]).permute([0,2,1])) for ch in range(5)]
                    #     # x = torch.cat(x_res, axis = 1).permute([0,2,1])

                    #     # m = torch.mean(x_res, axis=1)
                    #     # v = torch.var(x_res, axis=1)
                    #     # x = (x_res - m.unsqueeze(1))/v.unsqueeze(1)
                    #     x_feat = feat_enc(float_tensor(x))
                    # else:
                    x_feat = feat_enc(float_tensor(x)) # Converts [128, 119, 5] to [128, 60]

                    yp, _, vars, _, _, _, A = fnp_enc.predict(
                        x_feat, x_seq, float_tensor(X_ref), sample
                    )
                    # pu.db
                    val_err += torch.pow(yp - y, 2).mean().sqrt().detach().cpu().numpy()
                    YP.append(yp.detach().cpu().numpy())
                    T_target.append(y.detach().cpu().numpy())

                    all_As = [] 
        return val_err / (i + 1), np.array(YP).ravel(), np.array(T_target).ravel()

save_x, save_y, save_yp, save_var = None, None, None, None

def val_step_with_states(data_loader, X, Y, X_ref, sample=True):
    """
    Validation step
    """
    global save_x, save_y, save_yp, save_var 
    with torch.set_grad_enabled(False):
        feat_enc.eval()
        seq_enc.eval()
        fnp_enc.eval()
        val_err = 0.0
        YP = []
        T_target = []
        states_here = []
        all_vars = []
        all_As = []
        if options.smart_mode == 7:
            conv_here_1.eval()
            conv_here_2.eval()
            for i, (x, x_weeks, y, st) in enumerate(tqdm(data_loader)):
                # pu.db
                if options.bert_emb or options.rag:
                    opt_bert.zero_grad()
                if options.bert_emb:
                    # My method
                    inp = float_tensor(X_ref).unsqueeze(2)
                    mask = float_tensor(X_ref!=-100)
                    x_seq = seq_enc(past_values=inp[:,:,-1], past_time_features=float_tensor(raw_data_weeks).unsqueeze(2), past_observed_mask=mask).encoder_last_hidden_state[:,-1,:]
                    
                    inp = float_tensor(x)
                    mask = x != -100
                    mask = mask.float()
                    x_feat = feat_enc(past_values=inp,past_time_features=float_tensor(x_weeks).unsqueeze(2), past_observed_mask=mask).encoder_last_hidden_state[:,-1,:] # Final dimension is [128, 64]
                    
                else:
                    # Original method
                    # to_concat_batch = 
                    # np.random.seed(seed)
                    # torch.manual_seed(seed)
                    # random.seed(seed)
                    x_seq = seq_enc(float_tensor(X_ref).unsqueeze(2)) # Converts [51,119 (,1)] to [51, 60]
                    x_feat = feat_enc(float_tensor(x)) # Converts [128, 119, 5] to [128, 60]
                    
                    yp, _, vars, _, _, _, A = fnp_enc.predict(
                        x_feat, x_seq, float_tensor(X_ref), sample
                    )
                    val_err += torch.pow(yp - y, 2).mean().sqrt().detach().cpu().numpy()
                    YP.append(yp.detach().cpu().numpy())
                    T_target.append(y.detach().cpu().numpy())
                    states_here.extend(st)
                    all_As = [] 
        else:
            if options.smart_mode == 3 or options.smart_mode == 4 or options.smart_mode == 5 or options.smart_mode == 8 or "combine" in options.optionals or "fft" in options.optionals:
                for i, (x, x_smart, x_weeks, y, st) in enumerate(data_loader):
                    if options.bert_emb:
                        inp = float_tensor(X_ref).unsqueeze(2)
                        mask = float_tensor(torch.ones_like(inp))
                        x_seq = seq_enc(past_values=inp[:,:,-1], past_time_features=float_tensor(raw_data_weeks).unsqueeze(2), past_observed_mask=mask[:,:,-1]).encoder_last_hidden_state[:,-1,:]
                        
                        inp = float_tensor(x)
                        mask = float_tensor(torch.ones_like(inp))
                        x_feat = feat_enc(past_values=inp,past_time_features=float_tensor(x_weeks).unsqueeze(2), past_observed_mask=mask).encoder_last_hidden_state[:,-1,:] # Final dimension is [128, 64]
                    else:
                        if "combine" in options.optionals:
                            x_here = combine(x,x_smart)
                        else:
                            x_here = x

                        x_seq = seq_enc(float_tensor(X_ref).unsqueeze(2))
                        x_feat = feat_enc(x_here) # Converts [128, 119, 5] to [128, 60]

                        if "fft" in options.optionals:
                            x_here_fft = torch.fft.fft(combine_fft(x,x_smart)).real
                            x_seq_fft = seq_enc_fft(float_tensor(X_ref_fft).unsqueeze(2))
                            x_feat_fft = feat_enc_fft(x_here_fft) # Converts [128, 119, 5] to [128, 60]
                            x_seq = torch.cat([x_seq, x_seq_fft], dim=-1)
                            x_feat = torch.cat([x_feat, x_feat_fft], dim=-1)

                    yp, _, vars, _, _, _, A = fnp_enc.predict(
                        x_feat, x_seq, float_tensor(X_ref), sample
                    )
                    # if "fft" in options.optionals:
                    #     yp = torch.mean(torch.cat([yp, yp_fft], dim=-1), dim=-1).unsqueeze(-1)
                    val_err += torch.pow(yp - y, 2).mean().sqrt().detach().cpu().numpy()
                    YP.extend(yp.detach().cpu().numpy().tolist())
                    T_target.extend(y.detach().cpu().numpy().tolist())
                    all_vars.extend(vars.detach().cpu().numpy().tolist())
                    all_As.append(A.cpu().numpy())
                    states_here.extend(st)
                    save_x = x
                    save_y = y
                    save_yp = yp
                    save_var = vars
            else:
                for i, (x, x_weeks, y, st) in enumerate(data_loader):
                    if options.bert_emb:
                        # My method
                        # if options.smart_mode == 8:
                        #     inp = float_tensor(np.ones_like(X_ref)).unsqueeze(2)
                        # else:
                        inp = float_tensor(X_ref).unsqueeze(2)
                        mask = float_tensor(torch.ones_like(inp))
                        x_seq = seq_enc(past_values=inp[:,:,-1], past_time_features=float_tensor(raw_data_weeks).unsqueeze(2), past_observed_mask=mask[:,:,-1]).encoder_last_hidden_state[:,-1,:]
                        
                        inp = float_tensor(x)
                        mask = float_tensor(torch.ones_like(inp))
                        x_feat = feat_enc(past_values=inp,past_time_features=float_tensor(x_weeks).unsqueeze(2), past_observed_mask=mask).encoder_last_hidden_state[:,-1,:] # Final dimension is [128, 64]
                    else:
                        # np.random.seed(seed)
                        # torch.manual_seed(seed)
                        # random.seed(seed)
                        # if options.smart_mode == 8:
                        #     x_seq = seq_enc(float_tensor(np.ones_like(X_ref)).unsqueeze(2))
                        # else:
                        x_seq = seq_enc(float_tensor(X_ref).unsqueeze(2))
                        # if options.smart_mode == 7:
                        #     x_res = np.zeros_like(x.cpu().detach().numpy())
                        #     for a in range(x.shape[0]):
                        #         for b in range(x.shape[-1]):
                        #             x_res[a,:,b] = moving_wavg_5(moving_wavg_3(x[a,:,b].cpu().detach().numpy()))
                        #     scaler_here = ScalerFeat(x_res)
                        #     x = scaler_here.transform(x_res)
                        #     x[x_res==0] = x_res[x_res==0]
                        #     # pu.db
                        #     y = float_tensor(np.expand_dims(scaler_here.transform_idx(y[:,0].cpu().detach().numpy(), label_idx), axis=-1))
                        #     # x_res = []
                        #     # for ch in range(5):
                        #     #     x_res.append(conv_here_1(float_tensor(x[:,:,ch:ch+1]).permute([0,2,1])))
                        #     # # pu.db
                        #     # # x_res = [conv_here(float_tensor(x[:,:,ch:ch+1]).permute([0,2,1])) for ch in range(5)]
                        #     # x = torch.cat(x_res, axis = 1).permute([0,2,1])
                        #     # x_res = []
                        #     # for ch in range(5):
                        #     #     x_res.append(conv_here_2(float_tensor(x[:,:,ch:ch+1]).permute([0,2,1])))
                        #     # # pu.db
                        #     # # x_res = [conv_here(float_tensor(x[:,:,ch:ch+1]).permute([0,2,1])) for ch in range(5)]
                        #     # x = torch.cat(x_res, axis = 1).permute([0,2,1])

                        #     # m = torch.mean(x_res, axis=1)
                        #     # v = torch.var(x_res, axis=1)
                        #     # x = (x_res - m.unsqueeze(1))/v.unsqueeze(1)
                        #     x_feat = feat_enc(float_tensor(x) )
                        # else:
                        x_feat = feat_enc(float_tensor(x)) # Converts [128, 119, 5] to [128, 60]

                    yp, _, vars, _, _, _, A = fnp_enc.predict(
                        x_feat, x_seq, float_tensor(X_ref), sample
                    )
                    # pu.db
                    val_err += torch.pow(yp - y, 2).mean().sqrt().detach().cpu().numpy()
                    YP.extend(yp.detach().cpu().numpy().tolist())
                    T_target.extend(y.detach().cpu().numpy().tolist())
                    all_vars.extend(vars.detach().cpu().numpy().tolist())
                    all_As.append(A.cpu().numpy())
                    states_here.extend(st)
                    save_x = x
                    save_y = y
                    save_yp = yp
                    save_var = vars
            YP = [x[0] for x in YP]
            T_target = [x[0] for x in T_target]
            all_vars = [x[0] for x in all_vars]
        return val_err / (i + 1), np.array(YP, dtype=object).ravel(), np.array(T_target).ravel(), states_here, all_vars, all_As


def test_step(X, X_test_weeks, X_ref, samples=1000):
    """
    Test step
    """
    with torch.set_grad_enabled(False):
        feat_enc.eval()
        seq_enc.eval()
        fnp_enc.eval()
        if options.smart_mode == 7:
            conv_here_1.eval()
            conv_here_2.eval()
        YP = []
        As = []
        for i in tqdm(range(samples)):
            if options.bert_emb:
                # My method
                inp = float_tensor(X_ref).unsqueeze(2)
                mask = float_tensor(torch.ones_like(inp))
                x_seq = seq_enc(past_values=inp[:,:,-1], past_time_features=float_tensor(raw_data_weeks).unsqueeze(2), past_observed_mask=mask[:,:,-1]).encoder_last_hidden_state[:,-1,:]
                
                inp = float_tensor(X)
                mask = float_tensor(torch.ones_like(inp))
                x_feat = feat_enc(past_values=inp,past_time_features=float_tensor(X_test_weeks).unsqueeze(2), past_observed_mask=mask).encoder_last_hidden_state[:,-1,:] # Final dimension is [128, 64]
            else:
                x_seq = seq_enc(float_tensor(X_ref).unsqueeze(2))
                x_feat = feat_enc(float_tensor(X))
        
            yp, _, vars, _, _, _, A = fnp_enc.predict(
                x_feat, x_seq, float_tensor(X_ref), sample=False
            )
            YP.append(yp.detach().cpu().numpy())
            As.append(A.cpu().numpy())
        return np.array(YP), As


min_val_err = np.inf
min_val_epoch = 0
all_results = {}
for ep in range(epochs):
    print(f"Epoch {ep+1}")
    print("---------------Details-----------------")
    print("Epiweek: "+str(options.epiweek))
    print("Week ahead: "+str(options.day_ahead))
    print("Min val err: "+str(min_val_err))
    if options.auto_size_best_num is not None:
        print("Auto num: "+str(options.auto_size_best_num))
        print("Window size: "+str(ilk))
    if options.seed != 0:
        print("seed: "+str(options.seed))
    print("---------------------------------------")
    train_loss, train_err, yp, yt = train_step(train_loader, X_train, Y_train, X_ref)
    print(f"Train loss: {train_loss:.4f}, Train err: {train_err:.4f}")
    val_err, yp, yt, st, vars, As = val_step_with_states(val_loader_with_states, X_val, Y_val, X_ref)
    # val_err, yp, yt = val_step(val_loader, X_val, Y_val, X_ref)
    print(f"Val err: {val_err:.4f}")
    all_results[ep] = {"pred": yp, "gt": yt, "states": st, "vars": vars, "As": As, "train_err": train_err}
    # all_results[ep] = {"pred": yp, "gt": yt}
    if options.tb:
        writer.add_scalar('Train/RMSE', train_err, ep)
        writer.add_scalar('Train/loss', train_loss, ep)
        writer.add_scalar('Val/RMSE', val_err, ep)
    if val_err < min_val_err:
        min_val_err = val_err
        min_val_epoch = ep
        try:
            save_model("/localscratch/ssinha97/fnp_saved_models/"+disease+"hosp_models")
        except:
            save_model("/localscratch/ssinha97/fnp_saved_models/"+disease+"hosp_models")
        print("Saved model")
    print()
    print()
    if ep > 100 and ep - min_val_epoch > patience:
        break
    kkk += 1
# pu.db
if options.sliding_window:
    os.makedirs(f"/localscratch/ssinha97/fnp_evaluations/"+disease+"_val_predictions_slidingwindow", exist_ok=True)
    with open(f"/localscratch/ssinha97/fnp_evaluations/"+disease+"_val_predictions_slidingwindow/"+str(save_model_name)+"_predictions.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print("Saved val data at "+"/localscratch/ssinha97/fnp_evaluations/"+disease+"_val_predictions_slidingwindow/"+str(save_model_name)+"_predictions.pkl")

else:
    os.makedirs(f"/localscratch/ssinha97/fnp_evaluations/"+disease+"_val_predictions_normal", exist_ok=True)
    with open(f"/localscratch/ssinha97/fnp_evaluations/"+disease+"_val_predictions_normal/"+str(save_model_name)+"_predictions.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print("Saved val data at "+"/localscratch/ssinha97/fnp_evaluations/"+disease+"_val_predictions_normal/"+str(save_model_name)+"_predictions.pkl")

# Now we get results
try:
    load_model("/localscratch/ssinha97/fnp_saved_models/"+disease+"hosp_models")
except:
    load_model("/localscratch/ssinha97/fnp_saved_models/"+disease+"hosp_models")
X_test = raw_data[states_to_consider_indices]
Y_test, As = test_step(X_test, X_test_weeks, X_ref, samples=2000)
Y_test = Y_test.squeeze()

Y_test_unnorm = scaler.inverse_transform_idx_selected_states(Y_test, label_idx)
# Save predictions
if options.sliding_window:
    try:
        os.makedirs(f"/localscratch/ssinha97/fnp_evaluations/"+disease+"_hosp_stable_predictions_slidingwindow", exist_ok=True)
        with open(f"/localscratch/ssinha97/fnp_evaluations/"+disease+"_hosp_stable_predictions_slidingwindow/"+str(save_model_name)+"_predictions.pkl", "wb") as f:
            pickle.dump([Y_test_unnorm, all_labels[states_to_consider_indices], raw_data_unnorm[:,:,label_idx][states_to_consider_indices], As], f)
        print("Saved as "+"/localscratch/ssinha97/fnp_evaluations/"+disease+"_hosp_stable_predictions_slidingwindow/"+str(save_model_name)+"_predictions.pkl")
    except:
        os.makedirs(f"/localscratch/ssinha97/fnp_evaluations/"+disease+"_hosp_stable_predictions_slidingwindow", exist_ok=True)
        with open(f"/localscratch/ssinha97/fnp_evaluations/"+disease+"_hosp_stable_predictions_slidingwindow/"+str(save_model_name)+"_predictions.pkl", "wb") as f:
            pickle.dump([Y_test_unnorm, all_labels[states_to_consider_indices], raw_data_unnorm[:,:,label_idx][states_to_consider_indices], As], f)
        print("Saved as " +"/localscratch/ssinha97/fnp_evaluations/"+disease+"_hosp_stable_predictions_slidingwindow/"+str(save_model_name)+"_predictions.pkl")
else:
    try:
        os.makedirs(f"/localscratch/ssinha97/fnp_evaluations/"+disease+"_hosp_stable_predictions", exist_ok=True)
        with open(f"/localscratch/ssinha97/fnp_evaluations/"+disease+"_hosp_stable_predictions/"+str(save_model_name)+"_predictions.pkl", "wb") as f:
            pickle.dump([Y_test_unnorm, all_labels[states_to_consider_indices], raw_data_unnorm[:,:,label_idx][states_to_consider_indices], As], f)
        print("Saved as"+"/localscratch/ssinha97/fnp_evaluations/"+disease+"_hosp_stable_predictions/"+str(save_model_name)+"_predictions.pkl")
    except:
        os.makedirs(f"/localscratch/ssinha97/fnp_evaluations/"+disease+"_hosp_stable_predictions", exist_ok=True)
        with open(f"/localscratch/ssinha97/fnp_evaluations/"+disease+"_hosp_stable_predictions/"+str(save_model_name)+"_predictions.pkl", "wb") as f:
            pickle.dump([Y_test_unnorm, all_labels[states_to_consider_indices], raw_data_unnorm[:,:,label_idx][states_to_consider_indices], As], f)
        print("Saved as "+"/localscratch/ssinha97/fnp_evaluations/"+disease+"_hosp_stable_predictions/"+str(save_model_name)+"_predictions.pkl")
        