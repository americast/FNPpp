from curses import raw
import sys
import numpy as np
import pickle
import os

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
from models.fnpmodels import RegressionFNP2
from tqdm import tqdm
from itertools import product
from scipy.signal import argrelextrema
from copy import deepcopy
from functools import reduce

parser = OptionParser()
# parser.add_option("-p", "--epiweek_pres", dest="epiweek_pres", default="202240", type="string")
# parser.add_option("-e", "--epiweek", dest="epiweek", default="202140", type="string")
parser.add_option("--epochs", dest="epochs", default=3500, type="int")
parser.add_option("--lr", dest="lr", default=1e-5, type="float")
parser.add_option("--patience", dest="patience", default=1000, type="int")
parser.add_option("-d", "--day", dest="day_ahead", default=1, type="int")
parser.add_option("-s", "--seed", dest="seed", default=0, type="int")
parser.add_option("-b", "--batch", dest="batch_size", default=128, type="int")
parser.add_option("-m", "--save", dest="save_model", default="default", type="string")
parser.add_option("--start_model", dest="start_model", default="None", type="string")
parser.add_option("-c", "--cuda", dest="cuda", default=True, action="store_true")
parser.add_option("--start", dest="start_day", default=-120, type="int")
parser.add_option("-t", "--tb", action="store_true", dest="tb", default=False)
parser.add_option("-W", "--use-sliding-window", dest="sliding_window", default=False, action="store_true")
parser.add_option("--auto-size-best-num", dest="auto_size_best_num", default=None, type="int")
parser.add_option("--sliding-window-size", dest="window_size", type="int", default=17)
parser.add_option("--sliding-window-stride", dest="window_stride", type="int", default=15)
parser.add_option("--disease", dest="disease", type="string", default="power")
parser.add_option("--preprocess", dest="preprocess", action="store_true", default=False)
parser.add_option("--cnn", dest="cnn", action="store_true", default=False)
parser.add_option("--rag", dest="rag", action="store_true", default=False)
parser.add_option("--nn", dest="nn", default="none", type="choice", choices=["none", "simple", "bn", "dot", "bert"])
parser.add_option("--smart-mode", dest="smart_mode", default=0, type="int")
parser.add_option("--optionals", dest="optionals", default=" ", type="str")

(options, args) = parser.parse_args()
# epiweek_pres = options.epiweek_pres
# epiweek = options.epiweek
day_ahead = options.day_ahead
seed = options.seed
save_model_name = options.save_model
start_model = options.start_model
cuda = options.cuda
start_day = options.start_day
batch_size = options.batch_size
lr = options.lr
epochs = options.epochs
patience = options.patience
disease = options.disease
if options.rag and options.cnn:
    print("Cannot have cnn and rag together")
    sys.exit(0)
    
if options.cnn:
    disease = disease + "_cnn"
if options.rag:
    disease = disease + "_rag"
if options.nn != "none":
    disease = disease + "_nn-" + options.nn

if  options.smart_mode == 8:
    options.nn = "bn"
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
#     raw_data = np.array(raw_data)[:, :-5 + day_ahead, :]
# else:    
#     raw_data = np.array(raw_data)[:, :diff_epiweeks(epiweek, epiweek_pres) + day_ahead, :]  # states x days x featureslabel_idx = include_cols.index("cdc_hospitalized")
# if options.disease == "flu":
#     label_idx = include_cols.index("cdc_flu_hosp")
# else:
#     label_idx = include_cols.index("cdc_hospitalized")
# all_labels = raw_data[:, -1, label_idx]
# print(f"Diff epiweeks: {diff_epiweeks(epiweek, epiweek_pres)}")
# raw_data = raw_data[:, start_day:-day_ahead, :]

# raw_data_unnorm = raw_data.copy()

if options.tb:
    if options.sliding_window:
        if options.preprocess:
            writer = SummaryWriter("runs/"+disease+"/"+disease+"_preprocessedslidingwindow_weekahead_"+str(options.day_ahead)+"_windowsize_"+str(options.window_size)+"_stride_"+str(options.window_stride))
        else:
            writer = SummaryWriter("runs/"+disease+"/"+disease+"_slidingwindow_weekahead_"+str(options.day_ahead)+"_windowsize_"+str(options.window_size)+"_stride_"+str(options.window_stride))
    else:
        writer = SummaryWriter("runs/"+disease+"/"+disease+"_normal_weekahead_"+str(options.day_ahead))
label_idx = 0
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

# # Shuffle data
# perm = np.random.permutation(len(X_train))
# X_train, Y_train = X_train[perm], Y_train[perm]

# # Reference sequences
# X_ref = raw_data[:, :, label_idx]

# # Divide val and train
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
for d in tqdm(data):
    f = []
    for x in d[2:]:
        try:
            f.append(float(x))
        except:
            f.append(0.0)
    features.append(f)
features = np.array(features)
features = np.expand_dims(features, axis=0)
# pu.db
scaler = ScalerFeat(features)
features = scaler.transform(features)
features = np.squeeze(features, axis=0)
target = features[:, 0]

total_time = len(data)
test_start = int(total_time * 0.7)
test_end = int(total_time * 0.9)

X, X_symp, Y, mt, reg = [], [], [], [], []

def sample_train(n_samples, window = 20):
    X, X_smart, X_symp, X_symp_smart, Y, mt, reg = [], [], [], [], [], [], []
    start_seqs = np.random.randint(0, test_start, n_samples)
    for start_seq in start_seqs:
        if options.smart_mode == 2 or options.smart_mode == 3 or options.smart_mode == 5 or options.smart_mode == 8:
            X.append(target[start_seq:start_seq+window, np.newaxis])
            X_symp.append(features[start_seq:start_seq+window])
            X_smart.append(target_avg[start_seq:start_seq+window, np.newaxis])
            X_symp_smart.append(features_avg[start_seq:start_seq+window])
        else:
            X.append(target[start_seq:start_seq+window, np.newaxis])
            X_symp.append(features[start_seq:start_seq+window])
        
        if options.smart_mode == 2:
            Y.append(target_avg[start_seq+window+(options.day_ahead - 1)])
        else:
            Y.append(target[start_seq+window+(options.day_ahead - 1)])
        mt.append(month[start_seq+window])
        reg.append(tod[start_seq+window])
    X = np.array(X)
    X_symp = np.array(X_symp)
    X_smart = np.array(X_smart)
    X_symp_smart = np.array(X_symp_smart)
    Y = np.array(Y)
    mt = np.array(mt)
    reg = np.array(reg)
    return X, X_smart, X_symp, X_symp_smart, Y, mt, reg

def sample_val(n_samples, window = 20):
    X, X_smart, X_symp, X_symp_smart, Y, mt, reg = [], [], [], [], [], [], []
    start_seqs = np.random.randint(test_start, test_end-(window - (options.day_ahead -1)), n_samples)
    for start_seq in start_seqs:
        if options.smart_mode == 2 or options.smart_mode == 3 or options.smart_mode == 5 or options.smart_mode == 8:
            X.append(target[start_seq:start_seq+window, np.newaxis])
            X_symp.append(features[start_seq:start_seq+window])
            X_smart.append(target_avg[start_seq:start_seq+window, np.newaxis])
            X_symp_smart.append(features_avg[start_seq:start_seq+window])
        else:
            X.append(target[start_seq:start_seq+window, np.newaxis])
            X_symp.append(features[start_seq:start_seq+window])
        
        if options.smart_mode == 2:
            Y.append(target_avg[start_seq+window+(options.day_ahead - 1)])
        else:
            Y.append(target[start_seq+window+(options.day_ahead - 1)])
        mt.append(month[start_seq+window])
        reg.append(tod[start_seq+window])
    X = np.array(X)
    X_symp = np.array(X_symp)
    X_smart = np.array(X_smart)
    X_symp_smart = np.array(X_symp_smart)
    Y = np.array(Y)
    mt = np.array(mt)
    reg = np.array(reg)
    return X, X_smart, X_symp, X_symp_smart, Y, mt, reg

def sample_test(n_samples, window = 20):
    X, X_smart, X_symp, X_symp_smart, Y, mt, reg = [], [], [], [], [], [], []
    start_seqs = np.random.randint(test_end, total_time-(window - (options.day_ahead -1)), n_samples)
    for start_seq in start_seqs:
        X.append(target[start_seq:start_seq+window, np.newaxis])
        X_symp.append(features[start_seq:start_seq+window])
        X_smart.append(target_avg[start_seq:start_seq+window, np.newaxis])
        X_symp_smart.append(features_avg[start_seq:start_seq+window])
        Y.append(target[start_seq+window+(options.day_ahead - 1)])
        mt.append(month[start_seq+window])
        reg.append(tod[start_seq+window])
    X = np.array(X)
    X_symp = np.array(X_symp)
    X_smart = np.array(X_smart)
    X_symp_smart = np.array(X_symp_smart)
    Y = np.array(Y)
    mt = np.array(mt)
    reg = np.array(reg)
    return X, X_smart, X_symp, X_symp_smart, Y, mt, reg
# pu.db

# Reference points
# len_seq = test_start//splits


# X_ref = features[np.newaxis,:,0]
# seq_references = np.array([features[i: i+len_seq, 0, np.newaxis] for i in range(0, test_start, len_seq)])[:, :options.batch_size, :]
# symp_references = np.array([features[i: i+len_seq] for i in range(0, test_start, len_seq)])[:, :options.batch_size, :]
# month_references = np.arange(12)
# reg_references = np.arange(4)
# splits = 4
# if options.smart_mode != 0:
features_avg = np.zeros_like(features)
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
for i in tqdm(range(features.shape[1])):
    rw = features[:, i]
    rw_here = moving_avg(rw)
    features_avg[:, i] = rw_here
    if i == label_idx:
        target_avg = rw_here
kernel_size = 5
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
for i in tqdm(range(features.shape[1])):
    rw = features_avg[:, i]
    rw_here = moving_avg(rw)
    features_avg[:, i] = rw_here
    if i == label_idx:
        target_avg = rw_here

    # pu.db

if options.smart_mode == 2 or options.smart_mode == 3 or options.smart_mode == 4 or options.smart_mode == 6:
    X_ref = features_avg
else:
    X_ref = features
X_ref_smart = features_avg
# pu.db
train_seqs, train_seqs_smart, X_train, X_train_smart, Y_train, mt, reg = sample_train(options.batch_size)
val_seqs, val_seqs_smart, X_val, X_val_smart, Y_val, mt_val, reg_val = sample_val(options.batch_size)
test_seqs, test_seqs_smart, X_test, X_test_smart, Y_test, mt_test, reg_test = sample_test(options.batch_size)
# pu.db

kernel_size = 25
avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
def moving_avg(x, kernel_size=25):
    # padding on the both ends of time series
    front = x[0:1].repeat((kernel_size - 1) // 2)
    end = x[-1:].repeat((kernel_size - 1) // 2)
    x = np.concatenate([front, x, end]).tolist()
    with torch.no_grad():
        x = avg(torch.tensor([[x]]))
    x = x.detach().numpy().tolist()
    return np.array(x[0][0])

X_ref = np.expand_dims(X_ref, axis=0)
if options.sliding_window:
    if options.smart_mode == 1 or options.smart_mode == 4 or options.smart_mode == 8:
        X_ref_orig_shape = X_ref.shape
        to_concat = []
        for i in tqdm(range(features.shape[1])):
            series_here = features_avg[:,i]
            minimas =  argrelextrema(series_here, np.less)[0]
            derivative = np.diff(series_here)
            split_idxs= [np.where(derivative!=0)[0][0]]
            start_scan = split_idxs[0]
            # if states[i] == "ID":
            #     pu.db
            for stop_scan in minimas.tolist()+[len(series_here)-1]:
                vals = series_here[start_scan:stop_scan]
                max_here = np.max(vals)
                if max_here > 0:
                    split_idxs.append(start_scan)
                    split_idxs.append(stop_scan)
                start_scan = stop_scan
            # print("mean "+str(np.mean(series_here)))
            # print("std "+str(np.std(series_here)))
            # print("mean+std "+str(np.mean(series_here) + np.std(series_here)))
            if len(series_here) - 1 not in split_idxs:
                split_idxs.append(len(series_here) - 1)
            split_idxs_diffs = [100]+np.diff(split_idxs).tolist()
            split_idxs = np.array(split_idxs)[np.array(split_idxs_diffs) >= 10].tolist()
            # if states[i] == "AR":
            #     pu.db

            # print(split_idxs)
            # if states[i] == "AR":
            #     pu.db
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
            # print(len(split_idxs))
            # if len(split_idxs) > 1000:
            #     split_idxs = sorted(random.sample(split_idxs, 1000))

            for si in split_idxs:
                if options.smart_mode == 1 or options.smart_mode == 8:
                    to_concat.append(features[ci:si, i].tolist())
                else:
                    to_concat.append(features_avg[ci:si, i].tolist())
                ci = si

        if len(to_concat) > 1500:
            to_concat = random.sample(to_concat, 1500)
        max_len = 0
        for tc in to_concat:
            if len(tc) > max_len:
                max_len = len(tc)
        for t, tc in enumerate(to_concat):
            to_concat[t] = np.expand_dims(np.array([-100 for x in range(max_len-len(tc))] + tc), axis=0)
        X_ref = np.expand_dims(np.concatenate(to_concat), axis=-1)
        print(X_ref.shape)
    elif options.auto_size_best_num is not None:
        lags = [x for x in range(100,1200,100)]
        # for x in range(5,10:
        #     lags.append(2**x)
        idx_scores = [0 for x in range(len(lags))]
        to_concat = []
        seq_lengths = []
        for st_idx in tqdm(range(7)):
            series_here = X_ref[0, :, st_idx]
            series_here = series_here - moving_avg(series_here)
            acs = []
            for lag in lags:
                ac_here = []
                for it in range(len(series_here)-1, lag-1, -1):
                    ac_here.append(series_here[it] * series_here[it - lag])
                acs.append(np.mean(np.array(ac_here)))
            sorted_indices = np.flip(np.argsort(acs)).tolist()
            lag_needed = lags[sorted_indices[options.auto_size_best_num]]
            for w in range(0, X_ref.shape[1] - lag_needed + 1, lag_needed):
                seq_lengths.append(lag_needed)
                to_concat.append(X_ref[:,w:w + lag_needed, :])


            # for k, sindxs in enumerate(sorted_indices):
            #     seq_lengths.append(lag_needed)
            #     idx_scores[sindxs] += len(lags) - k

        max_length = np.max(seq_lengths)
        # pu.db
        for t, tc in enumerate(to_concat):
            len_here = tc.shape[1]
            diff_len = max_length - len_here
            to_concat[t] = np.concatenate([to_concat[t], np.zeros((1,diff_len,7))], axis=1)
        X_ref = np.concatenate(to_concat)
        to_choose = []
        for k in range(0,X_ref.shape[0],5):
            to_choose.append(k)
        X_ref = X_ref[to_choose]
        to_choose = []
        for k in range(0,X_ref.shape[1],5):
            to_choose.append(k)
        X_ref = X_ref[:,to_choose,:]
        ilk = ils = X_ref.shape[1]

        # lags_needed_idxs = np.flip(np.argsort(idx_scores)).tolist()
        # ilk = lags[lags_needed_idxs[options.auto_size_best_num]]
        # ils = lags[lags_needed_idxs[options.auto_size_best_num]]
    else:
        # pu.db
        ilk = options.window_size
        ils = options.window_stride
    # """
        to_concat = []
        for w in range(0, X_ref.shape[1] - ilk + 1, options.window_stride):
            to_concat.append(X_ref[:,w:w + ilk])
        
        X_ref_orig_shape = X_ref.shape
        X_ref = np.concatenate(to_concat)
        # pu.db
    if options.preprocess:
        all_idxs = np.zeros((X_ref.shape[0], X_train.shape[0], X_ref.shape[1]+X_train.shape[1]))
        for idx in tqdm(product(range(X_ref.shape[0]), range(X_train.shape[0]))):
            all_idxs[idx[0], idx[1]] = np.concatenate((X_ref[idx[0]], X_train[idx[1]][:,label_idx]), axis = -1)
        all_idxs = np.reshape(all_idxs,(-1, all_idxs.shape[-1]))
        pccs = batched_compute_pcc(torch.tensor(all_idxs[:,:X_ref.shape[1]]), torch.tensor(all_idxs[:,X_ref.shape[1]:]))
        pccs = pccs.numpy().reshape((X_ref.shape[0], X_train.shape[0]))
        summed_pccs = np.sum(pccs, axis=-1)
        best_ref_idxs = np.argsort(summed_pccs)[:1000]
        X_ref = X_ref[best_ref_idxs, :]
        # pu.db


# Build model
feat_enc = GRUEncoder(in_size=7, out_dim=60,).to(device)
if options.smart_mode == 1 or options.smart_mode == 4 or options.smart_mode == 8:
    seq_enc = GRUEncoder(in_size=1, out_dim=60,).to(device)
else:
    seq_enc = GRUEncoder(in_size=7, out_dim=60,).to(device)
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

if "fft" in options.optionals:
    feat_enc_fft = GRUEncoder(in_size=7, out_dim=60,).to(device)
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
    combine_fft = Combine(7).to(device)

def load_model(folder, file=save_model_name):
    """
    Load model
    """
    full_path = os.path.join(folder, file)
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
    
class SeqDataSmart(torch.utils.data.Dataset):
    def __init__(self, X, X_smart, Y):
        self.X = X
        self.X_smart = X_smart
        self.Y = Y[:, None]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            float_tensor(self.X[idx, :, :]),
            float_tensor(self.X_smart[idx, :, :]),
            float_tensor(self.Y[idx]),
        )

# Build dataset with state info
class SeqDataWithStates(torch.utils.data.Dataset):
    def __init__(self, X, Y, states):
        self.X = X
        self.Y = Y[:, None]
        self.states = states

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        try:
            return (
                float_tensor(self.X[idx, :, :]),
                float_tensor(self.Y[idx]),
                self.states[idx],
            )
        except:
            pu.db

train_dataset = SeqDataSmart(X_train, X_train_smart, Y_train)
val_dataset = SeqDataSmart(X_val, X_val_smart, Y_val)
val_dataset_with_states = SeqData(X_val, Y_val)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)
val_loader_with_states = torch.utils.data.DataLoader(
    val_dataset_with_states, batch_size=batch_size, shuffle=False
)
if start_model != "None":
    load_model("./"+disease+"power_models", file=start_model)
    print("Loaded model from", start_model)

models_to_train = [seq_enc, feat_enc, fnp_enc]

if "combine" in options.optionals:
    combine = Combine(7).to(device)
    models_to_train += [combine]

if "fft" in options.optionals:
    models_to_train += [seq_enc_fft, feat_enc_fft, combine_fft]

opt = torch.optim.Adam(
    reduce(lambda x, y: x + y, [list(m.parameters()) for m in models_to_train]), lr=lr,
)


def train_step(data_loader, X, Y, X_ref):
    """
    Train step
    """
    for model in models_to_train:
        model.train()
    total_loss = 0.0
    train_err = 0.0
    YP = []
    T_target = []
    for i, (x, x_smart, y) in enumerate(data_loader):
        opt.zero_grad()
        if "combine" in options.optionals:
            x_here = combine(x,x_smart)
        elif options.smart_mode == 8:
            x_here = x_smart
        else:
            x_here = x
        x_seq = seq_enc(float_tensor(X_ref))
        x_feat = feat_enc(x_here)

        if "fft" in options.optionals:
            x_here_fft = torch.fft.fft(combine_fft(x,x_smart)).real
            x_seq_fft = seq_enc_fft(float_tensor(X_ref))
            x_feat_fft = feat_enc_fft(x_here_fft) # Converts [128, 119, 5] to [128, 60]
            x_seq = torch.cat([x_seq, x_seq_fft], dim=-1)
            x_feat = torch.cat([x_feat, x_feat_fft], dim=-1)

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
        for model in models_to_train:
            model.eval()
        val_err = 0.0
        YP = []
        T_target = []
        all_vars = []
        all_As = []
        # load_model("/localscratch/ssinha97/"+disease+"power_models")
        for i, (x, x_smart, y) in enumerate(data_loader):
            x_seq = seq_enc(float_tensor(X_ref))
            if "combine" in options.optionals:
                x_here = combine(x,x_smart)
            elif options.smart_mode == 8:
                x_here = x_smart
            else:
                x_here = x
            x_feat = feat_enc(x_here)
            if "fft" in options.optionals:
                x_here_fft = torch.fft.fft(combine_fft(x,x_smart)).real
                x_seq_fft = seq_enc_fft(float_tensor(X_ref))
                x_feat_fft = feat_enc_fft(x_here_fft) # Converts [128, 119, 5] to [128, 60]
                x_seq = torch.cat([x_seq, x_seq_fft], dim=-1)
                x_feat = torch.cat([x_feat, x_feat_fft], dim=-1)
            yp, _, vars, _, _, _, A = fnp_enc.predict(
                x_feat, x_seq, float_tensor(X_ref), sample
            )
            # pu.db
            val_err += torch.pow(yp - y, 2).mean().sqrt().detach().cpu().numpy()
            YP.extend(yp.detach().cpu().numpy().squeeze(-1).tolist())
            T_target.extend(y.detach().cpu().numpy().squeeze(-1).tolist())
            all_vars.extend(vars.detach().cpu().numpy().squeeze(-1).tolist())
            all_As.append(A.cpu().numpy())
        # YP = [x[0] for x in YP]
        # T_target = [x[0] for x in T_target]
        # all_vars = [x[0] for x in all_vars]
        return val_err / (i + 1), np.array(YP, dtype=object).ravel(), np.array(T_target).ravel(), all_vars, all_As

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
        As = []
        for i in tqdm(range(samples)):
            x_seq = seq_enc(float_tensor(X_ref))
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
    print("Week ahead: "+str(options.day_ahead))
    print("Min val err: "+str(min_val_err))
    if options.auto_size_best_num is not None:
        print("Auto num: "+str(options.auto_size_best_num))
        print("Window size: "+str(ilk))
    print("num refs: "+str(X_ref.shape[0]))
    if options.seed != 0:
        print("Seed: "+str(options.seed))
    print("---------------------------------------")
    train_loss, train_err, yp, yt = train_step(train_loader, X_train, Y_train, X_ref)
    print(f"Train loss: {train_loss:.4f}, Train err: {train_err:.4f}")
    val_err, yp, yt, vars, As = val_step(val_loader, X_val, Y_val, X_ref)
    print(f"Val err: {val_err:.4f}")
    # pu.db
    all_results[ep] = {"pred": yp, "gt": yt, "vars": vars, "As": As}
    if options.tb:
        writer.add_scalar('Train/RMSE', train_err, ep)
        writer.add_scalar('Train/loss', train_loss, ep)
        writer.add_scalar('Val/RMSE', val_err, ep)
    if val_err < min_val_err:
        min_val_err = val_err
        min_val_epoch = ep
        try:
            save_model("/nvmescratch/ssinha97/"+disease+"power_models")
        except:
            save_model("/localscratch/ssinha97/"+disease+"power_models")
        print("Saved model")
    print()
    print()
    if ep > 100 and ep - min_val_epoch > patience:
        break

if options.optionals != " ":
    save_model_name = save_model_name+"_optionals_"+options.optionals

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

"""
# Now we get results
try:
    load_model("/nvmescratch/ssinha97/"+disease+"power_models")
except:
    load_model("/localscratch/ssinha97/"+disease+"power_models")
Y_pred, As = test_step(X_test, X_ref, samples=2000)
Y_pred = Y_pred.squeeze()
Y_pred_unnorm = scaler.inverse_transform_idx(Y_pred, label_idx)
X_test_unnorm = scaler.inverse_transform_idx(X_test, label_idx)
Y_test_unnorm = scaler.inverse_transform_idx(Y_test, label_idx)
# Save predictions
if options.sliding_window:
    os.makedirs(f"./"+disease+"_power_stable_predictions_slidingwindow", exist_ok=True)
    with open(f"./"+disease+"_power_stable_predictions_slidingwindow/"+str(save_model_name)+"_predictions.pkl", "wb") as f:
        pickle.dump([Y_pred_unnorm, Y_test_unnorm, X_test_unnorm[:, :, label_idx], As], f)
else:
    os.makedirs(f"./"+disease+"_power_stable_predictions", exist_ok=True)
    with open(f"./"+disease+"_power_stable_predictions/"+str(save_model_name)+"_predictions.pkl", "wb") as f:
        pickle.dump([Y_pred_unnorm, Y_test_unnorm, X_test_unnorm[:, :, label_idx], As], f)
"""