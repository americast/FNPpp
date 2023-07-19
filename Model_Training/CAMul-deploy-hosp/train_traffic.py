import os
import pickle
from optparse import OptionParser
import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from symp_extract.consts import include_cols
import dgl
from functools import reduce

from models.utils import device, float_tensor, long_tensor
from models.multimodels import (
    EmbedEncoder,
    GRUEncoder,
    EmbGCNEncoder,
    LatentEncoder,
    CorrEncoder,
    Decoder,
)
from tqdm import tqdm
from scipy.signal import argrelextrema
from datasets import load_dataset

import pudb
parser = OptionParser()
parser.add_option("-w", "--week", dest="week_ahead", type="int", default=2)
parser.add_option("-y", "--year", dest="year", type="int", default=2020)
parser.add_option("-n", "--num", dest="num", type="string")
parser.add_option("-e", "--epoch", dest="epochs", type="int", default="3000")
parser.add_option("-m", "--save", dest="save_model", default="default", type="string")
parser.add_option("--seed", dest="seed", default=0, type="int")
parser.add_option("--traffic-row", dest="traffic_row", default=0, type="int")
parser.add_option("--smart-mode", dest="smart_mode", default=0, type="int")
parser.add_option("--nn", dest="nn", default="none", type="choice", choices=["none", "simple", "bn", "dot", "bert"])

(options, args) = parser.parse_args()

regions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
week_ahead = options.week_ahead
num = options.num
epochs = options.epochs
year = options.year
seed = options.seed
start_year = 2018

if options.smart_mode == 8:
    options.nn = "bn"

np.random.seed(seed)
torch.manual_seed(seed)

import random
random.seed(seed)

# with open("./data/symptom_data/saves/combine.pkl", "rb") as f:
#     data = pickle.load(f)


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

    def transform_idx(self, data, idx):
        return (data - self.means[:, idx]) / self.vars[:, idx]

    def inverse_transform_idx(self, data, idx):
        return data * self.vars[:, idx] + self.means[:, idx]

weekly = True
if weekly:
    dataset = load_dataset("monash_tsf", "traffic_weekly")
else:
    dataset = load_dataset("monash_tsf", "traffic_hourly")
traffic_rows = [options.traffic_row]
if weekly:
    train_len = 88
    val_len = 96
    test_len = 104
    window = 20
else:
    train_len = 17448
    val_len = 48
    test_len = 48
    window = 96
# pu.db

features_traffic_test = []
time = []
for i in range(862):
    features_traffic_test.append(dataset["test"][i]["target"])
    time.append(i)
features_traffic_test = np.array(features_traffic_test).T

features = np.expand_dims(features_traffic_test, axis = 0)
time = np.array(time)
# time = (time / np.max(time)) - 0.5

scaler = ScalerFeat(features)
features = scaler.transform(features)
features = np.squeeze(features, axis=0)
target = features

total_time = test_len
val_start = train_len
val_end = val_len

X, X_symp, Y, mt, reg = [], [], [], [], []


def sample_train(window = window):
    X, Y, t_x, t_y = [], [], [], []
    start_seqs = list(range(0, val_start - (window + week_ahead)))
    for start_seq in tqdm(start_seqs):
        # X.append(target[start_seq:start_seq+window, np.newaxis])
        for traffic_row in traffic_rows:
            X.append(features[start_seq:start_seq+window])
            t_x.append(time[start_seq:start_seq+window])
            Y.append(features[start_seq+window+(week_ahead - 1), traffic_row])
            t_y.append(time[start_seq+window-1:start_seq+window+week_ahead])
        # mt.append(month[start_seq+window])
        # reg.append(tod[start_seq+window])
    # X = np.array(X)
    # idxs = random.sample(range(len(X)), 8192)
    X = np.array(X)
    Y = np.array(Y)
    # t_x = [x for i,x in enumerate(t_x) if i in idxs]
    # t_y = [x for i,x in enumerate(t_y) if i in idxs]
    # mt = np.array(mt)
    # reg = np.array(reg)
    return X,Y,t_x,t_y


def sample_val(window = 96):
    X, Y, t_x, t_y = [], [], [], []
    start_seqs = list(range(val_start - (window + week_ahead), val_end - (window + week_ahead)))
    for start_seq in tqdm(start_seqs):
        for traffic_row in traffic_rows:
            X.append(features[start_seq:start_seq+window])
            t_x.append(time[start_seq:start_seq+window])
            Y.append(features[start_seq+window+(week_ahead - 1), traffic_row])
            t_y.append(time[start_seq+window-1:start_seq+window+week_ahead])
    # idxs = random.sample(range(len(X)), 128)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y, t_x, t_y




X_train, Y_train, X_train_time, Y_train_time = sample_train()
X_val, Y_val, X_val_time, Y_val_time = sample_val()



X_ref = features[:val_start]



# Make sequence for all years
# def get_sequence(data: pd.DataFrame, hhs: int, year: int) -> np.ndarray:
#     d1 = data[(data.year == year - 1) & (data.hhs == hhs) & (data.epiweek > 20)][
#         include_cols + ["ili", "epiweek", "hhs"]
#     ]
#     d2 = data[(data.year == year) & (data.hhs == hhs) & (data.epiweek <= 20)][
#         include_cols + ["ili", "epiweek", "hhs"]
#     ]
#     d1 = np.array(d1)
#     d2 = np.array(d2)
#     print(len(d1), len(d2))
#     return np.vstack((d1, d2))


# train_years = [y for y in range(start_year, year)]
# test_years = [year]

# train_seqs = [get_sequence(data, hhs, year) for hhs in regions for year in train_years]
# test_seqs = [get_sequence(data, hhs, year) for hhs in regions for year in test_years]


# def seq_to_dataset(seq, week_ahead=week_ahead):
#     X, Y, wk, reg = [], [], [], []
#     start_idx = max(week_ahead, seq.shape[0] - 32 + week_ahead)
#     for i in range(start_idx, seq.shape[0]):
#         X.append(seq[: i - week_ahead + 1, :-2])
#         Y.append(seq[i, -3])
#         wk.append(seq[i, -2])
#         reg.append(seq[i, -1])
#     return X, Y, wk, reg

avg_3 = torch.nn.AvgPool1d(kernel_size=3, stride=1, padding=0)
def moving_avg_3(x, kernel_size=3):
    # padding on the both ends of time series
    front = x[0:1].repeat((kernel_size - 1) // 2)
    end = x[-1:].repeat((kernel_size - 1) // 2)
    x = np.concatenate([front, x, end]).tolist()
    with torch.no_grad():
        # if options.smart_mode == 7:
        #     x = conv_here_1(torch.tensor([[x]]))
        # else:
            x = avg_3(torch.tensor([[x]]))
    x = x.detach().numpy().tolist()
    return np.array(x[0][0])

avg_5 = torch.nn.AvgPool1d(kernel_size=5, stride=1, padding=0)
def moving_avg_5(x, kernel_size=5):
    # padding on the both ends of time series
    front = x[0:1].repeat((kernel_size - 1) // 2)
    end = x[-1:].repeat((kernel_size - 1) // 2)
    x = np.concatenate([front, x, end]).tolist()
    with torch.no_grad():
        # if options.smart_mode == 7:
        #     x = conv_here_2(torch.tensor([[x]]))
        # else:
            x = avg_5(torch.tensor([[x]]))
    x = x.detach().numpy().tolist()
    return np.array(x[0][0])

def smoothen(x):
    if len(x.shape) == 1:
        x_new = moving_avg_5(moving_avg_3(x))
        assert x_new.shape == x.shape
        return x_new
    else:
        x_new = np.zeros_like(x)
        for i in range(x.shape[1]):
            x_new[:,i] = moving_avg_5(moving_avg_3(x[:, i]))
            assert x_new.shape == x.shape
        return x_new

if options.smart_mode == 5:
    for i, xt in enumerate(tqdm(X_train)):
        X_train[i] = smoothen(xt)
    for i, xv in enumerate(tqdm(X_val)):
        X_val[i] = smoothen(xv)

# train_dataset = [seq_to_dataset(seq, week_ahead) for seq in train_seqs]
# X, X_symp, Y, wk, reg = [], [], [], [], []
# for x, y, w, r in train_dataset:
#     if options.smart_mode == 5:
#         X.extend([smoothen(l[:, -1]) for l in x])
#         X_symp.extend([smoothen(l[:, :-1]) for l in x])
#     else:
#         X.extend([l[:, -1] for l in x])
#         X_symp.extend([l[:, :-1] for l in x])
#     Y.extend(y)
#     wk.extend(w)
#     reg.extend(r)
# test_dataset = [seq_to_dataset(seq, week_ahead) for seq in test_seqs]
# X_test, X_symp_test, Y_test, wk_test, reg_test = [], [], [], [], []
# for x, y, w, r in test_dataset:
#     if options.smart_mode == 5:
#         X_test.extend([smoothen(l[:, -1]) for l in x])
#         X_symp_test.extend([smoothen(l[:, :-1]) for l in x])
#     else:
#         X_test.extend([l[:, -1] for l in x])
#         X_symp_test.extend([l[:, :-1] for l in x])
#     Y_test.extend(y)
#     wk_test.extend(w)
#     reg_test.extend(r)

# # Convert Epiweek to month
# from symp_extract.utils import epiweek_to_month

# mt = [epiweek_to_month(w) - 1 for w in wk]
# mt_test = [epiweek_to_month(w) - 1 for w in wk_test]

# # Get HHS adjacency graph
# adj = nx.Graph()
# adj.add_nodes_from(regions)
# from symp_extract.consts import hhs_neighbors

# for i in range(1, len(regions)):
#     adj.add_edges_from([(i, j) for j in hhs_neighbors[i]])

# graph = dgl.from_networkx(adj)
# graph = dgl.add_self_loop(graph)


# Reference points
# pu.db
seq_references = X_ref[:,options.traffic_row:options.traffic_row+1]
# pu.db
symp_references = X_ref
# symp_references = [x[:, :-3] for x in train_seqs]
month_references = np.arange(12)
# reg_references = np.array(regions) - 1.0
if options.smart_mode == 1 or options.smart_mode == 8:
    to_concat = []
    series_here = smoothen(seq_references)
    minimas =  argrelextrema(series_here, np.less)[0]
    derivative = np.diff(series_here[:,0])
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
    if len(series_here) - 1 not in split_idxs:
        split_idxs.append(len(series_here) - 1)
    split_idxs_diffs = [100]+np.diff(split_idxs).tolist()
    split_idxs = np.array(split_idxs)[np.array(split_idxs_diffs) >= 10].tolist()

    
    ci = 0
    # if len(split_idxs) > 3:
    #     split_idxs = split_idxs[:3]
    for si in split_idxs:
        to_append = seq_references[ci:si]
        if len(to_append) > 0:
            to_concat.append(to_append)
        ci = si
    to_concat.append(seq_references)



    max_len = 0
    for tc in to_concat:
        if len(tc) > max_len:
            max_len = len(tc)
    # pu.db
    for t, tc in enumerate(to_concat):
        to_concat[t] = np.expand_dims(np.array([-100 for x in range(max_len-len(tc[:,0]))] + tc[:,0].tolist()), axis=0)
    # pu.db
    seq_references = np.expand_dims(np.concatenate(to_concat), axis=-1)


def preprocess_seq_batch(seq_list: list):
    max_len = max([len(x) for x in seq_list])
    if len(seq_list[0].shape) == 2:
        ans = np.zeros((len(seq_list), max_len, len(seq_list[0][0])))
    else:
        ans = np.zeros((len(seq_list), max_len, 1))
        seq_list = [x[:, np.newaxis] for x in seq_list]
    for i, seq in enumerate(seq_list):
        ans[i, : len(seq), :] = seq
    return ans

seq_references = preprocess_seq_batch(seq_references)
symp_references = preprocess_seq_batch([symp_references])
train_seqs = preprocess_seq_batch(X_train[:,:,options.traffic_row:options.traffic_row+1])
train_y = np.array(Y_train)
train_symp_seqs = preprocess_seq_batch(X_train)
# pu.db
mt = np.array(X_train_time, dtype=np.int32)
mt_test = np.array(X_val_time, dtype=np.int32)
# reg = np.array(reg, dtype=np.int32) - 1
# reg_test = np.array(reg_test, dtype=np.int32) - 1
test_symp_seqs = preprocess_seq_batch(X_val)
test_seqs = preprocess_seq_batch(X_val[:,:,options.traffic_row:options.traffic_row+1])
# test_symp_seqs = preprocess_seq_batch(X_symp_test)
test_y = np.array(Y_val)


month_enc = EmbedEncoder(in_size=75, out_dim=60).to(device)
seq_encoder = GRUEncoder(in_size=1, out_dim=60).to(device)
symp_encoder = GRUEncoder(in_size=862, out_dim=60).to(device)
# reg_encoder = EmbGCNEncoder(
#     in_size=11, emb_dim=60, out_dim=60, num_layers=2, device=device
# ).to(device)

stoch_month_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_seq_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_symp_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_reg_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)

month_corr = CorrEncoder(
    nn_A=options.nn,
    in_data_dim=60,
    in_data_det_dim=60,
    in_ref_dim=60,
    in_ref_det_dim=60,
    hidden_dim=60,
    q_layers=2,
    same_decoder=True,
).to(device)
seq_corr = CorrEncoder(
    nn_A=options.nn,
    in_data_dim=60,
    in_data_det_dim=60,
    in_ref_dim=60,
    in_ref_det_dim=60,
    hidden_dim=60,
    q_layers=2,
    same_decoder=True,
).to(device)
symp_corr = CorrEncoder(
    nn_A=options.nn,
    in_data_dim=60,
    in_data_det_dim=60,
    in_ref_dim=60,
    in_ref_det_dim=60,
    hidden_dim=60,
    q_layers=2,
    same_decoder=True,
).to(device)
reg_corr = CorrEncoder(
    nn_A=options.nn,
    in_data_dim=60,
    in_data_det_dim=60,
    in_ref_dim=60,
    in_ref_det_dim=60,
    hidden_dim=60,
    q_layers=2,
    same_decoder=True,
).to(device)

decoder = Decoder(z_dim=60, sr_dim=60, latent_dim=60, hidden_dim=60, y_dim=1).to(device)

models = [
    month_enc,
    seq_encoder,
    symp_encoder,
    stoch_month_enc,
    stoch_seq_enc,
    stoch_symp_enc,
    stoch_reg_enc,
    month_corr,
    seq_corr,
    symp_corr,
    reg_corr,
    decoder,
]

opt = optim.Adam(
    reduce(lambda x, y: x + y, [list(m.parameters()) for m in models]), lr=1e-3
)

# Porbabilistic encode of reference points
ref_months = month_enc.forward(long_tensor(month_references))
ref_seq = seq_encoder.forward(float_tensor(seq_references))
ref_symp = symp_encoder.forward(float_tensor(symp_references))
# ref_reg = reg_encoder.forward(long_tensor(reg_references), graph.to(device))

stoch_ref_months = stoch_month_enc.forward(ref_months)[0]
stoch_ref_seq = stoch_seq_enc.forward(ref_seq)[0]
stoch_ref_symp = stoch_symp_enc.forward(ref_symp)[0]
# stoch_ref_reg = stoch_reg_enc.forward(ref_reg)[0]

# Probabilistic encode of training points

# pu.db
train_months = month_enc.forward(long_tensor(mt[:,0].astype(int)))
train_seq = seq_encoder.forward(float_tensor(train_seqs))
train_symp = symp_encoder.forward(float_tensor(train_symp_seqs))
# train_reg = torch.stack([ref_reg[i] for i in mt], dim=0)

stoch_train_months = stoch_month_enc.forward(train_months)[0]
stoch_train_seq = stoch_seq_enc.forward(train_seq)[0]
stoch_train_symp = stoch_symp_enc.forward(train_symp)[0]
# stoch_train_reg = stoch_reg_enc.forward(train_reg)[0]


def train(train_symp_seqs, mt, train_y):
    for m in models:
        m.train()
    opt.zero_grad()

    # Porbabilistic encode of reference points
    pu.db
    ref_months = month_enc.forward(long_tensor(month_references))
    ref_seq = seq_encoder.forward(float_tensor(seq_references))
    ref_symp = symp_encoder.forward(float_tensor(symp_references))
    # ref_reg = reg_encoder.forward(long_tensor(reg_references), graph.to(device))

    stoch_ref_months = stoch_month_enc.forward(ref_months)[0]
    stoch_ref_seq = stoch_seq_enc.forward(ref_seq)[0]
    stoch_ref_symp = stoch_symp_enc.forward(ref_symp)[0]
    # stoch_ref_reg = stoch_reg_enc.forward(ref_reg)[0]

    # Probabilistic encode of training points

    train_months = month_enc.forward(long_tensor(mt[:,0].astype(int)))
    train_seq = seq_encoder.forward(float_tensor(train_seqs))
    train_symp = symp_encoder.forward(float_tensor(train_symp_seqs))
    # train_reg = torch.stack([ref_reg[i] for i in reg], dim=0)

    stoch_train_months = stoch_month_enc.forward(train_months)[0]
    stoch_train_seq = stoch_seq_enc.forward(train_seq)[0]
    stoch_train_symp = stoch_symp_enc.forward(train_symp)[0]
    # stoch_train_reg = stoch_reg_enc.forward(train_reg)[0]
    # Get view-aware latent embeddings
    train_months_z, train_month_sr, _, month_loss, _ = month_corr.forward(
        stoch_ref_months, stoch_train_months, ref_months, train_months
    )

    train_seq_z, train_seq_sr, _, seq_loss, _ = seq_corr.forward(
        stoch_ref_seq, stoch_train_seq, ref_seq, train_seq
    )

    train_symp_z, train_symp_sr, _, symp_loss, _ = symp_corr.forward(
        stoch_ref_symp, stoch_train_symp, ref_symp, train_symp
    )


    # train_seq_z, train_seq_sr, _, seq_loss, _ = seq_corr.forward(
    #     stoch_ref_seq, stoch_train_seq, ref_seq, train_seq
    # )
    # train_symp_z, train_symp_sr, _, symp_loss, _ = symp_corr.forward(
    #     stoch_ref_symp, stoch_train_symp, ref_symp, train_symp
    # )
    # train_reg_z, train_reg_sr, _, reg_loss, _ = reg_corr.forward(
    #     stoch_ref_reg, stoch_train_reg, ref_reg, train_reg
    # )

    # Concat all latent embeddings
    train_z = torch.stack(
        [train_months_z, train_seq_z, train_symp_z], dim=1
    )
    train_sr = torch.stack(
        [train_month_sr, train_seq_sr, train_symp_sr], dim=1
    )

    loss, mean_y, _, _ = decoder.forward(
        train_z, train_sr, train_seq, float_tensor(train_y)[:, None]
    )

    losses = month_loss + seq_loss + symp_loss + loss
    losses.backward()
    opt.step()
    # print(f"Loss = {loss.detach().cpu().numpy()}")

    return (
        mean_y.detach().cpu().numpy(),
        losses.detach().cpu().numpy(),
        loss.detach().cpu().numpy(),
    )


def evaluate(test_symp_seqs, mt_test, test_y, sample=True):
    for m in models:
        m.eval()
    # Porbabilistic encode of reference points
    ref_months = month_enc.forward(long_tensor(month_references))
    ref_seq = seq_encoder.forward(float_tensor(seq_references))
    ref_symp = symp_encoder.forward(float_tensor(symp_references))
    # ref_reg = reg_encoder.forward(long_tensor(reg_references), graph.to(device))

    stoch_ref_months = stoch_month_enc.forward(ref_months)[0]
    stoch_ref_seq = stoch_seq_enc.forward(ref_seq)[0]
    stoch_ref_symp = stoch_symp_enc.forward(ref_symp)[0]
    # stoch_ref_reg = stoch_reg_enc.forward(ref_reg)[0]

    # Probabilistic encode of test points
    # pu.db
    test_months = month_enc.forward(long_tensor(mt_test[:,0].astype(int)))
    test_seq = seq_encoder.forward(float_tensor(test_seqs))
    test_symp = symp_encoder.forward(float_tensor(test_symp_seqs))
    # test_reg = torch.stack([ref_reg[i] for i in reg_test], dim=0)

    stoch_test_months = stoch_month_enc.forward(test_months)[0]
    stoch_test_seq = stoch_seq_enc.forward(test_seq)[0]
    stoch_test_symp = stoch_symp_enc.forward(test_symp)[0]
    # stoch_test_reg = stoch_reg_enc.forward(test_reg)[0]
    # Get view-aware latent embeddings
    test_months_z, test_month_sr, _, _, _, _ = month_corr.predict(
        stoch_ref_months, stoch_test_months, ref_months, test_months
    )

    test_seq_z, test_seq_sr, _, seq_loss, _ = seq_corr.forward(
        stoch_ref_seq, stoch_test_seq, ref_seq, test_seq
    )

    test_symp_z, test_symp_sr, _, _, _, _ = symp_corr.predict(
        stoch_ref_symp, stoch_test_symp, ref_symp, test_symp
    )


    # test_seq_z, test_symp_sr, _, _, _, _ = seq_corr.predict(
    #     stoch_ref_seq, stoch_test_symp, ref_seq, test_seq
    # )
    # test_symp_z, test_symp_sr, _, _, _, _ = symp_corr.predict(
    #     stoch_ref_symp, stoch_test_symp, ref_symp, test_symp
    # )
    # test_reg_z, test_reg_sr, _, _, _, _ = reg_corr.predict(
    #     stoch_ref_reg, stoch_test_reg, ref_reg, test_reg
    # )

    # Concat all latent embeddings
    test_z = torch.stack([test_months_z, test_seq_z, test_symp_z], dim=1)
    test_sr = torch.stack(
        [test_month_sr, test_seq_sr, test_symp_sr], dim=1
    )

    sample_y, mean_y, _, _ = decoder.predict(
        test_z, test_sr, test_seq, sample=sample
    )
    sample_y = sample_y.detach().cpu().numpy().ravel()
    mean_y = mean_y.detach().cpu().numpy().ravel()
    # RMSE loss
    rmse = np.sqrt(np.mean((sample_y - test_y.ravel()) ** 2))
    # Mean absolute error
    # mae = np.mean(np.abs(sample_y - test_y))

    print(f"RMSE = {rmse}")
    return rmse, sample_y, test_y

all_results = {}
min_loss = np.inf
counter = 0
for ep in tqdm(range(1, epochs + 1)):
    # pu.db
    # if options.smart_mode == 1:
    #     idxs = random.sample(range(len(train_symp_seqs)), 1024)
    # else:
    #     idxs = random.sample(range(len(train_symp_seqs)), 8192)
    # train_symp_seqs_here = train_symp_seqs[idxs]
    # train_y_here = train_y[idxs]
    # mt_here = mt[idxs]


    
    _,_, loss = train(train_symp_seqs, mt, train_y)

    print("Loss: "+str(loss))
    if loss < min_loss:
        min_loss = loss
        counter = 0
    else:
        counter += 1
        print("Patience counter: "+str(counter)+"/50")

    if counter >= 50:
        break

    if ep % 10 == 0:
        print("Evaluating")
        rmse_here, yp, yt = evaluate(test_symp_seqs, mt_test, test_y)
        all_results[ep] = {"rmse": rmse_here, "pred": yp, "gt": yt}

os.makedirs(f"/localscratch/ssinha97/fnp_evaluations/traffic_val_predictions_normal", exist_ok=True)
with open(f"/localscratch/ssinha97/fnp_evaluations/traffic_val_predictions_normal/"+str(options.save_model)+"_predictions.pkl", "wb") as f:
    pickle.dump(all_results, f)
print("Saved val data at "+"/localscratch/ssinha97/fnp_evaluations/traffic_val_predictions_normal/"+str(options.save_model)+"_predictions.pkl")
