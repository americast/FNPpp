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
    Combine
)
from tqdm import tqdm
from scipy.signal import argrelextrema
from scipy.fft import fft

import pudb
parser = OptionParser()
parser.add_option("-w", "--week", dest="week_ahead", type="int", default=2)
parser.add_option("-y", "--year", dest="year", type="int", default=2020)
parser.add_option("-n", "--num", dest="num", type="string")
parser.add_option("-e", "--epoch", dest="epochs", type="int", default="1500")
parser.add_option("-m", "--save", dest="save_model", default="default", type="string")
parser.add_option("--seed", dest="seed", default=0, type="int")
parser.add_option("--smart-mode", dest="smart_mode", default=0, type="int")
parser.add_option("--nn", dest="nn", default="none", type="choice", choices=["none", "simple", "bn", "dot", "bert"])
parser.add_option("--optionals", dest="optionals", default=" ", type="str")


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

with open("./data/symptom_data/saves/combine.pkl", "rb") as f:
    data = pickle.load(f)

# Make sequence for all years
def get_sequence(data: pd.DataFrame, hhs: int, year: int) -> np.ndarray:
    d1 = data[(data.year == year - 1) & (data.hhs == hhs) & (data.epiweek > 20)][
        include_cols + ["ili", "epiweek", "hhs"]
    ]
    d2 = data[(data.year == year) & (data.hhs == hhs) & (data.epiweek <= 20)][
        include_cols + ["ili", "epiweek", "hhs"]
    ]
    d1 = np.array(d1)
    d2 = np.array(d2)
    print(len(d1), len(d2))
    return np.vstack((d1, d2))


train_years = [y for y in range(start_year, year)]
test_years = [year]

train_seqs = [get_sequence(data, hhs, year) for hhs in regions for year in train_years]
test_seqs = [get_sequence(data, hhs, year) for hhs in regions for year in test_years]


def seq_to_dataset(seq, week_ahead=week_ahead):
    X, Y, wk, reg = [], [], [], []
    start_idx = max(week_ahead, seq.shape[0] - 32 + week_ahead)
    for i in range(start_idx, seq.shape[0]):
        X.append(seq[: i - week_ahead + 1, :-2])
        Y.append(seq[i, -3])
        wk.append(seq[i, -2])
        reg.append(seq[i, -1])
    return X, Y, wk, reg

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


train_dataset = [seq_to_dataset(seq, week_ahead) for seq in train_seqs]
X, X_smart, X_symp, X_symp_smart, Y, wk, reg = [], [], [], [], [], [], []
for x, y, w, r in train_dataset:
    if options.smart_mode == 5:
        X.extend([smoothen(l[:, -1]) for l in x])
        X_symp.extend([smoothen(l[:, :-1]) for l in x])
    else:
        X.extend([l[:, -1] for l in x])
        X_symp.extend([l[:, :-1] for l in x])
        X_smart.extend([smoothen(l[:, -1]) for l in x])
        X_symp_smart.extend([smoothen(l[:, :-1]) for l in x])
    Y.extend(y)
    wk.extend(w)
    reg.extend(r)
test_dataset = [seq_to_dataset(seq, week_ahead) for seq in test_seqs]
X_test, X_test_smart, X_symp_test, X_symp_test_smart, Y_test, wk_test, reg_test = [], [], [], [], [], [], []
for x, y, w, r in test_dataset:
    if options.smart_mode == 5:
        X_test.extend([smoothen(l[:, -1]) for l in x])
        X_symp_test.extend([smoothen(l[:, :-1]) for l in x])
    else:
        X_test.extend([l[:, -1] for l in x])
        X_symp_test.extend([l[:, :-1] for l in x])
        X_test_smart.extend([smoothen(l[:, -1]) for l in x])
        X_symp_test_smart.extend([smoothen(l[:, :-1]) for l in x])
    Y_test.extend(y)
    wk_test.extend(w)
    reg_test.extend(r)
# Convert Epiweek to month
from symp_extract.utils import epiweek_to_month

mt = [epiweek_to_month(w) - 1 for w in wk]
mt_test = [epiweek_to_month(w) - 1 for w in wk_test]

# Get HHS adjacency graph
adj = nx.Graph()
adj.add_nodes_from(regions)
from symp_extract.consts import hhs_neighbors

for i in range(1, len(regions)):
    adj.add_edges_from([(i, j) for j in hhs_neighbors[i]])

graph = dgl.from_networkx(adj)
graph = dgl.add_self_loop(graph)


# Reference points
seq_references = [x[:, -3] for x in train_seqs]
# pu.db
symp_references = [x[:, :-3] for x in train_seqs]
month_references = np.arange(12)
reg_references = np.array(regions) - 1.0
if options.smart_mode == 1 or options.smart_mode == 8:
    to_concat = []
    for i in range(len(seq_references)):
        series_here = seq_references[i]
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
        for si in split_idxs:
            to_append = seq_references[i][ci:si]
            if len(to_append) > 0:
                to_concat.append(to_append)
            ci = si
        to_concat.append(seq_references[i])
    # pu.db
    seq_references = to_concat

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
symp_references = preprocess_seq_batch(symp_references)
train_seqs = preprocess_seq_batch(X)
train_seqs_smart = preprocess_seq_batch(X_smart)
train_y = np.array(Y)
train_symp_seqs = preprocess_seq_batch(X_symp)
train_symp_seqs_smart = preprocess_seq_batch(X_symp_smart)
mt = np.array(mt, dtype=np.int32)
mt_test = np.array(mt_test, dtype=np.int32)
reg = np.array(reg, dtype=np.int32) - 1
reg_test = np.array(reg_test, dtype=np.int32) - 1
test_seqs = preprocess_seq_batch(X_test)
test_seqs_smart = preprocess_seq_batch(X_test_smart)
test_symp_seqs = preprocess_seq_batch(X_symp_test)
test_symp_seqs_smart = preprocess_seq_batch(X_symp_test_smart)
test_y = np.array(Y_test)

month_enc = EmbedEncoder(in_size=12, out_dim=60).to(device)
seq_encoder = GRUEncoder(in_size=1, out_dim=60).to(device)
symp_encoder = GRUEncoder(in_size=14, out_dim=60).to(device)
reg_encoder = EmbGCNEncoder(
    in_size=11, emb_dim=60, out_dim=60, num_layers=2, device=device
).to(device)

stoch_month_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_seq_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_symp_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_reg_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)

if "fft" in options.optionals:
    seq_encoder_fft = GRUEncoder(in_size=1, out_dim=60).to(device)
    symp_encoder_fft = GRUEncoder(in_size=14, out_dim=60).to(device)
    stoch_seq_enc_fft = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
    stoch_symp_enc_fft = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)

    seq_corr_fft = CorrEncoder(
        nn_A=options.nn,
        in_data_dim=60,
        in_data_det_dim=60,
        in_ref_dim=60,
        in_ref_det_dim=60,
        hidden_dim=60,
        q_layers=2,
        same_decoder=True,
    ).to(device)
    symp_corr_fft = CorrEncoder(
        nn_A=options.nn,
        in_data_dim=60,
        in_data_det_dim=60,
        in_ref_dim=60,
        in_ref_det_dim=60,
        hidden_dim=60,
        q_layers=2,
        same_decoder=True,
    ).to(device)

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
    reg_encoder,
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
# hidden_size_combine = 102
# if week_ahead == 2:
#     hidden_size_combine = 100
# if week_ahead == 3:
#     hidden_size_combine = 98
if "combine" in options.optionals:
    combine_seq = Combine(1).to(device)
    combine_symp = Combine(len(include_cols)).to(device)
    models += [
        combine_seq,
        combine_symp,
    ]
if "fft" in options.optionals:
    models += [
        seq_encoder_fft,
        symp_encoder_fft,
        stoch_seq_enc_fft,
        stoch_symp_enc_fft,
        seq_corr_fft,
        symp_corr_fft
    ]


opt = optim.Adam(
    reduce(lambda x, y: x + y, [list(m.parameters()) for m in models]), lr=1e-3
)

# Porbabilistic encode of reference points
# ref_months = month_enc.forward(long_tensor(month_references))
# ref_seq = seq_encoder.forward(float_tensor(seq_references))
# ref_symp = symp_encoder.forward(float_tensor(symp_references))
# ref_reg = reg_encoder.forward(long_tensor(reg_references), graph.to(device))

# stoch_ref_months = stoch_month_enc.forward(ref_months)[0]
# stoch_ref_seq = stoch_seq_enc.forward(ref_seq)[0]
# stoch_ref_symp = stoch_symp_enc.forward(ref_symp)[0]
# stoch_ref_reg = stoch_reg_enc.forward(ref_reg)[0]

# # Probabilistic encode of training points

# train_months = month_enc.forward(long_tensor(mt.astype(int)))
# train_seq = seq_encoder.forward(float_tensor(train_seqs))
# train_symp = symp_encoder.forward(float_tensor(train_symp_seqs))
# train_reg = torch.stack([ref_reg[i] for i in mt], dim=0)

# stoch_train_months = stoch_month_enc.forward(train_months)[0]
# stoch_train_seq = stoch_seq_enc.forward(train_seq)[0]
# stoch_train_symp = stoch_symp_enc.forward(train_symp)[0]
# stoch_train_reg = stoch_reg_enc.forward(train_reg)[0]

# if "fft" in options.optionals:
#     seq_references = np.concatenate([seq_references, fft(seq_references).real, fft(seq_references).imag], axis=0)
#     symp_references = np.concatenate([symp_references, fft(symp_references).real, fft(symp_references).imag], axis=0)

def train(train_seqs, train_symp_seqs, reg, mt, train_y):
    for m in models:
        m.train()
    opt.zero_grad()

    # Porbabilistic encode of reference points
    ref_months = month_enc.forward(long_tensor(month_references))
    ref_seq = seq_encoder.forward(float_tensor(seq_references))
    ref_symp = symp_encoder.forward(float_tensor(symp_references))
    ref_reg = reg_encoder.forward(long_tensor(reg_references), graph.to(device))

    stoch_ref_months = stoch_month_enc.forward(ref_months)[0]
    stoch_ref_seq = stoch_seq_enc.forward(ref_seq)[0]
    stoch_ref_symp = stoch_symp_enc.forward(ref_symp)[0]
    stoch_ref_reg = stoch_reg_enc.forward(ref_reg)[0]


    # Probabilistic encode of training points
    train_months = month_enc.forward(long_tensor(mt.astype(int)))
    if "combine" in options.optionals:
        combined_seq = combine_seq(float_tensor(train_seqs), float_tensor(train_seqs_smart))
        combined_symp = combine_symp(float_tensor(train_symp_seqs), float_tensor(train_symp_seqs_smart))
    else:
        combined_seq = float_tensor(train_seqs)
        combined_symp = float_tensor(train_symp_seqs)
    train_seq = seq_encoder.forward(combined_seq)
    train_symp = symp_encoder.forward(combined_symp)
    train_reg = torch.stack([ref_reg[i] for i in reg], dim=0)

    stoch_train_months = stoch_month_enc.forward(train_months)[0]
    stoch_train_seq = stoch_seq_enc.forward(train_seq)[0]
    stoch_train_symp = stoch_symp_enc.forward(train_symp)[0]
    stoch_train_reg = stoch_reg_enc.forward(train_reg)[0]


    if "fft" in options.optionals:
        combined_seq_fft = torch.fft.fft(combined_seq).real
        train_seq_fft = seq_encoder_fft.forward(combined_seq_fft)
        combined_symp_fft = torch.fft.fft(combined_symp).real
        train_symp_fft = symp_encoder_fft.forward(combined_symp_fft)
        train_reg = torch.stack([ref_reg[i] for i in reg], dim=0)

        stoch_train_seq_fft = stoch_seq_enc_fft.forward(train_seq_fft)[0]
        stoch_train_symp_fft = stoch_symp_enc_fft.forward(train_symp_fft)[0]
        if "ref" in options.optionals:
            ref_seq_fft = seq_encoder_fft.forward(torch.fft.fft(float_tensor(seq_references)).real)
            ref_symp_fft = symp_encoder_fft.forward(torch.fft.fft(float_tensor(symp_references)).real)
        else:
            ref_seq_fft = seq_encoder_fft.forward(float_tensor(seq_references))
            ref_symp_fft = symp_encoder_fft.forward(float_tensor(symp_references))
        stoch_ref_seq_fft = stoch_seq_enc_fft.forward(ref_seq_fft)[0]
        stoch_ref_symp_fft = stoch_symp_enc_fft.forward(ref_symp_fft)[0]

        # Get view-aware latent embeddings
        train_seq_z_fft, train_seq_sr_fft, _, seq_loss_fft, _ = seq_corr_fft.forward(
            stoch_ref_seq_fft, stoch_train_seq_fft, ref_seq_fft, train_seq_fft
        )

        train_symp_z_fft, train_symp_sr_fft, _, symp_loss_fft, _ = symp_corr_fft.forward(
            stoch_ref_symp_fft, stoch_train_symp_fft, ref_symp_fft, train_symp_fft
        )



    # Get view-aware latent embeddings
    train_months_z, train_month_sr, _, month_loss, _ = month_corr.forward(
        stoch_ref_months, stoch_train_months, ref_months, train_months
    )

    train_seq_z, train_seq_sr, _, seq_loss, _ = seq_corr.forward(
        stoch_ref_seq, stoch_train_seq, ref_seq, train_seq
    )
    # pu.db
    train_symp_z, train_symp_sr, _, symp_loss, _ = symp_corr.forward(
        stoch_ref_symp, stoch_train_symp, ref_symp, train_symp
    )
    train_reg_z, train_reg_sr, _, reg_loss, _ = reg_corr.forward(
        stoch_ref_reg, stoch_train_reg, ref_reg, train_reg
    )

    # Concat all latent embeddings
    if "epi" in options.optionals:
        train_z = torch.stack(
            [train_seq_z, train_symp_z], dim=1
        )
        train_sr = torch.stack(
            [train_seq_sr, train_symp_sr], dim=1
        )

    elif "fft" in options.optionals:
        train_z = torch.stack(
            [train_months_z, train_seq_z, train_seq_z_fft, train_symp_z, train_symp_z_fft, train_reg_z], dim=1
        )
        train_sr = torch.stack(
            [train_month_sr, train_seq_sr, train_seq_sr_fft, train_symp_sr, train_symp_sr_fft, train_reg_sr], dim=1
        )
    else:
        train_z = torch.stack(
            [train_months_z, train_seq_z, train_symp_z, train_reg_z], dim=1
        )
        train_sr = torch.stack(
            [train_month_sr, train_seq_sr, train_symp_sr, train_reg_sr], dim=1
        )

    loss, mean_y, _, _ = decoder.forward(
        train_z, train_sr, train_seq, float_tensor(train_y)[:, None]
    )
    if "epi" in options.optionals:
        losses = seq_loss + symp_loss + loss
    else:
        losses = month_loss + seq_loss + symp_loss + reg_loss + loss
    if "fft" in options.optionals:
        losses += seq_loss_fft + symp_loss_fft

    losses.backward()
    opt.step()
    # print(f"Loss = {loss.detach().cpu().numpy()}")
    mean_y = mean_y.cpu().detach().numpy()
    return (
        (((mean_y - train_y) ** 2).mean()) ** 0.5
    )


def evaluate(test_seqs, test_symp_seqs, reg_test, mt_test, test_y, sample=True):
    for m in models:
        m.eval()
    # Porbabilistic encode of reference points
    ref_months = month_enc.forward(long_tensor(month_references))
    ref_seq = seq_encoder.forward(float_tensor(seq_references))
    ref_symp = symp_encoder.forward(float_tensor(symp_references))
    ref_reg = reg_encoder.forward(long_tensor(reg_references), graph.to(device))

    stoch_ref_months = stoch_month_enc.forward(ref_months)[0]
    stoch_ref_seq = stoch_seq_enc.forward(ref_seq)[0]
    stoch_ref_symp = stoch_symp_enc.forward(ref_symp)[0]
    stoch_ref_reg = stoch_reg_enc.forward(ref_reg)[0]

    # Probabilistic encode of test points

    test_months = month_enc.forward(long_tensor(mt_test.astype(int)))
    if "combine" in options.optionals:
        combined_seq = combine_seq(float_tensor(test_seqs), float_tensor(test_seqs_smart))
        combined_symp = combine_symp(float_tensor(test_symp_seqs), float_tensor(test_symp_seqs_smart))
    else:
        combined_seq = float_tensor(test_seqs)
        combined_symp = float_tensor(test_symp_seqs)
    test_seq = seq_encoder.forward(combined_seq)
    test_symp = symp_encoder.forward(combined_symp)
    test_reg = torch.stack([ref_reg[i] for i in reg_test], dim=0)

    stoch_test_months = stoch_month_enc.forward(test_months)[0]
    stoch_test_seq = stoch_seq_enc.forward(test_seq)[0]
    stoch_test_symp = stoch_symp_enc.forward(test_symp)[0]
    stoch_test_reg = stoch_reg_enc.forward(test_reg)[0]

    if "fft" in options.optionals:
        combined_seq_fft = torch.fft.fft(combined_seq).real
        test_seq_fft = seq_encoder_fft.forward(combined_seq_fft)
        combined_symp_fft = torch.fft.fft(combined_symp).real
        test_symp_fft = symp_encoder_fft.forward(combined_symp_fft)
        test_reg = torch.stack([ref_reg[i] for i in reg], dim=0)

        stoch_test_seq_fft = stoch_seq_enc_fft.forward(test_seq_fft)[0]
        stoch_test_symp_fft = stoch_symp_enc_fft.forward(test_symp_fft)[0]
        if "ref" in options.optionals:
            ref_seq_fft = seq_encoder_fft.forward(torch.fft.fft(float_tensor(seq_references)).real)
            ref_symp_fft = symp_encoder_fft.forward(torch.fft.fft(float_tensor(symp_references)).real)
        else:
            ref_seq_fft = seq_encoder_fft.forward(float_tensor(seq_references))
            ref_symp_fft = symp_encoder_fft.forward(float_tensor(symp_references))
        stoch_ref_seq_fft = stoch_seq_enc_fft.forward(ref_seq_fft)[0]
        stoch_ref_symp_fft = stoch_symp_enc_fft.forward(ref_symp_fft)[0]

        # Get view-aware latent embeddings
        test_seq_z_fft, test_seq_sr_fft, _, seq_loss_fft, _ = seq_corr_fft.forward(
            stoch_ref_seq_fft, stoch_test_seq_fft, ref_seq_fft, test_seq_fft
        )

        test_symp_z_fft, test_symp_sr_fft, _, symp_loss_fft, _ = symp_corr_fft.forward(
            stoch_ref_symp_fft, stoch_test_symp_fft, ref_symp_fft, test_symp_fft
        )

    # Get view-aware latent embeddings
    test_months_z, test_month_sr, _, _, _, _ = month_corr.predict(
        stoch_ref_months, stoch_test_months, ref_months, test_months
    )
    test_seq_z, test_seq_sr, _, _, _, _ = seq_corr.predict(
        stoch_ref_seq, stoch_test_seq, ref_seq, test_seq
    )
    test_symp_z, test_symp_sr, _, _, _, _ = symp_corr.predict(
        stoch_ref_symp, stoch_test_symp, ref_symp, test_symp
    )
    test_reg_z, test_reg_sr, _, _, _, _ = reg_corr.predict(
        stoch_ref_reg, stoch_test_reg, ref_reg, test_reg
    )

    # Concat all latent embeddings
    if "epi" in options.optionals:
        test_z = torch.stack(
            [test_seq_z, test_symp_z], dim=1
        )
        test_sr = torch.stack(
            [test_seq_sr, test_symp_sr], dim=1
        )
    elif "fft" in options.optionals:
        test_z = torch.stack(
            [test_months_z, test_seq_z, test_seq_z_fft, test_symp_z, test_symp_z_fft, test_reg_z], dim=1
        )
        test_sr = torch.stack(
            [test_month_sr, test_seq_sr, test_seq_sr_fft, test_symp_sr, test_symp_sr_fft, test_reg_sr], dim=1
        )
    else:
        test_z = torch.stack(
            [test_months_z, test_seq_z, test_symp_z, test_reg_z], dim=1
        )
        test_sr = torch.stack(
            [test_month_sr, test_seq_sr, test_symp_sr, test_reg_sr], dim=1
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
for ep in tqdm(range(1, epochs + 1)):
    train_err = train(train_seqs, train_symp_seqs, reg, mt, train_y)
    if ep % 10 == 0:
        print("Evaluating")
        rmse_here, yp, yt = evaluate(test_seqs, test_symp_seqs, reg_test, mt_test, test_y)
        all_results[ep] = {"rmse": rmse_here, "pred": yp, "gt": yt, "train_err": train_err}

if options.optionals != " ":
    options.save_model = options.save_model+"_optionals_"+options.optionals

os.makedirs(f"/localscratch/ssinha97/fnp_evaluations/symp_val_predictions_normal", exist_ok=True)
with open(f"/localscratch/ssinha97/fnp_evaluations/symp_val_predictions_normal/"+str(options.save_model)+"_predictions.pkl", "wb") as f:
    pickle.dump(all_results, f)
print("Saved val data at "+"/localscratch/ssinha97/fnp_evaluations/symp_val_predictions_normal/"+str(options.save_model)+"_predictions.pkl")
