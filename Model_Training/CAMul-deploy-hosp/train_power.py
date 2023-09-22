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
from eval_metrics import crps_samples

parser = OptionParser()
parser.add_option("-w", "--ahead", dest="week_ahead", type="int", default=1)
parser.add_option("--seed", dest="seed", type="int", default=0)
parser.add_option("--optionals", dest="optionals", type="str", default=" ")
(options, args) = parser.parse_args()
ahead = options.week_ahead
seed = options.seed
np.random.seed(seed)
torch.manual_seed(seed)

import random
random.seed(seed)

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
features_smart = smoothen(features)
target = features[:, 0]
target_smart = features_smart[:, 0]

total_time = len(data)
test_start = int(total_time * 0.8)

X, X_symp, Y, mt, reg = [], [], [], [], []

def sample_train(n_samples, window = 20):
    X, X_smart, X_symp, X_symp_smart, Y, mt, reg = [], [], [], [], [], [], []
    start_seqs = np.random.randint(0, test_start - ahead - window, n_samples)
    for start_seq in start_seqs:
        X.append(target[start_seq:start_seq+window, np.newaxis])
        X_smart.append(target_smart[start_seq:start_seq+window, np.newaxis])
        X_symp.append(features[start_seq:start_seq+window])
        X_symp_smart.append(features_smart[start_seq:start_seq+window])
        Y.append(target[start_seq+window+ahead-1])
        mt.append(month[start_seq+window])
        reg.append(tod[start_seq+window])
    X = np.array(X)
    X_smart = np.array(X_smart)
    X_symp = np.array(X_symp)
    X_symp_smart = np.array(X_symp_smart)
    Y = np.array(Y)
    mt = np.array(mt)
    reg = np.array(reg)
    return X, X_smart, X_symp, X_symp_smart, Y, mt, reg

def sample_test(n_samples, window = 20):
    X, X_smart, X_symp, X_symp_smart, Y, mt, reg = [], [], [], [], [], [], []
    start_seqs = np.random.randint(test_start, total_time - ahead - window, n_samples)
    for start_seq in start_seqs:
        X.append(target[start_seq:start_seq+window, np.newaxis])
        X_smart.append(target_smart[start_seq:start_seq+window, np.newaxis])
        X_symp.append(features[start_seq:start_seq+window])
        X_symp_smart.append(features_smart[start_seq:start_seq+window])
        Y.append(target[start_seq+window+ahead-1])
        mt.append(month[start_seq+window])
        reg.append(tod[start_seq+window])
    X = np.array(X)
    X_smart = np.array(X_smart)
    X_symp = np.array(X_symp)
    X_symp_smart = np.array(X_symp_smart)
    Y = np.array(Y)
    mt = np.array(mt)
    reg = np.array(reg)
    return X, X_smart, X_symp, X_symp_smart, Y, mt, reg




# Reference points
splits = 10
len_seq = test_start//splits
seq_references = np.array([features[i: i+len_seq, 0, np.newaxis] for i in range(0, test_start, len_seq)])[:, :100, :]
symp_references = np.array([features[i: i+len_seq] for i in range(0, test_start, len_seq)])[:, :100, :]
month_references = np.arange(12)
reg_references = np.arange(4)

train_seqs, train_seqs_smart, train_symp_seqs, train_symp_seqs_smart, train_y, mt, reg = sample_train(100)




month_enc = EmbedEncoder(in_size=12, out_dim=60).to(device)
seq_encoder = GRUEncoder(in_size=1, out_dim=60).to(device)
symp_encoder = GRUEncoder(in_size=7, out_dim=60).to(device)
reg_encoder = EmbedEncoder(in_size=5, out_dim=60).to(device)

stoch_month_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_seq_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_symp_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
stoch_reg_enc = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)

if "fft" in options.optionals:
    seq_encoder_fft = GRUEncoder(in_size=1, out_dim=60).to(device)
    symp_encoder_fft = GRUEncoder(in_size=7, out_dim=60).to(device)
    stoch_seq_enc_fft = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
    stoch_symp_enc_fft = LatentEncoder(in_dim=60, hidden_layers=[60], out_dim=60).to(device)
    seq_corr_fft = CorrEncoder(
        in_data_dim=60,
        in_data_det_dim=60,
        in_ref_dim=60,
        in_ref_det_dim=60,
        hidden_dim=60,
        q_layers=2,
        same_decoder=True,
    ).to(device)
    symp_corr_fft = CorrEncoder(
        in_data_dim=60,
        in_data_det_dim=60,
        in_ref_dim=60,
        in_ref_det_dim=60,
        hidden_dim=60,
        q_layers=2,
        same_decoder=True,
    ).to(device)


month_corr = CorrEncoder(
    in_data_dim=60,
    in_data_det_dim=60,
    in_ref_dim=60,
    in_ref_det_dim=60,
    hidden_dim=60,
    q_layers=2,
    same_decoder=True,
).to(device)
seq_corr = CorrEncoder(
    in_data_dim=60,
    in_data_det_dim=60,
    in_ref_dim=60,
    in_ref_det_dim=60,
    hidden_dim=60,
    q_layers=2,
    same_decoder=True,
).to(device)
symp_corr = CorrEncoder(
    in_data_dim=60,
    in_data_det_dim=60,
    in_ref_dim=60,
    in_ref_det_dim=60,
    hidden_dim=60,
    q_layers=2,
    same_decoder=True,
).to(device)
reg_corr = CorrEncoder(
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

if "combine" in options.optionals:
    combine_seq = Combine(1).to(device)
    combine_symp = Combine(7).to(device)
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
ref_months = month_enc.forward(long_tensor(month_references))
ref_seq = seq_encoder.forward(float_tensor(seq_references))
ref_symp = symp_encoder.forward(float_tensor(symp_references))
ref_reg = reg_encoder.forward(long_tensor(reg_references))

stoch_ref_months = stoch_month_enc.forward(ref_months)[0]
stoch_ref_seq = stoch_seq_enc.forward(ref_seq)[0]
stoch_ref_symp = stoch_symp_enc.forward(ref_symp)[0]
stoch_ref_reg = stoch_reg_enc.forward(ref_reg)[0]

# Probabilistic encode of training points

train_months = month_enc.forward(long_tensor(mt.astype(int)))
train_seq = seq_encoder.forward(float_tensor(train_seqs))
train_symp = symp_encoder.forward(float_tensor(train_symp_seqs))
train_reg = reg_encoder.forward(long_tensor(reg.astype(int)))

stoch_train_months = stoch_month_enc.forward(train_months)[0]
stoch_train_seq = stoch_seq_enc.forward(train_seq)[0]
stoch_train_symp = stoch_symp_enc.forward(train_symp)[0]
stoch_train_reg = stoch_reg_enc.forward(train_reg)[0]
A_seq_ffts, A_symp_ffts, A_seqs, A_symps = None,None,None,None
A_seq_ffts_test, A_symp_ffts_test, A_seqs_test, A_symps_test = None,None,None,None
def train(train_seqs, train_seqs_smart, train_symp_seqs, train_symp_seqs_smart, reg, mt, train_y):
    global A_seq_ffts, A_symp_ffts, A_seqs, A_symps
    for m in models:
        m.train()
    opt.zero_grad()

    # Porbabilistic encode of reference points
    ref_months = month_enc.forward(long_tensor(month_references))
    ref_seq = seq_encoder.forward(float_tensor(seq_references))
    ref_symp = symp_encoder.forward(float_tensor(symp_references))
    ref_reg = reg_encoder.forward(long_tensor(reg_references))

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

    train_reg = reg_encoder.forward(long_tensor(reg.astype(int)))

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
        train_seq_z_fft, train_seq_sr_fft, _, seq_loss_fft, A_seq_ffts = seq_corr_fft.forward(
            stoch_ref_seq_fft, stoch_train_seq_fft, ref_seq_fft, train_seq_fft
        )

        train_symp_z_fft, train_symp_sr_fft, _, symp_loss_fft, A_symp_ffts = symp_corr_fft.forward(
            stoch_ref_symp_fft, stoch_train_symp_fft, ref_symp_fft, train_symp_fft
        )

        # A_seq_ffts.append(A_seq_fft)
        # A_symp_ffts.append(A_symp_fft)


    # Get view-aware latent embeddings
    train_months_z, train_month_sr, _, month_loss, _ = month_corr.forward(
        stoch_ref_months, stoch_train_months, ref_months, train_months
    )
    train_seq_z, train_seq_sr, _, seq_loss, A_seqs = seq_corr.forward(
        stoch_ref_seq, stoch_train_seq, ref_seq, train_seq
    )
    train_symp_z, train_symp_sr, _, symp_loss, A_symps = symp_corr.forward(
        stoch_ref_symp, stoch_train_symp, ref_symp, train_symp
    )
    train_reg_z, train_reg_sr, _, reg_loss, _ = reg_corr.forward(
        stoch_ref_reg, stoch_train_reg, ref_reg, train_reg
    )

    # A_seq_ffts.append(A_seq)
    # A_symp_ffts.append(A_symp)

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
    print(f"Loss = {loss.detach().cpu().numpy()}", end="\r")

    return (
        mean_y.detach().cpu().numpy(),
        losses.detach().cpu().numpy(),
        loss.detach().cpu().numpy(),
    )


def evaluate(test_seqs, test_seqs_smart, test_symp_seqs, test_symp_seqs_smart, reg_test, mt_test, test_y, sample=True):
    global A_seq_ffts_test, A_symp_ffts_test, A_seqs_test, A_symps_test

    for m in models:
        m.eval()
    # Porbabilistic encode of reference points
    ref_months = month_enc.forward(long_tensor(month_references))
    ref_seq = seq_encoder.forward(float_tensor(seq_references))
    ref_symp = symp_encoder.forward(float_tensor(symp_references))
    ref_reg = reg_encoder.forward(long_tensor(reg_references))

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
    test_reg = reg_encoder.forward(long_tensor(reg_test.astype(int)))
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
        test_seq_z_fft, test_seq_sr_fft, _, seq_loss_fft, A_seq_ffts_test = seq_corr_fft.forward(
            stoch_ref_seq_fft, stoch_test_seq_fft, ref_seq_fft, test_seq_fft
        )

        test_symp_z_fft, test_symp_sr_fft, _, symp_loss_fft, A_symp_ffts_test = symp_corr_fft.forward(
            stoch_ref_symp_fft, stoch_test_symp_fft, ref_symp_fft, test_symp_fft
        )

    # Get view-aware latent embeddings
    test_months_z, test_month_sr, _, _, _, _ = month_corr.predict(
        stoch_ref_months, stoch_test_months, ref_months, test_months
    )
    test_seq_z, test_seq_sr, _, _, _, A_seqs_test = seq_corr.predict(
        stoch_ref_seq, stoch_test_seq, ref_seq, test_seq
    )
    test_symp_z, test_symp_sr, _, _, _, A_symps_test = symp_corr.predict(
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
    crps_here = crps_samples(sample_y,  test_y.ravel())
    print(f"RMSE = {rmse}")
    print(f"CRPS = {crps_here}")
    return rmse, crps_here, mean_y, sample_y

loss_min = np.inf
eval_flag = True
for ep in range(1, 2000 + 1):
    train_seqs, train_seqs_smart, train_symp_seqs, train_symp_seqs_smart, train_y, mt, reg = sample_train(100)
    _,_, loss = train(train_seqs, train_seqs_smart, train_symp_seqs, train_symp_seqs_smart, reg, mt, train_y)
    if loss < loss_min:
        loss_min = loss
        eval_flag = True

    if ep % 10 == 0 and eval_flag:
        print("\n")
        print("Evaluating at "+str(ep))
        test_seqs, test_seqs_smart, test_symp_seqs, test_symp_seqs_smart, test_y, mt_test, reg_test = sample_test(100)
        rmse, crps, _, _ = evaluate(test_seqs, test_seqs_smart, test_symp_seqs, test_symp_seqs_smart, reg_test, mt_test, test_y)
        print("\n")
        eval_flag = False

f = open("power_results/ahead_"+str(ahead)+"_optionals_"+str(options.optionals)+"_seed_"+str(seed), "w")
f.write("RMSE:"+str(rmse)+"\n")
f.write("CRPS:"+str(crps)+"\n")
# plt.imshow(A_seq_ffts.detach().cpu().numpy(), cmap="hot", interpolation='nearest')
# plt.colorbar()
# plt.savefig("plots_power/A_seq_ffts.png")
# plt.imshow(A_symp_ffts.detach().cpu().numpy(), cmap="hot", interpolation='nearest')
# plt.colorbar()
# plt.savefig("plots_power/A_symp_ffts.png")
# plt.imshow(A_seqs.detach().cpu().numpy(), cmap="hot", interpolation='nearest')
# plt.colorbar()
# plt.savefig("plots_power/A_seqs.png")
# plt.imshow(A_symps.detach().cpu().numpy(), cmap="hot", interpolation='nearest')
# plt.colorbar()
# plt.savefig("plots_power/A_symps.png")

# plt.imshow(A_seq_ffts_test.detach().cpu().numpy(), cmap="hot", interpolation='nearest')
# plt.colorbar()
# plt.savefig("plots_power/A_seq_ffts_test.png")
# plt.imshow(A_symp_ffts_test.detach().cpu().numpy(), cmap="hot", interpolation='nearest')
# plt.colorbar()
# plt.savefig("plots_power/A_symp_ffts_test.png")
# plt.imshow(A_seqs_test.detach().cpu().numpy(), cmap="hot", interpolation='nearest')
# plt.colorbar()
# plt.savefig("plots_power/A_seqs_test.png")
# plt.imshow(A_symps_test.detach().cpu().numpy(), cmap="hot", interpolation='nearest')
# plt.colorbar()
# plt.savefig("plots_power/A_symps_test.png")
# pu.db