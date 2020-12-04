#!/usr/bin/env python3

import argparse

import numpy as np
import torch  # For building the networks
import torchtuples as tt  # Some useful functions
from pycox.datasets import metabric
from pycox.evaluation import EvalSurv
from pycox.models import LogisticHazard
# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

# We also set some seeds to make this reproducable.
# Note that on gpu, there is still some randomness.
np.random.seed(1234)
_ = torch.manual_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Set model", type=str, default='fir')
parser.add_argument("-t", "--type", help="Set model", type=str, default='fir')

args = parser.parse_args()

df_train = metabric.read_df()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
cols_leave = ['x4', 'x5', 'x6', 'x7']

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

num_durations = 10

labtrans = LogisticHazard.label_transform(num_durations)

get_target = lambda df: (df['duration'].values, df['event'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))

train = (x_train, y_train)
val = (x_val, y_val)

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)

in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = labtrans.out_features
batch_norm = False
dropout = 0

# net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout, w_init_=lambda w: torch.nn.init.kaiming_uniform_(w, nonlinearity='relu'))
net = torch.nn.Sequential(
    torch.nn.Linear(in_features, 32),
    torch.nn.ReLU(),
    #     torch.nn.BatchNorm1d(32),
    #     torch.nn.Dropout(0.1),

    torch.nn.Linear(32, 32),
    torch.nn.ReLU(),

    #     torch.nn.Linear(32,32),
    #     torch.nn.ReLU(),

    #     torch.nn.BatchNorm1d(32),
    #     torch.nn.Dropout(0.1),

    torch.nn.Linear(32, out_features)
)

model = LogisticHazard(net, tt.optim.SGD(lr=0.1, momentum=0.9, nesterov=True), duration_index=labtrans.cuts)

batch_size = 256
epochs = 30
callbacks = [tt.cb.EarlyStopping()]
# callbacks = [CheckBatchEnd()]


N_HIDDEN = 2
# folder_name = 'gcloud-ep100-adam0.1-reg0.01-torchinit'
folder_name = args.model
W = [[] for _ in range(N_HIDDEN + 1)]
b = [[] for _ in range(N_HIDDEN + 1)]
for ep in range(101):

    for l in range(N_HIDDEN + 1):
        W[l] = np.loadtxt('../mpc/{}/ecg_P1_{}_3_W{}.bin'.format(folder_name, ep, l))
        W[l] = np.transpose(W[l])

    for l in range(N_HIDDEN + 1):
        b[l] = np.loadtxt('../mpc/{}/ecg_P1_{}_3_b{}.bin'.format(folder_name, ep, l))

    w0_from_text = torch.from_numpy(W[0])
    w1_from_text = torch.from_numpy(W[1])
    w2_from_text = torch.from_numpy(W[2])
    model.net[0].weight = torch.nn.Parameter(w0_from_text.float())
    model.net[2].weight = torch.nn.Parameter(w1_from_text.float())
    model.net[4].weight = torch.nn.Parameter(w2_from_text.float())
    model.net[0].bias = torch.nn.Parameter(torch.from_numpy(b[0]).float())
    model.net[2].bias = torch.nn.Parameter(torch.from_numpy(b[1]).float())
    model.net[4].bias = torch.nn.Parameter(torch.from_numpy(b[2]).float())

    surv = model.interpolate(10).predict_surv_df(x_test)
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    val_loss = model.score_in_batches(val)['loss']
    c_index = ev.concordance_td('antolini')
    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    ibs = ev.integrated_brier_score(time_grid)
    inbll = ev.integrated_nbll(time_grid)
    if args.type == 'all':
        print('ep : {}, c-index : {:.3f}, IBS : {:.3f}, INBLL: {:.3f}, val_loss : {:.3f}'
              .format(ep, c_index, ibs, inbll, val_loss))
    elif args.type == 'cindex':
        print('{:.3f}'.format(c_index))
    else:
        print('Need appropriate type (current type : {})'.format(args.type))
