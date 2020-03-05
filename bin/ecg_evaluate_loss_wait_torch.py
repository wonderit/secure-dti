import datetime
import numpy as np
from comet_ml import Experiment
from sklearn import metrics
import sys
from sklearn.metrics import r2_score, mean_squared_error
import math
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epoch", help="Set epoch", type=int, default=0)
parser.add_argument("-el", "--epoch_limit", help="Set epoch limit", type=int, default=30)
parser.add_argument("-li", "--log_interval", help="Set batch interval for log", type=int, default=5)
parser.add_argument("-b", "--batch_size", help="Set batch size for log", type=int, default=20)
parser.add_argument("-t", "--is_test", help="Set isTest", action='store_true')
parser.add_argument("-c", "--is_comet", help="Set isTest", action='store_true')
parser.add_argument("-f", "--cache_folder", help="Set folder name", type=str, default='cache')
parser.add_argument("-p", "--comet_project", help="Set project name", type=str, default='secure-ecg-c')
parser.add_argument("-s", "--seed", help="Set random seed", type=int, default=1234)
args = parser.parse_args()

N_HIDDEN = 5
LOSS = 'mse'

# previous one
# MEAN = 59.3
# STD = 10.6

# outlier removed one
MEAN = 61.6
STD = 9.8
_ = torch.manual_seed(args.seed)

result_path = 'result'
batches = 5000 / args.batch_size


def scale(arr, m, s):
    arr = arr - m
    arr = arr / (s + 1e-7)
    return arr


def rescale(arr, m, s):
    arr = arr * s
    arr = arr + m
    return arr


def report_scores(X, y, trained_model):
    y_true = []
    y_pred = []
    # y_score = []

    # DON'T NORMALIZE X
    # X = scale(X, mean_x, std_x)
    # print('Example : X - ', X[0, 0:3], 'y - ', y[0])

    reshaped_X = X.reshape(X.shape[0], 3, 500)


    with torch.no_grad():
        scores = trained_model(torch.from_numpy(reshaped_X))
        #
        # output rescale
        scores = rescale(scores, MEAN, STD)
        y = rescale(y, MEAN, STD)

        mse_loss = metrics.mean_squared_error(y, scores)

        y_true.extend(list(y))
        y_pred.extend(scores)

    return y_true, y_pred, mse_loss


def load_model(model, epoch, batch):
    file_name_check = 'mpc/{}/ecg_P1_{}_{}_W0.bin'.format(args.cache_folder, epoch, batch)

    while not os.path.exists(file_name_check):
        print('Waiting 60s for the file to be generated : ', file_name_check)
        time.sleep(60)

    W = [[] for _ in range(N_HIDDEN + 1)]
    for l in range(N_HIDDEN + 1):
        W[l] = np.loadtxt('mpc/{}/ecg_P1_{}_{}_W{}.bin'.format(args.cache_folder, epoch, batch, l))

    # Initialize bias vector with zeros.
    b = [[] for _ in range(N_HIDDEN + 1)]
    for l in range(N_HIDDEN + 1):
        b[l] = np.loadtxt('mpc/{}/ecg_P1_{}_{}_b{}.bin'.format(args.cache_folder, epoch, batch, l))

    W[0] = np.transpose(W[0])
    w0_from_text = torch.from_numpy(W[0].reshape(6, 3, 7))
    model.conv1.weight = torch.nn.Parameter(w0_from_text)
    b0_from_text = torch.from_numpy(b[0])
    model.conv1.bias = torch.nn.Parameter(b0_from_text)

    W[1] = np.transpose(W[1])
    w1_from_text = torch.from_numpy(W[1].reshape(6, 6, 7))
    model.conv2.weight = torch.nn.Parameter(w1_from_text)
    b1_from_text = torch.from_numpy(b[1])
    model.conv2.bias = torch.nn.Parameter(b1_from_text)

    W[2] = np.transpose(W[2])
    w2_from_text = torch.from_numpy(W[2].reshape(6, 6, 7))
    model.conv3.weight = torch.nn.Parameter(w2_from_text)
    b2_from_text = torch.from_numpy(b[2])
    model.conv3.bias = torch.nn.Parameter(b2_from_text)

    W[3] = np.transpose(W[3])
    w3_from_text = torch.from_numpy(W[3])
    model.fc1.weight = torch.nn.Parameter(w3_from_text)

    b3_from_text = torch.from_numpy(b[3])
    model.fc1.bias = torch.nn.Parameter(b3_from_text)

    W[4] = np.transpose(W[4])
    w4_from_text = torch.from_numpy(W[4])
    model.fc2.weight = torch.nn.Parameter(w4_from_text)
    b4_from_text = torch.from_numpy(b[4])
    model.fc2.bias = torch.nn.Parameter(b4_from_text)

    w5_from_text = torch.from_numpy(W[5])
    w5_from_text = w5_from_text.reshape(1, 64)
    model.fc3.weight = torch.nn.Parameter(w5_from_text)
    b5_from_text = torch.from_numpy(b[5])
    model.fc3.bias = torch.nn.Parameter(b5_from_text)

    model.eval()

    return model


def r_squared_mse(y_true, y_pred, mse, sample_weight=None, multioutput=None):
    r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    # mse = mean_squared_error(y_true, y_pred,
    #                          sample_weight=sample_weight,
    #                          multioutput=multioutput)
    # Output aggregated scores.
    try:
        sys.stdout.write(str(datetime.datetime.now()) + ' | ')
        print('\ntype \t Actual \t\t Predictions'.format(np.std(y_true), np.std(y_pred)))
        print('std \t {0:.2f} \t\t {1:.2f}'.format(np.std(y_true), np.std(y_pred)))
        print('min \t {0:.2f} \t\t {1:.2f}'.format(np.min(y_true), np.min(y_pred)))
        print('max \t {0:.2f} \t\t {1:.2f}'.format(np.max(y_true), np.max(y_pred)))
        print('median \t {0:.2f} \t\t {1:.2f}'.format(np.median(y_true), np.median(y_pred)))
        print('mean \t {0:.2f} \t\t {1:.2f}'.format(np.mean(y_true), np.mean(y_pred)))
        print('MSE \t {0:.4f}'.format(mse))
        print('RMSE \t {0:.4f}'.format(math.sqrt(mse)))
        print('R2 \t {0:.4f}'.format(r2))

    except Exception as e:
        sys.stderr.write(str(e))
        sys.stderr.write('\n')

    result_message = 'r2:{:.3f}, mse:{:.3f}, std:{:.3f},{:.3f}'.format(r2, mse, np.std(y_true), np.std(y_pred))
    return result_message, r2


def scatter_plot(y_true, y_pred, message, epoch, batch):
    result = np.column_stack((y_true, y_pred))

    if not os.path.exists('{}/{}'.format(result_path, 'csv')):
        os.makedirs('{}/{}'.format(result_path, 'csv'))

    if not os.path.exists('{}/{}'.format(result_path, 'scatter')):
        os.makedirs('{}/{}'.format(result_path, 'scatter'))

    pd.DataFrame(result).to_csv("{}/csv/{}.csv".format(result_path, 1), index=False)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    plt.scatter(y_pred, y_true, s=3)
    plt.suptitle(message)
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    if args.is_comet:
        experiment.log_figure(figure=plt, figure_name='{}_{}.png'.format(epoch, batch))
    else:
        plt.savefig("{}/scatter/{}_{}.png".format(result_path, epoch, batch))
    plt.clf()


class CNNAVG(nn.Module):
    def __init__(self):
        super(CNNAVG, self).__init__()
        self.kernel_size = 7
        self.padding_size = 0
        self.channel_size = 6
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avgpool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv1d(3, self.channel_size, kernel_size=self.kernel_size, padding=self.padding_size)
        self.conv2 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=self.padding_size)
        self.conv3 = nn.Conv1d(self.channel_size, self.channel_size, kernel_size=self.kernel_size,
                               padding=self.padding_size)
        self.fc1 = nn.Linear(342, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)  # 32
        x = self.avgpool1(x)  # 32
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        y = F.relu(self.conv3(x))
        y = self.avgpool3(y)
        y = y.view(y.shape[0], -1)

        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y


if __name__ == '__main__':
    # Add the following code anywhere in your machine learning file
    if args.is_comet:
        experiment = Experiment(api_key="eIskxE43gdgwOiTV27APVUQtB", project_name=args.comet_project,
                                workspace="wonderit")
    else:
        experiment = None

    X_train = np.genfromtxt('../data/ecg/text_demo_5500/Xtrain', delimiter=',', dtype='float')
    y_train = np.genfromtxt('../data/ecg/text_demo_5500/ytrain', delimiter=',', dtype='float')

    X_test = np.genfromtxt('../data/ecg/text_demo_5500/Xtest', delimiter=',', dtype='float')
    y_test = np.genfromtxt('../data/ecg/text_demo_5500/ytest', delimiter=',', dtype='float')

    log_batches = int(batches / args.log_interval)
    step = 0

    for e in range(args.epoch_limit):
        if args.epoch > 0 and e < args.epoch:
            continue

        for i in range(log_batches):
            if i == 0:
                continue
            step = (log_batches * e + i)
            # W, b, act = load_model(e, i * args.log_interval)
            model = CNNAVG()

            model = load_model(model, e, i * args.log_interval)

            y_true_train, y_pred_train, train_mse_loss = report_scores(X_train, y_train, model)

            print('Training mse_loss: {0:.4f}'.format(train_mse_loss))

            y_true, y_pred, test_mse_loss = report_scores(X_test, y_test, model)
            print('Testing mse_loss: {0:.4f}'.format(test_mse_loss))

            _, train_r2 = r_squared_mse(y_true_train, y_pred_train, train_mse_loss)
            rm, test_r2 = r_squared_mse(y_true, y_pred, test_mse_loss)

            if args.is_comet:
                experiment.log_metric("train_mse", train_mse_loss, epoch=e + 1, step=step)
                experiment.log_metric("test_mse", test_mse_loss, epoch=e + 1, step=step)
                experiment.log_metric("train_r2", train_r2, epoch=e + 1, step=step)
                experiment.log_metric("test_r2", test_r2, epoch=e + 1, step=step)

            scatter_plot(y_true, y_pred, rm, e, i * args.log_interval)