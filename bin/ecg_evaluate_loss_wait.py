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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epoch", help="Set epoch", type=int, default=0)
parser.add_argument("-el", "--epoch_limit", help="Set epoch limit", type=int, default=30)
parser.add_argument("-li", "--log_interval", help="Set batch interval for log", type=int, default=10)
parser.add_argument("-t", "--is_test", help="Set isTest", action='store_true')
parser.add_argument("-c", "--is_comet", help="Set isTest", action='store_true')
parser.add_argument("-p", "--comet_project", help="Set project name", type=str, default='secure-ecg-c')
args = parser.parse_args()

N_HIDDEN = 5
LOSS = 'mse'
# MEAN = 59.3
# STD = 10.6
MEAN = 61.9
STD = 10.6

result_path = 'result'
# epoch = args.epoch
# batch = args.batch
batches = 500

# 5500 criteria
mean_x = 1.547
std_x = 156.820


def scale(arr, m, s):
    arr = arr - m
    arr = arr / (s + 1e-7)
    return arr


def rescale(arr, m, s):
    arr = arr * s
    arr = arr + m
    return arr


def avgpool(arr, kernel_size, stride):
    row = arr.shape[1] / stride
    if row % 2 == 1:
        row = row - 1
    row = int(row)
    new_shape = (arr.shape[0], row, arr.shape[2])
    new_arr = np.zeros(new_shape)

    for r in range(row):
        for k in range(kernel_size):
            new_arr[:, r, :] += arr[:, r * stride + k, :] / kernel_size

    return new_arr


def report_scores(X, y, W, b, act):
    y_true = []
    y_pred = []
    # y_score = []

    # X = scale(X, mean_x, std_x)
    # print('Example : X - ', X[0, 0:3], 'y - ', y[0])

    torchX = torch.from_numpy(np.array(X))
    reshape_img = conv1d(torchX, 3, 6)
    # print('reshape img' , reshape_img.shape)
    for l in range(N_HIDDEN):
        # print('l ==== ', l)
        if l == 0:
            act[l] = np.dot(reshape_img, W[l]) + b[l]
            act[l] = avgpool(act[l], 2, 2)
            # act[l] = np.maximum(0, np.dot(reshape_img, W[l]) + b[l])
        else:
            # if l == 1:
            #     reshape_img
            if l == 1 or l == 2:
                # print('before', act[l-1].shape)
                reshape_img2 = conv1d(torch.from_numpy(act[l - 1]), 6, 6)
                act[l - 1] = reshape_img2
                # print('after', act[l-1].shape)
            else:
                act_col = act[l - 1].shape[-1]
                w_row = W[l].shape[0]
                # print('act_col, w_row', act_col, w_row)
                if act_col != w_row:
                    act[l - 1] = act[l - 1].reshape(act[l - 1].shape[0], -1)

            if l == N_HIDDEN - 1:
                act[l] = np.dot(act[l - 1], W[l]) + b[l]
            else:
                act[l] = np.maximum(0, np.dot(act[l - 1], W[l]) + b[l])
                if l == 1 or l == 2:
                    act[l] = avgpool(act[l], 2, 2)

    # print('act', act)

    if N_HIDDEN == 0:
        scores = np.dot(reshape_img, W[-1]) + b[-1]
    else:
        scores = np.dot(act[-1], W[-1]) + b[-1]

    y = rescale(y, MEAN, STD)
    scores = rescale(scores, MEAN, STD)

    mse_loss = metrics.mean_squared_error(y, scores)

    y_true.extend(list(y))
    y_pred.extend(scores)

    return y_true, y_pred, mse_loss


def load_model(epoch, batch):
    file_name_check = 'mpc/cache/ecg_P1_{}_{}_W0.bin'.format(epoch, batch)

    while not os.path.exists(file_name_check):
        print('Waiting 60s for the file to be generated : ', file_name_check)
        time.sleep(60)

    W = [[] for _ in range(N_HIDDEN + 1)]
    for l in range(N_HIDDEN + 1):
        W[l] = np.loadtxt('mpc/cache/ecg_P1_{}_{}_W{}.bin'.format(epoch, batch, l))

    # Initialize bias vector with zeros.
    b = [[] for _ in range(N_HIDDEN + 1)]
    for l in range(N_HIDDEN + 1):
        b[l] = np.loadtxt('mpc/cache/ecg_P1_{}_{}_b{}.bin'.format(epoch, batch, l))

    # Initialize activations.
    act = [[] for _ in range(N_HIDDEN)]

    print('Model loaded from ', file_name_check)

    return W, b, act


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

    plt.scatter(y_pred, y_true, s=3)
    plt.suptitle(message)
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    if args.is_comet:
        experiment.log_figure(figure=plt, figure_name='{}_{}.png'.format(epoch, batch))
    else:
        plt.savefig("{}/scatter/{}_{}.png".format(result_path, epoch, batch))
    plt.clf()


##TODO WONDERIT conv1d
def conv1d(
        input,
        in_channels,
        out_channels,
):
    input = input.view(input.shape[0], in_channels, -1)
    assert len(input.shape) == 3

    # Change to tuple if not one
    stride = 1
    padding = 0
    dilation = 1

    # Extract a few useful values
    batch_size, nb_channels_in, nb_cols_in = input.shape
    # nb_channels_out, nb_channels_kernel, nb_cols_kernel = (3, 3, 7)
    nb_channels_out = out_channels
    nb_channels_kernel = in_channels
    nb_cols_kernel = 7

    # Check if inputs are coherent
    # assert nb_channels_in == nb_channels_kernel * groups

    # nb_cols_out = int(
    #     ((nb_cols_in + 2 * padding[0] - dilation[0] * (nb_cols_kernel - 1) - 1) / stride[0])
    #     + 1
    # )
    nb_cols_out = int(
        ((nb_cols_in - dilation * (nb_cols_kernel - 1) - 1) / stride)
        + 1
    )

    # Apply padding to the input
    # if padding != (0, 0):
    #     padding_mode = "constant" if padding_mode == "zeros" else padding_mode
    #     input = torch.nn.functional.pad(
    #         input, (padding[0], padding[0]), padding_mode
    #     )
    #     # Update shape after padding
    #     # nb_rows_in += 2 * padding[0]
    #     nb_cols_in += 2 * padding[0]

    # We want to get relative positions of values in the input tensor that are used by one filter convolution.
    # It basically is the position of the values used for the top left convolution.
    pattern_ind = []
    for ch in range(nb_channels_in):
        for c in range(nb_cols_kernel):
            pixel = c * dilation
            pattern_ind.append(pixel + ch * nb_cols_in)

    im_flat = input.view(batch_size, -1)
    im_reshaped = []
    for cur_col_out in range(nb_cols_out):
        # For each new output value, we just need to shift the receptive field
        offset = cur_col_out * stride
        tmp = [ind + offset for ind in pattern_ind]
        im_reshaped.append(im_flat[:, tmp])
    im_reshaped = torch.stack(im_reshaped).permute(1, 0, 2)
    return im_reshaped


if __name__ == '__main__':
    # Add the following code anywhere in your machine learning file
    if args.is_comet:
        experiment = Experiment(api_key="eIskxE43gdgwOiTV27APVUQtB", project_name=args.comet_project, workspace="wonderit")
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
            step = (log_batches * e + i) * args.log_interval
            W, b, act = load_model(e, i * args.log_interval)

            y_true_train, y_pred_train, train_mse_loss = report_scores(X_train, y_train, W, b, act)

            print('Training mse_loss: {0:.4f}'.format(train_mse_loss))

            y_true, y_pred, test_mse_loss = report_scores(X_test, y_test, W, b, act)
            print('Testing mse_loss: {0:.4f}'.format(test_mse_loss))

            _, train_r2 = r_squared_mse(y_true_train, y_pred_train, train_mse_loss)
            rm, test_r2 = r_squared_mse(y_true, y_pred, test_mse_loss)


            if args.is_comet:
                experiment.log_metric("train_mse", train_mse_loss, epoch=e+1, step=step)
                experiment.log_metric("test_mse", test_mse_loss, epoch=e+1, step=step)
                experiment.log_metric("train_r2", train_r2, epoch=e+1, step=step)
                experiment.log_metric("test_r2", test_r2, epoch=e+1, step=step)

            scatter_plot(y_true, y_pred, rm, e, i * args.log_interval)