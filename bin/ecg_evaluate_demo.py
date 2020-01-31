import datetime
import numpy as np
from sklearn import metrics
import sys
from sklearn.metrics import r2_score, mean_squared_error
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch

N_HIDDEN = 5
LOSS = 'mse'
MEAN = 59.3
STD = 10.6
result_path = 'result'
isTest = False
epoch = 0
batch = 110

# 11000 criteria
# mean_x = 1.66
# std_x = 155.51

# 11000 criteria
# mean_x = 1.91397
# std_x = 156.413

# 5500 criteria
mean_x = 1.547
std_x = 156.820

def scale(arr, m, s):
    arr = arr - m
    arr = arr / (s + 1e-7)
    return arr

# LOSS = 'hinge'

def rescale(arr, m, s):
    arr = arr * s
    arr = arr + m
    return arr


def report_scores(X, y, W, b, act):
    y_true = []
    y_pred = []
    y_score = []

    X = scale(X, mean_x, std_x)
    print(X[0, 0:3], y[0])

    torchX = torch.from_numpy(np.array(X))
    reshape_img = conv1d(torchX, 3, 6)
    # print('reshape img' , reshape_img.shape)
    for l in range(N_HIDDEN):
        # print('l ==== ', l)
        if l == 0:
            act[l] = np.dot(reshape_img, W[l]) + b[l]
            # act[l] = np.maximum(0, np.dot(reshape_img, W[l]) + b[l])
        else:
            # if l == 1:
            #     reshape_img
            if l == 1 or l == 2:
                # print('before', act[l-1].shape)
                reshape_img2 = conv1d(torch.from_numpy(act[l-1]), 6, 6)
                act[l-1] = reshape_img2
                # print('after', act[l-1].shape)
            else:
                act_col = act[l-1].shape[-1]
                w_row = W[l].shape[0]
                # print('act_col, w_row', act_col, w_row)
                if act_col != w_row:
                    act[l-1] = act[l-1].reshape(act[l-1].shape[0], -1)

            if l == N_HIDDEN - 1:
                act[l] = np.dot(act[l-1], W[l]) + b[l]
            else:
                act[l] = np.maximum(0, np.dot(act[l-1], W[l]) + b[l])

    # print('act', act)

    if N_HIDDEN == 0:
        scores = np.dot(reshape_img, W[-1]) + b[-1]
    else:
        scores = np.dot(act[-1], W[-1]) + b[-1]

    y = rescale(y, MEAN, STD)
    scores = rescale(scores, MEAN, STD)

    print(y.shape, scores.shape)
    print(y[0:5], scores[0:5])

    mse_loss = metrics.mean_squared_error(y, scores)

    # predicted_class = np.zeros(scores.shape)
    # if LOSS == 'hinge':
    #     predicted_class[scores > 0] = 1
    #     predicted_class[scores <= 0] = -1
    #     y = 2 * y - 1
    # elif LOSS == 'mse':
    #
    # else:
    #     predicted_class[scores >= 0.5] = 1

    sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    print('Batch mse: {0:.2f}'
          .format(mse_loss)
    )
    
    y_true.extend(list(y))
    # y_pred.extend(list(predicted_class))
    y_pred.extend(scores)
    # y_score.extend(list(mse_loss))
    y_score.append(mse_loss)
    
    # Output aggregated scores.
    # try:
    #     sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    #     print('Accuracy: {0:.2f}'.format(
    #         metrics.accuracy_score(y_true, y_pred))
    #     )
    #     sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    #     print('F1: {0:.2f}'.format(
    #         metrics.f1_score(y_true, y_pred))
    #     )
    #     sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    #     print('Precision: {0:.2f}'.format(
    #         metrics.precision_score(y_true, y_pred))
    #     )
    #     sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    #     print('Recall: {0:.2f}'.format(
    #         metrics.recall_score(y_true, y_pred))
    #     )
    #     sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    #     print('ROC AUC: {0:.2f}'.format(
    #         metrics.roc_auc_score(y_true, y_score))
    #     )
    #     sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    #     print('Avg. precision: {0:.2f}'.format(
    #         metrics.average_precision_score(y_true, y_score))
    #     )
    # except Exception as e:
    #     sys.stderr.write(str(e))
    #     sys.stderr.write('\n')

        
    return y_true, y_pred, y_score


def load_model():
    W = [ [] for _ in range(N_HIDDEN + 1) ]
    for l in range(N_HIDDEN+1):
        W[l] = np.loadtxt('mpc/cache/ecg_P1_W{}_{}_{}.bin'.format(l, epoch, batch))

     # Initialize bias vector with zeros.
    b = [ []  for _ in range(N_HIDDEN + 1) ]
    for l in range(N_HIDDEN+1):
        b[l] = np.loadtxt('mpc/cache/ecg_P1_b{}_{}_{}.bin'.format(l, epoch, batch))

    # Initialize activations.
    act = [ [] for _ in range(N_HIDDEN) ]

    # print(np.array(W[0]).shape, np.array(b[0]).shape, act)
    # print(np.array(W[1]).shape, np.array(b[1]).shape, act)
    # print(np.array(W[2]).shape, np.array(b[2]).shape, act)

    return W, b, act


def r_squared_mse(y_true, y_pred, sample_weight=None, multioutput=None):

    r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    mse = mean_squared_error(y_true, y_pred,
                             sample_weight=sample_weight,
                             multioutput=multioutput)
    # bounds_check = np.min(y_pred) > MIN_MOISTURE_BOUND
    # bounds_check = bounds_check&(np.max(y_pred) < MAX_MOISTURE_BOUND)

    print('Scoring - std', np.std(y_true), np.std(y_pred))
    print('Scoring - median', np.median(y_true), np.median(y_pred))
    print('Scoring - min', np.min(y_true), np.min(y_pred))
    print('Scoring - max', np.max(y_true), np.max(y_pred))
    print('Scoring - mean', np.mean(y_true), np.mean(y_pred))
    print('Scoring - MSE: ', mse, 'RMSE: ', math.sqrt(mse))
    print('Scoring - R2: ', r2)
    # print(y_pred)
    # exit()

    result_message = 'r2:{:.3f}, mse:{:.3f}, std:{:.3f},{:.3f}'.format(r2, mse, np.std(y_true), np.std(y_pred))
    return result_message


def scatter_plot(y_true, y_pred, message):
    result = np.column_stack((y_true,y_pred))

    if not os.path.exists('{}/{}'.format(result_path, 'csv')):
        os.makedirs('{}/{}'.format(result_path, 'csv'))

    if not os.path.exists('{}/{}'.format(result_path, 'scatter')):
        os.makedirs('{}/{}'.format(result_path, 'scatter'))

    pd.DataFrame(result).to_csv("{}/csv/{}.csv".format(result_path, 1), index=False)

    plt.scatter(y_pred, y_true, s=3)
    plt.suptitle(message)
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    plt.savefig("{}/scatter/{}_{}.png".format(result_path, epoch, batch))
    plt.clf()
    # plt.show()

##TODO WONDERIT conv1d
def conv1d(
        input,
        in_channels,
        out_channels,
):
    """
    Overloads torch.conv1d to be able to use MPC on convolutional networks.
    The idea is to build new tensors from input and weight to compute a
    matrix multiplication equivalent to the convolution.

    Args:
        input: input image
        weight: convolution kernels
        bias: optional additive bias
        stride: stride of the convolution kernels
        padding:
        dilation: spacing between kernel elements
        groups:
        padding_mode: type of padding, should be either 'zeros' or 'circular' but 'reflect' and 'replicate' accepted
    Returns:
        the result of the convolution as an AdditiveSharingTensor
    """
    # Currently, kwargs are not unwrapped by hook_args
    # So this needs to be done manually
    # if bias.is_wrapper:
    #     bias = bias.child
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

    # print('kkkkk')
    # print(batch_size, nb_channels_in, nb_cols_in)
    # print(nb_channels_out, nb_channels_kernel, nb_cols_kernel)

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

    # The image tensor is reshaped for the matrix multiplication:
    # on each row of the new tensor will be the input values used for each filter convolution
    # We will get a matrix [[in values to compute out value 0],
    #                       [in values to compute out value 1],
    #                       ...
    #                       [in values to compute out value nb_rows_out*nb_cols_out]]
    im_flat = input.view(batch_size, -1)
    im_reshaped = []
    # for cur_row_out in range(nb_rows_out):
    for cur_col_out in range(nb_cols_out):
        # For each new output value, we just need to shift the receptive field
        offset = cur_col_out * stride
        tmp = [ind + offset for ind in pattern_ind]
        im_reshaped.append(im_flat[:, tmp])

    # print('img reshaped shape', np.array(im_reshaped).shape)

    im_reshaped = torch.stack(im_reshaped).permute(1, 0, 2)
    # print('after img reshaped shape', im_reshaped.shape)
    # The convolution kernels are also reshaped for the matrix multiplication
    # We will get a matrix [[weights for out channel 0],
    #                       [weights for out channel 1],
    #                       ...
    #                       [weights for out channel nb_channels_out]].TRANSPOSE()
    # weight_reshaped = weight.view(nb_channels_out // groups, -1).t()

    # Now that everything is set up, we can compute the result
    # if groups > 1:
    #     res = []
    #     chunks_im = torch.chunk(im_reshaped, groups, dim=1)
    #     chunks_weights = torch.chunk(weight_reshaped, groups, dim=0)
    #     for g in range(groups):
    #         tmp = chunks_im[g].matmul(chunks_weights[g])
    #         res.append(tmp)
    #     res = torch.cat(res, dim=1)
    # else:
    #     res = im_reshaped.matmul(weight_reshaped)

    # Add a bias if needed
    # if bias is not None:
    #     res += bias
    #
    # # ... And reshape it back to an image
    # res = (
    #     res.permute(0, 2, 1)
    #         .view(batch_size, nb_channels_out, nb_cols_out)
    #         .contiguous()
    # )
    return im_reshaped

if __name__ == '__main__':
    W, b, act = load_model()

    # X_train = np.genfromtxt('demo_data/text_demo/Xtrain',
    #                         delimiter=',', dtype='float')
    # y_train = np.genfromtxt('demo_data/text_demo/ytrain',
    #                         delimiter=',', dtype='float')
    X_train = np.genfromtxt('../data/ecg/text_demo_5500/Xtrain',
                            delimiter=',', dtype='float')
    y_train = np.genfromtxt('../data/ecg/text_demo_5500/ytrain',
                            delimiter=',', dtype='float')

    if isTest:
        X_train = X_train[:20,:]
        y_train = y_train[:20]
        print(X_train.shape, y_train.shape)

    print('Training accuracy:')
    report_scores(X_train, y_train, W, b, act)

    # X_test = np.genfromtxt('demo_data/text_demo/Xtest',
    #                         delimiter=',', dtype='float')
    # y_test = np.genfromtxt('demo_data/text_demo/ytest',
    #                         delimiter=',', dtype='float')
    X_test = np.genfromtxt('../data/ecg/text_demo_5500/Xtest',
                            delimiter=',', dtype='float')
    y_test = np.genfromtxt('../data/ecg/text_demo_5500/ytest',
                            delimiter=',', dtype='float')

    if isTest:
        X_test = X_test[:20,:]
        y_test = y_test[:20]
        print(X_test.shape, y_test.shape)

    print('Testing accuracy:')
    y_true, y_pred, _ = report_scores(X_test, y_test, W, b, act)
    rm = r_squared_mse(y_true, y_pred)

    scatter_plot(y_true, y_pred, rm)