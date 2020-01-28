import datetime
import numpy as np
from sklearn import metrics
import sys

N_HIDDEN = 2
# LOSS = 'hinge'
LOSS = 'mse'

def report_scores(X, y, W, b, act):
    y_true = []
    y_pred = []
    y_score = []
    
    for l in range(N_HIDDEN):
        if l == 0:
            act[l] = np.maximum(0, np.dot(X, W[l]) + b[l])
        else:
            act[l] = np.maximum(0, np.dot(act[l-1], W[l]) + b[l])

    if N_HIDDEN == 0:
        scores = np.dot(X, W[-1]) + b[-1]
    else:
        scores = np.dot(act[-1], W[-1]) + b[-1]

    predicted_class = np.zeros(scores.shape)
    print('scores', scores.shape, scores[0])
    if LOSS == 'hinge':
        predicted_class[scores > 0] = 1
        predicted_class[scores <= 0] = -1
        y = 2 * y - 1
    else:
        predicted_class[scores >= 0.5] = 1


    print('predicted_class', predicted_class.shape, predicted_class[0])
    print('y', y.shape, y[0])

    sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    print('Batch accuracy: {}'
          .format(metrics.accuracy_score(
              y, predicted_class
          ))
    )
    # print('Batch accuracy: {}'
    #           .format(metrics.mean_squared_error(
    #               y, scores
    #           ))
    #     )

    # print(y.shape, scores.shape, act)
    # print(y[0:5], scores[0:5])
    predicted_class = scores.argmax(axis=1)
    y_true = y.argmax(axis=1)
    print('predicted_class', y_true.shape, y_true[0])
    y_pred = predicted_class
    y_score = scores
    # y_true.extend(list(y))
    # y_pred.extend(list(predicted_class))
    # y_score.extend(list(scores))
    
    # Output aggregated scores.
    try:
        sys.stdout.write(str(datetime.datetime.now()) + ' | ')
        print('Accuracy: {0:.2f}'.format(
            metrics.accuracy_score(y_true, y_pred))
        )
        # sys.stdout.write(str(datetime.datetime.now()) + ' | ')
        # print('F1: {0:.2f}'.format(
        #     metrics.f1_score(y_true, y_pred))
        # )
        # sys.stdout.write(str(datetime.datetime.now()) + ' | ')
        # print('Precision: {0:.2f}'.format(
        #     metrics.precision_score(y_true, y_pred))
        # )
        # sys.stdout.write(str(datetime.datetime.now()) + ' | ')
        # print('Recall: {0:.2f}'.format(
        #     metrics.recall_score(y_true, y_pred))
        # )
        # sys.stdout.write(str(datetime.datetime.now()) + ' | ')
        # print('ROC AUC: {0:.2f}'.format(
        #     metrics.roc_auc_score(y_true, y_score))
        # )
        # sys.stdout.write(str(datetime.datetime.now()) + ' | ')
        # print('Avg. precision: {0:.2f}'.format(
        #     metrics.average_precision_score(y_true, y_score))
        # )
    except Exception as e:
        sys.stderr.write(str(e))
        sys.stderr.write('\n')
        
    return y_true, y_pred, y_score


def load_model():
    W = [ [] for _ in range(N_HIDDEN + 1) ]
    for l in range(N_HIDDEN+1):
        W[l] = np.loadtxt('mpc/cache/mnist_P1_W{}_final.bin'.format(l))

     # Initialize bias vector with zeros.
    b = [ []  for _ in range(N_HIDDEN + 1) ]
    for l in range(N_HIDDEN+1):
        b[l] = np.loadtxt('mpc/cache/mnist_P1_b{}_final.bin'.format(l))

    # Initialize activations.
    act = [ [] for _ in range(N_HIDDEN) ]

    return W, b, act
    
if __name__ == '__main__':
    W, b, act = load_model()

    X_train = np.genfromtxt('../data/mnist/text_demo_1280/Xtrain',
                            delimiter=',', dtype='float')
    y_train = np.genfromtxt('../data/mnist/text_demo_1280/ytrain',
                            delimiter=',', dtype='float')
    
    print('Training accuracy:')
    report_scores(X_train, y_train, W, b, act)

    X_test = np.genfromtxt('../data/mnist/text_demo_1280/Xtest',
                            delimiter=',', dtype='float')
    y_test = np.genfromtxt('../data/mnist/text_demo_1280/ytest',
                            delimiter=',', dtype='float')
    
    print('Testing accuracy:')
    report_scores(X_test, y_test, W, b, act)
