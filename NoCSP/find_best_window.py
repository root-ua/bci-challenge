from utils import *
from sklearn import preprocessing
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

folder_name = '../../shrinked_data/'


def find_best_window(alg, starts, sizes, features, base_accuracy=0.58):
    log('find_best_window started with algorithm %s and features %s' % (alg, str(features)))

    Accuracy = np.zeros((len(sizes), len(starts)))
    Accuracy.fill(base_accuracy)

    data, train_labels = load_data(folder_name, 'train')

    best_acc = 0
    for i, window_start in enumerate(starts):
        for j, window_size in enumerate(sizes):
            windows = get_windows(data, window_start, window_size)

            w_data = np.array(windows)
            if alg == 'SVM':
                w_data = preprocessing.scale(w_data)

            acc = train_test_and_validate(alg, w_data, train_labels, features)

            acc = base_accuracy if acc < base_accuracy else acc
            Accuracy[j][i] = acc

            if acc > best_acc:
                color = True
                best_acc = acc
            else:
                color = False

            log(alg + ' Accuracy on start %3i, window size %3i train data %.8f%%' % (window_start, window_size, acc),
                color)

            try:
                plot_data = go.Data([go.Contour(z=Accuracy, x=starts, y=sizes)])
                py.plot(plot_data, filename='prediction-%s with features %s' % (alg, str(features)),
                                   fileopt='overwrite', auto_open=False)
            except BaseException as e:
                print 'error occurred when trying to plot: ' + e.message

