from NoCSP.load_data import *
from utils import *
from sklearn import cross_validation, metrics, preprocessing, svm
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

folder_name = '../../shrinked_data/'

starts = np.arange(-70, 100, 15) # 11  items
sizes = np.arange(30, 250, 15) # 14  items
features = [47, 13]

Accuracy = np.zeros((len(sizes), len(starts)))
# baseline, just a little smaller then best score we have
base_accuracy = 0.58
Accuracy.fill(base_accuracy)


data, train_labels = load_data(folder_name, 'train')

best_acc = 0
for i, window_start in enumerate(starts):
    for j, window_size in enumerate(sizes):
        windows = get_windows(data, window_start, window_size)

        w_data = np.array(windows)
        w_data = preprocessing.scale(w_data)

        accuracy = np.zeros(3)

        for state in range(0, len(accuracy)):
            rs = cross_validation.ShuffleSplit(n_train_subjects, n_iter=10, test_size=.1, random_state=state)
            rs = [[train_index, test_index] for train_index, test_index in rs][0]

            train_data = w_data[epochs_indices(rs[0])]
            train_y = train_labels[epochs_indices(rs[0])]
            test_data = w_data[epochs_indices(rs[1])]
            test_y = train_labels[epochs_indices(rs[1])]

            train_y = (train_y * 2) - 1

            train_x = extract_features(train_data, features)
            test_x = extract_features(test_data, features)

            clf = svm.SVC(probability=True)

            clf.fit(train_x, train_y)

            result = clf.predict_proba(test_x)

            fpr, tpr, thresholds = metrics.roc_curve(test_y, result[:, 1], pos_label=1)
            accuracy[state] = metrics.auc(fpr, tpr)

        acc = accuracy.min()
        acc = base_accuracy if acc < base_accuracy else acc
        Accuracy[j][i] = acc

        if acc > best_acc:
            color = True
            best_acc = acc
        else:
            color = False

        log('LDA Accuracy on start %3i, window size %3i train data %.8f%%' % (window_start, window_size, acc), color)

        try:
            plot_data = go.Data([go.Contour(z=Accuracy, x=starts, y=sizes)])
            plot_url = py.plot(plot_data, filename='prediction-SVM with features %s' % str(features),
                               fileopt='overwrite', auto_open=False)
        except BaseException as e:
            print 'error occurred when trying to plot: ' + e.message

log('done')
