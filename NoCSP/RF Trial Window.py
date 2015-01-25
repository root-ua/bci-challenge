from NoCSP.load_data import *
from utils import *
from sklearn import cross_validation, metrics, preprocessing, svm
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import sklearn.ensemble as ens

folder_name = '../../shrinked_data/'

starts = np.arange(-70, 100, 15) # 11  items
sizes = np.arange(30, 250, 15) # 14  items
features = [47]

Accuracy = np.zeros((len(sizes), len(starts)))
# fil with best of what we have right now
Accuracy.fill(0.55989712)

data, train_labels = load_data(folder_name, 'train')

best_acc = 0
for i, window_start in enumerate(starts):
    for j, window_size in enumerate(sizes):
        windows = get_windows(data, window_start, window_size)

        w_data = np.array(windows)

        accuracy = np.zeros(3)

        for state in range(0, len(accuracy)):
            rs = cross_validation.ShuffleSplit(n_train_subjects, n_iter=10, test_size=.1, random_state=state)
            rs = [[train_index, test_index] for train_index, test_index in rs][0]

            train_data = w_data[epochs_indices(rs[0])]
            train_y = train_labels[epochs_indices(rs[0])]
            test_data = w_data[epochs_indices(rs[1])]
            test_y = train_labels[epochs_indices(rs[1])]

            train_x = extract_features(train_data, features)
            test_x = extract_features(test_data, features)

            clf = ens.RandomForestClassifier(n_estimators=500, max_features=0.25, min_samples_split=1, random_state=0)

            clf.fit(train_x, train_y)

            result = clf.predict_proba(test_x)

            fpr, tpr, thresholds = metrics.roc_curve(test_y, result[:, 1], pos_label=1)
            accuracy[state] = metrics.auc(fpr, tpr)

        acc = accuracy.min()
        Accuracy[j][i] = acc

        if acc > best_acc:
            color = True
            best_acc = acc
        else:
            color = False

        log('Accuracy on start %3i, window size %3i train data %.8f%%' % (window_start, window_size, acc), color)

        try:
            plot_data = go.Data([go.Contour(z=Accuracy, x=starts, y=sizes)])
            plot_url = py.plot(plot_data, filename='prediction-RF with feature 47', fileopt='overwrite', auto_open=False)
        except BaseException as e:
            print 'error occurred when trying to plot: ' + e.message

log('done')
