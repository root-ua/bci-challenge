from scipy.io import savemat
from utils import *
from wyrm import processing as proc
from sklearn import cross_validation, metrics
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go


starts = np.arange(152, 167, 2) # 7  items
sizes = np.arange(16, 23, 2) # 3  items

Accuracy = np.zeros((len(sizes), len(starts)))
Accuracy.fill(0.489695)

train_subs = np.array(['02','06','07','11','12','13','14','16','17','18','20','21','22','23','24','26'])
n_epo_per_sub = 5440 / len(train_subs)


def epochs_indices(subj_indices):
    return np.hstack([np.arange(s_ind * n_epo_per_sub, (s_ind + 1) * n_epo_per_sub) for s_ind in subj_indices])

for i, start in enumerate(starts):
    for j, window_size in enumerate(sizes):
        accuracy = np.zeros(20)
        train_data = load_epo_data('train', start, window_size)

        for state in range(0, len(accuracy)):
            rs = cross_validation.ShuffleSplit(len(train_subs), n_iter=10, test_size=.15, random_state=state)
            rs = [[train_index, test_index] for train_index, test_index in rs][0]

            train_data_i = proc.select_epochs(train_data, epochs_indices(rs[0]))
            test_data_i = proc.select_epochs(train_data, epochs_indices(rs[1]))
            expected = test_data_i.axes[0]

            # creating a CSP filter, preprocessing train data
            fv_train, filt = preprocess(train_data_i)

            # training LDA
            cfy = proc.lda_train(fv_train)

            # preprocess test data
            fv_test, _ = preprocess(test_data_i, filt)

            # predicting result of the test data
            result = proc.lda_apply(fv_test, cfy)

            result = (np.sign(result) + 1) / 2
            fpr, tpr, thresholds = metrics.roc_curve(expected, result, pos_label=1)
            accuracy[state] = metrics.auc(fpr, tpr)

        acc = accuracy.mean()
        Accuracy[j][i] = acc

        log('LDA Accuracy on start %3i, window size %3i train data %.8f%%' % (start, window_size, acc))

        try:
            data = go.Data([go.Contour(z=Accuracy, x=starts, y=sizes)])
            plot_url = py.plot(data, filename='prediction-roc-accuracy-2-10', fileopt='overwrite', auto_open=False)
        except BaseException as e:
            print 'error occurred when trying to plot: ' + e.message

# save all data to files
savemat('accuracy.mat', {'starts': starts, 'sizes': sizes, 'Accuracy': Accuracy})
np.savetxt('accuracy.txt', Accuracy)
