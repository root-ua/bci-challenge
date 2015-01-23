from scipy.io import savemat
from sklearn.ensemble import RandomForestClassifier
from utils import *
from wyrm import processing as proc
from sklearn import cross_validation, metrics
import plotly.plotly as py
import plotly.graph_objs as go


starts = [160]#np.arange(130, 191, 15) # 4  items
sizes = [20]#np.arange(30, 91, 15) # 4  items

Accuracy = np.zeros((len(sizes), len(starts)))
Accuracy.fill(0.54588)

for i, start in enumerate(starts):
    for j, window_size in enumerate(sizes):
        train_data = load_epo_data('train', start, window_size)

        accuracy = np.zeros(10)

        for state in range(0, len(accuracy)):
            rs = cross_validation.ShuffleSplit(train_data.data.shape[0], n_iter=10, test_size=.15, random_state=state)
            rs = [[train_index, test_index] for train_index, test_index in rs][0]

            test_data_i = proc.select_epochs(train_data, rs[1])
            train_data_i = proc.select_epochs(train_data, rs[0])
            #expected = test_data_i.axes[0]
            expected = train_data_i.axes[0]

            # creating a CSP filter, preprocessing train data
            fv_train, filt = preprocess(train_data_i)

            # training LDA
            cfy = proc.lda_train(fv_train)

            # preprocess test data
            #fv_test, _ = preprocess(test_data_i, filt)

            # predicting result of the test data
            result = proc.lda_apply(fv_train, cfy)
            result = (np.sign(result) + 1) / 2

            fpr, tpr, thresholds = metrics.roc_curve(expected, result, pos_label=1)
            accuracy[state] = metrics.auc(fpr, tpr)

        Accuracy[j][i] = accuracy.mean()

        log('LDA Accuracy on start %3i, window size %3i train data %.8f%%' % (start, window_size, Accuracy[j][i]))
        #log('LDA Accuracy on start %3i, window size %3i train data %.8f%%' % (start, window_size, accuracy))

        try:
            data = go.Data([go.Contour(z=Accuracy, x=starts, y=sizes)])
            #plot_url = py.plot(data, filename='prediction-roc-accuracy-8', fileopt='overwrite', auto_open=False)
        except BaseException as e:
            print 'error occurred when trying to load data from .mat: ' + e.message

# save all data to files
savemat('accuracy.mat', {'starts': starts, 'sizes': sizes, 'Accuracy': Accuracy})
np.savetxt('accuracy.txt', Accuracy)
