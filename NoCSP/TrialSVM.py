from sklearn import cross_validation, metrics, preprocessing
from NoCSP.load_data import load_data, get_windows
import sklearn.ensemble as ens
from NoCSP.utils import *
from sklearn import svm

folder_name = '../../shrinked_data/'

window_start = 160
window_size = 20

all_features = np.arange(0, 57)

data, train_labels = load_data(folder_name, 'train')
data = np.array(get_windows(data, window_start, window_size))
data = preprocessing.scale(data)

features = [53, 16]

accuracy = np.zeros(5)

log('SVM test started')

for state in range(0, len(accuracy)):
    rs = cross_validation.ShuffleSplit(n_train_subjects, n_iter=10, test_size=.1, random_state=state)
    rs = [[train_index, test_index] for train_index, test_index in rs][0]

    train_data = data[epochs_indices(rs[0])]
    train_y = train_labels[epochs_indices(rs[0])]
    test_data = data[epochs_indices(rs[1])]
    test_y = train_labels[epochs_indices(rs[1])]

    train_y = (train_y * 2) - 1

    train_x = extract_features(train_data, features)
    test_x = extract_features(test_data, features)

    clf = svm.SVC(probability=True)
    clf.fit(train_x, train_y)

    result = clf.predict_proba(test_x)

    #print result[:, 1]
    #print result[:, 0]

    fpr, tpr, thresholds = metrics.roc_curve(test_y, result[:, 1], pos_label=1)
    accuracy[state] = metrics.auc(fpr, tpr)

acc = accuracy.min()

log('SVM accuracy %.8f%%' % acc)
print 'done'
