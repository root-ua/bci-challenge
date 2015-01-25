from scipy import signal
from sklearn import cross_validation, metrics, preprocessing
from NoCSP.load_data import load_data, get_windows
import sklearn.ensemble as ens
from NoCSP.utils import *
from sklearn import svm

folder_name = '../../shrinked_data/'

window_start = 20
window_size = 135
features = [47]
# SVM or RMF or GBM
alg = 'RMF'

data, train_labels = load_data(folder_name, 'train')
data = np.array(get_windows(data, window_start, window_size))
if alg == 'SVM':
    data = preprocessing.scale(data)

# TODO: change to 5
accuracy = np.zeros(5)

log(alg + ' quick test started with features %s, window start %i and window size %i'
    % (str(features), window_start, window_size))

for state in range(0, len(accuracy)):
    rs = cross_validation.ShuffleSplit(n_train_subjects, n_iter=10, test_size=.1, random_state=state)
    rs = [[train_index, test_index] for train_index, test_index in rs][0]

    train_data = data[epochs_indices(rs[0])]
    train_y = train_labels[epochs_indices(rs[0])]
    test_data = data[epochs_indices(rs[1])]
    test_y = train_labels[epochs_indices(rs[1])]

    if alg == 'SVM':
        train_y = (train_y * 2) - 1

    train_x = extract_features(train_data, features)
    test_x = extract_features(test_data, features)

    if alg == 'SVM':
        clf = svm.SVC(probability=True)
    elif alg == 'GBM':
        clf = ens.GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_features=0.25)
    else:
        clf = ens.RandomForestClassifier(n_estimators=500, max_features=0.25, min_samples_split=1, random_state=0)

    clf.fit(train_x, train_y)

    result = clf.predict_proba(test_x)

    fpr, tpr, thresholds = metrics.roc_curve(test_y, result[:, 1], pos_label=1)
    acc = metrics.auc(fpr, tpr)
    accuracy[state] = acc
    log('round %i. accuracy: %.8f%%' % (state, acc))

acc = accuracy.min()

log('accuracy with %s algorithm is %.8f%%' % (alg, acc))
print 'done'
