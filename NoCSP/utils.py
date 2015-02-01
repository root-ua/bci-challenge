from datetime import datetime
import numpy as np
from sklearn import cross_validation, metrics, preprocessing
from NoCSP.load_data import load_data, get_windows
import sklearn.ensemble as ens
from NoCSP.utils import *
from sklearn import svm

n_epo_per_sub = 340
n_train_subjects = 16


def log(message, cool_color=False):
    if cool_color:
        print bcolors.OKGREEN + "at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " " + message + bcolors.ENDC
    else:
        print "at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " " + message


def epochs_indices(subj_indices):
    return np.hstack([np.arange(s_ind * n_epo_per_sub, (s_ind + 1) * n_epo_per_sub) for s_ind in subj_indices])


def extract_features(data, features):
    return [np.hstack(([row[feature] for row in epo for feature in features],
                       epo[0][57], epo[0][58], epo[0][59], epo[0][60]))
            for epo in data]


def train_test_and_validate(alg, data, train_labels, features, quiet=True):
    accuracy = np.zeros(5)

    for state in range(0, len(accuracy)):
        rs = cross_validation.ShuffleSplit(n_train_subjects, n_iter=10, test_size=.05, random_state=state)
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
            clf = ens.GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_features=0.25,
                                                 random_state=0)
        else:
            clf = ens.RandomForestClassifier(n_estimators=500, max_features=0.25, min_samples_split=1,
                                             random_state=0)

        clf.fit(train_x, train_y)

        result = clf.predict_proba(test_x)

        fpr, tpr, thresholds = metrics.roc_curve(test_y, result[:, 1], pos_label=1)
        acc = metrics.auc(fpr, tpr)
        accuracy[state] = acc
        if not quiet:
            log('round %i. accuracy: %.8f%%' % (state, acc))

    return accuracy


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'