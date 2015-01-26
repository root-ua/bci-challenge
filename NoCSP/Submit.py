import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from NoCSP.load_data import load_data, get_windows
import sklearn.ensemble as ens
from NoCSP.utils import extract_features

submission = pd.read_csv('../../SampleSubmission.csv')
folder_name = '../../shrinked_data/'

window_start = 20
window_size = 110
features = [47]
# SVM or RMF or GBM
alg = 'GBM'

test_data, _ = load_data(folder_name, 'test')
test_data = np.array(get_windows(test_data, window_start, window_size))

train_data, train_labels = load_data(folder_name, 'train')
train_data = np.array(get_windows(train_data, window_start, window_size))

if alg == 'SVM':
    merged_data = np.vstack((train_data, test_data))
    train_lgth = len(train_data)

    preprocessed_data = preprocessing.scale(merged_data)
    train_data = preprocessed_data[:train_lgth]
    test_data = preprocessed_data[train_lgth:]

    train_labels = (train_labels * 2) - 1

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

clf.fit(train_x, train_labels)
result = clf.predict_proba(test_x)

submission['Prediction'] = result[:, 1]
submission.to_csv('../../submission.csv', index=False)

print 'done'
