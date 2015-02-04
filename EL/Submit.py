import random
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.decomposition import FastICA
from NoCSP.load_data import load_data, get_windows
import sklearn.ensemble as ens
from NoCSP.utils import extract_features, log

submission = pd.read_csv('../../SampleSubmission.csv')
folder_name = '../../shrinked_data/'

window_start = 60
window_size = 150
features = [34, 53, 17, 14]
# SVM or RMF or GBM
alg = 'SVM'

train_data, train_labels = load_data(folder_name, 'train')
train_data = np.array(get_windows(train_data, window_start, window_size))
n_train_epochs = train_data.shape[0]

log('submit started')

# select 1000 random epochs from data
random.seed(3)
train_eeg_matrix = np.vstack(train_data[random.sample(range(n_train_epochs), 2500), :, :56])

# Compute ICA
ica = FastICA(n_components=train_eeg_matrix.shape[1], random_state=9)
# train on part of the data
ica.fit(train_eeg_matrix)
del train_eeg_matrix
log('ICA computed')

# 2d matrix with all training data we have
data_matrix = np.vstack(train_data[:, :, :])

train_data = ica.transform(data_matrix[:, :56])                    # transform channels to sources data
train_data = np.concatenate((train_data, data_matrix[:, 56:]), 1)  # append additional features
train_data = np.array_split(train_data, n_train_epochs)            # split to epochs
del data_matrix
log('train source data retrieved')


test_data, _ = load_data(folder_name, 'test')
test_data = np.array(get_windows(test_data, window_start, window_size))
n_test_epochs = test_data.shape[0]

# 2d matrix with all test data we have
data_matrix = np.vstack(test_data[:, :, :])

test_data = ica.transform(data_matrix[:, :56])                   # transform channels to sources data
test_data = np.concatenate((test_data, data_matrix[:, 56:]), 1)  # append additional features
test_data = np.array_split(test_data, n_test_epochs)             # split to epochs
del data_matrix, ica
log('test source data retrieved')

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
