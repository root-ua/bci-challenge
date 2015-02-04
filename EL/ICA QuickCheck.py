import random
import gc
from sklearn import cross_validation, preprocessing
from sklearn.decomposition import FastICA
from NoCSP.find_best_features import find_best_features
from NoCSP.load_data import load_data, get_windows
from NoCSP.utils import log, train_test_and_validate
import numpy as np

folder_name = '../../shrinked_data/'

window_start = 60
window_size = 150
features = [34, 53, 17, 14]

# SVM or RMF or GBM
alg = 'GBM'

data, train_labels = load_data(folder_name, 'train')
n_epochs = data.shape[0]

log('quick check started')

# select 1000 random epochs from data
random.seed(3)
train_eeg_matrix = np.vstack(data[random.sample(range(n_epochs), 2500), :, :56])

# Compute ICA
ica = FastICA(n_components=train_eeg_matrix.shape[1], random_state=9)
# train on part of the data
ica.fit(train_eeg_matrix)
del train_eeg_matrix
log('ICA computed')

# 2d matrix with all training data we have
data_matrix = np.vstack(data[:, :, :])
del data

s_data = ica.transform(data_matrix[:, :56])                # transform channels to sources data
s_data = np.concatenate((s_data, data_matrix[:, 56:]), 1)  # append additional features (like subject and session)
s_data = np.array_split(s_data, n_epochs)                  # split to epochs
del data_matrix
log('source data retrieved')

data = np.array(get_windows(s_data, window_start, window_size))
if alg == 'SVM':
    data = preprocessing.scale(data)

log(alg + ' quick test started with features %s, window start %i and window size %i'
    % (str(features), window_start, window_size))

accs = train_test_and_validate(alg, data, train_labels, features, False)

log('%s algorithm: features %s, window start %i, window size %i mean accuracy: %.8f%%, min accuracy: %.8f%% %s'
    % (alg, str(features), window_start, window_size, accs.mean(), accs.min(), str(accs)))

print 'done'