import random
import gc
from sklearn import cross_validation
from sklearn.decomposition import FastICA, PCA
from NoCSP.find_best_features import find_best_features
from NoCSP.load_data import load_data
from NoCSP.utils import log
import numpy as np

folder_name = '../../shrinked_data/'

window_start = 50
window_size = 270

# SVM or RMF or GBM
alg = 'SVM'
seed = 3

log('test FastICA with random epochs and seed %i started' % seed)

data, train_labels = load_data(folder_name, 'train')
n_epochs = data.shape[0]

# select 1000 random epochs from data
random.seed(seed)
train_eeg_matrix = np.vstack(data[random.sample(range(n_epochs), 2500), :, :56])

# Compute ICA
ica = FastICA(n_components=train_eeg_matrix.shape[1], random_state=9)
# train on part of the data
ica.fit(train_eeg_matrix)
del train_eeg_matrix

# 2d matrix with all training data we have
data_matrix = np.vstack(data[:, :, :])
del data

s_data = ica.transform(data_matrix[:, :56])                # transform channels to sources data
s_data = np.concatenate((s_data, data_matrix[:, 56:]), 1)  # append additional features (like subject and session)
s_data = np.array_split(s_data, n_epochs)             # split to epochs
del data_matrix

find_best_features(alg, window_start, window_size, False, s_data, train_labels)

log('done')