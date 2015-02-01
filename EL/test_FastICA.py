from sklearn import cross_validation
from sklearn.decomposition import FastICA, PCA
from NoCSP.find_best_features import find_best_features
from NoCSP.load_data import load_data
from NoCSP.utils import log
import numpy as np

folder_name = '../../shrinked_data/'

window_start = 50
window_size = 270
features = [8, 14]

# SVM or RMF or GBM
alg = 'SVM'

data, train_labels = load_data(folder_name, 'train')

# 2d matrix with all training data we have
data_matrix = np.vstack(data[:, :, :])

train_eeg_matrix = data_matrix[:1000, :56]

log('test FastICA started')

# Compute ICA
ica = FastICA(n_components=train_eeg_matrix.shape[1])
ica.fit(train_eeg_matrix)                                  # train on some channels data
s_data = ica.transform(data_matrix[:, :56])                # transform channels to sources data
s_data = np.concatenate((s_data, data_matrix[:, 56:]), 1)  # append additional features
s_data = np.array_split(s_data, data.shape[0])             # split to epochs
# A = ica.mixing_                                          # get estimated mixing matrix

find_best_features(alg, window_start, window_size, s_data, train_labels)

