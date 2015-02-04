import random
import gc
from sklearn import cross_validation, preprocessing
from sklearn.decomposition import FastICA
from find_best_mean_feature import find_best_mean_feature
from NoCSP.find_best_features import find_best_features
from NoCSP.load_data import load_data, get_windows
from NoCSP.utils import log, train_test_and_validate
import numpy as np

folder_name = '../../shrinked_data/'

window_start = 50
window_size = 270

log('quick feature check started')

data, train_labels = load_data(folder_name, 'train')
n_epochs = data.shape[0]

data = preprocessing.scale(data)

find_best_mean_feature(data, train_labels)

log('done')