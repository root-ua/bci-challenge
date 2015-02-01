from NoCSP.find_best_features import find_best_features
from NoCSP.load_data import *
from NoCSP.utils import *

folder_name = '../../shrinked_data/'


window_start = 50
window_size = 270
features = [39]

# SVM or RMF or GBM
alg = 'GBM'

W = np.loadtxt('data/weights.txt')

find_best_features(alg, window_start, window_size, ICA_W=W)

print 'done'