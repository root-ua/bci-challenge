from NoCSP.find_best_features import find_best_features
from NoCSP.utils import *

folder_name = '../../shrinked_data/'


window_start = 50
window_size = 270
features = [39]

# SVM or RMF or GBM
alg = 'SVM'

W = np.loadtxt('data/weights.txt')
Wt = W.transpose()
find_best_features(alg, window_start, window_size, ICA_W=Wt)

print 'done'