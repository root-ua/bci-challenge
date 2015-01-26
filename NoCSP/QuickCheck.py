from sklearn import cross_validation, metrics, preprocessing
from NoCSP.load_data import load_data, get_windows
import sklearn.ensemble as ens
from NoCSP.utils import *
from sklearn import svm

folder_name = '../../shrinked_data/'

window_start = 20
window_size = 110
features = [55, 15]
# SVM or RMF or GBM
alg = 'SVM'

data, train_labels = load_data(folder_name, 'train')
data = np.array(get_windows(data, window_start, window_size))
if alg == 'SVM':
    data = preprocessing.scale(data)

log(alg + ' quick test started with features %s, window start %i and window size %i'
    % (str(features), window_start, window_size))

acc = train_test_and_validate(alg, data, train_labels, features, False)

log('%s algorithm: features %s, window start %i, window size %i accuracy is %.8f%%'
    % (alg, str(features), window_start, window_size, acc))

print 'done'
