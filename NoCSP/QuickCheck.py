from NoCSP.utils import *


folder_name = '../../shrinked_data/'

window_start = 0
window_size = 260
features = [55, 15, 50, 24]
# SVM or RMF or GBM
alg = 'SVM'

data, train_labels = load_data(folder_name, 'train')
data = np.array(get_windows(data, window_start, window_size))
if alg == 'SVM':
    data = preprocessing.scale(data)

log(alg + ' quick test started with features %s, window start %i and window size %i'
    % (str(features), window_start, window_size))

acc, accs = train_test_and_validate(alg, data, train_labels, features, False)

log('%s algorithm: features %s, window start %i, window size %i %s best accuracy is %.8f%%'
    % (alg, str(features), window_start, window_size, str(accs), acc))

print 'done'
