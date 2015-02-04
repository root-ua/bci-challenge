from NoCSP.utils import *


folder_name = '../../shrinked_data/'

window_start = 50
window_size = 270
features = [19]
# SVM or RMF or GBM
alg = 'SVM'

data, train_labels = load_data(folder_name, 'train')
data = np.array(get_windows(data, window_start, window_size))
if alg == 'SVM':
    data = preprocessing.scale(data)

log(alg + ' quick test started with features %s, window start %i and window size %i'
    % (str(features), window_start, window_size))

accs = train_test_and_validate(alg, data, train_labels, features, False)

log('%s algorithm: features %s, window start %i, window size %i mean accuracy: %.8f%%, min accuracy: %.8f%% %s'
    % (alg, str(features), window_start, window_size, accs.mean(), accs.min(), str(accs)))

print 'done'
