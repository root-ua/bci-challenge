from NoCSP.find_best_features import find_best_features


window_start = 60
window_size = 150

# SVM or RMF or GBM
alg = 'SVM'

features, best_score = find_best_features(alg, window_start, window_size, init_features=[38])

print features
print best_score
print 'done'
