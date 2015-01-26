from NoCSP.find_best_features import find_best_features


window_start = 0
window_size = 260

# SVM or RMF or GBM
alg = 'GBM'

features, best_score = find_best_features(alg, window_start, window_size)

print features
print best_score
print 'done'
