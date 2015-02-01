from NoCSP.find_best_features import find_best_features


window_start = 50
window_size = 270

# SVM or RMF or GBM
alg = 'GBM'

features, best_score = find_best_features(alg, window_start, window_size, mean=True)

print features
print best_score
print 'done'
