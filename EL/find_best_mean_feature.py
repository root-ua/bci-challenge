from NoCSP.utils import log
import numpy as np


def find_best_mean_feature(data, labels):
    positive_all = data[labels == 1]
    negative_all = data[labels == 0]

    best_feature = -1
    best_result = -1
    for i in range(0, 56):
        mean_pos = positive_all[:, :, i].mean(0)
        mean_neg = negative_all[:, :, i].mean(0)

        result = np.linalg.norm(mean_pos - mean_neg)
        if result > best_result:
            best_result = result
            best_feature = i
            log('found better feature %i' % i)

    log('best feature %i' % best_feature)