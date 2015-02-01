from NoCSP.load_data import apply_ICA_wgts
from NoCSP.utils import *


folder_name = '../../shrinked_data/'


def find_best_features(alg, window_start, window_size, mean=False, existing_data=None, existing_labels=None):
    log('find_best_features started with algorithm %s, window start %i and window size %i'
        % (alg, window_start, window_size))

    all_features = np.arange(0, 57)

    if existing_data is None or existing_labels is None:
        data, train_labels = load_data(folder_name, 'train')
    else:
        data, train_labels = existing_data, existing_labels

    data = np.array(get_windows(data, window_start, window_size))
    if alg == 'SVM':
        data = preprocessing.scale(data)

    features = []
    best_score = 0

    while True:
        best_feature_score = best_score
        best_feature_set = features

        for feature in all_features:
            if feature in features:
                continue

            new_features = features + [feature]

            log('started testing with features ' + str(new_features))

            accs = train_test_and_validate(alg, data, train_labels, new_features)
            acc = accs.mean() if mean else accs.min()

            log('old score: %.8f%%, old features set: %s. new score: %.8f%%, new features: set %s '
                % (best_feature_score, str(best_feature_set), acc, str(new_features)))

            if acc > best_feature_score:
                log(bcolors.OKGREEN + 'new feature is better' + bcolors.ENDC)
                best_feature_score = acc
                best_feature_set = new_features
            else:
                log('new feature is not better')

        print '*************************************'
        log('old score: %.8f%%, old features set: %s. new score: %.8f%%, new features set %s'
            % (best_score, str(features), best_feature_score, str(best_feature_set)))

        if best_feature_score - best_score > 0.002:
            features = best_feature_set
            best_score = best_feature_score
            log("let's repeat once again")
        else:
            log("done")
            break

    return features, best_score
