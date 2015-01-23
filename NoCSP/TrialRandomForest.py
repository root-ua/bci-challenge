from sklearn import cross_validation, metrics, preprocessing
from NoCSP.load_data import load_data, get_windows
import sklearn.ensemble as ens
from NoCSP.utils import *
from sklearn import svm

folder_name = '../../shrinked_data/'

window_start = 160
window_size = 20

all_features = np.arange(0, 57)

data, train_labels = load_data(folder_name, 'train')
data = np.array(get_windows(data, window_start, window_size))

features = []
best_score = 0

while True:
    best_feature_score = best_score
    best_feature_set = features

    for feature in all_features:
        if feature in features:
            continue

        new_features = features + [feature]

        accuracy = np.zeros(5)

        log('started testing with features ' + str(new_features))

        for state in range(0, len(accuracy)):
            rs = cross_validation.ShuffleSplit(n_train_subjects, n_iter=10, test_size=.1, random_state=state)
            rs = [[train_index, test_index] for train_index, test_index in rs][0]

            train_data = data[epochs_indices(rs[0])]
            train_y = train_labels[epochs_indices(rs[0])]
            test_data = data[epochs_indices(rs[1])]
            test_y = train_labels[epochs_indices(rs[1])]

            #add for SVM
            #train_y = (train_y * 2) - 1

            train_x = extract_features(train_data, new_features)
            test_x = extract_features(test_data, new_features)

            clf = ens.RandomForestClassifier(n_estimators=500, max_features=0.25, min_samples_split=1, random_state=0)
            #clf = svm.SVC(probability=True)

            clf.fit(train_x, train_y)

            result = clf.predict_proba(test_x)

            fpr, tpr, thresholds = metrics.roc_curve(test_y, result[:, 1], pos_label=1)
            accuracy[state] = metrics.auc(fpr, tpr)

        acc = accuracy.min()

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

    if best_feature_score - best_score > 0.02:
        features = best_feature_set
        best_score = best_feature_score
        log("let's repeat once again")
    else:
        log("done")
        break


print features
print best_score
print 'done'
