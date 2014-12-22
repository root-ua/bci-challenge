## author: phalaris
## kaggle bci challenge gbm benchmark

from __future__ import division
import numpy as np
import pandas as pd
import sklearn.ensemble as ens
from load_data import *
from sklearn import cross_validation, svm

for offset in range(0, 10):
    for nrows in range(240,260):
        offset = 1
        nrows = 50
        train_x, train_y = load_train_data(offset, nrows)
        train_x.to_csv('train_x.csv',ignore_index=True)

        #clf = svm.SVC(kernel='linear', C=1)
        clf = ens.GradientBoostingClassifier(n_estimators=500,learning_rate=0.05, max_features=0.25)

        cv = cross_validation.KFold(len(train_y), 5, indices=False, shuffle=True)

        scores = cross_validation.cross_val_score(clf, train_x.values, train_y, cv=cv)

        with open("results.csv", "a") as myfile:
            myfile.write(offset + "," + nrows + ","+scores.mean())

        print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
