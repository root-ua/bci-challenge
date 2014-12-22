# coding: utf-8

"""
Beating the Benchmark 
BCI Challenge @ Kaggle

__author__ : Abhishek (abhishek4 AT gmail)
"""

import numpy as np
import pandas as pd
from sklearn import ensemble

labels = pd.read_csv('TrainLabels.csv')
submission = pd.read_csv('SampleSubmission.csv')

training_files = []
for filename in labels.IdFeedBack.values:
    training_files.append(filename[:-6])  

testing_files = []
for filename in submission.IdFeedBack.values:
    testing_files.append(filename[:-6])  

for i, filename in enumerate(np.unique(training_files)):
    print i, filename
    path = 'train/Data_' + str(filename) + '.csv'
    df = pd.read_csv(path)
    df = df[df.FeedBackEvent != 0]
    df = df.drop('FeedBackEvent', axis = 1)
    if i == 0:
        train = np.array(df)
    else:
        train = np.vstack((train, np.array(df)))

for i, filename in enumerate(np.unique(testing_files)):
    print i, filename
    path = 'test/Data_' + str(filename) + '.csv'
    df = pd.read_csv(path)
    df = df[df.FeedBackEvent != 0]
    df = df.drop('FeedBackEvent', axis = 1)
    if i == 0:
        test = np.array(df)
    else:
        test = np.vstack((test, np.array(df)))


clf = ensemble.RandomForestClassifier(n_jobs = -1, 
				     n_estimators=150, 
			             random_state=42)

clf.fit(train, labels.Prediction.values)
preds = clf.predict_proba(test)[:,1]

submission['Prediction'] = preds
submission.to_csv('benchmark.csv', index = False)

