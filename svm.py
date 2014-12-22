import numpy as np
from sklearn import svm
from sklearn import cross_validation
import pandas as pd
from pprint import pprint

print "Starting to load data..."

test = pd.DataFrame.from_csv('test_cz.csv')
train = pd.DataFrame.from_csv('train_cz.csv')
train_labels = pd.read_csv('TrainLabels.csv')
submission = pd.read_csv('SampleSubmission.csv')

print "Data loaded successfully!\n"

clf = svm.SVC(kernel='linear', probability=True)

print "Starting to train..."
rf = clf.fit(train.values[:, :], train_labels.values[:, 1].ravel())
print "Training finished!\n"

print "Predicting ..."
#preds = clf.predict(test)
preds = clf.predict_proba(test.values[:, :])
print "Predicted!\n"

preds = preds[:, 1]
submission['Prediction'] = preds
submission.to_csv('svm_benchmark_2.csv', index=False)
