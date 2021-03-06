import numpy as np
from sklearn import svm
from sklearn import cross_validation
import pandas as pd
from pprint import pprint
import sklearn.ensemble as ens
from sklearn.cross_validation import cross_val_score

print "     RANDOM FOREST METHOD!\n"
print "Starting to load data..."

test = pd.DataFrame.from_csv('test_cz.csv')
train = pd.DataFrame.from_csv('train_cz.csv')
train_labels = pd.read_csv('TrainLabels.csv')
submission = pd.read_csv('SampleSubmission.csv')

print "Data loaded successfully!\n"

sp = np.array(train_labels.values[:, 1].ravel(), dtype=int)

rfc = ens.RandomForestClassifier(n_estimators=500, max_features=0.25, min_samples_split=1, random_state=0)

print "Starting to train..."
rfc.fit(train.values[:, :], sp)
print "Training finished!\n"

print "Starting cross-validation..."
scores = cross_val_score(rfc, train.values[:, :], sp)
print "Cross-validation accuracy: {}".format(scores.mean())

print "Predicting ..."
preds = rfc.predict_proba(test.values[:, :])
print "Predicted!\n"

preds = preds[:, 1]
submission['Prediction'] = preds
submission.to_csv('rfc_benchmark.csv', index=False)
