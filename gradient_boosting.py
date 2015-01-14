import numpy as np
from sklearn import svm
from sklearn import cross_validation
import pandas as pd
from pprint import pprint
import sklearn.ensemble as ens
from sklearn.cross_validation import cross_val_score

print "     GRADIENT BOOSTING METHOD!\n"
print "Starting to load data..."

test = pd.DataFrame.from_csv('test_cz.csv')
train = pd.DataFrame.from_csv('train_cz.csv')
train_labels = pd.read_csv('TrainLabels.csv')
submission = pd.read_csv('SampleSubmission.csv')

print "Data loaded successfully!\n"

sp = np.array(train_labels.values[:, 1].ravel(), dtype=int)

rfc = ens.GradientBoostingClassifier(n_estimators=500,learning_rate=0.05, max_features=0.25)

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
submission.to_csv('gbmv2_benchmark.csv', index=False)
