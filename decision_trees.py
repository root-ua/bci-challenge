import numpy as np
from sklearn import svm
from sklearn import cross_validation
import pandas as pd
from pprint import pprint
import sklearn.ensemble as ens
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.utils.multiclass import type_of_target
from sklearn.externals.six import string_types

print "     DESCISION TREES METHOD!\n"
print "Starting to load data..."

test = pd.DataFrame.from_csv('test_cz.csv')
train = pd.DataFrame.from_csv('train_cz.csv')
train_labels = pd.read_csv('TrainLabels.csv')
submission = pd.read_csv('SampleSubmission.csv')

print "Data loaded successfully!\n"

cls = DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0)

sp = np.array(train_labels.values[:, 1], dtype=int)

#print "Starting to train..."
#rfc.fit(train.values[:, :], train_labels.values[:, 1].ravel())
#print "Training finished!\n"

#k = type_of_target(sp)
#pprint(k)
print "Starting cross-validation..."
scores = cross_val_score(cls, train.values[:, :], sp)
print "Cross-validation accuracy: "
pprint(scores.mean())

#print "Predicting ..."
#preds = rfc.predict_proba(test.values[:, :])
#print "Predicted!\n"

#preds = preds[:, 1]
#submission['Prediction'] = preds
#submission.to_csv('rfc_benchmark.csv', index=False)
