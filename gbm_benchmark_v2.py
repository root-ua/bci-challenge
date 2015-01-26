from __future__ import division
import numpy as np
import pandas as pd
import sklearn.ensemble as ens
from datetime import datetime

print('Start at {}'.format(datetime.now()))

######################
## index: 47
#column = 'Pz'

## index: 29
#column = 'Cz'

# index: 37
column = 'CP1'

column_index = 37

window_size = 261
algorithm = 'GBM'
run_time = 1
total_sessions = 5
######################

train_subs = ['02', '06', '07', '11', '12', '13', '14', '16', '17', '18', '20', '21', '22', '23', '24', '26']
test_subs = ['01', '03', '04', '05', '08', '09', '10', '15', '19', '25']
train_labels = pd.read_csv('TrainLabels.csv')
submission = pd.read_csv('SampleSubmission.csv')

train = pd.DataFrame(columns=['subject', 'session', 'feedback_num', 'start_pos'] +
                             [column + '_' + s for s in map(str, range(window_size))], index=range(5440))
counter = 0

print 'Loading TRAINING data ...'

data = {}
for train_sub in train_subs:
    for session in range(1, total_sessions+1):
        temp = pd.read_csv('train/Data_S{}_Sess0{}.csv'.format(train_sub, session))
        fb = temp.query('FeedBackEvent == 1', engine='python')['FeedBackEvent']
        counter2 = 0
        for k in fb.index:
                temp2 = temp.loc[int(k):int(k)+(window_size-1), column]
                temp2.index = [column + '_' + s for s in map(str, range(window_size))]
                train.loc[counter, [column + '_' + s for s in map(str, range(window_size))]] = temp2
                train.loc[counter, 'Session'] = session
                train.loc[counter, 'Subject'] = train_sub
                train.loc[counter, 'FeedbackNumber'] = counter2
                train.loc[counter, 'StartPosition'] = k
                counter += 1
                counter2 += 1
    print '   Subject {}'.format(train_sub)

train.to_csv('Train_{}_window{}.csv'.format(column, window_size), ignore_index=True)

###############

test = pd.DataFrame(columns=['subject', 'session', 'feedback_num', 'start_pos'] +
                            [column + '_' + s for s in map(str, range(window_size))], index=range(3400))
print 'Loading TESTING data...'

counter = 0
data = {}
for i in test_subs:
    for j in range(1, total_sessions+1):
        temp = pd.read_csv('test/Data_S' + i + '_Sess0' + str(j) + '.csv')
        fb = temp.query('FeedBackEvent == 1', engine='python')['FeedBackEvent']
        counter2 = 0
        for k in fb.index:
                temp2 = temp.loc[int(k):int(k)+(window_size-1), column]
                temp2.index = [column + '_' + s for s in map(str,range(window_size))]
                test.loc[counter, [column + '_' + s for s in map(str,range(window_size))]] = temp2
                test.loc[counter, 'session'] = j
                test.loc[counter, 'subject'] = i
                test.loc[counter, 'feedback_num'] = counter2
                test.loc[counter, 'start_pos'] = k
                counter += 1
                counter2 += 1
    print '   Subject ', i

test.to_csv('Test_{}_window{}.csv'.format(column, window_size), ignore_index=True)

print 'Training {}...'.format(algorithm)

############
if algorithm == 'GBM':
    cls = ens.GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_features=0.25)
    cls.fit(train, train_labels.values[:, 1].ravel())

############

predictions = cls.predict_proba(test)
predictions = predictions[:, 1]
submission['Prediction'] = predictions

submission.to_csv('{}_t{}_{}_benchmark.csv'.format(algorithm, run_time, column), index=False)

print 'Done!'
print('End at {}'.format(datetime.now()))
