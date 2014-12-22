from __future__ import division
import pandas as pd

train_subs = ['02','06','07','11','12','13','14','16','17','18','20','21','22','23','24','26']
test_subs = ['01','03','04','05','08','09','10','15','19','25']
train_labels = pd.read_csv('../shrinked_data/TrainLabels.csv')
submission = pd.read_csv('../shrinked_data/SampleSubmission.csv')

def load_train_data(offset, nrows):
    print 'loading train data'
    train = pd.DataFrame(columns=['subject','session','feedback_num','start_pos'] + ['Cz_' + s for s in map(str,range(nrows+1))],index=range(5440))
    counter = 0

    for i in train_subs:
        for j in range(1,6):
            temp = pd.read_csv('../shrinked_data/train/Data_S' + i + '_Sess0'  + str(j) + '.csv')
            fb = temp.query('FeedBackEvent == 1',engine='python')['FeedBackEvent']
            feedback_index = 0
            for k in fb.index:
                    start_pos = k + offset
                    temp2 = temp.loc[int(start_pos):int(start_pos)+nrows,'Cz']
                    temp2.index = ['Cz_' + s for s in map(str,range(nrows+1))]
                    train.loc[counter,['Cz_' + s for s in map(str,range(nrows+1))]] = temp2
                    train.loc[counter,'session'] = j
                    train.loc[counter, 'subject'] = i
                    train.loc[counter, 'feedback_num'] = feedback_index
                    train.loc[counter, 'start_pos'] = start_pos
                    counter +=1
                    feedback_index +=1
        print 'subject ', i
    return (train, train_labels.values[:,1].ravel())


def load_test_data(offset, nrows):
    print 'loading test data'
    test = pd.DataFrame(columns=['subject','session','feedback_num','start_pos'] + ['Cz_' + s for s in map(str,range(nrows+1))],index=range(5440))
    counter = 0
    for i in test_subs:
        for j in range(1,6):
            temp = pd.read_csv('../shrinked_data/test/Data_S' + i + '_Sess0'  + str(j) + '.csv')
            fb = temp.query('FeedBackEvent == 1',engine='python')['FeedBackEvent']
            feedback_index = 0
            for k in fb.index:
                    start_pos = k + offset
                    temp2 = temp.loc[int(start_pos):int(start_pos)+nrows,'Cz']
                    temp2.index = ['Cz_' + s for s in map(str,range(nrows+1))]
                    test.loc[counter,['Cz_' + s for s in map(str,range(nrows+1))]] = temp2
                    test.loc[counter,'session'] = j
                    test.loc[counter, 'subject'] = i
                    test.loc[counter, 'feedback_num'] = feedback_index
                    test.loc[counter, 'start_pos'] = start_pos
                    counter +=1
                    feedback_index +=1
        print 'subject ', i
    return test