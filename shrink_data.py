from __future__ import division
import pandas as pd

train_subs = ['02','06','07','11','12','13','14','16','17','18','20','21','22','23','24','26']
test_subs = ['01','03','04','05','08','09','10','15','19','25']

def shrink_files(subjects, folder):

    for subj in subjects:
        for session in range(1, 6):
            oldFileName = '../' + folder + '/Data_S' + subj + '_Sess0' + str(session) + '.csv'
            newFileName = '../shrinked_data/' + folder + '/Data_S' + subj + '_Sess0' + str(session) + '.csv'

            # whole file
            temp = pd.read_csv(oldFileName)
            # feedbackss
            fb = temp.query('FeedBackEvent == 1', engine='python')['FeedBackEvent']

            concatenated = None

            # join blocks of 510 lines near each feedback(10 before and 500 after)
            for k in fb.index:
                start_index = int(k) - 10
                temp2 = temp.loc[start_index:int(k) + 500]
                if concatenated is None:
                    concatenated = temp2
                else:
                    concatenated = pd.concat((concatenated, temp2))

            concatenated.to_csv(newFileName, ignore_index=True)
            print 'subject ', subj, ', session ', session

# shrink_files(train_subs, 'train')
shrink_files(test_subs, 'test')

print 'done'
