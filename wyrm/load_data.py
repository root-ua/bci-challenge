from __future__ import division
import os
import pandas as pd
import numpy as np

from scipy.io import savemat, loadmat
from utils import log


def load_data(fs, folder_name, data_cat):
    mat_file_name = folder_name + "kaggle_BCI_" + data_cat + "_data.mat"

    if os.path.isfile(mat_file_name):
        try:
            log('loading data from .mat file')
            data = loadmat(mat_file_name)

            markers = data['markers']
            markers = [(float(markers[i][0]), markers[i][1]) for i in range(len(markers))]

            return data['data'], data['channels'],  markers
        except BaseException as e:
            log('error occurred when trying to load data from .mat: ' + e.message)

    log('loading data from .csv files')
    sampling_period = 1000 / fs

    subs = ['02','06','07','11','12','13','14','16','17','18','20','21','22','23','24','26'] if data_cat == 'train' \
        else ['01','03','04','05','08','09','10','15','19','25']
    sessions = range(1, 6)

    channels = None
    data = []
    markers = []

    last_global_index = 0
    for i in subs:
        for j in sessions:
            temp = pd.read_csv(folder_name + data_cat + '/Data_S' + i + '_Sess0' + str(j) + '.csv')

            # append current data matrix
            new_data = temp.values[:, 2:-2]
            if channels is None:
                channels = temp.columns[2:-2].values.tolist()
                data = new_data
            else:
                data = np.concatenate((data, new_data))

            # append markers in format: [EVENT_TIME, MARKER_NAME]
            # time means number of milliseconds passed from start of entire matrix
            feedbacks = temp.query('FeedBackEvent == 1', engine='python')['FeedBackEvent']
            feedback_index = 1
            for k in feedbacks.index:
                markers.append([(last_global_index + k) * sampling_period, 'S' + i + '_Sess0' + str(j) + '_FB'
                                + '{:03}'.format(feedback_index)])
                feedback_index += 1

            # update markers index
            last_global_index += new_data.shape[0]

    savemat(mat_file_name, {'data': data, 'channels': channels, 'markers': markers})

    return data, channels, markers



