from __future__ import division
import os
import pandas as pd
from scipy.io import savemat, loadmat
from pprint import pprint
import numpy as np

init_start = 100
init_size = 500


def load_data(folder_name, data_cat, subjects=None):
    mat_file_name = folder_name + data_cat + "_data.mat"
    train_labels = pd.read_csv(folder_name + 'TrainLabels.csv')

    if os.path.isfile(mat_file_name):
        try:
            # loading data from .mat file
            data = loadmat(mat_file_name)
            return data['X'], data['Y'][0]

        except BaseException as e:
            print 'error occurred when trying to load data from .mat: ' + e.message

    # loading data from .csv files
    train_subs = ['02', '06','07','11','12','13','14','16','17','18','20','21','22','23','24','26']
    test_subs = ['01','03','04','05','08','09','10','15','19','25']

    subs = subjects
    if subs is None:
        subs = train_subs if data_cat == 'train' else test_subs

    sessions = range(1, 6)

    X = []
    Y = train_labels.Prediction.values

    for i in subs:
        for j in sessions:
            temp = pd.read_csv(folder_name + data_cat + '/Data_S' + i + '_Sess0' + str(j) + '.csv')

            # get entire data matrix
            new_data = temp.values[:, 2:-1]

            feedback_num = 0
            # take windows
            feedbacks = temp.query('FeedBackEvent == 1', engine='python')['FeedBackEvent']
            for k in feedbacks.index:
                window = np.column_stack((new_data[k - init_start:k + init_size], arange_column(i),
                                          arange_column(j), arange_column(k), arange_column(feedback_num)))

                X.append(window)
                feedback_num += 1

    savemat(mat_file_name, {'X': X, 'Y': Y})

    return X, Y


def get_windows(data, window_start, window_size):

    new_data = []
    for d in data:
        new_data.append(d[init_start + window_start:init_start + window_start + window_size - 1])

    return new_data


def arange_column(val):
    column = np.arange(init_start + init_size)
    column.fill(val)
    return column