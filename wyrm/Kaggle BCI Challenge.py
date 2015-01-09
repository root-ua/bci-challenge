from __future__ import division
from numpy.ma import vstack, array
import pandas as pd
import numpy
from wyrm.processing import segment_dat
from wyrm.io import convert_mushu_data

FS = 200

train_subs = ['02']
train_labels = pd.read_csv('shrinked_data/TrainLabels.csv')

n_train_sessions = 2 #6

channels = None
data = None
markers = []

last_global_index = 0
for i in train_subs:
    for j in range(1, n_train_sessions):
        temp = pd.read_csv('shrinked_data/train/Data_S' + i + '_Sess0' + str(j) + '.csv')
        if channels is None:
            channels = temp.columns[2:-2]
            data = numpy.zeros((2000, 56))

        # TODO: read data and append to data
        # TODO: !!! convert markers indexes to ms !!!
        markers.append([last_global_index + 11,   'S' + i + '_Sess0' + str(j) + '_FB001'])
        markers.append([last_global_index + 522,  'S' + i + '_Sess0' + str(j) + '_FB002'])
        markers.append([last_global_index + 1033, 'S' + i + '_Sess0' + str(j) + '_FB003'])
        markers.append([last_global_index + 1544, 'S' + i + '_Sess0' + str(j) + '_FB004'])

        last_global_index = markers[-1][0]

        # TODO: append entire file to a data
        # data = vstack()

# TODO: save data in a .mat format

cnt = convert_mushu_data(data, markers, FS, channels)

# Define the markers belonging to class 1 and 2
md = {'class 1': ['S02_Sess01_FB001', 'S02_Sess01_FB002'], 'class 2': ['S02_Sess01_FB003', 'S02_Sess01_FB004']}
# Epoch the data -25ms(5 rows) and +500ms(100 rows) around the markers defined in
# md
epo = segment_dat(cnt, md, [-15, 500])

a = 5
