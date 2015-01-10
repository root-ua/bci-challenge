from __future__ import division
from time import gmtime, strftime

import pandas as pd
from utils import log
from load_data import load_train_data
from wyrm.processing import segment_dat
from wyrm.io import convert_mushu_data

FS = 200

folder_name = '../../shrinked_data/'
train_labels = pd.read_csv(folder_name + 'TrainLabels.csv')

log('loading train data')

data, channels, markers = load_train_data(FS, folder_name)

log('train data loaded')
log('converting data to epo object')

cnt = convert_mushu_data(data, markers, FS, channels)

# Define the markers belonging to class 1 and 2
markers_definitions = {'class 1': train_labels.query('Prediction == 0', engine='python')['IdFeedBack'],
                       'class 2': train_labels.query('Prediction == 1', engine='python')['IdFeedBack']}

# Epoch the data -25ms(5 rows) and +500ms(100 rows) around the markers defined in markers_definitions
epo = segment_dat(cnt, markers_definitions, [-15, 500])

log('epo is ready!')

