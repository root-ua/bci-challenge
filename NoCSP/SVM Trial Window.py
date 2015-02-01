from NoCSP.find_best_window import find_best_window
from utils import *
import numpy as np


folder_name = '../../shrinked_data/'

starts = np.arange(30, 85, 10) # 4  items
sizes = np.arange(100, 235, 10) # 13  items
features = [39, 0, 40]
alg = 'SVM'

find_best_window(alg, starts, sizes, features, 0.6, '2')

log('done')
