from NoCSP.find_best_window import find_best_window
from utils import *
import numpy as np


folder_name = '../../shrinked_data/'

starts = np.arange(-10, 75, 10) # 8  items
sizes = np.arange(240, 275, 10) # 4  items
features = [55, 15]
alg = 'SVM'

find_best_window(alg, starts, sizes, features, 0.5, 'top')

log('done')
