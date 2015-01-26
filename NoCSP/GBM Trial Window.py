from NoCSP.find_best_window import find_best_window
from utils import *
import numpy as np


folder_name = '../../shrinked_data/'

starts = np.arange(-70, 100, 15) # 11  items
sizes = np.arange(30, 250, 15) # 14  items
features = [47, 13]
alg = 'GBM'

find_best_window(alg, starts, sizes, features)

log('done')
