from NoCSP.find_best_window import find_best_window
from utils import *
import numpy as np


folder_name = '../../shrinked_data/'

starts = np.arange(-70, 100, 15) # 11  items
sizes = np.arange(30, 260, 15) # 14  items
features = [47, 18]
alg = 'GBM'

find_best_window(alg, starts, sizes, features, 0.5)

log('done')
