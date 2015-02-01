from NoCSP.find_best_window import find_best_window
from utils import *
import numpy as np


folder_name = '../../shrinked_data/'

starts = np.arange(-10, 75, 10) # 8  items
sizes = np.arange(220, 350, 10) # 13  items
features = [47, 18]
alg = 'GBM'

find_best_window(alg, starts, sizes, features, 0.5, 'top', mean=True)

log('done')
