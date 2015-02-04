from scipy.io import savemat
import numpy as np
from NoCSP.load_data import load_data, apply_ICA_wgts

folder_name = '../../shrinked_data/'

# generate EegLab data
data, train_labels = load_data(folder_name, 'train')
X = data[0:500, :, :56].transpose()

savemat('data/test.mat', {'test': X})

print 'done'