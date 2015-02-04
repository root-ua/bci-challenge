from scipy.io import savemat, loadmat
from NoCSP.load_data import load_data

# folder_name = '../../shrinked_data/'

# data, train_labels = load_data(folder_name, 'train')

# one_epoch = data[0, :, :56]

data = loadmat('sample_data/eeglab_data_epochs_ica.set')

a = 5