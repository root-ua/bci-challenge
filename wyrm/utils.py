from time import gmtime, strftime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from load_data import load_data
from wyrm.processing import segment_dat
from wyrm import processing as proc
from wyrm.io import convert_mushu_data

FS = 200

folder_name = '../../shrinked_data/'
train_labels = pd.read_csv(folder_name + 'TrainLabels.csv')


def log(message):
    print "at " + str(strftime("%Y-%m-%d %H:%M:%S", gmtime())) + " " + message


def plot_csp_pattern(a):
    # get symmetric min/max values for the color bar from first and last column of the pattern
    maxv = np.max(np.abs(a[:, [0, -1]]))
    minv = -maxv

    im_args = {'interpolation' : 'None',
           'vmin' : minv,
           'vmax' : maxv
           }

    # plot
    ax1 = plt.subplot2grid((1,11), (0,0), colspan=5)
    ax2 = plt.subplot2grid((1,11), (0,5), colspan=5)
    ax3 = plt.subplot2grid((1,11), (0,10))

    ax1.imshow(a[:, 0].astype(int).reshape(7, 8), **im_args)
    ax1.set_title('Pinky')

    ax = ax2.imshow(a[:, -1].astype(int).reshape(7, 8), **im_args)
    ax2.set_title('Tongue')

    plt.colorbar(ax, cax=ax3)
    plt.tight_layout()
    plt.show()


def preprocess(data, filt=None):
    # copying data
    dat = data.copy()
    fs_n = dat.fs / 2.0

    # butter filtering low
    b, a = proc.signal.butter(5, [13 / fs_n], btype='low')
    #dat = proc.filtfilt(dat, b, a)

    # butter filtering high
    b, a = proc.signal.butter(4, [9 / fs_n], btype='high')
    dat = proc.filtfilt(dat, b, a)

    # subsampling
    #dat = proc.subsample(dat, 50)

    if filt is None:
        # calculate_csp
        filt, pattern, _ = proc.calculate_csp(dat)

        # plot_csp_pattern
        #plot_csp_pattern(pattern)

    # apply_csp
    dat = proc.apply_csp(dat, filt)

    # variance and logarithm
    dat = proc.variance(dat)
    #dat = proc.logarithm(dat)
    return dat, filt


def load_epo_data(data_cat, n_before=-3, n_len=100, subjects=None):
    # loading 'data_cat' data
    data, channels, markers = load_data(FS, folder_name, data_cat, subjects)

    # converting plain data to continuous Data object
    cnt = convert_mushu_data(data, markers, FS, channels)

    # Define the markers belonging to class 1 and 2
    markers_definitions = None
    if data_cat == 'train':
        markers_definitions = {'class 1': (train_labels.query('Prediction == 0', engine='python')['IdFeedBack']).tolist(),
                           'class 2': (train_labels.query('Prediction == 1', engine='python')['IdFeedBack']).tolist()}
    else:
        # marker classes doesn't matter for test data
        markers_definitions = {'class 1': [m[1] for m in markers], 'class 2': []}

    # segmenting continuous Data object into epoched data
    # Epoch the data -25ms(5 rows) and +500ms(100 rows) around the markers defined in markers_definitions
    return segment_dat(cnt, markers_definitions, [n_before*5, (n_before + n_len)*5])