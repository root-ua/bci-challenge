import pandas as pd
from matplotlib import pyplot as plt
from utils import log, plot_csp_pattern
from load_data import load_data
from wyrm.processing import segment_dat
from wyrm import processing as proc
from wyrm import plot
from wyrm.io import convert_mushu_data

plot.beautify()

FS = 200

folder_name = '../../shrinked_data/'
train_labels = pd.read_csv(folder_name + 'TrainLabels.csv')


def preprocess(data, filt=None):
    log('preprocess: copying data')
    dat = data.copy()
    fs_n = dat.fs / 2

    log('preprocess: butter filtering low')
    b, a = proc.signal.butter(5, [13 / fs_n], btype='low')
    dat = proc.filtfilt(dat, b, a)

    log('preprocess: butter filtering high')
    b, a = proc.signal.butter(5, [9 / fs_n], btype='high')
    dat = proc.filtfilt(dat, b, a)

    log('preprocess: subsampling')
    dat = proc.subsample(dat, 50)

    if filt is None:
        log('preprocess: calculate_csp')
        filt, pattern, _ = proc.calculate_csp(dat)

        log('preprocess: plot_csp_pattern')
        plot_csp_pattern(pattern)

    log('preprocess: apply_csp')
    dat = proc.apply_csp(dat, filt)

    log('preprocess: variance and logarithm')
    dat = proc.variance(dat)
    dat = proc.logarithm(dat)
    return dat, filt


def load_epo_data(data_cat):
    log('loading ' + data_cat + ' data')
    data, channels, markers = load_data(FS, folder_name, data_cat)

    log('converting plain data to continuous Data object')
    cnt = convert_mushu_data(data, markers, FS, channels)

    # Define the markers belonging to class 1 and 2
    markers_definitions = None
    if data_cat == 'train':
        markers_definitions = {'class 1': (train_labels.query('Prediction == 0', engine='python')['IdFeedBack']).tolist(),
                           'class 2': (train_labels.query('Prediction == 1', engine='python')['IdFeedBack']).tolist()}
    else:
        # marker classes doesn't matter for test data
        markers_definitions = {'class 1': [m[1] for m in markers], 'class 2': []}

    log('segmenting continuous Data object into epoched data')
    # Epoch the data -25ms(5 rows) and +500ms(100 rows) around the markers defined in markers_definitions
    epo = segment_dat(cnt, markers_definitions, [-15, 500])
    log('train data is ready!')

    return epo


test_data = load_epo_data('test')
train_data = load_epo_data('train')

log('creating a CSP filter, preprocessing train data')
fv_train, filt = preprocess(train_data)
log('preprocessing test data')
fv_test, _ = preprocess(test_data, filt)

log('training LDA')
cfy = proc.lda_train(fv_train)

log('predicting result of the test data')
result = proc.lda_apply(fv_test, cfy)

plt.show()
log('done')

"""fv_test, _ = preprocess(dat_test, filt)

cfy = proc.lda_train(fv_train)
result = proc.lda_apply(fv_test, cfy)
result = (np.sign(result) + 1) / 2
print 'LDA Accuracy %.2f%%' % (len(set(result).intersection(true_labels)) / len(result))"""



