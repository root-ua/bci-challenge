from scipy import signal
from sklearn import cross_validation, metrics, preprocessing
from NoCSP.load_data import load_data, get_windows
import sklearn.ensemble as ens
from NoCSP.utils import *
from sklearn import svm
import plotly.plotly as py
import plotly.graph_objs as go

folder_name = '../../shrinked_data/'

window_start = 20
window_size = 110

feature = 37
subset_size = 500

data, train_labels = load_data(folder_name, 'train')
data = get_windows(data, window_start, window_size)
data = preprocessing.scale(data)
train_labels = np.array(train_labels)

# extract exact channel from all examples
log('extracting data')
data = np.array(extract_features(data, [feature]))


# take some positive and negative examples separately
positive_all = data[train_labels == 1]
negative_all = data[train_labels == 0]
positive_subset = positive_all[:subset_size]
negative_subset = negative_all[:subset_size]

mean_positive = positive_all.mean(0)
mean_negative = negative_all.mean(0)


# log('generating plot data')
# positive_means_dif = []
# negative_means_dif = []

# for index in range(0, subset_size):
#     positive_y = positive_subset[index]
#     negative_y = negative_subset[index]
#
#     positive_means_dif.append((positive_y.mean() - absolute_mean) ** 2 - positive_dif_mean)
#     negative_means_dif.append((negative_y.mean() - absolute_mean) ** 2 - negative_dif_mean)

traces = [
    go.Scatter(
        x=range(0, len(mean_positive)),
        y=mean_positive,
        mode='lines',
        name='positive mean',
        line=go.Line(color='rgb(0,128,0)')),
    go.Scatter(
        x=range(0, len(mean_positive)),
        y=mean_negative,
        mode='lines',
        name='negative mean',
        line=go.Line(color='rgb(153,0,0)'))]


log('plotting')
plot_url = py.plot(go.Data(traces), filename='visualization of mean channel %i' % feature, fileopt='overwrite')

log('done', True)
