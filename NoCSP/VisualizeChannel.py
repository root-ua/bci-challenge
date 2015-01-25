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
subset_size = 20

data, train_labels = load_data(folder_name, 'train')
data = get_windows(data, window_start, window_size)
train_labels = np.array(train_labels)
data = preprocessing.scale(data)

# extract exact channel from all examples
log('extracting data')
data = np.array(extract_features(data, [feature]))

# take some positive and negative examples separately
positive_subset = data[train_labels == 1][:subset_size]
negative_subset = data[train_labels == 0][:subset_size]

log('generating plot data')
traces = []
for index in range(0, subset_size):
    positive_y = positive_subset[index]
    negative_y = negative_subset[index]

    vertical_margin = 1 - index

    positive_y -= positive_y.mean()
    positive_y *= 4
    positive_y -= vertical_margin

    negative_y -= negative_y.mean()
    negative_y *= 4
    negative_y -= vertical_margin

    x = range(0, len(positive_subset[0]))
    traces.append(go.Scatter(
        x=x,
        y=positive_y,
        mode='lines',
        name='ch. %i, %i' % (feature, index),
        line=go.Line(color='rgb(0,128,0)', width=1)))

    traces.append(go.Scatter(
        x=x,
        y=negative_y - 25,
        mode='lines',
        name='ch. %i, %i' % (feature, index),
        line=go.Line(color='rgb(153,0,0)', width=1)))

log('plotting')
plot_url = py.plot(go.Data(traces), filename='visualization of channel %i' % feature, fileopt='overwrite')

log('done', True)
