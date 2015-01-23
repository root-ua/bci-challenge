from sklearn import cross_validation
from utils import *
from wyrm import processing as proc

# best window found
window_start = 160
window_size = 20

log('loading train data')
train_data = load_epo_data('train', window_start, window_size)

log('loading test data')
test_data = load_epo_data('test',  window_start, window_size)

log('creating a CSP filter, preprocessing train data')
fv_train, filt = preprocess(train_data)

log('training LDA')
cfy = proc.lda_train(fv_train)

# preprocess test data
fv_test, _ = preprocess(test_data, filt)

# predicting result of the test data
result = proc.lda_apply(fv_test, cfy)
result = (np.sign(result) + 1) / 2

result_feedbacks = [[test_data.markers[i][1], '%i' % result[i]] for i in np.arange(0, len(result))]

np.savetxt('result_feedbacks.txt', result_feedbacks, fmt='%s,%s')

log('done')