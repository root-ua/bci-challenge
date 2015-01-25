from datetime import datetime
import numpy as np

n_epo_per_sub = 340
n_train_subjects = 16


def log(message, cool_color=False):
    if cool_color:
        print bcolors.OKGREEN + "at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " " + message + bcolors.ENDC
    else:
        print "at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " " + message


def epochs_indices(subj_indices):
    return np.hstack([np.arange(s_ind * n_epo_per_sub, (s_ind + 1) * n_epo_per_sub) for s_ind in subj_indices])


def extract_features(data, features):
    return [np.hstack([row[feature] for row in epo for feature in features]) for epo in data]


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'