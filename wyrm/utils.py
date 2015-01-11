import numpy as np
from time import gmtime, strftime
from matplotlib import pyplot as plt


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
