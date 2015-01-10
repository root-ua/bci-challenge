from __future__ import division
from time import gmtime, strftime


def log(message):
    print "at " + str(strftime("%Y-%m-%d %H:%M:%S", gmtime())) + " " + message
