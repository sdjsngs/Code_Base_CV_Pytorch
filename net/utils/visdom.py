"""
torch visdom
"""
from visdom import Visdom
import numpy as np
import math
import os.path
import getpass
from sys import platform as _platform
from six.moves import urllib

viz = Visdom()
assert viz.check_connection()

try:
    import matplotlib.pyplot as plt
    plt.plot([1, 23, 2, 4])
    plt.ylabel('some numbers')
    viz.matplot(plt)
except BaseException as err:
    print('Skipped matplotlib example')
    print('Error message: ', err)



if __name__=="__main__":
    print("torch visdom")
