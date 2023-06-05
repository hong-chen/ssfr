import os
import sys
import glob
import datetime
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.image as mpl_img
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cartopy.crs as ccrs
# mpl.use('Agg')


import ssfr


def test_process_lasp_spns():

    fdir = 'data/SPN-S/2023-06-05'

    fname_dif = sorted(glob.glob('%s/Diffuse.txt' % fdir))[0]
    data_dif0 = ssfr.lasp_spn.read_spns(fname=fname_dif)

    fname_tot = sorted(glob.glob('%s/Total.txt' % fdir))[0]
    data_tot0 = ssfr.lasp_spn.read_spns(fname=fname_tot)

    print(data_dif0.data['general_info'])
    print(data_dif0.data['wavelength'])

    # print(fname_dif)
    # print(fname_tot)


if __name__ == '__main__':

    test_process_lasp_spns()
