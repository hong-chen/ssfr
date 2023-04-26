import os
import sys
import glob
import datetime
import h5py
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
import numpy as np
from scipy import interpolate
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


def test():

    fnames0 = {
            'dark':'/data/hong/mygit/ssfr/examples/data/camp2ex/2019/p3/raw/ssfr/spc10380.OSA2',\
             'cal':'/data/hong/mygit/ssfr/examples/data/camp2ex/2019/p3/raw/ssfr/spc10381.OSA2',\
            }

    fnames1 = {
            'dark':'/data/hong/mygit/ssfr/examples/data/camp2ex/2019/p3/raw/ssfr/spc10382.OSA2',\
             'cal':'/data/hong/mygit/ssfr/examples/data/camp2ex/2019/p3/raw/ssfr/spc10383.OSA2',\
            }

    ssfr.vis.quicklook_ssfr_raw(fnames0['dark'], extra_tag='CAMP2Ex_')
    ssfr.vis.quicklook_ssfr_raw(fnames0['cal'] , extra_tag='CAMP2Ex_')
    ssfr.vis.quicklook_ssfr_raw(fnames1['dark'], extra_tag='CAMP2Ex_')
    ssfr.vis.quicklook_ssfr_raw(fnames1['cal'] , extra_tag='CAMP2Ex_')

    data0_cal  = ssfr.nasa_ssfr.read_ssfr_raw(fnames0['cal'] , verbose=False)
    data0_dark = ssfr.nasa_ssfr.read_ssfr_raw(fnames0['dark'], verbose=False)

    data1_cal  = ssfr.nasa_ssfr.read_ssfr_raw(fnames1['cal'] , verbose=False)
    data1_dark = ssfr.nasa_ssfr.read_ssfr_raw(fnames1['dark'], verbose=False)


if __name__ == '__main__':

    test()
