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


def test_process_lasp_ssfr():

    fdir = 'data/20221208_ssfr-lasp_pri-cal'

    fnames = ssfr.util.get_all_files(fdir)
    print(fnames)
    sys.exit()

    fnames0 = {
            'dark':'data/20221208_ssfr-lasp_pri-cal/20221208_CALIBRATION_75_150/20221208_spc00001.SKS',\
            'cal' :'data/20221208_ssfr-lasp_pri-cal/20221208_CALIBRATION_75_150/20221208_spc00002.SKS',
            }

    fnames1 = {
            'dark':'data/20221208_ssfr-lasp_pri-cal/20221208_CALIBRATION_250_500/20221208_spc00001.SKS',\
            'cal' :'data/20221208_ssfr-lasp_pri-cal/20221208_CALIBRATION_250_500/20221208_spc00002.SKS',
            }

    ssfr.vis.quicklook_ssfr_raw(fnames0['dark'], extra_tag='Skywatch-test_075-150_')
    ssfr.vis.quicklook_ssfr_raw(fnames0['cal'] , extra_tag='Skywatch-test_075-150_')
    ssfr.vis.quicklook_ssfr_raw(fnames1['dark'], extra_tag='Skywatch-test_250-500_')
    ssfr.vis.quicklook_ssfr_raw(fnames1['cal'] , extra_tag='Skywatch-test_250-500_')

    data0_cal  = ssfr.lasp_ssfr.read_ssfr_raw(fnames0['cal'] , verbose=False)
    data0_dark = ssfr.lasp_ssfr.read_ssfr_raw(fnames0['dark'], verbose=False)

    data1_cal  = ssfr.lasp_ssfr.read_ssfr_raw(fnames1['cal'] , verbose=False)
    data1_dark = ssfr.lasp_ssfr.read_ssfr_raw(fnames1['dark'], verbose=False)


if __name__ == '__main__':

    test_process_lasp_ssfr()
