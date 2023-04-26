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

    for fname in fnames:
        data0 = ssfr.lasp_ssfr.read_ssfr_raw(fname, verbose=False)
        ssfr.vis.quicklook_ssfr_raw(fname, extra_tag='%s_' % fname.split('/')[-2])


if __name__ == '__main__':

    test_process_lasp_ssfr()
