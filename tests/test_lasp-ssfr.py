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


def test_process_lasp_ssfr():

    fdir = 'data/SSFR/2023-10-02'

    # for fdir0  in sorted(glob.glob('%s/*' % fdir)):
    for fdir0  in [fdir]:

        fnames = ssfr.util.get_all_files(fdir0)

        ssfr0 = ssfr.lasp_ssfr.read_ssfr(fnames, dark_corr_mode='interp')

        dset = 'data_raw'
        data0 = getattr(ssfr0, dset)
        extra_tag = '%s_dset-raw_' % os.path.basename(fdir0)
        ssfr.vis.quicklook_mpl_ssfr_raw(data0, extra_tag=extra_tag)

        for i in range(ssfr0.Ndset):
            dset = 'data%d' % i
            extra_tag = '%s_dset-%d_' % (os.path.basename(fdir0), i)
            data0 = getattr(ssfr0, dset)
            ssfr.vis.quicklook_mpl_ssfr_raw(data0, extra_tag=extra_tag)


if __name__ == '__main__':

    test_process_lasp_ssfr()
