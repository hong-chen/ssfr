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

    fdir = 'data/SSFR/2023-10-06'

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


def test_lasp_ssfr_skywatch_zenith():

    fdir = 'data/SSFR/2023-10-06'

    fnames = ssfr.util.get_all_files(fdir)

    ssfr0 = ssfr.lasp_ssfr.read_ssfr(fnames, dark_corr_mode='interp')


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.scatter(ssfr0.data0['tmhr'], ssfr0.data0['spectra'][:, 100, 0], s=6, c='black', lw=0.0)
        ax1.scatter(ssfr0.data1['tmhr'], ssfr0.data1['spectra'][:, 100, 0], s=6, c='gray' , lw=0.0)

        ax1.scatter(ssfr0.data0['tmhr'], ssfr0.data0['spectra'][:, 100, 1], s=6, c='red'    , lw=0.0)
        ax1.scatter(ssfr0.data1['tmhr'], ssfr0.data1['spectra'][:, 100, 1], s=6, c='magenta', lw=0.0)

        ax1.set_ylim((-35000, 35000))

        ax1.set_xlabel('Time [Hour]')
        ax1.set_ylabel('Counts')
        ax1.set_title('Zenith (LC6 + CAMP2Ex Zenith Cable)')
        #\--------------------------------------------------------------/#

        patches_legend = [
                         mpatches.Patch(color='black', label='Zenith Silicon (%d ms)' % ssfr0.data0['info']['int_time']['Zenith Silicon']), \
                         mpatches.Patch(color='gray' , label='Zenith Silicon (%d ms)' % ssfr0.data1['info']['int_time']['Zenith Silicon']), \
                         mpatches.Patch(color='red'     , label='Zenith InGaAs (%d ms)' % ssfr0.data0['info']['int_time']['Zenith InGaAs']), \
                         mpatches.Patch(color='magenta' , label='Zenith InGaAs (%d ms)' % ssfr0.data1['info']['int_time']['Zenith InGaAs']), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper left', fontsize=16)

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % (_metadata['Function']), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
    #\----------------------------------------------------------------------------/#



def test_lasp_ssfr_skywatch_nadir():

    fdir = 'data/SSFR/2023-10-06'

    fnames = ssfr.util.get_all_files(fdir)

    ssfr0 = ssfr.lasp_ssfr.read_ssfr(fnames, dark_corr_mode='interp')


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.scatter(ssfr0.data0['tmhr'], ssfr0.data0['spectra'][:, 100, 2], s=6, c='black', lw=0.0)
        ax1.scatter(ssfr0.data1['tmhr'], ssfr0.data1['spectra'][:, 100, 2], s=6, c='gray', lw=0.0)

        ax1.scatter(ssfr0.data0['tmhr'], ssfr0.data0['spectra'][:, 100, 3], s=6, c='red'    , lw=0.0)
        ax1.scatter(ssfr0.data1['tmhr'], ssfr0.data1['spectra'][:, 100, 3], s=6, c='magenta', lw=0.0)

        ax1.set_ylim((-35000, 35000))

        ax1.set_xlabel('Time [Hour]')
        ax1.set_ylabel('Counts')
        ax1.set_title('Nadir (LC4 + Stainless Steel Cable)')
        #\--------------------------------------------------------------/#

        patches_legend = [
                         mpatches.Patch(color='black', label='Nadir Silicon (%d ms)' % ssfr0.data0['info']['int_time']['Nadir Silicon']), \
                         mpatches.Patch(color='gray' , label='Nadir Silicon (%d ms)' % ssfr0.data1['info']['int_time']['Nadir Silicon']), \
                         mpatches.Patch(color='red'     , label='Nadir InGaAs (%d ms)' % ssfr0.data0['info']['int_time']['Nadir InGaAs']), \
                         mpatches.Patch(color='magenta' , label='Nadir InGaAs (%d ms)' % ssfr0.data1['info']['int_time']['Nadir InGaAs']), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper left', fontsize=16)

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % (_metadata['Function']), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
    #\----------------------------------------------------------------------------/#



if __name__ == '__main__':

    test_lasp_ssfr_skywatch_zenith()
    test_lasp_ssfr_skywatch_nadir()
