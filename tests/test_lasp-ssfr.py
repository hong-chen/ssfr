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

    fdir = 'data/SSFR'

    for fdir0  in sorted(glob.glob('%s/*' % fdir)):
        fnames = ssfr.util.get_all_files(fdir0)

        ssfr0 = ssfr.lasp_ssfr.read_ssfr(fnames, dark_corr_mode='interp')

        dset = 'data_raw'
        extra_tag = '%s_dset-raw_' % os.path.basename(fdir0)
        data0 = getattr(ssfr0, dset)
        ssfr.vis.quicklook_ssfr_raw(data0, extra_tag=extra_tag)

        for i in range(ssfr0.Ndata):
            dset = 'data%d' % i
            extra_tag = '%s_dset-%d_' % (os.path.basename(fdir0), i)
            data0 = getattr(ssfr0, dset)
            ssfr.vis.quicklook_ssfr_raw(data0, extra_tag=extra_tag)


def figure_ssfr_wavelength():

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(15, 7))
        # fig.suptitle('Figure')

        # plot
        #/--------------------------------------------------------------\#
        wvls = ssfr.lasp_ssfr.get_ssfr_wavelength()
        ax1 = fig.add_subplot(121)
        ax1.plot(np.arange(256), wvls['zen_si'], lw=8.0, color='red', alpha=0.8)
        ax1.plot(np.arange(256), wvls['zen_in'], lw=8.0, color='blue', alpha=0.8)
        ax1.plot(np.arange(256), wvls['nad_si'], lw=3.0, color='green')
        ax1.plot(np.arange(256), wvls['nad_in'], lw=3.0, color='orange')
        ax1.set_xlabel('Channel Number')
        ax1.set_ylabel('Wavelength [nm]')
        ax1.set_title('CU LASP SSFR')

        patches_legend = [
                         mpatches.Patch(color='red'   , label='Zenith Silicon'), \
                         mpatches.Patch(color='blue'  , label='Zenith InGaAs'), \
                         mpatches.Patch(color='green' , label='Nadir Silicon'), \
                         mpatches.Patch(color='orange', label='Nadir InGaAs'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)

        ax1.grid()
        #\--------------------------------------------------------------/#

        # plot
        #/--------------------------------------------------------------\#
        wvls = ssfr.nasa_ssfr.get_ssfr_wavelength()
        ax1 = fig.add_subplot(122)
        ax1.plot(np.arange(256), wvls['zen_si'], lw=8.0, color='red', alpha=0.8)
        ax1.plot(np.arange(256), wvls['zen_in'], lw=8.0, color='blue', alpha=0.8)
        ax1.plot(np.arange(256), wvls['nad_si'], lw=3.0, color='green')
        ax1.plot(np.arange(256), wvls['nad_in'], lw=3.0, color='orange')
        ax1.set_xlabel('Channel Number')
        ax1.set_ylabel('Wavelength [nm]')
        ax1.set_title('NASA Ames SSFR')

        patches_legend = [
                         mpatches.Patch(color='red'   , label='Zenith Silicon'), \
                         mpatches.Patch(color='blue'  , label='Zenith InGaAs'), \
                         mpatches.Patch(color='green' , label='Nadir Silicon'), \
                         mpatches.Patch(color='orange', label='Nadir InGaAs'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)

        ax1.grid()
        #\--------------------------------------------------------------/#

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
    #\----------------------------------------------------------------------------/#


def figure_ssfr_slit():

    si_slit = ssfr.util.get_slit_func(500.0)
    in_slit = ssfr.util.get_slit_func(2000.0)

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(15, 7))
        # fig.suptitle('Figure')

        # plot
        #/--------------------------------------------------------------\#
        wvls = ssfr.lasp_ssfr.get_ssfr_wavelength()
        ax1 = fig.add_subplot(111)
        ax1.plot(si_slit[:, 0]+940.0, si_slit[:, 1], color='r', lw=2.0)
        ax1.plot(in_slit[:, 0]+960.0, in_slit[:, 1], color='b', lw=2.0)
        ax1.axvline(950.0, color='k', ls='--', lw=2.0)
        ax1.set_ylabel('Probability Density Function (Weight)')
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_title('SSFR Line Shape')

        patches_legend = [
                         mpatches.Patch(color='red'   , label='Silicon'), \
                         mpatches.Patch(color='blue'  , label='InGaAs'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)

        ax1.grid()
        #\--------------------------------------------------------------/#

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
    #\----------------------------------------------------------------------------/#



if __name__ == '__main__':

    # test_process_lasp_ssfr()

    # figure_ssfr_wavelength()
    figure_ssfr_slit()
