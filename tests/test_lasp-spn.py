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
    data0_dif = ssfr.lasp_spn.read_spns(fname=fname_dif)

    fname_tot = sorted(glob.glob('%s/Total.txt' % fdir))[0]
    data0_tot = ssfr.lasp_spn.read_spns(fname=fname_tot)


    wvl0 = 555.0

    iwvl0_dif = np.argmin(np.abs(wvl0-data0_dif.data['wavelength']))
    iwvl0_tot = np.argmin(np.abs(wvl0-data0_tot.data['wavelength']))

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle('SPN-S Test on %s' % os.path.basename(fdir))

        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(211)
        ax1.scatter(data0_dif.data['tmhr'], data0_dif.data['flux'][:, iwvl0_dif], s=3, c='gray', lw=0.0)
        ax1.scatter(data0_tot.data['tmhr'], data0_tot.data['flux'][:, iwvl0_tot], s=3, c='black', lw=0.0)
        ax1.set_xlabel('UTC Time [Hour]')
        ax1.set_ylabel('Irradiance [$W m^{-2} nm^{-1}$]')
        ax1.set_title('Time Series of Downwelling Irradiance at %d nm' % wvl0)
        ax1.set_ylim((0.0, 0.3))

        patches_legend = [
                          mpatches.Patch(color='black' , label='Total'), \
                          mpatches.Patch(color='gray'  , label='Diffuse'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)

        percentiles = np.array([10, 50, 90])
        colors = ['red', 'blue', 'green']
        selected_tmhr = np.percentile(data0_tot.data['tmhr'], percentiles)
        for i in range(percentiles.size):
            ax1.axvline(selected_tmhr[i], color=colors[i], lw=1.5, ls='--')
        #\--------------------------------------------------------------/#

        #
        #/--------------------------------------------------------------\#
        ax2 = fig.add_subplot(212)
        for i in range(percentiles.size):
            itmhr_dif = np.argmin(np.abs(data0_dif.data['tmhr']-selected_tmhr[i]))
            itmhr_tot = np.argmin(np.abs(data0_tot.data['tmhr']-selected_tmhr[i]))
            ax2.plot(data0_tot.data['wavelength'], data0_tot.data['flux'][itmhr_tot, :], ls='-' , color=colors[i])
            ax2.plot(data0_dif.data['wavelength'], data0_dif.data['flux'][itmhr_dif, :], ls='--', color=colors[i])
        ax2.axvline(wvl0, color='k', lw=1.5, ls='--')
        ax2.set_xlabel('Wavelength [nm]')
        ax2.set_ylabel('Irradiance [$W m^{-2} nm^{-1}$]')
        ax2.set_title('Spectral Downwelling Irradiance')

        patches_legend = [
                          mpatches.Patch(edgecolor='black', ls='-' , facecolor='None', label='Total'), \
                          mpatches.Patch(edgecolor='black', ls='--', facecolor='None', label='Diffuse'), \
                         ]
        ax2.legend(handles=patches_legend, loc='upper right', fontsize=16)
        #\--------------------------------------------------------------/#

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
    #\----------------------------------------------------------------------------/#


if __name__ == '__main__':

    test_process_lasp_spns()
