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



def fig_cos_resp():

    # fname = '2024-03-15|2024-03-19|vaa-180|dset0|cos-resp|lasp|ssfr-a|zen|si-080|in-250.h5'
    # fname = '2024-03-15|2024-03-19|vaa-180|dset1|cos-resp|lasp|ssfr-a|zen|si-120|in-350.h5'
    # fname = '2024-03-16|2024-03-19|vaa-180|dset0|cos-resp|lasp|ssfr-a|zen|si-080|in-250.h5'
    # fname = '2024-03-16|2024-03-19|vaa-180|dset1|cos-resp|lasp|ssfr-a|zen|si-120|in-350.h5'
    # fname = '2024-03-18|2024-03-19|vaa-180|dset0|cos-resp|lasp|ssfr-a|nad|si-080|in-250.h5'
    # fname = '2024-03-18|2024-03-19|vaa-180|dset1|cos-resp|lasp|ssfr-a|nad|si-120|in-350.h5'
    # fname = '2024-03-18|2024-03-19|vaa-300|dset0|cos-resp|lasp|ssfr-a|nad|si-080|in-250.h5'
    fname = '2024-03-18|2024-03-19|vaa-300|dset1|cos-resp|lasp|ssfr-a|nad|si-120|in-350.h5'

    f = h5py.File(fname, 'r')
    mu = f['mu'][...]
    cos_resp = f['cos_resp'][...]
    wvl = f['wvl'][...]

    ang_ = f['raw/ang'][...]
    mu_  = f['raw/mu'][...]
    wvl_ = f['raw/nad|si/wvl'][...]
    cos_resp_ = f['raw/nad|si/cos_resp'][...]
    mu0  = f['raw/mu0'][...]
    cos_resp0 = f['raw/nad|si/cos_resp0'][...]
    cos_resp_std0 = f['raw/nad|si/cos_resp_std0'][...]
    f.close()

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.scatter(mu, cos_resp[:, np.argmin(np.abs(wvl-555.0))], s=6, c='k', lw=0.0)
        ax1.scatter(mu_[:19], cos_resp_[:19, np.argmin(np.abs(wvl_-555.0))], s=60, c='r', lw=0.0, alpha=0.6)
        ax1.scatter(mu_[19:], cos_resp_[19:, np.argmin(np.abs(wvl_-555.0))], s=30, c='b', lw=0.0, alpha=0.6)
        ax1.errorbar(mu0, cos_resp0[:, np.argmin(np.abs(wvl_-555.0))], yerr=cos_resp_std0[:, np.argmin(np.abs(wvl_-555.0))], color='g', lw=1.0)
        ax1.axhline(1.0, color='red', ls='--')
        ax1.plot([0.0, 1.0], [0.0, 1.0], color='gray', ls='--')
        ax1.set_xlim((0.0, 1.0))
        ax1.set_ylim((0.0, 1.1))
        ax1.set_xlabel('$cos(\\theta)$')
        ax1.set_ylabel('Response')
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        # fig.subplots_adjust(hspace=0.3, wspace=0.3)
        # _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        # fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
    #\----------------------------------------------------------------------------/#


if __name__ == '__main__':

    fig_cos_resp()
