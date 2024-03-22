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



def fig_cos_resp(fname, wvl0=555.0):

    f = h5py.File(fname, 'r')
    mu = f['mu'][...]
    cos_resp = f['cos_resp'][...]
    wvl = f['wvl'][...]

    ang_ = f['raw/ang'][...]
    mu_  = f['raw/mu'][...]
    mu0  = f['raw/mu0'][...]

    try:
        wvl_ = f['raw/nad|si/wvl'][...]
        cos_resp_ = f['raw/nad|si/cos_resp'][...]
        cos_resp0 = f['raw/nad|si/cos_resp0'][...]
        cos_resp_std0 = f['raw/nad|si/cos_resp_std0'][...]
    except:
        wvl_ = f['raw/zen|si/wvl'][...]
        cos_resp_ = f['raw/zen|si/cos_resp'][...]
        cos_resp0 = f['raw/zen|si/cos_resp0'][...]
        cos_resp_std0 = f['raw/zen|si/cos_resp_std0'][...]
    f.close()

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        title = os.path.basename(fname).replace('.h5', '').upper()
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        fig.suptitle('Cosine Response (%d nm)' % (wvl0), fontsize=18)
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.scatter(mu, cos_resp[:, np.argmin(np.abs(wvl-wvl0))], s=6, c='k', lw=0.0, alpha=0.2)
        ax1.plot(mu_[:19]  , cos_resp_[:19, np.argmin(np.abs(wvl_-wvl0))]  , marker='o', markersize=8, color='r', lw=1.0, alpha=0.6)

        angle_offset = -2.5
        mu_new = np.cos(np.deg2rad(np.rad2deg(np.arccos(mu_[19:-1])) + angle_offset))
        ax1.plot(mu_[19:-1], cos_resp_[19:-1, np.argmin(np.abs(wvl_-wvl0))], marker='o', markersize=8, color='b', lw=1.0, alpha=0.2)
        ax1.plot(mu_new, cos_resp_[19:-1, np.argmin(np.abs(wvl_-wvl0))], marker='o', markersize=8, color='b', lw=1.0, alpha=0.6)
        ax1.errorbar(mu0, cos_resp0[:, np.argmin(np.abs(wvl_-wvl0))], yerr=cos_resp_std0[:, np.argmin(np.abs(wvl_-wvl0))], color='g', lw=1.0)
        ax1.axhline(1.0, color='gray', ls='--')
        ax1.plot([0.0, 1.0], [0.0, 1.0], color='gray', ls='--')
        ax1.set_xlim((0.0, 1.0))
        ax1.set_ylim((0.0, 1.1))
        ax1.set_xlabel('$cos(\\theta)$')
        ax1.set_ylabel('Response')
        ax1.set_title('%s' % (title), fontsize=12)

        patches_legend = [
                          mpatches.Patch(color='black' , label='Average&Interpolated'), \
                          mpatches.Patch(color='red'   , label='Pos. Angles (C.C.W.)'), \
                          mpatches.Patch(color='blue'  , label='Neg. Angles (C.W.)'), \
                          mpatches.Patch(color='green' , label='Average&Std.'), \
                         ]
        # ax1.legend(handles=patches_legend, loc='lower right', fontsize=16)
        ax1.legend(handles=patches_legend, loc='upper left', fontsize=16)
        #\--------------------------------------------------------------/#

        # save figure
        #/--------------------------------------------------------------\#
        fname_png = os.path.basename(fname).replace('.h5', '.png')
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig(fname_png, bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
    #\----------------------------------------------------------------------------/#


if __name__ == '__main__':

    fnames = sorted(glob.glob('data/*cos-resp*.h5'))
    for fname in [fnames[-1]]:
        fig_cos_resp(fname)
