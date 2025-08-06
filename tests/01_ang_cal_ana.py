import os
import sys
import glob
import datetime
import h5py
import argparse
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



def fig_belana_darks_si():

    fname = 'ARCSIX_SSFR-B_2024-03-21_v0.h5'
    f = h5py.File(fname)
    zen_si_cnt0 = f['dset0/spectra'][...][:30, :, 0]
    nad_si_cnt0 = f['dset0/spectra'][...][:30, :, 2]
    zen_si_cnt1 = f['dset1/spectra'][...][:30, :, 0]
    nad_si_cnt1 = f['dset1/spectra'][...][:30, :, 2]
    f.close()

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(14, 10))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        colors = mpl.cm.jet(np.linspace(0.0, 1.0, 30))

        for i in range(0, 30, 3):
            ax1.plot(np.arange(256), zen_si_cnt0[i, :], alpha=1.0, lw=1.0, color=colors[i, ...])
            ax2.plot(np.arange(256), nad_si_cnt0[i, :], alpha=1.0, lw=1.0, color=colors[i, ...])
            ax3.plot(np.arange(256), zen_si_cnt1[i, :], alpha=1.0, lw=1.0, color=colors[i, ...])
            ax4.plot(np.arange(256), nad_si_cnt0[i, :], alpha=1.0, lw=1.0, color=colors[i, ...])
        ax3.set_xlabel('Channel #')
        ax4.set_xlabel('Channel #')
        ax1.set_ylabel('Counts')
        ax3.set_ylabel('Counts')
        ax1.set_title('Belana Zenith Silicon (dset0)')
        ax2.set_title('Belana Nadir Silicon (dset0)')
        ax3.set_title('Belana Zenith Silicon (dset1)')
        ax4.set_title('Belana Nadir Silicon (dset1)')
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
    #\----------------------------------------------------------------------------/#

def fig_alvin_darks_si():

    fname = 'ARCSIX_SSFR-A_2024-03-20_v0.h5'
    f = h5py.File(fname)
    zen_si_cnt0 = f['dset0/spectra'][...][:30, :, 0]
    nad_si_cnt0 = f['dset0/spectra'][...][:30, :, 2]
    zen_si_cnt1 = f['dset1/spectra'][...][:30, :, 0]
    nad_si_cnt1 = f['dset1/spectra'][...][:30, :, 2]
    f.close()

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(14, 10))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        colors = mpl.cm.jet(np.linspace(0.0, 1.0, 30))

        for i in range(0, 30, 3):
            ax1.plot(np.arange(256), zen_si_cnt0[i, :], alpha=1.0, lw=1.0, color=colors[i, ...])
            ax2.plot(np.arange(256), nad_si_cnt0[i, :], alpha=1.0, lw=1.0, color=colors[i, ...])
            ax3.plot(np.arange(256), zen_si_cnt1[i, :], alpha=1.0, lw=1.0, color=colors[i, ...])
            ax4.plot(np.arange(256), nad_si_cnt0[i, :], alpha=1.0, lw=1.0, color=colors[i, ...])
        ax3.set_xlabel('Channel #')
        ax4.set_xlabel('Channel #')
        ax1.set_ylabel('Counts')
        ax3.set_ylabel('Counts')
        ax1.set_title('Alvin Zenith Silicon (dset0)')
        ax2.set_title('Alvin Nadir Silicon (dset0)')
        ax3.set_title('Alvin Zenith Silicon (dset1)')
        ax4.set_title('Alvin Nadir Silicon (dset1)')
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
    #\----------------------------------------------------------------------------/#




def fig_cos_resp(fname, fdir_out, wvl0=555.0):

    f = h5py.File(fname, 'r')
    mu = f['mu'][...]
    wvl = f['wvl'][...]

    ang_ = f['raw/ang'][...]
    mu_  = f['raw/mu'][...]
    mu0  = f['raw/mu0'][...]

    try:
        cos_resp = f['ang_resp'][...]
    except:
        cos_resp = f['cos_resp'][...]

    if '|nad|' in fname:
        try:
            wvl_ = f['raw/nad|si/wvl'][...]
            cos_resp_ = f['raw/nad|si/ang_resp'][...]
            cos_resp0 = f['raw/nad|si/ang_resp0'][...]
            cos_resp_std0 = f['raw/nad|si/ang_resp_std0'][...]
        except:
            wvl_ = f['raw/nad|si/wvl'][...]
            cos_resp_ = f['raw/nad|si/cos_resp'][...]
            cos_resp0 = f['raw/nad|si/cos_resp0'][...]
            cos_resp_std0 = f['raw/nad|si/cos_resp_std0'][...]

    elif '|zen|' in fname:
        try:
            wvl_ = f['raw/zen|si/wvl'][...]
            cos_resp_ = f['raw/zen|si/ang_resp'][...]
            cos_resp0 = f['raw/zen|si/ang_resp0'][...]
            cos_resp_std0 = f['raw/zen|si/ang_resp_std0'][...]
        except:
            wvl_ = f['raw/zen|si/wvl'][...]
            cos_resp_ = f['raw/zen|si/cos_resp'][...]
            cos_resp0 = f['raw/zen|si/cos_resp0'][...]
            cos_resp_std0 = f['raw/zen|si/cos_resp_std0'][...]
    f.close()

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        fontsize = 20
        title = os.path.basename(fname).replace('.h5', '').upper()
        plt.close('all')
        plt.rcParams.update({'font.size': fontsize})
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle('Cosine Response (%d nm)' % (wvl0), fontsize=fontsize+4)
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        # ax1.scatter(mu, cos_resp[:, np.argmin(np.abs(wvl-wvl0))], s=6, c='k', lw=0.0, alpha=0.2)
        ax1.plot(mu_[ang_>=0.0]  , cos_resp_[ang_>=0.0, np.argmin(np.abs(wvl_-wvl0))]  , marker='o', markersize=10, color='r', lw=2.0, alpha=0.6)
        ax1.plot(mu_[ang_<0.0]  , cos_resp_[ang_<0.0, np.argmin(np.abs(wvl_-wvl0))]  , marker='o', markersize=10, color='b', lw=2.0, alpha=0.6)

        # angle_offset = -2.5
        angle_offset = 0.0
        # mu_new = np.cos(np.deg2rad(np.rad2deg(np.arccos(mu_[19:-1])) + angle_offset))
        # ax1.plot(mu_[19:-1], cos_resp_[19:-1, np.argmin(np.abs(wvl_-wvl0))], marker='o', markersize=8, color='b', lw=1.0, alpha=0.2)
        # ax1.plot(mu_new, cos_resp_[19:-1, np.argmin(np.abs(wvl_-wvl0))], marker='o', markersize=8, color='b', lw=1.0, alpha=0.6)
        # ax1.errorbar(mu0, cos_resp0[:, np.argmin(np.abs(wvl_-wvl0))], yerr=cos_resp_std0[:, np.argmin(np.abs(wvl_-wvl0))], color='g', lw=1.0)
        ax1.axhline(1.0, color='gray', ls='--')
        ax1.plot([0.0, 1.0], [0.0, 1.0], color='gray', ls='--')
        ax1.set_xlim((0.0, 1.0))
        ax1.set_ylim((0.0, 1.1))
        ax1.set_xlabel('$cos(\\theta)$')
        ax1.set_ylabel('Response')
        ax1.set_title('%s' % (title), fontsize=fontsize)

        patches_legend = [
                          # mpatches.Patch(color='black' , label='Average&Interpolated'), \
                          mpatches.Patch(color='red'   , label='Pos. Angles (C.C.W.)'), \
                          mpatches.Patch(color='blue'  , label='Neg. Angles (C.W.)'), \
                          # mpatches.Patch(color='green' , label='Average&Std.'), \
                         ]
        ax1.legend(handles=patches_legend, loc='lower right', fontsize=16)
        # ax1.legend(handles=patches_legend, loc='upper left', fontsize=16)
        #\--------------------------------------------------------------/#

        # save figure
        #/--------------------------------------------------------------\#
        fname = os.path.splitext(fname)[0] + '_wvl-{}_cos_resp.h5'.format(int(wvl0))
        fname_png = os.path.basename(fname).replace('.h5', '.png')
        if fdir_out is not None:
            if not os.path.exists(fdir_out):
                os.makedirs(fdir_out)
            fname_png = os.path.join(fdir_out, fname_png)

        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig(fname_png, bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        # plt.show()
        # sys.exit()
    #\----------------------------------------------------------------------------/#

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='SSFR Angular Calibration Analysis')
    parser.add_argument('--fdir', type=str, help='Directory containing the processed calibration data files (.h5).')
    parser.add_argument('--fdir_out', type=str, default='./', help='Directory where the processed data will be saved.')
    parser.add_argument('--wvl', type=float, default=555.0, help='Wavelength (in nm) for cosine response plot.')
    args = parser.parse_args()

    # fig_belana_darks_si()
    # fig_alvin_darks_si()

    # fnames = sorted(glob.glob('processed/2025-08-04/2025-08-04*.h5'))
    # fdir_out = 'plots/2025-08-04/'
    fnames = sorted(glob.glob(os.path.join(args.fdir, '*.h5')))
    for fname in fnames:
        fig_cos_resp(fname, fdir_out=args.fdir_out, wvl0=args.wvl)

    print('Plots visualized for {} nm and saved in {}'.format(args.wvl, args.fdir_out))
