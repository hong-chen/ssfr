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




def fig_cos_resp(fname, fdir_out=None, wvl0=555.0):

    f = h5py.File(fname, 'r')
    mu = f['mu'][...]
    wvl = f['wvl'][...]

    ang_ = f['raw/ang'][...]
    mu_  = f['raw/mu'][...]
    mu0  = f['raw/mu0'][...]

    try:
        cos_resp = f['ang_resp'][...]

    except Exception:
        cos_resp = f['cos_resp'][...]

    # determine joinder wavelength from file attributes (default to 950nm if not found)
    try:
        wvl_joint = f.attrs.get('joint_wavelength_nm', 950.0)
    except Exception:
        wvl_joint = 950.0

    # select appropriate channel based on wavelength
    if wvl0 <= wvl_joint:
        channel = 'si'  # Silicon channel for wavelengths <= joinder wavelength
    else:
        channel = 'in'  # InGaAs channel for wavelengths > joinder wavelength

    # nadir
    if '|nad|' in fname:
        direction_channel_tag = f'nad|{channel}' # one of ['nad|si', 'nad|in']
        try:
            wvl_ = f[f'raw/{direction_channel_tag}/wvl'][...]
            cos_resp_ = f[f'raw/{direction_channel_tag}/ang_resp'][...]
            cos_resp0 = f[f'raw/{direction_channel_tag}/ang_resp0'][...]
            cos_resp_std0 = f[f'raw/{direction_channel_tag}/ang_resp_std0'][...]

        except Exception:
            # Fallback to old naming convention
            try:
                wvl_ = f[f'raw/{direction_channel_tag}/wvl'][...]
                cos_resp_ = f[f'raw/{direction_channel_tag}/cos_resp'][...]
                cos_resp0 = f[f'raw/{direction_channel_tag}/cos_resp0'][...]
                cos_resp_std0 = f[f'raw/{direction_channel_tag}/cos_resp_std0'][...]

            except Exception as e2:
                print(f"Error reading nadir {channel} channel data: {e2}")
                f.close()
                return

    # zenith
    elif '|zen|' in fname:
        direction_channel_tag = f'zen|{channel}' # one of ['zen|si', 'zen|in']
        try:
            wvl_ = f[f'raw/{direction_channel_tag}/wvl'][...]
            cos_resp_ = f[f'raw/{direction_channel_tag}/ang_resp'][...]
            cos_resp0 = f[f'raw/{direction_channel_tag}/ang_resp0'][...]
            cos_resp_std0 = f[f'raw/{direction_channel_tag}/ang_resp_std0'][...]

        except Exception:
            # Fallback to old naming convention
            try:
                wvl_ = f[f'raw/{direction_channel_tag}/wvl'][...]
                cos_resp_ = f[f'raw/{direction_channel_tag}/cos_resp'][...]
                cos_resp0 = f[f'raw/{direction_channel_tag}/cos_resp0'][...]
                cos_resp_std0 = f[f'raw/{direction_channel_tag}/cos_resp_std0'][...]

            except Exception as e2:
                print(f"Error reading zenith {channel} channel data: {e2}")
                f.close()
                return

    # check if requested wavelength is within the available range and check if channel is in Si or InGaAs
    wvl_min, wvl_max = wvl_.min(), wvl_.max()
    if wvl0 < wvl_min or wvl0 > wvl_max:
        print(f"Warning: Requested wavelength {wvl0}nm is outside the {channel.upper()} channel range [{wvl_min:.1f}-{wvl_max:.1f}nm]")
        print(f"Joint wavelength: {wvl_joint}nm")
        if wvl0 > wvl_joint:
            print("Try using the InGaAs channel for wavelengths > 950nm")
        else:
            print("Try using the Silicon channel for wavelengths <= 950nm")

    f.close()

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        fontsize = 20
        title = os.path.basename(fname).replace('.h5', '').upper()
        plt.close('all')
        plt.rcParams.update({'font.size': fontsize})
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle('Cosine Response (%d nm, %s channel)' % (wvl0, channel.upper()), fontsize=fontsize+4)
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        # ax1.scatter(mu, cos_resp[:, np.argmin(np.abs(wvl-wvl0))], s=6, c='k', lw=0.0, alpha=0.2)

        # find the closest wavelength index but also print actual vs requested wavelength used
        wvl_idx = np.argmin(np.abs(wvl_-wvl0))
        actual_wvl = wvl_[wvl_idx]

        ax1.plot(mu_[ang_ >= 0.0], cos_resp_[ang_ >= 0.0, wvl_idx], marker='o', markersize=10, color='r', lw=2.0, alpha=0.6)
        ax1.plot(mu_[ang_ < 0.0], cos_resp_[ang_ < 0.0, wvl_idx], marker='o', markersize=10, color='b', lw=2.0, alpha=0.6)

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
        ax1.set_title('%s (Actual: %.1f nm)' % (title, actual_wvl), fontsize=fontsize)

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


def print_wavelength_info(fname):
    """
    Print wavelength information for both channels in a calibration file.

    Args:
    ----
        fname (str): Path to the HDF5 calibration file
    """
    try:
        with h5py.File(fname, 'r') as f:
            # Get joint wavelength
            try:
                wvl_joint = f.attrs.get('joint_wavelength_nm', 950.0)
            except Exception:
                wvl_joint = 950.0

            print(f"\nWavelength information for: {os.path.basename(fname)}")
            print(f"Joint wavelength: {wvl_joint} nm")
            print("-" * 60)

            # Check available channels
            for lc in ['zen', 'nad']:
                for channel in ['si', 'in']:
                    path = f'raw/{lc}|{channel}'
                    if path in f:
                        wvl = f[f'{path}/wvl'][...]
                        channel_name = "Silicon" if channel == 'si' else "InGaAs"
                        print(f"{lc.upper()} {channel_name:7s}: {wvl.min():6.1f} - {wvl.max():6.1f} nm ({len(wvl):3d} channels)")

            print("\nRecommended usage:")
            print(f"- Use wavelengths ≤ {wvl_joint} nm for Silicon channel")
            print(f"- Use wavelengths > {wvl_joint} nm for InGaAs channel")

    except Exception as e:
        print(f"Error reading wavelength info from {fname}: {e}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # parse arguments only when there is a command line input
        parser = argparse.ArgumentParser(description='SSFR Angular Calibration Analysis')
        parser.add_argument('--fdir', type=str, help='Directory containing the processed calibration data files (.h5).')
        parser.add_argument('--fname', type=str, help='Name of the processed calibration data file(s) (.h5).')
        parser.add_argument('--fdir_out', type=str, default='./', help='Directory where the processed data will be saved.')
        parser.add_argument('--wvl', type=float, default=555.0, help='Wavelength (in nm) for cosine response plot.')
        parser.add_argument('--info', action='store_true', help='Print wavelength information for calibration files.')
        args = parser.parse_args()

    else:
        class Args:
            # fdir = './'
            fdir = '../projects/2024-arcsix/'
            fname = '*ang-resp*si-120|in-350.h5'
            fdir_out = './'
            wvl = 555.0
            info = False
        args = Args()

    # fig_belana_darks_si()
    # fig_alvin_darks_si()

    # fnames = sorted(glob.glob('processed/2025-08-04/2025-08-04*.h5'))
    # fdir_out = 'plots/2025-08-04/'
    if args.fname:
        print(os.path.join(args.fdir, args.fname))
        fnames = sorted(glob.glob(os.path.join(args.fdir, args.fname)))
    else:
        fnames = sorted(glob.glob(os.path.join(args.fdir, '*.h5')))
    
    if args.info:
        # Print wavelength information for all files
        print("WAVELENGTH INFORMATION FOR CALIBRATION FILES")
        print("=" * 80)
        for fname in fnames:
            print_wavelength_info(fname)
    else:
        # Generate plots
        for fname in fnames:
            fig_cos_resp(fname, fdir_out=args.fdir_out, wvl0=args.wvl)

        print('Plots visualized for {} nm and saved in {}'.format(args.wvl, args.fdir_out))
