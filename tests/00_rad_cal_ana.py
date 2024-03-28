import os
import sys
import glob
import datetime
import warnings
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
        ax1.legend(handles=patches_legend, loc='lower right', fontsize=16)
        # ax1.legend(handles=patches_legend, loc='upper left', fontsize=16)
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







def rad_cal(
        fdir_pri,
        fdir_tra,
        fdir_sec=None
        ):

    # get calibration files of primary
    #/----------------------------------------------------------------------------\#
    date_cal_s_pri, ssfr_tag_pri, lc_tag_pri, cal_tag_pri, lamp_tag_pri, si_int_tag_pri, in_int_tag_pri = os.path.basename(fdir_pri).split('_')
    fnames_pri_ = sorted(glob.glob('%s/*.SKS' % (fdir_pri)))
    fnames_pri = [fnames_pri_[-1]]
    if len(fnames_pri) > 1:
        msg = '\nWarning [rad_cal]: find more than one file for "%s", selected "%s" ...' % (fdir_pri, fnames_pri[0])
        warnings.warn(msg)
    #\----------------------------------------------------------------------------/#


    # get calibration files of transfer
    #/----------------------------------------------------------------------------\#
    date_cal_s_tra, ssfr_tag_tra, lc_tag_tra, cal_tag_tra, lamp_tag_tra, si_int_tag_tra, in_int_tag_tra = os.path.basename(fdir_tra).split('_')
    fnames_tra_ = sorted(glob.glob('%s/*.SKS' % (fdir_tra)))
    fnames_tra = [fnames_tra_[-1]]
    if len(fnames_tra) > 1:
        msg = '\nWarning [rad_cal]: find more than one file for "%s", selected "%s" ...' % (fdir_tra, fnames_tra[0])
        warnings.warn(msg)
    #\----------------------------------------------------------------------------/#


    # placeholder for calibration files of transfer
    #/----------------------------------------------------------------------------\#
    if fdir_sec is None:
        fdir_sec = fdir_tra
    date_cal_s_sec, ssfr_tag_sec, lc_tag_sec, cal_tag_sec, lamp_tag_sec, si_int_tag_sec, in_int_tag_sec = os.path.basename(fdir_sec).split('_')
    fnames_sec_ = sorted(glob.glob('%s/*.SKS' % (fdir_sec)))
    fnames_sec = [fnames_sec_[-1]]
    if len(fnames_sec) > 1:
        msg = '\nWarning [rad_cal]: find more than one file for "%s", selected "%s" ...' % (fdir_sec, fnames_sec[0])
        warnings.warn(msg)
    #\----------------------------------------------------------------------------/#

    if (ssfr_tag_pri==ssfr_tag_tra):
        ssfr_tag = ssfr_tag_pri
    if (lc_tag_pri==lc_tag_tra):
        lc_tag = lc_tag_pri

    date_today_s = datetime.datetime.now().strftime('%Y-%m-%d')

    ssfr_ = ssfr.lasp_ssfr.read_ssfr(fnames_pri)

    for i in range(ssfr_.Ndset):
        dset_tag = 'dset%d' % i
        dset_ = getattr(ssfr_, dset_tag)
        int_time = dset_['info']['int_time']

        cal_tag = '%s_%s|%s_%s|%s_%s' % (date_cal_s_pri, lamp_tag_pri, date_cal_s_tra, lamp_tag_tra, date_cal_s_sec, lamp_tag_sec)
        filename_tag = '%s|%s|%s' % (cal_tag, date_today_s, dset_tag)

        ssfr.cal.cdata_rad_resp(fnames_pri=fnames_pri, fnames_tra=fnames_tra, fnames_sec=fnames_sec, which_ssfr='lasp|%s' % ssfr_tag, which_lc=lc_tag, int_time=int_time, which_lamp=lamp_tag_pri, filename_tag=filename_tag)


def main_calibration():

    """
    Notes:
        irradiance setup:
            SSFR-A (Alvin)
              - nadir : LC6 + stainless steel cased fiber
              - zenith: LC4 + black plastic cased fiber
    """

    # wavelength calibration
    #/----------------------------------------------------------------------------\#
    # for ssfr_tag in ['SSFR-A', 'SSFR-B']:
    #     for lc_tag in ['zen', 'nad']:
    #         for lamp_tag in ['kr', 'hg']:
    #             wvl_cal(ssfr_tag, lc_tag, lamp_tag)
    #\----------------------------------------------------------------------------/#

    # radiometric calibration
    #/----------------------------------------------------------------------------\#
    fdir_pri = '/argus/field/arcsix/cal/rad-cal/2024-03-27_SSFR-A_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350'
    fdir_tra = '/argus/field/arcsix/cal/rad-cal/2024-03-26_SSFR-A_zen-lc4_transfer_lamp-150e_si-080-120_in-250-350'

    # fdir_pri = '/argus/field/arcsix/cal/rad-cal/2024-03-20_SSFR-A_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350'
    # fdir_tra = '/argus/field/arcsix/cal/rad-cal/2024-03-20_SSFR-A_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350'

    # fdir_pri = '/argus/field/arcsix/cal/rad-cal/2024-03-20_SSFR-A_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350'
    # fdir_tra = '/argus/field/arcsix/cal/rad-cal/2024-03-25_SSFR-A_nad-lc6_transfer_lamp-150e_si-080-120_in-250-350'

    # fdir_pri = '/argus/field/arcsix/cal/rad-cal/2024-03-25_SSFR-A_nad-lc6_pri-cal_lamp-506_si-080-120_in-250-350'
    # fdir_tra = '/argus/field/arcsix/cal/rad-cal/2024-03-25_SSFR-A_nad-lc6_transfer_lamp-150e_si-080-120_in-250-350'

    # fdir_pri = '/argus/field/arcsix/cal/rad-cal/2024-03-20_SSFR-A_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350'
    # fdir_tra = '/argus/field/arcsix/cal/rad-cal/2024-03-26_SSFR-A_zen-lc4_transfer_lamp-150e_si-080-120_in-250-350'

    rad_cal(fdir_pri, fdir_tra)
    #\----------------------------------------------------------------------------/#

    # angular calibration
    #/----------------------------------------------------------------------------\#
    # fdirs = [
    #         'data/arcsix/cal/ang-cal/2024-03-15_SSFR-A_zen_vaa-180_507',
    #         'data/arcsix/cal/ang-cal/2024-03-16_SSFR-A_zen_vaa-180_507',
    #         'data/arcsix/cal/ang-cal/2024-03-18_SSFR-A_nad_vaa-180_507',
    #         'data/arcsix/cal/ang-cal/2024-03-18_SSFR-A_nad_vaa-300_507',
    #         'data/arcsix/cal/ang-cal/2024-03-19_SSFR-A_nad_vaa-060_507',
    #         'data/arcsix/cal/ang-cal/2024-03-19_SSFR-A_zen_vaa-060_507',
    #         'data/arcsix/cal/ang-cal/2024-03-19_SSFR-A_zen_vaa-300_507',
    #         ]
    # for fdir in fdirs:
    #     ang_cal(fdir)
    #\----------------------------------------------------------------------------/#



def field_lamp_consis_check():

    fnames = sorted(glob.glob('*150e*.h5'))

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)

        for fname in fnames:
            f = h5py.File(fname, 'r')
            wvl = f['wvl'][...]
            transfer = f['pri_resp'][...]
            f.close()
        # cs = ax1.imshow(.T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
            ax1.scatter(wvl, transfer, s=6, lw=0.0)
        # ax1.hist(.ravel(), bins=100, histtype='stepfilled', alpha=0.5, color='black')
        # ax1.plot([0, 1], [0, 1], color='k', ls='--')
        # ax1.set_xlim(())
        # ax1.set_ylim(())
        # ax1.set_xlabel('')
        # ax1.set_ylabel('')
        # ax1.set_title('')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
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

    # fig_belana_darks_si()
    # fig_alvin_darks_si()
    # fnames = sorted(glob.glob('data/*cos-resp*.h5'))
    # for fname in [fnames[-1]]:
    #     fig_cos_resp(fname)

    main_calibration()

    # field_lamp_consis_check()

    pass
