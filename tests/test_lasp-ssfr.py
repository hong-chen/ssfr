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


_mission_   = 'arcsix'
_spns_      = 'spns-b'
_ssfr_      = 'ssfr-b'
_fdir_data_ = 'data/%s/pre-mission' % _mission_
_fdir_hsk_  = '%s/raw/hsk'
_fdir_ssfr_ = '%s/raw/%s' % (_fdir_data_, _ssfr_)
_fdir_spns_ = '%s/raw/%s' % (_fdir_data_, _spns_)
_fdir_v0_   = 'data/processed'
_fdir_v1_   = 'data/processed'
_fdir_v2_   = 'data/processed'



def test_joint_wvl_cal(ssfr_tag, lc_tag, lamp_tag, Nchan=256):


    # si and in tags
    #/----------------------------------------------------------------------------\#
    si_tag = '%s|si' % lc_tag
    in_tag = '%s|in' % lc_tag
    #\----------------------------------------------------------------------------/#

    # si and in index
    #/----------------------------------------------------------------------------\#
    indices_spec = {
            'zen|si': 0,
            'zen|in': 1,
            'nad|si': 2,
            'nad|in': 3,
            }
    index_si = indices_spec[si_tag]
    index_in = indices_spec[in_tag]
    #\----------------------------------------------------------------------------/#

    # get wavelength
    #/----------------------------------------------------------------------------\#
    wvls = ssfr.lasp_ssfr.get_ssfr_wvl('lasp|%s' % ssfr_tag.lower())
    wvl_si = wvls[si_tag]
    wvl_in = wvls[in_tag]
    #\----------------------------------------------------------------------------/#

    # get spectra counts data
    #/----------------------------------------------------------------------------\#
    fdir_data = '/argus/field/arcsix/cal/rad-cal'
    fdir   =  sorted(glob.glob('%s/*%s*%s*%s*' % (fdir_data, ssfr_tag, lc_tag, lamp_tag)))[0]
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))
    ssfr_data = ssfr.lasp_ssfr.read_ssfr(fnames)

    cnt_si_dset0 = ssfr_data.dset0['spectra_dark-corr'][:, :, index_si]/ssfr_data.dset0['info']['int_time'][si_tag]
    cnt_in_dset0 = ssfr_data.dset0['spectra_dark-corr'][:, :, index_in]/ssfr_data.dset0['info']['int_time'][in_tag]

    cnt_si_dset1 = ssfr_data.dset1['spectra_dark-corr'][:, :, index_si]/ssfr_data.dset1['info']['int_time'][si_tag]
    cnt_in_dset1 = ssfr_data.dset1['spectra_dark-corr'][:, :, index_in]/ssfr_data.dset1['info']['int_time'][in_tag]
    #\----------------------------------------------------------------------------/#

    # get response
    #/----------------------------------------------------------------------------\#
    dset_s = 'dset0'
    fnames_cal_dset0 = sorted(glob.glob('%s/cal/*cal-rad-pri|lasp|%s|%s|%s*.h5' % (ssfr.common.fdir_data, ssfr_tag.lower(), lc_tag.lower(), dset_s.lower())))
    f = h5py.File(fnames_cal_dset0[-1], 'r')
    resp_si_dset0 = f[si_tag][...]
    resp_in_dset0 = f[in_tag][...]
    f.close()

    dset_s = 'dset1'
    fnames_cal_dset1 = sorted(glob.glob('%s/cal/*cal-rad-pri|lasp|%s|%s|%s*.h5' % (ssfr.common.fdir_data, ssfr_tag.lower(), lc_tag.lower(), dset_s.lower())))
    f = h5py.File(fnames_cal_dset1[-1], 'r')
    resp_si_dset1 = f[si_tag][...]
    resp_in_dset1 = f[in_tag][...]
    f.close()
    #\----------------------------------------------------------------------------/#


    # get flux
    #/----------------------------------------------------------------------------\#
    flux_si_dset0 = cnt_si_dset0 / resp_si_dset0
    flux_in_dset0 = cnt_in_dset0 / resp_in_dset0
    flux_si_dset1 = cnt_si_dset1 / resp_si_dset1
    flux_in_dset1 = cnt_in_dset1 / resp_in_dset1
    #\----------------------------------------------------------------------------/#


    wvl_joint = 950.0
    x = np.arange(flux_si_dset0.shape[0])

    index_joint_si = np.where(wvl_si< wvl_joint)[0][-1]
    index_joint_in = np.where(wvl_in>=wvl_joint)[0][-1]

    # figure
    #/----------------------------------------------------------------------------\#
    for index in np.arange(30, 91):
        plt.close('all')
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle('LASP|%s|%s %5.5d' % (ssfr_tag.upper(), lc_tag.upper(), index))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(312)
        ax1.scatter(wvl_si, flux_si_dset0[index, :], s=6, c='r'      , lw=0.0, alpha=0.5)
        ax1.scatter(wvl_si, flux_si_dset1[index, :], s=6, c='magenta', lw=0.0, alpha=0.5)
        ax1.scatter(wvl_in, flux_in_dset0[index, :], s=6, c='b'      , lw=0.0, alpha=0.5)
        ax1.scatter(wvl_in, flux_in_dset1[index, :], s=6, c='cyan'   , lw=0.0, alpha=0.5)

        ax1.scatter(wvl_si[index_joint_si], flux_si_dset0[index, index_joint_si], s=250, c='r', lw=0.0, marker='*')
        ax1.scatter(wvl_si[index_joint_si], flux_si_dset1[index, index_joint_si], s=250, c='magenta', lw=0.0, marker='*')
        ax1.scatter(wvl_in[index_joint_in], flux_in_dset0[index, index_joint_in], s=250, c='b', lw=0.0, marker='*')
        ax1.scatter(wvl_in[index_joint_in], flux_in_dset1[index, index_joint_in], s=250, c='cyan', lw=0.0, marker='*')

        ax1.axvline(wvl_joint, color='gray', lw=1.5, ls='--')
        ax1.axvspan(750, 1150, color='gray', alpha=0.1, lw=0.0)
        ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylim((0.0, 0.3))
        ax1.set_title('Spectra')

        ax2 = fig.add_subplot(313)
        ax2.scatter(wvl_si, flux_si_dset0[index, :], s=6, c='r'      , lw=0.0, alpha=0.5)
        ax2.scatter(wvl_si, flux_si_dset1[index, :], s=6, c='magenta', lw=0.0, alpha=0.5)
        ax2.scatter(wvl_in, flux_in_dset0[index, :], s=6, c='b'      , lw=0.0, alpha=0.5)
        ax2.scatter(wvl_in, flux_in_dset1[index, :], s=6, c='cyan'   , lw=0.0, alpha=0.5)

        ax2.scatter(wvl_si[index_joint_si], flux_si_dset0[index, index_joint_si], s=250, c='r', lw=0.0, marker='*')
        ax2.scatter(wvl_si[index_joint_si], flux_si_dset1[index, index_joint_si], s=250, c='magenta', lw=0.0, marker='*')
        ax2.scatter(wvl_in[index_joint_in], flux_in_dset0[index, index_joint_in], s=250, c='b', lw=0.0, marker='*')
        ax2.scatter(wvl_in[index_joint_in], flux_in_dset1[index, index_joint_in], s=250, c='cyan', lw=0.0, marker='*')

        ax2.axvline(wvl_joint, color='gray', lw=1.5, ls='--')
        ax2.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
        ax2.set_xlabel('Wavelength [nm]')
        ax2.set_xlim((750, 1150))
        ax2.set_ylim((0.18, 0.26))
        ax2.set_title('Spectra [Zoomed In]')


        ax3 = fig.add_subplot(311)
        ax3.plot(x, flux_si_dset0[:, index_joint_si], lw=2, c='r'      , alpha=0.5, marker='o', markersize=1)
        ax3.plot(x, flux_si_dset1[:, index_joint_si], lw=2, c='magenta', alpha=0.5, marker='o', markersize=1)
        ax3.plot(x, flux_in_dset0[:, index_joint_in], lw=2, c='b'      , alpha=0.5, marker='o', markersize=1)
        ax3.plot(x, flux_in_dset1[:, index_joint_in], lw=2, c='cyan'   , alpha=0.5, marker='o', markersize=1)

        ax3.scatter(index, flux_si_dset0[index, index_joint_si], s=250, c='r', lw=0.0, marker='*')
        ax3.scatter(index, flux_si_dset1[index, index_joint_si], s=250, c='magenta', lw=0.0, marker='*')
        ax3.scatter(index, flux_in_dset0[index, index_joint_in], s=250, c='b', lw=0.0, marker='*')
        ax3.scatter(index, flux_in_dset1[index, index_joint_in], s=250, c='cyan', lw=0.0, marker='*')

        ax3.axvline(index, color='gray', lw=1.5, ls='--')
        ax3.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
        ax3.set_xlabel('Index')
        ax3.set_ylim((0.18, 0.26))
        ax3.set_title('Time Series')
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.35, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%5.5d_%s_[lasp|%s|%s].png' % (index, _metadata['Function'], ssfr_tag.lower(), lc_tag.lower()), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#

def main_test_joint_wvl_cal():

    # radiometric calibration
    #/----------------------------------------------------------------------------\#
    for ssfr_tag in ['SSFR-A', 'SSFR-B']:
        for lc_tag in ['zen', 'nad']:
            for lamp_tag in ['1324']:
                test_joint_wvl_cal(ssfr_tag, lc_tag, lamp_tag)
    #\----------------------------------------------------------------------------/#



def test_joint_wvl_skywatch(ssfr_tag, lc_tag, date_tag, Nchan=256):

    # si and in tags
    #/----------------------------------------------------------------------------\#
    si_tag = '%s|si' % lc_tag
    in_tag = '%s|in' % lc_tag
    #\----------------------------------------------------------------------------/#

    # si and in index
    #/----------------------------------------------------------------------------\#
    indices_spec = {
            'zen|si': 0,
            'zen|in': 1,
            'nad|si': 2,
            'nad|in': 3,
            }
    index_si = indices_spec[si_tag]
    index_in = indices_spec[in_tag]
    #\----------------------------------------------------------------------------/#

    # get wavelength
    #/----------------------------------------------------------------------------\#
    wvls = ssfr.lasp_ssfr.get_ssfr_wvl('lasp|%s' % ssfr_tag.lower())
    wvl_si = wvls[si_tag]
    wvl_in = wvls[in_tag]
    #\----------------------------------------------------------------------------/#

    # get spectra counts data
    #/----------------------------------------------------------------------------\#
    fdir_data = '../examples/data/arcsix/pre-mission/raw/%s' % (ssfr_tag.lower())
    fnames = sorted(glob.glob('%s/%s/*.SKS' % (fdir_data, date_tag)))
    ssfr_data = ssfr.lasp_ssfr.read_ssfr(fnames, dark_corr_mode='interp', which_ssfr='lasp|%s' % _ssfr_.lower(), dark_extend=4, light_extend=4)

    cnt_si_dset0 = ssfr_data.dset0['spectra_dark-corr'][:, :, index_si]/ssfr_data.dset0['info']['int_time'][si_tag]
    cnt_in_dset0 = ssfr_data.dset0['spectra_dark-corr'][:, :, index_in]/ssfr_data.dset0['info']['int_time'][in_tag]

    cnt_si_dset1 = ssfr_data.dset1['spectra_dark-corr'][:, :, index_si]/ssfr_data.dset1['info']['int_time'][si_tag]
    cnt_in_dset1 = ssfr_data.dset1['spectra_dark-corr'][:, :, index_in]/ssfr_data.dset1['info']['int_time'][in_tag]
    #\----------------------------------------------------------------------------/#

    # get temperature
    #/----------------------------------------------------------------------------\#
    # 1 housing temp
    # 2 TEC 1
    # 3 INGAS 1
    # 4 INGAS 2
    # 5 Wavelength controllor 1
    # 6 Wavelength controller 2
    # 7 Silicon temp
    # 8 Relative humidity
    # 9 TEC 2
    # 10 CRio Temperature
    #
    # 1 ambient temp
    # 2 INGAS 1 zenith temp
    # 3 INGAS 2 nadir temp
    # 4 plate temp
    # 5 relative humidity
    # 6 TEC1
    # 7 TEC2
    # 8 Wavelength controllor 1
    # 9 nothing
    # 10 CRio Temperature
    temp0 = ssfr_data.dset0['temp'][:, 5] # 5 or 6
    temp1 = ssfr_data.dset1['temp'][:, 5] # 5 or 6
    #\----------------------------------------------------------------------------/#

    # get response
    #/----------------------------------------------------------------------------\#
    dset_s = 'dset0'
    fnames_cal_dset0 = sorted(glob.glob('%s/cal/*cal-rad-pri|lasp|%s|%s|%s*.h5' % (ssfr.common.fdir_data, ssfr_tag.lower(), lc_tag.lower(), dset_s.lower())))
    f = h5py.File(fnames_cal_dset0[-1], 'r')
    resp_si_dset0 = f[si_tag][...]
    resp_in_dset0 = f[in_tag][...]
    f.close()

    dset_s = 'dset1'
    fnames_cal_dset1 = sorted(glob.glob('%s/cal/*cal-rad-pri|lasp|%s|%s|%s*.h5' % (ssfr.common.fdir_data, ssfr_tag.lower(), lc_tag.lower(), dset_s.lower())))
    f = h5py.File(fnames_cal_dset1[-1], 'r')
    resp_si_dset1 = f[si_tag][...]
    resp_in_dset1 = f[in_tag][...]
    f.close()
    #\----------------------------------------------------------------------------/#


    # get flux
    #/----------------------------------------------------------------------------\#
    flux_si_dset0 = cnt_si_dset0 / resp_si_dset0
    flux_in_dset0 = cnt_in_dset0 / resp_in_dset0
    flux_si_dset1 = cnt_si_dset1 / resp_si_dset1
    flux_in_dset1 = cnt_in_dset1 / resp_in_dset1
    #\----------------------------------------------------------------------------/#


    wvl_joint = 950.0
    # wvl_joint = 1000.0
    x0 = np.arange(flux_si_dset0.shape[0])
    x1 = np.arange(flux_si_dset1.shape[0])

    index_joint_si = np.where(wvl_si< wvl_joint)[0][-1]
    index_joint_in = np.where(wvl_in>=wvl_joint)[0][-1]

    flux_si_dset0_ = flux_si_dset0[:, index_joint_si]
    flux_in_dset0_ = flux_in_dset0[:, index_joint_in]
    flux_si_dset1_ = flux_si_dset1[:, index_joint_si]
    flux_in_dset1_ = flux_in_dset1[:, index_joint_in]

    diff_dset0 = flux_in_dset0_ - flux_si_dset0_
    diff_dset1 = flux_in_dset1_ - flux_si_dset1_

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.scatter(temp0, diff_dset0, s=6, c='r', lw=0.0)
        ax1.scatter(temp1, diff_dset1, s=6, c='b', lw=0.0)
        # ax1.hist(.ravel(), bins=100, histtype='stepfilled', alpha=0.5, color='black')
        # ax1.plot([0, 1], [0, 1], color='k', ls='--')
        # ax1.set_xlim(())
        # ax1.set_ylim(())
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel('Irradiance Difference')
        ax1.set_title('SSFR-B Skywatch 2023-10-19')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #\--------------------------------------------------------------/#

        patches_legend = [
                         mpatches.Patch(color='red'   , label='Int Time 250 ms'), \
                         mpatches.Patch(color='blue'  , label='Int Time 350 ms'), \
        #                  mpatches.Patch(color='green' , label='D'), \
                         ]
        ax1.legend(handles=patches_legend, loc='lower left', fontsize=16)

        # save figure
        #/--------------------------------------------------------------\#
        # fig.subplots_adjust(hspace=0.3, wspace=0.3)
        # _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        # fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
    #\----------------------------------------------------------------------------/#

    # figure
    #/----------------------------------------------------------------------------\#
    for index in np.arange(0, x0.size, 60):
        plt.close('all')
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle('LASP|%s|%s %s %5.5d' % (ssfr_tag.upper(), lc_tag.upper(), date_tag, index))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(312)
        ax1.scatter(wvl_si, flux_si_dset0[index, :], s=6, c='r'      , lw=0.0, alpha=0.5)
        ax1.scatter(wvl_si, flux_si_dset1[index, :], s=6, c='magenta', lw=0.0, alpha=0.5)
        ax1.scatter(wvl_in, flux_in_dset0[index, :], s=6, c='b'      , lw=0.0, alpha=0.5)
        ax1.scatter(wvl_in, flux_in_dset1[index, :], s=6, c='cyan'   , lw=0.0, alpha=0.5)

        ax1.scatter(wvl_si[index_joint_si], flux_si_dset0[index, index_joint_si], s=250, c='r', lw=0.0, marker='*')
        ax1.scatter(wvl_si[index_joint_si], flux_si_dset1[index, index_joint_si], s=250, c='magenta', lw=0.0, marker='*')
        ax1.scatter(wvl_in[index_joint_in], flux_in_dset0[index, index_joint_in], s=250, c='b', lw=0.0, marker='*')
        ax1.scatter(wvl_in[index_joint_in], flux_in_dset1[index, index_joint_in], s=250, c='cyan', lw=0.0, marker='*')

        ax1.axvline(wvl_joint, color='gray', lw=1.5, ls='--')
        ax1.axvspan(750, 1150, color='gray', alpha=0.1, lw=0.0)
        ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylim((0.0, 1.0))
        ax1.set_title('Spectra')

        ax2 = fig.add_subplot(313)
        ax2.scatter(wvl_si, flux_si_dset0[index, :], s=6, c='r'      , lw=0.0, alpha=0.5)
        ax2.scatter(wvl_si, flux_si_dset1[index, :], s=6, c='magenta', lw=0.0, alpha=0.5)
        ax2.scatter(wvl_in, flux_in_dset0[index, :], s=6, c='b'      , lw=0.0, alpha=0.5)
        ax2.scatter(wvl_in, flux_in_dset1[index, :], s=6, c='cyan'   , lw=0.0, alpha=0.5)

        ax2.scatter(wvl_si[index_joint_si], flux_si_dset0[index, index_joint_si], s=250, c='r', lw=0.0, marker='*')
        ax2.scatter(wvl_si[index_joint_si], flux_si_dset1[index, index_joint_si], s=250, c='magenta', lw=0.0, marker='*')
        ax2.scatter(wvl_in[index_joint_in], flux_in_dset0[index, index_joint_in], s=250, c='b', lw=0.0, marker='*')
        ax2.scatter(wvl_in[index_joint_in], flux_in_dset1[index, index_joint_in], s=250, c='cyan', lw=0.0, marker='*')

        ax2.axvline(wvl_joint, color='gray', lw=1.5, ls='--')
        ax2.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
        ax2.set_xlabel('Wavelength [nm]')
        ax2.set_xlim((750, 1150))
        # ax2.set_ylim((0.18, 0.26))
        ax2.set_ylim((0.0, 0.4))
        ax2.set_title('Spectra [Zoomed In]')


        ax3 = fig.add_subplot(311)
        ax3.plot(x0, flux_si_dset0[:, index_joint_si], lw=2, c='r'      , alpha=0.5, marker='o', markersize=1)
        ax3.plot(x1, flux_si_dset1[:, index_joint_si], lw=2, c='magenta', alpha=0.5, marker='o', markersize=1)
        ax3.plot(x0, flux_in_dset0[:, index_joint_in], lw=2, c='b'      , alpha=0.5, marker='o', markersize=1)
        ax3.plot(x1, flux_in_dset1[:, index_joint_in], lw=2, c='cyan'   , alpha=0.5, marker='o', markersize=1)

        ax3.scatter(index, flux_si_dset0[index, index_joint_si], s=250, c='r', lw=0.0, marker='*')
        ax3.scatter(index, flux_si_dset1[index, index_joint_si], s=250, c='magenta', lw=0.0, marker='*')
        ax3.scatter(index, flux_in_dset0[index, index_joint_in], s=250, c='b', lw=0.0, marker='*')
        ax3.scatter(index, flux_in_dset1[index, index_joint_in], s=250, c='cyan', lw=0.0, marker='*')

        ax3.axvline(index, color='gray', lw=1.5, ls='--')
        ax3.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
        ax3.set_xlabel('Index')
        # ax3.set_ylim((0.18, 0.26))
        ax3.set_ylim((0.0, 0.4))
        ax3.set_title('Time Series')
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.35, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%5.5d_%s_%s_[lasp|%s|%s].png' % (index, date_tag, _metadata['Function'], ssfr_tag.lower(), lc_tag.lower()), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#

def main_test_joint_wvl_skywatch():

    # skywatch
    #/----------------------------------------------------------------------------\#
    # for ssfr_tag in ['SSFR-A']:
    #     for lc_tag in ['zen', 'nad']:
    #         for date_tag in ['2023-10-27', '2023-10-30']:
    #             test_joint_wvl_skywatch(ssfr_tag, lc_tag, date_tag)
    #\----------------------------------------------------------------------------/#

    # skywatch
    #/----------------------------------------------------------------------------\#
    for ssfr_tag in ['SSFR-B']:
        # for lc_tag in ['zen', 'nad']:
        #     for date_tag in ['2023-10-19', '2023-10-20']:
        for lc_tag in ['zen']:
            for date_tag in ['2023-10-19']:
                test_joint_wvl_skywatch(ssfr_tag, lc_tag, date_tag)
    #\----------------------------------------------------------------------------/#


def cal_mean_stdv(fnames):

    ssfr_data = ssfr.lasp_ssfr.read_ssfr(fnames)


    index_spec = 3
    shutter0 = ssfr_data.data_raw['shutter']
    int_time_zen_in0 = ssfr_data.data_raw['int_time'][:, index_spec]
    cnt_zen_in_dset0 = ssfr_data.data_raw['spectra'][(shutter0==1)&(int_time_zen_in0==250.0), :, index_spec]
    cnt_zen_in_dset1 = ssfr_data.data_raw['spectra'][(shutter0==1)&(int_time_zen_in0==350.0), :, index_spec]

    # mean_dset0 = np.nanmean(cnt_zen_in_dset0, axis=0)
    mean_dset0 = np.nanmin(cnt_zen_in_dset0, axis=0)
    stdv_dset0 = np.nanstd(cnt_zen_in_dset0, axis=0)

    # mean_dset1 = np.nanmean(cnt_zen_in_dset1, axis=0)
    mean_dset1 = np.nanmin(cnt_zen_in_dset1, axis=0)
    stdv_dset1 = np.nanstd(cnt_zen_in_dset1, axis=0)

    return mean_dset0, mean_dset1, stdv_dset0, stdv_dset1

def cal_time_series(fnames):

    ssfr_data = ssfr.lasp_ssfr.read_ssfr(fnames)


    index_spec = 3
    shutter0 = ssfr_data.data_raw['shutter']
    int_time_zen_in0 = ssfr_data.data_raw['int_time'][:, index_spec]
    cnt_zen_in_dset0 = ssfr_data.data_raw['spectra'][(shutter0==1)&(int_time_zen_in0==250.0), :, index_spec]
    cnt_zen_in_dset1 = ssfr_data.data_raw['spectra'][(shutter0==1)&(int_time_zen_in0==350.0), :, index_spec]

    # mean_dset0 = np.nanmean(cnt_zen_in_dset0, axis=0)
    # mean_dset0 = np.nanmin(cnt_zen_in_dset0, axis=0)
    # stdv_dset0 = np.nanstd(cnt_zen_in_dset0, axis=0)
    mean_dset0 = cnt_zen_in_dset0[:, -6]
    stdv_dset0 = cnt_zen_in_dset0[:, -16]

    # mean_dset1 = np.nanmean(cnt_zen_in_dset1, axis=0)
    # mean_dset1 = np.nanmin(cnt_zen_in_dset1, axis=0)
    # stdv_dset1 = np.nanstd(cnt_zen_in_dset1, axis=0)
    mean_dset1 = cnt_zen_in_dset1[:, -6]
    stdv_dset1 = cnt_zen_in_dset1[:, -6]

    return mean_dset0, mean_dset1, stdv_dset0, stdv_dset1



def test_dark_cnt_spectra():

    wvls = ssfr.lasp_ssfr.get_ssfr_wvl('lasp|ssfr-b')
    wvl_zen_in = wvls['nad|in']

    fdir = '../examples/data/arcsix/cal/rad-cal/SSFR-B_2023-11-16_lab-rad-cal-zen-1324'
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))
    mean_dset0, mean_dset1, stdv_dset0, stdv_dset1 = cal_mean_stdv(fnames)

    fdir = '../examples/data/arcsix/pre-mission/raw/ssfr-b/2023-10-19'
    fnames = sorted(glob.glob('%s/*.SKS' % (fdir)))
    mean_dset0_, mean_dset1_, stdv_dset0_, stdv_dset1_ = cal_mean_stdv(fnames)


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.plot(wvl_zen_in, mean_dset0, lw=2, c='b')
        ax1.plot(wvl_zen_in, mean_dset1, lw=2, c='r')
        ax1.plot(wvl_zen_in, mean_dset0_, lw=1, c='cyan')
        ax1.plot(wvl_zen_in, mean_dset1_, lw=1, c='magenta')
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

def test_dark_cnt_time_series():

    wvls = ssfr.lasp_ssfr.get_ssfr_wvl('lasp|ssfr-b')
    wvl_zen_in = wvls['nad|in']

    fdir = '../examples/data/arcsix/cal/rad-cal/SSFR-B_2023-11-16_lab-rad-cal-zen-1324'
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))
    mean_dset0, mean_dset1, stdv_dset0, stdv_dset1 = cal_time_series(fnames)

    fdir = '../examples/data/arcsix/pre-mission/raw/ssfr-b/2023-10-19'
    fnames = sorted(glob.glob('%s/*.SKS' % (fdir)))
    mean_dset0_, mean_dset1_, stdv_dset0_, stdv_dset1_ = cal_time_series(fnames)


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        # ax1.plot(wvl_zen_in, mean_dset0, lw=2, c='b')
        # ax1.plot(wvl_zen_in, mean_dset1, lw=2, c='r')
        # ax1.plot(wvl_zen_in, mean_dset0_, lw=1, c='cyan')
        # ax1.plot(wvl_zen_in, mean_dset1_, lw=1, c='magenta')
        ax1.plot(mean_dset0, lw=2, c='b')
        ax1.plot(mean_dset1, lw=2, c='r')
        ax1.plot(mean_dset0_, lw=1, c='cyan')
        ax1.plot(mean_dset1_, lw=1, c='magenta')
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


def test_dark_cnt_spectra_ssfr_b():

    wvls = ssfr.lasp_ssfr.get_ssfr_wvl('lasp|ssfr-b')
    wvl_zen_in = wvls['nad|in']

    fdir = '../examples/data/arcsix/cal/rad-cal/SSFR-B_2023-11-16_lab-rad-cal-zen-1324'
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))
    mean_dset0, mean_dset1, stdv_dset0, stdv_dset1 = cal_mean_stdv(fnames)

    fdir = '../examples/data/arcsix/pre-mission/raw/ssfr-b/2023-10-19'
    fnames = sorted(glob.glob('%s/*.SKS' % (fdir)))
    mean_dset0_, mean_dset1_, stdv_dset0_, stdv_dset1_ = cal_mean_stdv(fnames)


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.plot(wvl_zen_in, mean_dset0, lw=2, c='b')
        ax1.plot(wvl_zen_in, mean_dset1, lw=2, c='r')
        ax1.plot(wvl_zen_in, mean_dset0_, lw=1, c='cyan')
        ax1.plot(wvl_zen_in, mean_dset1_, lw=1, c='magenta')
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

def test_dark_cnt_time_series_ssfr_b():

    wvls = ssfr.lasp_ssfr.get_ssfr_wvl('lasp|ssfr-b')
    wvl_zen_in = wvls['zen|in']

    fdir = '../examples/data/arcsix/cal/rad-cal/SSFR-B_2023-11-16_lab-rad-cal-zen-1324'
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))
    mean_dset0, mean_dset1, stdv_dset0, stdv_dset1 = cal_time_series(fnames)

    fdir = '../examples/data/arcsix/pre-mission/raw/ssfr-b/2023-10-19'
    fnames = sorted(glob.glob('%s/*.SKS' % (fdir)))
    mean_dset0_, mean_dset1_, stdv_dset0_, stdv_dset1_ = cal_time_series(fnames)


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        # ax1.plot(wvl_zen_in, mean_dset0, lw=2, c='b')
        # ax1.plot(wvl_zen_in, mean_dset1, lw=2, c='r')
        # ax1.plot(wvl_zen_in, mean_dset0_, lw=1, c='cyan')
        # ax1.plot(wvl_zen_in, mean_dset1_, lw=1, c='magenta')
        ax1.plot(mean_dset0, lw=2, c='b')
        ax1.plot(mean_dset1, lw=2, c='r')
        ax1.plot(mean_dset0_, lw=1, c='cyan')
        ax1.plot(mean_dset1_, lw=1, c='magenta')
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

def test_dark_cnt_spectra_ssfr_a():

    wvls = ssfr.lasp_ssfr.get_ssfr_wvl('lasp|ssfr-a')
    wvl_zen_in = wvls['nad|in']

    fdir = '../examples/data/arcsix/cal/rad-cal/SSFR-A_2023-11-16_lab-rad-cal-zen-1324'
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))
    mean_dset0, mean_dset1, stdv_dset0, stdv_dset1 = cal_mean_stdv(fnames)

    fdir = '../examples/data/arcsix/pre-mission/raw/ssfr-a/2023-10-27'
    fnames = sorted(glob.glob('%s/*.SKS' % (fdir)))
    mean_dset0_, mean_dset1_, stdv_dset0_, stdv_dset1_ = cal_mean_stdv(fnames)


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.plot(wvl_zen_in, mean_dset0, lw=2, c='b')
        ax1.plot(wvl_zen_in, mean_dset1, lw=2, c='r')
        ax1.plot(wvl_zen_in, mean_dset0_, lw=1, c='cyan')
        ax1.plot(wvl_zen_in, mean_dset1_, lw=1, c='magenta')
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

def test_dark_cnt_time_series_ssfr_a():

    wvls = ssfr.lasp_ssfr.get_ssfr_wvl('lasp|ssfr-a')
    wvl_zen_in = wvls['nad|in']

    fdir = '../examples/data/arcsix/cal/rad-cal/SSFR-A_2023-11-16_lab-rad-cal-nad-1324'
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))
    mean_dset0, mean_dset1, stdv_dset0, stdv_dset1 = cal_time_series(fnames)

    fdir = '../examples/data/arcsix/pre-mission/raw/ssfr-a/2023-10-27'
    fnames = sorted(glob.glob('%s/*.SKS' % (fdir)))
    mean_dset0_, mean_dset1_, stdv_dset0_, stdv_dset1_ = cal_time_series(fnames)


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        # ax1.plot(wvl_zen_in, mean_dset0, lw=2, c='b')
        # ax1.plot(wvl_zen_in, mean_dset1, lw=2, c='r')
        # ax1.plot(wvl_zen_in, mean_dset0_, lw=1, c='cyan')
        # ax1.plot(wvl_zen_in, mean_dset1_, lw=1, c='magenta')
        ax1.plot(mean_dset0, lw=2, c='b')
        ax1.plot(mean_dset1, lw=2, c='r')
        ax1.plot(mean_dset0_, lw=1, c='cyan')
        ax1.plot(mean_dset1_, lw=1, c='magenta')
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








def test_dark_cnt_time_series():

    which_lc = 'nad'
    index_spec = 3

    int_time0  = 250.0
    index_chan = -6
    index_temp = 2

    # wavelength ssfr-a
    #/----------------------------------------------------------------------------\#
    wvls = ssfr.lasp_ssfr.get_ssfr_wvl('lasp|ssfr-a')
    wvl_in_a = wvls['%s|in' % which_lc]
    #\----------------------------------------------------------------------------/#

    # ssfr-a cal data
    #/----------------------------------------------------------------------------\#
    fdir = '../examples/data/arcsix/cal/rad-cal/SSFR-A_2023-11-16_lab-rad-cal-%s-1324' % which_lc
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))
    ssfr_a_cal = ssfr.lasp_ssfr.read_ssfr(fnames)
    shutter0 = ssfr_a_cal.data_raw['shutter']
    int_time_in0 = ssfr_a_cal.data_raw['int_time'][:, index_spec]
    dark_cnt_in_cal_a0 = ssfr_a_cal.data_raw['spectra'][(shutter0==1)&(int_time_in0==int_time0), index_chan, index_spec]
    #\----------------------------------------------------------------------------/#

    # ssfr-a skywatch data
    #/----------------------------------------------------------------------------\#
    fdir = '../examples/data/arcsix/pre-mission/raw/ssfr-a/2023-10-27'
    fnames = sorted(glob.glob('%s/*.SKS' % (fdir)))
    ssfr_a_sky = ssfr.lasp_ssfr.read_ssfr(fnames)
    shutter0 = ssfr_a_sky.data_raw['shutter']
    int_time_in0 = ssfr_a_sky.data_raw['int_time'][:, index_spec]
    dark_cnt_in_sky_a0 = ssfr_a_sky.data_raw['spectra'][(shutter0==1)&(int_time_in0==int_time0), index_chan, index_spec]
    temp_a0 = ssfr_a_sky.data_raw['temp'][(shutter0==1)&(int_time_in0==int_time0), index_temp]
    #\----------------------------------------------------------------------------/#

    # wavelength ssfr-b
    #/----------------------------------------------------------------------------\#
    wvls = ssfr.lasp_ssfr.get_ssfr_wvl('lasp|ssfr-b')
    wvl_in_b = wvls['%s|in' % which_lc]
    #\----------------------------------------------------------------------------/#

    # ssfr-b cal data
    #/----------------------------------------------------------------------------\#
    fdir = '../examples/data/arcsix/cal/rad-cal/SSFR-B_2023-11-16_lab-rad-cal-%s-1324' % which_lc
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))
    ssfr_b_cal = ssfr.lasp_ssfr.read_ssfr(fnames)
    shutter0 = ssfr_b_cal.data_raw['shutter']
    int_time_in0 = ssfr_b_cal.data_raw['int_time'][:, index_spec]
    dark_cnt_in_cal_b0 = ssfr_b_cal.data_raw['spectra'][(shutter0==1)&(int_time_in0==int_time0), index_chan, index_spec]
    #\----------------------------------------------------------------------------/#

    # ssfr-b skywatch data
    #/----------------------------------------------------------------------------\#
    fdir = '../examples/data/arcsix/pre-mission/raw/ssfr-b/2023-10-19'
    fnames = sorted(glob.glob('%s/*.SKS' % (fdir)))
    ssfr_b_sky = ssfr.lasp_ssfr.read_ssfr(fnames)
    shutter0 = ssfr_b_sky.data_raw['shutter']
    int_time_in0 = ssfr_b_sky.data_raw['int_time'][:, index_spec]
    dark_cnt_in_sky_b0 = ssfr_b_sky.data_raw['spectra'][(shutter0==1)&(int_time_in0==int_time0), index_chan, index_spec]
    temp_b0 = ssfr_b_sky.data_raw['temp'][(shutter0==1)&(int_time_in0==int_time0), index_temp]
    #\----------------------------------------------------------------------------/#


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        fig.suptitle('%s' % which_lc.upper())
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        # ax1.plot(dark_cnt_in_cal_a0, lw=2, c='r')
        ax1.plot(dark_cnt_in_sky_a0, lw=2, c='magenta')
        # ax1.plot(dark_cnt_in_cal_b0, lw=2, c='magenta')
        ax1.plot(dark_cnt_in_sky_b0, lw=2, c='cyan')
        ax1.set_ylabel('Counts')
        ax1.set_xlabel('Time Index')


        ax2 = ax1.twinx()
        ax2.plot(temp_a0, lw=4.0, c='red' , ls='--')
        ax2.plot(temp_b0, lw=4.0, c='blue', ls='--')
        ax2.set_ylabel('Temperature', rotation=270)

        patches_legend = [
                         mpatches.Patch(color='red'   , label='SSFR-A Temp'), \
                         mpatches.Patch(color='blue'  , label='SSFR-B Temp'), \
                         mpatches.Patch(color='magenta', label='SSFR-A Dark'), \
                         mpatches.Patch(color='cyan'   , label='SSFR-B Dark'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)
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

    # main_test_joint_wvl_cal()
    # main_test_joint_wvl_skywatch()
    # test_dark_cnt_spectra_ssfr_b()
    # test_dark_cnt_time_series_ssfr_b()
    # test_dark_cnt_spectra_ssfr_a()
    # test_dark_cnt_time_series_ssfr_a()

    test_dark_cnt_time_series()
