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
mpl.use('Agg')


import ssfr


_mission_   = 'arcsix'
_spns_      = 'spns-b'
_ssfr_      = 'ssfr-b'
_fdir_data_ = '/argus/pre-mission/%s' % _mission_
_fdir_hsk_  = '%s/raw/hsk'
_fdir_ssfr_ = '%s/raw/%s' % (_fdir_data_, _ssfr_)
_fdir_spns_ = '%s/raw/%s' % (_fdir_data_, _spns_)
_fdir_v0_   = '%s/processed'  % _fdir_data_
_fdir_v1_   = '%s/processed'  % _fdir_data_
_fdir_v2_   = '%s/processed'  % _fdir_data_



# calibration
#/----------------------------------------------------------------------------\#


def test_data_a(ssfr_tag, lc_tag, lamp_tag, Nchan=256):

    fdir_data = '/argus/field/arcsix/cal/rad-cal'

    indices_spec = {
            'zen': [0, 1],
            'nad': [2, 3]
            }

    fdir =  sorted(glob.glob('%s/*%s*%s*%s*' % (fdir_data, ssfr_tag, lc_tag, lamp_tag)))[0]
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))


    date_cal_s   = '2023-11-16'
    date_today_s = datetime.datetime.now().strftime('%Y-%m-%d')

    ssfr_ = ssfr.lasp_ssfr.read_ssfr(fnames)
    for i in range(ssfr_.Ndset):
        dset_tag = 'dset%d' % i
        dset_ = getattr(ssfr_, dset_tag)
        int_time = dset_['info']['int_time']

        fname = '%s/cal/%s|RAD-CAL-PRI|LASP|%s|%s|%s-SI%3.3d-IN%3.3d|%s.h5' % (ssfr.common.fdir_data, date_cal_s, ssfr_tag.upper(), lc_tag.upper(), dset_tag.upper(), int_time['%s|si' % lc_tag], int_time['%s|in' % lc_tag], date_today_s)
        f = h5py.File(fname, 'w')

        resp_pri = ssfr.cal.cal_rad_resp(fnames, which_ssfr='lasp|%s' % ssfr_tag.lower(), which_lc=lc_tag.lower(), int_time=int_time, which_lamp=lamp_tag.lower())

        for key in resp_pri.keys():
            f[key] = resp_pri[key]

        f.close()

def test_data_b(
        date,
        fdir_data=_fdir_v1_,
        fdir_out=_fdir_v2_,
        pitch_angle=0.0,
        roll_angle=0.0,
        ):

    date_s = date.strftime('%Y-%m-%d')

    fname_h5 = '%s/%s_%s_%s_v2.h5' % (fdir_out, _mission_.upper(), _ssfr_.upper(), date_s)
    f = h5py.File(fname_h5, 'w')

    fname_h5 = '%s/%s_%s_%s_v1.h5' % (fdir_data, _mission_.upper(), _ssfr_.upper(), date_s)
    f_ = h5py.File(fname_h5, 'r')
    tmhr = f_['tmhr'][...]
    for dset_s in f_.keys():

        if 'dset' in dset_s:

            # primary calibration (from pre-mission arcsix in lab on 2023-11-16)
            #/----------------------------------------------------------------------------\#
            wvls = ssfr.lasp_ssfr.get_ssfr_wavelength()
            wvl_start = 350.0
            wvl_end   = 2100.0
            wvl_join  = 950.0

            # zenith wavelength
            #/----------------------------------------------------------------------------\#
            logic_zen_si = (wvls['zen|si'] >= wvl_start) & (wvls['zen|si'] <= wvl_join)
            logic_zen_in = (wvls['zen|in'] >  wvl_join)  & (wvls['zen|in'] <= wvl_end)

            wvl_zen = np.concatenate((wvls['zen|si'][logic_zen_si], wvls['zen|in'][logic_zen_in]))

            indices_sort_zen = np.argsort(wvl_zen)
            wvl_zen = wvl_zen[indices_sort_zen]
            #\----------------------------------------------------------------------------/#

            # nadir wavelength
            #/----------------------------------------------------------------------------\#
            logic_nad_si = (wvls['nad|si'] >= wvl_start) & (wvls['nad|si'] <= wvl_join)
            logic_nad_in = (wvls['nad|in'] >  wvl_join)  & (wvls['nad|in'] <= wvl_end)

            wvl_nad = np.concatenate((wvls['nad|si'][logic_nad_si], wvls['nad|in'][logic_nad_in]))

            indices_sort_nad = np.argsort(wvl_nad)
            wvl_nad = wvl_nad[indices_sort_nad]
            #\----------------------------------------------------------------------------/#

            fnames_zen = sorted(glob.glob('%s/cal/*RAD-CAL-PRI|LASP|%s|ZEN|%s*.h5' % (ssfr.common.fdir_data, _ssfr_.upper(), dset_s.upper())))
            fnames_nad = sorted(glob.glob('%s/cal/*RAD-CAL-PRI|LASP|%s|NAD|%s*.h5' % (ssfr.common.fdir_data, _ssfr_.upper(), dset_s.upper())))
            if len(fnames_zen) == 1 and len(fnames_nad) == 1:
                fname_zen = fnames_zen[0]
                fname_nad = fnames_nad[0]

                f_zen = h5py.File(fname_zen, 'r')
                sec_resp_zen_si = f_zen['zen|si'][...]
                sec_resp_zen_in = f_zen['zen|in'][...]
                f_zen.close()

                f_nad = h5py.File(fname_nad, 'r')
                sec_resp_nad_si = f_nad['nad|si'][...]
                sec_resp_nad_in = f_nad['nad|in'][...]
                f_nad.close()

                sec_resp_zen = np.concatenate((sec_resp_zen_si[logic_zen_si], sec_resp_zen_in[logic_zen_in]))[indices_sort_zen]
                sec_resp_nad = np.concatenate((sec_resp_nad_si[logic_nad_si], sec_resp_nad_in[logic_nad_in]))[indices_sort_nad]
            #\----------------------------------------------------------------------------/#

            # zenith
            #/--------------------------------------------------------------\#
            cnt_zen = f_['%s/cnt_zen' % dset_s][...]
            wvl_zen = f_['%s/wvl_zen' % dset_s][...]

            # sec_resp_zen = np.interp(wvl_zen, wvl_resp_zen_, sec_resp_zen_)

            flux_zen = cnt_zen.copy()
            for i in range(tmhr.size):
                if np.isnan(cnt_zen[i, :]).sum() == 0:
                    flux_zen[i, :] = cnt_zen[i, :] / sec_resp_zen
            #\--------------------------------------------------------------/#

            # nadir
            #/--------------------------------------------------------------\#
            cnt_nad = f_['%s/cnt_nad' % dset_s][...]
            wvl_nad = f_['%s/wvl_nad' % dset_s][...]

            # sec_resp_nad = np.interp(wvl_nad, wvl_resp_nad_, sec_resp_nad_)

            flux_nad = cnt_nad.copy()
            for i in range(tmhr.size):
                if np.isnan(cnt_nad[i, :]).sum() == 0:
                    flux_nad[i, :] = cnt_nad[i, :] / sec_resp_nad
            #\--------------------------------------------------------------/#

            g = f.create_group(dset_s)
            g['flux_zen'] = flux_zen
            g['flux_nad'] = flux_nad
            g['wvl_zen']  = wvl_zen
            g['wvl_nad']  = wvl_nad

        else:

            f[dset_s] = f_[dset_s][...]

    f_.close()

    f.close()

    return



def test_jointer_wavelength():

    pass





def test_joint_wvl_spectra(ssfr_tag, lc_tag, lamp_tag, Nchan=256):


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
    f = h5py.File(fnames_cal_dset0[0], 'r')
    resp_si_dset0 = f[si_tag][...]
    resp_in_dset0 = f[in_tag][...]
    f.close()

    dset_s = 'dset1'
    fnames_cal_dset1 = sorted(glob.glob('%s/cal/*cal-rad-pri|lasp|%s|%s|%s*.h5' % (ssfr.common.fdir_data, ssfr_tag.lower(), lc_tag.lower(), dset_s.lower())))
    f = h5py.File(fnames_cal_dset1[0], 'r')
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


    # index = 50
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


def main_test_joint_wvl_spectra():

    # radiometric calibration
    #/----------------------------------------------------------------------------\#
    for ssfr_tag in ['SSFR-A', 'SSFR-B']:
        for lc_tag in ['zen', 'nad']:
            for lamp_tag in ['1324']:
                test_joint_wvl_spectra(ssfr_tag, lc_tag, lamp_tag)
    #\----------------------------------------------------------------------------/#


if __name__ == '__main__':

    main_test_joint_wvl_spectra()
