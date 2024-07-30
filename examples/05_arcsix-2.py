"""
Code for processing data collected by "radiation instruments" during NASA ARCSIX 2024.

Acknowledgements:
    Instrument engineering:
        Jeffery Drouet, Sebastian Schmidt
    Pre-mission calibration and analysis:
        Hong Chen, Yu-Wen Chen, Ken Hirata, Sebastian Schmidt, Bruce Kindel
    In-field calibration and on-flight operation:
        Arabella Chamberlain, Ken Hirata, Vikas Nataraja, Sebastian Becker, Sebastian Schmidt
"""

import os
import sys
import glob
import datetime
import warnings
from collections import OrderedDict
from tqdm import tqdm
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav
from scipy.optimize import curve_fit
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



# parameters
#/----------------------------------------------------------------------------\#
_mission_     = 'arcsix'
_platform_    = 'p3b'

_hsk_         = 'hsk'
_alp_         = 'alp'
_ssfr1_       = 'ssfr-a'
_ssfr2_       = 'ssfr-b'
_cam_         = 'nac'

_spns_        = 'spns-b'
_which_ssfr_for_flux_ = 'ssfr-b'

_fdir_hsk_   = 'data/arcsix/2024/p3/aux/hsk'
_fdir_cal_   = 'data/%s/cal' % _mission_

_fdir_data_  = 'data/%s' % _mission_
_fdir_out_   = '%s/processed' % _fdir_data_


_verbose_   = True

_fnames_ = {}

_alp_time_offset_ = {
        '20240708':   -17.85,
        '20240709':   -17.85,
        '20240722':   -17.85,
        '20240724':   -17.85,
        '20240725':   -17.89,
        '20240726':   -17.89,
        '20240729':   -17.89,
        '20240730':   -17.89,
        }
_spns_time_offset_ = {
        '20240708': 0.0,
        '20240709': 0.0,
        '20240722': 0.0,
        '20240724': 0.0,
        '20240725': 0.0,
        '20240726': 0.0,
        '20240729': 9.69,
        '20240730': 9.69,
        }
_ssfr1_time_offset_ = {
        '20240708': -196.06,
        '20240709': -196.06,
        '20240722': -196.06,
        '20240724': -196.06,
        '20240725': -299.86,
        '20240726': -299.86,
        '20240729': -299.86,
        '20240730': -299.86,
        }
_ssfr2_time_offset_ = {
        '20240708': -273.59,
        '20240709': -273.59,
        '20240722': -273.59,
        '20240724': -273.59, #? inaccurate
        '20240725': -397.91,
        '20240726': -397.91,
        '20240729': -397.91,
        '20240730': -397.91,
        }
#\----------------------------------------------------------------------------/#



# functions for ssfr calibrations
#/----------------------------------------------------------------------------\#
def wvl_cal_old(ssfr_tag, lc_tag, lamp_tag, Nchan=256):

    fdir_data = '/argus/field/arcsix/cal/wvl-cal'

    indices_spec = {
            'zen': [0, 1],
            'nad': [2, 3]
            }

    fdir =  sorted(glob.glob('%s/*%s*%s*%s*' % (fdir_data, ssfr_tag, lc_tag, lamp_tag)))[0]
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))

    ssfr0 = ssfr.lasp_ssfr.read_ssfr(fnames, dark_corr_mode='interp')

    xchan = np.arange(Nchan)

    spectra0 = np.nanmean(ssfr0.dset0['spectra_dark-corr'][:, :, indices_spec[lc_tag]], axis=0)
    spectra1 = np.nanmean(ssfr0.dset1['spectra_dark-corr'][:, :, indices_spec[lc_tag]], axis=0)

    # spectra_inp = {lamp_tag.lower(): spectra0[:, 0]}
    # ssfr.cal.cal_wvl_coef(spectra_inp, which_spec='lasp|%s|%s|si' % (ssfr_tag.lower(), lc_tag.lower()))

    spectra_inp = {lamp_tag.lower(): spectra0[:, 1]}
    ssfr.cal.cal_wvl_coef(spectra_inp, which_spec='lasp|%s|%s|in' % (ssfr_tag.lower(), lc_tag.lower()))
    sys.exit()

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('%s %s (illuminated by %s Lamp)' % (ssfr_tag.upper(), lc_tag.title(), lamp_tag.upper()))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(121)
        ax1.plot(xchan, spectra0[:, 0], lw=1, c='r')
        ax1.plot(xchan, spectra1[:, 0], lw=1, c='b')
        ax1.set_xlabel('Channel #')
        ax1.set_ylabel('Counts')
        ax1.set_ylim(bottom=0)
        ax1.set_title('Silicon')

        ax2 = fig.add_subplot(122)
        ax2.plot(xchan, spectra0[:, 1], lw=1, c='r')
        ax2.plot(xchan, spectra1[:, 1], lw=1, c='b')
        ax2.set_xlabel('Channel #')
        ax2.set_ylabel('Counts')
        ax2.set_ylim(bottom=0)
        ax2.set_title('InGaAs')
        #\--------------------------------------------------------------/#

        patches_legend = [
                          mpatches.Patch(color='red' , label='IntTime set 1'), \
                          mpatches.Patch(color='blue', label='IntTime set 2'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_%s_%s_%s.png' % (_metadata['Function'], ssfr_tag.lower(), lc_tag.lower(), lamp_tag.lower()), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#

def rad_cal_old(ssfr_tag, lc_tag, lamp_tag, Nchan=256):

    fdir_data = 'data/arcsix/cal/rad-cal'

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

        fname = '%s/cal/%s|cal-rad-pri|lasp|%s|%s|%s-si%3.3d-in%3.3d|%s.h5' % (ssfr.common.fdir_data, date_cal_s, ssfr_tag.lower(), lc_tag.lower(), dset_tag.lower(), int_time['%s|si' % lc_tag], int_time['%s|in' % lc_tag], date_today_s)
        f = h5py.File(fname, 'w')

        resp_pri = ssfr.cal.cal_rad_resp(fnames, which_ssfr='lasp|%s' % ssfr_tag.lower(), which_lc=lc_tag.lower(), int_time=int_time, which_lamp=lamp_tag.lower())

        for key in resp_pri.keys():
            f[key] = resp_pri[key]

        f.close()

def ang_cal_old(fdir):

    """

    Notes:
        angular calibration is done for three different azimuth angles (reference to the vaccum port)
        60, 180, 300

        angles
    """

    tags = os.path.basename(fdir).split('_')
    ssfr_tag = tags[1]
    lc_tag   = tags[2]

    # get angles
    #/----------------------------------------------------------------------------\#
    angles_pos = np.concatenate((np.arange(0.0, 30.0, 3.0), np.arange(30.0, 50.0, 5.0), np.arange(50.0, 91.0, 10.0)))
    angles_neg = -angles_pos
    angles = np.concatenate((angles_pos, angles_neg, np.array([0.0])))
    #\----------------------------------------------------------------------------/#

    # make fnames, a dictionary <key:value> with file name as key, angle as value
    #/----------------------------------------------------------------------------\#
    fnames_ = sorted(glob.glob('%s/*.SKS' % fdir))
    fnames  = {
            fnames_[i]: angles[i] for i in range(angles.size)
            }
    #\----------------------------------------------------------------------------/#

    date_today_s = datetime.datetime.now().strftime('%Y-%m-%d')

    ssfr_ = ssfr.lasp_ssfr.read_ssfr([fnames_[0]])
    for i in range(ssfr_.Ndset):
        dset_tag = 'dset%d' % i
        dset_ = getattr(ssfr_, dset_tag)
        int_time = dset_['info']['int_time']

        filename_tag = '%s|%s|%s|%s' % (tags[0], tags[4], date_today_s, dset_tag)

        ssfr.cal.cdata_ang_resp(fnames, filename_tag=filename_tag, which_ssfr='lasp|%s' % ssfr_tag, which_lc=lc_tag, int_time=int_time)

def main_calibration_old():

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
    for ssfr_tag in ['SSFR-A', 'SSFR-B']:
        for lc_tag in ['zen', 'nad']:
            for lamp_tag in ['1324']:
                rad_cal(ssfr_tag, lc_tag, lamp_tag)
    #\----------------------------------------------------------------------------/#

    # angular calibration
    #/----------------------------------------------------------------------------\#
    # fdirs = [
    #         '/argus/field/arcsix/cal/ang-cal/2024-03-19_SSFR-A_zen-lc4_ang-cal_vaa-060_lamp-507_si-080-120_in-250-350',
    #         '/argus/field/arcsix/cal/ang-cal/2024-03-15_SSFR-A_zen-lc4_ang-cal_vaa-180_lamp-507_si-080-120_in-250-350',
    #         '/argus/field/arcsix/cal/ang-cal/2024-03-19_SSFR-A_zen-lc4_ang-cal_vaa-300_lamp-507_si-080-120_in-250-350',
    #         '/argus/field/arcsix/cal/ang-cal/2024-03-18_SSFR-A_nad-lc6_ang-cal_vaa-060_lamp-507_si-080-120_in-250-350',
    #         '/argus/field/arcsix/cal/ang-cal/2024-03-18_SSFR-A_nad-lc6_ang-cal_vaa-180_lamp-507_si-080-120_in-250-350',
    #         '/argus/field/arcsix/cal/ang-cal/2024-03-18_SSFR-A_nad-lc6_ang-cal_vaa-300_lamp-507_si-080-120_in-250-350',
    #         ]
    # for fdir in fdirs:
    #     ang_cal(fdir)
    #\----------------------------------------------------------------------------/#
    sys.exit()
#\----------------------------------------------------------------------------/#


# instrument calibrations
#/----------------------------------------------------------------------------\#
def rad_cal(
        fdir_pri,
        fdir_tra,
        fdir_sec=None,
        spec_reverse=False,
        ):

    # get calibration files of primary
    #/----------------------------------------------------------------------------\#
    tags_pri = os.path.basename(fdir_pri).split('_')
    fnames_pri_ = sorted(glob.glob('%s/*.SKS' % (fdir_pri)))
    fnames_pri = [fnames_pri_[-1]]
    if len(fnames_pri) > 1:
        msg = '\nWarning [rad_cal]: find more than one file for "%s", selected "%s" ...' % (fdir_pri, fnames_pri[0])
        warnings.warn(msg)
    #\----------------------------------------------------------------------------/#


    # get calibration files of transfer
    #/----------------------------------------------------------------------------\#
    tags_tra = os.path.basename(fdir_tra).split('_')
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
    tags_sec = os.path.basename(fdir_sec).split('_')
    fnames_sec_ = sorted(glob.glob('%s/*.SKS' % (fdir_sec)))
    fnames_sec = [fnames_sec_[-1]]
    if len(fnames_sec) > 1:
        msg = '\nWarning [rad_cal]: find more than one file for "%s", selected "%s" ...' % (fdir_sec, fnames_sec[0])
        warnings.warn(msg)
    #\----------------------------------------------------------------------------/#


    # tags
    #/----------------------------------------------------------------------------\#
    if (tags_pri[1]==tags_tra[1]) and (tags_tra[1]==tags_sec[1]):
        ssfr_tag = tags_pri[1]
    if (tags_pri[2]==tags_tra[2]) and (tags_tra[2]==tags_sec[2]):
        lc_tag = tags_pri[2]
    #\----------------------------------------------------------------------------/#

    date_today_s = datetime.datetime.now().strftime('%Y-%m-%d')

    ssfr_ = ssfr.lasp_ssfr.read_ssfr(fnames_pri, verbose=False)

    for i in range(ssfr_.Ndset):
        dset_tag = 'dset%d' % i
        int_time = ssfr_.dset_info[dset_tag]

        if len(tags_pri) == 7:
            cal_tag = '%s_%s' % (tags_pri[0], tags_pri[4])
        elif len(tags_pri) == 8:
            cal_tag = '%s_%s_%s' % (tags_pri[0], tags_pri[4], tags_pri[7])

        if len(tags_tra) == 7:
            cal_tag = '%s|%s_%s' % (cal_tag, tags_tra[0], tags_tra[4])
        elif len(tags_tra) == 8:
            cal_tag = '%s|%s_%s_%s' % (cal_tag, tags_tra[0], tags_tra[4], tags_tra[7])

        if len(tags_sec) == 7:
            cal_tag = '%s|%s_%s' % (cal_tag, tags_sec[0], tags_sec[4])
        elif len(tags_sec) == 8:
            cal_tag = '%s|%s_%s_%s' % (cal_tag, tags_sec[0], tags_sec[4], tags_sec[7])

        filename_tag = '%s|%s_processed-for-arcsix' % (cal_tag, date_today_s)

        ssfr.cal.cdata_rad_resp(fnames_pri=fnames_pri, fnames_tra=fnames_tra, fnames_sec=fnames_sec, which_ssfr='lasp|%s' % ssfr_tag, which_lc=lc_tag, int_time=int_time, which_lamp=tags_pri[4], filename_tag=filename_tag, verbose=True, spec_reverse=spec_reverse)

def main_calibration_rad():

    """
    Notes:
        irradiance setup:
            SSFR-A (Alvin)
              - zenith: LC4 + black plastic cased fiber
              - nadir : LC6 + stainless steel cased fiber

        irradiance backup setup:
            SSFR-B (Belana)
              - zenith: LC4 + black plastic cased fiber
              - nadir : LC6 + stainless steel cased fiber

    Available options for primary calibrations (pre-mission):
        data/arcsix/cal/rad-cal/2023-11-16_SSFR-A_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2023-11-16_SSFR-A_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2024-03-20_SSFR-A_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2024-03-20_SSFR-A_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2024-03-27_SSFR-A_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350_restart

        data/arcsix/cal/rad-cal/2024-03-25_SSFR-A_nad-lc6_pri-cal_lamp-506_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_nad-lc6_pri-cal_lamp-506_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_zen-lc4_pri-cal_lamp-506_si-080-120_in-250-350

        data/arcsix/cal/rad-cal/2023-11-16_SSFR-B_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2023-11-16_SSFR-B_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2024-03-27_SSFR-B_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350

    Available options for transfer (pre-mission):
        data/arcsix/cal/rad-cal/2024-03-20_SSFR-A_nad-lc6_transfer_lamp-150c_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2024-03-20_SSFR-A_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_nad-lc6_transfer_lamp-150c_si-080-120_in-250-350_after-pri
        data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350_after-pri

        data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_nad-lc6_transfer_lamp-150c_si-080-160_in-250-350
        data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350

        data/arcsix/cal/rad-cal/2024-03-25_SSFR-A_nad-lc6_transfer_lamp-150e_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2024-03-26_SSFR-A_zen-lc4_transfer_lamp-150e_si-080-120_in-250-350
        data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_nad-lc6_transfer_lamp-150e_si-080-120_in-250-350_after-pri
        data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_nad-lc6_transfer_lamp-150e_si-080-120_in-250-350_before-pri
        data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_nad-lc6_transfer_lamp-150e_si-080-120_in-250-350_fiber-zen
        data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_nad-lc6_transfer_lamp-150e_si-080-120_in-250-350_spec-zen
        data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_zen-lc4_transfer_lamp-150e_si-080-120_in-250-350_after-pri
        data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_zen-lc4_transfer_lamp-150e_si-080-120_in-250-350_before-pri
        data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_zen-lc4_transfer_lamp-150e_si-080-120_in-250-350_fiber-nad
        data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_zen-lc4_transfer_lamp-150e_si-080-120_in-250-350_restart
        data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_zen-lc4_transfer_lamp-150e_si-080-120_in-250-350_spec-nad

    Avaiable options for secondary calibrations (or known as field calibrations):
        data/arcsix/cal/rad-cal/2024-05-26_SSFR-A_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-05-27_SSFR-A_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-06-02_SSFR-A_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-06-02_SSFR-A_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-06-09_SSFR-A_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-06-09_SSFR-A_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik

        data/arcsix/cal/rad-cal/2024-06-02_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-06-02_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-07-23_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-07-26_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik

    fdirs = [
            {'zen': '',
             'nad': ''},
            ]
    """

    # radiometric calibration
    #/----------------------------------------------------------------------------\#
    fdir_cal = '%s/rad-cal' % _fdir_cal_

    # primary calibrations (pre-mission)
    #/----------------------------------------------------------------------------\#
    # fdirs_pri_cal = ssfr.util.get_all_folders(fdir_cal, pattern='*pri-cal_lamp-1324*si-080-120*in-250-350*')
    # fdirs_pri_cal = ssfr.util.get_all_folders(fdir_cal, pattern='*pri-cal_lamp-506*si-080-120*in-250-350*')
    # for fdir_pri in fdirs_pri_cal:
    #     print(fdir_pri)
    #\----------------------------------------------------------------------------/#

    # transfer (pre-mission)
    #/----------------------------------------------------------------------------\#
    # fdirs_transfer = ssfr.util.get_all_folders(fdir_cal, pattern='*transfer_lamp-150c*si-080-120*in-250-350*')
    # fdirs_transfer = ssfr.util.get_all_folders(fdir_cal, pattern='*transfer_lamp-150e*si-080-120*in-250-350*')
    # for fdir_transfer in fdirs_transfer:
    #     print(fdir_transfer)
    #\----------------------------------------------------------------------------/#

    # secondary calibrations (in-field)
    #/----------------------------------------------------------------------------\#
    # fdirs_sec_cal = ssfr.util.get_all_folders(fdir_cal, pattern='*sec-cal_lamp-150c*si-080-120*in-250-350*')
    # fdirs_sec_cal = ssfr.util.get_all_folders(fdir_cal, pattern='*sec-cal_lamp-150e*si-080-120*in-250-350*')
    # for fdir_sec_cal in fdirs_sec_cal:
    #     print(fdir_sec_cal)
    #\----------------------------------------------------------------------------/#


    # SSFR-A (regular setup for measuring irradiance)
    # /--------------------------------------------------------------------------\ #
    fdirs_pri = [
            {'zen': 'data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350',
             'nad': 'data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350'},
            ]

    fdirs_tra = [
            {'zen': 'data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350_after-pri',
             'nad': 'data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_nad-lc6_transfer_lamp-150c_si-080-120_in-250-350_after-pri'},
            ]

    fdirs_sec = [
            # {'zen': 'data/arcsix/cal/rad-cal/2024-05-27_SSFR-A_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
            #  'nad': 'data/arcsix/cal/rad-cal/2024-05-26_SSFR-A_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
            # {'zen': 'data/arcsix/cal/rad-cal/2024-06-02_SSFR-A_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
            #  'nad': 'data/arcsix/cal/rad-cal/2024-06-02_SSFR-A_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
            {'zen': 'data/arcsix/cal/rad-cal/2024-06-09_SSFR-A_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
             'nad': 'data/arcsix/cal/rad-cal/2024-06-09_SSFR-A_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
            ]
    # \--------------------------------------------------------------------------/ #


    # SSFR-B (backup setup for measuring irradiance)
    # /--------------------------------------------------------------------------\ #
    fdirs_pri = [
            {'zen': 'data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350',
             'nad': 'data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350'},
            ]

    fdirs_tra = [
            {'zen': 'data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350',
             'nad': 'data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_nad-lc6_transfer_lamp-150c_si-080-160_in-250-350'},
            ]

    fdirs_sec = [
            {'zen': 'data/arcsix/cal/rad-cal/2024-06-02_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
             'nad': 'data/arcsix/cal/rad-cal/2024-06-02_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
            {'zen': 'data/arcsix/cal/rad-cal/2024-07-26_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
             'nad': 'data/arcsix/cal/rad-cal/2024-07-23_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
            ]
    # \--------------------------------------------------------------------------/ #

    for fdir_pri in fdirs_pri:
        for fdir_tra in fdirs_tra:
            for fdir_sec in fdirs_sec:
                for spec_tag in fdir_sec.keys():
                    fdir_pri0 = fdir_pri[spec_tag]
                    fdir_tra0 = fdir_tra[spec_tag]
                    fdir_sec0 = fdir_sec[spec_tag]

                    print(spec_tag)
                    print(fdir_pri0)
                    print(fdir_tra0)
                    print(fdir_sec0)
                    rad_cal(fdir_pri0, fdir_tra0, fdir_sec=fdir_sec0, spec_reverse=False)
    #\----------------------------------------------------------------------------/#
    sys.exit()
#\----------------------------------------------------------------------------/#


# functions for processing HSK and ALP
#/----------------------------------------------------------------------------\#
def cdata_arcsix_hsk_v0(
        date,
        fdir_data=_fdir_data_,
        fdir_out=_fdir_out_,
        run=True,
        ):

    """
    For processing aricraft housekeeping file

    Notes:
        The housekeeping data would require some corrections before its release by the
        data system team, we usually request the raw IWG file (similar data but with a
        slightly different data formatting) from the team right after each flight to
        facilitate our data processing in a timely manner.
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v0.h5' % (fdir_out, _mission_.upper(), _hsk_.upper(), _platform_.upper(), date_s)
    if run:

        # this would change if we are processing IWG file
        #/--------------------------------------------------------------\#
        try:

            fname = ssfr.util.get_all_files(fdir_data, pattern='*%4.4d*%2.2d*%2.2d*.ict' % (date.year, date.month, date.day))[-1]
            data_hsk = ssfr.util.read_ict(fname)
            var_dict = {
                    'lon': 'longitude',
                    'lat': 'latitude',
                    'alt': 'gps_altitude',
                    'tmhr': 'tmhr',
                    'ang_pit': 'pitch_angle',
                    'ang_rol': 'roll_angle',
                    'ang_hed': 'true_heading',
                    'ir_surf_temp': 'ir_surf_temp',
                    }

        except Exception as error:
            print(error)

            fname = ssfr.util.get_all_files(fdir_data, pattern='*%4.4d*%2.2d*%2.2d*.iwg' % (date.year, date.month, date.day))[0]
            data_hsk = ssfr.util.read_iwg_nsrc(fname)
            var_dict = {
                    'tmhr': 'tmhr',
                    'lon': 'longitude',
                    'lat': 'latitude',
                    'alt': 'gps_alt_msl',
                    'ang_pit': 'pitch_angle',
                    'ang_rol': 'roll_angle',
                    'ang_hed': 'true_heading',
                    }

            # fname = ssfr.util.get_all_files(fdir_data, pattern='*%4.4d*%2.2d*%2.2d*.mts' % (date.year, date.month, date.day))[0]
            # data_hsk = ssfr.util.read_iwg_mts(fname)
            # var_dict = {
            #         'tmhr': 'tmhr',
            #         'lon': 'longitude',
            #         'lat': 'latitude',
            #         'alt': 'gps_msl_altitude',
            #         'ang_pit': 'pitch',
            #         'ang_rol': 'roll',
            #         'ang_hed': 'true_heading',
            #         }

        print()
        print('Processing HSK file:', fname)
        print()
        #\--------------------------------------------------------------/#


        # fake hsk for PSB (Pituffik Space Base)
        #/----------------------------------------------------------------------------\#
        # tmhr_range = [10.0, 13.5]
        # tmhr = np.arange(tmhr_range[0]*3600.0, tmhr_range[-1]*3600.0, 1.0)/3600.0
        # lon0 = -68.6471 # PSB longitude
        # lat0 = 76.5324  # PSB latitude
        # alt0 =  4.0     # airplane altitude
        # pit0 = 0.0
        # rol0 = 0.0
        # hed0 = 0.0
        # data_hsk = {
        #         'tmhr': {'data': tmhr, 'units': 'hour'},
        #         'long': {'data': np.repeat(lon0, tmhr.size), 'units': 'degree'},
        #         'lat' : {'data': np.repeat(lat0, tmhr.size), 'units': 'degree'},
        #         'palt': {'data': np.repeat(alt0, tmhr.size), 'units': 'meter'},
        #         'pitch'   : {'data': np.repeat(pit0, tmhr.size), 'units': 'degree'},
        #         'roll'    : {'data': np.repeat(rol0, tmhr.size), 'units': 'degree'},
        #         'heading' : {'data': np.repeat(hed0, tmhr.size), 'units': 'degree'},
        #         }
        # var_dict = {
        #         'lon': 'long',
        #         'lat': 'lat',
        #         'alt': 'palt',
        #         'tmhr': 'tmhr',
        #         'ang_pit': 'pitch',
        #         'ang_rol': 'roll',
        #         'ang_hed': 'heading',
        #         }
        #\----------------------------------------------------------------------------/#


        # fake hsk for NASA WFF
        #/----------------------------------------------------------------------------\#
        # if date == datetime.datetime(2024, 7, 8):
        #     dtime_s = datetime.datetime(2024, 7, 8, 18, 24)
        #     dtime_e = datetime.datetime(2024, 7, 8, 19, 1)
        # elif date == datetime.datetime(2024, 7, 9):
        #     dtime_s = datetime.datetime(2024, 7, 9, 15, 15)
        #     dtime_e = datetime.datetime(2024, 7, 9, 16, 5)
        # sec_s = (dtime_s - date).total_seconds()
        # sec_e = (dtime_e - date).total_seconds()
        # tmhr = np.arange(sec_s, sec_e, 1.0)/3600.0
        # lon0 = -75.47058922297123
        # lat0 = 37.94080738931773
        # alt0 =  4.0                # airplane altitude
        # pit0 = 0.0
        # rol0 = 0.0
        # hed0 = 0.0
        # data_hsk = {
        #         'tmhr': {'data': tmhr, 'units': 'hour'},
        #         'long': {'data': np.repeat(lon0, tmhr.size), 'units': 'degree'},
        #         'lat' : {'data': np.repeat(lat0, tmhr.size), 'units': 'degree'},
        #         'palt': {'data': np.repeat(alt0, tmhr.size), 'units': 'meter'},
        #         'pitch'   : {'data': np.repeat(pit0, tmhr.size), 'units': 'degree'},
        #         'roll'    : {'data': np.repeat(rol0, tmhr.size), 'units': 'degree'},
        #         'heading' : {'data': np.repeat(hed0, tmhr.size), 'units': 'degree'},
        #         }
        # var_dict = {
        #         'lon': 'long',
        #         'lat': 'lat',
        #         'alt': 'palt',
        #         'tmhr': 'tmhr',
        #         'ang_pit': 'pitch',
        #         'ang_rol': 'roll',
        #         'ang_hed': 'heading',
        #         }
        #\----------------------------------------------------------------------------/#


        # solar geometries
        #/----------------------------------------------------------------------------\#
        jday0 = ssfr.util.dtime_to_jday(date)
        jday  = jday0 + data_hsk[var_dict['tmhr']]['data']/24.0
        sza, saa = ssfr.util.cal_solar_angles(jday, data_hsk[var_dict['lon']]['data'], data_hsk[var_dict['lat']]['data'], data_hsk[var_dict['alt']]['data'])
        #\----------------------------------------------------------------------------/#

        # save processed data
        #/----------------------------------------------------------------------------\#
        f = h5py.File(fname_h5, 'w')
        for var in var_dict.keys():
            f[var] = data_hsk[var_dict[var]]['data']
        f['jday'] = jday
        f['sza']  = sza
        f['saa']  = saa
        f.close()
        #\----------------------------------------------------------------------------/#

    return fname_h5

def cdata_arcsix_hsk_from_alp_v0(
        date,
        fname_alp_v0,
        fdir_data=_fdir_data_,
        fdir_out=_fdir_out_,
        run=True,
        ):

    """
    For processing aricraft housekeeping file

    Notes:
        The housekeeping data would require some corrections before its release by the
        data system team, we usually request the raw IWG file (similar data but with a
        slightly different data formatting) from the team right after each flight to
        facilitate our data processing in a timely manner.
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v0.h5' % (fdir_out, _mission_.upper(), _hsk_.upper(), _platform_.upper(), date_s)
    if run:
        data_alp = ssfr.util.load_h5(fname_alp_v0)

        tmhr_ = data_alp['tmhr'][(data_alp['tmhr']>=0.0) & (data_alp['tmhr']<=48.0)]
        seconds_s = np.round(np.quantile(tmhr_, 0.02)*3600.0, decimals=0)
        seconds_e = np.round(np.quantile(tmhr_, 0.98)*3600.0, decimals=0)
        tmhr = (np.arange(seconds_s, seconds_e+1.0, 1.0) + _alp_time_offset_[date_s]) / 3600.0

        data_hsk = {}
        data_hsk['tmhr'] = {'data': tmhr}

        jday0 = ssfr.util.dtime_to_jday(date)
        jday  = jday0 + data_hsk['tmhr']['data']/24.0

        var_dict = {
                'lon': 'lon',
                'lat': 'lat',
                'alt': 'alt',
                'ang_pit': 'ang_pit_s',
                'ang_rol': 'ang_rol_s',
                'ang_hed': 'ang_hed',
                }

        for vname in var_dict.keys():

            data_hsk[vname] = {
                    'data': ssfr.util.interp(jday, data_alp['jday']+_alp_time_offset_[date_s]/86400.0, data_alp[var_dict[vname]], mode='linear')
                    }

        # solar geometries
        #/----------------------------------------------------------------------------\#
        sza, saa = ssfr.util.cal_solar_angles(jday, data_hsk['lon']['data'], data_hsk['lat']['data'], data_hsk['alt']['data'])
        #\----------------------------------------------------------------------------/#

        # save processed data
        #/----------------------------------------------------------------------------\#
        f = h5py.File(fname_h5, 'w')
        for var in data_hsk.keys():
            f[var] = data_hsk[var]['data']
        f['jday'] = jday
        f['sza']  = sza
        f['saa']  = saa
        f.close()
        #\----------------------------------------------------------------------------/#

    return fname_h5

def cdata_arcsix_alp_v0(
        date,
        fdir_data=_fdir_data_,
        fdir_out=_fdir_out_,
        run=True,
        ):

    """
    v0: directly read raw ALP (Active Leveling Platform) data

    Notes:
        ALP raw data has a finer temporal resolution than 1Hz and a higher measurement
        precision (or sensitivity) of the aircraft attitude.
    """

    date_s = date.strftime('%Y%m%d')

    # read ALP raw data
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/%s-%s_%s_%s_v0.h5' % (fdir_out, _mission_.upper(), _alp_.upper(), _platform_.upper(), date_s)
    if run:
        fnames_alp = ssfr.util.get_all_files(fdir_data, pattern='*.plt3')
        alp0 = ssfr.lasp_alp.read_alp(fnames_alp, date=date)
        alp0.save_h5(fname_h5)
    #\----------------------------------------------------------------------------/#

    return os.path.abspath(fname_h5)

def cdata_arcsix_alp_v1(
        date,
        fname_v0,
        fname_hsk,
        fdir_out=_fdir_out_,
        run=True
        ):

    """
    v1:
    1) calculate time offset (seconds) between aircraft housekeeping data and ALP raw data
       (referencing to aircraft housekeeping)
    2) interpolate raw alp data to aircraft housekeeping time

    Notes:
        ALP raw data has a finer temporal resolution than 1Hz and a higher measurement
        precision (or sensitivity) of the aircraft attitude.
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v1.h5' % (fdir_out, _mission_.upper(), _alp_.upper(), _platform_.upper(), date_s)

    if run:

        data_hsk = ssfr.util.load_h5(fname_hsk)
        data_alp = ssfr.util.load_h5(fname_v0)

        time_offset = _alp_time_offset_[date_s]


        f = h5py.File(fname_h5, 'w')
        f.attrs['description'] = 'v1:\n  1) raw data interpolated to HSK time frame;\n  2) time offset (seconds) was calculated and applied.'

        f['tmhr']        = data_hsk['tmhr']
        f['jday']        = data_hsk['jday']
        f['tmhr_ori']    = data_hsk['tmhr'] - time_offset/3600.0
        f['jday_ori']    = data_hsk['jday'] - time_offset/86400.0
        f['time_offset'] = time_offset
        f['sza']         = data_hsk['sza']
        f['saa']         = data_hsk['saa']

        jday_corr        = data_alp['jday'] + time_offset/86400.0
        for vname in data_alp.keys():
            if vname not in ['tmhr', 'jday']:
                f[vname] = ssfr.util.interp(data_hsk['jday'], jday_corr, data_alp[vname], mode='linear')
        f.close()

    return fname_h5
#\----------------------------------------------------------------------------/#


# functions for processing SPNS
#/----------------------------------------------------------------------------\#
def cdata_arcsix_spns_v0(
        date,
        fdir_data=_fdir_data_,
        fdir_out=_fdir_out_,
        run=True,
        ):

    """
    Process raw SPN-S data
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v0.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), _platform_.upper(), date_s)

    if run:

        # read spn-s raw data
        #/----------------------------------------------------------------------------\#
        fname_dif = ssfr.util.get_all_files(fdir_data, pattern='*Diffuse*.txt')[-1]
        data0_dif = ssfr.lasp_spn.read_spns(fname=fname_dif)

        fname_tot = ssfr.util.get_all_files(fdir_data, pattern='*Total*.txt')[-1]
        data0_tot = ssfr.lasp_spn.read_spns(fname=fname_tot)
        #/----------------------------------------------------------------------------\#

        # read wavelengths and calculate toa downwelling solar flux
        #/----------------------------------------------------------------------------\#
        flux_toa = ssfr.util.get_solar_kurudz()

        wvl_tot = data0_tot.data['wvl']
        f_dn_sol_tot = np.zeros_like(wvl_tot)
        for i, wvl0 in enumerate(wvl_tot):
            f_dn_sol_tot[i] = ssfr.util.cal_weighted_flux(wvl0, flux_toa[:, 0], flux_toa[:, 1])*ssfr.util.cal_solar_factor(date)
        #\----------------------------------------------------------------------------/#

        f = h5py.File(fname_h5, 'w')

        g1 = f.create_group('dif')
        for key in data0_dif.data.keys():
            if key in ['tmhr', 'jday', 'wvl', 'flux']:
                dset0 = g1.create_dataset(key, data=data0_dif.data[key], compression='gzip', compression_opts=9, chunks=True)

        g2 = f.create_group('tot')
        for key in data0_tot.data.keys():
            if key in ['tmhr', 'jday', 'wvl', 'flux']:
                dset0 = g2.create_dataset(key, data=data0_tot.data[key], compression='gzip', compression_opts=9, chunks=True)
        g2['toa0'] = f_dn_sol_tot

        f.close()

    return fname_h5

def cdata_arcsix_spns_v1(
        date,
        fname_spns_v0,
        fname_hsk,
        fdir_out=_fdir_out_,
        time_offset=0.0,
        run=True,
        ):

    """
    Check for time offset and merge SPN-S data with aircraft data
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v1.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), _platform_.upper(), date_s)

    if run:
        # read spn-s v0
        #/----------------------------------------------------------------------------\#
        data_spns_v0 = ssfr.util.load_h5(fname_spns_v0)
        #/----------------------------------------------------------------------------\#

        # read hsk v0
        #/----------------------------------------------------------------------------\#
        data_hsk= ssfr.util.load_h5(fname_hsk)
        #\----------------------------------------------------------------------------/#

        time_offset = _spns_time_offset_[date_s]

        # interpolate spn-s data to hsk time frame
        #/----------------------------------------------------------------------------\#
        flux_dif = np.zeros((data_hsk['jday'].size, data_spns_v0['dif/wvl'].size), dtype=np.float64)
        for i in range(flux_dif.shape[-1]):
            flux_dif[:, i] = ssfr.util.interp(data_hsk['jday'], data_spns_v0['dif/jday']+time_offset/86400.0, data_spns_v0['dif/flux'][:, i], mode='nearest')

        flux_tot = np.zeros((data_hsk['jday'].size, data_spns_v0['tot/wvl'].size), dtype=np.float64)
        for i in range(flux_tot.shape[-1]):
            flux_tot[:, i] = ssfr.util.interp(data_hsk['jday'], data_spns_v0['tot/jday']+time_offset/86400.0, data_spns_v0['tot/flux'][:, i], mode='nearest')
        #\----------------------------------------------------------------------------/#

        f = h5py.File(fname_h5, 'w')

        for key in data_hsk.keys():
            f[key] = data_hsk[key]

        f['time_offset'] = time_offset
        f['tmhr_ori'] = data_hsk['tmhr'] - time_offset/3600.0
        f['jday_ori'] = data_hsk['jday'] - time_offset/86400.0

        g1 = f.create_group('dif')
        g1['wvl']   = data_spns_v0['dif/wvl']
        dset0 = g1.create_dataset('flux', data=flux_dif, compression='gzip', compression_opts=9, chunks=True)

        g2 = f.create_group('tot')
        g2['wvl']   = data_spns_v0['tot/wvl']
        g2['toa0']  = data_spns_v0['tot/toa0']
        dset0 = g2.create_dataset('flux', data=flux_tot, compression='gzip', compression_opts=9, chunks=True)

        f.close()

    return fname_h5

def cdata_arcsix_spns_v2(
        date,
        fname_spns_v1,
        fname_hsk, # interchangable with fname_alp_v1
        wvl_range=None,
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        fdir_out=_fdir_out_,
        run=True,
        ):

    """
    Apply attitude correction to account for aircraft attitude (pitch, roll, heading)
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v2.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), _platform_.upper(), date_s)

    if run:

        # read spn-s v1
        #/----------------------------------------------------------------------------\#
        data_spns_v1 = ssfr.util.load_h5(fname_spns_v1)
        #/----------------------------------------------------------------------------\#

        # read hsk v0
        #/----------------------------------------------------------------------------\#
        data_hsk = ssfr.util.load_h5(fname_hsk)
        #/----------------------------------------------------------------------------\#

        # correction factor
        #/----------------------------------------------------------------------------\#
        mu = np.cos(np.deg2rad(data_hsk['sza']))

        try:
            iza, iaa = ssfr.util.prh2za(data_hsk['ang_pit']+ang_pit_offset, data_hsk['ang_rol']+ang_rol_offset, data_hsk['ang_hed'])
        except Exception as error:
            print(error)
            iza, iaa = ssfr.util.prh2za(data_hsk['ang_pit_s']+ang_pit_offset, data_hsk['ang_rol_s']+ang_rol_offset, data_hsk['ang_hed'])
        dc = ssfr.util.muslope(data_hsk['sza'], data_hsk['saa'], iza, iaa)

        factors = mu / dc
        #\----------------------------------------------------------------------------/#

        # attitude correction
        #/----------------------------------------------------------------------------\#
        f_dn_dir = data_spns_v1['tot/flux'] - data_spns_v1['dif/flux']
        f_dn_dir_corr = np.zeros_like(f_dn_dir)
        f_dn_tot_corr = np.zeros_like(f_dn_dir)
        for iwvl in range(data_spns_v1['tot/wvl'].size):
            f_dn_dir_corr[..., iwvl] = f_dn_dir[..., iwvl]*factors
            f_dn_tot_corr[..., iwvl] = f_dn_dir_corr[..., iwvl] + data_spns_v1['dif/flux'][..., iwvl]
        #\----------------------------------------------------------------------------/#

        f = h5py.File(fname_h5, 'w')


        g0 = f.create_group('att_corr')
        g0['mu'] = mu
        g0['dc'] = dc
        g0['factors'] = factors
        for key in data_hsk.keys():
            if key in ['sza', 'saa', 'ang_pit', 'ang_rol', 'ang_hed']:
                g0[key] = data_hsk[key]
            else:
                f[key] = data_hsk[key]

        if wvl_range is None:
            wvl_range = [0.0, 2200.0]

        logic_wvl_dif = (data_spns_v1['dif/wvl']>=wvl_range[0]) & (data_spns_v1['dif/wvl']<=wvl_range[1])
        logic_wvl_tot = (data_spns_v1['tot/wvl']>=wvl_range[0]) & (data_spns_v1['tot/wvl']<=wvl_range[1])

        g1 = f.create_group('dif')
        g1['wvl']   = data_spns_v1['dif/wvl'][logic_wvl_dif]
        dset0 = g1.create_dataset('flux', data=data_spns_v1['dif/flux'][:, logic_wvl_dif], compression='gzip', compression_opts=9, chunks=True)

        g2 = f.create_group('tot')
        g2['wvl']   = data_spns_v1['tot/wvl'][logic_wvl_tot]
        g2['toa0']  = data_spns_v1['tot/toa0'][logic_wvl_tot]
        dset0 = g2.create_dataset('flux', data=f_dn_tot_corr[:, logic_wvl_tot], compression='gzip', compression_opts=9, chunks=True)

        f.close()

    return fname_h5

def cdata_arcsix_spns_archive(
        date,
        fname_spns_v2,
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        wvl_range=[400.0, 800.0],
        platform_info = 'p3',
        principal_investigator_info = 'Chen, Hong',
        affiliation_info = 'University of Colorado Boulder',
        instrument_info = 'SPN-S (Sunshine Pyranometer - Spectral)',
        mission_info = 'ARCSIX 2024',
        project_info = '',
        file_format_index = '1001',
        file_volume_number = '1, 1',
        data_interval = '1.0',
        scale_factor = '1.0',
        fill_value = 'NaN',
        version='RA',
        fdir_out=_fdir_out_,
        run=True,
        ):


    # placeholder for additional information such as calibration
    #/----------------------------------------------------------------------------\#
    #\----------------------------------------------------------------------------/#


    # date info
    #/----------------------------------------------------------------------------\#
    date_s = date.strftime('%Y%m%d')
    date_today = datetime.date.today()
    date_info  = '%4.4d, %2.2d, %2.2d, %4.4d, %2.2d, %2.2d' % (date.year, date.month, date.day, date_today.year, date_today.month, date_today.day)
    #\----------------------------------------------------------------------------/#


    # version info
    #/----------------------------------------------------------------------------\#
    version = version.upper()
    version_info = {
            'RA': 'field data',
            }
    version_info = version_info[version]
    #\----------------------------------------------------------------------------/#


    # data info
    #/----------------------------------------------------------------------------\#
    data_info = 'Shortwave Total and Diffuse Downwelling Spectral Irradiance from %s %s' % (platform_info.upper(), instrument_info)
    #\----------------------------------------------------------------------------/#


    # routine comments
    #/----------------------------------------------------------------------------\#
    comments_routine_list = OrderedDict({
            'PI_CONTACT_INFO': 'Address: University of Colorado Boulder, LASP, 3665 Discovery Drive, Boulder, CO 80303; E-mail: hong.chen@lasp.colorado.edu and sebastian.schmidt@lasp.colorado.edu',
            'PLATFORM': platform_info.upper(),
            'LOCATION': 'N/A',
            'ASSOCIATED_DATA': 'N/A',
            'INSTRUMENT_INFO': instrument_info,
            'DATA_INFO': 'Reported are only of a selected wavelength range (%d-%d nm), time/lat/lon/alt/pitch/roll/heading from aircraft, sza calculated from time/lon/lat.' % (wvl_range[0], wvl_range[1]),
            'UNCERTAINTY': 'Nominal SPN-S uncertainty (shortwave): total: N/A; diffuse: N/A',
            'ULOD_FLAG': '-7777',
            'ULOD_VALUE': 'N/A',
            'LLOD_FLAG': '-8888',
            'LLOD_VALUE': 'N/A',
            'DM_CONTACT_INFO': 'N/A',
            'PROJECT_INFO': 'ARCSIX field experiment out of Pituffik, Greenland, May - August 2024',
            'STIPULATIONS_ON_USE': 'This is initial in-field release of the ARCSIX-2024 data set. Please consult the PI, both for updates to the data set, and for the proper and most recent interpretation of the data for specific science use.',
            'OTHER_COMMENTS': 'Minimal corrections were applied.\n',
            'REVISION': version,
            version: version_info
            })

    comments_routine = '\n'.join(['%s: %s' % (var0, comments_routine_list[var0]) for var0 in comments_routine_list.keys()])
    #\----------------------------------------------------------------------------/#


    # special comments
    #/----------------------------------------------------------------------------\#
    comments_special_dict = {
            '20240530': 'Noticed icing on dome after flight',
            }
    if date_s in comments_special_dict.keys():
        comments_special = comments_special_dict[date_s]
    else:
        comments_special = ''

    if comments_special != '':
        Nspecial = len(comments_special.split('\n'))
    else:
        Nspecial = 0
    #\----------------------------------------------------------------------------/#


    # data processing
    #/----------------------------------------------------------------------------\#
    data_v2 = ssfr.util.load_h5(fname_spns_v2)
    data_v2['tot/flux'][data_v2['tot/flux']<0.0] = np.nan
    data_v2['dif/flux'][data_v2['dif/flux']<0.0] = np.nan

    logic_tot = (data_v2['tot/wvl']>=wvl_range[0]) & (data_v2['tot/wvl']<=wvl_range[1])
    logic_dif = (data_v2['dif/wvl']>=wvl_range[0]) & (data_v2['dif/wvl']<=wvl_range[1])

    data = OrderedDict({
            'Time_Start': {
                'data': data_v2['tmhr']*3600.0,
                'unit': 'second',
                'description': 'UTC time in seconds from the midnight 00:00:00',
                },

            'jday': {
                'data': data_v2['jday'],
                'unit': 'day',
                'description': 'UTC time in decimal day from 0001-01-01 00:00:00',
                },

            'tmhr': {
                'data': data_v2['tmhr'],
                'unit': 'hour',
                'description': 'UTC time in decimal hour from the midnight 00:00:00',
                },

            'lon': {
                'data': data_v2['lon'],
                'unit': 'degree',
                'description': 'longitude',
                },

            'lat': {
                'data': data_v2['lat'],
                'unit': 'degree',
                'description': 'latitude',
                },

            'alt': {
                'data': data_v2['alt'],
                'unit': 'meter',
                'description': 'altitude',
                },

            'sza': {
                'data': data_v2['att_corr/sza'],
                'unit': 'degree',
                'description': 'solar zenith angle',
                },

            'tot/flux': {
                'data': data_v2['tot/flux'][:, logic_tot],
                'unit': 'W m^-2 nm^-1',
                'description': 'total downwelling spectral irradiance',
                },

            'tot/toa0': {
                'data': data_v2['tot/toa0'][logic_tot],
                'unit': 'W m^-2 nm^-1',
                'description': 'Kurucz reference total downwelling spectral irradiance',
                },

            'tot/wvl': {
                'data': data_v2['tot/wvl'][logic_tot],
                'unit': 'nm',
                'description': 'wavelength for total downwelling spectral irradiance',
                },

            'dif/flux': {
                'data': data_v2['dif/flux'][:, logic_dif],
                'unit': 'W m^-2 nm^-1',
                'description': 'diffuse downwelling spectral irradiance',
                },

            'dif/wvl': {
                'data': data_v2['dif/wvl'][logic_dif],
                'unit': 'nm',
                'description': 'wavelength for diffuse downwelling spectral irradiance',
                },
            })
    for key in data.keys():
        data[key]['description'] = '%s: %s, %s' % (key, data[key]['unit'], data[key]['description'])

    Nvar = len(data.keys())
    comments_routine = '%s\n%s' % (comments_routine, ','.join(data.keys()))
    Nroutine = len(comments_routine.split('\n'))
    #\----------------------------------------------------------------------------/#


    header_list = [file_format_index,
                   principal_investigator_info,
                   affiliation_info,       # Organization/affiliation of PI.
                   data_info,              # Data source description (e.g., instrument name, platform name, model name, etc.).
                   mission_info,           # Mission name (usually the mission acronym).
                   file_volume_number,     # File volume number, number of file volumes (these integer values are used when the data require more than one file per day; for data that require only one file these values are set to 1, 1) - comma delimited.
                   date_info,              # UTC date when data begin, UTC date of data reduction or revision - comma delimited (yyyy, mm, dd, yyyy, mm, dd).
                   data_interval,          # Data Interval (This value describes the time spacing (in seconds) between consecutive data records. It is the (constant) interval between values of the independent variable. For 1 Hz data the data interval value is 1 and for 10 Hz data the value is 0.1. All intervals longer than 1 second must be reported as Start and Stop times, and the Data Interval value is set to 0. The Mid-point time is required when it is not at the average of Start and Stop times. For additional information see Section 2.5 below.).
                   data['Time_Start']['description'],                # Description or name of independent variable (This is the name chosen for the start time. It always refers to the number of seconds UTC from the start of the day on which measurements began. It should be noted here that the independent variable should monotonically increase even when crossing over to a second day.).
                   str(Nvar-1),                                      # Number of variables (Integer value showing the number of dependent variables: the total number of columns of data is this value plus one.).
                   ', '.join([scale_factor for i in range(Nvar-1)]), # Scale factors (1 for most cases, except where grossly inconvenient) - comma delimited.
                   ', '.join([fill_value for i in range(Nvar-1)]),   # Missing data indicators (This is -9999 (or -99999, etc.) for any missing data condition, except for the main time (independent) variable which is never missing) - comma delimited.
                   '\n'.join([data[vname]['description'] for vname in data.keys() if vname != 'Time_Start']), # Variable names and units (Short variable name and units are required, and optional long descriptive name, in that order, and separated by commas. If the variable is unitless, enter the keyword "none" for its units. Each short variable name and units (and optional long name) are entered on one line. The short variable name must correspond exactly to the name used for that variable as a column header, i.e., the last header line prior to start of data.).
                   str(Nspecial),                                   # Number of SPECIAL comment lines (Integer value indicating the number of lines of special comments, NOT including this line.).
                   comments_special,
                   str(Nroutine),
                   comments_routine,
                ]


    header = '\n'.join([header0 for header0 in header_list if header0 != ''])

    Nline = len(header.split('\n'))
    header = '%d, %s' % (Nline, header)

    print(header)

    fname_h5 = '%s/%s-SPNS_%s_%s_%s.h5' % (fdir_out, _mission_.upper(), _platform_.upper(), date_s, version.upper())
    if run:
        f = h5py.File(fname_h5, 'w')

        dset = f.create_dataset('header', data=header)
        dset.attrs['description'] = 'header follows ICT format'

        for key in data.keys():
            dset = f.create_dataset(key, data=data[key]['data'], compression='gzip', compression_opts=9, chunks=True)
            dset.attrs['description'] = data[key]['description']
            dset.attrs['unit'] = data[key]['unit']
        f.close()

    return fname_h5
#\----------------------------------------------------------------------------/#


# functions for processing SSFR
#/----------------------------------------------------------------------------\#
def cdata_arcsix_ssfr_v0(
        date,
        fdir_data=_fdir_data_,
        fdir_out=_fdir_out_,
        which_ssfr='ssfr-a',
        run=True,
        ):

    """
    version 0: counts after dark correction
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v0.h5' % (fdir_out, _mission_.upper(), which_ssfr.upper(), _platform_.upper(), date_s)

    if run:
        fnames_ssfr = ssfr.util.get_all_files(fdir_data, pattern='*.SKS')

        ssfr0 = ssfr.lasp_ssfr.read_ssfr(fnames_ssfr, dark_corr_mode='interp', which_ssfr='lasp|%s' % which_ssfr.lower())

        # data that are useful
        #   wvl_zen [nm]
        #   cnt_zen [counts/ms]
        #   sat_zen: saturation tag, 1 means saturation
        #   wvl_nad [nm]
        #   cnt_nad [counts/ms]
        #   sat_nad: saturation tag, 1 means saturation
        #/----------------------------------------------------------------------------\#
        f = h5py.File(fname_h5, 'w')

        g = f.create_group('raw')
        for key in ssfr0.data_raw.keys():
            if isinstance(ssfr0.data_raw[key], np.ndarray):
                g.create_dataset(key, data=ssfr0.data_raw[key], compression='gzip', compression_opts=9, chunks=True)

        g = f.create_group('spec')
        for key in ssfr0.data_spec.keys():
            if isinstance(ssfr0.data_spec[key], np.ndarray):
                g.create_dataset(key, data=ssfr0.data_spec[key], compression='gzip', compression_opts=9, chunks=True)

        f.close()
        #\----------------------------------------------------------------------------/#

    return fname_h5

def cdata_arcsix_ssfr_v1(
        date,
        fname_ssfr_v0,
        fname_hsk,
        fdir_out=_fdir_out_,
        time_offset=0.0,
        which_ssfr_for_flux=_which_ssfr_for_flux_,
        run=True,
        ):

    """
    version 1: 1) time adjustment          : check for time offset and merge SSFR data with aircraft housekeeping data
               2) time synchronization     : interpolate raw SSFR data into the time frame of the housekeeping data
               3) counts-to-flux conversion: apply primary and secondary calibration to convert counts to fluxes
    """

    date_s = date.strftime('%Y%m%d')

    which_ssfr = os.path.basename(fname_ssfr_v0).split('_')[0].replace('%s-' % _mission_.upper(), '').lower()

    fname_h5 = '%s/%s-%s_%s_%s_v1.h5' % (fdir_out, _mission_.upper(), which_ssfr.upper(), _platform_.upper(), date_s)

    if run:

        # load ssfr v0 data
        #/----------------------------------------------------------------------------\#
        data_ssfr_v0 = ssfr.util.load_h5(fname_ssfr_v0)
        #\----------------------------------------------------------------------------/#


        # load hsk
        #/----------------------------------------------------------------------------\#
        data_hsk = ssfr.util.load_h5(fname_hsk)
        #\----------------------------------------------------------------------------/#


        # read wavelengths and calculate toa downwelling solar flux
        #/----------------------------------------------------------------------------\#
        flux_toa = ssfr.util.get_solar_kurudz()

        wvl_zen = data_ssfr_v0['spec/wvl_zen']
        f_dn_sol_zen = np.zeros_like(wvl_zen)
        for i, wvl0 in enumerate(wvl_zen):
            f_dn_sol_zen[i] = ssfr.util.cal_weighted_flux(wvl0, flux_toa[:, 0], flux_toa[:, 1])*ssfr.util.cal_solar_factor(date)
        #\----------------------------------------------------------------------------/#

        f = h5py.File(fname_h5, 'w')

        # processing data - since we have dual integration times, SSFR data with different
        # integration time will be processed seperately
        #/----------------------------------------------------------------------------\#
        jday     = data_ssfr_v0['raw/jday']
        dset_num = data_ssfr_v0['raw/dset_num']

        wvl_zen  = data_ssfr_v0['spec/wvl_zen']
        wvl_nad  = data_ssfr_v0['spec/wvl_nad']

        cnt_zen  = data_ssfr_v0['spec/cnt_zen']
        spec_zen = np.zeros_like(cnt_zen)
        cnt_nad  = data_ssfr_v0['spec/cnt_nad']
        spec_nad = np.zeros_like(cnt_nad)

        for idset in np.unique(dset_num):

            if which_ssfr_for_flux == which_ssfr:
                # select calibration file (can later be adjusted for different integration time sets)
                #/----------------------------------------------------------------------------\#
                fdir_cal = '%s/rad-cal' % _fdir_cal_

                jday_today = ssfr.util.dtime_to_jday(date)

                int_time_tag_zen = 'si-%3.3d|in-%3.3d' % (data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num']==idset][0, 0], data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num']==idset][0, 1])
                int_time_tag_nad = 'si-%3.3d|in-%3.3d' % (data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num']==idset][0, 2], data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num']==idset][0, 3])

                # fnames_cal_zen = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324|*lamp-150c_after-pri|*pituffik*%s*zen*%s*' % (which_ssfr_for_flux.lower(), int_time_tag_zen)), key=os.path.getmtime)
                fnames_cal_zen = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324|*lamp-150c*|*pituffik*%s*zen*%s*' % (which_ssfr_for_flux.lower(), int_time_tag_zen)), key=os.path.getmtime)
                jday_cal_zen = np.zeros(len(fnames_cal_zen), dtype=np.float64)
                for i in range(jday_cal_zen.size):
                    dtime0_s = os.path.basename(fnames_cal_zen[i]).split('|')[2].split('_')[0]
                    dtime0 = datetime.datetime.strptime(dtime0_s, '%Y-%m-%d')
                    jday_cal_zen[i] = ssfr.util.dtime_to_jday(dtime0)
                fname_cal_zen = fnames_cal_zen[np.argmin(np.abs(jday_cal_zen-jday_today))]
                data_cal_zen = ssfr.util.load_h5(fname_cal_zen)

                msg = '\nMessage [cdata_arcsix_ssfr_v1]: Using <%s> for %s zenith irradiance ...' % (os.path.basename(fname_cal_zen), which_ssfr.upper())
                print(msg)

                # fnames_cal_nad = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324|*lamp-150c_after-pri|*pituffik*%s*nad*%s*' % (which_ssfr_for_flux.lower(), int_time_tag_nad)), key=os.path.getmtime)
                fnames_cal_nad = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324|*lamp-150c*|*pituffik*%s*nad*%s*' % (which_ssfr_for_flux.lower(), int_time_tag_nad)), key=os.path.getmtime)
                jday_cal_nad = np.zeros(len(fnames_cal_nad), dtype=np.float64)
                for i in range(jday_cal_nad.size):
                    dtime0_s = os.path.basename(fnames_cal_nad[i]).split('|')[2].split('_')[0]
                    dtime0 = datetime.datetime.strptime(dtime0_s, '%Y-%m-%d')
                    jday_cal_nad[i] = ssfr.util.dtime_to_jday(dtime0)
                fname_cal_nad = fnames_cal_nad[np.argmin(np.abs(jday_cal_nad-jday_today))]
                data_cal_nad = ssfr.util.load_h5(fname_cal_nad)

                msg = '\nMessage [cdata_arcsix_ssfr_v1]: Using <%s> for %s nadir irradiance ...' % (os.path.basename(fname_cal_nad), which_ssfr.upper())
                print(msg)
                #\----------------------------------------------------------------------------/#
            else:
                # radiance (scale the data to 0 - 2.0 for now,
                # later we will apply radiometric response after mission to retrieve spectral RADIANCE)

                factor_zen = (np.nanmax(cnt_zen)-np.nanmin(cnt_zen)) / 2.0
                data_cal_zen = {
                        'sec_resp': np.repeat(factor_zen, wvl_zen.size)
                        }

                msg = '\nMessage [cdata_arcsix_ssfr_v1]: Using [0, 2.0] scaling for %s zenith radiance ...' % (which_ssfr.upper())
                print(msg)

                factor_nad = (np.nanmax(cnt_nad)-np.nanmin(cnt_nad)) / 2.0
                data_cal_nad = {
                        'sec_resp': np.repeat(factor_nad, wvl_nad.size)
                        }

                msg = '\nMessage [cdata_arcsix_ssfr_v1]: Using [0, 2.0] scaling for %s nadir radiance ...' % (which_ssfr.upper())
                print(msg)

            logic_dset = (dset_num == idset)

            # convert counts to flux
            #/----------------------------------------------------------------------------\#
            for i in range(wvl_zen.size):
                spec_zen[logic_dset, i] = cnt_zen[logic_dset, i] / data_cal_zen['sec_resp'][i]

            for i in range(wvl_nad.size):
                spec_nad[logic_dset, i] = cnt_nad[logic_dset, i] / data_cal_nad['sec_resp'][i]
            #\----------------------------------------------------------------------------/#

            # set saturation to 0
            #/--------------------------------------------------------------\#
            spec_zen[data_ssfr_v0['spec/sat_zen']==1] = -0.05
            spec_nad[data_ssfr_v0['spec/sat_nad']==1] = -0.05
            #\--------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#


        # check time offset
        #/----------------------------------------------------------------------------\#
        if which_ssfr == 'ssfr-a':
            time_offset = _ssfr1_time_offset_[date_s]
        elif which_ssfr == 'ssfr-b':
            time_offset = _ssfr2_time_offset_[date_s]
        #\----------------------------------------------------------------------------/#


        # interpolate ssfr data to hsk time frame
        # and convert counts to flux
        #/----------------------------------------------------------------------------\#
        cnt_zen_hsk  = np.zeros((data_hsk['jday'].size, wvl_zen.size), dtype=np.float64)
        spec_zen_hsk = np.zeros_like(cnt_zen_hsk)

        for i in range(wvl_zen.size):
            cnt_zen_hsk[:, i]  = ssfr.util.interp(data_hsk['jday'], jday+time_offset/86400.0, cnt_zen[:, i], mode='nearest')
            spec_zen_hsk[:, i] = ssfr.util.interp(data_hsk['jday'], jday+time_offset/86400.0, spec_zen[:, i], mode='nearest')

        cnt_nad_hsk  = np.zeros((data_hsk['jday'].size, wvl_nad.size), dtype=np.float64)
        spec_nad_hsk = np.zeros_like(cnt_nad_hsk)
        for i in range(wvl_nad.size):
            cnt_nad_hsk[:, i]  = ssfr.util.interp(data_hsk['jday'], jday+time_offset/86400.0, cnt_nad[:, i], mode='nearest')
            spec_nad_hsk[:, i] = ssfr.util.interp(data_hsk['jday'], jday+time_offset/86400.0, spec_nad[:, i], mode='nearest')
        #\----------------------------------------------------------------------------/#


        # save processed data
        #/----------------------------------------------------------------------------\#
        g0 = f.create_group('v0')
        g0.create_dataset('jday', data=jday+time_offset/86400.0, compression='gzip', compression_opts=9, chunks=True)
        g0.create_dataset('wvl_zen', data=wvl_zen, compression='gzip', compression_opts=9, chunks=True)
        g0.create_dataset('wvl_nad', data=wvl_nad, compression='gzip', compression_opts=9, chunks=True)
        g0.create_dataset('spec_zen', data=spec_zen, compression='gzip', compression_opts=9, chunks=True)
        g0.create_dataset('spec_nad', data=spec_nad, compression='gzip', compression_opts=9, chunks=True)

        g1 = f.create_group('zen')
        g1.create_dataset('wvl' , data=wvl_zen     , compression='gzip', compression_opts=9, chunks=True)
        g1.create_dataset('cnt' , data=cnt_zen_hsk , compression='gzip', compression_opts=9, chunks=True)
        if which_ssfr_for_flux == which_ssfr:
            g1.create_dataset('flux', data=spec_zen_hsk, compression='gzip', compression_opts=9, chunks=True)
            g1.create_dataset('toa0', data=f_dn_sol_zen, compression='gzip', compression_opts=9, chunks=True)
        else:
            g1.create_dataset('rad', data=spec_zen_hsk, compression='gzip', compression_opts=9, chunks=True)

        g2 = f.create_group('nad')
        g2.create_dataset('wvl' , data=wvl_nad     , compression='gzip', compression_opts=9, chunks=True)
        g2.create_dataset('cnt' , data=cnt_nad_hsk , compression='gzip', compression_opts=9, chunks=True)
        if which_ssfr_for_flux == which_ssfr:
            g2.create_dataset('flux', data=spec_nad_hsk, compression='gzip', compression_opts=9, chunks=True)
        else:
            g2.create_dataset('rad', data=spec_nad_hsk, compression='gzip', compression_opts=9, chunks=True)
        #\----------------------------------------------------------------------------/#


        # save processed data
        #/----------------------------------------------------------------------------\#
        for key in data_hsk.keys():
            f[key] = data_hsk[key]

        f['time_offset'] = time_offset
        f['tmhr_ori'] = data_hsk['tmhr'] - time_offset/3600.0
        f['jday_ori'] = data_hsk['jday'] - time_offset/86400.0

        f.close()
        #\----------------------------------------------------------------------------/#

    return fname_h5

def cdata_arcsix_ssfr_v2(
        date,
        fname_ssfr_v1,
        fname_alp_v1,
        fname_spns_v2,
        fdir_out=_fdir_out_,
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        run=True,
        run_aux=True,
        ):

    """
    version 2: apply cosine correction to correct for non-linear angular resposne
               diffuse radiation: use cosine response integrated over the entire angular space
               direct radiation: use cosine response over the entire angular space measured in the lab

               diffuse and direct seperation is guided by the diffuse ratio measured by SPNS
    """

    def func_diff_ratio(x, a, b, c):

        return a * (x/500.0)**(b) + c

    def fit_diff_ratio(wavelength, ratio):

        popt, pcov = curve_fit(func_diff_ratio, wavelength, ratio, maxfev=1000000, bounds=(np.array([0.0, -np.inf, 0.0]), np.array([np.inf, 0.0, np.inf])))

        return popt, pcov

    date_s = date.strftime('%Y%m%d')

    which_ssfr = os.path.basename(fname_ssfr_v1).split('_')[0].replace('%s-' % _mission_.upper(), '').lower()

    fname_h5 = '%s/%s-%s_%s_%s_v2.h5' % (fdir_out, _mission_.upper(), which_ssfr.upper(), _platform_.upper(), date_s)

    if run:

        data_ssfr_v1 = ssfr.util.load_h5(fname_ssfr_v1)

        # temporary fix to bypass the attitude correction for SSFR-B
        # /--------------------------------------------------------------------------\ #
        if data_ssfr_v1['zen/wvl'].size > 424:
            data_ssfr_v1['zen/toa0'] = data_ssfr_v1['zen/toa0'][:424]
            data_ssfr_v1['zen/wvl'] = data_ssfr_v1['zen/wvl'][:424]
            data_ssfr_v1['zen/flux'] = data_ssfr_v1['zen/flux'][:, :424]
            data_ssfr_v1['zen/cnt'] = data_ssfr_v1['zen/cnt'][:, :424]
            data_ssfr_v1['v0/spec_zen'] = data_ssfr_v1['v0/spec_zen'][:, :424]
            data_ssfr_v1['v0/wvl_zen'] = data_ssfr_v1['v0/wvl_zen'][:424]
        # \--------------------------------------------------------------------------/ #

        fname_aux = '%s/%s-%s_%s_%s_v2-aux.h5' % (fdir_out, _mission_.upper(), which_ssfr.upper(), _platform_.upper(), date_s)

        if run_aux:

            # calculate diffuse/global ratio from SPNS data
            #/----------------------------------------------------------------------------\#
            data_spns_v2 = ssfr.util.load_h5(fname_spns_v2)

            f_ = h5py.File(fname_aux, 'w')

            wvl_ssfr_zen = data_ssfr_v1['zen/wvl']
            wvl_spns     = data_spns_v2['tot/wvl']

            Nt, Nwvl = data_ssfr_v1['zen/flux'].shape

            diff_ratio = np.zeros((Nt, Nwvl), dtype=np.float64)
            diff_ratio[...] = np.nan

            poly_coefs = np.zeros((Nt, 3), dtype=np.float64)
            poly_coefs[...] = np.nan

            qual_flag = np.repeat(0, Nt)

            # do spectral fit based on 400 nm - 750 nm observations
            #/--------------------------------------------------------------\#
            for i in tqdm(range(Nt)):

                diff_ratio0_spns = data_spns_v2['dif/flux'][i, :] / data_spns_v2['tot/flux'][i, :]
                logic_valid = (~np.isnan(diff_ratio0_spns)) & (diff_ratio0_spns>=0.0) & (diff_ratio0_spns<=1.0) & (wvl_spns>=400.0) & (wvl_spns<=750.0)
                if logic_valid.sum() > 20:

                    x = data_spns_v2['tot/wvl'][logic_valid]
                    y = diff_ratio0_spns[logic_valid]
                    popt, pcov = fit_diff_ratio(x, y)

                    diff_ratio[i, :] = func_diff_ratio(wvl_ssfr_zen, *popt)
                    poly_coefs[i, :] = popt

                    qual_flag[i] = 1

            diff_ratio[diff_ratio<0.0] = 0.0
            diff_ratio[diff_ratio>1.0] = 1.0
            #\--------------------------------------------------------------/#

            # fill in nan values in time space
            #/--------------------------------------------------------------\#
            for i in range(Nwvl):

                logic_nan   = np.isnan(diff_ratio[:, i])
                logic_valid = ~logic_nan
                f_interp = interpolate.interp1d(data_ssfr_v1['tmhr'][logic_valid], diff_ratio[:, i][logic_valid], bounds_error=None, fill_value='extrapolate')
                diff_ratio[logic_nan, i] = f_interp(data_ssfr_v1['tmhr'][logic_nan])

            diff_ratio[diff_ratio<0.0] = 0.0
            diff_ratio[diff_ratio>1.0] = 1.0
            #\--------------------------------------------------------------/#

            # save data
            #/--------------------------------------------------------------\#
            f_.create_dataset('diff_ratio', data=diff_ratio  , compression='gzip', compression_opts=9, chunks=True)
            g_ = f_.create_group('diff_ratio_aux')
            g_.create_dataset('wvl'       , data=wvl_ssfr_zen, compression='gzip', compression_opts=9, chunks=True)
            g_.create_dataset('coef'      , data=poly_coefs  , compression='gzip', compression_opts=9, chunks=True)
            g_.create_dataset('qual_flag' , data=qual_flag   , compression='gzip', compression_opts=9, chunks=True)
            #\--------------------------------------------------------------/#
            #\----------------------------------------------------------------------------/#


            # alp
            #/----------------------------------------------------------------------------\#
            data_alp_v1  = ssfr.util.load_h5(fname_alp_v1)
            for key in data_alp_v1.keys():
                try:
                    f_.create_dataset(key, data=data_alp_v1[key], compression='gzip', compression_opts=9, chunks=True)
                except TypeError as error:
                    print(error)
                    f_[key] = data_alp_v1[key]
            f_.close()
            #\----------------------------------------------------------------------------/#

        # aux data processing ends here

        data_aux = ssfr.util.load_h5(fname_aux)

        # diffuse ratio
        #/----------------------------------------------------------------------------\#
        diff_ratio = data_aux['diff_ratio']
        #\----------------------------------------------------------------------------/#


        # angles
        #/----------------------------------------------------------------------------\#
        angles = {}
        angles['sza'] = data_aux['sza']
        angles['saa'] = data_aux['saa']
        angles['ang_pit']   = data_aux['ang_pit_s'] # pitch angle from SPAN-CPT
        angles['ang_rol']   = data_aux['ang_rol_s'] # roll angle from SPAN-CPT
        angles['ang_hed']   = data_aux['ang_hed']
        angles['ang_pit_m'] = data_aux['ang_pit_m']
        angles['ang_rol_m'] = data_aux['ang_rol_m']
        angles['ang_pit_offset'] = ang_pit_offset
        angles['ang_rol_offset'] = ang_rol_offset
        #\----------------------------------------------------------------------------/#


        # select calibration file for attitude correction
        # angular response is relative change, thus irrelavant to integration time (ideally)
        # and is intrinsic property of light collector, thus fixed to use SSFR-A with larger
        # integration time for consistency and simplicity, will revisit this after mission
        #/----------------------------------------------------------------------------\#
        dset_s = 'dset1'
        fdir_cal = '%s/ang-cal' % _fdir_cal_
        fname_cal_zen = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*vaa-180|*%s*%s*zen*' % (dset_s, 'ssfr-a')), key=os.path.getmtime)[-1]
        fname_cal_nad = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*vaa-180|*%s*%s*nad*' % (dset_s, 'ssfr-a')), key=os.path.getmtime)[-1]
        #\----------------------------------------------------------------------------/#


        # calculate attitude correction factors
        #/----------------------------------------------------------------------------\#
        fnames_cal = {
                'zen': fname_cal_zen,
                'nad': fname_cal_nad,
                }
        factors = ssfr.corr.att_corr(fnames_cal, angles, diff_ratio=diff_ratio)
        #\----------------------------------------------------------------------------/#

        # save data
        #/----------------------------------------------------------------------------\#
        f = h5py.File(fname_h5, 'w')
        for key in ['tmhr', 'jday', 'lon', 'lat', 'alt']:
            f.create_dataset(key, data=data_aux[key], compression='gzip', compression_opts=9, chunks=True)

        g1 = f.create_group('att_corr')
        g1.create_dataset('factors_zen', data=factors['zen'], compression='gzip', compression_opts=9, chunks=True)
        g1.create_dataset('factors_nad', data=factors['nad'], compression='gzip', compression_opts=9, chunks=True)
        for key in ['sza', 'saa', 'ang_pit_s', 'ang_rol_s', 'ang_hed', 'ang_pit_m', 'ang_rol_m']:
            g1.create_dataset(key, data=data_aux[key], compression='gzip', compression_opts=9, chunks=True)

        # apply attitude correction
        #/----------------------------------------------------------------------------\#
        g2 = f.create_group('zen')
        g2.create_dataset('flux', data=data_ssfr_v1['zen/flux']*factors['zen'], compression='gzip', compression_opts=9, chunks=True)
        g2.create_dataset('wvl' , data=data_ssfr_v1['zen/wvl']                , compression='gzip', compression_opts=9, chunks=True)
        g2.create_dataset('toa0', data=data_ssfr_v1['zen/toa0']               , compression='gzip', compression_opts=9, chunks=True)

        g3 = f.create_group('nad')
        g3.create_dataset('flux', data=data_ssfr_v1['nad/flux']*factors['nad'], compression='gzip', compression_opts=9, chunks=True)
        g3.create_dataset('wvl' , data=data_ssfr_v1['nad/wvl']                , compression='gzip', compression_opts=9, chunks=True)
        #\----------------------------------------------------------------------------/#

        f.close()

    return fname_h5

def cdata_arcsix_ssfr_archive(
        date,
        fname_ssfr_v2,
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        wvl_range=[350.0, 2000.0],
        platform_info = 'P3',
        principal_investigator_info = 'Chen, Hong',
        affiliation_info = 'University of Colorado Boulder',
        mission_info = 'ARCSIX 2024',
        project_info = '',
        file_format_index = '1001',
        file_volume_number = '1, 1',
        data_interval = '1.0',
        scale_factor = '1.0',
        fill_value = 'NaN',
        version='RA',
        fdir_out=_fdir_out_,
        run=True,
        ):


    # placeholder for additional information such as calibration
    #/----------------------------------------------------------------------------\#
    # comments_list = []
    # comments_list.append('Bandwidth of Silicon channels (wavelength < 950nm) as defined by the FWHM: 6 nm')
    # comments_list.append('Bandwidth of InGaAs channels (wavelength > 950nm) as defined by the FWHM: 12 nm')
    # comments_list.append('Pitch angle offset: %.1f degree' % pitch_angle)
    # comments_list.append('Roll angle offset: %.1f degree' % roll_angle)

    # for key in fnames_rad_cal.keys():
    #     comments_list.append('Radiometric calibration file (%s): %s' % (key, os.path.basename(fnames_rad_cal[key])))
    # for key in fnames_ang_cal.keys():
    #     comments_list.append('Angular calibration file (%s): %s' % (key, os.path.basename(fnames_ang_cal[key])))
    # comments = '\n'.join(comments_list)

    # print(date_s)
    # print(comments)
    # print()
    #\----------------------------------------------------------------------------/#


    # date info
    #/----------------------------------------------------------------------------\#
    date_s = date.strftime('%Y%m%d')
    date_today = datetime.date.today()
    date_info  = '%4.4d, %2.2d, %2.2d, %4.4d, %2.2d, %2.2d' % (date.year, date.month, date.day, date_today.year, date_today.month, date_today.day)

    which_ssfr = os.path.basename(fname_ssfr_v2).split('_')[0].replace('%s-' % _mission_.upper(), '').lower()
    if which_ssfr == _ssfr1_:
        instrument_info = 'SSFR-A (Solar Spectral Flux Radiometer - Alvin)'
    else:
        instrument_info = 'SSFR-B (Solar Spectral Flux Radiometer - Belana)'
    #\----------------------------------------------------------------------------/#


    # version info
    #/----------------------------------------------------------------------------\#
    version = version.upper()
    version_info = {
            'RA': 'field data',
            }
    version_info = version_info[version]
    #\----------------------------------------------------------------------------/#


    # data info
    #/----------------------------------------------------------------------------\#
    data_info = 'Shortwave Total Downwelling and Upwelling Spectral Irradiance from %s %s' % (platform_info.upper(), instrument_info)
    #\----------------------------------------------------------------------------/#


    # routine comments
    #/----------------------------------------------------------------------------\#
    comments_routine_list = OrderedDict({
            'PI_CONTACT_INFO': 'Address: University of Colorado Boulder, LASP, 3665 Discovery Drive, Boulder, CO 80303; E-mail: hong.chen@lasp.colorado.edu and sebastian.schmidt@lasp.colorado.edu',
            'PLATFORM': platform_info.upper(),
            'LOCATION': 'N/A',
            'ASSOCIATED_DATA': 'N/A',
            'INSTRUMENT_INFO': instrument_info,
            'DATA_INFO': 'Reported are only of a selected wavelength range (%d-%d nm), pitch/roll from leveling platform INS or aircraft, time/lat/lon/alt/heading from aircraft, sza calculated from time/lon/lat.' % (wvl_range[0], wvl_range[1]),
            'UNCERTAINTY': 'Nominal SSFR uncertainty (shortwave): nadir: N/A; zenith: N/A',
            'ULOD_FLAG': '-7777',
            'ULOD_VALUE': 'N/A',
            'LLOD_FLAG': '-8888',
            'LLOD_VALUE': 'N/A',
            'DM_CONTACT_INFO': 'N/A',
            'PROJECT_INFO': 'ARCSIX field experiment out of Pituffik, Greenland, May - August 2024',
            'STIPULATIONS_ON_USE': 'This is initial in-field release of the ARCSIX-2024 data set. Please consult the PI, both for updates to the data set, and for the proper and most recent interpretation of the data for specific science use.',
            'OTHER_COMMENTS': 'Minimal corrections were applied.\n',
            'REVISION': version,
            version: version_info
            })

    comments_routine = '\n'.join(['%s: %s' % (var0, comments_routine_list[var0]) for var0 in comments_routine_list.keys()])
    #\----------------------------------------------------------------------------/#


    # special comments
    #/----------------------------------------------------------------------------\#
    comments_special_dict = {
            '20240530': 'Noticed icing on zenith light collector dome after flight',
            '20240531': 'Encountered temperature control issue (after around 1:30 UTC)',
            }
    if date_s in comments_special_dict.keys():
        comments_special = comments_special_dict[date_s]
    else:
        comments_special = ''

    if comments_special != '':
        Nspecial = len(comments_special.split('\n'))
    else:
        Nspecial = 0
    #\----------------------------------------------------------------------------/#


    # data processing
    #/----------------------------------------------------------------------------\#
    data_v2 = ssfr.util.load_h5(fname_ssfr_v2)
    data_v2['zen/flux'][data_v2['zen/flux']<0.0] = np.nan
    data_v2['nad/flux'][data_v2['nad/flux']<0.0] = np.nan

    logic_zen = (data_v2['zen/wvl']>=wvl_range[0]) & (data_v2['zen/wvl']<=wvl_range[1])
    logic_nad = (data_v2['nad/wvl']>=wvl_range[0]) & (data_v2['nad/wvl']<=wvl_range[1])

    data = OrderedDict({
            'Time_Start': {
                'data': data_v2['tmhr']*3600.0,
                'unit': 'second',
                'description': 'UTC time in seconds from the midnight 00:00:00',
                },

            'jday': {
                'data': data_v2['jday'],
                'unit': 'day',
                'description': 'UTC time in decimal day from 0001-01-01 00:00:00',
                },

            'tmhr': {
                'data': data_v2['tmhr'],
                'unit': 'hour',
                'description': 'UTC time in decimal hour from the midnight 00:00:00',
                },

            'lon': {
                'data': data_v2['lon'],
                'unit': 'degree',
                'description': 'longitude',
                },

            'lat': {
                'data': data_v2['lat'],
                'unit': 'degree',
                'description': 'latitude',
                },

            'alt': {
                'data': data_v2['alt'],
                'unit': 'meter',
                'description': 'altitude',
                },

            'sza': {
                'data': data_v2['att_corr/sza'],
                'unit': 'degree',
                'description': 'solar zenith angle',
                },

            'zen/flux': {
                'data': data_v2['zen/flux'][:, logic_zen],
                'unit': 'W m^-2 nm^-1',
                'description': 'total downwelling spectral irradiance (zenith)',
                },

            'zen/toa0': {
                'data': data_v2['zen/toa0'][logic_zen],
                'unit': 'W m^-2 nm^-1',
                'description': 'Kurucz reference total downwelling spectral irradiance (zenith)',
                },

            'zen/wvl': {
                'data': data_v2['zen/wvl'][logic_zen],
                'unit': 'nm',
                'description': 'wavelength for total downwelling spectral irradiance (zenith)',
                },

            'nad/flux': {
                'data': data_v2['nad/flux'][:, logic_nad],
                'unit': 'W m^-2 nm^-1',
                'description': 'total upwelling spectral irradiance (nadir)',
                },

            'nad/wvl': {
                'data': data_v2['nad/wvl'][logic_nad],
                'unit': 'nm',
                'description': 'wavelength for total upwelling spectral irradiance (nadir)',
                },
            })
    for key in data.keys():
        data[key]['description'] = '%s: %s, %s' % (key, data[key]['unit'], data[key]['description'])

    Nvar = len(data.keys())
    comments_routine = '%s\n%s' % (comments_routine, ','.join(data.keys()))
    Nroutine = len(comments_routine.split('\n'))
    #\----------------------------------------------------------------------------/#


    header_list = [file_format_index,
                   principal_investigator_info,
                   affiliation_info,       # Organization/affiliation of PI.
                   data_info,              # Data source description (e.g., instrument name, platform name, model name, etc.).
                   mission_info,           # Mission name (usually the mission acronym).
                   file_volume_number,     # File volume number, number of file volumes (these integer values are used when the data require more than one file per day; for data that require only one file these values are set to 1, 1) - comma delimited.
                   date_info,              # UTC date when data begin, UTC date of data reduction or revision - comma delimited (yyyy, mm, dd, yyyy, mm, dd).
                   data_interval,          # Data Interval (This value describes the time spacing (in seconds) between consecutive data records. It is the (constant) interval between values of the independent variable. For 1 Hz data the data interval value is 1 and for 10 Hz data the value is 0.1. All intervals longer than 1 second must be reported as Start and Stop times, and the Data Interval value is set to 0. The Mid-point time is required when it is not at the average of Start and Stop times. For additional information see Section 2.5 below.).
                   data['Time_Start']['description'],                # Description or name of independent variable (This is the name chosen for the start time. It always refers to the number of seconds UTC from the start of the day on which measurements began. It should be noted here that the independent variable should monotonically increase even when crossing over to a second day.).
                   str(Nvar-1),                                      # Number of variables (Integer value showing the number of dependent variables: the total number of columns of data is this value plus one.).
                   ', '.join([scale_factor for i in range(Nvar-1)]), # Scale factors (1 for most cases, except where grossly inconvenient) - comma delimited.
                   ', '.join([fill_value for i in range(Nvar-1)]),   # Missing data indicators (This is -9999 (or -99999, etc.) for any missing data condition, except for the main time (independent) variable which is never missing) - comma delimited.
                   '\n'.join([data[vname]['description'] for vname in data.keys() if vname != 'Time_Start']), # Variable names and units (Short variable name and units are required, and optional long descriptive name, in that order, and separated by commas. If the variable is unitless, enter the keyword "none" for its units. Each short variable name and units (and optional long name) are entered on one line. The short variable name must correspond exactly to the name used for that variable as a column header, i.e., the last header line prior to start of data.).
                   str(Nspecial),                                   # Number of SPECIAL comment lines (Integer value indicating the number of lines of special comments, NOT including this line.).
                   comments_special,
                   str(Nroutine),
                   comments_routine,
                ]


    header = '\n'.join([header0 for header0 in header_list if header0 != ''])

    Nline = len(header.split('\n'))
    header = '%d, %s' % (Nline, header)

    print(header)

    fname_h5 = '%s/%s-SSFR_%s_%s_%s.h5' % (fdir_out, _mission_.upper(), _platform_.upper(), date_s, version.upper())
    if run:
        f = h5py.File(fname_h5, 'w')

        dset = f.create_dataset('header', data=header)
        dset.attrs['description'] = 'header follows ICT format'

        for key in data.keys():
            dset = f.create_dataset(key, data=data[key]['data'], compression='gzip', compression_opts=9, chunks=True)
            dset.attrs['description'] = data[key]['description']
            dset.attrs['unit'] = data[key]['unit']
        f.close()

    return fname_h5
#\----------------------------------------------------------------------------/#


# additional functions under development
#/----------------------------------------------------------------------------\#
def run_time_offset_check(date):

    date_s = date.strftime('%Y%m%d')
    data_hsk = ssfr.util.load_h5(_fnames_['%s_hsk_v0' % date_s])
    data_alp = ssfr.util.load_h5(_fnames_['%s_alp_v0' % date_s])
    data_spns_v0 = ssfr.util.load_h5(_fnames_['%s_spns_v0' % date_s])
    if _which_ssfr_for_flux_ == _ssfr1_:
        data_ssfr1_v0 = ssfr.util.load_h5(_fnames_['%s_ssfr1_v0' % date_s])
        data_ssfr2_v0 = ssfr.util.load_h5(_fnames_['%s_ssfr2_v0' % date_s])
    else:
        data_ssfr1_v0 = ssfr.util.load_h5(_fnames_['%s_ssfr2_v0' % date_s])
        data_ssfr2_v0 = ssfr.util.load_h5(_fnames_['%s_ssfr1_v0' % date_s])

    # data_spns_v0['tot/jday'] += 1.0
    # data_spns_v0['dif/jday'] += 1.0

    # _offset_x_range_ = [-6000.0, 6000.0]
    _offset_x_range_ = [-600.0, 600.0]

    # ALP pitch vs HSK pitch
    #/----------------------------------------------------------------------------\#
    data_offset = {
            'x0': data_hsk['jday']*86400.0,
            'y0': data_hsk['ang_pit'],
            'x1': data_alp['jday'][::10]*86400.0,
            'y1': data_alp['ang_pit_s'][::10],
            }
    ssfr.vis.find_offset_bokeh(
            data_offset,
            offset_x_range=_offset_x_range_,
            offset_y_range=[-10, 10],
            x_reset=True,
            y_reset=False,
            description='ALP Pitch vs. HSK Pitch',
            fname_html='alp-pit_offset_check_%s.html' % date_s)
    #\----------------------------------------------------------------------------/#

    # ALP roll vs HSK roll
    #/----------------------------------------------------------------------------\#
    data_offset = {
            'x0': data_hsk['jday']*86400.0,
            'y0': data_hsk['ang_rol'],
            'x1': data_alp['jday'][::10]*86400.0,
            'y1': data_alp['ang_rol_s'][::10],
            }
    ssfr.vis.find_offset_bokeh(
            data_offset,
            offset_x_range=_offset_x_range_,
            offset_y_range=[-10, 10],
            x_reset=True,
            y_reset=False,
            description='ALP Roll vs. HSK Roll',
            fname_html='alp-rol_offset_check_%s.html' % date_s)
    #\----------------------------------------------------------------------------/#

    # ALP altitude vs HSK altitude
    #/----------------------------------------------------------------------------\#
    data_offset = {
            'x0': data_hsk['jday']*86400.0,
            'y0': data_hsk['alt'],
            'x1': data_alp['jday'][::10]*86400.0,
            'y1': data_alp['alt'][::10],
            }
    ssfr.vis.find_offset_bokeh(
            data_offset,
            offset_x_range=_offset_x_range_,
            offset_y_range=[-10, 10],
            x_reset=True,
            y_reset=True,
            description='ALP Altitude vs. HSK Altitude',
            fname_html='alp-alt_offset_check_%s.html' % date_s)
    #\----------------------------------------------------------------------------/#


    # SPNS vs TOA
    #/----------------------------------------------------------------------------\#
    index_wvl = np.argmin(np.abs(745.0-data_spns_v0['tot/wvl']))
    data_y1   = data_spns_v0['tot/flux'][:, index_wvl]

    mu = np.cos(np.deg2rad(data_hsk['sza']))
    iza, iaa = ssfr.util.prh2za(data_hsk['ang_pit'], data_hsk['ang_rol'], data_hsk['ang_hed'])
    dc = ssfr.util.muslope(data_hsk['sza'], data_hsk['saa'], iza, iaa)
    factors = mu/dc
    data_y0   = data_spns_v0['tot/toa0'][index_wvl]*np.cos(np.deg2rad(data_hsk['sza']))/factors

    data_offset = {
            'x0': data_hsk['jday']*86400.0,
            'y0': data_y0,
            'x1': data_spns_v0['tot/jday']*86400.0,
            'y1': data_y1,
            }
    ssfr.vis.find_offset_bokeh(
            data_offset,
            offset_x_range=_offset_x_range_,
            offset_y_range=[-10, 10],
            x_reset=True,
            y_reset=True,
            description='SPNS Total vs. TOA (745 nm)',
            fname_html='spns-toa_offset_check_%s.html' % date_s)
    #\----------------------------------------------------------------------------/#


    # SSFR-A vs SPNS
    #/----------------------------------------------------------------------------\#
    index_wvl_spns = np.argmin(np.abs(745.0-data_spns_v0['tot/wvl']))
    data_y0 = data_spns_v0['tot/flux'][:, index_wvl_spns]

    index_wvl_ssfr = np.argmin(np.abs(745.0-data_ssfr1_v0['spec/wvl_zen']))
    data_y1 = data_ssfr1_v0['spec/cnt_zen'][:, index_wvl_ssfr]
    data_offset = {
            'x0': data_spns_v0['tot/jday']*86400.0,
            'y0': data_y0,
            'x1': data_ssfr1_v0['raw/jday']*86400.0,
            'y1': data_y1,
            }
    ssfr.vis.find_offset_bokeh(
            data_offset,
            offset_x_range=_offset_x_range_,
            offset_y_range=[-10, 10],
            x_reset=True,
            y_reset=True,
            description='SSFR-A Zenith Count vs. SPNS Total (745nm)',
            fname_html='ssfr-a_offset_check_%s.html' % (date_s))
    #\----------------------------------------------------------------------------/#

    # SSFR-B vs SSFR-A
    #/----------------------------------------------------------------------------\#
    index_wvl_ssfr = np.argmin(np.abs(745.0-data_ssfr1_v0['spec/wvl_nad']))
    data_y0 = data_ssfr1_v0['spec/cnt_nad'][:, index_wvl_ssfr]

    index_wvl_ssfr = np.argmin(np.abs(745.0-data_ssfr2_v0['spec/wvl_nad']))
    data_y1 = data_ssfr2_v0['spec/cnt_nad'][:, index_wvl_ssfr]
    data_offset = {
            'x0': data_ssfr1_v0['raw/jday']*86400.0,
            'y0': data_y0,
            'x1': data_ssfr2_v0['raw/jday']*86400.0,
            'y1': data_y1,
            }
    ssfr.vis.find_offset_bokeh(
            data_offset,
            offset_x_range=_offset_x_range_,
            offset_y_range=[-10, 10],
            x_reset=True,
            y_reset=True,
            description='SSFR-B Nadir Count vs. SSFR-A Nadir (745nm)',
            fname_html='ssfr-b_offset_check_%s.html' % (date_s))
    #\----------------------------------------------------------------------------/#

    sys.exit()

def run_angle_offset_check(
        date,
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        wvl0=745.0,
        ):

    date_s = date.strftime('%Y%m%d')
    data_hsk = ssfr.util.load_h5(_fnames_['%s_hsk_v0' % date_s])


    # SPNS v1
    #/----------------------------------------------------------------------------\#
    data_spns_v1 = ssfr.util.load_h5(_fnames_['%s_spns_v1' % date_s])
    index_wvl_spns = np.argmin(np.abs(wvl0-data_spns_v1['tot/wvl']))
    data_y1 = data_spns_v1['tot/flux'][:, index_wvl_spns]
    #\----------------------------------------------------------------------------/#


    # SPNS v2
    #/----------------------------------------------------------------------------\#
    fname_spns_v2 = cdata_arcsix_spns_v2(date, _fnames_['%s_spns_v1' % date_s], _fnames_['%s_hsk_v0' % date_s],
            fdir_out=_fdir_out_,
            run=True,
            ang_pit_offset=ang_pit_offset,
            ang_rol_offset=ang_rol_offset,
            )
    data_spns_v2 = ssfr.util.load_h5(_fnames_['%s_spns_v2' % date_s])
    data_y2 = data_spns_v2['tot/flux'][:, index_wvl_spns]
    #\----------------------------------------------------------------------------/#


    # SSFR-A v2
    #/----------------------------------------------------------------------------\#
    data_ssfr1_v2 = ssfr.util.load_h5(_fnames_['%s_ssfr1_v2' % date_s])
    index_wvl_ssfr = np.argmin(np.abs(wvl0-data_ssfr1_v2['zen/wvl']))
    data_y0 = data_ssfr1_v2['zen/flux'][:, index_wvl_ssfr]
    #\----------------------------------------------------------------------------/#

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(15, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        # cs = ax1.imshow(.T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        ax1.scatter(data_hsk['tmhr'], data_y1, s=3, c='r', lw=0.0)
        ax1.scatter(data_hsk['tmhr'], data_y0, s=3, c='k', lw=0.0)
        ax1.scatter(data_hsk['tmhr'], data_y2, s=3, c='g', lw=0.0)
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
    #\----------------------------------------------------------------------------/#
    sys.exit()

def dark_corr_temp(date, iChan=100, idset=0):

    date_s = date.strftime('%Y%m%d')
    data_ssfr1_v0 = ssfr.util.load_h5(_fnames_['%s_ssfr1_v0' % date_s])

    tmhr = data_ssfr1_v0['raw/tmhr']
    x_temp_zen = data_ssfr1_v0['raw/temp'][:, 1]
    x_temp_nad = data_ssfr1_v0['raw/temp'][:, 2]
    shutter = data_ssfr1_v0['raw/shutter_dark-corr']
    dset_num = data_ssfr1_v0['raw/dset_num']

    logic_dark = (shutter==1) & (dset_num==idset)
    logic_light = (shutter==0) & (dset_num==idset)

    # figure
    #/----------------------------------------------------------------------------\#
    if True:


        plt.close('all')
        fig = plt.figure(figsize=(13, 19))
        fig.suptitle('Channel #%d' % iChan)
        # plot
        #/--------------------------------------------------------------\#
        ax0 = fig.add_subplot(12,1,1)
        ax00 = fig.add_subplot(12,1,2)
        ax000 = fig.add_subplot(12,1,3)
        ax0000 = fig.add_subplot(12,1,4)

        ax1 = fig.add_subplot(323)
        logic_fit = (x_temp_zen>25.0) & logic_dark
        logic_x   = (x_temp_zen>25.0) & logic_light
        coef = np.polyfit(x_temp_zen[logic_fit], data_ssfr1_v0['raw/count_raw'][logic_fit, iChan, 0], 5)
        xx = np.linspace(x_temp_zen[logic_x].min(), x_temp_zen[logic_x].max(), 1000)
        yy = np.polyval(coef, xx)

        ax1.scatter(x_temp_zen[logic_x], data_ssfr1_v0['raw/count_raw'][logic_x, iChan, 0]-data_ssfr1_v0['raw/count_dark-corr'][logic_x, iChan, 0], c=tmhr[logic_x], s=6, cmap='jet', alpha=0.2, zorder=0)
        ax1.plot(xx, yy, color='gray', zorder=1)
        ax1.scatter(x_temp_zen[logic_dark], data_ssfr1_v0['raw/count_raw'][logic_dark, iChan, 0], color='k', s=10, alpha=0.2, zorder=2)
        ax1.set_title('Zenith Silicon (%.2f nm)' % data_ssfr1_v0['raw/wvl_zen_si'][iChan])
        ax1.set_xlabel('Zenith InGaAs Temperature')
        ax1.set_ylabel('Counts')

        ax0.scatter(tmhr[logic_x], data_ssfr1_v0['raw/count_raw'][logic_x, iChan, 0]-data_ssfr1_v0['raw/count_dark-corr'][logic_x, iChan, 0], c=tmhr[logic_x], s=6, cmap='jet', alpha=0.2, zorder=0)

        logic_fit = (x_temp_zen>25.0) & logic_dark
        logic_x   = (x_temp_zen>25.0) & logic_light
        coef = np.polyfit(x_temp_zen[logic_fit], data_ssfr1_v0['raw/count_raw'][logic_fit, iChan, 1], 5)
        xx = np.linspace(x_temp_zen[logic_x].min(), x_temp_zen[logic_x].max(), 1000)
        yy = np.polyval(coef, xx)

        ax2 = fig.add_subplot(324)
        ax2.scatter(x_temp_zen[logic_x], data_ssfr1_v0['raw/count_raw'][logic_x, iChan, 1]-data_ssfr1_v0['raw/count_dark-corr'][logic_x, iChan, 1], c=tmhr[logic_x], s=6, cmap='jet', alpha=0.2, zorder=0)
        ax2.plot(xx, yy, color='gray', zorder=1)
        ax2.scatter(x_temp_zen[logic_dark], data_ssfr1_v0['raw/count_raw'][logic_dark, iChan, 1], color='k', s=10, alpha=0.2, zorder=2)
        ax2.set_title('Zenith InGaAs (%.2f nm)' % data_ssfr1_v0['raw/wvl_zen_in'][iChan])
        ax2.set_xlabel('Zenith InGaAs Temperature')
        ax2.set_ylabel('Counts')

        ax00.scatter(tmhr[logic_x], data_ssfr1_v0['raw/count_raw'][logic_x, iChan, 1]-data_ssfr1_v0['raw/count_dark-corr'][logic_x, iChan, 1], c=tmhr[logic_x], s=6, cmap='jet', alpha=0.2, zorder=0)

        logic_fit = (x_temp_nad>25.0) & logic_dark
        logic_x   = (x_temp_nad>25.0) & logic_light
        coef = np.polyfit(x_temp_nad[logic_fit], data_ssfr1_v0['raw/count_raw'][logic_fit, iChan, 2], 5)
        xx = np.linspace(x_temp_nad[logic_x].min(), x_temp_nad[logic_x].max(), 1000)
        yy = np.polyval(coef, xx)

        ax3 = fig.add_subplot(325)
        ax3.scatter(x_temp_nad[logic_x], data_ssfr1_v0['raw/count_raw'][logic_x, iChan, 2]-data_ssfr1_v0['raw/count_dark-corr'][logic_x, iChan, 2], c=tmhr[logic_x], s=6, cmap='jet', alpha=0.2, zorder=0)
        ax3.plot(xx, yy, color='gray', zorder=1)
        ax3.scatter(x_temp_nad[logic_dark], data_ssfr1_v0['raw/count_raw'][logic_dark, iChan, 2], color='k', s=10, alpha=0.2, zorder=2)
        ax3.set_title('Nadir Silicon (%.2f nm)' % data_ssfr1_v0['raw/wvl_nad_si'][iChan])
        ax3.set_xlabel('Nadir InGaAs Temperature')
        ax3.set_ylabel('Counts')

        ax000.scatter(tmhr[logic_x], data_ssfr1_v0['raw/count_raw'][logic_x, iChan, 2]-data_ssfr1_v0['raw/count_dark-corr'][logic_x, iChan, 2], c=tmhr[logic_x], s=6, cmap='jet', alpha=0.2, zorder=0)

        logic_fit = (x_temp_nad>25.0) & logic_dark
        logic_x   = (x_temp_nad>25.0) & logic_light
        coef = np.polyfit(x_temp_nad[logic_fit], data_ssfr1_v0['raw/count_raw'][logic_fit, iChan, 3], 5)
        xx = np.linspace(x_temp_nad[logic_x].min(), x_temp_nad[logic_x].max(), 1000)
        yy = np.polyval(coef, xx)

        ax4 = fig.add_subplot(326)
        ax4.scatter(x_temp_nad[logic_x], data_ssfr1_v0['raw/count_raw'][logic_x, iChan, 3]-data_ssfr1_v0['raw/count_dark-corr'][logic_x, iChan, 3], c=tmhr[logic_x], s=6, cmap='jet', alpha=0.2, zorder=0)
        ax4.plot(xx, yy, color='gray', zorder=1)
        ax4.scatter(x_temp_nad[logic_dark], data_ssfr1_v0['raw/count_raw'][logic_dark, iChan, 3], color='k', s=10, alpha=0.2, zorder=2)
        ax4.set_title('Nadir InGaAs (%.2f nm)' % data_ssfr1_v0['raw/wvl_nad_in'][iChan])
        ax4.set_xlabel('Nadir InGaAs Temperature')
        ax4.set_ylabel('Counts')

        ax0000.scatter(tmhr[logic_x], data_ssfr1_v0['raw/count_raw'][logic_x, iChan, 3]-data_ssfr1_v0['raw/count_dark-corr'][logic_x, iChan, 3], c=tmhr[logic_x], s=6, cmap='jet', alpha=0.2, zorder=0)
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.4)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_dset%d_%3.3d.png' % (_metadata['Function'], idset, iChan), bbox_inches='tight', metadata=_metadata, dpi=150)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
    #\----------------------------------------------------------------------------/#
#\----------------------------------------------------------------------------/#


# main program
#/----------------------------------------------------------------------------\#
def main_process_data_instrument(date, run=True):

    date_s = date.strftime('%Y%m%d')

    # 1&2. aircraft housekeeping file (need to request data from the P-3 data system)
    #      active leveling platform
    #    - longitude
    #    - latitude
    #    - altitude
    #    - UTC time
    #    - pitch angle
    #    - roll angle
    #    - heading angle
    #    - motor pitch angle
    #    - motor roll angle
    process_alp_data(date, run=False)

    # 3. SPNS - irradiance (400nm - 900nm)
    #    - spectral downwelling diffuse
    #    - spectral downwelling global/direct (direct=global-diffuse)
    process_spns_data(date, run=False)

    # 4. SSFR-A - irradiance (350nm - 2200nm)
    #    - spectral downwelling global
    #    - spectral upwelling global
    # process_ssfr_data(date, which_ssfr='ssfr-a', run=True)

    # 5. SSFR-B - radiance (350nm - 2200nm)
    #    - spectral downwelling global
    #    - spectral upwelling global
    process_ssfr_data(date, which_ssfr='ssfr-b', run=True)

def main_process_data_v0(date, run=True):

    fdir_out = _fdir_out_
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    date_s = date.strftime('%Y%m%d')

    # # ALP v0: raw data
    # #/----------------------------------------------------------------------------\#
    # fdirs = ssfr.util.get_all_folders(_fdir_data_, pattern='*%4.4d*%2.2d*%2.2d*raw?%s' % (date.year, date.month, date.day, _alp_))
    # fdir_data_alp = sorted(fdirs, key=os.path.getmtime)[-1]
    # fnames_alp = ssfr.util.get_all_files(fdir_data_alp, pattern='*.plt3')
    # if run and len(fnames_alp) == 0:
    #     pass
    # else:
    #     fname_alp_v0 = cdata_arcsix_alp_v0(date, fdir_data=fdir_data_alp,
    #             fdir_out=fdir_out, run=run)
    #     _fnames_['%s_alp_v0' % date_s]   = fname_alp_v0
    # #\----------------------------------------------------------------------------/#

    # # HSK v0: raw data
    # #/----------------------------------------------------------------------------\#
    # fnames_hsk = ssfr.util.get_all_files(_fdir_hsk_, pattern='*%4.4d*%2.2d*%2.2d*.???' % (date.year, date.month, date.day))
    # if run and len(fnames_hsk) == 0:
    #     # * not preferred, use ALP lon/lat if P3 housekeeping file is not available (e.g., for immediate data processing)
    #     fname_hsk_v0 = cdata_arcsix_hsk_from_alp_v0(date, _fnames_['%s_alp_v0' % date_s], fdir_data=_fdir_hsk_,
    #             fdir_out=fdir_out, run=run)
    # else:
    #     # * preferred, use P3 housekeeping file, ict > iwg > mts
    #     fname_hsk_v0 = cdata_arcsix_hsk_v0(date, fdir_data=_fdir_hsk_,
    #             fdir_out=fdir_out, run=run)
    # _fnames_['%s_hsk_v0' % date_s] = fname_hsk_v0
    # #\----------------------------------------------------------------------------/#
    # sys.exit()


    # SSFR-A v0: raw data
    #/----------------------------------------------------------------------------\#
    fdirs = ssfr.util.get_all_folders(_fdir_data_, pattern='*%4.4d*%2.2d*%2.2d*raw?%s' % (date.year, date.month, date.day, _ssfr1_))
    fdir_data_ssfr1 = sorted(fdirs, key=os.path.getmtime)[-1]
    fnames_ssfr1 = ssfr.util.get_all_files(fdir_data_ssfr1, pattern='*.SKS')
    if run and len(fnames_ssfr1) == 0:
        pass
    else:
        fname_ssfr1_v0 = cdata_arcsix_ssfr_v0(date, fdir_data=fdir_data_ssfr1,
                which_ssfr='ssfr-a', fdir_out=fdir_out, run=run)
        _fnames_['%s_ssfr1_v0' % date_s] = fname_ssfr1_v0
    #\----------------------------------------------------------------------------/#


    # SSFR-B v0: raw data
    #/----------------------------------------------------------------------------\#
    fdirs = ssfr.util.get_all_folders(_fdir_data_, pattern='*%4.4d*%2.2d*%2.2d*raw?%s' % (date.year, date.month, date.day, _ssfr2_))
    fdir_data_ssfr2 = sorted(fdirs, key=os.path.getmtime)[-1]
    fnames_ssfr2 = ssfr.util.get_all_files(fdir_data_ssfr2, pattern='*.SKS')
    if run and len(fnames_ssfr2) == 0:
        pass
    else:
        fname_ssfr2_v0 = cdata_arcsix_ssfr_v0(date, fdir_data=fdir_data_ssfr2,
                which_ssfr='ssfr-b', fdir_out=fdir_out, run=run)
        _fnames_['%s_ssfr2_v0' % date_s] = fname_ssfr2_v0
    #\----------------------------------------------------------------------------/#


    # ALP v0: raw data
    #/----------------------------------------------------------------------------\#
    fdirs = ssfr.util.get_all_folders(_fdir_data_, pattern='*%4.4d*%2.2d*%2.2d*raw?%s' % (date.year, date.month, date.day, _alp_))
    fdir_data_alp = sorted(fdirs, key=os.path.getmtime)[-1]
    fnames_alp = ssfr.util.get_all_files(fdir_data_alp, pattern='*.plt3')
    if run and len(fnames_alp) == 0:
        pass
    else:
        fname_alp_v0 = cdata_arcsix_alp_v0(date, fdir_data=fdir_data_alp,
                fdir_out=fdir_out, run=run)
        _fnames_['%s_alp_v0' % date_s]   = fname_alp_v0
    #\----------------------------------------------------------------------------/#


    # SPNS v0: raw data
    #/----------------------------------------------------------------------------\#
    fdirs = ssfr.util.get_all_folders(_fdir_data_, pattern='*%4.4d*%2.2d*%2.2d*raw?%s*' % (date.year, date.month, date.day, 'spns'))
    fdir_data_spns = sorted(fdirs, key=os.path.getmtime)[-1]
    fnames_spns = ssfr.util.get_all_files(fdir_data_spns, pattern='*.txt')
    if run and len(fnames_spns) == 0:
        pass
    else:
        fname_spns_v0 = cdata_arcsix_spns_v0(date, fdir_data=fdir_data_spns,
                fdir_out=fdir_out, run=run)
        _fnames_['%s_spns_v0' % date_s]  = fname_spns_v0
    #\----------------------------------------------------------------------------/#


    # HSK v0: raw data
    #/----------------------------------------------------------------------------\#
    fnames_hsk = ssfr.util.get_all_files(_fdir_hsk_, pattern='*%4.4d*%2.2d*%2.2d*.???' % (date.year, date.month, date.day))
    if run and len(fnames_hsk) == 0:
        # * not preferred, use ALP lon/lat if P3 housekeeping file is not available (e.g., for immediate data processing)
        fname_hsk_v0 = cdata_arcsix_hsk_from_alp_v0(date, _fnames_['%s_alp_v0' % date_s], fdir_data=_fdir_hsk_,
                fdir_out=fdir_out, run=run)
    else:
        # * preferred, use P3 housekeeping file, ict > iwg > mts
        fname_hsk_v0 = cdata_arcsix_hsk_v0(date, fdir_data=_fdir_hsk_,
                fdir_out=fdir_out, run=run)
    _fnames_['%s_hsk_v0' % date_s] = fname_hsk_v0
    #\----------------------------------------------------------------------------/#

def main_process_data_v1(date, run=True):

    fdir_out = _fdir_out_
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    date_s = date.strftime('%Y%m%d')

    # ALP v1: time synced with hsk time with time offset applied
    #/----------------------------------------------------------------------------\#
    fname_alp_v1 = cdata_arcsix_alp_v1(date, _fnames_['%s_alp_v0' % date_s], _fnames_['%s_hsk_v0' % date_s],
            fdir_out=fdir_out, run=run)

    _fnames_['%s_alp_v1'   % date_s] = fname_alp_v1
    #\----------------------------------------------------------------------------/#

    # SPNS v1: time synced with hsk time with time offset applied
    #/----------------------------------------------------------------------------\#
    fname_spns_v1 = cdata_arcsix_spns_v1(date, _fnames_['%s_spns_v0' % date_s], _fnames_['%s_hsk_v0' % date_s],
            fdir_out=fdir_out, run=run)

    _fnames_['%s_spns_v1'  % date_s] = fname_spns_v1
    #\----------------------------------------------------------------------------/#

    # SSFR-A v1: time synced with hsk time with time offset applied
    #/----------------------------------------------------------------------------\#
    fname_ssfr1_v1 = cdata_arcsix_ssfr_v1(date, _fnames_['%s_ssfr1_v0' % date_s], _fnames_['%s_hsk_v0' % date_s],
            which_ssfr_for_flux=_which_ssfr_for_flux_, fdir_out=fdir_out, run=run)

    _fnames_['%s_ssfr1_v1' % date_s] = fname_ssfr1_v1
    #\----------------------------------------------------------------------------/#

    # SSFR-B v1: time synced with hsk time with time offset applied
    #/----------------------------------------------------------------------------\#
    fname_ssfr2_v1 = cdata_arcsix_ssfr_v1(date, _fnames_['%s_ssfr2_v0' % date_s], _fnames_['%s_hsk_v0' % date_s],
            which_ssfr_for_flux=_which_ssfr_for_flux_, fdir_out=fdir_out, run=run)

    _fnames_['%s_ssfr2_v1' % date_s] = fname_ssfr2_v1
    #\----------------------------------------------------------------------------/#

def main_process_data_v2(date, run=True):

    """
    v0: raw data directly read out from the data files
    v1: data collocated/synced to aircraft nav
    v2: attitude corrected data
    """

    date_s = date.strftime('%Y%m%d')

    fdir_out = _fdir_out_
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    # SPNS v2
    #/----------------------------------------------------------------------------\#
    # * based on ALP v1
    # fname_spns_v2 = cdata_arcsix_spns_v2(date, _fnames_['%s_spns_v1' % date_s], _fnames_['%s_alp_v1' % date_s],
    #         fdir_out=fdir_out, run=run)
    # fname_spns_v2 = cdata_arcsix_spns_v2(date, _fnames_['%s_spns_v1' % date_s], _fnames_['%s_alp_v1' % date_s],
    #         fdir_out=fdir_out, run=True)

    # * based on HSK v0
    fname_spns_v2 = cdata_arcsix_spns_v2(date, _fnames_['%s_spns_v1' % date_s], _fnames_['%s_hsk_v0' % date_s],
            fdir_out=fdir_out, run=run)
    #\----------------------------------------------------------------------------/#
    _fnames_['%s_spns_v2' % date_s] = fname_spns_v2


    # SSFR v2
    #/----------------------------------------------------------------------------\#
    if _which_ssfr_for_flux_ == _ssfr1_:
        _vname_ssfr_v1_ = '%s_ssfr1_v1' % date_s
        _vname_ssfr_v2_ = '%s_ssfr1_v2' % date_s
    else:
        _vname_ssfr_v1_ = '%s_ssfr2_v1' % date_s
        _vname_ssfr_v2_ = '%s_ssfr2_v2' % date_s

    fname_ssfr_v2 = cdata_arcsix_ssfr_v2(date, _fnames_[_vname_ssfr_v1_], _fnames_['%s_alp_v1' % date_s], _fnames_['%s_spns_v2' % date_s],
            fdir_out=fdir_out, run=run, run_aux=True)
    #\----------------------------------------------------------------------------/#
    _fnames_[_vname_ssfr_v2_] = fname_ssfr_v2

def main_process_data_archive(date, run=True):

    """
    ra: in-field data to be uploaded to https://www-air.larc.nasa.gov/cgi-bin/ArcView/arcsix
    """

    date_s = date.strftime('%Y%m%d')

    fdir_out = _fdir_out_
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    # SPNS RA
    #/----------------------------------------------------------------------------\#
    fname_spns_ra = cdata_arcsix_spns_archive(date, _fnames_['%s_spns_v2' % date_s],
            fdir_out=fdir_out, run=run)
    #\----------------------------------------------------------------------------/#
    _fnames_['%s_spns_ra' % date_s] = fname_spns_ra


    # SSFR RA
    #/----------------------------------------------------------------------------\#
    if _which_ssfr_for_flux_ == _ssfr1_:
        _vname_ssfr_v2_ = '%s_ssfr1_v2' % date_s
        _vname_ssfr_ra_ = '%s_ssfr1_ra' % date_s
    else:
        _vname_ssfr_v2_ = '%s_ssfr2_v2' % date_s
        _vname_ssfr_ra_ = '%s_ssfr2_ra' % date_s

    fname_ssfr_ra = cdata_arcsix_ssfr_archive(date, _fnames_[_vname_ssfr_v2_],
            fdir_out=fdir_out, run=run)
    #\----------------------------------------------------------------------------/#
    _fnames_[_vname_ssfr_ra_] = fname_ssfr_ra
#\----------------------------------------------------------------------------/#


if __name__ == '__main__':

    warnings.warn('\n!!!!!!!! Under development !!!!!!!!')

    # process field calibration
    #/----------------------------------------------------------------------------\#
    # main_calibration_rad()
    # sys.exit()
    #\----------------------------------------------------------------------------/#

    # process field data
    #/----------------------------------------------------------------------------\#
    dates = [
             # datetime.datetime(2024, 7, 8), # ARCSIX-2 pre-mission test data after SARP, collected inside NASA WFF hangar
             # datetime.datetime(2024, 7, 9), # ARCSIX-2 pre-mission test data after SARP, collected inside NASA WFF hangar
             # datetime.datetime(2024, 7, 22), # ARCSIX-2 transit from WFF to Pituffik, noticed TEC2 (SSFR-A nadir) issue, operator - Ken Hirata, Vikas Nataraja
             # datetime.datetime(2024, 7, 24), # ARCSIX-2 science flight #11, cancelled due to weather condition, data from ground, operator - Arabella Chamberlain, Ken Hirata
             # datetime.datetime(2024, 7, 25), # ARCSIX-2 science flight #11, cloud walls, operator - Arabella Chamberlain
             # datetime.datetime(2024, 7, 26), # ARCSIX-2 science flight #12, cancelled due to weather condition, data from ground, operator - Ken Hirata, Vikas Nataraja
             # datetime.datetime(2024, 7, 29), # ARCSIX-2 science flight #12, clear-sky BRDF, operator - Ken Hirata, Vikas Nataraja
             datetime.datetime(2024, 7, 30), # ARCSIX-2 science flight #13, clear-sky BRDF, operator - Ken Hirata
            ]

    for date in dates[::-1]:

        # step 1
        #/--------------------------------------------------------------\#
        main_process_data_v0(date, run=True)
        sys.exit()
        #\--------------------------------------------------------------/#

        # step 2
        #/--------------------------------------------------------------\#
        # main_process_data_v0(date, run=False)
        # run_time_offset_check(date)
        # sys.exit()
        #\--------------------------------------------------------------/#

        # step 3
        #/--------------------------------------------------------------\#
        # main_process_data_v0(date, run=False)
        # main_process_data_v1(date, run=True)
        # sys.exit()
        #\--------------------------------------------------------------/#

        # step 4
        #/--------------------------------------------------------------\#
        # main_process_data_v0(date, run=False)
        # main_process_data_v1(date, run=False)
        # main_process_data_v2(date, run=True)
        # sys.exit()
        #\--------------------------------------------------------------/#

        # step 5
        #/--------------------------------------------------------------\#
        # main_process_data_v0(date, run=False)
        # main_process_data_v1(date, run=False)
        # main_process_data_v2(date, run=False)
        # main_process_data_archive(date, run=True)
        # sys.exit()
        #\--------------------------------------------------------------/#

        pass
    #\----------------------------------------------------------------------------/#
