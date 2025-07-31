"""
Code for processing data collected by SSFR Team during NASA ARCSIX 2024.

SSFR-A: Solar Spectral Flux Radiometer - Alvin
SSFR-B: Solar Spectral Flux Radiometer - Belana
HSR1-A: Hyper-Spectral Radiometer 1 - Alvin
HSR1-B: Hyper-Spectral Radiometer 1 - Belana
ALP: Active Leveling Platform

Acknowledgements:
    Instrument engineering:
        Jeffery Drouet, Sebastian Schmidt
    Pre-mission and post-mission calibration and data analysis:
        Hong Chen, Yu-Wen Chen, Ken Hirata, Vikas Nataraja, Sebastian Schmidt, Bruce Kindel
    In-field calibration and on-flight operation:
        Vikas Nataraja, Arabella Chamberlain, Ken Hirata, Sebastian Becker, Jeffery Drouet, Sebastian Schmidt
"""

import os
import sys
import glob
import datetime
import warnings
import importlib
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
from matplotlib.gridspec import GridSpec
# import cartopy.crs as ccrs
# mpl.use('Agg')



import ssfr


# parameters
#╭────────────────────────────────────────────────────────────────────────────╮#
_MISSION_     = 'arcsix'
_PLATFORM_    = 'p3b'

_HSK_         = 'hsk'
_ALP_         = 'alp'
_SSFR1_       = 'ssfr-a'
_SSFR2_       = 'ssfr-b'
_CAM_         = 'nac'
_SSRR1_       = 'ssrr-a'
_SSRR2_       = 'ssrr-b'

_SPNS_        = 'spns-a'
_WHICH_SSFR_ = 'ssfr-a'
# _SPNS_        = 'spns-b'
# _WHICH_SSFR_ = 'ssfr-b'

# _FDIR_HSK_   = 'data/arcsix/2024/p3/aux/hsk'
# _FDIR_CAL_   = 'data/%s/cal' % _MISSION_
_FDIR_HSK_   = '/Volumes/argus/field/arcsix/2024/p3/aux/hsk'
_FDIR_CAL_   = '/Volumes/argus/field/%s/cal' % _MISSION_

# _FDIR_DATA_  = 'data/%s' % _MISSION_
# _FDIR_OUT_   = '%s/processed' % _FDIR_DATA_
_FDIR_DATA_  = '/Volumes/argus/field/%s' % _MISSION_
_FDIR_OUT_   = './%s/processed' % _MISSION_

_VERBOSE_   = True
_FNAMES_ = {}
#╰────────────────────────────────────────────────────────────────────────────╯#


# functions for ssfr calibrations
#╭────────────────────────────────────────────────────────────────────────────╮#
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
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('%s %s (illuminated by %s Lamp)' % (ssfr_tag.upper(), lc_tag.title(), lamp_tag.upper()))
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
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
        #╰──────────────────────────────────────────────────────────────╯#

        patches_legend = [
                          mpatches.Patch(color='red' , label='IntTime set 1'), \
                          mpatches.Patch(color='blue', label='IntTime set 2'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_%s_%s_%s.png' % (_metadata['Function'], ssfr_tag.lower(), lc_tag.lower(), lamp_tag.lower()), bbox_inches='tight', metadata=_metadata)
        #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#

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
    #╭────────────────────────────────────────────────────────────────────────────╮#
    angles_pos = np.concatenate((np.arange(0.0, 30.0, 3.0), np.arange(30.0, 50.0, 5.0), np.arange(50.0, 91.0, 10.0)))
    angles_neg = -angles_pos
    angles = np.concatenate((angles_pos, angles_neg, np.array([0.0])))
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # make fnames, a dictionary <key:value> with file name as key, angle as value
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fnames_ = sorted(glob.glob('%s/*.SKS' % fdir))
    fnames  = {
            fnames_[i]: angles[i] for i in range(angles.size)
            }
    #╰────────────────────────────────────────────────────────────────────────────╯#

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
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # for ssfr_tag in ['SSFR-A', 'SSFR-B']:
    #     for lc_tag in ['zen', 'nad']:
    #         for lamp_tag in ['kr', 'hg']:
    #             wvl_cal(ssfr_tag, lc_tag, lamp_tag)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # radiometric calibration
    #╭────────────────────────────────────────────────────────────────────────────╮#
    for ssfr_tag in ['SSFR-A', 'SSFR-B']:
        for lc_tag in ['zen', 'nad']:
            for lamp_tag in ['1324']:
                rad_cal(ssfr_tag, lc_tag, lamp_tag)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # angular calibration
    #╭────────────────────────────────────────────────────────────────────────────╮#
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
    #╰────────────────────────────────────────────────────────────────────────────╯#
    sys.exit()
#╰────────────────────────────────────────────────────────────────────────────╯#


# radiometric calibrations (1. primary, 2. transfer, 3. secondary)
#╭────────────────────────────────────────────────────────────────────────────╮#
def retrieve_rad_cal(
        date,
        which_ssfr='lasp|ssfr-a',
        which_ssfr_for_flux='lasp|ssfr-a',
        run=True,
        ):

    """
    version 1: 1) time adjustment          : check for time offset and merge SSFR data with aircraft housekeeping data
               2) time synchronization     : interpolate raw SSFR data into the time frame of the housekeeping data
               3) counts-to-flux conversion: apply primary and secondary calibration to convert counts to fluxes
    """

    date_s = date.strftime('%Y%m%d')

    for idset in np.unique(dset_num):

        if which_ssfr_for_flux == which_ssfr:
            # select calibration file (can later be adjusted for different integration time sets)
            #╭──────────────────────────────────────────────────────────────╮#
            fdir_cal = '%s/rad-cal' % _FDIR_CAL_

            jday_today = ssfr.util.dtime_to_jday(date)

            int_time_tag_zen = 'si-%3.3d|in-%3.3d' % (data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num']==idset][0, 0], data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num']==idset][0, 1])
            int_time_tag_nad = 'si-%3.3d|in-%3.3d' % (data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num']==idset][0, 2], data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num']==idset][0, 3])

            # fnames_cal_zen = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324|*lamp-150c_after-pri|*pituffik*%s*zen*%s*' % (which_ssfr_for_flux.lower(), int_time_tag_zen)), key=os.path.getmtime)
            fnames_cal_zen = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324|*lamp-150c*|*pituffik*%s*zen*%s*' % (which_ssfr_for_flux.lower(), int_time_tag_zen)), key=os.path.getmtime)
            jday_cal_zen = np.zeros(len(fnames_cal_zen), dtype=np.float64)
            for i in range(jday_cal_zen.size):
                dtime0_s = os.path.basename(fnames_cal_zen[i]).split('|')[2].split('_')[0]
                dtime0 = datetime.datetime.strptime(dtime0_s, '%Y-%m-%d')
                jday_cal_zen[i] = ssfr.util.dtime_to_jday(dtime0) + i/86400.0
            fname_cal_zen = fnames_cal_zen[np.argmin(np.abs(jday_cal_zen-jday_today))]
            data_cal_zen = ssfr.util.load_h5(fname_cal_zen)

            msg = '\nMessage [cdata_ssfr_v1]: Using <%s> for %s zenith irradiance ...' % (os.path.basename(fname_cal_zen), which_ssfr.upper())
            print(msg)

            # fnames_cal_nad = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324|*lamp-150c_after-pri|*pituffik*%s*nad*%s*' % (which_ssfr_for_flux.lower(), int_time_tag_nad)), key=os.path.getmtime)
            fnames_cal_nad = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324|*lamp-150c*|*pituffik*%s*nad*%s*' % (which_ssfr_for_flux.lower(), int_time_tag_nad)), key=os.path.getmtime)
            jday_cal_nad = np.zeros(len(fnames_cal_nad), dtype=np.float64)
            for i in range(jday_cal_nad.size):
                dtime0_s = os.path.basename(fnames_cal_nad[i]).split('|')[2].split('_')[0]
                dtime0 = datetime.datetime.strptime(dtime0_s, '%Y-%m-%d')
                jday_cal_nad[i] = ssfr.util.dtime_to_jday(dtime0) + i/86400.0
            fname_cal_nad = fnames_cal_nad[np.argmin(np.abs(jday_cal_nad-jday_today))]
            data_cal_nad = ssfr.util.load_h5(fname_cal_nad)

            msg = '\nMessage [cdata_ssfr_v1]: Using <%s> for %s nadir irradiance ...' % (os.path.basename(fname_cal_nad), which_ssfr.upper())
            print(msg)
            #╰──────────────────────────────────────────────────────────────╯#
        else:
            # radiance (scale the data to 0 - 2.0 for now,
            # later we will apply radiometric response after mission to retrieve spectral RADIANCE)

            factor_zen = (np.nanmax(cnt_zen)-np.nanmin(cnt_zen)) / 2.0
            data_cal_zen = {
                    'sec_resp': np.repeat(factor_zen, wvl_zen.size)
                    }

            msg = '\nMessage [cdata_ssfr_v1]: Using [0, 2.0] scaling for %s zenith radiance ...' % (which_ssfr.upper())
            print(msg)

            factor_nad = (np.nanmax(cnt_nad)-np.nanmin(cnt_nad)) / 2.0
            data_cal_nad = {
                    'sec_resp': np.repeat(factor_nad, wvl_nad.size)
                    }

            msg = '\nMessage [cdata_ssfr_v1]: Using [0, 2.0] scaling for %s nadir radiance ...' % (which_ssfr.upper())
            print(msg)

        logic_dset = (dset_num == idset)

    return fname_h5

def ssfr_rad_cal(
        fdir_pri,
        fdir_tra,
        fdir_sec=None,
        spec_reverse=False,
        ):

    # get calibration files of primary
    #╭────────────────────────────────────────────────────────────────────────────╮#
    tags_pri = os.path.basename(fdir_pri).split('_')
    fnames_pri_ = sorted(glob.glob('%s/*.SKS' % (fdir_pri)))
    fnames_pri = [fnames_pri_[-1]]
    if len(fnames_pri) > 1:
        msg = '\nWarning [rad_cal]: find more than one file for "%s", selected "%s" ...' % (fdir_pri, fnames_pri[0])
        warnings.warn(msg)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # get calibration files of transfer
    #╭────────────────────────────────────────────────────────────────────────────╮#
    tags_tra = os.path.basename(fdir_tra).split('_')
    fnames_tra_ = sorted(glob.glob('%s/*.SKS' % (fdir_tra)))
    fnames_tra = [fnames_tra_[-1]]
    if len(fnames_tra) > 1:
        msg = '\nWarning [rad_cal]: find more than one file for "%s", selected "%s" ...' % (fdir_tra, fnames_tra[0])
        warnings.warn(msg)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # secondary calibration files from field
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if fdir_sec is None:
        fdir_sec = fdir_tra
    tags_sec = os.path.basename(fdir_sec).split('_')
    fnames_sec_ = sorted(glob.glob('%s/*.SKS' % (fdir_sec)))
    fnames_sec = [fnames_sec_[-1]]
    if len(fnames_sec) > 1:
        msg = '\nWarning [rad_cal]: find more than one file for "%s", selected "%s" ...' % (fdir_sec, fnames_sec[0])
        warnings.warn(msg)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # tags
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if (tags_pri[1]==tags_tra[1]) and (tags_tra[1]==tags_sec[1]):
        ssfr_tag = tags_pri[1]
    if (tags_pri[2]==tags_tra[2]) and (tags_tra[2]==tags_sec[2]):
        lc_tag = tags_pri[2]
    #╰────────────────────────────────────────────────────────────────────────────╯#

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
#╰────────────────────────────────────────────────────────────────────────────╯#

# angular calibrations
#╭────────────────────────────────────────────────────────────────────────────╮#
def ssfr_ang_cal(fdir):

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
    #╭────────────────────────────────────────────────────────────────────────────╮#
    angles_pos = np.concatenate((np.arange(0.0, 30.0, 3.0), np.arange(30.0, 50.0, 5.0), np.arange(50.0, 91.0, 10.0)))
    angles_neg = -angles_pos
    angles = np.concatenate((angles_pos, angles_neg, np.array([0.0])))
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # make fnames, a dictionary <key:value> with file name as key, angle as value
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fnames_ = sorted(glob.glob('%s/*.SKS' % fdir))
    fnames  = {
            fnames_[i]: angles[i] for i in range(angles.size)
            }
    #╰────────────────────────────────────────────────────────────────────────────╯#

    date_today_s = datetime.datetime.now().strftime('%Y-%m-%d')

    ssfr_ = ssfr.lasp_ssfr.read_ssfr([fnames_[0]])
    for i in range(ssfr_.Ndset):
        dset_tag = 'dset%d' % i
        int_time = ssfr_.dset_info[dset_tag]

        filename_tag = '%s|%s|%s|%s' % (tags[0], tags[4], date_today_s, dset_tag)

        ssfr.cal.cdata_ang_resp(fnames, filename_tag=filename_tag, which_ssfr='lasp|%s' % ssfr_tag, which_lc=lc_tag, int_time=int_time)
#╰────────────────────────────────────────────────────────────────────────────╯#


def main_ssfr_rad_cal(
        which_ssfr='lasp|ssfr-a',
        ):

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

        data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350_postdeployment0
        data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350_postdeployment
        data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350_postdeployment

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

        data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_nad-lc6_transfer_lamp-150c_si-080-120_in-250-350_postdeployment
        data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350_postdeployment

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
        data/arcsix/cal/rad-cal/2024-07-31_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-07-31_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-08-04_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-08-04_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-08-05_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik # after disassembly
        data/arcsix/cal/rad-cal/2024-08-10_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-08-10_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik1
        data/arcsix/cal/rad-cal/2024-08-10_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik2 # after disassembly

    fdirs = [
            {'zen': '',
             'nad': ''},
            ]
    """

    # radiometric calibration
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fdir_cal = '%s/rad-cal' % _FDIR_CAL_

    # primary calibrations (pre-mission)
    #╭──────────────────────────────────────────────────────────────╮#
    # fdirs_pri_cal = ssfr.util.get_all_folders(fdir_cal, pattern='*pri-cal_lamp-1324*si-080-120*in-250-350*')
    # fdirs_pri_cal = ssfr.util.get_all_folders(fdir_cal, pattern='*pri-cal_lamp-506*si-080-120*in-250-350*')
    # for fdir_pri in fdirs_pri_cal:
    #     print(fdir_pri)
    #╰──────────────────────────────────────────────────────────────╯#

    # transfer (pre-mission)
    #╭──────────────────────────────────────────────────────────────╮#
    # fdirs_transfer = ssfr.util.get_all_folders(fdir_cal, pattern='*transfer_lamp-150c*si-080-120*in-250-350*')
    # fdirs_transfer = ssfr.util.get_all_folders(fdir_cal, pattern='*transfer_lamp-150e*si-080-120*in-250-350*')
    # for fdir_transfer in fdirs_transfer:
    #     print(fdir_transfer)
    #╰──────────────────────────────────────────────────────────────╯#

    # secondary calibrations (in-field)
    #╭──────────────────────────────────────────────────────────────╮#
    # fdirs_sec_cal = ssfr.util.get_all_folders(fdir_cal, pattern='*sec-cal_lamp-150c*si-080-120*in-250-350*')
    # fdirs_sec_cal = ssfr.util.get_all_folders(fdir_cal, pattern='*sec-cal_lamp-150e*si-080-120*in-250-350*')
    # for fdir_sec_cal in fdirs_sec_cal:
    #     print(fdir_sec_cal)
    #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if 'ssfr-a' in which_ssfr.lower():

        # SSFR-A (regular setup for measuring irradiance)
        #╭────────────────────────────────────────────────────────────────────────────╮#
        fdirs_pri = [
                {'zen': 'data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350',
                 'nad': 'data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350_post0',
                #  'nad': 'data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350_post'},
                ]

        fdirs_tra = [
                {'zen': 'data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350_after-pri',
                 'nad': 'data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_nad-lc6_transfer_lamp-150c_si-080-120_in-250-350_after-pri'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350_post',
                #  'nad': 'data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_nad-lc6_transfer_lamp-150c_si-080-120_in-250-350_post'},
                ]

        fdirs_sec = [
                # {'zen': 'data/arcsix/cal/rad-cal/2024-05-27_SSFR-A_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                #  'nad': 'data/arcsix/cal/rad-cal/2024-05-26_SSFR-A_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
                {'zen': 'data/arcsix/cal/rad-cal/2024-06-02_SSFR-A_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                 'nad': 'data/arcsix/cal/rad-cal/2024-06-02_SSFR-A_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
                # {'zen': 'data/arcsix/cal/rad-cal/2024-06-09_SSFR-A_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                #  'nad': 'data/arcsix/cal/rad-cal/2024-06-09_SSFR-A_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
                ]
        #╰────────────────────────────────────────────────────────────────────────────╯#

    elif 'ssfr-b' in which_ssfr.lower():


        # SSFR-B (backup setup for measuring irradiance)
        #╭────────────────────────────────────────────────────────────────────────────╮#
        fdirs_pri = [
                {'zen': 'data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350',
                 'nad': 'data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-02-25_SSFR-B_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350_post',
                #  'nad': 'data/arcsix/cal/rad-cal/2025-02-25_SSFR-B_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350_post'},
                ]

        fdirs_tra = [
                {'zen': 'data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350',
                 'nad': 'data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_nad-lc6_transfer_lamp-150c_si-080-160_in-250-350'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-02-25_SSFR-B_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350_post',
                #  'nad': 'data/arcsix/cal/rad-cal/2025-02-25_SSFR-B_nad-lc6_transfer_lamp-150c_si-080-120_in-250-350_post'},
                ]

        fdirs_sec = [
                # {'nad': 'data/arcsix/cal/rad-cal/2024-06-02_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                #  'zen': 'data/arcsix/cal/rad-cal/2024-06-02_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
                # {'nad': 'data/arcsix/cal/rad-cal/2024-07-23_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                #  'zen': 'data/arcsix/cal/rad-cal/2024-07-26_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
                # {'nad': 'data/arcsix/cal/rad-cal/2024-07-31_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                #  'zen': 'data/arcsix/cal/rad-cal/2024-07-31_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
                # {'nad': 'data/arcsix/cal/rad-cal/2024-08-04_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                #  'zen': 'data/arcsix/cal/rad-cal/2024-08-04_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
                # {'nad': 'data/arcsix/cal/rad-cal/2024-08-04_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                #  'zen': 'data/arcsix/cal/rad-cal/2024-08-05_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
                # {'nad': 'data/arcsix/cal/rad-cal/2024-08-10_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                #  'zen': 'data/arcsix/cal/rad-cal/2024-08-10_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik1'},
                {'nad': 'data/arcsix/cal/rad-cal/2024-08-10_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                 'zen': 'data/arcsix/cal/rad-cal/2024-08-10_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik2'},
                ]
        #╰────────────────────────────────────────────────────────────────────────────╯#

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
                    ssfr_rad_cal(fdir_pri0, fdir_tra0, fdir_sec=fdir_sec0, spec_reverse=False)
    return

def field_calibration_check(
        ssfr_tag='ssfr-a',
        lc_tag='zen',
        int_time={'si':80.0, 'in':250.0},
        ):


    tag = '%s|%s|si-%3.3d|in-%3.3d' % (ssfr_tag, lc_tag, int_time['si'], int_time['in'])

    # for post-mission
    # fnames = sorted(glob.glob('*lamp-1324_post|*pituffik*%s*.h5' % tag))

    # for pre-mission
    fnames = sorted(glob.glob('*lamp-1324|*pituffik*%s*.h5' % tag))

    # for SSFR-A post-mission azimuth sensitivity study
    # fnames = sorted(glob.glob('*lamp-1324_post0|*pituffik*%s*.h5' % tag))

    print(fnames)

    # colors = plt.cm.jet(np.linspace(0.0, 1.0, len(fnames)))

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(16, 8))

        ax1 = fig.add_subplot(111)

        for i, fname in enumerate(fnames):
            tags = os.path.basename(fname).replace('.h5', '').split('|')
            f = h5py.File(fname, 'r')
            wvl = f['wvl'][...]
            sec_resp = f['sec_resp'][...]
            pri_resp = f['pri_resp'][...]
            f.close()

            if i == 0:
                ax1.plot(wvl, pri_resp, marker='o', markersize=2, lw=0.5, label='%s (Primary)' % tags[0])

            ax1.plot(wvl, sec_resp, marker='o', markersize=2, lw=0.5, label='%s (Secondary)' % tags[2])

        ax1.set_title('%s (%s)' % (ssfr_tag.upper(), lc_tag.upper()))
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Irradiance [$W m^{-2} nm^{-1}$]')
        ax1.set_xlim((200, 2400))
        if lc_tag == 'zen':
            ax1.set_ylim((0, 600))
        else:
            ax1.set_ylim((0, 800))
        plt.legend(fontsize=12)
        # fig.suptitle('Field Calibration (%s|%s|%s|%s)' % (ssfr_tag.upper(), lc_tag.upper(), tags[-2].upper(), tags[-1].upper()), fontsize=24)


        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        # fig.savefig('%s_%s_%s_%s_%s.png' % (_metadata['Function'], ssfr_tag, lc_tag, tags[-2], tags[-1]), bbox_inches='tight', metadata=_metadata)
        fig.savefig('%s_%s_%s.png' % (_metadata['Function'], ssfr_tag, lc_tag), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#



def main_ssfr_rad_cal_all(
        which_ssfr='lasp|ssfr-a',
        ):

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

        data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350_postdeployment0
        data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350_postdeployment
        data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350_postdeployment

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

        data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_nad-lc6_transfer_lamp-150c_si-080-120_in-250-350_postdeployment
        data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350_postdeployment

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
        data/arcsix/cal/rad-cal/2024-07-31_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-07-31_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-08-04_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-08-04_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-08-05_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik # after disassembly
        data/arcsix/cal/rad-cal/2024-08-10_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik
        data/arcsix/cal/rad-cal/2024-08-10_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik1
        data/arcsix/cal/rad-cal/2024-08-10_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik2 # after disassembly

    fdirs = [
            {'zen': '',
             'nad': ''},
            ]
    """

    # radiometric calibration
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fdir_cal = '%s/rad-cal' % _FDIR_CAL_

    # primary calibrations (pre-mission)
    #╭──────────────────────────────────────────────────────────────╮#
    # fdirs_pri_cal = ssfr.util.get_all_folders(fdir_cal, pattern='*pri-cal_lamp-1324*si-080-120*in-250-350*')
    # fdirs_pri_cal = ssfr.util.get_all_folders(fdir_cal, pattern='*pri-cal_lamp-506*si-080-120*in-250-350*')
    # for fdir_pri in fdirs_pri_cal:
    #     print(fdir_pri)
    #╰──────────────────────────────────────────────────────────────╯#

    # transfer (pre-mission)
    #╭──────────────────────────────────────────────────────────────╮#
    # fdirs_transfer = ssfr.util.get_all_folders(fdir_cal, pattern='*transfer_lamp-150c*si-080-120*in-250-350*')
    # fdirs_transfer = ssfr.util.get_all_folders(fdir_cal, pattern='*transfer_lamp-150e*si-080-120*in-250-350*')
    # for fdir_transfer in fdirs_transfer:
    #     print(fdir_transfer)
    #╰──────────────────────────────────────────────────────────────╯#

    # secondary calibrations (in-field)
    #╭──────────────────────────────────────────────────────────────╮#
    # fdirs_sec_cal = ssfr.util.get_all_folders(fdir_cal, pattern='*sec-cal_lamp-150c*si-080-120*in-250-350*')
    # fdirs_sec_cal = ssfr.util.get_all_folders(fdir_cal, pattern='*sec-cal_lamp-150e*si-080-120*in-250-350*')
    # for fdir_sec_cal in fdirs_sec_cal:
    #     print(fdir_sec_cal)
    #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if 'ssfr-a' in which_ssfr.lower():

        # SSFR-A (regular setup for measuring irradiance)
        #╭────────────────────────────────────────────────────────────────────────────╮#
        fdirs_pri = [
                {'zen': 'data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350',
                 'nad': 'data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350_post0',
                #  'nad': 'data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350_post'},
                ]

        fdirs_tra = [
                {'zen': 'data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350_after-pri',
                 'nad': 'data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_nad-lc6_transfer_lamp-150c_si-080-120_in-250-350_after-pri'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350_post',
                #  'nad': 'data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_nad-lc6_transfer_lamp-150c_si-080-120_in-250-350_post'},
                ]

        fdirs_sec = [
                {'zen': 'data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350_after-pri',
                 'nad': 'data/arcsix/cal/rad-cal/2024-03-29_SSFR-A_nad-lc6_transfer_lamp-150c_si-080-120_in-250-350_after-pri'},
                {'zen': 'data/arcsix/cal/rad-cal/2024-05-27_SSFR-A_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                 'nad': 'data/arcsix/cal/rad-cal/2024-05-26_SSFR-A_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
                {'zen': 'data/arcsix/cal/rad-cal/2024-06-02_SSFR-A_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                 'nad': 'data/arcsix/cal/rad-cal/2024-06-02_SSFR-A_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
                {'zen': 'data/arcsix/cal/rad-cal/2024-06-09_SSFR-A_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                 'nad': 'data/arcsix/cal/rad-cal/2024-06-09_SSFR-A_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
                {'zen': 'data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350_post',
                 'nad': 'data/arcsix/cal/rad-cal/2025-02-18_SSFR-A_nad-lc6_transfer_lamp-150c_si-080-120_in-250-350_post'},
                ]
        #╰────────────────────────────────────────────────────────────────────────────╯#

    elif 'ssfr-b' in which_ssfr.lower():


        # SSFR-B (backup setup for measuring irradiance)
        #╭────────────────────────────────────────────────────────────────────────────╮#
        fdirs_pri = [
                {'zen': 'data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350',
                 'nad': 'data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-02-25_SSFR-B_zen-lc4_pri-cal_lamp-1324_si-080-120_in-250-350_post',
                #  'nad': 'data/arcsix/cal/rad-cal/2025-02-25_SSFR-B_nad-lc6_pri-cal_lamp-1324_si-080-120_in-250-350_post'},
                ]

        fdirs_tra = [
                {'zen': 'data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350',
                 'nad': 'data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_nad-lc6_transfer_lamp-150c_si-080-160_in-250-350'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-02-25_SSFR-B_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350_post',
                #  'nad': 'data/arcsix/cal/rad-cal/2025-02-25_SSFR-B_nad-lc6_transfer_lamp-150c_si-080-120_in-250-350_post'},
                ]

        fdirs_sec = [
                {'zen': 'data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350',
                 'nad': 'data/arcsix/cal/rad-cal/2024-03-21_SSFR-B_nad-lc6_transfer_lamp-150c_si-080-160_in-250-350'},
                {'nad': 'data/arcsix/cal/rad-cal/2024-06-02_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                 'zen': 'data/arcsix/cal/rad-cal/2024-06-02_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
                {'nad': 'data/arcsix/cal/rad-cal/2024-07-23_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                 'zen': 'data/arcsix/cal/rad-cal/2024-07-26_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
                {'nad': 'data/arcsix/cal/rad-cal/2024-07-31_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                 'zen': 'data/arcsix/cal/rad-cal/2024-07-31_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
                {'nad': 'data/arcsix/cal/rad-cal/2024-08-04_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                 'zen': 'data/arcsix/cal/rad-cal/2024-08-04_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
                {'nad': 'data/arcsix/cal/rad-cal/2024-08-04_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                 'zen': 'data/arcsix/cal/rad-cal/2024-08-05_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik'},
                {'nad': 'data/arcsix/cal/rad-cal/2024-08-10_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                 'zen': 'data/arcsix/cal/rad-cal/2024-08-10_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik1'},
                {'nad': 'data/arcsix/cal/rad-cal/2024-08-10_SSFR-B_nad-lc6_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik',
                 'zen': 'data/arcsix/cal/rad-cal/2024-08-10_SSFR-B_zen-lc4_sec-cal_lamp-150c_si-080-120_in-250-350_pituffik2'},
                {'zen': 'data/arcsix/cal/rad-cal/2025-02-25_SSFR-B_zen-lc4_transfer_lamp-150c_si-080-120_in-250-350_post',
                 'nad': 'data/arcsix/cal/rad-cal/2025-02-25_SSFR-B_nad-lc6_transfer_lamp-150c_si-080-120_in-250-350_post'},
                ]
        #╰────────────────────────────────────────────────────────────────────────────╯#

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
                    ssfr_rad_cal(fdir_pri0, fdir_tra0, fdir_sec=fdir_sec0, spec_reverse=False)
    return

def plot_time_series_all(
        which_ssfr='lasp|ssfr-a',
        which_lc='zen',
        int_time={'si':80, 'in':250},
        ):


    if 'ssfr-a' in which_ssfr.lower():
        pattern = '*lamp-150c_after-pri|*%s*%s*si-%3.3d*in-%3.3d*.h5' % (which_ssfr, which_lc, int_time['si'], int_time['in'])
    elif 'ssfr-b' in which_ssfr.lower():
        pattern = '*lamp-150c|*%s*%s*si-%3.3d*in-%3.3d*.h5' % (which_ssfr, which_lc, int_time['si'], int_time['in'])

    fnames = sorted(glob.glob(pattern))

    wvl0 = 550
    wvl1 = 1600

    data0 = np.zeros(len(fnames), dtype=np.float64)
    data1 = np.zeros(len(fnames), dtype=np.float64)
    xlabels = []

    for i, fname in enumerate(fnames):
        # date_s = os.path.basename(fname).split('|')[2].split('_')[0]
        date_s = os.path.basename(fname).split('|')[2].replace('lamp-150c_', '')
        f = h5py.File(fname, 'r')
        wvl = f['wvl'][...]
        resp = f['sec_resp'][...]
        f.close()

        data0[i] = resp[np.argmin(np.abs(wvl-wvl0))]
        data1[i] = resp[np.argmin(np.abs(wvl-wvl1))]

        xlabels.append(date_s)

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    x = np.arange(len(fnames))
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        # fig.suptitle('Figure')
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        ax1.plot(x, data0, marker='o', markersize=8, color='r', lw=1.0)
        ax1.plot(x, data1, marker='o', markersize=8, color='b', lw=1.0)

        ax1.xaxis.set_major_locator(FixedLocator(x))
        ax1.set_xticklabels(xlabels, rotation=45)

        if which_lc == 'zen':
            if 'ssfr-a' in which_ssfr.lower():
                ax1.set_ylim((0, 300))
            elif 'ssfr-b' in which_ssfr.lower():
                ax1.set_ylim((0, 600))
        else:
            if 'ssfr-a' in which_ssfr.lower():
                ax1.set_ylim((0, 600))
            elif 'ssfr-b' in which_ssfr.lower():
                ax1.set_ylim((0, 600))

        ax1.grid()
        ax1.set_ylabel('Secondary Response')
        ax1.set_title('%s (%s)' % (which_ssfr.upper(), which_lc.upper()))
        #╰──────────────────────────────────────────────────────────────╯#
        patches_legend = [
                          mpatches.Patch(color='red'  , label='550 nm'), \
                          mpatches.Patch(color='blue' , label='1600 nm'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = '%s_%s.png' % (_metadata_['Function'], pattern.replace('*', '_'))
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        # plt.show()
        # sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

# radiance calibrations
#╭────────────────────────────────────────────────────────────────────────────╮#
def ssrr_rad_cal(
        fdir_pri,
        which_ssrr='lasp|ssrr-a',
        which_lc='zen',
        spec_reverse=False,
        ):

    # get calibration files of primary
    #╭────────────────────────────────────────────────────────────────────────────╮#
    tags_pri = os.path.basename(fdir_pri).split('_')
    fnames_pri_ = sorted(glob.glob('%s/*.SKS' % (fdir_pri)))
    fnames_pri = [fnames_pri_[-1]]
    if len(fnames_pri) > 1:
        msg = '\nWarning [rad_cal]: find more than one file for "%s", selected "%s" ...' % (fdir_pri, fnames_pri[0])
        warnings.warn(msg)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # tags
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # ssrr_tag = tags_pri[1]
    # lc_tag = tags_pri[2]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # check SSFR spectrometer
    #/----------------------------------------------------------------------------\#
    which_ssrr = which_ssrr.lower()
    which_ssfr = which_ssrr.replace('ssrr', 'ssfr')
    which_lab  = which_ssfr.split('|')[0]
    if which_lab == 'nasa':
        import ssfr.nasa_ssfr as ssfr_toolbox
    elif which_lab == 'lasp':
        import ssfr.lasp_ssfr as ssfr_toolbox
    else:
        msg = '\nError [cdata_rad_resp]: <which_ssfr=> does not support <\'%s\'> (only supports <\'nasa|ssfr-6\'> or <\'lasp|ssfr-a\'> or <\'lasp|ssfr-b\'>).' % which_ssfr
        raise ValueError(msg)
    #\----------------------------------------------------------------------------/#

    # check light collector
    #/----------------------------------------------------------------------------\#
    which_lc = which_lc.lower()
    if (which_lc in ['zenith', 'zen', 'z']) | ('zen' in which_lc):
        which_lc = 'zen'
        if not spec_reverse:
            which_spec = 'zen'
        else:
            which_spec = 'nad'
    elif (which_lc in ['nadir', 'nad', 'n']) | ('nad' in which_lc):
        which_lc = 'nad'
        if not spec_reverse:
            which_spec = 'nad'
        else:
            which_spec = 'zen'
    else:
        msg = '\nError [cdata_cos_resp]: <which_lc=> does not support <\'%s\'> (only supports <\'zenith, zen, z\'> or <\'nadir, nad, n\'>).' % which_lc
        raise ValueError(msg)
    #\----------------------------------------------------------------------------/#

    # lamp
    #/----------------------------------------------------------------------------\#
    which_lamp = tags_pri[4].lower()
    #\----------------------------------------------------------------------------/#

    date_today_s = datetime.datetime.now().strftime('%Y-%m-%d')

    ssrr_ = ssfr.lasp_ssfr.read_ssfr(fnames_pri, verbose=False)

    for i in range(ssrr_.Ndset):
        dset_tag = 'dset%d' % i
        int_time = ssrr_.dset_info[dset_tag]

        # si/in tag
        #/----------------------------------------------------------------------------\#
        si_tag = '%s|si' % which_spec
        in_tag = '%s|in' % which_spec

        if si_tag not in int_time.keys():
            int_time[si_tag] = int_time.pop('si')

        if in_tag not in int_time.keys():
            int_time[in_tag] = int_time.pop('in')
        #\----------------------------------------------------------------------------/#

        if len(tags_pri) == 7:
            cal_tag = '%s_%s' % (tags_pri[0], tags_pri[4])
        elif len(tags_pri) == 8:
            cal_tag = '%s_%s_%s' % (tags_pri[0], tags_pri[4], tags_pri[7])

        filename_tag = '%s|%s_processed-for-arcsix' % (cal_tag, date_today_s)

        pri_resp = ssfr.cal.cal_rad_resp(
                fnames_pri,
                resp=None,
                which_ssfr=which_ssfr,
                which_lc=which_lc,
                spec_reverse=spec_reverse,
                which_lamp=which_lamp,
                int_time=int_time,
                verbose=True,
                )
        
        # wavelength
        #/----------------------------------------------------------------------------\#
        wvls = ssfr_toolbox.get_ssfr_wvl(which_ssfr)

        wvl_start = 350.0
        wvl_end   = 2200.0
        wvl_joint = 950.0
        logic_si  = (wvls[si_tag] >= wvl_start)  & (wvls[si_tag] <= wvl_joint)
        logic_in  = (wvls[in_tag] >  wvl_joint)  & (wvls[in_tag] <= wvl_end)

        wvl_data      = np.concatenate((wvls[si_tag][logic_si], wvls[in_tag][logic_in]))
        pri_resp_data = np.concatenate((pri_resp[si_tag][logic_si], pri_resp[in_tag][logic_in]))

        indices_sort = np.argsort(wvl_data)
        wvl_      = wvl_data[indices_sort]
        pri_resp_ = pri_resp_data[indices_sort]
        #\----------------------------------------------------------------------------/#

        # flux to radiance (reflectance panel)
        #/----------------------------------------------------------------------------\#
        # reflectance panel efficiency
        # effic_refl = 0.97 # 97% reflectance panel efficiency (assumed)
        fname_panel = '%s/panel/12x12Spectralon_ASDwavs.ascii' % (ssfr.common.fdir_data)
        if not os.path.exists(fname_panel):
            msg = '\nError [ssfr_rad_cal]: cannot locate calibration file for panel <%s>.' % fname_panel
            raise OSError(msg)
        data_panel = np.loadtxt(fname_panel)
        wvl_panel = data_panel[:, 0]
        refl_panel = data_panel[:, 1]
        pri_resp_rad = {}
        pri_resp_rad[si_tag] = pri_resp[si_tag] * np.pi / np.interp(wvls[si_tag], wvl_panel, refl_panel)
        pri_resp_rad[in_tag] = pri_resp[in_tag] * np.pi / np.interp(wvls[in_tag], wvl_panel, refl_panel)
        pri_resp_rad_ = pri_resp_ * np.pi / np.interp(wvl_, wvl_panel, refl_panel)
        #\----------------------------------------------------------------------------/#

        # brute force filtering for low response
        #/----------------------------------------------------------------------------\#
        # resp_threshold = 60. # counts / (W m^{-2} nm^{-1} sr^{-1} s)
        resp_threshold = 10. # counts / (W m^{-2} nm^{-1} sr^{-1} s)
        pri_resp_rad[si_tag][pri_resp_rad[si_tag] < resp_threshold] = np.nan
        pri_resp_rad[in_tag][pri_resp_rad[in_tag] < resp_threshold] = np.nan
        pri_resp_rad_[pri_resp_rad_ < resp_threshold] = np.nan
        #\----------------------------------------------------------------------------/#

        # Silicon scaling based on the joint wavelength signals
        #/----------------------------------------------------------------------------\#
        if   which_ssrr.lower() == 'lasp|ssrr-a' and which_lc == 'zen':
            si_in_diff = 0.9036051272130695
        elif which_ssrr.lower() == 'lasp|ssrr-a' and which_lc == 'nad':
            si_in_diff = 0.9808717432930821
        elif which_ssrr.lower() == 'lasp|ssrr-b' and which_lc == 'zen':
            si_in_diff = 0.9279473175434759
        elif which_ssrr.lower() == 'lasp|ssrr-b' and which_lc == 'nad':
            si_in_diff = 0.9887711508106611
        else:
            msg = '\nError [ssfr_rad_cal]: <which_ssrr=> does not support <\'%s\'> (only supports <\'lasp|ssrr-a\'> or <\'lasp|ssrr-b\'>).' % which_ssrr
            raise ValueError(msg)
        scaling_factor = -(si_in_diff - 1.) * np.exp( 0.015 * (350. - wvl_[wvl_ < wvl_joint])) + si_in_diff
        pri_resp_rad_[wvl_ < wvl_joint] = pri_resp_rad_[wvl_ < wvl_joint] * scaling_factor 
        #\----------------------------------------------------------------------------/#


        # save file
        #/----------------------------------------------------------------------------\#
        if filename_tag is not None:
            fname_out = '%s|rad-resp|%s|%s|si-%3.3d|in-%3.3d.h5' % (filename_tag, which_ssrr, which_spec, int_time[si_tag], int_time[in_tag])
        else:
            fname_out = 'rad-resp|%s|%s|si-%3.3d|in-%3.3d.h5' % (which_ssrr, which_spec, int_time[si_tag], int_time[in_tag])

        f = h5py.File(fname_out, 'w')
        f['wvl']       = wvl_
        f['pri_resp']  = pri_resp_rad_

        g = f.create_group('raw')
        
        g_si = g.create_group('si')
        g_si['wvl'] = wvls[si_tag]
        g_si['pri_resp'] = pri_resp_rad[si_tag]

        g_in = g.create_group('in')
        g_in['wvl'] = wvls[in_tag]
        g_in['pri_resp'] = pri_resp_rad[in_tag]

        f.close()
        #\----------------------------------------------------------------------------/#

def main_ssrr_rad_cal_all(
        which_ssrr='lasp|ssrr-a',
        ):

    """
    Notes:
        raiance setup:
            SSFR-A (Alvin)
              - zenith: (what like?)
              - nadir : LC (what like?) + exposed stainless steel fiber

        radiance backup setup:
            SSFR-B (Belana)
              - zenith: (what like?)
              - nadir : LC (what like?) + exposed stainless steel fiber

    Available options for primary calibrations (post-mission):
        data/arcsix/cal/rad-cal/2025-06-03_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeployment
        data/arcsix/cal/rad-cal/2025-06-03_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeployment

    fdirs = [
            {'zen': '',
             'nad': ''},
            ]
    """

    # radiometric calibration
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fdir_cal = '%s/rad-cal' % _FDIR_CAL_

    # primary calibrations (post-mission)
    #╭──────────────────────────────────────────────────────────────╮#
    # fdirs_pri_cal = ssfr.util.get_all_folders(fdir_cal, pattern='*pri-cal_lamp-1324*si-030-050*in-080-180*')
    # fdirs_pri_cal = ssfr.util.get_all_folders(fdir_cal, pattern='*pri-cal_lamp-506*si-030-050*in-080-180*')
    # for fdir_pri in fdirs_pri_cal:
    #     print(fdir_pri)
    #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if 'ssrr-b' in which_ssrr.lower():

        # SSRR-B (regular setup for measuring radiance)
        #╭────────────────────────────────────────────────────────────────────────────╮#
        fdirs_pri = [
                {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeploymentafterwiggle',
                 'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-B_zen-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-03_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-06_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-06_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-060-100_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-06_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-120-200_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-06_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-06_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-240-400_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeploymentafterwiggle'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeploymentwiggleccw'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeploymentwigglecw'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-060-100_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-090-150_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-120-200_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeploymentafterwiggle'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeploymentwiggleccw'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-B_nad-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeploymentwigglecw'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-13_SSRR-B_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentuncal'}, # uncalibrated lamp
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-13_SSRR-B_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentuncalccw'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-13_SSRR-B_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentuncalcw'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-13_SSRR-B_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentuncaldiag'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-13_SSRR-B_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentuncaldiagccw'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-13_SSRR-B_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentuncaldiagcw'},
                
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-B_zen-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeployment'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-B_zen-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeployment123testfiberoutin'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-B_zen-lcx_pri-cal_lamp-1324_si-060-100_in-080-180_postdeployment'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-B_zen-lcx_pri-cal_lamp-1324_si-090-150_in-080-180_postdeployment'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-B_zen-lcx_pri-cal_lamp-1324_si-120-200_in-080-180_postdeployment'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-B_zen-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeployment'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-B_zen-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeployment123test'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-B_zen-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeployment123testfiberoutin'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-B_zen-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeploymentwiggleccw'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-B_zen-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeploymentwigglecw'},
                
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentnad2nadaz000'}, # uncalibrated lamp
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentnad2nadaz030'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentnad2nadaz090'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentnad2nadaz180'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentnad2nadaz270'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentnad2nadaz330'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentnad2nadaz360'},
                
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_zen-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentzen2nadaz000'}, # uncalibrated lamp
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_zen-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentzen2nadaz030'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_zen-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentzen2nadaz090'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_zen-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentzen2nadaz180'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_zen-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentzen2nadaz270'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_zen-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentzen2nadaz330'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_zen-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentzen2nadaz360'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_zen-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentzen2zenaz000'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_zen-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentzen2zenaz030'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_zen-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentzen2zenaz090'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_zen-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentzen2zenaz180'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_zen-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentzen2zenaz270'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_zen-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentzen2zenaz330'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-B_zen-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentzen2zenaz360'},
        ]
        #╰────────────────────────────────────────────────────────────────────────────╯#
    
    elif 'ssrr-a' in which_ssrr.lower():

        # SSRR-A (backup setup for measuring radiance)
        #╭────────────────────────────────────────────────────────────────────────────╮#
        fdirs_pri = [
                {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeploymentafterwiggle',
                 'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-A_zen-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-03_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-03_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeploymentazimuthallyrotated'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-03_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeploymentnopump'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-06_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-06_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeploymentwiggle'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-06_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-060-100_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-06_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-120-200_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-06_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-06_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-240-400_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-06_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-240-400_in-080-180_postdeploymentwiggle'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeploymentafterwiggle'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeploymentwiggleccw'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeploymentwigglecw'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-060-100_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-090-150_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-120-200_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeployment'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeploymentafterwiggle'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeploymentwiggleccw'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-09_SSRR-A_nad-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeploymentwigglecw'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-13_SSRR-A_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentuncal'}, # uncalibrated lamp
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-13_SSRR-A_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentuncalccw'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-13_SSRR-A_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentuncalcw'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-13_SSRR-A_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentuncaldiag'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-13_SSRR-A_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentuncaldiagccw'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-13_SSRR-A_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentuncaldiagcw'},

                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-A_zen-lcx_pri-cal_lamp-1324_si-030-050_in-080-180_postdeployment'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-A_zen-lcx_pri-cal_lamp-1324_si-060-100_in-080-180_postdeployment'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-A_zen-lcx_pri-cal_lamp-1324_si-090-150_in-080-180_postdeployment'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-A_zen-lcx_pri-cal_lamp-1324_si-120-200_in-080-180_postdeployment'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-A_zen-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeployment'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-A_zen-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeploymentwiggleccw'},
                # {'zen': 'data/arcsix/cal/rad-cal/2025-06-16_SSRR-A_zen-lcx_pri-cal_lamp-1324_si-180-300_in-080-180_postdeploymentwigglecw'},
                
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-A_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentnad2nadaz000'}, # uncalibrated lamp
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-A_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentnad2nadaz030'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-A_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentnad2nadaz090'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-A_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentnad2nadaz180'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-A_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentnad2nadaz270'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-A_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentnad2nadaz330'},
                # {'nad': 'data/arcsix/cal/rad-cal/2025-06-18_SSRR-A_nad-lcx_pri-cal_lamp-507_si-180-300_in-080-180_postdeploymentnad2nadaz360'},
                ]
        #╰────────────────────────────────────────────────────────────────────────────╯#

    for fdir_pri in fdirs_pri:
        for spec_tag in ['nad', 'zen']:
        # for spec_tag in ['nad',]:
        # for spec_tag in ['zen',]:
            fdir_pri0 = fdir_pri[spec_tag]
            print(spec_tag)
            print(fdir_pri0)
            ssrr_rad_cal(
                fdir_pri0,
                which_ssrr=which_ssrr,
                which_lc=spec_tag,
                spec_reverse=False,
                )
    return

def plot_response(
        which_ssfr='lasp|ssrr-a',
        which_lc='nad',
        fdir='.',
        ):

    search_path = os.path.join(fdir, '*|*processed-for-arcsix|rad-resp|%s|%s|si-*|in-*.h5' % (which_ssfr, which_lc))
    fnames = sorted(glob.glob(search_path))
    if not fnames:
        raise OSError('No file found for pattern: %s' % search_path)

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)

    for fname in fnames:
        # Optionally parse params from filename if needed
        # params = parse_fname(fname)
        f = h5py.File(fname, 'r')
        wvl = f['wvl'][...]
        resp = f['pri_resp'][...]
        f.close()
        label = os.path.basename(fname)
        ax1.plot(wvl, resp, lw=1.0, label=label)

    ax1.set_xlim(350.0, 2200.0)
    ax1.set_ylim(0.0, None)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Response (counts / (W m$^{-2}$ nm$^{-1}$ sr$^{-1}$ $\cdot$ s))')
    ax1.set_title('%s (%s)' % (which_ssfr.upper(), which_lc.upper()))
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=10)

    fname_fig = '%s_%s.png' % (which_ssfr, which_lc)
    fig.savefig(fname_fig, bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)

def plot_response_lc_wiggle(
        caldate='2024-08-10',
        which_ssfr='lasp|ssrr-a',
        which_lc='nad',
        fdir='.',
        ):
    
    search_path = os.path.join(fdir, '*|*processed-for-arcsix|rad-resp|%s|%s|si-*|in-*.h5' % (which_ssfr, which_lc))
    fnames = sorted(glob.glob(search_path))
    if not fnames:
        raise OSError('No file found for pattern: %s' % search_path)

    target_types = {
        "postdeployment|": "original",
        "postdeploymentwigglecw|": "CW",
        "postdeploymentwiggleccw|": "CCW",
        "postdeploymentuncal|": "original",
        "postdeploymentuncalcw|": "CW",
        "postdeploymentuncalccw|": "CCW",
        "postdeploymentuncaldiag|": "original",
        "postdeploymentuncaldiagcw|": "CW",
        "postdeploymentuncaldiagccw|": "CCW",
        "postdeploymentwiggle|": "wiggle",
        "postdeployment123test|": "Repeat",
        "postdeployment123testfiberoutin|": "Post-swap",
    }
    type_linestyles = {
        "original": "solid",
        "CW": "dashed",
        "CCW": "dotted",
        "wiggle": "dashed",
        "Repeat": "dashed",
        "Post-swap": "dotted",
    }

    remark_titl = ''
    # remark_titl = '(default distance)'
    # remark_titl = '(shorter distance)'

    # target_wvls = [450, 550, 650]
    # colors = ['blue', 'green', 'red']
    target_wvls = [450, 650, 1050, 1250]
    colors = ['blue', 'red', 'cyan', 'magenta']
    # target_wvls = [450, 550,]
    # colors = ['blue', 'green',]

    # Collect data by type
    type_data = {k: {"si_times": [], "resp": {w: [] for w in target_wvls}} for k in target_types}


    for fname in fnames:
        base = os.path.basename(fname)
        if caldate not in base:
            continue
        # Determine type
        type_key = None
        for k in target_types:
            if k in base:
                type_key = k
                break
        if type_key is None:
            continue
        # Extract silicon integration time without re
        si_val = np.nan
        parts = base.split('si-')
        if len(parts) > 1:
            si_str = parts[1].split('|')[0]
            try:
                si_val = float(si_str)
            except Exception:
                si_val = np.nan
        f = h5py.File(fname, 'r')
        wvl = f['wvl'][...]
        resp = f['pri_resp'][...]
        f.close()
        type_data[type_key]["si_times"].append(si_val)
        for twvl in target_wvls:
            type_data[type_key]["resp"][twvl].append(resp[np.argmin(np.abs(wvl - twvl))])

    # Plot
    fig2 = plt.figure(figsize=(5, 6))
    ax2 = fig2.add_subplot(111)
    handles = []
    labels = []
    for i, twvl in enumerate(target_wvls):
        for k, label in target_types.items():
            si_times = np.array(type_data[k]["si_times"])
            if len(si_times) == 0:
                continue
            resp_vals = np.array(type_data[k]["resp"][twvl])
            # Sort by si_times for line plotting
            sort_idx = np.argsort(si_times)
            si_times = si_times[sort_idx]
            resp_vals = resp_vals[sort_idx]
            linestyle = type_linestyles.get(label, 'solid')
            line, = ax2.plot(
                si_times,
                resp_vals,
                marker='o',
                color=colors[i],
                linestyle=linestyle,
                label=f"{twvl} nm ({label})"
            )
            handles.append(line)
            labels.append(f"{twvl} nm ({label})")

            print(f"Plotting {twvl} nm for {label}: SI times = {si_times}, Response = {resp_vals}")

    ax2.set_xlabel('Silicon Integration Time (ms)')
    ax2.set_ylabel('Response (counts / (W m$^{-2}$ nm$^{-1}$ sr$^{-1}$ $\cdot$ s))')
    # ax2.set_title('%s (%s): Angular Wiggle test %s' % (which_ssfr.upper(), which_lc.upper(), remark_titl))
    ax2.set_title('%s (%s): Fiber in/out test %s' % (which_ssfr.upper(), which_lc.upper(), remark_titl))
    ax2.set_xlim(0., None)
    # ax2.set_ylim(2000, 4500)
    # Move legend below the panel, use 3 columns
    ax2.legend(handles=handles, labels=labels, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3)
    fig2.tight_layout()
    fname_fig2 = '%s_%s_vs_si_postwiggle.png' % (which_ssfr, which_lc)
    fig2.savefig(fname_fig2, bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig2)

def plot_response_nonlinearity(
        which_ssfr='lasp|ssrr-a',
        which_lc='nad',
        fdir='.',
        ):
    
    search_path = os.path.join(fdir, '*|*processed-for-arcsix|rad-resp|%s|%s|si-*|in-*.h5' % (which_ssfr, which_lc))
    fnames = sorted(glob.glob(search_path))
    if not fnames:
        raise OSError('No file found for pattern: %s' % search_path)

    # target_wvls = [450, 550, 650,]
    # colors = ['blue', 'green', 'red',]
    target_wvls = [450, 550, 650, 750, 1050, 1150, 1250,] # Note the int time is for Silicon
    colors = ['blue', 'green', 'red', 'brown', 'cyan', 'teal', 'magenta',]
    # target_wvls = [450, 550,]
    # colors = ['blue', 'green',]
    # target_wvls = [550,]
    # colors = ['green',]
    si_times = []
    resp_dict = {wvl: [] for wvl in target_wvls}
    labels = []
    date_list = []
    id_list = []

    # Define marker shapes for up to 10 unique dates
    marker_shapes = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']

    for fname in fnames:
        base = os.path.basename(fname)
        try:
            si_str = base.split('si-')[1].split('|')[0]
            si_val = float(si_str)
        except Exception:
            si_val = np.nan
        si_times.append(si_val)
        # Extract date from filename (first part before '_')
        date_str = base.split('_')[0]
        id_str = base.split('|')[0].split('_')[-1]
        if id_str not in id_list:
            id_list.append(id_str)
        id_idx = id_list.index(id_str)
        date_list.append(date_str + ' no.%d' % (id_idx + 1))
        print(base, date_str + ' no.%d' % (id_idx + 1))
        f = h5py.File(fname, 'r')
        wvl = f['wvl'][...]
        resp = f['pri_resp'][...]
        f.close()
        for twvl in target_wvls:
            resp_dict[twvl].append(resp[np.argmin(np.abs(wvl - twvl))])
        labels.append(base)

    si_times = np.array(si_times)
    date_list = np.array(date_list)
    for twvl in target_wvls:
        resp_dict[twvl] = np.array(resp_dict[twvl])

    unique_dates = np.unique(date_list)
    marker_map = {d: marker_shapes[i % len(marker_shapes)] for i, d in enumerate(unique_dates)}

    fig2 = plt.figure(figsize=(6, 5))
    gs = GridSpec(2, 1, height_ratios=[10, 1.5])
    ax2 = fig2.add_subplot(gs[0])
    for i, twvl in enumerate(target_wvls):
        for d in unique_dates:
            mask = date_list == d
            marker = marker_map[d]
            ax2.scatter(
                si_times[mask], resp_dict[twvl][mask],
                color=colors[i], marker=marker, s=50, edgecolor='k', linewidth=0.5
            )
            if np.sum(mask) > 1:
                ax2.plot(si_times[mask], resp_dict[twvl][mask], color=colors[i], alpha=0.5, linewidth=1)

        print(f"\nWavelength: {twvl} nm")
        for si in np.unique(si_times):
            mask = si_times == si
            if np.sum(mask) > 0:
                vals = resp_dict[twvl][mask]
                mean_val = np.nanmean(vals)
                min_val = np.nanmin(vals)
                max_val = np.nanmax(vals)
                err_val = (max_val - min_val) / mean_val
                print(f"  SI={si:.0f} ms: mean={mean_val:.2f}, min={min_val:.2f}, max={max_val:.2f}, err={err_val:.2%}")
    ax2.set_xlabel('Silicon Integration Time (ms)')
    ax2.set_ylabel('Response (counts / (W m$^{-2}$ nm$^{-1}$ sr$^{-1}$ $\cdot$ s))')
    ax2.set_title('%s (%s): Response at %s nm' % (which_ssfr.upper(), which_lc.upper(), ', '.join(str(w) for w in target_wvls)))
    ax2.set_xlim(0., None)
    ax2.set_ylim(1750, 4750)

    handles_dates = [
        plt.Line2D([0], [0], marker=marker_map[d], color='w', label=d, markerfacecolor='gray', markeredgecolor='k', markersize=8)
        for d in unique_dates
    ]
    handles_wvls = [mpatches.Patch(color=colors[i], label=f'{target_wvls[i]} nm') for i in range(len(target_wvls))]

    ax_leg = fig2.add_subplot(gs[1])
    ax_leg.axis('off')

    legend1 = ax_leg.legend(handles=handles_dates, title='Date', loc='center left', bbox_to_anchor=(0.05, 0.5), fontsize=8, ncol=2)
    legend2 = ax_leg.legend(handles=handles_wvls, title='Wavelength', loc='center right', bbox_to_anchor=(0.95, 0.5), fontsize=8)
    ax_leg.add_artist(legend1)

    fig2.tight_layout()
    fname_fig2 = '%s_%s_vs_si.png' % (which_ssfr, which_lc)
    fig2.savefig(fname_fig2, bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig2)

def plot_response_fiber_rotation(
        which_ssfr='lasp|ssrr-a',
        which_lc='nad',
        fdir='.',
        ):
    
    search_path = os.path.join(fdir, '*|*processed-for-arcsix|rad-resp|%s|%s|si-*|in-*.h5' % (which_ssfr, which_lc))
    fnames = sorted(glob.glob(search_path))
    if not fnames:
        raise OSError('No file found for pattern: %s' % search_path)

    # Plot each response relative to the mean for each unique (si, in) integration time set
    # Use different linestyles for different unique integration times

    si_times = []
    in_times = []
    wvl_list = []
    resp_list = []

    for fname in fnames:
        base = os.path.basename(fname)
        if 'az' not in base:
            continue
        try:
            si_str = base.split('si-')[1].split('|')[0]
            in_str = base.split('in-')[1].split('.')[0]
            si_val = float(si_str)
            in_val = float(in_str)
        except Exception:
            si_val = np.nan
            in_val = np.nan
        si_times.append(si_val)
        in_times.append(in_val)
        f = h5py.File(fname, 'r')
        wvl = f['wvl'][...]
        resp = f['pri_resp'][...]
        f.close()
        wvl_list.append(wvl)
        resp_list.append(resp)

    si_times = np.array(si_times)
    in_times = np.array(in_times)
    wvl_arr = np.array(wvl_list)
    resp_arr = np.array(resp_list)

    ### Fitting function
    def fit_func(x, a):
        return -(a - 1.) * np.exp(0.015*(350. - x)) + a

    # Find all unique (si, in) pairs
    unique_pairs = np.unique(np.stack([si_times, in_times], axis=1), axis=0)

    # Define linestyles for up to 10 unique integration time sets
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (3, 5, 1, 5)), (0, (5, 1)), (0, (1, 10))]
    linestyle_map = {}
    for idx, pair in enumerate(unique_pairs):
        linestyle_map[tuple(pair)] = linestyles[idx % len(linestyles)]

    fig3, ax3 = plt.subplots(figsize=(10, 6))

    for pair in unique_pairs:
        mask = (si_times == pair[0]) & (in_times == pair[1])
        n_case = np.sum(mask)
        if n_case < 1:
            continue
        # Interpolate all responses to a common wavelength grid
        wvl_common = wvl_list[np.where(mask)[0][0]]
        resp_interp = []
        for i in np.where(mask)[0]:
            resp_interp.append(np.interp(wvl_common, wvl_list[i], resp_list[i]))
        resp_interp = np.array(resp_interp)
        mean_vals = np.nanmean(resp_interp, axis=0)
        # Plot each case relative to the mean
        for j, i in enumerate(np.where(mask)[0]):
            rel_val = resp_interp[j] / mean_vals
            coeff_diff = rel_val[np.where((wvl_common < 950) & (wvl_common >= 850))].mean() / rel_val[np.where((wvl_common > 950) & (wvl_common <= 1050))].mean()
            print(f"Pair: {pair}, Case {j+1}/{n_case}, Coeff Diff: {coeff_diff:.4f}")
            out_rel_val = rel_val.copy()
            out_rel_val[np.where(wvl_common < 950)] /= fit_func(wvl_common[np.where(wvl_common < 950)], coeff_diff)
            label = f"si={int(pair[0])}, in={int(pair[1])} (case {j+1}/{n_case})"
            ax3.plot(
                wvl_common, out_rel_val,
                linestyle=linestyle_map[tuple(pair)],
                alpha=0.7,
                label=label
            )
    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Response / Mean')
    ax3.set_title('Response Relative to Mean for Each Integration Time Set')
    ax3.set_xlim(350, 2200)
    ax3.set_ylim(0.8, 1.2)
    ax3.legend(fontsize=9, loc='upper right')
    fig3.tight_layout()
    fig3.savefig('%s_%s_rel2mean_vs_wvl.png' % (which_ssfr, which_lc), bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig3)

    target_wvls = [450, 550, 650, 750, 1050, 1150, 1250,]
    colors = ['blue', 'green', 'red', 'brown', 'cyan', 'teal', 'magenta',]
    si_times = []
    resp_dict = {wvl: [] for wvl in target_wvls}
    labels = []
    date_list = []
    ang_list = []
    id_list = []

    # Define marker shapes for up to 10 unique entries
    marker_shapes = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']

    for fname in fnames:
        base = os.path.basename(fname)
        if 'az' not in base:
            continue
        try:
            si_str = base.split('si-')[1].split('|')[0]
            si_val = float(si_str)
        except Exception:
            si_val = np.nan
        si_times.append(si_val)
        # Extract date from filename (first part before '_')
        date_str = base.split('_')[0]
        id_str = base.split('|')[0].split('_')[-1]
        if id_str not in id_list:
            id_list.append(id_str)
        id_idx = id_list.index(id_str)
        date_list.append(date_str + ' no.%d' % (id_idx + 1))
        ang_list.append(int(base.split('az')[1].split('|')[0]) if 'az' in base else 'unknown')
        f = h5py.File(fname, 'r')
        wvl = f['wvl'][...]
        resp = f['pri_resp'][...]
        f.close()
        for twvl in target_wvls:
            resp_dict[twvl].append(resp[np.argmin(np.abs(wvl - twvl))])
        labels.append(base)

    si_times = np.array(si_times)
    date_list = np.array(date_list)
    for twvl in target_wvls:
        resp_dict[twvl] = np.array(resp_dict[twvl])

    unique_dates = np.unique(date_list)
    marker_map = {d: marker_shapes[i % len(marker_shapes)] for i, d in enumerate(unique_dates)}

    fig2 = plt.figure(figsize=(6, 5))
    gs = GridSpec(2, 1, height_ratios=[10, 1.5])

    ax2 = fig2.add_subplot(gs[0])
    for i, twvl in enumerate(target_wvls):
        for d in unique_dates:
            mask = date_list == d
            marker = marker_map[d]
            ax2.scatter(
                si_times[mask], resp_dict[twvl][mask],
                color=colors[i], marker=marker, s=50, edgecolor='k', linewidth=0.5
            )
            if np.sum(mask) > 1:
                ax2.plot(si_times[mask], resp_dict[twvl][mask], color=colors[i], alpha=0.5, linewidth=1)
        # For each integration time, print mean, std, min, max of response across all unique dates
        print(f"\nWavelength: {twvl} nm")
        for si in np.unique(si_times):
            mask = si_times == si
            if np.sum(mask) > 0:
                vals = resp_dict[twvl][mask]
                mean_val = np.nanmean(vals)
                min_val = np.nanmin(vals)
                max_val = np.nanmax(vals)
                err_val = (max_val - min_val) / mean_val
                print(f"  SI={si:.0f} ms: mean={mean_val:.2f}, min={min_val:.2f}, max={max_val:.2f}, err={err_val:.2%}")
    ax2.set_xlabel('Silicon Integration Time (ms)')
    ax2.set_ylabel('Response (counts / (W m$^{-2}$ nm$^{-1}$ sr$^{-1}$ $\cdot$ s))')
    ax2.set_title('%s (%s): Response at %s nm' % (which_ssfr.upper(), which_lc.upper(), ', '.join(str(w) for w in target_wvls)))
    ax2.set_xlim(0., None)
    ax2.set_ylim(1750, 4750)

    handles_dates = [
        plt.Line2D([0], [0], marker=marker_map[d], color='w', label=d, markerfacecolor='gray', markeredgecolor='k', markersize=8)
        for d in unique_dates
    ]
    handles_wvls = [mpatches.Patch(color=colors[i], label=f'{target_wvls[i]} nm') for i in range(len(target_wvls))]

    ax_leg = fig2.add_subplot(gs[1])
    ax_leg.axis('off')

    legend1 = ax_leg.legend(handles=handles_dates, title='Date', loc='center left', bbox_to_anchor=(0.05, 0.5), fontsize=8, ncol=2)
    legend2 = ax_leg.legend(handles=handles_wvls, title='Wavelength', loc='center right', bbox_to_anchor=(0.95, 0.5), fontsize=8)
    ax_leg.add_artist(legend1)

    fig2.tight_layout()
    fname_fig2 = '%s_%s_vs_si.png' % (which_ssfr, which_lc)
    fig2.savefig(fname_fig2, bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig2)

    # For each unique (si, in) integration time set, plot response vs date for all target wavelengths
    target_wvls = [450, 550,]
    colors = ['blue', 'green']
    # target_wvls = [1150, 1250] # note that the integration time is currently for Silicon, not InGaAs
    # colors = ['teal', 'magenta']

    # print(si_times)
    # print(in_times)
    # print(date_list)
    # print(ang_list)
    # datenum_to_datedeg = {
    #     '2025-06-18 no.1': 'no. 1   0 deg',
    #     '2025-06-18 no.2': 'no. 2   30 deg',
    #     '2025-06-18 no.3': 'no. 3   90 deg',
    #     '2025-06-18 no.4': 'no. 4   180 deg',
    #     '2025-06-18 no.5': 'no. 5   270 deg',
    #     '2025-06-18 no.6': 'no. 6   330 deg',
    #     '2025-06-18 no.7': 'no. 7   360 deg',
    # }
    datenum_to_datedeg = {
        date: f'{date.split(" ")[1]} {ang} deg' for iang, (date, ang) in enumerate(zip(date_list, ang_list))
    }

    date_list = np.array([datenum_to_datedeg.get(d, d) for d in date_list])

    for pair in unique_pairs:
        mask_pair = (si_times == pair[0]) & (in_times == pair[1])
        if np.sum(mask_pair) < 1:
            continue
        fig_pair = plt.figure(figsize=(6, 4.5))
        gs_pair = GridSpec(1, 1)
        ax_pair = fig_pair.add_subplot(gs_pair[0])

        unique_dates_pair = np.unique(date_list)
        for i, twvl in enumerate(target_wvls):
            yvals = []
            for d in unique_dates_pair:
                mask_date = (date_list == d) & mask_pair
                vals_d = resp_dict[twvl][mask_date]
                if len(vals_d) > 0:
                    mean_val = np.nanmean(vals_d)
                    yvals.append(mean_val)
                else:
                    yvals.append(np.nan)
            ax_pair.plot(
                np.arange(len(unique_dates_pair)), yvals,
                marker='o', color=colors[i], label=f'{twvl} nm'
            )

        ax_pair.set_xticks(np.arange(len(unique_dates_pair)))
        ax_pair.set_xticklabels(unique_dates_pair, rotation=45, ha='right')
        ax_pair.set_ylabel('Response (counts / (W m$^{-2}$ nm$^{-1}$ sr$^{-1}$ $\cdot$ s))')
        ax_pair.set_title('%s (%s): Response vs Azimuth angle (si=%d, in=%d)' % (
            which_ssfr.upper(), which_lc.upper(), int(pair[0]), int(pair[1])))
        ax_pair.legend(fontsize=9, loc='upper right')

        fig_pair.tight_layout()
        fname_fig_pair = '%s_%s_vs_date_si%d_in%d.png' % (
            which_ssfr, which_lc, int(pair[0]), int(pair[1]))
        fig_pair.savefig(fname_fig_pair, bbox_inches='tight', transparent=False, dpi=300)
        plt.close(fig_pair)


if __name__ == '__main__':

    # process field calibration (SSFR-A)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # main_ssfr_rad_cal(which_ssfr='lasp|ssfr-a')

    # for lc_tag in ['zen', 'nad']:
    #     for int_time in [{'si':80, 'in':250}, {'si':120, 'in':350}]:
    #         field_calibration_check(lc_tag=lc_tag, int_time=int_time)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # process field calibration (SSFR-B)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # main_ssfr_rad_cal(which_ssfr='lasp|ssfr-b')

    # for lc_tag in ['zen', 'nad']:
    #     for int_time in [{'si':80, 'in':250}, {'si':120, 'in':350}]:
    #         field_calibration_check(ssfr_tag='ssfr-b', lc_tag=lc_tag, int_time=int_time)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    main_ssfr_rad_cal_all(which_ssfr='lasp|ssfr-a')
    # plot_time_series_all(which_ssfr='lasp|ssfr-a', which_lc='zen')
    # plot_time_series_all(which_ssfr='lasp|ssfr-a', which_lc='nad')

    # main_ssfr_rad_cal_all(which_ssfr='lasp|ssfr-b')
    # plot_time_series_all(which_ssfr='lasp|ssfr-b', which_lc='zen')
    # plot_time_series_all(which_ssfr='lasp|ssfr-b', which_lc='nad')


    # angular calibrations(SSFR-B, zen-lc4,  post)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # fdir = 'data/arcsix/cal/ang-cal/2025-03-05_SSFR-B_zen-lc4_ang-cal_vaa-000_lamp-507_si-080-120_in-250-350_post'
    # ssfr_ang_cal(fdir)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # post-mission SSRR calibration
    #╭────────────────────────────────────────────────────────────────────────────╮#
    main_ssrr_rad_cal_all(which_ssrr='lasp|ssrr-a')
    main_ssrr_rad_cal_all(which_ssrr='lasp|ssrr-b')
    plot_response(which_ssfr='lasp|ssrr-a', which_lc='nad', fdir='.',)
    plot_response(which_ssfr='lasp|ssrr-b', which_lc='nad', fdir='.',)
    plot_response(which_ssfr='lasp|ssrr-a', which_lc='zen', fdir='.',)
    plot_response(which_ssfr='lasp|ssrr-b', which_lc='zen', fdir='.',)

    # Nonlinearity test
    # plot_response_nonlinearity(which_ssfr='lasp|ssrr-a', which_lc='nad', fdir='.')
    # plot_response_nonlinearity(which_ssfr='lasp|ssrr-b', which_lc='nad', fdir='.')

    # wiggle test
    # plot_response_lc_wiggle(caldate='2025-06-13', which_ssfr='lasp|ssrr-a', which_lc='nad', fdir='.')
    # plot_response_lc_wiggle(caldate='2025-06-13', which_ssfr='lasp|ssrr-b', which_lc='nad', fdir='.')

    # Fiber rotation test
    # plot_response_fiber_rotation(which_ssfr='lasp|ssrr-a', which_lc='nad', fdir='.')
    # plot_response_fiber_rotation(which_ssfr='lasp|ssrr-b', which_lc='nad', fdir='.')
    # plot_response_fiber_rotation(which_ssfr='lasp|ssrr-b', which_lc='zen', fdir='.')
    #╰────────────────────────────────────────────────────────────────────────────╯#

    pass
