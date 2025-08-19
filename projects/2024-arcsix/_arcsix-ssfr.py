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
        Vikas Nataraja, Arabella Chamberlain, Ken Hirata, Sebastian Becker, Sebastian Schmidt
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
# import cartopy.crs as ccrs
# mpl.use('Agg')



import ssfr


_SSFR1_TIME_OFFSET_ = {
        '20240517': 185.0,
        '20240521': 182.0,
        '20240524': -145.75,
        '20240528': -156.26,
        '20240530': -158.04,
        '20240531': -161.38,
        '20240603': -170.42,
        '20240605': -176.88,
        '20240606': -180.41,
        '20240607': -181.44,
        '20240610': -188.70,
        '20240611': -190.69,
        '20240613': -196.06,
        '20240708': -196.06,
        '20240709': -196.06,
        '20240722': -196.06,
        '20240724': -196.06,
        '20240725': -299.86,
        '20240726': -299.86,
        '20240729': -307.87,
        '20240730': -307.64,
        '20240801': -315.90,
        '20240802': -317.40,
        '20240807': -328.88,
        '20240808': -331.98,
        '20240809': -333.53,
        '20240815': -353.13,
        '20240816': -353.13,
        }

_SSFR2_TIME_OFFSET_ = {
        '20240517': 115.0,
        '20240521': -6.0,
        '20240524': -208.22,
        '20240528': -222.66,
        '20240530': -229.45,
        '20240531': -227.00,
        '20240603': -241.66,
        '20240605': -250.48,
        '20240606': -256.90,
        '20240607': -255.45,
        '20240610': -261.64,
        '20240611': -271.93,
        '20240613': -273.59,
        '20240708': -273.59,
        '20240709': -273.59,
        '20240722': -273.59,
        '20240724': -273.59, #? inaccurate
        '20240725': -397.91,
        '20240726': -397.91,
        '20240729': -408.39,
        '20240730': -408.13,
        '20240801': -416.93,
        '20240802': -419.59,
        '20240807': -434.47,
        '20240808': -437.18,
        '20240809': -439.71,
        '20240815': -457.82,
        '20240816': -457.82,
        }


# functions for processing SSFR
#╭────────────────────────────────────────────────────────────────────────────╮#
def cdata_ssfr_v0(
        date,
        fnames_ssfr,
        fname_h5='SSFR_v0.h5',
        fdir_out='./',
        which_ssfr='lasp|ssfr-a',
        wvl_s=350.0,
        wvl_e=2000.0,
        wvl_j=950.0,
        dark_extend=1,
        light_extend=1,
        dark_corr_mode='interp',
        run=True,
        ):

    """
    version 0: counts after dark correction
    """

    date_s = date.strftime('%Y%m%d')

    if run:

        ssfr0 = ssfr.lasp_ssfr.read_ssfr(
                fnames_ssfr,
                which_ssfr=which_ssfr.lower(),
                wvl_s=wvl_s,
                wvl_e=wvl_e,
                wvl_j=wvl_j,
                dark_extend=dark_extend,
                light_extend=light_extend,
                dark_corr_mode=dark_corr_mode,
                )

        # data that are useful
        #   wvl_zen [nm]
        #   cnt_zen [counts/ms]
        #   sat_zen: saturation tag, 1 means saturation
        #   wvl_nad [nm]
        #   cnt_nad [counts/ms]
        #   sat_nad: saturation tag, 1 means saturation
        #╭────────────────────────────────────────────────────────────────────────────╮#
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
        #╰────────────────────────────────────────────────────────────────────────────╯#

    return fname_h5

def cdata_ssfr_v1(
        date,
        fname_ssfr_v0,
        fname_hsk,
        fname_h5='SSFR_v1.h5',
        fdir_out='./',
        time_offset=0.0,
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

    if run:

        # load ssfr v0 data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        data_ssfr_v0 = ssfr.util.load_h5(fname_ssfr_v0)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # load hsk
        #╭────────────────────────────────────────────────────────────────────────────╮#
        data_hsk = ssfr.util.load_h5(fname_hsk)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # read wavelengths and calculate toa downwelling solar flux
        #╭────────────────────────────────────────────────────────────────────────────╮#
        flux_toa = ssfr.util.get_solar_kurudz()

        wvl_zen = data_ssfr_v0['spec/wvl_zen']
        f_dn_sol_zen = np.zeros_like(wvl_zen)
        for i, wvl0 in enumerate(wvl_zen):
            f_dn_sol_zen[i] = ssfr.util.cal_weighted_flux(wvl0, flux_toa[:, 0], flux_toa[:, 1])*ssfr.util.cal_solar_factor(date)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        f = h5py.File(fname_h5, 'w')

        # processing data - since we have dual integration times, SSFR data with different
        # integration time will be processed seperately
        #╭────────────────────────────────────────────────────────────────────────────╮#
        jday     = data_ssfr_v0['raw/jday']
        dset_num = data_ssfr_v0['raw/dset_num']

        wvl_zen  = data_ssfr_v0['spec/wvl_zen']
        wvl_nad  = data_ssfr_v0['spec/wvl_nad']

        cnt_zen  = data_ssfr_v0['spec/cnt_zen']
        spec_zen = np.zeros_like(cnt_zen)
        cnt_nad  = data_ssfr_v0['spec/cnt_nad']
        spec_nad = np.zeros_like(cnt_nad)

        for idset in np.unique(dset_num):

            logic_dset = (dset_num == idset)

            if which_ssfr_for_flux == which_ssfr:
                # select calibration file (can later be adjusted for different integration time sets)
                #╭──────────────────────────────────────────────────────────────╮#
                fdir_cal = '%s/rad-cal' % cfg.fdir_cal #_FDIR_CAL_

                jday_today = ssfr.util.dtime_to_jday(date)

                int_time_tag_zen = 'si-%3.3d|in-%3.3d' % (data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num']==idset][0, 0], data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num']==idset][0, 1])
                int_time_tag_nad = 'si-%3.3d|in-%3.3d' % (data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num']==idset][0, 2], data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num']==idset][0, 3])

                # fnames_cal_zen = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324|*lamp-150c_after-pri|*pituffik*%s*zen*%s*' % (which_ssfr_for_flux.lower(), int_time_tag_zen)), key=os.path.getmtime)
                fnames_cal_zen = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324|*lamp-150c*|*pituffik*%s*zen*%s*' % (which_ssfr_for_flux.lower(), int_time_tag_zen)), key=os.path.getmtime)
                if len(fnames_cal_zen) == 0:
                    msg = '\nWarnings [cdata_ssfr_v1]: No zenith calibration file found for <%s> ...' % (int_time_tag_zen)
                    warnings.warn(msg)
                    int_time_tag_zen = 'si-080|in-250'  # default integration time tag for zenith calibration
                    msg = '\nMessage [cdata_ssfr_v1]: Using the zenith calibration file for <%s> ...' % (int_time_tag_zen)
                    print(msg)
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
                if len(fnames_cal_nad) == 0:
                    msg = '\nWarnings [cdata_ssfr_v1]: No nadir calibration file found for <%s> ...' % (int_time_tag_nad)
                    warnings.warn(msg)
                    int_time_tag_nad = 'si-080|in-250'  # default integration time tag for nadir calibration
                    msg = '\nMessage [cdata_ssfr_v1]: Using the nadir calibration file for <%s> ...' % (int_time_tag_nad)
                    print(msg)
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

                # convert counts to flux
                #╭──────────────────────────────────────────────────────────────╮#
                for i in range(wvl_zen.size):
                    spec_zen[logic_dset, i] = cnt_zen[logic_dset, i] / data_cal_zen['sec_resp'][i]

                for i in range(wvl_nad.size):
                    spec_nad[logic_dset, i] = cnt_nad[logic_dset, i] / data_cal_nad['sec_resp'][i]
                #╰──────────────────────────────────────────────────────────────╯#
                
            else:
                fdir_cal = '%s/rad-cal' % cfg.fdir_cal #_FDIR_CAL_

                jday_today = ssfr.util.dtime_to_jday(date)

                int_time_tag_zen = 'si-%3.3d|in-%3.3d' % (data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num']==idset][0, 0], data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num']==idset][0, 1])
                int_time_tag_nad = 'si-%3.3d|in-%3.3d' % (data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num']==idset][0, 2], data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num']==idset][0, 3])

                fnames_cal_zen = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324_postdeployment|*%s*zen*%s*' % (which_ssfr.lower().replace('ssfr', 'ssrr'), int_time_tag_zen)), key=os.path.getmtime)
                jday_cal_zen = np.zeros(len(fnames_cal_zen), dtype=np.float64)
                for i in range(jday_cal_zen.size):
                    dtime0_s = os.path.basename(fnames_cal_zen[i]).split('|')[1].split('_')[0]
                    dtime0 = datetime.datetime.strptime(dtime0_s, '%Y-%m-%d')
                    jday_cal_zen[i] = ssfr.util.dtime_to_jday(dtime0) + i/86400.0
                fname_cal_zen = fnames_cal_zen[np.argmin(np.abs(jday_cal_zen-jday_today))]
                data_cal_zen = ssfr.util.load_h5(fname_cal_zen)

                msg = '\nMessage [cdata_ssfr_v1]: Using <%s> for %s zenith radince ...' % (os.path.basename(fname_cal_zen), which_ssfr.upper())
                print(msg)

                fnames_cal_nad = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324_postdeployment|*%s*nad*%s*' % (which_ssfr.lower().replace('ssfr', 'ssrr'), int_time_tag_nad)), key=os.path.getmtime)
                jday_cal_nad = np.zeros(len(fnames_cal_nad), dtype=np.float64)
                for i in range(jday_cal_nad.size):
                    dtime0_s = os.path.basename(fnames_cal_nad[i]).split('|')[1].split('_')[0]
                    dtime0 = datetime.datetime.strptime(dtime0_s, '%Y-%m-%d')
                    jday_cal_nad[i] = ssfr.util.dtime_to_jday(dtime0) + i/86400.0
                fname_cal_nad = fnames_cal_nad[np.argmin(np.abs(jday_cal_nad-jday_today))]
                data_cal_nad = ssfr.util.load_h5(fname_cal_nad)

                msg = '\nMessage [cdata_ssfr_v1]: Using <%s> for %s nadir radince ...' % (os.path.basename(fname_cal_nad), which_ssfr.upper())
                print(msg)

                # # radiance (scale the data to 0 - 2.0 for now,
                # # later we will apply radiometric response after mission to retrieve spectral RADIANCE)

                # factor_zen = (np.nanmax(cnt_zen)-np.nanmin(cnt_zen)) / 2.0
                # data_cal_zen = {
                #         'sec_resp': np.repeat(factor_zen, wvl_zen.size)
                #         }

                # msg = '\nMessage [cdata_ssfr_v1]: Using [0, 2.0] scaling for %s zenith radiance ...' % (which_ssfr.upper())
                # print(msg)

                # factor_nad = (np.nanmax(cnt_nad)-np.nanmin(cnt_nad)) / 2.0
                # data_cal_nad = {
                #         'sec_resp': np.repeat(factor_nad, wvl_nad.size)
                #         }

                # msg = '\nMessage [cdata_ssfr_v1]: Using [0, 2.0] scaling for %s nadir radiance ...' % (which_ssfr.upper())
                # print(msg)

                # convert counts to radiance
                #╭──────────────────────────────────────────────────────────────╮#
                for i in range(wvl_zen.size):
                    spec_zen[logic_dset, i] = cnt_zen[logic_dset, i] / data_cal_zen['pri_resp'][i]

                for i in range(wvl_nad.size):
                    spec_nad[logic_dset, i] = cnt_nad[logic_dset, i] / data_cal_nad['pri_resp'][i]
                #╰──────────────────────────────────────────────────────────────╯#

                ### (tentative solution) Force the lower integration time data to be NaN
                #╭──────────────────────────────────────────────────────────────╮#
                idset_zen_max_int = np.argmax([data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num'] == i][0, 0] for i in np.unique(dset_num)])
                if idset != idset_zen_max_int:
                    for i in range(wvl_zen.size):
                        spec_zen[logic_dset, :] = np.nan
                idset_zen_max_int = np.argmax([data_ssfr_v0['raw/int_time'][data_ssfr_v0['raw/dset_num'] == i][0, 2] for i in np.unique(dset_num)])
                if idset != idset_zen_max_int:
                    for i in range(wvl_zen.size):
                        spec_nad[logic_dset, :] = np.nan
                #╰──────────────────────────────────────────────────────────────╯#



            # set saturation to 0
            #╭──────────────────────────────────────────────────────────────╮#
            spec_zen[data_ssfr_v0['spec/sat_zen']==1] = -0.05
            spec_nad[data_ssfr_v0['spec/sat_nad']==1] = -0.05
            #╰──────────────────────────────────────────────────────────────╯#
        #╰────────────────────────────────────────────────────────────────────────────╯#


        # interpolate ssfr data to hsk time frame
        # and convert counts to flux
        #╭────────────────────────────────────────────────────────────────────────────╮#
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
        #╰────────────────────────────────────────────────────────────────────────────╯#


        # save processed data
        #╭────────────────────────────────────────────────────────────────────────────╮#
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
        #╰────────────────────────────────────────────────────────────────────────────╯#


        # save processed data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        for key in data_hsk.keys():
            f[key] = data_hsk[key]

        f['time_offset'] = time_offset
        f['tmhr_ori'] = data_hsk['tmhr'] - time_offset/3600.0
        f['jday_ori'] = data_hsk['jday'] - time_offset/86400.0

        f.close()
        #╰────────────────────────────────────────────────────────────────────────────╯#

    return fname_h5

def cdata_ssfr_v2(
        date,
        fname_ssfr_v1,
        fname_alp_v1,
        fname_hsr1_v2,
        fname_h5='SSFR_v2.h5',
        fdir_out='./',
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        run=True,
        run_aux=True,
        ):

    """
    version 2: apply cosine correction to correct for non-linear angular resposne
               diffuse radiation: use cosine response integrated over the entire angular space
               direct radiation: use cosine response over the entire angular space measured in the lab

               diffuse and direct seperation is guided by the diffuse ratio measured by HSR1
    """

    def func_diff_ratio(x, a, b, c):

        return a * (x/500.0)**(b) + c

    def fit_diff_ratio(wavelength, ratio):

        popt, pcov = curve_fit(func_diff_ratio, wavelength, ratio, maxfev=1000000, bounds=(np.array([0.0, -np.inf, 0.0]), np.array([np.inf, 0.0, np.inf])))

        return popt, pcov

    date_s = date.strftime('%Y%m%d')

    if run:

        data_ssfr_v1 = ssfr.util.load_h5(fname_ssfr_v1)

        # temporary fix to bypass the attitude correction for SSFR-B
        #╭────────────────────────────────────────────────────────────────────────────╮#
        if data_ssfr_v1['zen/wvl'].size > 424:
            data_ssfr_v1['zen/toa0'] = data_ssfr_v1['zen/toa0'][:424]
            data_ssfr_v1['zen/wvl'] = data_ssfr_v1['zen/wvl'][:424]
            data_ssfr_v1['zen/flux'] = data_ssfr_v1['zen/flux'][:, :424]
            data_ssfr_v1['zen/cnt'] = data_ssfr_v1['zen/cnt'][:, :424]
            data_ssfr_v1['v0/spec_zen'] = data_ssfr_v1['v0/spec_zen'][:, :424]
            data_ssfr_v1['v0/wvl_zen'] = data_ssfr_v1['v0/wvl_zen'][:424]
        # ╰────────────────────────────────────────────────────────────────────────────╯#

        fname_aux = fname_h5.replace('_v2.h5', '-aux_v2.h5')

        if run_aux:

            # calculate diffuse/global ratio from HSR1 data
            #╭────────────────────────────────────────────────────────────────────────────╮#
            data_hsr1_v2 = ssfr.util.load_h5(fname_hsr1_v2)

            f_ = h5py.File(fname_aux, 'w')

            wvl_ssfr_zen = data_ssfr_v1['zen/wvl']
            wvl_hsr1     = data_hsr1_v2['tot/wvl']

            Nt, Nwvl = data_ssfr_v1['zen/flux'].shape

            diff_ratio = np.zeros((Nt, Nwvl), dtype=np.float64)
            diff_ratio[...] = np.nan

            poly_coefs = np.zeros((Nt, 3), dtype=np.float64)
            poly_coefs[...] = np.nan

            qual_flag = np.repeat(0, Nt)

            # do spectral fit based on 400 nm - 750 nm observations
            #╭──────────────────────────────────────────────────────────────╮#
            for i in tqdm(range(Nt)):

                diff_ratio0_hsr1 = data_hsr1_v2['dif/flux'][i, :] / data_hsr1_v2['tot/flux'][i, :]
                logic_valid = (~np.isnan(diff_ratio0_hsr1)) & (diff_ratio0_hsr1>=0.0) & (diff_ratio0_hsr1<=1.0) & (wvl_hsr1>=400.0) & (wvl_hsr1<=750.0)
                if logic_valid.sum() > 20:

                    x = data_hsr1_v2['tot/wvl'][logic_valid]
                    y = diff_ratio0_hsr1[logic_valid]
                    popt, pcov = fit_diff_ratio(x, y)

                    diff_ratio[i, :] = func_diff_ratio(wvl_ssfr_zen, *popt)
                    poly_coefs[i, :] = popt

                    qual_flag[i] = 1

            diff_ratio[diff_ratio<0.0] = 0.0
            diff_ratio[diff_ratio>1.0] = 1.0
            #╰──────────────────────────────────────────────────────────────╯#

            # fill in nan values in time space
            #╭──────────────────────────────────────────────────────────────╮#
            for i in range(Nwvl):

                logic_nan   = np.isnan(diff_ratio[:, i])
                logic_valid = ~logic_nan
                f_interp = interpolate.interp1d(data_ssfr_v1['tmhr'][logic_valid], diff_ratio[:, i][logic_valid], bounds_error=None, fill_value='extrapolate')
                diff_ratio[logic_nan, i] = f_interp(data_ssfr_v1['tmhr'][logic_nan])

            diff_ratio[diff_ratio<0.0] = 0.0
            diff_ratio[diff_ratio>1.0] = 1.0
            #╰──────────────────────────────────────────────────────────────╯#

            # save data
            #╭──────────────────────────────────────────────────────────────╮#
            f_.create_dataset('diff_ratio', data=diff_ratio  , compression='gzip', compression_opts=9, chunks=True)
            g_ = f_.create_group('diff_ratio_aux')
            g_.create_dataset('wvl'       , data=wvl_ssfr_zen, compression='gzip', compression_opts=9, chunks=True)
            g_.create_dataset('coef'      , data=poly_coefs  , compression='gzip', compression_opts=9, chunks=True)
            g_.create_dataset('qual_flag' , data=qual_flag   , compression='gzip', compression_opts=9, chunks=True)
            #╰──────────────────────────────────────────────────────────────╯#
            #╰────────────────────────────────────────────────────────────────────────────╯#


            # alp
            #╭────────────────────────────────────────────────────────────────────────────╮#
            data_alp_v1  = ssfr.util.load_h5(fname_alp_v1)
            for key in data_alp_v1.keys():
                try:
                    f_.create_dataset(key, data=data_alp_v1[key], compression='gzip', compression_opts=9, chunks=True)
                except TypeError as error:
                    print(error)
                    f_[key] = data_alp_v1[key]
            f_.close()
            #╰────────────────────────────────────────────────────────────────────────────╯#


        data_aux = ssfr.util.load_h5(fname_aux)

        # diffuse ratio
        #╭────────────────────────────────────────────────────────────────────────────╮#
        diff_ratio = data_aux['diff_ratio']
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # angles
        #╭────────────────────────────────────────────────────────────────────────────╮#
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
        #╰────────────────────────────────────────────────────────────────────────────╯#


        # select calibration file for attitude correction
        # angular response is relative change, thus irrelavant to integration time (ideally)
        # and is intrinsic property of light collector, thus fixed to use SSFR-A with larger
        # integration time for consistency and simplicity, will revisit this after mission
        #╭────────────────────────────────────────────────────────────────────────────╮#
        dset_s = 'dset1'
        fdir_cal = '%s/ang-cal' % cfg.fdir_cal #_FDIR_CAL_
        fname_cal_zen = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*vaa-180|*%s*%s*zen*' % (dset_s, 'ssfr-a')), key=os.path.getmtime)[-1]
        fname_cal_nad = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*vaa-180|*%s*%s*nad*' % (dset_s, 'ssfr-a')), key=os.path.getmtime)[-1]
        #╰────────────────────────────────────────────────────────────────────────────╯#


        # calculate attitude correction factors
        #╭────────────────────────────────────────────────────────────────────────────╮#
        fnames_cal = {
                'zen': fname_cal_zen,
                'nad': fname_cal_nad,
                }
        factors = ssfr.corr.att_corr(fnames_cal, angles, diff_ratio=diff_ratio)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # save data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        f = h5py.File(fname_h5, 'w')
        for key in ['tmhr', 'jday', 'lon', 'lat', 'alt']:
            f.create_dataset(key, data=data_aux[key], compression='gzip', compression_opts=9, chunks=True)

        g1 = f.create_group('att_corr')
        g1.create_dataset('factors_zen', data=factors['zen'], compression='gzip', compression_opts=9, chunks=True)
        g1.create_dataset('factors_nad', data=factors['nad'], compression='gzip', compression_opts=9, chunks=True)
        for key in ['sza', 'saa', 'ang_pit_s', 'ang_rol_s', 'ang_hed', 'ang_pit_m', 'ang_rol_m']:
            g1.create_dataset(key, data=data_aux[key], compression='gzip', compression_opts=9, chunks=True)

        # apply attitude correction
        #╭──────────────────────────────────────────────────────────────╮#
        g2 = f.create_group('zen')
        g2.create_dataset('flux', data=data_ssfr_v1['zen/flux']*factors['zen'], compression='gzip', compression_opts=9, chunks=True)
        g2.create_dataset('wvl' , data=data_ssfr_v1['zen/wvl']                , compression='gzip', compression_opts=9, chunks=True)
        g2.create_dataset('toa0', data=data_ssfr_v1['zen/toa0']               , compression='gzip', compression_opts=9, chunks=True)

        g3 = f.create_group('nad')
        g3.create_dataset('flux', data=data_ssfr_v1['nad/flux']*factors['nad'], compression='gzip', compression_opts=9, chunks=True)
        g3.create_dataset('wvl' , data=data_ssfr_v1['nad/wvl']                , compression='gzip', compression_opts=9, chunks=True)
        #╰──────────────────────────────────────────────────────────────╯#

        f.close()
        #╰────────────────────────────────────────────────────────────────────────────╯#

    return fname_h5

def cdata_ssfr_archive(
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
        fdir_out='./',
        run=True,
        ):


    # placeholder for additional information such as calibration
    #╭────────────────────────────────────────────────────────────────────────────╮#
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
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # date info
    #╭────────────────────────────────────────────────────────────────────────────╮#
    date_s = date.strftime('%Y%m%d')
    date_today = datetime.date.today()
    date_info  = '%4.4d, %2.2d, %2.2d, %4.4d, %2.2d, %2.2d' % (date.year, date.month, date.day, date_today.year, date_today.month, date_today.day)

    which_ssfr = os.path.basename(fname_ssfr_v2).split('_')[0].replace('%s-' % _MISSION_.upper(), '').lower()
    if which_ssfr == _SSFR1_:
        instrument_info = 'SSFR-A (Solar Spectral Flux Radiometer - Alvin)'
    else:
        instrument_info = 'SSFR-B (Solar Spectral Flux Radiometer - Belana)'
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # version info
    #╭────────────────────────────────────────────────────────────────────────────╮#
    version = version.upper()
    version_info = {
            'RA': 'field data',
            }
    version_info = version_info[version]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # data info
    #╭────────────────────────────────────────────────────────────────────────────╮#
    data_info = 'Shortwave Total Downwelling and Upwelling Spectral Irradiance from %s %s' % (platform_info.upper(), instrument_info)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # routine comments
    #╭────────────────────────────────────────────────────────────────────────────╮#
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
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # special comments
    #╭────────────────────────────────────────────────────────────────────────────╮#
    comments_special_dict = {
            '20240530': 'Noticed icing on dome outside zenith light collector after flight',
            '20240531': 'Encountered temperature control issue (after around 1:30 UTC)',
            '20240730': 'Noticed icing on dome inside zenith light collector after flight',
            '20240801': 'Noticed condensation on dome inside zenith light collector before flight',
            '20240807': 'Noticed condensation on dome inside zenith light collector after flight',
            '20240808': 'Noticed condensation on dome inside zenith light collector after flight',
            '20240809': 'Noticed condensation on dome inside zenith light collector after flight',
            }
    if date_s in comments_special_dict.keys():
        comments_special = comments_special_dict[date_s]
    else:
        comments_special = ''

    if comments_special != '':
        Nspecial = len(comments_special.split('\n'))
    else:
        Nspecial = 0
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # data processing
    #╭────────────────────────────────────────────────────────────────────────────╮#
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
    #╰────────────────────────────────────────────────────────────────────────────╯#


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

    fname_h5 = '%s/%s-SSFR_%s_%s_%s.h5' % (fdir_out, _MISSION_.upper(), _PLATFORM_.upper(), date_s, version.upper())
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
#╰────────────────────────────────────────────────────────────────────────────╯#


# additional functions under development
#╭────────────────────────────────────────────────────────────────────────────╮#
def run_time_offset_check(cfg):

    main_process_data_v0(cfg, run=False)

    date = cfg.common['date']
    date_s = date.strftime('%Y%m%d')
    data_hsr1_v0 = ssfr.util.load_h5(cfg.hsr1['fname_v0'])
    data_ssfr_v0 = ssfr.util.load_h5(cfg.ssfr['fname_v0'])

    # data_hsr1_v0['tot/jday'] += 1.0
    # data_hsr1_v0['dif/jday'] += 1.0

    # _offset_x_range_ = [-6000.0, 6000.0]
    _offset_x_range_ = [-600.0, 600.0]


    # SSFR vs HSR1
    #╭────────────────────────────────────────────────────────────────────────────╮#
    index_wvl_hsr1 = np.argmin(np.abs(745.0-data_hsr1_v0['tot/wvl']))
    data_y0 = data_hsr1_v0['tot/flux'][:, index_wvl_hsr1]

    index_wvl_ssfr = np.argmin(np.abs(745.0-data_ssfr_v0['spec/wvl_zen']))
    data_y1 = data_ssfr_v0['spec/cnt_zen'][:, index_wvl_ssfr]
    data_offset = {
            'x0': data_hsr1_v0['tot/jday']*86400.0,
            'y0': data_y0,
            'x1': data_ssfr_v0['raw/jday']*86400.0,
            'y1': data_y1,
            }
    ssfr.vis.find_offset_bokeh(
            data_offset,
            offset_x_range=_offset_x_range_,
            offset_y_range=[-10, 10],
            x_reset=True,
            y_reset=True,
            description='SSFR Zenith Count vs. HSR1 Total (745nm)',
            fname_html='ssfr_offset_check_%s.html' % (date_s))
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return

def run_angle_offset_check(
        date,
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        wvl0=745.0,
        ):

    date_s = date.strftime('%Y%m%d')
    data_hsk = ssfr.util.load_h5(_FNAMES_['%s_hsk_v0' % date_s])


    # HSR1 v1
    #╭────────────────────────────────────────────────────────────────────────────╮#
    data_hsr1_v1 = ssfr.util.load_h5(_FNAMES_['%s_hsr1_v1' % date_s])
    index_wvl_hsr1 = np.argmin(np.abs(wvl0-data_hsr1_v1['tot/wvl']))
    data_y1 = data_hsr1_v1['tot/flux'][:, index_wvl_hsr1]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # HSR1 v2
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsr1_v2 = cdata_hsr1_v2(date, _FNAMES_['%s_hsr1_v1' % date_s], _FNAMES_['%s_hsk_v0' % date_s],
            fdir_out='./',
            run=True,
            ang_pit_offset=ang_pit_offset,
            ang_rol_offset=ang_rol_offset,
            )
    data_hsr1_v2 = ssfr.util.load_h5(_FNAMES_['%s_hsr1_v2' % date_s])
    data_y2 = data_hsr1_v2['tot/flux'][:, index_wvl_hsr1]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # SSFR-A v2
    #╭────────────────────────────────────────────────────────────────────────────╮#
    data_ssfr1_v2 = ssfr.util.load_h5(_FNAMES_['%s_ssfr1_v2' % date_s])
    index_wvl_ssfr = np.argmin(np.abs(wvl0-data_ssfr1_v2['zen/wvl']))
    data_y0 = data_ssfr1_v2['zen/flux'][:, index_wvl_ssfr]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(15, 6))
        # fig.suptitle('Figure')
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
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
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        # fig.subplots_adjust(hspace=0.3, wspace=0.3)
        # _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        # fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
    #╰────────────────────────────────────────────────────────────────────────────╯#
    sys.exit()

def dark_corr_temp(date, iChan=100, idset=0):

    date_s = date.strftime('%Y%m%d')
    data_ssfr1_v0 = ssfr.util.load_h5(_FNAMES_['%s_ssfr1_v0' % date_s])

    tmhr = data_ssfr1_v0['raw/tmhr']
    x_temp_zen = data_ssfr1_v0['raw/temp'][:, 1]
    x_temp_nad = data_ssfr1_v0['raw/temp'][:, 2]
    shutter = data_ssfr1_v0['raw/shutter_dark-corr']
    dset_num = data_ssfr1_v0['raw/dset_num']

    logic_dark = (shutter==1) & (dset_num==idset)
    logic_light = (shutter==0) & (dset_num==idset)

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:

        plt.close('all')
        fig = plt.figure(figsize=(13, 19))
        fig.suptitle('Channel #%d' % iChan)
        # plot
        #╭──────────────────────────────────────────────────────────────╮#
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
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.4)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_dset%d_%3.3d.png' % (_metadata['Function'], idset, iChan), bbox_inches='tight', metadata=_metadata, dpi=150)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
    #╰────────────────────────────────────────────────────────────────────────────╯#
#╰────────────────────────────────────────────────────────────────────────────╯#


# main program
#╭────────────────────────────────────────────────────────────────────────────╮#
def main_process_data_v0(cfg, run=True):

    date = cfg.common['date']
    date_s = cfg.common['date_s']

    fdir_out = cfg.common['fdir_out']
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    # SSFR v0: raw data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fnames_ssfr = cfg.ssfr['fnames']
    fname_h5 = cfg.ssfr['fname_v0']
    if run and (len(fnames_ssfr) == 0):
        pass
    else:
        fname_ssfr_v0 = cdata_ssfr_v0(
                date,
                fnames_ssfr,
                fname_h5=fname_h5,
                which_ssfr=cfg.ssfr['which_ssfr'],
                wvl_s=cfg.ssfr['wvl_s'],
                wvl_e=cfg.ssfr['wvl_e'],
                wvl_j=cfg.ssfr['wvl_j'],
                dark_extend=cfg.ssfr['dark_extend'],
                light_extend=cfg.ssfr['light_extend'],
                dark_corr_mode=cfg.ssfr['dark_corr_mode'],
                fdir_out=fdir_out,
                run=run
                )
    #╰────────────────────────────────────────────────────────────────────────────╯#

def main_process_data_v1(cfg, run=True):

    date = cfg.common['date']
    date_s = cfg.common['date_s']

    fdir_out = cfg.common['fdir_out']
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    # SSFR v1: time synced with hsk time with time offset applied
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_h5 = cfg.ssfr['fname_v1']

    fname_ssfr_v1 = cdata_ssfr_v1(
            date,
            cfg.ssfr['fname_v0'],
            cfg.hsk['fname_v0'],
            fname_h5=fname_h5,
            time_offset=cfg.ssfr['time_offset'],
            which_ssfr=cfg.ssfr['which_ssfr'],
            which_ssfr_for_flux=cfg.ssfr['which_ssfr'],
            fdir_out=fdir_out,
            run=run
            )
    #╰────────────────────────────────────────────────────────────────────────────╯#

def main_process_data_v2(cfg, run=True):

    """
    v0: raw data directly read out from the data files
    v1: data collocated/synced to aircraft nav
    v2: attitude corrected data
    """

    date = cfg.common['date']
    date_s = cfg.common['date_s']

    fdir_out = './'
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    # SSFR v2
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_h5 = cfg.ssfr['fname_v2']

    fname_ssfr_v2 = cdata_ssfr_v2(
            date,
            cfg.ssfr['fname_v1'],
            cfg.alp['fname_v1'],
            cfg.hsr1['fname_v2'],
            fname_h5=fname_h5,
            fdir_out=fdir_out,
            run=run,
            run_aux=True
            )
    #╰────────────────────────────────────────────────────────────────────────────╯#
#╰────────────────────────────────────────────────────────────────────────────╯#


if __name__ == '__main__':


    # dates
    #╭────────────────────────────────────────────────────────────────────────────╮#
    dates = [
             datetime.datetime(2024, 5, 24), #
            #  datetime.datetime(2024, 5, 28), # ARCSIX-1 science flight #1
            #  datetime.datetime(2024, 5, 30), # ARCSIX-1 science flight #2, cloud wall, operator - Vikas Nataraja
            #  datetime.datetime(2024, 5, 31), # ARCSIX-1 science flight #3, bowling alley; surface BRDF, operator - Vikas Nataraja
            #  datetime.datetime(2024, 6, 3),  # ARCSIX-1 science flight #4, cloud wall, operator - Vikas Nataraja
            #  datetime.datetime(2024, 6, 5),  # ARCSIX-1 science flight #5
            #  datetime.datetime(2024, 6, 6),  # ARCSIX-1 science flight #6
            #  datetime.datetime(2024, 6, 7),  # ARCSIX-1 science flight #7, cloud wall, operator - Vikas Nataraja, Arabella Chamberlain
            #  datetime.datetime(2024, 6, 10), # ARCSIX-1 science flight #8, operator - Jeffery Drouet
            #  datetime.datetime(2024, 6, 11), # ARCSIX-1 science flight #9, operator - Arabella Chamberlain, Sebastian Becker
            #  datetime.datetime(2024, 6, 13), # ARCSIX-1 science flight #10, operator - Arabella Chamberlain
            #  datetime.datetime(2024, 7, 22), #
            #  datetime.datetime(2024, 7, 25), # ARCSIX-2 science flight #11, cloud walls, operator - Arabella Chamberlain
            #  datetime.datetime(2024, 7, 29), # ARCSIX-2 science flight #12, clear-sky BRDF, operator - Ken Hirata, Vikas Nataraja
            #  datetime.datetime(2024, 7, 30), # ARCSIX-2 science flight #13, clear-sky BRDF, operator - Ken Hirata
            #  datetime.datetime(2024, 8, 1),  # ARCSIX-2 science flight #14, cloud walls, operator - Ken Hirata
            #  datetime.datetime(2024, 8, 2),  # ARCSIX-2 science flight #15, cloud walls, operator - Ken Hirata, Arabella Chamberlain
            #  datetime.datetime(2024, 8, 7),  # ARCSIX-2 science flight #16, cloud walls, operator - Arabella Chamberlain
            #  datetime.datetime(2024, 8, 8),  # ARCSIX-2 science flight #17, cloud walls, operator - Arabella Chamberlain
            #  datetime.datetime(2024, 8, 9),  # ARCSIX-2 science flight #18, cloud walls, operator - Arabella Chamberlain
            #  datetime.datetime(2024, 8, 15), # ARCSIX-2 science flight #19, cloud walls, operator - Ken Hirata, Sebastian Schmidt
            #  datetime.datetime(2024, 8, 16), # 
            ]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    for date in dates[::-1]:

        # configuration file
        #╭────────────────────────────────────────────────────────────────────────────╮#
        global cfg
        fname_cfg = date.strftime('cfg_%Y%m%d')
        cfg = importlib.import_module(fname_cfg)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # step 1
        # process raw data (text, binary etc.) into HDF5 file
        #╭────────────────────────────────────────────────────────────────────────────╮#
        main_process_data_v0(cfg, run=True)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # step 2
        # create bokeh interactive plots to retrieve time offset
        #╭────────────────────────────────────────────────────────────────────────────╮#
        run_time_offset_check(cfg)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # step 3
        # apply time offsets to sync data to aircraft housekeeping file
        #╭────────────────────────────────────────────────────────────────────────────╮#
        main_process_data_v1(cfg, run=True)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # step 4
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # main_process_data_v2(cfg, run=True)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        pass
