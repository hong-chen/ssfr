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


# parameters
#╭────────────────────────────────────────────────────────────────────────────╮#
_FNAMES_ = {}
#╰────────────────────────────────────────────────────────────────────────────╯#


_ALP_TIME_OFFSET_ = {
        '20240517':   5.55,
        '20240521': -17.94,
        '20240524': -18.39,
        '20240528': -17.19,
        '20240530': -17.41,
        '20240531': -17.41,
        '20240603': -17.41,
        '20240605': -17.58,
        '20240606': -18.08,
        '20240607': -17.45,
        '20240610': -17.45,
        '20240611': -17.52,
        '20240613': -17.85,
        '20240708': -17.85,
        '20240709': -17.85,
        '20240722': -17.85,
        '20240724': -17.85,
        '20240725': -17.89,
        '20240726': -17.89,
        '20240729': -18.22,
        '20240730': -17.43,
        '20240801': -17.74,
        '20240802': -17.97,
        '20240807': -17.67,
        '20240808': -18.04,
        '20240809': -18.01,
        '20240815': -18.10,
        '20240816': -18.10,
        }
_HSR1_TIME_OFFSET_ = {
        '20240517': 0.0,
        '20240521': 0.0,
        '20240524': 86400.0,
        '20240528': 0.0,
        '20240530': 0.0,
        '20240531': 0.0,
        '20240603': 0.0,
        '20240605': 0.0,
        '20240606': 0.0,
        '20240607': 0.0,
        '20240610': 0.0,
        '20240611': 0.0,
        '20240613': 0.0,
        '20240708': 0.0,
        '20240709': 0.0,
        '20240722': 0.0,
        '20240724': 0.0,
        '20240725': 0.0,
        '20240726': 0.0,
        '20240729': 0.0,
        '20240730': 0.0,
        '20240801': 0.0,
        '20240802': 0.0,
        '20240807': 0.0,
        '20240808': 0.0,
        '20240809': 0.0,
        '20240815': 0.0,
        '20240816': 0.0,
        }

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


# functions for processing HSK and ALP
#╭────────────────────────────────────────────────────────────────────────────╮#
def cdata_hsk_v0(
        date,
        fname_hsk,
        fname_h5='HSK_v0.h5',
        fdir_out='./',
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

    if run:

        # ict file from P3 data system team, best quality but cannot be accessed immediately
        #╭────────────────────────────────────────────────────────────────────────────╮#
        data_hsk = ssfr.util.read_ict(fname_hsk)
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
        #╰────────────────────────────────────────────────────────────────────────────╯#

        print()
        print('Processing HSK file:', fname_hsk)
        print()

        # solar geometries
        #╭────────────────────────────────────────────────────────────────────────────╮#
        jday0 = ssfr.util.dtime_to_jday(date)
        jday  = jday0 + data_hsk[var_dict['tmhr']]['data']/24.0
        sza, saa = ssfr.util.cal_solar_angles(jday, data_hsk[var_dict['lon']]['data'], data_hsk[var_dict['lat']]['data'], data_hsk[var_dict['alt']]['data'])
        #╰────────────────────────────────────────────────────────────────────────────╯#


        # save processed data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        f = h5py.File(fname_h5, 'w')
        for var in var_dict.keys():
            f[var] = data_hsk[var_dict[var]]['data']
        f['jday'] = jday
        f['sza']  = sza
        f['saa']  = saa
        f.close()
        #╰────────────────────────────────────────────────────────────────────────────╯#

    return fname_h5

def cdata_hsk_from_alp_v0(
        date,
        fname_alp_v0,
        fname_h5='HSK_v0.h5',
        fdir_out='./',
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

    if run:

        data_alp = ssfr.util.load_h5(fname_alp_v0)

        tmhr_ = data_alp['tmhr'][(data_alp['tmhr']>=0.0) & (data_alp['tmhr']<=48.0)]
        seconds_s = np.round(np.quantile(tmhr_, 0.02)*3600.0, decimals=0)
        seconds_e = np.round(np.quantile(tmhr_, 0.98)*3600.0, decimals=0)
        tmhr = (np.arange(seconds_s, seconds_e+1.0, 1.0) + _ALP_TIME_OFFSET_[date_s]) / 3600.0

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
                    'data': ssfr.util.interp(jday, data_alp['jday']+_ALP_TIME_OFFSET_[date_s]/86400.0, data_alp[var_dict[vname]], mode='linear')
                    }

        # empirical offset angles between ALP and HSK
        #╭────────────────────────────────────────────────────────────────────────────╮#
        data_hsk['ang_pit']['data'] += 3.85
        data_hsk['ang_rol']['data'] += 0.45
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # solar geometries
        #╭────────────────────────────────────────────────────────────────────────────╮#
        sza, saa = ssfr.util.cal_solar_angles(jday, data_hsk['lon']['data'], data_hsk['lat']['data'], data_hsk['alt']['data'])
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # save processed data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        f = h5py.File(fname_h5, 'w')
        for var in data_hsk.keys():
            f[var] = data_hsk[var]['data']
        f['jday'] = jday
        f['sza']  = sza
        f['saa']  = saa
        f.close()
        #╰────────────────────────────────────────────────────────────────────────────╯#

    return fname_h5

def cdata_alp_v0(
        date,
        fnames_alp,
        fname_h5='ALP_v0.h5',
        fdir_out='./',
        run=True,
        ):

    """
    v0: directly read raw ALP (Active Leveling Platform) data

    Notes:
        ALP raw data has a finer temporal resolution than 1Hz and a higher measurement
        precision (or sensitivity) of the aircraft attitude.
    """


    # read ALP raw data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if run:
        alp0 = ssfr.lasp_alp.read_alp(fnames_alp, date=date)
        alp0.save_h5(fname_h5)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return os.path.abspath(fname_h5)

def cdata_alp_v1(
        date,
        fname_v0,
        fname_hsk,
        fname_h5='ALP_v1.h5',
        time_offset=0.0,
        fdir_out='./',
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

    if run:

        data_hsk = ssfr.util.load_h5(fname_hsk)
        data_alp = ssfr.util.load_h5(fname_v0)

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
#╰────────────────────────────────────────────────────────────────────────────╯#


# additional functions under development
#╭────────────────────────────────────────────────────────────────────────────╮#
def run_time_offset_check(cfg):

    main_process_data_v0(cfg, run=False)

    date = cfg.common['date']
    date_s = date.strftime('%Y%m%d')
    data_hsk = ssfr.util.load_h5(_FNAMES_['%s_hsk_v0' % date_s])
    data_alp = ssfr.util.load_h5(_FNAMES_['%s_alp_v0' % date_s])
    data_hsr1_v0 = ssfr.util.load_h5(_FNAMES_['%s_hsr1_v0' % date_s])
    data_ssfr_v0 = ssfr.util.load_h5(_FNAMES_['%s_ssfr_v0' % date_s])
    data_ssrr_v0 = ssfr.util.load_h5(_FNAMES_['%s_ssrr_v0' % date_s])

    # data_hsr1_v0['tot/jday'] += 1.0
    # data_hsr1_v0['dif/jday'] += 1.0

    # _offset_x_range_ = [-6000.0, 6000.0]
    _offset_x_range_ = [-600.0, 600.0]

    # ALP pitch vs HSK pitch
    #╭────────────────────────────────────────────────────────────────────────────╮#
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
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # ALP roll vs HSK roll
    #╭────────────────────────────────────────────────────────────────────────────╮#
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
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # ALP altitude vs HSK altitude
    #╭────────────────────────────────────────────────────────────────────────────╮#
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
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # HSR1 vs TOA
    #╭────────────────────────────────────────────────────────────────────────────╮#
    index_wvl = np.argmin(np.abs(745.0-data_hsr1_v0['tot/wvl']))
    data_y1   = data_hsr1_v0['tot/flux'][:, index_wvl]

    mu = np.cos(np.deg2rad(data_hsk['sza']))
    iza, iaa = ssfr.util.prh2za(data_hsk['ang_pit'], data_hsk['ang_rol'], data_hsk['ang_hed'])
    dc = ssfr.util.muslope(data_hsk['sza'], data_hsk['saa'], iza, iaa)
    factors = mu/dc
    data_y0   = data_hsr1_v0['tot/toa0'][index_wvl]*np.cos(np.deg2rad(data_hsk['sza']))/factors

    data_offset = {
            'x0': data_hsk['jday']*86400.0,
            'y0': data_y0,
            'x1': data_hsr1_v0['tot/jday']*86400.0,
            'y1': data_y1,
            }
    ssfr.vis.find_offset_bokeh(
            data_offset,
            offset_x_range=_offset_x_range_,
            offset_y_range=[-10, 10],
            x_reset=True,
            y_reset=True,
            description='HSR1 Total vs. TOA (745 nm)',
            fname_html='hsr1-toa_offset_check_%s.html' % date_s)
    #╰────────────────────────────────────────────────────────────────────────────╯#


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


    # SSRR vs SSFR
    #╭────────────────────────────────────────────────────────────────────────────╮#
    index_wvl_ssfr = np.argmin(np.abs(745.0-data_ssfr_v0['spec/wvl_nad']))
    data_y0 = data_ssfr_v0['spec/cnt_nad'][:, index_wvl_ssfr]

    index_wvl_ssfr = np.argmin(np.abs(745.0-data_ssrr_v0['spec/wvl_nad']))
    data_y1 = data_ssrr_v0['spec/cnt_nad'][:, index_wvl_ssfr]
    data_offset = {
            'x0': data_ssfr_v0['raw/jday']*86400.0,
            'y0': data_y0,
            'x1': data_ssrr_v0['raw/jday']*86400.0,
            'y1': data_y1,
            }
    ssfr.vis.find_offset_bokeh(
            data_offset,
            offset_x_range=_offset_x_range_,
            offset_y_range=[-10, 10],
            x_reset=True,
            y_reset=True,
            description='SSRR Nadir Count vs. SSFR Nadir (745nm)',
            fname_html='ssrr_offset_check_%s.html' % (date_s))
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

    # ALP v0: raw data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fnames_alp = cfg.alp['fnames']
    fname_h5 = '%s/%s-%s_%s_%s_v0.h5' % (fdir_out, cfg.common['mission'].upper(), cfg.alp['aka'].upper(), cfg.common['platform'].upper(), date_s)
    if run and (len(fnames_alp)==0):
        pass
    else:
        fname_alp_v0 = cdata_alp_v0(
                date,
                fnames_alp,
                fname_h5=fname_h5,
                fdir_out=fdir_out,
                run=run
                )
        _FNAMES_['%s_alp_v0' % date_s]   = fname_alp_v0
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # HSK v0: raw data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsk = cfg.hsk['fname']
    fname_h5 = '%s/%s-%s_%s_%s_v0.h5' % (fdir_out, cfg.common['mission'].upper(), cfg.hsk['aka'].upper(), cfg.common['platform'].upper(), date_s)
    if run and (fname_hsk is None):
        # * not preferred, use ALP lon/lat if P3 housekeeping file is not available (e.g., for immediate data processing after flight)
        fname_hsk_v0 = cdata_hsk_from_alp_v0(
                date,
                _FNAMES_['%s_alp_v0' % date_s],
                fname_h5=fname_h5,
                fdir_out=fdir_out,
                run=run
                )
    else:
        # * preferred, use P3 housekeeping file, ict > iwg > mts
        fname_hsk_v0 = cdata_hsk_v0(
                date,
                fname_hsk,
                fname_h5=fname_h5,
                fdir_out=fdir_out,
                run=run
                )

    _FNAMES_['%s_hsk_v0' % date_s] = fname_hsk_v0
    #╰────────────────────────────────────────────────────────────────────────────╯#

def main_process_data_v1(cfg, run=True):

    date = cfg.common['date']
    date_s = cfg.common['date_s']

    fdir_out = cfg.common['fdir_out']
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    main_process_data_v0(cfg, run=False)

    # # ALP v1: time synced with hsk time with time offset applied
    # #╭────────────────────────────────────────────────────────────────────────────╮#
    # fname_h5 = '%s/%s-%s_%s_%s_v1.h5' % (fdir_out, cfg.common['mission'].upper(), cfg.alp['aka'].upper(), cfg.common['platform'].upper(), date_s)

    # fname_alp_v1 = cdata_alp_v1(
    #         date,
    #         _FNAMES_['%s_alp_v0' % date_s],
    #         _FNAMES_['%s_hsk_v0' % date_s],
    #         fname_h5=fname_h5,
    #         time_offset=cfg.alp['time_offset'],
    #         fdir_out=fdir_out,
    #         run=run
    #         )

    # _FNAMES_['%s_alp_v1' % date_s] = fname_alp_v1
    # #╰────────────────────────────────────────────────────────────────────────────╯#


    # # HSR1 v1: time synced with hsk time with time offset applied
    # #╭────────────────────────────────────────────────────────────────────────────╮#
    # fname_h5 = '%s/%s-%s_%s_%s_v1.h5' % (fdir_out, cfg.common['mission'].upper(), cfg.hsr1['aka'].upper(), cfg.common['platform'].upper(), date_s)

    # fname_hsr1_v1 = cdata_hsr1_v1(
    #         date,
    #         _FNAMES_['%s_hsr1_v0' % date_s],
    #         _FNAMES_['%s_hsk_v0' % date_s],
    #         fname_h5=fname_h5,
    #         time_offset=cfg.hsr1['time_offset'],
    #         fdir_out=fdir_out,
    #         run=run
    #         )

    # _FNAMES_['%s_hsr1_v1' % date_s] = fname_hsr1_v1
    # #╰────────────────────────────────────────────────────────────────────────────╯#


    # SSFR v1: time synced with hsk time with time offset applied
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_h5 = '%s/%s-%s_%s_%s_v1.h5' % (fdir_out, cfg.common['mission'].upper(), cfg.ssfr['aka'].upper(), cfg.common['platform'].upper(), date_s)

    fname_ssfr_v1 = cdata_ssfr_v1(
            date,
            _FNAMES_['%s_ssfr_v0' % date_s],
            _FNAMES_['%s_hsk_v0' % date_s],
            fname_h5=fname_h5,
            time_offset=cfg.ssfr['time_offset'],
            which_ssfr=cfg.ssfr['which_ssfr'],
            which_ssfr_for_flux=cfg.ssfr['which_ssfr'],
            fdir_out=fdir_out,
            run=run
            )

    _FNAMES_['%s_ssfr_v1' % date_s] = fname_ssfr_v1
    #╰────────────────────────────────────────────────────────────────────────────╯#
    # sys.exit()


    # SSRR v1: time synced with hsk time with time offset applied
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_h5 = '%s/%s-%s_%s_%s_v1.h5' % (fdir_out, cfg.common['mission'].upper(), cfg.ssrr['aka'].upper(), cfg.common['platform'].upper(), date_s)

    fname_ssrr_v1 = cdata_ssfr_v1(
            date,
            _FNAMES_['%s_ssrr_v0' % date_s],
            _FNAMES_['%s_hsk_v0' % date_s],
            fname_h5=fname_h5,
            time_offset=cfg.ssrr['time_offset'],
            which_ssfr=cfg.ssrr['which_ssfr'],
            which_ssfr_for_flux=cfg.ssfr['which_ssfr'],
            fdir_out=fdir_out,
            run=run
            )

    _FNAMES_['%s_ssrr_v1' % date_s] = fname_ssrr_v1
    #╰────────────────────────────────────────────────────────────────────────────╯#
    # sys.exit()

def main_process_data_v2(date, run=True):

    """
    v0: raw data directly read out from the data files
    v1: data collocated/synced to aircraft nav
    v2: attitude corrected data
    """

    date_s = date.strftime('%Y%m%d')

    fdir_out = './'
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    main_process_data_v0(cfg, run=False)
    main_process_data_v1(cfg, run=False)

    # HSR1 v2
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # * based on ALP v1
    # fname_hsr1_v2 = cdata_hsr1_v2(date, _FNAMES_['%s_hsr1_v1' % date_s], _FNAMES_['%s_alp_v1' % date_s],
    #         fdir_out=fdir_out, run=run)
    # fname_hsr1_v2 = cdata_hsr1_v2(date, _FNAMES_['%s_hsr1_v1' % date_s], _FNAMES_['%s_alp_v1' % date_s],
    #         fdir_out=fdir_out, run=True)

    # * based on HSK v0
    fname_hsr1_v2 = cdata_hsr1_v2(date, _FNAMES_['%s_hsr1_v1' % date_s], _FNAMES_['%s_hsk_v0' % date_s],
            fdir_out=fdir_out, run=run)
    #╰────────────────────────────────────────────────────────────────────────────╯#
    _FNAMES_['%s_hsr1_v2' % date_s] = fname_hsr1_v2


    # SSFR v2
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if _WHICH_SSFR_ == _SSFR1_:
        _vname_ssfr_v1_ = '%s_ssfr1_v1' % date_s
        _vname_ssfr_v2_ = '%s_ssfr1_v2' % date_s
    else:
        _vname_ssfr_v1_ = '%s_ssfr2_v1' % date_s
        _vname_ssfr_v2_ = '%s_ssfr2_v2' % date_s

    fname_ssfr_v2 = cdata_ssfr_v2(date, _FNAMES_[_vname_ssfr_v1_], _FNAMES_['%s_alp_v1' % date_s], _FNAMES_['%s_hsr1_v2' % date_s],
            fdir_out=fdir_out, run=run, run_aux=True)
    #╰────────────────────────────────────────────────────────────────────────────╯#
    _FNAMES_[_vname_ssfr_v2_] = fname_ssfr_v2
#╰────────────────────────────────────────────────────────────────────────────╯#


if __name__ == '__main__':


    # dates
    #╭────────────────────────────────────────────────────────────────────────────╮#
    dates = [
             # datetime.datetime(2024, 5, 28),
             # datetime.datetime(2024, 5, 30), # ARCSIX-1 science flight #2, cloud wall, operator - Vikas Nataraja
             # datetime.datetime(2024, 5, 31), # ARCSIX-1 science flight #3, bowling alley; surface BRDF, operator - Vikas Nataraja
             # datetime.datetime(2024, 6, 3),  # ARCSIX-1 science flight #4, cloud wall, operator - Vikas Nataraja
             # datetime.datetime(2024, 6, 5),
             datetime.datetime(2024, 6, 6),  # ARCSIX-1 science flight #6, cloud wall, operator - Vikas Nataraja, Jeffery Drouet
             # datetime.datetime(2024, 6, 7),  # ARCSIX-1 science flight #7, cloud wall, operator - Vikas Nataraja, Arabella Chamberlain
             # datetime.datetime(2024, 6, 10), # ARCSIX-1 science flight #8, operator - Jeffery Drouet
             # datetime.datetime(2024, 6, 11), # ARCSIX-1 science flight #9, operator - Arabella Chamberlain, Sebastian Becker
             # datetime.datetime(2024, 6, 13), # ARCSIX-1 science flight #10, operator - Arabella Chamberlain
             # datetime.datetime(2024, 7, 25), # ARCSIX-2 science flight #11, cloud walls, operator - Arabella Chamberlain
             # datetime.datetime(2024, 7, 29), # ARCSIX-2 science flight #12, clear-sky BRDF, operator - Ken Hirata, Vikas Nataraja
             # datetime.datetime(2024, 7, 30), # ARCSIX-2 science flight #13, clear-sky BRDF, operator - Ken Hirata
             # datetime.datetime(2024, 8, 1),  # ARCSIX-2 science flight #14, cloud walls, operator - Ken Hirata
             # datetime.datetime(2024, 8, 2),  # ARCSIX-2 science flight #15, cloud walls, operator - Ken Hirata, Arabella Chamberlain
             # datetime.datetime(2024, 8, 7),  # ARCSIX-2 science flight #16, cloud walls, operator - Arabella Chamberlain
             # datetime.datetime(2024, 8, 8),  # ARCSIX-2 science flight #17, cloud walls, operator - Arabella Chamberlain
             # datetime.datetime(2024, 8, 9),  # ARCSIX-2 science flight #18, cloud walls, operator - Arabella Chamberlain
             # datetime.datetime(2024, 8, 15), # ARCSIX-2 science flight #19, cloud walls, operator - Ken Hirata, Sebastian Schmidt
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
        # run_time_offset_check(cfg)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # step 3
        # apply time offsets to sync data to aircraft housekeeping file
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # main_process_data_v1(cfg, run=True)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # step 4
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # main_process_data_v2(cfg, run=True)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # step 5
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # main_process_data_archive(date, run=True)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        pass
