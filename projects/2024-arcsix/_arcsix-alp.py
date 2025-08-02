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

    return
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

    # ALP v1: time synced with hsk time with time offset applied
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_h5 = '%s/%s-%s_%s_%s_v1.h5' % (fdir_out, cfg.common['mission'].upper(), cfg.alp['aka'].upper(), cfg.common['platform'].upper(), date_s)

    fname_alp_v1 = cdata_alp_v1(
            date,
            _FNAMES_['%s_alp_v0' % date_s],
            _FNAMES_['%s_hsk_v0' % date_s],
            fname_h5=fname_h5,
            time_offset=cfg.alp['time_offset'],
            fdir_out=fdir_out,
            run=run
            )

    _FNAMES_['%s_alp_v1' % date_s] = fname_alp_v1
    #╰────────────────────────────────────────────────────────────────────────────╯#
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
        run_time_offset_check(cfg)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # step 3
        # apply time offsets to sync data to aircraft housekeeping file
        #╭────────────────────────────────────────────────────────────────────────────╮#
        main_process_data_v1(cfg, run=True)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        pass
