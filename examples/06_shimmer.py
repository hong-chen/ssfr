"""
Code for processing data collected by "HSR1-B" (used to be called SPNS-B) during NRL SHIMMER 2024.

Acknowledgements:
    Instrument engineering:
        John Wood
    Data analysis:
        Hong Chen, Arabella Chamberlain, Sebastian Schmidt
    Instrument operation:
        Anthony Bucholtz
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
#╭────────────────────────────────────────────────────────────────────────────╮#
_MISSION_     = 'shimmer'
_PLATFORM_    = 'dhc6'

_HSK_         = 'hsk'
_HSR1_        = 'hsr1-b'

_FDIR_HSK_   = 'data/arcsix/2024/p3/aux/hsk'

_FDIR_DATA_  = 'data/%s' % _MISSION_
_FDIR_OUT_   = '%s/processed' % _FDIR_DATA_

_VERBOSE_   = True
_FNAMES_ = {}
#╰────────────────────────────────────────────────────────────────────────────╯#


_HSR1_TIME_OFFSET_ = {
        '20241106': 0.0,
        }

# functions for processing HSK
#╭────────────────────────────────────────────────────────────────────────────╮#
def cdata_arcsix_hsk_v0(
        date,
        fdir_data=_FDIR_DATA_,
        fdir_out=_FDIR_OUT_,
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

    fname_h5 = '%s/%s-%s_%s_%s_v0.h5' % (fdir_out, _MISSION_.upper(), _HSK_.upper(), _PLATFORM_.upper(), date_s)
    if run:

        try:

            # ict file from P3 data system team, best quality but cannot be accessed immediately
            #╭────────────────────────────────────────────────────────────────────────────╮#
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
            #╰────────────────────────────────────────────────────────────────────────────╯#

        except Exception as error:
            print(error)

            # iwg file from <https://asp-archive.arc.nasa.gov>, secondary option
            #╭────────────────────────────────────────────────────────────────────────────╮#
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
            #╰────────────────────────────────────────────────────────────────────────────╯#

            # wts file from <https://mts2.nasa.gov/> -> Telemetry, immediately availale after flight but poor quality
            #╭────────────────────────────────────────────────────────────────────────────╮#
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
            #╰────────────────────────────────────────────────────────────────────────────╯#

        print()
        print('Processing HSK file:', fname)
        print()


        # fake hsk for PSB (Pituffik Space Base)
        #╭────────────────────────────────────────────────────────────────────────────╮#
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
        #╰────────────────────────────────────────────────────────────────────────────╯#


        # fake hsk for NASA WFF
        #╭────────────────────────────────────────────────────────────────────────────╮#
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
        #╰────────────────────────────────────────────────────────────────────────────╯#


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
#╰────────────────────────────────────────────────────────────────────────────╯#


# functions for processing HSR1
#╭────────────────────────────────────────────────────────────────────────────╮#
def cdata_arcsix_hsr1_v0(
        date,
        fdir_data=_FDIR_DATA_,
        fdir_out=_FDIR_OUT_,
        run=True,
        ):

    """
    Process raw HSR1 data
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v0.h5' % (fdir_out, _MISSION_.upper(), _HSR1_.split('-')[0].upper(), _PLATFORM_.upper(), date_s)

    if run:

        # read hsr1 raw data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        fname_dif = ssfr.util.get_all_files(fdir_data, pattern='*Diffuse*.txt')[-1]
        data0_dif = ssfr.lasp_hsr.read_hsr1(fname=fname_dif)

        fname_tot = ssfr.util.get_all_files(fdir_data, pattern='*Total*.txt')[-1]
        data0_tot = ssfr.lasp_hsr.read_hsr1(fname=fname_tot)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # read wavelengths and calculate toa downwelling solar flux
        #╭────────────────────────────────────────────────────────────────────────────╮#
        flux_toa = ssfr.util.get_solar_kurudz()

        wvl_tot = data0_tot.data['wvl']
        f_dn_sol_tot = np.zeros_like(wvl_tot)
        for i, wvl0 in enumerate(wvl_tot):
            f_dn_sol_tot[i] = ssfr.util.cal_weighted_flux(wvl0, flux_toa[:, 0], flux_toa[:, 1])*ssfr.util.cal_solar_factor(date)
        #╰────────────────────────────────────────────────────────────────────────────╯#

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

def cdata_arcsix_hsr1_v1(
        date,
        fname_hsr1_v0,
        fname_hsk,
        fdir_out=_FDIR_OUT_,
        time_offset=0.0,
        run=True,
        ):

    """
    Check for time offset and merge HSR1 data with aircraft data
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v1.h5' % (fdir_out, _MISSION_.upper(), _HSR1_.split('-')[0].upper(), _PLATFORM_.upper(), date_s)

    if run:
        # read hsr1 v0
        #╭────────────────────────────────────────────────────────────────────────────╮#
        data_hsr1_v0 = ssfr.util.load_h5(fname_hsr1_v0)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # read hsk v0
        #╭────────────────────────────────────────────────────────────────────────────╮#
        data_hsk= ssfr.util.load_h5(fname_hsk)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        time_offset = _HSR1_TIME_OFFSET_[date_s]

        # interpolate hsr1 data to hsk time frame
        #╭────────────────────────────────────────────────────────────────────────────╮#
        flux_dif = np.zeros((data_hsk['jday'].size, data_hsr1_v0['dif/wvl'].size), dtype=np.float64)
        for i in range(flux_dif.shape[-1]):
            flux_dif[:, i] = ssfr.util.interp(data_hsk['jday'], data_hsr1_v0['dif/jday']+time_offset/86400.0, data_hsr1_v0['dif/flux'][:, i], mode='nearest')

        flux_tot = np.zeros((data_hsk['jday'].size, data_hsr1_v0['tot/wvl'].size), dtype=np.float64)
        for i in range(flux_tot.shape[-1]):
            flux_tot[:, i] = ssfr.util.interp(data_hsk['jday'], data_hsr1_v0['tot/jday']+time_offset/86400.0, data_hsr1_v0['tot/flux'][:, i], mode='nearest')
        #╰────────────────────────────────────────────────────────────────────────────╯#

        f = h5py.File(fname_h5, 'w')

        for key in data_hsk.keys():
            f[key] = data_hsk[key]

        f['time_offset'] = time_offset
        f['tmhr_ori'] = data_hsk['tmhr'] - time_offset/3600.0
        f['jday_ori'] = data_hsk['jday'] - time_offset/86400.0

        g1 = f.create_group('dif')
        g1['wvl']   = data_hsr1_v0['dif/wvl']
        dset0 = g1.create_dataset('flux', data=flux_dif, compression='gzip', compression_opts=9, chunks=True)

        g2 = f.create_group('tot')
        g2['wvl']   = data_hsr1_v0['tot/wvl']
        g2['toa0']  = data_hsr1_v0['tot/toa0']
        dset0 = g2.create_dataset('flux', data=flux_tot, compression='gzip', compression_opts=9, chunks=True)

        f.close()

    return fname_h5

def cdata_arcsix_hsr1_v2(
        date,
        fname_hsr1_v1,
        fname_hsk, # interchangable with fname_alp_v1
        wvl_range=None,
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        fdir_out=_FDIR_OUT_,
        run=True,
        ):

    """
    Apply attitude correction to account for aircraft attitude (pitch, roll, heading)
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v2.h5' % (fdir_out, _MISSION_.upper(), _HSR1_.split('-')[0].upper(), _PLATFORM_.upper(), date_s)

    if run:

        # read hsr1 v1
        #╭────────────────────────────────────────────────────────────────────────────╮#
        data_hsr1_v1 = ssfr.util.load_h5(fname_hsr1_v1)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # read hsk v0
        #╭────────────────────────────────────────────────────────────────────────────╮#
        data_hsk = ssfr.util.load_h5(fname_hsk)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # correction factor
        #╭────────────────────────────────────────────────────────────────────────────╮#
        mu = np.cos(np.deg2rad(data_hsk['sza']))

        try:
            iza, iaa = ssfr.util.prh2za(data_hsk['ang_pit']+ang_pit_offset, data_hsk['ang_rol']+ang_rol_offset, data_hsk['ang_hed'])
        except Exception as error:
            print(error)
            iza, iaa = ssfr.util.prh2za(data_hsk['ang_pit_s']+ang_pit_offset, data_hsk['ang_rol_s']+ang_rol_offset, data_hsk['ang_hed'])
        dc = ssfr.util.muslope(data_hsk['sza'], data_hsk['saa'], iza, iaa)

        factors = mu / dc
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # attitude correction
        #╭────────────────────────────────────────────────────────────────────────────╮#
        f_dn_dir = data_hsr1_v1['tot/flux'] - data_hsr1_v1['dif/flux']
        f_dn_dir_corr = np.zeros_like(f_dn_dir)
        f_dn_tot_corr = np.zeros_like(f_dn_dir)
        for iwvl in range(data_hsr1_v1['tot/wvl'].size):
            f_dn_dir_corr[..., iwvl] = f_dn_dir[..., iwvl]*factors
            f_dn_tot_corr[..., iwvl] = f_dn_dir_corr[..., iwvl] + data_hsr1_v1['dif/flux'][..., iwvl]
        #╰────────────────────────────────────────────────────────────────────────────╯#

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

        logic_wvl_dif = (data_hsr1_v1['dif/wvl']>=wvl_range[0]) & (data_hsr1_v1['dif/wvl']<=wvl_range[1])
        logic_wvl_tot = (data_hsr1_v1['tot/wvl']>=wvl_range[0]) & (data_hsr1_v1['tot/wvl']<=wvl_range[1])

        g1 = f.create_group('dif')
        g1['wvl']   = data_hsr1_v1['dif/wvl'][logic_wvl_dif]
        dset0 = g1.create_dataset('flux', data=data_hsr1_v1['dif/flux'][:, logic_wvl_dif], compression='gzip', compression_opts=9, chunks=True)

        g2 = f.create_group('tot')
        g2['wvl']   = data_hsr1_v1['tot/wvl'][logic_wvl_tot]
        g2['toa0']  = data_hsr1_v1['tot/toa0'][logic_wvl_tot]
        dset0 = g2.create_dataset('flux', data=f_dn_tot_corr[:, logic_wvl_tot], compression='gzip', compression_opts=9, chunks=True)

        f.close()

    return fname_h5

def cdata_arcsix_hsr1_archive(
        date,
        fname_hsr1_v2,
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        wvl_range=[400.0, 800.0],
        platform_info = 'p3',
        principal_investigator_info = 'Chen, Hong',
        affiliation_info = 'University of Colorado Boulder',
        instrument_info = 'HSR1-B (Sunshine Pyranometer - Spectral)',
        mission_info = 'ARCSIX 2024',
        project_info = '',
        file_format_index = '1001',
        file_volume_number = '1, 1',
        data_interval = '1.0',
        scale_factor = '1.0',
        fill_value = 'NaN',
        version='RA',
        fdir_out=_FDIR_OUT_,
        run=True,
        ):


    # placeholder for additional information such as calibration
    #╭────────────────────────────────────────────────────────────────────────────╮#
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # date info
    #╭────────────────────────────────────────────────────────────────────────────╮#
    date_s = date.strftime('%Y%m%d')
    date_today = datetime.date.today()
    date_info  = '%4.4d, %2.2d, %2.2d, %4.4d, %2.2d, %2.2d' % (date.year, date.month, date.day, date_today.year, date_today.month, date_today.day)
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
    data_info = 'Shortwave Total and Diffuse Downwelling Spectral Irradiance from %s %s' % (platform_info.upper(), instrument_info)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # routine comments
    #╭────────────────────────────────────────────────────────────────────────────╮#
    comments_routine_list = OrderedDict({
            'PI_CONTACT_INFO': 'Address: University of Colorado Boulder, LASP, 3665 Discovery Drive, Boulder, CO 80303; E-mail: hong.chen@lasp.colorado.edu and sebastian.schmidt@lasp.colorado.edu',
            'PLATFORM': platform_info.upper(),
            'LOCATION': 'N/A',
            'ASSOCIATED_DATA': 'N/A',
            'INSTRUMENT_INFO': instrument_info,
            'DATA_INFO': 'Reported are only of a selected wavelength range (%d-%d nm), time/lat/lon/alt/pitch/roll/heading from aircraft, sza calculated from time/lon/lat.' % (wvl_range[0], wvl_range[1]),
            'UNCERTAINTY': 'Nominal HSR1 uncertainty (shortwave): total: N/A; diffuse: N/A',
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
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # data processing
    #╭────────────────────────────────────────────────────────────────────────────╮#
    data_v2 = ssfr.util.load_h5(fname_hsr1_v2)
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

    fname_h5 = '%s/%s-HSR1_%s_%s_%s.h5' % (fdir_out, _MISSION_.upper(), _PLATFORM_.upper(), date_s, version.upper())
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
def run_time_offset_check(date):

    date_s = date.strftime('%Y%m%d')
    data_hsk = ssfr.util.load_h5(_FNAMES_['%s_hsk_v0' % date_s])
    data_alp = ssfr.util.load_h5(_FNAMES_['%s_alp_v0' % date_s])
    data_hsr1_v0 = ssfr.util.load_h5(_FNAMES_['%s_hsr1_v0' % date_s])
    if _WHICH_SSFR_ == _SSFR1_:
        data_ssfr1_v0 = ssfr.util.load_h5(_FNAMES_['%s_ssfr1_v0' % date_s])
        data_ssfr2_v0 = ssfr.util.load_h5(_FNAMES_['%s_ssfr2_v0' % date_s])
    else:
        data_ssfr1_v0 = ssfr.util.load_h5(_FNAMES_['%s_ssfr2_v0' % date_s])
        data_ssfr2_v0 = ssfr.util.load_h5(_FNAMES_['%s_ssfr1_v0' % date_s])

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


    # SSFR-A vs HSR1
    #╭────────────────────────────────────────────────────────────────────────────╮#
    index_wvl_hsr1 = np.argmin(np.abs(745.0-data_hsr1_v0['tot/wvl']))
    data_y0 = data_hsr1_v0['tot/flux'][:, index_wvl_hsr1]

    index_wvl_ssfr = np.argmin(np.abs(745.0-data_ssfr1_v0['spec/wvl_zen']))
    data_y1 = data_ssfr1_v0['spec/cnt_zen'][:, index_wvl_ssfr]
    data_offset = {
            'x0': data_hsr1_v0['tot/jday']*86400.0,
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
            description='SSFR-A Zenith Count vs. HSR1 Total (745nm)',
            fname_html='ssfr-a_offset_check_%s.html' % (date_s))
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # SSFR-B vs SSFR-A
    #╭────────────────────────────────────────────────────────────────────────────╮#
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
    fname_hsr1_v2 = cdata_arcsix_hsr1_v2(date, _FNAMES_['%s_hsr1_v1' % date_s], _FNAMES_['%s_hsk_v0' % date_s],
            fdir_out=_FDIR_OUT_,
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
def main_process_data_v0(date, run=True):

    fdir_out = _FDIR_OUT_
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    date_s = date.strftime('%Y%m%d')


    # # HSK v0: raw data
    # #╭────────────────────────────────────────────────────────────────────────────╮#
    # fnames_hsk = ssfr.util.get_all_files(_FDIR_HSK_, pattern='*%4.4d*%2.2d*%2.2d*.???' % (date.year, date.month, date.day))
    # if run and len(fnames_hsk) == 0:
    #     # * not preferred, use ALP lon/lat if P3 housekeeping file is not available (e.g., for immediate data processing)
    #     fname_hsk_v0 = cdata_arcsix_hsk_from_alp_v0(date, _FNAMES_['%s_alp_v0' % date_s], fdir_data=_FDIR_HSK_,
    #             fdir_out=fdir_out, run=run)
    # else:
    #     # * preferred, use P3 housekeeping file, ict > iwg > mts
    #     fname_hsk_v0 = cdata_arcsix_hsk_v0(date, fdir_data=_FDIR_HSK_,
    #             fdir_out=fdir_out, run=run)
    # _FNAMES_['%s_hsk_v0' % date_s] = fname_hsk_v0
    # #╰────────────────────────────────────────────────────────────────────────────╯#
    # sys.exit()


    # HSR1 v0: raw data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fdirs = ssfr.util.get_all_folders(_FDIR_DATA_, pattern='*%4.4d*%2.2d*%2.2d*raw?%s*' % (date.year, date.month, date.day, _HSR1_.lower()))
    fdir_data_hsr1 = sorted(fdirs, key=os.path.getmtime)[-1]
    fnames_hsr1 = ssfr.util.get_all_files(fdir_data_hsr1, pattern='*.txt')
    if run and len(fnames_hsr1) == 0:
        pass
    else:
        fname_hsr1_v0 = cdata_arcsix_hsr1_v0(date, fdir_data=fdir_data_hsr1,
                fdir_out=fdir_out, run=run)
        _FNAMES_['%s_hsr1_v0' % date_s]  = fname_hsr1_v0
    #╰────────────────────────────────────────────────────────────────────────────╯#
    sys.exit()


    # HSK v0: raw data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fnames_hsk = ssfr.util.get_all_files(_FDIR_HSK_, pattern='*%4.4d*%2.2d*%2.2d*.???' % (date.year, date.month, date.day))
    if run and len(fnames_hsk) == 0:
        # * not preferred, use ALP lon/lat if P3 housekeeping file is not available (e.g., for immediate data processing)
        fname_hsk_v0 = cdata_arcsix_hsk_from_alp_v0(date, _FNAMES_['%s_alp_v0' % date_s], fdir_data=_FDIR_HSK_,
                fdir_out=fdir_out, run=run)
    else:
        # * preferred, use P3 housekeeping file, ict > iwg > mts
        fname_hsk_v0 = cdata_arcsix_hsk_v0(date, fdir_data=_FDIR_HSK_,
                fdir_out=fdir_out, run=run)
    _FNAMES_['%s_hsk_v0' % date_s] = fname_hsk_v0
    #╰────────────────────────────────────────────────────────────────────────────╯#

def main_process_data_v1(date, run=True):

    fdir_out = _FDIR_OUT_
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    date_s = date.strftime('%Y%m%d')

    # HSR1 v1: time synced with hsk time with time offset applied
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsr1_v1 = cdata_arcsix_hsr1_v1(date, _FNAMES_['%s_hsr1_v0' % date_s], _FNAMES_['%s_hsk_v0' % date_s],
            fdir_out=fdir_out, run=run)

    _FNAMES_['%s_hsr1_v1'  % date_s] = fname_hsr1_v1
    #╰────────────────────────────────────────────────────────────────────────────╯#

def main_process_data_v2(date, run=True):

    """
    v0: raw data directly read out from the data files
    v1: data collocated/synced to aircraft nav
    v2: attitude corrected data
    """

    date_s = date.strftime('%Y%m%d')

    fdir_out = _FDIR_OUT_
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    # HSR1 v2
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # * based on ALP v1
    # fname_hsr1_v2 = cdata_arcsix_hsr1_v2(date, _FNAMES_['%s_hsr1_v1' % date_s], _FNAMES_['%s_alp_v1' % date_s],
    #         fdir_out=fdir_out, run=run)
    # fname_hsr1_v2 = cdata_arcsix_hsr1_v2(date, _FNAMES_['%s_hsr1_v1' % date_s], _FNAMES_['%s_alp_v1' % date_s],
    #         fdir_out=fdir_out, run=True)

    # * based on HSK v0
    fname_hsr1_v2 = cdata_arcsix_hsr1_v2(date, _FNAMES_['%s_hsr1_v1' % date_s], _FNAMES_['%s_hsk_v0' % date_s],
            fdir_out=fdir_out, run=run)
    #╰────────────────────────────────────────────────────────────────────────────╯#
    _FNAMES_['%s_hsr1_v2' % date_s] = fname_hsr1_v2

    _FNAMES_[_vname_ssfr_v2_] = fname_ssfr_v2

def main_process_data_archive(date, run=True):

    """
    ra: in-field data to be uploaded to https://www-air.larc.nasa.gov/cgi-bin/ArcView/arcsix
    """

    date_s = date.strftime('%Y%m%d')

    fdir_out = _FDIR_OUT_
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    # HSR1 RA
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsr1_ra = cdata_arcsix_hsr1_archive(date, _FNAMES_['%s_hsr1_v2' % date_s],
            fdir_out=fdir_out, run=run)
    #╰────────────────────────────────────────────────────────────────────────────╯#
    _FNAMES_['%s_hsr1_ra' % date_s] = fname_hsr1_ra


    # SSFR RA
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if _WHICH_SSFR_ == _SSFR1_:
        _vname_ssfr_v2_ = '%s_ssfr1_v2' % date_s
        _vname_ssfr_ra_ = '%s_ssfr1_ra' % date_s
    else:
        _vname_ssfr_v2_ = '%s_ssfr2_v2' % date_s
        _vname_ssfr_ra_ = '%s_ssfr2_ra' % date_s

    fname_ssfr_ra = cdata_arcsix_ssfr_archive(date, _FNAMES_[_vname_ssfr_v2_],
            fdir_out=fdir_out, run=run)
    #╰────────────────────────────────────────────────────────────────────────────╯#
    _FNAMES_[_vname_ssfr_ra_] = fname_ssfr_ra
#╰────────────────────────────────────────────────────────────────────────────╯#


# check data
#╭────────────────────────────────────────────────────────────────────────────╮#
def check(date):

    date_s = date.strftime('%Y%m%d')

    fname = ssfr.util.get_all_files(_FDIR_OUT_, pattern='*%s*%s*v0*' % (_HSR1_.upper(), date_s))[0]

    data = ssfr.util.load_h5(fname)

    logic_nan = np.isnan(data['tot/tmhr']) | np.isnan(data['dif/tmhr'])
    for key in data.keys():
        if data['tot/tmhr'].size in data[key].shape:
            data[key] = data[key][~logic_nan]

    wvl0 = 555.0
    index_wvl = np.argmin(np.abs(data['tot/wvl']-wvl0))

    indices = np.unravel_index(np.argmax(data['tot/flux']), data['tot/flux'].shape)

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('%s on %s (%dnm)' % (_MISSION_.upper(), date.strftime('%Y-%m-%d'), wvl0), y=0.94)

        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(211)
        ax1.scatter(data['tot/tmhr'], data['tot/flux'][:, index_wvl], s=6, c='green', lw=0.0)
        ax1.scatter(data['dif/tmhr'], data['dif/flux'][:, index_wvl], s=6, c='springgreen', lw=0.0)
        ax1.axvline(data['tot/tmhr'][indices[0]], color='k', ls='--')
        ax1.set_ylabel('Irradiance [$\\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.set_xlabel('Time [Hour]')
        ax1.set_xlim((22, 24.2))
        ax1.set_ylim(bottom=0.0)

        patches_legend = [
                          mpatches.Patch(color='green'      , label='Total'), \
                          mpatches.Patch(color='springgreen', label='Diffuse'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)
        #╰──────────────────────────────────────────────────────────────╯#

        # plot2
        #╭──────────────────────────────────────────────────────────────╮#
        ax2 = fig.add_subplot(212)
        ax2.scatter(data['tot/wvl'], data['tot/flux'][indices[0], :], s=6, c='green', lw=0.0)
        ax2.scatter(data['dif/wvl'], data['dif/flux'][indices[0], :], s=6, c='springgreen', lw=0.0)
        ax2.set_ylabel('Irradiance [$\\mathrm{W m^{-2} nm^{-1}}$]')
        ax2.set_xlabel('Wavelength [nm]')
        ax2.set_ylim(bottom=0.0)
        #╰──────────────────────────────────────────────────────────────╯#

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        plt.savefig('%s_%s_%s.png' % (_metadata_['Function'], _MISSION_.lower(), date_s), bbox_inches='tight', metadata=_metadata_)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return
#╰────────────────────────────────────────────────────────────────────────────╯#


if __name__ == '__main__':

    warnings.warn('\n!!!!!!!! Under development !!!!!!!!')

    # dates
    #╭────────────────────────────────────────────────────────────────────────────╮#
    dates = [
             datetime.datetime(2024, 11, 6),  # SHIMMER flight #1
            ]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    for date in dates[::-1]:

        # check(date)
        # sys.exit()

        # step 1
        #╭────────────────────────────────────────────────────────────────────────────╮#
        main_process_data_v0(date, run=True)
        main_process_data_v0_metnav(date, run=True)
        sys.exit()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # step 2
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # main_process_data_v0(date, run=False)
        # run_time_offset_check(date)
        # sys.exit()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # step 3
        #╭────────────────────────────────────────────────────────────────────────────╮#
        main_process_data_v0(date, run=False)
        main_process_data_v1(date, run=True)
        # sys.exit()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # step 4
        #╭────────────────────────────────────────────────────────────────────────────╮#
        main_process_data_v0(date, run=False)
        main_process_data_v1(date, run=False)
        main_process_data_v2(date, run=True)
        # sys.exit()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # step 5
        #╭────────────────────────────────────────────────────────────────────────────╮#
        main_process_data_v0(date, run=False)
        main_process_data_v1(date, run=False)
        main_process_data_v2(date, run=False)
        main_process_data_archive(date, run=True)
        # sys.exit()
        #╰────────────────────────────────────────────────────────────────────────────╯#

        pass
