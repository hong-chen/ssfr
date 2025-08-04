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


# functions for processing HSR1
#╭────────────────────────────────────────────────────────────────────────────╮#
def cdata_hsr1_v0(
        date,
        fnames_hsr,
        fname_h5='HSR1_v0.h5',
        fdir_out='./',
        run=True,
        ):

    """
    Process raw HSR1 data
    """

    date_s = date.strftime('%Y%m%d')

    if run:

        # read hsr1 raw data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        fname_dif = [fname for fname in fnames_hsr if 'diffuse' in fname.lower()][0]
        data0_dif = ssfr.lasp_hsr.read_hsr1(fname=fname_dif)

        fname_tot = [fname for fname in fnames_hsr if 'total' in fname.lower()][0]
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

def cdata_hsr1_v1(
        date,
        fname_hsr1_v0,
        fname_hsk,
        fname_h5='HSR1_v1.h5',
        fdir_out='./',
        time_offset=0.0,
        run=True,
        ):

    """
    Check for time offset and merge HSR1 data with aircraft data
    """

    date_s = date.strftime('%Y%m%d')

    if run:
        # read hsr1 v0
        #╭────────────────────────────────────────────────────────────────────────────╮#
        data_hsr1_v0 = ssfr.util.load_h5(fname_hsr1_v0)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # read hsk v0
        #╭────────────────────────────────────────────────────────────────────────────╮#
        data_hsk= ssfr.util.load_h5(fname_hsk)
        #╰────────────────────────────────────────────────────────────────────────────╯#

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

def cdata_hsr1_v2(
        date,
        fname_hsr1_v1,
        fname_hsk, # interchangable with fname_alp_v1
        fname_h5='HSR1_v2.h5',
        wvl_range=None,
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        fdir_out='./',
        run=True,
        ):

    """
    Apply attitude correction to account for aircraft attitude (pitch, roll, heading)
    """

    date_s = date.strftime('%Y%m%d')

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

def cdata_hsr1_archive(
        date,
        fname_hsr1_v2,
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        wvl_range=[400.0, 800.0],
        platform_info = 'p3',
        principal_investigator_info = 'Chen, Hong',
        affiliation_info = 'University of Colorado Boulder',
        instrument_info = 'HSR1 (Hyper-Spectral Radiometer 1)',
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
def run_time_offset_check(cfg):

    main_process_data_v0(cfg, run=False)

    date = cfg.common['date']
    date_s = date.strftime('%Y%m%d')
    data_hsk = ssfr.util.load_h5(_FNAMES_['%s_hsk_v0' % date_s])
    data_hsr1_v0 = ssfr.util.load_h5(_FNAMES_['%s_hsr1_v0' % date_s])

    # data_hsr1_v0['tot/jday'] += 1.0
    # data_hsr1_v0['dif/jday'] += 1.0

    # _offset_x_range_ = [-6000.0, 6000.0]
    _offset_x_range_ = [-600.0, 600.0]


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
#╰────────────────────────────────────────────────────────────────────────────╯#


# main program
#╭────────────────────────────────────────────────────────────────────────────╮#
def main_process_data_v0(cfg, run=True):

    date = cfg.common['date']
    date_s = cfg.common['date_s']

    fdir_out = cfg.common['fdir_out']
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    # HSR1 v0: raw data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fnames_hsr1 = cfg.hsr1['fnames']
    fname_h5 = '%s/%s-%s_%s_%s_v0.h5' % (fdir_out, cfg.common['mission'].upper(), cfg.hsr1['aka'].upper(), cfg.common['platform'].upper(), date_s)
    if run and len(fnames_hsr1) == 0:
        pass
    else:
        fname_hsr1_v0 = cdata_hsr1_v0(
                date,
                fnames_hsr1,
                fname_h5=fname_h5,
                fdir_out=fdir_out,
                run=run
                )
        _FNAMES_['%s_hsr1_v0' % date_s]  = fname_hsr1_v0
    #╰────────────────────────────────────────────────────────────────────────────╯#

def main_process_data_v1(cfg, run=True):

    date = cfg.common['date']
    date_s = cfg.common['date_s']

    fdir_out = cfg.common['fdir_out']
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    main_process_data_v0(cfg, run=False)

    # HSR1 v1: time synced with hsk time with time offset applied
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_h5 = '%s/%s-%s_%s_%s_v1.h5' % (fdir_out, cfg.common['mission'].upper(), cfg.hsr1['aka'].upper(), cfg.common['platform'].upper(), date_s)

    fname_hsr1_v1 = cdata_hsr1_v1(
            date,
            _FNAMES_['%s_hsr1_v0' % date_s],
            _FNAMES_['%s_hsk_v0' % date_s],
            fname_h5=fname_h5,
            time_offset=cfg.hsr1['time_offset'],
            fdir_out=fdir_out,
            run=run
            )

    _FNAMES_['%s_hsr1_v1' % date_s] = fname_hsr1_v1
    #╰────────────────────────────────────────────────────────────────────────────╯#

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

def main_process_data_archive(date, run=True):

    """
    ra: in-field data to be uploaded to https://www-air.larc.nasa.gov/cgi-bin/ArcView/arcsix
    """

    date_s = date.strftime('%Y%m%d')

    fdir_out = './'
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    main_process_data_v0(date, run=False)
    main_process_data_v1(date, run=False)
    main_process_data_v2(date, run=False)

    # HSR1 RA
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsr1_ra = cdata_hsr1_archive(date, _FNAMES_['%s_hsr1_v2' % date_s],
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

    fname_ssfr_ra = cdata_ssfr_archive(date, _FNAMES_[_vname_ssfr_v2_],
            fdir_out=fdir_out, run=run)
    #╰────────────────────────────────────────────────────────────────────────────╯#
    _FNAMES_[_vname_ssfr_ra_] = fname_ssfr_ra
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
             datetime.datetime(2024, 6, 6),
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
        # main_process_data_v0(cfg, run=True)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # step 2
        # create bokeh interactive plots to retrieve time offset
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # run_time_offset_check(cfg)
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

        # step 5
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # main_process_data_archive(date, run=True)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        pass
