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
def cdata_hsr1_archive(
        cfg,
        fname_hsr1_v2,
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
    date = cfg.common['date']
    date_s = cfg.common['date_s']
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

    fname_h5 = '%s/%s-HSR1_%s_%s_%s.h5' % (fdir_out, cfg.common['mission'].upper(), cfg.common['platform'].upper(), date_s, version.upper())
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


# functions for processing SSFR
#╭────────────────────────────────────────────────────────────────────────────╮#
def cdata_ssfr_archive(
        cfg,
        fname_ssfr_v2,
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
    date = cfg.common['date']
    date_s = cfg.common['date_s']
    date_today = datetime.date.today()
    date_info  = '%4.4d, %2.2d, %2.2d, %4.4d, %2.2d, %2.2d' % (date.year, date.month, date.day, date_today.year, date_today.month, date_today.day)

    if  cfg.ssfr['tag'].lower() == 'ssfr-a':
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

    fname_h5 = '%s/%s-SSFR_%s_%s_%s.h5' % (fdir_out, cfg.common['mission'].upper(), cfg.common['platform'].upper(), date_s, version.upper())
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

# functions for processing SSRR
#╭────────────────────────────────────────────────────────────────────────────╮#
def cdata_ssrr_archive(
        cfg,
        fname_ssrr_v1,
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
    date = cfg.common['date']
    date_s = cfg.common['date_s']
    date_today = datetime.date.today()
    date_info  = '%4.4d, %2.2d, %2.2d, %4.4d, %2.2d, %2.2d' % (date.year, date.month, date.day, date_today.year, date_today.month, date_today.day)

    if  cfg.ssrr['tag'].lower() == 'ssrr-a':
        instrument_info = 'SSRR-A (Solar Spectral Radiance Radiometer - Alvin)'
    else:
        instrument_info = 'SSRR-B (Solar Spectral Radiance Radiometer - Belana)'
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
    data_info = 'Shortwave Downwelling and Upwelling Spectral Radiance from %s %s' % (platform_info.upper(), instrument_info)
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
            'UNCERTAINTY': 'Nominal SSRR uncertainty (shortwave): nadir: N/A; zenith: N/A',
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
            # '20240530': 'Noticed icing on dome outside zenith light collector after flight',
            # '20240531': 'Encountered temperature control issue (after around 1:30 UTC)',
            # '20240730': 'Noticed icing on dome inside zenith light collector after flight',
            # '20240801': 'Noticed condensation on dome inside zenith light collector before flight',
            # '20240807': 'Noticed condensation on dome inside zenith light collector after flight',
            # '20240808': 'Noticed condensation on dome inside zenith light collector after flight',
            # '20240809': 'Noticed condensation on dome inside zenith light collector after flight',
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
    data_v1 = ssfr.util.load_h5(fname_ssrr_v1)
    data_v1['zen/rad'][data_v1['zen/rad']<0.0] = np.nan
    data_v1['nad/rad'][data_v1['nad/rad']<0.0] = np.nan

    logic_zen = (data_v1['zen/wvl']>=wvl_range[0]) & (data_v1['zen/wvl']<=wvl_range[1])
    logic_nad = (data_v1['nad/wvl']>=wvl_range[0]) & (data_v1['nad/wvl']<=wvl_range[1])

    data = OrderedDict({
            'Time_Start': {
                'data': data_v1['tmhr']*3600.0,
                'unit': 'second',
                'description': 'UTC time in seconds from the midnight 00:00:00',
                },

            'jday': {
                'data': data_v1['jday'],
                'unit': 'day',
                'description': 'UTC time in decimal day from 0001-01-01 00:00:00',
                },

            'tmhr': {
                'data': data_v1['tmhr'],
                'unit': 'hour',
                'description': 'UTC time in decimal hour from the midnight 00:00:00',
                },

            'lon': {
                'data': data_v1['lon'],
                'unit': 'degree',
                'description': 'longitude',
                },

            'lat': {
                'data': data_v1['lat'],
                'unit': 'degree',
                'description': 'latitude',
                },

            'alt': {
                'data': data_v1['alt'],
                'unit': 'meter',
                'description': 'altitude',
                },

            'sza': {
                'data': data_v1['sza'],
                'unit': 'degree',
                'description': 'solar zenith angle',
                },

            'zen/rad': {
                'data': data_v1['zen/rad'][:, logic_zen],
                'unit': 'W m^-2 nm^-1 sr^-1',
                'description': 'downwelling spectral radiance (zenith)',
                },

            'zen/wvl': {
                'data': data_v1['zen/wvl'][logic_zen],
                'unit': 'nm',
                'description': 'wavelength for downwelling spectral radiance (zenith)',
                },

            'nad/rad': {
                'data': data_v1['nad/rad'][:, logic_nad],
                'unit': 'W m^-2 nm^-1 sr^-1',
                'description': 'upwelling spectral radiance (nadir)',
                },

            'nad/wvl': {
                'data': data_v1['nad/wvl'][logic_nad],
                'unit': 'nm',
                'description': 'wavelength for upwelling spectral radiance (nadir)',
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

    fname_h5 = '%s/%s-SSRR_%s_%s_%s.h5' % (fdir_out, cfg.common['mission'].upper(), cfg.common['platform'].upper(), date_s, version.upper())
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

# functions for processing HSR1 and SSFR (R0 version)
#╭────────────────────────────────────────────────────────────────────────────╮#
def cdata_hsr1_archive_r0_from_ra(
        cfg,
        fname_hsr1_ra,
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
        version='R0',
        fdir_out='./',
        run=True,
        ):


    # placeholder for additional information such as calibration
    #╭────────────────────────────────────────────────────────────────────────────╮#
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # date info
    #╭────────────────────────────────────────────────────────────────────────────╮#
    date = cfg.common['date']
    date_s = date.strftime('%Y%m%d')
    date_today = datetime.date.today()
    date_info  = '%4.4d, %2.2d, %2.2d, %4.4d, %2.2d, %2.2d' % (date.year, date.month, date.day, date_today.year, date_today.month, date_today.day)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # version info
    #╭────────────────────────────────────────────────────────────────────────────╮#
    version = version.upper()
    version_info = {
            'RA': 'field data',
            'R0': 'first public release'
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
            '20240530': 'Additional notes: noticed icing on dome after 20240530 flight - this might affect the downwelling total and diffuse irradiance - compare with SSFR measurements for consistency.',
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
    data_ra = ssfr.util.load_h5(fname_hsr1_ra)

    # fname_hsk = 'data/arcsix/processed/ARCSIX-HSK_P3B_%s_v0.h5' % date_s
    # data_hsk = ssfr.util.load_h5(fname_hsk)

    data = OrderedDict({
            'time': {
                # 'data': data_hsk['tmhr']*3600.0,
                'data': data_ra['tmhr']*3600.0,
                'unit': 'second',
                'long_name': 'UTC time of measurement instant',
                'units': 'seconds since %s' % date,
                },

            'gps_lon': {
                # 'data': data_hsk['lon'],
                'data': data_ra['lon'],
                'long_name': 'longitude of aircraft',
                'units': 'degrees_east',
                },

            'gps_lat': {
                # 'data': data_hsk['lat'],
                'data': data_ra['lat'],
                'long_name': 'latitude of aircraft',
                'units': 'degrees_north',
                },

            'gps_alt': {
                # 'data': data_hsk['alt'],
                'data': data_ra['alt'],
                'long_name': 'altitude of aircraft',
                'units': 'meter',
                },

            'wvl_dn_dif': {
                'data': data_ra['dif/wvl'],
                'long_name': 'wavelength for spectral downwelling diffuse irradiance',
                'units': 'nm',
                },

            'wvl_dn_tot': {
                'data': data_ra['tot/wvl'],
                'long_name': 'wavelength for spectral downwelling total irradiance',
                'units': 'nm',
                },

            'f_dn_dif': {
                'data': data_ra['dif/flux'],
                'ACVSN_standard_name': 'Rad_IrradianceDownwellingDiffuse_InSitu_SP',
                'long_name': 'spectral downwelling diffuse irradiance',
                'units': 'W.m-2.nm-1',
                'coordinates': ['time', 'wvl_dn_dif'],
                '_FillValue': 'nan',
                },

            'f_dn_tot': {
                'data': data_ra['tot/flux'],
                'ACVSN_standard_name': 'Rad_IrradianceDownwelling_InSitu_SP',
                'long_name': 'spectral downwelling total irradiance',
                'units': 'W.m-2.nm-1',
                'coordinates': ['time', 'wvl_dn_tot'],
                '_FillValue': 'nan',
                },
            })

    for key in data.keys():
        data[key]['description'] = '%s: %s, %s' % (key, data[key]['units'], data[key]['long_name'])

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
                   data['time']['long_name'],                        # Description or name of independent variable (This is the name chosen for the start time. It always refers to the number of seconds UTC from the start of the day on which measurements began. It should be noted here that the independent variable should monotonically increase even when crossing over to a second day.).
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
    # sys.exit()
    global_attrs = {
            'ACVSN_standard_name_URL'        : '10.5067/DOC/ESCO/ESDS-RFC-043v1 (under User Resources)',
            'ACVSN_standard_name_version'    : '1.0',
            'Conventions'                    : 'CF-1.10',
            'Format'                         : 'HDF5',
            'PI_contact'                     : 'hong.chen@lasp.colorado.edu, sebastian.schmidt@lasp.colorado.edu',
            'PI_name'                        : 'Hong Chen, K. Sebastian Schmidt',
            'ProcessingLevel'                : 'L1',
            'VersionID'                      : 'R0',
            'aircraft_data_stream'           : 'IWG1',
            'associated_data'                : 'The navigation data (longitude, latitude, and altitude) are from MetNav RA/RB.',
            'data_processing_note'           : 'This version is based on pre-mission calibrations. Post-mission calibrations will be applied for the next public release (R1, expected end of March 2025). Attitude correction needs further investigation. %s' % comments_special,
            'data_product_groups'            : 'root',
            'data_use_guideline'             : 'For responsible scientific use of the data sets provided, data users are strongly encouraged to carefully study the file headers and directly consult with the instrument PIs. Please acknowledge the data source and offer co-authorship to relevant instrument PIs when appropriate.',
            'file_originator'                : 'Hong Chen, K. Sebastian Schmidt',
            'file_originator_contact'        : 'hong.chen@lasp.colorado.edu, sebastian.schmidt@lasp.colorado.edu',
            'flight_start_date'              : date_s,
            'geospatial_lat_max'             : '%.4fdegrees_north' % data['gps_lat']['data'].max(),
            'geospatial_lat_min'             : '%.4fdegrees_north' % data['gps_lat']['data'].min(),
            'geospatial_lon_max'             : '%.4fdegrees_east' % data['gps_lon']['data'].max(),
            'geospatial_lon_min'             : '%.4fdegrees_east' % data['gps_lon']['data'].min(),
            'history'                        : 'R0: First public data release.',
            'institution'                    : 'University of Colorado',
            'keywords'                       : 'radiation, irradiance, spectral, diffuse, total',
            'last_modified_date'             : str(datetime.datetime.now()),
            'measurement_platform'           : 'NASA P3-B N426NA',
            'platform_identifier'            : 'P3B',
            'platform_type'                  : 'AirMobile',
            'project'                        : 'ARCSIX 2024',
            # 'references'                     : 'N/A',
            'source'                         : 'HSR1',
            'source_description'             : 'Hyper-Spectral Radiometer 1',
            'summary'                        : 'ARCSIX HSR1 measurement of spectral downwelling diffuse and total irradiance on %s flight. Publication quality data.' % date_s,
            'time_coverage_end'              : str(ssfr.util.jday_to_dtime(data_ra['jday'][-1])),
            'time_coverage_resolution'       : '1 s',
            'time_coverage_start'            : str(ssfr.util.jday_to_dtime(data_ra['jday'][0])),
            'title'                          : 'ARCSIX HSR1 measurement of spectral downwelling diffuse and total irradiance',
            'unit_convention'                : 'http://codes.wmo.int/wmdr/unit',
            }

    dims = {}

    fname_h5 = '%s/%s-HSR1_%s_%s_%s.h5' % (fdir_out, cfg.common['mission'].upper(), cfg.common['platform'].upper(), date_s, version.upper())
    if run:
        f = h5py.File(fname_h5, 'w')

        for attr in global_attrs.keys():
            f.attrs[attr] = global_attrs[attr]

        for key in data.keys():
            dset = f.create_dataset(key, data=data[key]['data'], compression='gzip', compression_opts=9, chunks=True)
            for attr in data[key].keys():
                if attr not in ['data', 'description']:
                    dset.attrs[attr] = data[key][attr]
                if attr in ['coordinates']:
                    dset.attrs[attr] = ' '.join(data[key][attr])

            if key in ['time', 'wvl_dn_dif', 'wvl_dn_tot']:
                dset.make_scale(key)
                dims[key] = dset

            if key in ['f_dn_dif', 'f_dn_tot']:
                dset.dims[0].attach_scale(dims[data[key]['coordinates'][0]])
                dset.dims[1].attach_scale(dims[data[key]['coordinates'][1]])
        f.close()

    return fname_h5

def cdata_ssfr_archive_r0_from_ra(
        cfg,
        fname_ssfr_ra,
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
        version='R0',
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
    date = cfg.common['date']
    date_s = date.strftime('%Y%m%d')
    date_today = datetime.date.today()
    date_info  = '%4.4d, %2.2d, %2.2d, %4.4d, %2.2d, %2.2d' % (date.year, date.month, date.day, date_today.year, date_today.month, date_today.day)

    if cfg.ssfr['tag'].lower() == 'ssfr-a':
        instrument_info = 'SSFR-A (Solar Spectral Flux Radiometer - Alvin)'
    else:
        instrument_info = 'SSFR-B (Solar Spectral Flux Radiometer - Belana)'
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # version info
    #╭────────────────────────────────────────────────────────────────────────────╮#
    version = version.upper()
    version_info = {
            'RA': 'field data',
            'R0': 'First public release.',
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
            '20240530': 'Additional notes: noticed icing on dome outside zenith light collector after 20240530 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency.',
            '20240531': 'Additional notes: encountered temperature control issue (after around 1:30 UTC) on 20240531 flight - downwelling and upwelling irradiance might be compromised after this time - contact the PI if you use this data.',
            '20240730': 'Additional notes: noticed icing on dome inside zenith light collector after 20240730 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency.',
            '20240801': 'Additional notes: noticed condensation on dome inside zenith light collector before 20240801 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency.',
            '20240807': 'Additional notes: noticed condensation on dome inside zenith light collector after 20240807 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency.',
            '20240808': 'Additional notes: noticed condensation on dome inside zenith light collector after 20240808 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency.',
            '20240809': 'Additional notes: noticed condensation on dome inside zenith light collector after 20240809 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency.',
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
    data_ra = ssfr.util.load_h5(fname_ssfr_ra)

    # fname_hsk = 'data/arcsix/processed/ARCSIX-HSK_P3B_%s_v0.h5' % date_s
    # data_hsk = ssfr.util.load_h5(fname_hsk)

    data = OrderedDict({
            'time': {
                # 'data': data_hsk['tmhr']*3600.0,
                'data': data_ra['tmhr']*3600.0,
                'unit': 'second',
                'long_name': 'UTC time of measurement instant',
                'units': 'seconds since %s' % date,
                },

            'gps_lon': {
                # 'data': data_hsk['lon'],
                'data': data_ra['lon'],
                'long_name': 'longitude of aircraft',
                'units': 'degrees_east',
                },

            'gps_lat': {
                # 'data': data_hsk['lat'],
                'data': data_ra['lat'],
                'long_name': 'latitude of aircraft',
                'units': 'degrees_north',
                },

            'gps_alt': {
                # 'data': data_hsk['alt'],
                'data': data_ra['alt'],
                'long_name': 'altitude of aircraft',
                'units': 'meter',
                },

            'wvl_up': {
                'data': data_ra['nad/wvl'],
                'long_name': 'wavelength for spectral upwelling irradiance',
                'units': 'nm',
                },

            'wvl_dn': {
                'data': data_ra['zen/wvl'],
                'long_name': 'wavelength for spectral downwelling irradiance',
                'units': 'nm',
                },

            'f_dn': {
                'data': data_ra['zen/flux'],
                'ACVSN_standard_name': 'Rad_IrradianceDownwelling_InSitu_SP',
                'long_name': 'spectral downwelling irradiance',
                'units': 'W.m-2.nm-1',
                'coordinates': ['time', 'wvl_dn'],
                '_FillValue': 'nan',
                },

            'f_up': {
                'data': data_ra['nad/flux'],
                'ACVSN_standard_name': 'Rad_IrradianceUpwelling_InSitu_SP',
                'long_name': 'spectral upwelling irradiance',
                'units': 'W.m-2.nm-1',
                'coordinates': ['time', 'wvl_up'],
                '_FillValue': 'nan',
                },
            })

    for key in data.keys():
        data[key]['description'] = '%s: %s, %s' % (key, data[key]['units'], data[key]['long_name'])

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
                   data['time']['description'],                # Description or name of independent variable (This is the name chosen for the start time. It always refers to the number of seconds UTC from the start of the day on which measurements began. It should be noted here that the independent variable should monotonically increase even when crossing over to a second day.).
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

    global_attrs = {
            'ACVSN_standard_name_URL'        : '10.5067/DOC/ESCO/ESDS-RFC-043v1 (under User Resources)',
            'ACVSN_standard_name_version'    : '1.0',
            'Conventions'                    : 'CF-1.10',
            'Format'                         : 'HDF5',
            'PI_contact'                     : 'hong.chen@lasp.colorado.edu, sebastian.schmidt@lasp.colorado.edu',
            'PI_name'                        : 'Hong Chen, K. Sebastian Schmidt',
            'ProcessingLevel'                : 'L1',
            'VersionID'                      : 'R0',
            'aircraft_data_stream'           : 'IWG1',
            'associated_data'                : 'The navigation data (longitude, latitude, and altitude) are from MetNav RA/RB.',
            'data_processing_note'           : 'This version is based on pre-mission calibrations. Post-mission calibrations will be applied for the next public release (R1, expected end of March 2025). Attitude correction needs further investigation. %s' % comments_special,
            'data_product_groups'            : 'root',
            'data_use_guideline'             : 'For responsible scientific use of the data sets provided, data users are strongly encouraged to carefully study the file headers and directly consult with the instrument PIs. Please acknowledge the data source and offer co-authorship to relevant instrument PIs when appropriate.',
            'file_originator'                : 'Hong Chen, K. Sebastian Schmidt',
            'file_originator_contact'        : 'hong.chen@lasp.colorado.edu, sebastian.schmidt@lasp.colorado.edu',
            'flight_start_date'              : date_s,
            'geospatial_lat_max'             : '%.4fdegrees_north' % data['gps_lat']['data'].max(),
            'geospatial_lat_min'             : '%.4fdegrees_north' % data['gps_lat']['data'].min(),
            'geospatial_lon_max'             : '%.4fdegrees_east' % data['gps_lon']['data'].max(),
            'geospatial_lon_min'             : '%.4fdegrees_east' % data['gps_lon']['data'].min(),
            'history'                        : 'R0: First public data release.',
            'institution'                    : 'University of Colorado',
            'keywords'                       : 'radiation, irradiance, spectral, upwelling, downwelling',
            'last_modified_date'             : str(datetime.datetime.now()),
            'measurement_platform'           : 'NASA P3-B N426NA',
            'platform_identifier'            : 'P3B',
            'platform_type'                  : 'AirMobile',
            'project'                        : 'ARCSIX 2024',
            # 'references'                     : 'N/A',
            'source'                         : 'SSFR',
            'source_description'             : 'Solar Spectral Flux Radiometer',
            'summary'                        : 'ARCSIX SSFR measurement of spectral downwelling and upwelling irradiance on %s flight. Publication quality data.' % date_s,
            'time_coverage_end'              : str(ssfr.util.jday_to_dtime(data_ra['jday'][-1])),
            'time_coverage_resolution'       : '1 s',
            'time_coverage_start'            : str(ssfr.util.jday_to_dtime(data_ra['jday'][0])),
            'title'                          : 'ARCSIX SSFR measurement of spectral downwelling and upwelling irradiance',
            'unit_convention'                : 'http://codes.wmo.int/wmdr/unit',
            }

    dims = {}

    fname_h5 = '%s/%s-SSFR_%s_%s_%s.h5' % (fdir_out, cfg.common['mission'].upper(), cfg.common['platform'].upper(), date_s, version.upper())
    if run:
        f = h5py.File(fname_h5, 'w')

        for attr in global_attrs.keys():
            f.attrs[attr] = global_attrs[attr]

        for key in data.keys():
            dset = f.create_dataset(key, data=data[key]['data'], compression='gzip', compression_opts=9, chunks=True)
            for attr in data[key].keys():
                if attr not in ['data', 'description']:
                    dset.attrs[attr] = data[key][attr]
                if attr in ['coordinates']:
                    dset.attrs[attr] = ' '.join(data[key][attr])

            if key in ['time', 'wvl_dn', 'wvl_up']:
                dset.make_scale(key)
                dims[key] = dset

            if key in ['f_dn', 'f_up']:
                dset.dims[0].attach_scale(dims[data[key]['coordinates'][0]])
                dset.dims[1].attach_scale(dims[data[key]['coordinates'][1]])
        f.close()

    return fname_h5

def cdata_ssrr_archive_r0_from_ra(
        cfg,
        fname_ssrr_ra,
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
        version='R0',
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
    date = cfg.common['date']
    date_s = date.strftime('%Y%m%d')
    date_today = datetime.date.today()
    date_info  = '%4.4d, %2.2d, %2.2d, %4.4d, %2.2d, %2.2d' % (date.year, date.month, date.day, date_today.year, date_today.month, date_today.day)

    if cfg.ssrr['tag'].lower() == 'ssrr-a':
        instrument_info = 'SSRR-A (Solar Spectral Radiance Radiometer - Alvin)'
    else:
        instrument_info = 'SSRR-B (Solar Spectral Radiance Radiometer - Belana)'
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # version info
    #╭────────────────────────────────────────────────────────────────────────────╮#
    version = version.upper()
    version_info = {
            'RA': 'field data',
            'R0': 'First public release.',
            }
    version_info = version_info[version]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # data info
    #╭────────────────────────────────────────────────────────────────────────────╮#
    data_info = 'Shortwave Downwelling and Upwelling Spectral Radiance from %s %s' % (platform_info.upper(), instrument_info)
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
            'UNCERTAINTY': 'Nominal SSRR uncertainty (shortwave): nadir: N/A; zenith: N/A',
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
            # '20240530': 'Additional notes: noticed icing on dome outside zenith light collector after 20240530 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency.',
            # '20240531': 'Additional notes: encountered temperature control issue (after around 1:30 UTC) on 20240531 flight - downwelling and upwelling irradiance might be compromised after this time - contact the PI if you use this data.',
            # '20240730': 'Additional notes: noticed icing on dome inside zenith light collector after 20240730 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency.',
            # '20240801': 'Additional notes: noticed condensation on dome inside zenith light collector before 20240801 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency.',
            # '20240807': 'Additional notes: noticed condensation on dome inside zenith light collector after 20240807 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency.',
            # '20240808': 'Additional notes: noticed condensation on dome inside zenith light collector after 20240808 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency.',
            # '20240809': 'Additional notes: noticed condensation on dome inside zenith light collector after 20240809 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency.',
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
    data_ra = ssfr.util.load_h5(fname_ssrr_ra)

    # fname_hsk = 'data/arcsix/processed/ARCSIX-HSK_P3B_%s_v0.h5' % date_s
    # data_hsk = ssfr.util.load_h5(fname_hsk)

    data = OrderedDict({
            'time': {
                # 'data': data_hsk['tmhr']*3600.0,
                'data': data_ra['tmhr']*3600.0,
                'unit': 'second',
                'long_name': 'UTC time of measurement instant',
                'units': 'seconds since %s' % date,
                },

            'gps_lon': {
                # 'data': data_hsk['lon'],
                'data': data_ra['lon'],
                'long_name': 'longitude of aircraft',
                'units': 'degrees_east',
                },

            'gps_lat': {
                # 'data': data_hsk['lat'],
                'data': data_ra['lat'],
                'long_name': 'latitude of aircraft',
                'units': 'degrees_north',
                },

            'gps_alt': {
                # 'data': data_hsk['alt'],
                'data': data_ra['alt'],
                'long_name': 'altitude of aircraft',
                'units': 'meter',
                },

            'wvl_up': {
                'data': data_ra['nad/wvl'],
                'long_name': 'wavelength for spectral upwelling irradiance',
                'units': 'nm',
                },

            'wvl_dn': {
                'data': data_ra['zen/wvl'],
                'long_name': 'wavelength for spectral downwelling irradiance',
                'units': 'nm',
                },

            'i_dn': {
                'data': data_ra['zen/rad'],
                'ACVSN_standard_name': 'Rad_RadianceDownwellingZenith_InSitu_SP',
                'long_name': 'spectral downwelling radiance',
                'units': 'W.m-2.nm-1.sr-1',
                'coordinates': ['time', 'wvl_dn'],
                '_FillValue': 'nan',
                },

            'i_up': {
                'data': data_ra['nad/rad'],
                'ACVSN_standard_name': 'Rad_Radiance_InSitu_SP',
                'long_name': 'spectral upwelling radiance',
                'units': 'W.m-2.nm-1.sr-1',
                'coordinates': ['time', 'wvl_up'],
                '_FillValue': 'nan',
                },
            })

    for key in data.keys():
        data[key]['description'] = '%s: %s, %s' % (key, data[key]['units'], data[key]['long_name'])

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
                   data['time']['description'],                # Description or name of independent variable (This is the name chosen for the start time. It always refers to the number of seconds UTC from the start of the day on which measurements began. It should be noted here that the independent variable should monotonically increase even when crossing over to a second day.).
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

    global_attrs = {
            'ACVSN_standard_name_URL'        : '10.5067/DOC/ESCO/ESDS-RFC-043v1 (under User Resources)',
            'ACVSN_standard_name_version'    : '1.0',
            'Conventions'                    : 'CF-1.10',
            'Format'                         : 'HDF5',
            'PI_contact'                     : 'hong.chen@lasp.colorado.edu, sebastian.schmidt@lasp.colorado.edu',
            'PI_name'                        : 'Hong Chen, K. Sebastian Schmidt',
            'ProcessingLevel'                : 'L1',
            'VersionID'                      : 'R0',
            'aircraft_data_stream'           : 'IWG1',
            'associated_data'                : 'The navigation data (longitude, latitude, and altitude) are from MetNav RA/RB.',
            'data_processing_note'           : 'This version is based on pre-mission calibrations. Post-mission calibrations will be applied for the next public release (R1, expected end of March 2025). Attitude correction needs further investigation. %s' % comments_special,
            'data_product_groups'            : 'root',
            'data_use_guideline'             : 'For responsible scientific use of the data sets provided, data users are strongly encouraged to carefully study the file headers and directly consult with the instrument PIs. Please acknowledge the data source and offer co-authorship to relevant instrument PIs when appropriate.',
            'file_originator'                : 'Hong Chen, K. Sebastian Schmidt',
            'file_originator_contact'        : 'hong.chen@lasp.colorado.edu, sebastian.schmidt@lasp.colorado.edu',
            'flight_start_date'              : date_s,
            'geospatial_lat_max'             : '%.4fdegrees_north' % data['gps_lat']['data'].max(),
            'geospatial_lat_min'             : '%.4fdegrees_north' % data['gps_lat']['data'].min(),
            'geospatial_lon_max'             : '%.4fdegrees_east' % data['gps_lon']['data'].max(),
            'geospatial_lon_min'             : '%.4fdegrees_east' % data['gps_lon']['data'].min(),
            'history'                        : 'R0: First public data release.',
            'institution'                    : 'University of Colorado',
            'keywords'                       : 'radiation, radiance, spectral, upwelling, downwelling',
            'last_modified_date'             : str(datetime.datetime.now()),
            'measurement_platform'           : 'NASA P3-B N426NA',
            'platform_identifier'            : 'P3B',
            'platform_type'                  : 'AirMobile',
            'project'                        : 'ARCSIX 2024',
            # 'references'                     : 'N/A',
            'source'                         : 'SSFR',
            'source_description'             : 'Solar Spectral Radiance Radiometer',
            'summary'                        : 'ARCSIX SSFR measurement of spectral downwelling and upwelling radiance on %s flight. Publication quality data.' % date_s,
            'time_coverage_end'              : str(ssfr.util.jday_to_dtime(data_ra['jday'][-1])),
            'time_coverage_resolution'       : '1 s',
            'time_coverage_start'            : str(ssfr.util.jday_to_dtime(data_ra['jday'][0])),
            'title'                          : 'ARCSIX SSFR measurement of spectral downwelling and upwelling radiance',
            'unit_convention'                : 'http://codes.wmo.int/wmdr/unit',
            }

    dims = {}

    fname_h5 = '%s/%s-SSRR_%s_%s_%s.h5' % (fdir_out, cfg.common['mission'].upper(), cfg.common['platform'].upper(), date_s, version.upper())
    if run:
        f = h5py.File(fname_h5, 'w')

        for attr in global_attrs.keys():
            f.attrs[attr] = global_attrs[attr]

        for key in data.keys():
            dset = f.create_dataset(key, data=data[key]['data'], compression='gzip', compression_opts=9, chunks=True)
            for attr in data[key].keys():
                if attr not in ['data', 'description']:
                    dset.attrs[attr] = data[key][attr]
                if attr in ['coordinates']:
                    dset.attrs[attr] = ' '.join(data[key][attr])

            if key in ['time', 'wvl_dn', 'wvl_up']:
                dset.make_scale(key)
                dims[key] = dset

            if key in ['i_dn', 'i_up']:
                dset.dims[0].attach_scale(dims[data[key]['coordinates'][0]])
                dset.dims[1].attach_scale(dims[data[key]['coordinates'][1]])
        f.close()

    return fname_h5
#╰────────────────────────────────────────────────────────────────────────────╯#

# main program
#╭────────────────────────────────────────────────────────────────────────────╮#
def main_process_data_archive_ra(cfg, fdir_out='./', run=True):

    """
    ra: in-field data to be uploaded to https://www-air.larc.nasa.gov/cgi-bin/ArcView/arcsix
    """

    # fdir_out = './'
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    # HSR1 RA
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsr1_ra = cdata_hsr1_archive(cfg, cfg.hsr1['fname_v2'],
            fdir_out=fdir_out, run=run)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # SSFR RA
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ssfr_ra = cdata_ssfr_archive(cfg, cfg.ssfr['fname_v2'],
            fdir_out=fdir_out, run=run)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # SSRR RA
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ssrr_ra = cdata_ssrr_archive(cfg, cfg.ssrr['fname_v1'],
            fdir_out=fdir_out, run=run)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return fname_hsr1_ra, fname_ssfr_ra, fname_ssrr_ra
#╰────────────────────────────────────────────────────────────────────────────╯#

#╭────────────────────────────────────────────────────────────────────────────╮#
def main_process_data_archive_r0_from_ra(cfg, fdir_out='./', run=True):

    """
    ra: in-field data to be uploaded to https://www-air.larc.nasa.gov/cgi-bin/ArcView/arcsix
    """

    # fdir_out = './'
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    fname_hsr1_ra, fname_ssfr_ra, fname_ssrr_ra = main_process_data_archive_ra(cfg, fdir_out=fdir_out, run=False)

    # HSR1 R0
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_hsr1_r0 = cdata_hsr1_archive_r0_from_ra(cfg, fname_hsr1_ra,
            fdir_out=fdir_out, run=run)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # SSFR R0
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ssfr_r0 = cdata_ssfr_archive_r0_from_ra(cfg, fname_ssfr_ra,
            fdir_out=fdir_out, run=run)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # SSRR R0
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ssrr_r0 = cdata_ssrr_archive_r0_from_ra(cfg, fname_ssrr_ra,
            fdir_out=fdir_out, run=run)
    #╰────────────────────────────────────────────────────────────────────────────╯#
#╰────────────────────────────────────────────────────────────────────────────╯#


if __name__ == '__main__':


    # dates
    #╭────────────────────────────────────────────────────────────────────────────╮#
    dates = [
             datetime.datetime(2024, 5, 24),
            #  datetime.datetime(2024, 5, 28),
            #  datetime.datetime(2024, 5, 30), # ARCSIX-1 science flight #2, cloud wall, operator - Vikas Nataraja
            #  datetime.datetime(2024, 5, 31),
            #  datetime.datetime(2024, 6, 3),  # ARCSIX-1 science flight #4, cloud wall, operator - Vikas Nataraja
            #  datetime.datetime(2024, 6, 5),
            #  datetime.datetime(2024, 6, 6),
            #  datetime.datetime(2024, 6, 7),  # ARCSIX-1 science flight #7, cloud wall, operator - Vikas Nataraja, Arabella Chamberlain
            #  datetime.datetime(2024, 6, 10), # ARCSIX-1 science flight #8, operator - Jeffery Drouet
            #  datetime.datetime(2024, 6, 11), # ARCSIX-1 science flight #9, operator - Arabella Chamberlain, Sebastian Becker
            #  datetime.datetime(2024, 6, 13), # ARCSIX-1 science flight #10, operator - Arabella Chamberlain
            #  datetime.datetime(2024, 7, 22),
            #  datetime.datetime(2024, 7, 25), # ARCSIX-2 science flight #11, cloud walls, operator - Arabella Chamberlain
            #  datetime.datetime(2024, 7, 29), # ARCSIX-2 science flight #12, clear-sky BRDF, operator - Ken Hirata, Vikas Nataraja
            #  datetime.datetime(2024, 7, 30), # ARCSIX-2 science flight #13, clear-sky BRDF, operator - Ken Hirata
            #  datetime.datetime(2024, 8, 1),
            #  datetime.datetime(2024, 8, 2),  # ARCSIX-2 science flight #15, cloud walls, operator - Ken Hirata, Arabella Chamberlain
            #  datetime.datetime(2024, 8, 7),  # ARCSIX-2 science flight #16, cloud walls, operator - Arabella Chamberlain
            #  datetime.datetime(2024, 8, 8),  # ARCSIX-2 science flight #17, cloud walls, operator - Arabella Chamberlain
            #  datetime.datetime(2024, 8, 9),  # ARCSIX-2 science flight #18, cloud walls, operator - Arabella Chamberlain
            #  datetime.datetime(2024, 8, 15), # ARCSIX-2 science flight #19, cloud walls, operator - Ken Hirata, Sebastian Schmidt
            #  datetime.datetime(2024, 8, 16),
            ]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    run_hsr1 = True
    # run_hsr1 = False
    run_ssfr = True
    # run_ssfr = False
    run_ssrr = True
    # run_ssrr = False

    for date in dates[::-1]:


        # configuration file
        #╭────────────────────────────────────────────────────────────────────────────╮#
        global cfg
        fname_cfg = date.strftime('cfg_%Y%m%d')
        cfg = importlib.import_module(fname_cfg)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # fdir_out = '/Users/kehi6101/Downloads/ssfr_test/arcsix/archive'
        fdir_out = cfg.fdir_out
        if not os.path.exists(fdir_out):
            os.makedirs(fdir_out)
        
        # Step 1 (RA)
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # HSR1 RA
        if run_hsr1:
            fname_hsr1_ra = cdata_hsr1_archive(cfg, cfg.hsr1['fname_v2'],
                    fdir_out=fdir_out, run=True)

        # SSFR RA
        if run_ssfr:
            fname_ssfr_ra = cdata_ssfr_archive(cfg, cfg.ssfr['fname_v2'],
                    fdir_out=fdir_out, run=True)

        # SSRR RA
        if run_ssrr:
            fname_ssrr_ra = cdata_ssrr_archive(cfg, cfg.ssrr['fname_v1'],
                    fdir_out=fdir_out, run=True)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        # Step 2 (R0)
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # HSR1 R0
        if run_hsr1:
            fname_hsr1_r0 = cdata_hsr1_archive_r0_from_ra(cfg, fname_hsr1_ra,
                    fdir_out=fdir_out, run=True)

        # SSFR R0
        if run_ssfr:
            fname_ssfr_r0 = cdata_ssfr_archive_r0_from_ra(cfg, fname_ssfr_ra,
                    fdir_out=fdir_out, run=True)

        # SSRR R0
        if run_ssrr:
            fname_ssrr_r0 = cdata_ssrr_archive_r0_from_ra(cfg, fname_ssrr_ra,
                    fdir_out=fdir_out, run=True)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        pass
