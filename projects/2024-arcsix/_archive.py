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
_MISSION_ = 'arcsix'
_SSFR1_ = 'ssfr-a'
_SSFR2_ = 'ssfr-b'
_FNAMES_ = {}
#╰────────────────────────────────────────────────────────────────────────────╯#


# functions for processing HSR1 and SSFR (RA version)
#╭────────────────────────────────────────────────────────────────────────────╮#
def cdata_hsr1_archive_ra(
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

def cdata_ssfr_archive_ra(
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


# functions for processing HSR1 and SSFR (R0 version)
#╭────────────────────────────────────────────────────────────────────────────╮#
def cdata_hsr1_archive_r0_from_ra(
        date,
        fname_hsr1_ra,
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
        version='R0',
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
            '20240530': 'Noticed icing on dome after 20240530 flight - this might affect the downwelling total and diffuse irradiance - compare with SSFR measurements for consistency',
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


    data = OrderedDict({
            'time': {
                'data': data_ra['Time_Start']*3600.0,
                'unit': 'second',
                'long_name': 'UTC time of measurement instant',
                'units': 'seconds since %s' % date,
                },

            'lon': {
                'data': data_ra['lon'],
                'long_name': 'longitude of aircraft',
                'units': 'degrees_east',
                },

            'lat': {
                'data': data_ra['lat'],
                'long_name': 'latitude of aircraft',
                'units': 'degrees_north',
                },

            'alt': {
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
                'units': 'W m^-2 nm^-1',
                'coordinates': ['time', 'wvl_dn_dif'],
                '_FillValue': 'nan',
                },

            'f_dn_tot': {
                'data': data_ra['tot/flux'],
                'ACVSN_standard_name': 'Rad_IrradianceDownwelling_InSitu_SP',
                'long_name': 'spectral downwelling total irradiance',
                'units': 'W m^-2 nm^-1',
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

    # print(header)
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
            'geospatial_lat_max'             : '%.4fdegrees_north' % data['lat']['data'].max(),
            'geospatial_lat_min'             : '%.4fdegrees_north' % data['lat']['data'].min(),
            'geospatial_lon_max'             : '%.4fdegrees_east' % data['lon']['data'].max(),
            'geospatial_lon_min'             : '%.4fdegrees_east' % data['lon']['data'].min(),
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

    fname_h5 = '%s/%s-HSR1_%s_%s_%s.h5' % (fdir_out, 'arcsix'.upper(), 'p3b'.upper(), date_s, version.upper())
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
        date,
        fname_ssfr_ra,
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
    date_s = date.strftime('%Y%m%d')
    date_today = datetime.date.today()
    date_info  = '%4.4d, %2.2d, %2.2d, %4.4d, %2.2d, %2.2d' % (date.year, date.month, date.day, date_today.year, date_today.month, date_today.day)

    which_ssfr = os.path.basename(fname_ssfr_ra).split('_')[0].replace('%s-' % _MISSION_.upper(), '').lower()
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
            '20240530': 'Noticed icing on dome outside zenith light collector after 20240530 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency',
            '20240531': 'Encountered temperature control issue (after around 1:30 UTC) on 20240531 flight - downwelling and upwelling irradiance might be compromised after this time - contact the PI if you use this data',
            '20240730': 'Noticed icing on dome inside zenith light collector after 20240730 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency',
            '20240801': 'Noticed condensation on dome inside zenith light collector before 20240801 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency',
            '20240807': 'Noticed condensation on dome inside zenith light collector after 20240807 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency',
            '20240808': 'Noticed condensation on dome inside zenith light collector after 20240808 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency',
            '20240809': 'Noticed condensation on dome inside zenith light collector after 20240809 flight - this might affect the downwelling irradiance - compare with HSR measurements for consistency',
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
    data_ra['lon'][data_ra['lon']>180.0] -= 360.0

    data = OrderedDict({
            'time': {
                'data': data_ra['Time_Start']*3600.0,
                'unit': 'second',
                'long_name': 'UTC time of measurement instant',
                'units': 'seconds since %s' % date,
                },

            'lon': {
                'data': data_ra['lon'],
                'long_name': 'longitude of aircraft',
                'units': 'degrees_east',
                },

            'lat': {
                'data': data_ra['lat'],
                'long_name': 'latitude of aircraft',
                'units': 'degrees_north',
                },

            'alt': {
                'data': data_ra['alt'],
                'long_name': 'altitude of aircraft',
                'units': 'meter',
                },

            'wvl_up': {
                'data': data_ra['zen/wvl'],
                'long_name': 'wavelength for spectral upwelling irradiance',
                'units': 'nm',
                },

            'wvl_dn': {
                'data': data_ra['nad/wvl'],
                'long_name': 'wavelength for spectral downwelling irradiance',
                'units': 'nm',
                },

            'f_dn': {
                'data': data_ra['zen/flux'],
                'ACVSN_standard_name': 'Rad_IrradianceDownwelling_InSitu_SP',
                'long_name': 'spectral downwelling irradiance',
                'units': 'W m^-2 nm^-1',
                'coordinates': ['time', 'wvl_dn'],
                '_FillValue': 'nan',
                },

            'f_up': {
                'data': data_ra['nad/flux'],
                'ACVSN_standard_name': 'Rad_IrradianceUpwelling_InSitu_SP',
                'long_name': 'spectral upwelling irradiance',
                'units': 'W m^-2 nm^-1',
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
            'geospatial_lat_max'             : '%.4fdegrees_north' % data['lat']['data'].max(),
            'geospatial_lat_min'             : '%.4fdegrees_north' % data['lat']['data'].min(),
            'geospatial_lon_max'             : '%.4fdegrees_east' % data['lon']['data'].max(),
            'geospatial_lon_min'             : '%.4fdegrees_east' % data['lon']['data'].min(),
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

    fname_h5 = '%s/%s-SSFR_%s_%s_%s.h5' % (fdir_out, 'arcsix'.upper(), 'p3b'.upper(), date_s, version.upper())
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
#╰────────────────────────────────────────────────────────────────────────────╯#


# main program
#╭────────────────────────────────────────────────────────────────────────────╮#
def main_process_data_archive_ra(date, run=True):

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

def main_process_data_archive_r0_from_ra(date, run=True):

    """
    ra: in-field data to be uploaded to https://www-air.larc.nasa.gov/cgi-bin/ArcView/arcsix
    """

    date_s = date.strftime('%Y%m%d')

    fdir_out = './'
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    # main_process_data_v0(date, run=False)
    # main_process_data_v1(date, run=False)
    # main_process_data_v2(date, run=False)

    # HSR1 R0
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ra = 'data/arcsix/processed/ARCSIX-HSR1_P3B_%s_RA.h5' % date_s
    fname_hsr1_ra = cdata_hsr1_archive_r0_from_ra(date, fname_ra,
            fdir_out=fdir_out, run=run)
    #╰────────────────────────────────────────────────────────────────────────────╯#
    _FNAMES_['%s_hsr1_ra' % date_s] = fname_hsr1_ra



    # SSFR R0
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_ra = 'data/arcsix/processed/ARCSIX-SSFR_P3B_%s_RA.h5' % date_s
    fname_ssfr_ra = cdata_ssfr_archive_r0_from_ra(date, fname_ra,
            fdir_out=fdir_out, run=run)
    #╰────────────────────────────────────────────────────────────────────────────╯#
    _FNAMES_['%s_ssfr_ra' % date_s] = fname_ssfr_ra
#╰────────────────────────────────────────────────────────────────────────────╯#

if __name__ == '__main__':


    # dates
    #╭────────────────────────────────────────────────────────────────────────────╮#
    dates = [
             datetime.datetime(2024, 5, 28),
             datetime.datetime(2024, 5, 30), # ARCSIX-1 science flight #2, cloud wall, operator - Vikas Nataraja
             datetime.datetime(2024, 5, 31), # ARCSIX-1 science flight #3, bowling alley; surface BRDF, operator - Vikas Nataraja
             datetime.datetime(2024, 6, 3),  # ARCSIX-1 science flight #4, cloud wall, operator - Vikas Nataraja
             datetime.datetime(2024, 6, 5),  # ARCSIX-1 science flight #5, bowling alley; surface BRDF, operator - Vikas Nataraja, Sebastian Becker
             datetime.datetime(2024, 6, 6),  # ARCSIX-1 science flight #6, cloud wall, operator - Vikas Nataraja, Jeffery Drouet
             datetime.datetime(2024, 6, 7),  # ARCSIX-1 science flight #7, cloud wall, operator - Vikas Nataraja, Arabella Chamberlain
             datetime.datetime(2024, 6, 10), # ARCSIX-1 science flight #8, operator - Jeffery Drouet
             datetime.datetime(2024, 6, 11), # ARCSIX-1 science flight #9, operator - Arabella Chamberlain, Sebastian Becker
             datetime.datetime(2024, 6, 13), # ARCSIX-1 science flight #10, operator - Arabella Chamberlain
             datetime.datetime(2024, 7, 25), # ARCSIX-2 science flight #11, cloud walls, operator - Arabella Chamberlain
             datetime.datetime(2024, 7, 29), # ARCSIX-2 science flight #12, clear-sky BRDF, operator - Ken Hirata, Vikas Nataraja
             datetime.datetime(2024, 7, 30), # ARCSIX-2 science flight #13, clear-sky BRDF, operator - Ken Hirata
             datetime.datetime(2024, 8, 1),  # ARCSIX-2 science flight #14, cloud walls, operator - Ken Hirata
             datetime.datetime(2024, 8, 2),  # ARCSIX-2 science flight #15, cloud walls, operator - Ken Hirata, Arabella Chamberlain
             datetime.datetime(2024, 8, 7),  # ARCSIX-2 science flight #16, cloud walls, operator - Arabella Chamberlain
             datetime.datetime(2024, 8, 8),  # ARCSIX-2 science flight #17, cloud walls, operator - Arabella Chamberlain
             datetime.datetime(2024, 8, 9),  # ARCSIX-2 science flight #18, cloud walls, operator - Arabella Chamberlain
             datetime.datetime(2024, 8, 15), # ARCSIX-2 science flight #19, cloud walls, operator - Ken Hirata, Sebastian Schmidt
            ]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    for date in dates[::-1]:

        # configuration file
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # global cfg
        # fname_cfg = date.strftime('cfg_%Y%m%d')
        # cfg = importlib.import_module(fname_cfg)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        #╭────────────────────────────────────────────────────────────────────────────╮#
        main_process_data_archive_r0_from_ra(date, run=True)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        pass
