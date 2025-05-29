import os
import sys
import glob
import h5py
import struct
import fnmatch
import pysolar
from tqdm import tqdm
from scipy import interpolate
from collections import OrderedDict
import xml.etree.ElementTree as ET
import numpy as np
import datetime
from scipy import stats

import ssfr



__all__ = [
        'get_all_files',
        'get_all_folders',
        'if_file_exists',
        'cal_heading',
        'cal_solar_angles',
        'cal_solar_factor',
        'cal_step_offset',
        'prh2za',
        'muslope',
        'dtime_to_jday',
        'jday_to_dtime',
        'interp',
        'load_h5',
        'save_h5',
        'get_solar_kurudz',
        'get_slit_func',
        'cal_weighted_flux',
        'read_ict',
        'write_ict',
        'read_iwg_nsrc',
        'read_iwg_mts',
        'read_cabin',
        ]



def get_all_files(fdir, pattern='*'):

    fnames = []
    for fdir_root, fdir_sub, fnames_tmp in os.walk(fdir):
        for fname_tmp in fnames_tmp:
            if fnmatch.fnmatch(fname_tmp, pattern):
                fnames.append(os.path.join(fdir_root, fname_tmp))
    return sorted(fnames)

def get_all_folders(fdir, pattern='*'):

    folders = [os.path.abspath(fdir0[0]) for fdir0 in os.walk(fdir) if fnmatch.fnmatch(fdir0[0], pattern)]

    return folders

def if_file_exists(fname, exitTag=True):

    """
    Check whether file exists.

    Input:
        fname: file path of the data
    Output:
        None
    """

    if not os.path.exists(fname):
        if exitTag is True:
            exit("Error   [if_file_exists]: cannot find '{fname}'".format(fname=fname))
        else:
            print("Warning [if_file_exists]: cannot find '{fname}'".format(fname=fname))

def cal_heading(lon, lat):

    dx = lat[1:]-lat[:-1]
    dy = lon[1:]-lon[:-1]

    heading = np.rad2deg(np.arctan2(dy, dx))
    heading = np.append(heading[0], heading) % 360.0

    return heading

def cal_solar_angles(julian_day, longitude, latitude, altitude, verbose=ssfr.common.karg['verbose']):

    dateRef = datetime.datetime(1, 1, 1)
    jdayRef = 1.0

    sza = np.zeros_like(julian_day)
    saa = np.zeros_like(julian_day)

    if verbose:
        print('Message [cal_solar_angles]: Calculating solar zenith and azimuth angles (total of %d) ...' % sza.size)

    for i in tqdm(range(julian_day.size)):

        jday = julian_day[i]

        try:
            dtime_i = (dateRef + datetime.timedelta(days=jday-jdayRef)).replace(tzinfo=datetime.timezone.utc)

            sza_i = 90.0 - pysolar.solar.get_altitude(latitude[i], longitude[i], dtime_i, elevation=altitude[i])
            if sza_i < 0.0 or sza_i > 90.0:
                sza_i = np.nan
            sza[i] = sza_i

            saa_i = pysolar.solar.get_azimuth(latitude[i], longitude[i], dtime_i, elevation=altitude[i])
            if saa_i >= 0.0:
                if 0.0<=saa_i<=180.0:
                    saa_i = 180.0 - saa_i
                elif 180.0<saa_i<=360.0:
                    saa_i = 540.0 - saa_i
                else:
                    saa_i = np.nan
            elif saa_i < 0.0:
                if -180.0<=saa_i<0.0:
                    saa_i = -saa_i + 180.0
                elif -360.0<=saa_i<-180.0:
                    saa_i = -saa_i - 180.0
                else:
                    saa_i = np.nan
            saa[i] = saa_i

        except Exception as error:
            print('\nError [cal_solar_angles]: received the following error for [i=%d] ...' % i)
            print(error)
            sza[i] = np.nan
            saa[i] = np.nan

    return sza, saa

def cal_solar_factor(dtime):

    """
    Calculate solar factor that accounts for Sun-Earth distance
    Input:
        dtime: datetime.datetime object
    Output:
        solfac: solar factor
    """

    doy = dtime.timetuple().tm_yday
    eps = 0.0167086
    perh= 4.0
    rsun = (1.0 - eps*np.cos(0.017202124161707175*(doy-perh)))
    solfac = 1.0/(rsun**2)

    return solfac

def cal_step_offset(x_ref, x_target, fill_value=np.nan, offset_range=[-900, 900]):

    from scipy.stats import pearsonr
    from scipy.ndimage import shift

    logic = (np.logical_not(np.isnan(x_ref))) & (np.logical_not(np.isnan(x_target)))
    x_ref    = x_ref[logic]
    x_target = x_target[logic]
    x_ref    = x_ref/x_ref.max()
    x_target = x_target/x_target.max()

    offsets = np.arange(offset_range[0], offset_range[1], dtype=np.float64)
    cross_corr = np.zeros_like(offsets)
    for i, offset in enumerate(offsets):
        x0 = shift(x_target, offset, cval=fill_value)
        logic = (np.logical_not(np.isnan(x0))) & (np.logical_not(np.isnan(x_ref)))
        coef = pearsonr(x_ref[logic], x0[logic])
        cross_corr[i] = coef[0]

    step_offset = offsets[np.argmax(cross_corr)]

    return step_offset

def prh2za(ang_pit, ang_rol, ang_head, is_rad=False):

    """
    input:
    ang_pit   (Pitch)   [deg]: positive (+) values indicate nose up (tail down)
    ang_rol   (Roll)    [deg]: positive (+) values indicate right wing down (left side up)
    ang_head  (Heading) [deg]: positive (+) values clockwise, w.r.t. north

    "vec": normal vector of the surface of the sensor

    return:
    ang_zenith : angle of "vec" [deg]
    ang_azimuth: angle of "vec" [deg]: positive (+) values clockwise, w.r.t. north
    """

    if not is_rad:
        rad_pit  = np.deg2rad(ang_pit)
        rad_rol  = np.deg2rad(ang_rol)
        rad_head = np.deg2rad(ang_head)

    uz =  np.cos(rad_rol)*np.cos(rad_pit)
    ux =  np.sin(rad_rol)
    uy = -np.cos(rad_rol)*np.sin(rad_pit)

    vz = uz.copy()
    vx = ux*np.cos(rad_head) + uy*np.sin(rad_head)
    vy = uy*np.cos(rad_head) - ux*np.sin(rad_head)

    ang_zenith  = np.rad2deg(np.arccos(vz))
    ang_azimuth = np.rad2deg(np.arctan2(vx,vy))

    ang_azimuth[ang_azimuth<0.0] += 360.0

    return ang_zenith, ang_azimuth

def muslope(sza, saa, iza, iaa, is_rad=False):

    if not is_rad:
        rad_sza = np.deg2rad(sza)
        rad_saa = np.deg2rad(saa)
        rad_iza = np.deg2rad(iza)
        rad_iaa = np.deg2rad(iaa)

    zs = np.cos(rad_sza)
    ys = np.sin(rad_sza) * np.cos(rad_saa)
    xs = np.sin(rad_sza) * np.sin(rad_saa)

    zi = np.cos(rad_iza)
    yi = np.sin(rad_iza) * np.cos(rad_iaa)
    xi = np.sin(rad_iza) * np.sin(rad_iaa)

    mu = xs*xi + ys*yi + zs*zi

    return mu

def load_h5(fname):

    def get_variable_names(obj, prefix=''):

        """
        Purpose: Walk through the file and extract information of data groups and data variables

        Input: h5py file object <f>, e.g., f = h5py.File('file.h5', 'r')

        Outputs:
            data variable path in the format of <['group1/variable1']> to
            mimic the style of accessing HDF5 data variables using h5py, e.g.,
            <f['group1/variable1']>
        """

        for key in obj.keys():

            item = obj[key]
            path = '{prefix}/{key}'.format(prefix=prefix, key=key)
            if isinstance(item, h5py.Dataset):
                yield path
            elif isinstance(item, h5py.Group):
                yield from get_variable_names(item, prefix=path)

    data = {}
    f = h5py.File(fname, 'r')
    keys = get_variable_names(f)
    for key in keys:
        data[key[1:]] = f[key[1:]][...]
    f.close()
    return data

def save_h5(fname, data):

    data = {}
    f = h5py.File(fname, 'w')
    for key in data.keys():
        f[key] = data[key]
    f.close()

    print('Message [save_h5]: Data has been successfully saved into \'%s\'.' % fname)

def get_slit_func(wvl, slit_func_file=None, wvl_joint=950.0):

    if slit_func_file is None:
        if wvl <= wvl_joint:
            slit_func_file = '%s/slit/vis_0.1nm_s.dat' % ssfr.common.fdir_data
        else:
            slit_func_file = '%s/slit/nir_0.1nm_s.dat' % ssfr.common.fdir_data

    data_slt = np.loadtxt(slit_func_file)

    return data_slt

def get_solar_kurudz(kurudz_file=None):

    if kurudz_file is None:
        kurudz_file = '%s/solar/kurudz_0.1nm.dat' % ssfr.common.fdir_data

    data_sol = np.loadtxt(kurudz_file)
    data_sol[:, 1] /= 1000.0

    return data_sol

def cal_weighted_flux(wvl, data_wvl, data_flux, slit_func_file=None, wvl_joint=950.0):

    data_slt = get_slit_func(wvl, slit_func_file=slit_func_file, wvl_joint=wvl_joint)

    weights  = data_slt[:, 1]
    wvl_x    = data_slt[:, 0] + wvl
    flux     = np.average(np.interp(wvl_x, data_wvl, data_flux), weights=weights)

    return flux

def dtime_to_jday(dtime):

    jday = (dtime - datetime.datetime(1, 1, 1)).total_seconds()/86400.0 + 1.0

    return jday

def jday_to_dtime(jday):

    dtime = datetime.datetime(1, 1, 1) + datetime.timedelta(seconds=np.round(((jday-1)*86400.0), decimals=0))

    return dtime

def interp(x, x0, y0, mode='linear'):

    if mode == 'nearest':
        f = interpolate.interp1d(x0, y0, bounds_error=False, kind=mode)
    else:
        logic = (~np.isnan(x0) & ~np.isnan(y0))
        f = interpolate.interp1d(x0[logic], y0[logic], bounds_error=False, kind=mode)

    return f(x)

def var_info_ict(line):

    words = line.strip().split(',')
    vname = words[0].strip()
    unit  = ','.join([word.strip() for word in words[1:]])
    return vname, unit

def read_ict(fname, tmhr_range=None, Nskip=7, lower=True):

    f = open(fname, 'r')

    # read first line to get the number of lines to skip
    firstLine   = f.readline()
    skip_header = int(firstLine.split(',')[0])

    vnames = []
    units  = []
    for i in range(Nskip):
        f.readline()
    vname0, unit0 = var_info_ict(f.readline())
    vnames.append(vname0); units.append(unit0)

    Nvar = int(f.readline())
    f.readline()
    fill_values = np.array([float(word) for word in f.readline().strip().split(',')], dtype=np.float64)

    for i in range(Nvar):
        vname0, unit0 = var_info_ict(f.readline())
        vnames.append(vname0); units.append(unit0)
    f.close()

    data_all = np.genfromtxt(fname, skip_header=skip_header, delimiter=',', invalid_raise=False)

    data = OrderedDict()
    data['tmhr'] = dict(data=data_all[:, 0]/3600.0, units='Hour')

    if tmhr_range != None:
        logic = (data['tmhr']['data']>=tmhr_range[0]) & (data['tmhr']['data']<=tmhr_range[1])
        for i, vname in enumerate(vnames):
            data0 = data_all[:, i][logic]
            if i > 0:
                data0[data0==fill_values[i-1]] = np.nan
            if lower:
                vname = vname.lower()
            data[vname] = dict(data=data0, units=units[i])
    else:
        for i, vname in enumerate(vnames):
            data0 = data_all[:, i]
            if i > 0:
                data0[data0==fill_values[i-1]] = np.nan
            if lower:
                vname = vname.lower()
            data[vname] = dict(data=data0, units=units[i])

    return data

def write_ict(
        date,
        data,
        fname,
        comments = None,
        platform_info = 'p3',
        principal_investigator_info = 'Schmidt, Sebastian',
        affiliation_info = 'University of Colorado Boulder',
        instrument_info = 'Solar Spectral Flux Radiometer',
        mission_info = 'CAMP2Ex 2019',
        project_info = '',
        file_format_index = '1001',
        file_volume_number = '1, 1',
        data_interval = '1.0',
        scale_factor = '1.0',
        fill_value = '-9999.0',
        special_comments = '',
        ):

    data_info = 'Shortwave Spectral Irradiance from %s %s' % (platform_info.upper(), instrument_info)

    date_today = datetime.date.today()
    date_info  = '%4.4d, %2.2d, %2.2d, %4.4d, %2.2d, %2.2d' % (date.year, date.month, date.day, date_today.year, date_today.month, date_today.day)

    Nvar = len(data.keys())

    if special_comments != '':
        Nspecial = len(special_comments.split('\n'))
    else:
        Nspecial = 0

    normal_comments = '\n'.join(['%s: %s' % (var0, comments[var0]) for var0 in comments.keys()])
    normal_comments = '%s\n%s' % (normal_comments, ','.join(['%16s' % var0 for var0 in data.keys()]))
    Nnormal  = len(normal_comments.split('\n'))

    header_list = [file_format_index,
                   principal_investigator_info,
                   affiliation_info,       # Organization/affiliation of PI.
                   data_info,              # Data source description (e.g., instrument name, platform name, model name, etc.).
                   mission_info,           # Mission name (usually the mission acronym).
                   file_volume_number,     # File volume number, number of file volumes (these integer values are used when the data require more than one file per day; for data that require only one file these values are set to 1, 1) - comma delimited.
                   date_info,              # UTC date when data begin, UTC date of data reduction or revision - comma delimited (yyyy, mm, dd, yyyy, mm, dd).
                   data_interval,          # Data Interval (This value describes the time spacing (in seconds) between consecutive data records. It is the (constant) interval between values of the independent variable. For 1 Hz data the data interval value is 1 and for 10 Hz data the value is 0.1. All intervals longer than 1 second must be reported as Start and Stop times, and the Data Interval value is set to 0. The Mid-point time is required when it is not at the average of Start and Stop times. For additional information see Section 2.5 below.).
                   data['Time_start']['description'],                # Description or name of independent variable (This is the name chosen for the start time. It always refers to the number of seconds UTC from the start of the day on which measurements began. It should be noted here that the independent variable should monotonically increase even when crossing over to a second day.).
                   str(Nvar-1),                                      # Number of variables (Integer value showing the number of dependent variables: the total number of columns of data is this value plus one.).
                   ', '.join([scale_factor for i in range(Nvar-1)]), # Scale factors (1 for most cases, except where grossly inconvenient) - comma delimited.
                   ', '.join([fill_value for i in range(Nvar-1)]),   # Missing data indicators (This is -9999 (or -99999, etc.) for any missing data condition, except for the main time (independent) variable which is never missing) - comma delimited.
                   '\n'.join([data[vname]['description'] for vname in data.keys() if vname != 'Time_start']), # Variable names and units (Short variable name and units are required, and optional long descriptive name, in that order, and separated by commas. If the variable is unitless, enter the keyword "none" for its units. Each short variable name and units (and optional long name) are entered on one line. The short variable name must correspond exactly to the name used for that variable as a column header, i.e., the last header line prior to start of data.).
                   str(Nspecial),                                   # Number of SPECIAL comment lines (Integer value indicating the number of lines of special comments, NOT including this line.).
                   special_comments,
                   str(Nnormal),
                   normal_comments
                ]


    header = '\n'.join([header0 for header0 in header_list if header0 != ''])

    Nline = len(header.split('\n'))
    header = '%d, %s' % (Nline, header)

    Ndata = data['Time_start']['data'].size
    data_all = np.concatenate([data[vname]['data'] for vname in data.keys()]).reshape((Nvar, Ndata)).T
    data_all[np.isnan(data_all)] = float(fill_value)

    with open(fname, 'w') as f:
        f.write(header)
        f.write('\n')
        for i in range(Ndata):
            line = ','.join(['%16.8f' % data0 for data0 in data_all[i, :]])
            f.write(line)
            f.write('\n')

    return fname





def read_iwg_nsrc(fname, date_ref=None, tmhr_range=None):

    fname_xml = '%s.xml' % fname

    vnames = []
    with open(fname_xml, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'parameter' in line and 'id=' in line:
                vnames.append(line.split('"')[1])

    vnames[0] = 'jday'

    Nvar = len(vnames)
    convert_func = lambda x: (datetime.datetime.strptime(x.decode('utf-8').split('.')[0], '%Y-%m-%dT%H:%M:%S')-datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0
    data_all = np.genfromtxt(fname, delimiter=',', names=vnames, converters={1:convert_func}, usecols=range(1, Nvar+1))

    if date_ref is None:
        jday_int = np.int_(data_all['jday'])
        jday_unique, counts = np.unique(jday_int, return_counts=True)
        jdayRef = jday_unique[np.argmax(counts)]
    else:
        jdayRef = (date_ref - datetime.datetime(1, 1, 1)).total_seconds()/86400.0 + 1.0

    data = OrderedDict()
    data['tmhr'] = dict(data=(data_all['jday']-jdayRef)*24.0, units='Hour')

    if tmhr_range != None:
        logic = (data['tmhr']['data']>=tmhr_range[0]) & (data['tmhr']['data']<=tmhr_range[1])
        for i, vname in enumerate(vnames):
            data0 = data_all[vname][logic]
            data[vname.lower()] = dict(data=data0, units='N/A')
    else:
        for i, vname in enumerate(vnames):
            data0 = data_all[vname]
            data[vname.lower()] = dict(data=data0, units='N/A')

    return data





def read_iwg_mts(fname, date_ref=None, tmhr_range=None):

    vnames = []
    with open(fname, 'r') as f:
        header = f.readline().strip()
        vnames = header.split(',')

    vnames = [vname.replace(' ', '_') for vname in vnames[1:]]
    vnames[0] = 'jday'

    Nvar = len(vnames)
    convert_func = lambda x: (datetime.datetime.strptime(x.decode('utf-8').split('.')[0], '%Y-%m-%dT%H:%M:%S')-datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0
    data_all = np.genfromtxt(fname, delimiter=',', names=vnames, converters={1:convert_func}, usecols=range(1, Nvar+1), skip_header=1)

    if date_ref is None:
        jday_int = np.int_(data_all['jday'])
        jday_unique, counts = np.unique(jday_int, return_counts=True)
        jdayRef = jday_unique[np.argmax(counts)]
    else:
        jdayRef = (date_ref - datetime.datetime(1, 1, 1)).total_seconds()/86400.0 + 1.0

    data = OrderedDict()
    data['tmhr'] = dict(data=(data_all['jday']-jdayRef)*24.0, units='Hour')

    if tmhr_range != None:
        logic = (data['tmhr']['data']>=tmhr_range[0]) & (data['tmhr']['data']<=tmhr_range[1])
        for i, vname in enumerate(vnames):
            data0 = data_all[vname][logic]
            data[vname.lower()] = dict(data=data0, units='N/A')
    else:
        for i, vname in enumerate(vnames):
            data0 = data_all[vname]
            data[vname.lower()] = dict(data=data0, units='N/A')

    return data





def read_cabin(fname, tmhr_range=None, Nskip=1, lower=True, time_units='sec', split='\t'):

    """
    Reader for Cabin file from Twin Otter aircraft (data shared by Dr. Anthony Bucholtz)

    Please note this reader might not be universally applicable and only work for the MAGPIE mission only.
    """

    # variable parsing
    #/----------------------------------------------------------------------------\#
    f = open(fname, 'r')

    vnames = []
    units  = []

    line = f.readline()
    vnames_ = line.strip().split(split)
    for vname_ in vnames_:
        words = vname_.replace(')', '').split('(')
        if len(words) > 1:
            vnames.append(words[0].strip())
            units.append(words[-1].strip())
        else:
            vnames.append(words[0].strip())
            units.append('N/A')

    Nvar = len(vnames)
    f.close()
    #\----------------------------------------------------------------------------/#

    data_all = np.genfromtxt(fname, skip_header=Nskip, delimiter=split, invalid_raise=False)

    data = OrderedDict()
    if time_units.lower() == 'sec':
        data['tmhr'] = dict(data=data_all[:, 1]/3600.0, units='Hour')
    elif time_units.lower() == 'hour':
        data['tmhr'] = dict(data=data_all[:, 1], units='Hour')

    if tmhr_range != None:
        logic = (data['tmhr']['data']>=tmhr_range[0]) & (data['tmhr']['data']<=tmhr_range[1])
        for i, vname in enumerate(vnames):
            data0 = data_all[:, i][logic]
            # if i > 0:
            #     data0[data0==fill_values[i-1]] = np.nan
            if lower:
                vname = vname.lower()
            data[vname] = dict(data=data0, units=units[i])
        data['tmhr']['data'] = data['tmhr']['data'][logic]
    else:
        for i, vname in enumerate(vnames):
            data0 = data_all[:, i]
            # if i > 0:
            #     data0[data0==fill_values[i-1]] = np.nan
            if lower:
                vname = vname.lower()
            data[vname] = dict(data=data0, units=units[i])


    return data





if __name__ == '__main__':

    pass
