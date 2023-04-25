import os
import glob
import h5py
import struct
import pysolar
from collections import OrderedDict
import xml.etree.ElementTree as ET
import numpy as np
import datetime
from scipy import stats

import ssfr
from ssfr.common import fdir_data
# from ssfr.corr import dark_corr
import ssfr.corr



__all__ = ['if_file_exists', 'cal_julian_day', 'cal_solar_angles', 'prh2za', 'muslope', 'dtime_to_jday', 'jday_to_dtime', \
           'load_h5', 'save_h5', 'cal_time_offset', 'cal_weighted_flux', 'read_ict', 'write_ict', \
           'read_iwg', 'read_lasp_ssfr', 'lasp_ssfr', 'read_nasa_ssfr', 'nasa_ssfr', \
           'get_lasp_ssfr_wavelength', 'get_nasa_ssfr_wavelength']



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

def cal_julian_day(date, tmhr):

    julian_day = np.zeros_like(tmhr, dtype=np.float64)

    for i in range(tmhr.size):
        tmhr0 = tmhr[i]
        julian_day[i] = (date - datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0 + tmhr0/24.0

    return julian_day

def cal_solar_angles(julian_day, longitude, latitude, altitude):

    dateRef = datetime.datetime(1, 1, 1)
    jdayRef = 1.0

    sza = np.zeros_like(julian_day)
    saa = np.zeros_like(julian_day)

    for i in range(julian_day.size):

        jday = julian_day[i]

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

    return sza, saa

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

    data = {}
    f = h5py.File(fname, 'r')
    for key in f.keys():
        data[key] = f[key][...]
    f.close()
    return data

def save_h5(fname, data):

    data = {}
    f = h5py.File(fname, 'w')
    for key in data.keys():
        f[key] = data[key]
    f.close()

    print('Message [save_h5]: Data has been successfully saved into \'%s\'.' % fname)

def cal_time_offset(time, data, time_ref, data_ref):

    pass

def cal_weighted_flux(wvl, data_wvl, data_flux, slit_func_file=None, wvl_join=950.0):

    if slit_func_file is None:
        if wvl <= wvl_join:
            slit_func_file = '%s/vis_0.1nm_s.dat' % fdir_data
        else:
            slit_func_file = '%s/nir_0.1nm_s.dat' % fdir_data

    data_slt = np.loadtxt(slit_func_file)
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





def var_info(line):

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
    vname0, unit0 = var_info(f.readline())
    vnames.append(vname0); units.append(unit0)

    Nvar = int(f.readline())
    f.readline()
    fill_values = np.array([float(word) for word in f.readline().strip().split(',')], dtype=np.float64)

    for i in range(Nvar):
        vname0, unit0 = var_info(f.readline())
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





def read_iwg(fname, date_ref=None, tmhr_range=None):

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





def get_lasp_ssfr_wavelength(chanNum=256):

    xChan = np.arange(chanNum)

    coef_zen_si = np.array([301.946,  3.31877,  0.00037585,  -1.76779e-6, 0])
    coef_zen_in = np.array([2202.33, -4.35275, -0.00269498,   3.84968e-6, -2.33845e-8])

    coef_nad_si = np.array([302.818,  3.31912,  0.000343831, -1.81135e-6, 0])
    coef_nad_in = np.array([2210.29,  -4.5998,  0.00102444,  -1.60349e-5, 1.29122e-8])

    wvl_zen_si = coef_zen_si[0] + coef_zen_si[1]*xChan + coef_zen_si[2]*xChan**2 + coef_zen_si[3]*xChan**3 + coef_zen_si[4]*xChan**4
    wvl_zen_in = coef_zen_in[0] + coef_zen_in[1]*xChan + coef_zen_in[2]*xChan**2 + coef_zen_in[3]*xChan**3 + coef_zen_in[4]*xChan**4

    wvl_nad_si = coef_nad_si[0] + coef_nad_si[1]*xChan + coef_nad_si[2]*xChan**2 + coef_nad_si[3]*xChan**3 + coef_nad_si[4]*xChan**4
    wvl_nad_in = coef_nad_in[0] + coef_nad_in[1]*xChan + coef_nad_in[2]*xChan**2 + coef_nad_in[3]*xChan**3 + coef_nad_in[4]*xChan**4

    wvl_dict = {
            'zenith_si': wvl_zen_si,
            'zenith_in': wvl_zen_in,
            'nadir_si' : wvl_nad_si,
            'nadir_in' : wvl_nad_in
            }

    return wvl_dict

def read_lasp_ssfr(fname, headLen=148, dataLen=2276, verbose=False):

    '''
    Reader code for Solar Spectral Flux Radiometer (SSFR) developed by Dr. Sebastian Schmidt's group
    at the University of Colorado Bouder.

    Input:
        fname: string, file path of the SSFR data
        headLen=: integer, number of bytes for the header
        dataLen=: integer, number of bytes for each data record
        verbose=: boolen, verbose tag

    Output:
        comment
        spectra
        shutter
        int_time
        temp
        jday_ARINC
        jday_cRIO
        qual_flag
        iterN

    How to use:
    fname = '/some/path/2015022000001.SKS'
    comment, spectra, shutter, int_time, temp, jday_ARINC, jday_cRIO, qual_flag, iterN = read_lasp_ssfr(fname, verbose=False)

    comment  (str)        [N/A]    : comment in header
    spectra  (numpy array)[N/A]    : counts of Silicon and InGaAs for both zenith and nadir
    shutter  (numpy array)[N/A]    : shutter status (1:closed(dark), 0:open(light))
    int_time (numpy array)[ms]     : integration time of Silicon and InGaAs for both zenith and nadir
    temp (numpy array)    [Celsius]: temperature variables
    jday_ARINC (numpy array)[day]  : julian days (w.r.t 0001-01-01) of aircraft nagivation system
    jday_cRIO(numpy array)[day]    : julian days (w.r.t 0001-01-01) of SSFR Inertial Navigation System (INS)
    qual_flag(numpy array)[N/A]    : quality flag(1:good, 0:bad)
    iterN (numpy array)   [N/A]    : number of data record

    by Hong Chen (me@hongchen.cz), Sebastian Schmidt (sebastian.schmidt@lasp.colorado.edu)
    '''

    if_file_exists(fname, exitTag=True)

    fileSize = os.path.getsize(fname)
    if fileSize > headLen:
        iterN   = (fileSize-headLen) // dataLen
        residual = (fileSize-headLen) %  dataLen
        if residual != 0:
            print('Warning [read_lasp_ssfr]: %s contains unreadable data, omit the last data record...' % fname)
    else:
        exit('Error [read_lasp_ssfr]: %s has invalid file size.' % fname)

    spectra    = np.zeros((iterN, 256, 4), dtype=np.float64) # spectra
    shutter    = np.zeros(iterN          , dtype=np.int32  ) # shutter status (1:closed, 0:open)
    int_time   = np.zeros((iterN, 4)     , dtype=np.float64) # integration time [ms]
    temp       = np.zeros((iterN, 11)    , dtype=np.float64) # temperature
    qual_flag  = np.ones(iterN           , dtype=np.int32)   # quality flag (1:good, 0:bad)
    jday_ARINC = np.zeros(iterN          , dtype=np.float64)
    jday_cRIO  = np.zeros(iterN          , dtype=np.float64)

    f           = open(fname, 'rb')

    # read head
    headRec   = f.read(headLen)
    head      = struct.unpack('<B144s3B', headRec)
    if head[0] != 144:
        f.seek(0)
    else:
        comment = head[1]

    if verbose:
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Comments in %s...' % fname.split('/')[-1])
        print(comment)
        print('--------------------------------------------------')

    # read data record
    for i in range(iterN):
        dataRec = f.read(dataLen)
        # ---------------------------------------------------------------------------------------------------------------
        # d9l: frac_second[d] , second[l] , minute[l] , hour[l] , day[l] , month[l] , year[l] , dow[l] , doy[l] , DST[l]
        # d9l: frac_second0[d], second0[l], minute0[l], hour0[l], day0[l], month0[l], year0[l], dow0[l], doy0[l], DST0[l]
        # l11d: null[l], temp(11)[11d]
        # --------------------------          below repeat for sz, sn, iz, in          ----------------------------------
        # l2Bl: int_time[l], shutter[B], EOS[B], null[l]
        # 257h: spectra(257)
        # ---------------------------------------------------------------------------------------------------------------
        data     = struct.unpack('<d9ld9ll11dl2Bl257hl2Bl257hl2Bl257hlBBl257h', dataRec)

        dataHead = data[:32]
        dataSpec = np.transpose(np.array(data[32:]).reshape((4, 261)))[:, [0, 2, 1, 3]]
        # [0, 2, 1, 3]: change order from 'sz, sn, iz, in' to 'sz, iz, sn, in'
        # transpose: change shape from (4, 261) to (261, 4)

        shutter_logic = (np.unique(dataSpec[1, :]).size != 1)
        eos_logic     = any(dataSpec[2, :] != 1)
        null_logic    = any(dataSpec[3, :] != 257)
        order_logic   = not np.array_equal(dataSpec[4, :], np.array([0, 2, 1, 3]))

        if any([shutter_logic, eos_logic, null_logic, order_logic]):
            qual_flag[i] = 0

        spectra[i, :, :]  = dataSpec[5:, :]
        shutter[i]        = dataSpec[1, 0]
        int_time[i, :]    = dataSpec[0, :]
        temp[i, :]        = dataHead[21:]

        dtime          = datetime.datetime(dataHead[6] , dataHead[5] , dataHead[4] , dataHead[3] , dataHead[2] , dataHead[1] , int(round(dataHead[0]*1000000.0)))
        dtime0         = datetime.datetime(dataHead[16], dataHead[15], dataHead[14], dataHead[13], dataHead[12], dataHead[11], int(round(dataHead[10]*1000000.0)))

        # calculate the proleptic Gregorian ordinal of the date
        jday_ARINC[i]  = (dtime  - datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0
        jday_cRIO[i]   = (dtime0 - datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0

    f.close()

    return comment, spectra, shutter, int_time, temp, jday_ARINC, jday_cRIO, qual_flag, iterN

class lasp_ssfr:

    def __init__(self, fnames, Ndata=600, whichTime='arinc', timeOffset=0.0, dark_corr_mode='interp'):

        '''
        Description:
        fnames    : list of SSFR files to read
        Ndata     : pre-defined number of data records, any number larger than the "number of data records per file" will work
        whichTime : "ARINC" or "cRIO"
        timeOffset: time offset in [seconds]
        '''

        if not isinstance(fnames, list):
            exit("Error [lasp_ssfr]: input variable 'fnames' should be a Python list.")

        if len(fnames) == 0:
            exit("Error [lasp_ssfr]: input variable 'fnames' is empty.")

        # +
        # read in all the data
        Nx         = Ndata * len(fnames)
        comment    = []
        spectra    = np.zeros((Nx, 256, 4), dtype=np.float64)
        shutter    = np.zeros(Nx          , dtype=np.int32  )
        int_time   = np.zeros((Nx, 4)     , dtype=np.float64)
        temp       = np.zeros((Nx, 11)    , dtype=np.float64)
        qual_flag  = np.zeros(Nx          , dtype=np.int32)
        jday_ARINC = np.zeros(Nx          , dtype=np.float64)
        jday_cRIO  = np.zeros(Nx          , dtype=np.float64)

        Nstart = 0
        for fname in fnames:
            comment0, spectra0, shutter0, int_time0, temp0, jday_ARINC0, jday_cRIO0, qual_flag0, iterN0 = read_lasp_ssfr(fname, verbose=False)

            Nend = iterN0 + Nstart

            comment.append(comment0)
            spectra[Nstart:Nend, ...]    = spectra0
            shutter[Nstart:Nend, ...]    = shutter0
            int_time[Nstart:Nend, ...]   = int_time0
            temp[Nstart:Nend, ...]       = temp0
            jday_ARINC[Nstart:Nend, ...] = jday_ARINC0
            jday_cRIO[Nstart:Nend, ...]  = jday_cRIO0
            qual_flag[Nstart:Nend, ...]  = qual_flag0

            Nstart = Nend

        self.comment    = comment
        self.spectra    = spectra[:Nend, ...]
        self.shutter    = shutter[:Nend, ...]
        self.int_time   = int_time[:Nend, ...]
        self.temp       = temp[:Nend, ...]
        self.jday_ARINC = jday_ARINC[:Nend, ...]
        self.jday_cRIO  = jday_cRIO[:Nend, ...]
        self.qual_flag  = qual_flag[:Nend, ...]

        if whichTime.lower() == 'arinc':
            self.jday = self.jday_ARINC.copy()
        elif whichTime.lower() == 'crio':
            self.jday = self.jday_cRIO.copy()
        self.tmhr = (self.jday - int(self.jday[0])) * 24.0

        self.jday_corr = self.jday.copy() + float(timeOffset)/86400.0
        self.tmhr_corr = self.tmhr.copy() + float(timeOffset)/3600.0
        # -

        self.port_info     = {0:'Zenith Silicon', 1:'Zenith InGaAs', 2:'Nadir Silicon', 3:'Nadir InGaAs'}
        self.int_time_info = {}
        # +
        # dark correction (light-dark)
        # variable name: self.spectra_dark_corr
        fillValue = np.nan
        self.fillValue = fillValue
        self.spectra_dark_corr      = self.spectra.copy()
        self.spectra_dark_corr[...] = fillValue
        for iSen in range(4):
            intTimes = np.unique(self.int_time[:, iSen])
            self.int_time_info[self.port_info[iSen]] = intTimes
            for intTime in intTimes:
                indices = np.where(self.int_time[:, iSen]==intTime)[0]
                self.spectra_dark_corr[indices, :, iSen] = DARK_CORRECTION(self.tmhr[indices], self.shutter[indices], self.spectra[indices, :, iSen], mode=dark_corr_mode, fillValue=fillValue)
        # -

    def count_to_radiation(self, cal, wvl_zen_join=900.0, wvl_nad_join=900.0, whichRadiation={'zenith':'radiance', 'nadir':'irradiance'}, wvlRange=[350, 2100]):

        """
        Convert digital count to radiation (radiance or irradiance)
        """

        self.whichRadiation = whichRadiation

        logic_zen_si = (cal.wvl_zen_si >= wvlRange[0])  & (cal.wvl_zen_si <= wvl_zen_join)
        logic_zen_in = (cal.wvl_zen_in >= wvl_zen_join) & (cal.wvl_zen_in <= wvlRange[1])
        n_zen_si = logic_zen_si.sum()
        n_zen_in = logic_zen_in.sum()
        n_zen    = n_zen_si + n_zen_in
        self.wvl_zen = np.append(cal.wvl_zen_si[logic_zen_si], cal.wvl_zen_in[logic_zen_in][::-1])

        logic_nad_si = (cal.wvl_nad_si >= wvlRange[0])  & (cal.wvl_nad_si <= wvl_nad_join)
        logic_nad_in = (cal.wvl_nad_in >= wvl_nad_join) & (cal.wvl_nad_in <= wvlRange[1])
        n_nad_si = logic_nad_si.sum()
        n_nad_in = logic_nad_in.sum()
        n_nad    = n_nad_si + n_nad_in
        self.wvl_nad = np.append(cal.wvl_nad_si[logic_nad_si], cal.wvl_nad_in[logic_nad_in][::-1])

        self.spectra_zen = np.zeros((self.tmhr.size, n_zen), dtype=np.float64)
        self.spectra_nad = np.zeros((self.tmhr.size, n_nad), dtype=np.float64)

        for i in range(self.tmhr.size):
            # for zenith
            intTime_si = self.int_time[i, 0]
            if intTime_si not in cal.primary_response_zen_si.keys():
                intTime_si_tag = -1
            else:
                intTime_si_tag = intTime_si

            intTime_in = self.int_time[i, 1]
            if intTime_in not in cal.primary_response_zen_in.keys():
                # intTime_in_tag = -1
                intTime_in_tag = 250
            else:
                intTime_in_tag = intTime_in

            if whichRadiation['zenith'] == 'radiance':
                self.spectra_zen[i, :n_zen_si] =  (self.spectra_dark_corr[i, logic_zen_si, 0]/intTime_si)/(np.pi * cal.primary_response_zen_si[intTime_si_tag][logic_zen_si])
                self.spectra_zen[i, n_zen_si:] = ((self.spectra_dark_corr[i, logic_zen_in, 1]/intTime_in)/(np.pi * cal.primary_response_zen_in[intTime_in_tag][logic_zen_in]))[::-1]
            elif whichRadiation['zenith'] == 'irradiance':
                self.spectra_zen[i, :n_zen_si] =  (self.spectra_dark_corr[i, logic_zen_si, 0]/intTime_si)/(cal.secondary_response_zen_si[intTime_si_tag][logic_zen_si])
                self.spectra_zen[i, n_zen_si:] = ((self.spectra_dark_corr[i, logic_zen_in, 1]/intTime_in)/(cal.secondary_response_zen_in[intTime_in_tag][logic_zen_in]))[::-1]

            # for nadir
            intTime_si = self.int_time[i, 2]
            if intTime_si not in cal.primary_response_nad_si.keys():
                intTime_si_tag = -1
            else:
                intTime_si_tag = intTime_si

            intTime_in = self.int_time[i, 3]
            if intTime_in not in cal.primary_response_nad_in.keys():
                # intTime_in_tag = -1
                intTime_in_tag = 250
            else:
                intTime_in_tag = intTime_in

            if whichRadiation['nadir'] == 'radiance':
                self.spectra_nad[i, :n_nad_si] =  (self.spectra_dark_corr[i, logic_nad_si, 2]/intTime_si)/(np.pi * cal.primary_response_nad_si[intTime_si_tag][logic_nad_si])
                self.spectra_nad[i, n_nad_si:] = ((self.spectra_dark_corr[i, logic_nad_in, 3]/intTime_in)/(np.pi * cal.primary_response_nad_in[intTime_in_tag][logic_nad_in]))[::-1]
            elif whichRadiation['nadir'] == 'irradiance':
                self.spectra_nad[i, :n_nad_si] =  (self.spectra_dark_corr[i, logic_nad_si, 2]/intTime_si)/(cal.secondary_response_nad_si[intTime_si_tag][logic_nad_si])
                self.spectra_nad[i, n_nad_si:] = ((self.spectra_dark_corr[i, logic_nad_in, 3]/intTime_in)/(cal.secondary_response_nad_in[intTime_in_tag][logic_nad_in]))[::-1]





def get_nasa_ssfr_wavelength(chanNum=256):

    xChan = np.arange(chanNum)

    # Silicon Zenith 033161
    coef_zen_si = np.array([303.087,  3.30588,  4.09568e-4,  -1.63269e-6, 0])
    # InGaAs Zenith 044832
    coef_zen_in = np.array([2213.37, -4.46844, -0.00111879,  -2.76593e-6, -1.57883e-8])

    # Silicon Nadir 045924
    coef_nad_si = np.array([302.255,  3.30977,  4.38733e-4, -1.90935e-6, 0])
    # InGaAs Nadir 044829
    coef_nad_in = np.array([2225.74, -4.37926, -0.00220588,  2.80201e-6, -2.2624e-8])

    wvl_zen_si = coef_zen_si[0] + coef_zen_si[1]*xChan + coef_zen_si[2]*xChan**2 + coef_zen_si[3]*xChan**3 + coef_zen_si[4]*xChan**4
    wvl_zen_in = coef_zen_in[0] + coef_zen_in[1]*xChan + coef_zen_in[2]*xChan**2 + coef_zen_in[3]*xChan**3 + coef_zen_in[4]*xChan**4

    wvl_nad_si = coef_nad_si[0] + coef_nad_si[1]*xChan + coef_nad_si[2]*xChan**2 + coef_nad_si[3]*xChan**3 + coef_nad_si[4]*xChan**4
    wvl_nad_in = coef_nad_in[0] + coef_nad_in[1]*xChan + coef_nad_in[2]*xChan**2 + coef_nad_in[3]*xChan**3 + coef_nad_in[4]*xChan**4

    wvl_dict = {
            'zenith_si': wvl_zen_si,
            'zenith_in': wvl_zen_in,
            'nadir_si' : wvl_nad_si,
            'nadir_in' : wvl_nad_in
            }

    return wvl_dict

def read_nasa_ssfr(fname, headLen=0, dataLen=2124, verbose=False):

    '''
    Reader code for Solar Spectral Flux Radiometer (SSFR) developed by Warren Gore's group
    at the NASA AMES.

    How to use:
    fname = '/some/path/2015022000001.OSA2'
    spectra, shutter, int_time, temp, jday, qual_flag, iterN = read_nasa_ssfr(fname, verbose=False)

    spectra  (numpy array)[N/A]    : counts of Silicon and InGaAs for both zenith and nadir
    shutter  (numpy array)[N/A]    : shutter status (1:closed(dark), 0:open(light))
    int_time (numpy array)[ms]     : integration time of Silicon and InGaAs for both zenith and nadir
    temp (numpy array)    [Celsius]: temperature variables
    jday(numpy array)[day]         : julian days (w.r.t 0001-01-01) of SSFR
    qual_flag(numpy array)[N/A]    : quality flag(1:good, 0:bad)
    iterN (numpy array)   [N/A]    : number of data record

    by Hong Chen (me@hongchen.cz), Sebastian Schmidt (sebastian.schmidt@lasp.colorado.edu)
    '''

    if_file_exists(fname, exitTag=True)

    filename = os.path.basename(fname)
    filetype = filename.split('.')[-1].lower()
    if filetype != 'osa2':
        exit('Error   [read_nasa_ssfr]: Do not support \'%s\'.' % filetype)

    fileSize = os.path.getsize(fname)
    if fileSize > headLen:
        iterN   = (fileSize-headLen) // dataLen
        residual = (fileSize-headLen) %  dataLen
        if residual != 0:
            print('Warning [read_nasa_ssfr]: %s contains unreadable data, omit the last data record...' % fname)
    else:
        exit('Error [read_nasa_ssfr]: %s has invalid file size.' % fname)

    spectra    = np.zeros((iterN, 256, 4), dtype=np.float64) # spectra
    shutter    = np.zeros(iterN          , dtype=np.int32  ) # shutter status (1:closed, 0:open)
    int_time   = np.zeros((iterN, 4)     , dtype=np.float64) # integration time [ms]
    temp       = np.zeros((iterN, 8)     , dtype=np.float64) # temperature
    qual_flag  = np.ones(iterN           , dtype=np.int32)   # quality flag (1:good, 0:bad)
    jday       = np.zeros(iterN          , dtype=np.float64)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 2l1   : btime
    # 12B   : bcdtimstp(12)
    # 6l    : intime1, intime2, intime3, intime4, accum, shsw
    # 8L    : zsit, nsit, zirt, nirt, zirx, nirx, xt, it
    # 1024h : zspecsi(256), zspecir(256), nspecsi(256), nspecir(256)
    # ---------------------------------------------------------------------------------------------------------------
    binFmt  = '<2l12B6l8L1024h'

    spectra    = np.zeros((iterN, 256, 4), dtype=np.float64) # spectra
    shutter    = np.zeros(iterN          , dtype=np.int32  ) # shutter status (1:closed, 0:open)
    int_time   = np.zeros((iterN, 4)     , dtype=np.float64) # integration time [ms]
    temp       = np.zeros((iterN, 8)     , dtype=np.float64) # temperature
    qual_flag  = np.ones(iterN           , dtype=np.int32)   # quality flag (1:good, 0:bad)
    jday       = np.zeros(iterN          , dtype=np.float64)

    f           = open(fname, 'rb')

    # +++++++++++++++++++++++++++ read head ++++++++++++++++++++++++++++++
    if headLen > 0:
        headRec   = f.read(headLen)
    # --------------------------------------------------------------------

    # read data record
    for i in range(iterN):
        dataRec = f.read(dataLen)

        data     = struct.unpack(binFmt, dataRec)

        dataSpec = np.transpose(np.array(data[28:]).reshape((4, 256)))
        # 0, 1, 2, 3 represent 'sz, iz, sn, in'
        # transpose: change shape from (4, 256) to (256, 4)

        spectra[i, :, :]  = dataSpec
        shutter[i]        = data[19]
        int_time[i, :]    = np.array(data[14:18])
        temp[i, :]        = np.array(data[20:28])

        # calculate the proleptic Gregorian ordinal of the date
        dtime      = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=data[0])
        jday[i]    = (dtime  - datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0

    return spectra, shutter, int_time, temp, jday, qual_flag, iterN

class nasa_ssfr:

    """
    Read NASA AMES SSFR data files (.OSA2) into nasa_ssfr object

    input:
        fnames: Python list, file paths of the data
        tmhr_range=: two elements Python list, e.g., [0, 24], starting and ending time in hours to slice the data
        Ndata=: maximum number of data records per data file
        time_offset=: float, time offset in seconds

    Output:
        nasa_ssfr object that contains
                .jday     : julian day
                .tmhr     : time in hour
                .spectra  : ssfr spectra
                .shutter  : shutter status
                .int_time : integration time
                .temp     : temperatures
                .qual_flag: quality flags
    """

    def __init__(self, fnames, fname_raw=None, date_ref=None, tmhr_range=None,  Ndata=600, time_add_offset=0.0, verbose=False):

        if fname_raw is not None:
            data_v0 = load_h5(fname_raw)
            self.jday = data_v0['jday']
            self.tmhr = data_v0['tmhr']
            self.shutter = data_v0['shutter']
            self.nad_cnt = data_v0['nad_cnt']
            self.nad_wvl = data_v0['nad_wvl']
            self.nad_int_time = data_v0['nad_int_time']
            self.zen_cnt = data_v0['zen_cnt']
            self.zen_wvl = data_v0['zen_wvl']
            self.zen_int_time = data_v0['zen_int_time']

        else:

            if len(fnames) == 0:
                sys.exit('Error   [nasa_ssfr]: No files are found in \'fnames\'.')

            # initialize
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            Nx         = Ndata * len(fnames)
            spectra    = np.zeros((Nx, 256, 4), dtype=np.float64) # spectra
            shutter    = np.zeros(Nx          , dtype=np.int32  ) # shutter status (1:closed, 0:open)
            int_time   = np.zeros((Nx, 4)     , dtype=np.float64) # integration time [ms]
            temp       = np.zeros((Nx, 8)     , dtype=np.float64) # temperature
            qual_flag  = np.zeros(Nx          , dtype=np.int32)
            jday       = np.zeros(Nx          , dtype=np.float64)
            # ------------------------------------------------------------------------------------------

            # read in data file by file
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            Nstart = 0
            for fname in fnames:
                spectra0, shutter0, int_time0, temp0, jday0, qual_flag0, iterN0 = read_nasa_ssfr(fname)
                Nend = iterN0 + Nstart
                spectra[Nstart:Nend, ...]    = spectra0
                shutter[Nstart:Nend, ...]    = shutter0
                int_time[Nstart:Nend, ...]   = int_time0
                temp[Nstart:Nend, ...]       = temp0
                jday[Nstart:Nend, ...]       = jday0
                qual_flag[Nstart:Nend, ...]  = qual_flag0
                Nstart = Nend
            # ------------------------------------------------------------------------------------------

            # remove redundant data
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            spectra    = spectra[:Nend, ...]
            shutter    = shutter[:Nend, ...]
            int_time   = int_time[:Nend, ...]
            temp       = temp[:Nend, ...]
            jday       = jday[:Nend, ...]
            qual_flag  = qual_flag[:Nend, ...]
            # ------------------------------------------------------------------------------------------

            # correct time by adding time_add_offset
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            jday    = jday + time_add_offset/86400.0
            # ------------------------------------------------------------------------------------------

            # find the most frequent julian day
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if date_ref is None:
                jday_int = np.int_(jday)
                jday_unique, counts = np.unique(jday_int, return_counts=True)
                jdayRef = jday_unique[np.argmax(counts)]
            else:
                jdayRef = (date_ref-datetime.datetime(1, 1, 1)).total_seconds()/86400.0 + 1.0
            # ------------------------------------------------------------------------------------------

            # calculate tmhr: time in hour
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            tmhr    = (jday-jdayRef)*24.0
            # ------------------------------------------------------------------------------------------

            # slice data using input time_range
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if tmhr_range is not None:
                logic = (tmhr>=tmhr_range[0]) & (tmhr<=tmhr_range[1])
            else:
                logic = (jday>=0.0)
            # ------------------------------------------------------------------------------------------

            # add data to the attributes
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            self.jday       = jday[logic]
            self.tmhr       = tmhr[logic]
            self.spectra    = spectra[logic, ...]
            self.shutter    = shutter[logic, ...]
            self.int_time   = int_time[logic, ...]
            self.temp       = temp[logic, ...]
            self.qual_flag  = qual_flag[logic, ...]
            # ------------------------------------------------------------------------------------------

    def pre_process(self, wvl_join=950.0, wvl_start=350.0, wvl_end=2200.0, intTime={'si':60, 'in':300}):

        wvls = get_nasa_ssfr_wavelength()

        # zenith
        logic_si = (wvls['zenith_si'] >= wvl_start) & (wvls['zenith_si'] <= wvl_join)
        logic_in = (wvls['zenith_in'] >  wvl_join)  & (wvls['zenith_in'] <= wvl_end)

        wvl_data = np.concatenate((wvls['zenith_si'][logic_si], wvls['zenith_in'][logic_in]))
        cnt_data = np.hstack((self.spectra[:, logic_si, 0], self.spectra[:, logic_in, 1]))

        indices_sort  = np.argsort(wvl_data)
        self.zen_wvl  = wvl_data[indices_sort]
        self.zen_int_time  = np.concatenate((np.repeat(intTime['si'], logic_si.sum()), np.repeat(intTime['in'], logic_in.sum())))
        tmp, self.zen_cnt  = ssfr.corr.dark_corr(self.tmhr, self.shutter, cnt_data[:, indices_sort])

        # nadir
        logic_si = (wvls['nadir_si'] >= wvl_start) & (wvls['nadir_si'] <= wvl_join)
        logic_in = (wvls['nadir_in'] >  wvl_join)  & (wvls['nadir_in'] <= wvl_end)

        wvl_data = np.concatenate((wvls['nadir_si'][logic_si], wvls['nadir_in'][logic_in]))
        cnt_data = np.hstack((self.spectra[:, logic_si, 2], self.spectra[:, logic_in, 3]))

        indices_sort  = np.argsort(wvl_data)
        self.nad_wvl  = wvl_data[indices_sort]
        self.nad_int_time  = np.concatenate((np.repeat(intTime['si'], logic_si.sum()), np.repeat(intTime['in'], logic_in.sum())))
        tmp, self.nad_cnt  = ssfr.corr.dark_corr(self.tmhr, self.shutter, cnt_data[:, indices_sort])

    def cal_flux(self, fnames, wvl0=550.0):

        resp_zen = ssfr.cal.load_rad_resp_h5(fnames['zenith'])
        self.zen_flux = np.zeros_like(self.zen_cnt)

        resp_nad = ssfr.cal.load_rad_resp_h5(fnames['nadir'])
        self.nad_flux = np.zeros_like(self.nad_cnt)

        for i in range(self.tmhr.size):

            self.zen_flux[i, :] = self.zen_cnt[i, :] / self.zen_int_time / resp_zen['sec_resp']
            self.nad_flux[i, :] = self.nad_cnt[i, :] / self.nad_int_time / resp_nad['sec_resp']

        index_zen = np.argmin(np.abs(self.zen_wvl-wvl0))
        index_nad = np.argmin(np.abs(self.nad_wvl-wvl0))
        logic_bad = (self.shutter==1) | (self.zen_flux[:, index_zen]<=0.0) | (self.nad_flux[:, index_nad]<=0.0)
        self.zen_flux[logic_bad, :] = np.nan
        self.nad_flux[logic_bad, :] = np.nan






if __name__ == '__main__':

    pass
