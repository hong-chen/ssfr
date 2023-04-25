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



__all__ = [
           'read_ssfr_raw', \
           'read_ssfr', \
           'get_ssfr_wavelength'
           ]



def get_ssfr_wavelength(chanNum=256):

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

def read_ssfr_raw(fname, headLen=0, dataLen=2124, verbose=False):

    '''
    Reader code for Solar Spectral Flux Radiometer (SSFR) developed by Warren Gore's group
    at the NASA Ames.

    How to use:
    fname = '/some/path/2015022000001.OSA2'
    spectra, shutter, int_time, temp, jday, qual_flag, iterN = read_ssfr_raw(fname, verbose=False)

    spectra  (numpy array)[N/A]    : counts of Silicon and InGaAs for both zenith and nadir
    shutter  (numpy array)[N/A]    : shutter status (1:closed(dark), 0:open(light))
    int_time (numpy array)[ms]     : integration time of Silicon and InGaAs for both zenith and nadir
    temp (numpy array)    [Celsius]: temperature variables
    jday(numpy array)[day]         : julian days (w.r.t 0001-01-01) of SSFR
    qual_flag(numpy array)[N/A]    : quality flag(1:good, 0:bad)
    iterN (numpy array)   [N/A]    : number of data record

    by Hong Chen (hong.chen@lasp.colorado.edu), Sebastian Schmidt (sebastian.schmidt@lasp.colorado.edu)
    '''

    if_file_exists(fname, exitTag=True)

    filename = os.path.basename(fname)
    filetype = filename.split('.')[-1].lower()
    if filetype != 'osa2':
        exit('Error   [read_ssfr_raw]: Do not support \'%s\'.' % filetype)

    fileSize = os.path.getsize(fname)
    if fileSize > headLen:
        iterN   = (fileSize-headLen) // dataLen
        residual = (fileSize-headLen) %  dataLen
        if residual != 0:
            print('Warning [read_ssfr_raw]: %s contains unreadable data, omit the last data record...' % fname)
    else:
        exit('Error [read_ssfr_raw]: %s has invalid file size.' % fname)

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

    data_ = {
          'spectra' : spectra,
          'shutter' : shutter,
         'int_time' : int_time,
             'temp' : temp,
             'jday' : jday,
        'qual_flag' : qual_flag,
            'iterN' : iterN,
            }
    return data_

class read_ssfr:

    """
    Read NASA Ames SSFR data files (.OSA2) into read_ssfr object

    input:
        fnames: Python list, file paths of the data
        tmhr_range=: two elements Python list, e.g., [0, 24], starting and ending time in hours to slice the data
        Ndata=: maximum number of data records per data file
        time_offset=: float, time offset in seconds

    Output:
        read_ssfr object that contains
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
            data_v0 = ssfr.util.load_h5(fname_raw)
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
                sys.exit('Error   [read_ssfr]: No files are found in \'fnames\'.')

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

                data0 = read_ssfr_raw(fname)
                Nend = data0['iterN'] + Nstart
                spectra[Nstart:Nend, ...]    = data0['spectra']
                shutter[Nstart:Nend, ...]    = data0['shutter']
                int_time[Nstart:Nend, ...]   = data0['int_time']
                temp[Nstart:Nend, ...]       = data0['temp']
                jday[Nstart:Nend, ...]       = data0['jday']
                qual_flag[Nstart:Nend, ...]  = data0['qual_flag']
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

        wvls = get_ssfr_wavelength()

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
