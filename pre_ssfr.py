import os
import glob
import h5py
import struct
import numpy as np
import datetime
from scipy.io import readsav
from scipy import stats

def READ_CU_SSFR_V1(fname, headLen=148, dataLen=2260, verbose=False):

    '''
    Description:
    Reader code for Solar Spectral Flux Radiometer (SSFR) developed by Dr. Sebastian Schmidt's group
    at the University of Colorado Bouder.

    How to use:
    fname = '/some/path/2015022000001.SKS'
    comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR_V1(fname, filetype='sks1', verbose=False)

    comment  (str)        [N/A]    : comment in header
    spectra  (numpy array)[N/A]    : counts of Silicon and InGaAs for both zenith and nadir
    shutter  (numpy array)[N/A]    : shutter status (1:closed(dark), 0:open(light))
    int_time (numpy array)[ms]     : integration time of Silicon and InGaAs for both zenith and nadir
    temp (numpy array)    [Celsius]: temperature variables
    jday_NSF (numpy array)[day]    : julian days (w.r.t 0001-01-01) of aircraft nagivation system
    jday_cRIO(numpy array)[day]    : julian days (w.r.t 0001-01-01) of SSFR Inertial Navigation System (INS)
    qual_flag(numpy array)[N/A]    : quality flag(1:good, 0:bad)
    iterN (numpy array)   [N/A]    : number of data record

    Written by: Hong Chen (me@hongchen.cz)
    '''

    fileSize = os.path.getsize(fname)
    iterN    = (fileSize-headLen) // dataLen
    residual = (fileSize-headLen) %  dataLen
    if not (fileSize>headLen and residual==0 and iterN>0):

        print('Warning [READ_CU_SSFR_V1]: %s has invalid data size, return empty arrays.' % fname)
        iterN      = 0
        comment    = ''
        spectra    = np.zeros((iterN, 256, 4), dtype=np.float64) # spectra
        shutter    = np.zeros(iterN          , dtype=np.int32  ) # shutter status (1:closed, 0:open)
        int_time   = np.zeros((iterN, 4)     , dtype=np.float64) # integration time [ms]
        temp       = np.zeros((iterN, 9)     , dtype=np.float64) # temperature
        qual_flag  = np.ones(iterN           , dtype=np.int32)   # quality flag (1:good, 0:bad)
        jday_NSF   = np.zeros(iterN          , dtype=np.float64)
        jday_cRIO  = np.zeros(iterN          , dtype=np.float64)

        return comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # d9l: frac_second[d] , second[l] , minute[l] , hour[l] , day[l] , month[l] , year[l] , dow[l] , doy[l] , DST[l]
    # d9l: frac_second0[d], second0[l], minute0[l], hour0[l], day0[l], month0[l], year0[l], dow0[l], doy0[l], DST0[l]
    # l9d: null[l], temp(9)[9d]
    # --------------------------          below repeat for sz, sn, iz, in          ----------------------------------
    # l2Bl: int_time[l], shutter[B], EOS[B], null[l]
    # 257h: spectra(257)
    # ---------------------------------------------------------------------------------------------------------------
    binFmt  = '<d9ld9ll9dl2Bl257hl2Bl257hl2Bl257hlBBl257h'

    if verbose:
        print('+' % fname)
        print('Message [READ_CU_SSFR_V1]: Reading %s...' % fname)

    f       = open(fname, 'rb')
    headRec = f.read(headLen)
    head    = struct.unpack('<B144s3B', headRec)
    comment = head[1]

    spectra    = np.zeros((iterN, 256, 4), dtype=np.float64)
    shutter    = np.zeros(iterN          , dtype=np.int32  )
    int_time   = np.zeros((iterN, 4)     , dtype=np.float64)
    temp       = np.zeros((iterN, 9)     , dtype=np.float64)
    qual_flag  = np.ones(iterN           , dtype=np.int32)
    jday_NSF   = np.zeros(iterN          , dtype=np.float64)
    jday_cRIO  = np.zeros(iterN          , dtype=np.float64)

    # read data record
    for i in range(iterN):
        dataRec  = f.read(dataLen)
        data     = struct.unpack(binFmt, dataRec)

        dataHead = data[:30]
        dataSpec = np.transpose(np.array(data[30:]).reshape((4, 261)))[:, [0, 2, 1, 3]]
        # [0, 2, 1, 3]: change order from 'sz, sn, iz, in' to 'sz, iz, sn, in'
        # transpose: change shape from (4, 261) to (261, 4)

        shutter_logic = (np.unique(dataSpec[1, :]).size != 1)
        eos_logic     = any(dataSpec[2, :] != 1)
        null_logic    = any(dataSpec[3, :] != 257)
        order_logic   = not np.array_equal(dataSpec[4, :], np.array([0, 2, 1, 3]))

        logic         = any([shutter_logic, eos_logic, null_logic, order_logic])
        if logic:
            qual_flag[i] = 0
            spectra[i, :, :]  = -99999
            shutter[i]        = -99999
            int_time[i, :]    = -99999
            temp[i, :]        = -99999
        else:
            spectra[i, :, :]  = dataSpec[5:, :]
            shutter[i]        = dataSpec[1, 0]
            int_time[i, :]    = dataSpec[0, :]
            temp[i, :]        = dataHead[21:]

        dtime          = datetime.datetime(dataHead[6] , dataHead[5] , dataHead[4] , dataHead[3] , dataHead[2] , dataHead[1] , int(round(dataHead[0]*1000000.0)))
        dtime0         = datetime.datetime(dataHead[16], dataHead[15], dataHead[14], dataHead[13], dataHead[12], dataHead[11], int(round(dataHead[10]*1000000.0)))

        # calculate the proleptic Gregorian ordinal of the date
        jday_NSF[i]    = (dtime  - datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0
        jday_cRIO[i]   = (dtime0 - datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0

    if verbose:
        print('-' % fname)

    return comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN

def READ_CU_SSFR_V2_OLD(fname, headLen=148, dataLen=2276, verbose=False):

    '''
    Description:
    Reader code for Solar Spectral Flux Radiometer (SSFR) developed by Dr. Sebastian Schmidt's group
    at the University of Colorado Bouder.

    How to use:
    fname = '/some/path/2015022000001.SKS'
    comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR_V2(fname, verbose=False)

    comment  (str)        [N/A]    : comment in header
    spectra  (numpy array)[N/A]    : counts of Silicon and InGaAs for both zenith and nadir
    shutter  (numpy array)[N/A]    : shutter status (1:closed(dark), 0:open(light))
    int_time (numpy array)[ms]     : integration time of Silicon and InGaAs for both zenith and nadir
    temp (numpy array)    [Celsius]: temperature variables
    jday_NSF (numpy array)[day]    : julian days (w.r.t 0001-01-01) of aircraft nagivation system
    jday_cRIO(numpy array)[day]    : julian days (w.r.t 0001-01-01) of SSFR Inertial Navigation System (INS)
    qual_flag(numpy array)[N/A]    : quality flag(1:good, 0:bad)
    iterN (numpy array)   [N/A]    : number of data record

    Written by: Hong Chen (me@hongchen.cz)
    '''

    fileSize = os.path.getsize(fname)
    iterN    = (fileSize-headLen) // dataLen
    residual = (fileSize-headLen) %  dataLen
    if not (fileSize>headLen and residual==0 and iterN>0):

        print('Warning [READ_CU_SSFR_V2]: %s has invalid data size, return empty arrays.' % fname)
        iterN      = 0
        comment    = ''
        spectra    = np.zeros((iterN, 256, 4), dtype=np.float64) # spectra
        shutter    = np.zeros(iterN          , dtype=np.int32  ) # shutter status (1:closed, 0:open)
        int_time   = np.zeros((iterN, 4)     , dtype=np.float64) # integration time [ms]
        temp       = np.zeros((iterN, 11)    , dtype=np.float64) # temperature
        qual_flag  = np.ones(iterN           , dtype=np.int32)   # quality flag (1:good, 0:bad)
        jday_NSF   = np.zeros(iterN          , dtype=np.float64)
        jday_cRIO  = np.zeros(iterN          , dtype=np.float64)

        return comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # d9l: frac_second[d] , second[l] , minute[l] , hour[l] , day[l] , month[l] , year[l] , dow[l] , doy[l] , DST[l]
    # d9l: frac_second0[d], second0[l], minute0[l], hour0[l], day0[l], month0[l], year0[l], dow0[l], doy0[l], DST0[l]
    # l11d: null[l], temp(11)[11d]
    # --------------------------          below repeat for sz, sn, iz, in          ----------------------------------
    # l2Bl: int_time[l], shutter[B], EOS[B], null[l]
    # 257h: spectra(257)
    # ---------------------------------------------------------------------------------------------------------------
    binFmt  = '<d9ld9ll11dl2Bl257hl2Bl257hl2Bl257hlBBl257h'

    if verbose:
        print('+' % fname)
        print('Message [READ_CU_SSFR_V2]: Reading %s...' % fname)

    f       = open(fname, 'rb')
    headRec = f.read(headLen)
    head    = struct.unpack('<B144s3B', headRec)
    comment = head[1]

    spectra    = np.zeros((iterN, 256, 4), dtype=np.float64)
    shutter    = np.zeros(iterN          , dtype=np.int32  )
    int_time   = np.zeros((iterN, 4)     , dtype=np.float64)
    temp       = np.zeros((iterN, 11)    , dtype=np.float64)
    qual_flag  = np.ones(iterN           , dtype=np.int32)
    jday_NSF   = np.zeros(iterN          , dtype=np.float64)
    jday_cRIO  = np.zeros(iterN          , dtype=np.float64)

    # read data record
    for i in range(iterN):
        dataRec  = f.read(dataLen)
        data     = struct.unpack(binFmt, dataRec)

        dataHead = data[:30]
        dataSpec = np.transpose(np.array(data[30:]).reshape((4, 261)))[:, [0, 2, 1, 3]]
        # [0, 2, 1, 3]: change order from 'sz, sn, iz, in' to 'sz, iz, sn, in'
        # transpose: change shape from (4, 261) to (261, 4)

        shutter_logic = (np.unique(dataSpec[1, :]).size != 1)
        eos_logic     = any(dataSpec[2, :] != 1)
        null_logic    = any(dataSpec[3, :] != 257)
        order_logic   = not np.array_equal(dataSpec[4, :], np.array([0, 2, 1, 3]))

        logic         = any([shutter_logic, eos_logic, null_logic, order_logic])
        if logic:
            qual_flag[i] = 0
            spectra[i, :, :]  = -99999
            shutter[i]        = -99999
            int_time[i, :]    = -99999
            temp[i, :]        = -99999
        else:
            spectra[i, :, :]  = dataSpec[5:, :]
            shutter[i]        = dataSpec[1, 0]
            int_time[i, :]    = dataSpec[0, :]
            temp[i, :]        = dataHead[21:]

        dtime          = datetime.datetime(dataHead[6] , dataHead[5] , dataHead[4] , dataHead[3] , dataHead[2] , dataHead[1] , int(round(dataHead[0]*1000000.0)))
        dtime0         = datetime.datetime(dataHead[16], dataHead[15], dataHead[14], dataHead[13], dataHead[12], dataHead[11], int(round(dataHead[10]*1000000.0)))

        # calculate the proleptic Gregorian ordinal of the date
        jday_NSF[i]    = (dtime  - datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0
        jday_cRIO[i]   = (dtime0 - datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0

    if verbose:
        print('-' % fname)

    return comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN

def READ_CU_SSFR_V2(fname, headLen=148, dataLen=2276, verbose=False):

    fileSize = os.path.getsize(fname)
    if fileSize > headLen:
        iterN   = (fileSize-headLen) // dataLen
        residual = (fileSize-headLen) %  dataLen
        if residual != 0:
            print('Warning [READ_SKS_V2]: %s has invalid data size.' % fname)
    else:
        exit('Error [READ_SKS_V2]: %s has invalid file size.' % fname)

    spectra    = np.zeros((iterN, 256, 4), dtype=np.float64) # spectra
    shutter    = np.zeros(iterN          , dtype=np.int32  ) # shutter status (1:closed, 0:open)
    int_time   = np.zeros((iterN, 4)     , dtype=np.float64) # integration time [ms]
    temp       = np.zeros((iterN, 11)    , dtype=np.float64) # temperature
    qual_flag  = np.ones(iterN           , dtype=np.int32)   # quality flag (1:good, 0:bad)
    jday_NSF   = np.zeros(iterN          , dtype=np.float64)
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
        # l9d: null[l], temp(9)[9d]
        # --------------------------          below repeat for sz, sn, iz, in          ----------------------------------
        # l2Bl: int_time[l], shutter[B], EOS[B], null[l]
        # 257h: spectra(257)
        # ---------------------------------------------------------------------------------------------------------------
        #data     = struct.unpack('<d9ld9ll9dl2Bl257hl2Bl257hl2Bl257hlBBl257h', dataRec)
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

        if True:
            spectra[i, :, :]  = dataSpec[5:, :]
            shutter[i]        = dataSpec[1, 0]
            int_time[i, :]    = dataSpec[0, :]
            temp[i, :]        = dataHead[21:]

            dtime          = datetime.datetime(dataHead[6] , dataHead[5] , dataHead[4] , dataHead[3] , dataHead[2] , dataHead[1] , int(round(dataHead[0]*1000000.0)))
            dtime0         = datetime.datetime(dataHead[16], dataHead[15], dataHead[14], dataHead[13], dataHead[12], dataHead[11], int(round(dataHead[10]*1000000.0)))

            # calculate the proleptic Gregorian ordinal of the date
            jday_NSF[i]    = (dtime  - datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0
            jday_cRIO[i]   = (dtime0 - datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0

    return comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN




def READ_NASA_SSFR_V1(fname, filetype='osa2', headLen=0, dataLen=2124, verbose=False):

    fileSize = os.path.getsize(fname)
    iterN    = (fileSize-headLen) // dataLen
    residual = (fileSize-headLen) %  dataLen

    filetype = filetype.lower()
    if filetype != 'osa2':
        exit('Error [READ_NASA_SSFR_V1]: do not support \'%s\'.' % filetype)

    if fileSize < headLen:
        print('Warning [READ_NASA_SSFR_V1]: %s has invalid data size, return empty arrays.' % fname)
        iterN      = 0
        spectra    = np.zeros((iterN, 256, 4), dtype=np.float64) # spectra
        shutter    = np.zeros(iterN          , dtype=np.int32  ) # shutter status (1:closed, 0:open)
        int_time   = np.zeros((iterN, 4)     , dtype=np.float64) # integration time [ms]
        temp       = np.zeros((iterN, 8)     , dtype=np.float64) # temperature
        qual_flag  = np.ones(iterN           , dtype=np.int32)   # quality flag (1:good, 0:bad)
        jday       = np.zeros(iterN          , dtype=np.float64)
        return comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN
    elif residual != 0:
        print('Warning [READ_NASA_SSFR_V1]: %s has invalid data size, read what\'s in...' % fname)

    headLen = 0
    dataLen = 2124
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


class READ_NASA_SSFR:

    def __init__(self, fnames, tmhr_range=None,  Ndata=600, secOffset=0.0, verbose=False):

        # read in SSFR files
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        Nx         = Ndata * len(fnames)
        spectra    = np.zeros((Nx, 256, 4), dtype=np.float64) # spectra
        shutter    = np.zeros(Nx          , dtype=np.int32  ) # shutter status (1:closed, 0:open)
        int_time   = np.zeros((Nx, 4)     , dtype=np.float64) # integration time [ms]
        temp       = np.zeros((Nx, 8)     , dtype=np.float64) # temperature
        qual_flag  = np.zeros(Nx          , dtype=np.int32)
        jday       = np.zeros(Nx          , dtype=np.float64)

        Nstart = 0
        if verbose:
            print('+++++++++++++++++++ Reading SSFR files ++++++++++++++++++++')
        for fname in fnames:
            if verbose:
                print('Reading %s...' % fname)

            spectra0, shutter0, int_time0, temp0, jday0, qual_flag0, iterN0 = READ_NASA_SSFR_V1(fname)

            Nend = iterN0 + Nstart

            spectra[Nstart:Nend, ...]    = spectra0
            shutter[Nstart:Nend, ...]    = shutter0
            int_time[Nstart:Nend, ...]   = int_time0
            temp[Nstart:Nend, ...]       = temp0
            jday[Nstart:Nend, ...]       = jday0
            qual_flag[Nstart:Nend, ...]  = qual_flag0

            Nstart = Nend

        spectra    = spectra[:Nend, ...]
        shutter    = shutter[:Nend, ...]
        int_time   = int_time[:Nend, ...]
        temp       = temp[:Nend, ...]
        jday       = jday[:Nend, ...]
        qual_flag  = qual_flag[:Nend, ...]

        jdayRef = int(jday[0])
        self.tmhr_range = tmhr_range
        logic = (jday-jdayRef>=0.0)&(jday-jdayRef<=2.0)

        spectra    = spectra[logic, ...]
        shutter    = shutter[logic, ...]
        int_time   = int_time[logic, ...]
        temp       = temp[logic, ...]
        jday       = jday[logic, ...]
        qual_flag  = qual_flag[logic, ...]
        # ------------------------------------------------------------------------------------------

        self.jday_corr = jday - secOffset/86400.0
        self.tmhr_corr = (jday-jdayRef)*24.0  - secOffset/3600.0
        self.spectra   = spectra
        self.shutter   = shutter
        self.int_time  = int_time
        self.temp      = temp
        self.qual_flag = qual_flag

        # dark correction
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.shutter_ori = self.shutter.copy()
        if verbose:
            print('Dark correction...')
        self.DARK_CORR()
        # ------------------------------------------------------------------------------------------

    def DARK_CORR(self, mode=-1, darkExtend=2, lightExtend=2, countOffset=0, lightThr=10, darkThr=5):

        if self.shutter[0] == 0:
            darkL = np.array([], dtype=np.int32)
            darkR = np.array([0], dtype=np.int32)
        else:
            darkR = np.array([], dtype=np.int32)
            darkL = np.array([0], dtype=np.int32)

        darkL0 = np.squeeze(np.argwhere((self.shutter[1:]-self.shutter[:-1]) ==  1)) + 1
        darkL  = np.hstack((darkL, darkL0))

        darkR0 = np.squeeze(np.argwhere((self.shutter[1:]-self.shutter[:-1]) == -1)) + 1
        darkR  = np.hstack((darkR, darkR0))

        if self.shutter[-1] == 0:
            darkL = np.hstack((darkL, self.shutter.size))
        else:
            darkR = np.hstack((darkR, self.shutter.size))

        self.spectra_dark_corr = self.spectra.copy() + countOffset
        self.dark_offset       = np.zeros(self.spectra.shape, dtype=np.float64)
        self.dark_std          = np.zeros(self.spectra.shape, dtype=np.float64)

        Nrecord, Nchannel, Nsensor = self.spectra.shape
        if mode == -1:
            if darkL.size-darkR.size==0:
                if darkL[0]>darkR[0] and darkL[-1]>darkR[-1]:
                    darkL = darkL[:-1]
                    darkR = darkR[1:]
            elif darkL.size-darkR.size==1:
                if darkL[0]>darkR[0] and darkL[-1]<darkR[-1]:
                    darkL = darkL[1:]
                elif darkL[0]<darkR[0] and darkL[-1]>darkR[-1]:
                    darkL = darkL[:-1]
            elif darkR.size-darkL.size==1:
                if darkL[0]>darkR[0] and darkL[-1]<darkR[-1]:
                    darkR = darkR[1:]
                elif darkL[0]<darkR[0] and darkL[-1]>darkR[-1]:
                    darkR = darkR[:-1]
            else:
                exit('Error [READ_SKS.DARK_CORR]: darkL and darkR are wrong.')

            for i in range(darkL.size-1):
                if darkR[i] < darkL[i]:
                    exit('Error [READ_SKS.DARK_CORR]: darkL > darkR.')

                darkLL = darkL[i] + darkExtend
                darkLR = darkR[i] - darkExtend
                darkRL = darkL[i+1] + darkExtend
                darkRR = darkR[i+1] - darkExtend

                if i == 0:
                    self.shutter[:darkLL] = -1  # omit the data before the first dark cycle

                lightL = darkR[i]   + lightExtend
                lightR = darkL[i+1] - lightExtend

                if lightR-lightL>lightThr and darkLR-darkLL>darkThr and darkRR-darkRL>darkThr:

                    self.shutter[darkL[i]:darkLL] = -1
                    self.shutter[darkLR:darkR[i]] = -1
                    self.shutter[darkR[i]:lightL] = -1
                    self.shutter[lightR:darkL[i+1]] = -1
                    self.shutter[darkL[i+1]:darkRL] = -1
                    self.shutter[darkRR:darkR[i+1]] = -1

                    int_dark  = np.append(self.int_time[darkLL:darkLR], self.int_time[darkRL:darkRR]).mean()
                    int_light = self.int_time[lightL:lightR].mean()

                    if np.abs(int_dark - int_light) > 0.0001:
                        self.shutter[lightL:lightR] = -1
                    else:
                        interp_x  = np.append(self.tmhr_corr[darkLL:darkLR], self.tmhr_corr[darkRL:darkRR])
                        if i==darkL.size-2:
                            target_x  = self.tmhr_corr[darkLL:darkRR]
                        else:
                            target_x  = self.tmhr_corr[darkLL:darkRL]

                        for ichan in range(Nchannel):
                            for isen in range(Nsensor):
                                interp_y = np.append(self.spectra[darkLL:darkLR,ichan,isen], self.spectra[darkRL:darkRR,ichan,isen])
                                slope, intercept, r_value, p_value, std_err  = stats.linregress(interp_x, interp_y)
                                if i==darkL.size-2:
                                    self.dark_offset[darkLL:darkRR, ichan, isen] = target_x*slope + intercept
                                    self.spectra_dark_corr[darkLL:darkRR, ichan, isen] -= self.dark_offset[darkLL:darkRR, ichan, isen]
                                    self.dark_std[darkLL:darkRR, ichan, isen] = np.std(interp_y)
                                else:
                                    self.dark_offset[darkLL:darkRL, ichan, isen] = target_x*slope + intercept
                                    self.spectra_dark_corr[darkLL:darkRL, ichan, isen] -= self.dark_offset[darkLL:darkRL, ichan, isen]
                                    self.dark_std[darkLL:darkRL, ichan, isen] = np.std(interp_y)

                else:
                    self.shutter[darkL[i]:darkR[i+1]] = -1

            self.shutter[darkRR:] = -1  # omit the data after the last dark cycle

        elif mode == -2:
            print('Message [DARK_CORR]: Not implemented...')

        elif mode == -3:

            #if darkL.size-darkR.size==0:
                #if darkL[0]>darkR[0] and darkL[-1]>darkR[-1]:
                    #darkL = darkL[:-1]
                    #darkR = darkR[1:]
            #elif darkL.size-darkR.size==1:
                #if darkL[0]>darkR[0] and darkL[-1]<darkR[-1]:
                    #darkL = darkL[1:]
                #elif darkL[0]<darkR[0] and darkL[-1]>darkR[-1]:
                    #darkL = darkL[:-1]
            #elif darkR.size-darkL.size==1:
                #if darkL[0]>darkR[0] and darkL[-1]<darkR[-1]:
                    #darkR = darkR[1:]
                #elif darkL[0]<darkR[0] and darkL[-1]>darkR[-1]:
                    #darkR = darkR[:-1]
            #else:
                #exit('Error [READ_SKS.DARK_CORR]: darkL and darkR are wrong.')

            for i in range(darkR.size):
                darkLL = darkL[i]   + darkExtend
                darkLR = darkR[i]   - darkExtend
                lightL = darkR[i]   + lightExtend
                lightR = darkL[i+1] - lightExtend

                self.shutter[darkL[i]:darkLL] = -1
                self.shutter[darkLR:darkR[i]] = -1
                self.shutter[darkR[i]:lightL] = -1
                self.shutter[lightR:darkL[i+1]] = -1

                int_dark  = self.int_time[darkLL:darkLR].mean()
                int_light = self.int_time[lightL:lightR].mean()
                if np.abs(int_dark - int_light) > 0.0001:
                    self.shutter[lightL:lightR] = -1
                    exit('Error [READ_SKS.DARK_CORR]: inconsistent integration time.')
                else:
                    for itmhr in range(darkLR, lightR):
                        for isen in range(Nsensor):
                            dark_offset0 = np.mean(self.spectra[darkLL:darkLR, :, isen], axis=0)
                            self.dark_offset[itmhr, :, isen] = dark_offset0
                    self.spectra_dark_corr[lightL:lightR,:,:] -= self.dark_offset[lightL:lightR,:,:]

        elif mode == -4:
            print('Message [DARK_CORR]: Not implemented...')


if __name__ == '__main__':

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator

    fname = '/Users/hoch4240/Chen/work/00_reuse/SSFR-util/data/20180302/zenith/s60i300/spc00000.OSA2'
    f_ssfr = READ_NASA_SSFR([fname])

    # figure settings
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(f_ssfr.tmhr_corr, f_ssfr.shutter)
    # ax1.legend(loc='best', fontsize=12, framealpha=0.4)
    plt.show()

    exit()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # python         :          IDL
    #   l            :    long or lonarr
    #   B            :    byte or bytarr
    #   L            :    ulong
    #   h            :    intarr
    # E.g., in IDL:
    #   spec  = {btime:lonarr(2)   , bcdtimstp:bytarr(12),$     2l12B
    #            intime1:long(0)   , intime2:long(0)     ,$     6l
    #            intime3:long(0)   , intime4:long(0)     ,$
    #            accum:long(0)     , shsw:long(0)        ,$
    #            zsit:ulong(0)     , nsit:ulong(0)       ,$     8L
    #            zirt:ulong(0)     , nirt:ulong(0)       ,$
    #            zirx:ulong(0)     , nirx:ulong(0)       ,$
    #            xt:ulong(0)       , it:ulong(0)         ,$
    #            zspecsi:intarr(np), zspecir:intarr(np)  ,$     1024h
    #            nspecsi:intarr(np), nspecir:intarr(np)}
    # in Python:
    # '<2l12B6l8L1024h'
    # ---------------------------------------------------------------------------------------------------------------
