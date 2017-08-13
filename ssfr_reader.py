import os
import glob
import h5py
import struct
import numpy as np
import datetime
from scipy.io import readsav
from scipy import stats

def READ_CU_SSFR(fname, filetype='sks1', headLen=148, dataLen=2260, verbose=False):

    '''
    Description:
    Reader code for Solar Spectral Flux Radiometer (SSFR) developed by Dr. Sebastian Schmidt's group
    at the University of Colorado Bouder.

    How to use:
    fname = '/some/path/2015022000001.SKS'
    comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR(fname, filetype='sks1', verbose=False)

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
        print('Warning [READ_CU_SSFR]: %s has invalid data size, return empty arrays.' % fname)
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

    filetype = filetype.lower()
    if filetype == 'sks1':
        headLen = 148
        dataLen = 2260
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # d9l: frac_second[d] , second[l] , minute[l] , hour[l] , day[l] , month[l] , year[l] , dow[l] , doy[l] , DST[l]
        # d9l: frac_second0[d], second0[l], minute0[l], hour0[l], day0[l], month0[l], year0[l], dow0[l], doy0[l], DST0[l]
        # l9d: null[l], temp(9)[9d]
        # --------------------------          below repeat for sz, sn, iz, in          ----------------------------------
        # l2Bl: int_time[l], shutter[B], EOS[B], null[l]
        # 257h: spectra(257)
        # ---------------------------------------------------------------------------------------------------------------
        binFmt  = '<d9ld9ll9dl2Bl257hl2Bl257hl2Bl257hlBBl257h'
    elif filetype == 'sks2':
        headLen = 148
        dataLen = 2276
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # d9l: frac_second[d] , second[l] , minute[l] , hour[l] , day[l] , month[l] , year[l] , dow[l] , doy[l] , DST[l]
        # d9l: frac_second0[d], second0[l], minute0[l], hour0[l], day0[l], month0[l], year0[l], dow0[l], doy0[l], DST0[l]
        # l11d: null[l], temp(11)[11d]
        # --------------------------          below repeat for sz, sn, iz, in          ----------------------------------
        # l2Bl: int_time[l], shutter[B], EOS[B], null[l]
        # 257h: spectra(257)
        # ---------------------------------------------------------------------------------------------------------------
        binFmt  = '<d9ld9ll11dl2Bl257hl2Bl257hl2Bl257hlBBl257h'
    else:
        exit('Error [READ_SSFR]: do not support file type "%s".' % filetype)

    if verbose:
        print('+' % fname)
        print('Message [READ_CU_SSFR]: Reading %s...' % fname)

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

def READ_NASA_SSFR(fname, filetype='osa2', headLen=0, dataLen=2124, verbose=False):

    fileSize = os.path.getsize(fname)
    iterN    = (fileSize-headLen) // dataLen
    residual = (fileSize-headLen) %  dataLen
    if not (fileSize>headLen and residual==0 and iterN>0):
        print('Warning [READ_NASA_SSFR]: %s has invalid data size, return empty arrays.' % fname)
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

    filetype = filetype.lower()
    if filetype == 'osa2':
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

            if True:
                spectra[i, :, :]  = dataSpec
                shutter[i]        = data[19]
                int_time[i, :]    = np.array(data[14:18])
                temp[i, :]        = np.array(data[20:28])

                # calculate the proleptic Gregorian ordinal of the date
                dtime      = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=data[0])
                jday[i]    = (dtime  - datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0

    return spectra, shutter, int_time, temp, jday, qual_flag, iterN

if __name__ == '__main__':

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator
    from mpl_toolkits.basemap import Basemap

    fname = '/Users/hoch4240/Google Drive/CU LASP/ORACLES/Data/ORACLES 2017/p3/20170812/SSFR/spc01581.OSA2'
    spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR(fname)

    print(spectra.shape)

    # figure settings
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(np.arange(256), spectra[10, :, 1])
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
    pass
