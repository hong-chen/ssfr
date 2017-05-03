import os
import glob
import h5py
import struct
import numpy as np
import datetime
from scipy.io import readsav
from scipy import stats

def READ_SSFR(fname, filetype='sks1', verbose=False):

    fileSize = os.path.getsize(fname)
    iterN    = (fileSize-headLen) // dataLen
    residual = (fileSize-headLen) %  dataLen
    if not (fileSize>headLen and residual==0 and iterN>0):
        print('Warning [READ_SSFR]: %s has invalid data size, return empty arrays.' % fname)
        iterN = 0
        comment    = []
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
        binFmt  = '<d9ld9ll9dl2Bl257hl2Bl257hl2Bl257hlBBl257h'
    elif filetype == 'sks2':
        headLen = 148
        dataLen = 2276
        binFmt  = '<d9ld9ll11dl2Bl257hl2Bl257hl2Bl257hlBBl257h'
    elif filetype == 'osa2':
        headLen = 0
        dataLen = 2124
        binFmt  = '<2l12B6l8L1024h'
    else:
        exit('Error [READ_SSFR]: do not support file type "%s".' % filetype)


    spectra    = np.zeros((iterN, 256, 4), dtype=np.float64) # spectra
    shutter    = np.zeros(iterN          , dtype=np.int32  ) # shutter status (1:closed, 0:open)
    int_time   = np.zeros((iterN, 4)     , dtype=np.float64) # integration time [ms]
    temp       = np.zeros((iterN, 9)     , dtype=np.float64) # temperature
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
        data     = struct.unpack('<d9ld9ll9dl2Bl257hl2Bl257hl2Bl257hlBBl257h', dataRec)

        dataHead = data[:30]
        dataSpec = np.transpose(np.array(data[30:]).reshape((4, 261)))[:, [0, 2, 1, 3]]
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

if __name__ == '__main__':
    pass
