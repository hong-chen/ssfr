import os
import glob
import h5py
import struct
import numpy as np
import datetime
from scipy import stats



def READ_CU_SSFR(fname, headLen=148, dataLen=2276, verbose=False):

    '''
    Description:
    Reader code for Solar Spectral Flux Radiometer (SSFR) developed by Dr. Sebastian Schmidt's group
    at the University of Colorado Bouder.

    How to use:
    fname = '/some/path/2015022000001.SKS'
    comment, spectra, shutter, int_time, temp, jday_ARINC, jday_cRIO, qual_flag, iterN = READ_CU_SSFR(fname, verbose=False)

    comment  (str)        [N/A]    : comment in header
    spectra  (numpy array)[N/A]    : counts of Silicon and InGaAs for both zenith and nadir
    shutter  (numpy array)[N/A]    : shutter status (1:closed(dark), 0:open(light))
    int_time (numpy array)[ms]     : integration time of Silicon and InGaAs for both zenith and nadir
    temp (numpy array)    [Celsius]: temperature variables
    jday_ARINC (numpy array)[day]  : julian days (w.r.t 0001-01-01) of aircraft nagivation system
    jday_cRIO(numpy array)[day]    : julian days (w.r.t 0001-01-01) of SSFR Inertial Navigation System (INS)
    qual_flag(numpy array)[N/A]    : quality flag(1:good, 0:bad)
    iterN (numpy array)   [N/A]    : number of data record

    by Hong Chen (me@hongchen.cz)
    '''

    fileSize = os.path.getsize(fname)
    if fileSize > headLen:
        iterN   = (fileSize-headLen) // dataLen
        residual = (fileSize-headLen) %  dataLen
        if residual != 0:
            print('Warning [READ_CU_SSFR]: %s contains unreadable data, omit the last data record...' % fname)
    else:
        exit('Error [READ_CU_SSFR]: %s has invalid file size.' % fname)

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

    return comment, spectra, shutter, int_time, temp, jday_ARINC, jday_cRIO, qual_flag, iterN



def DARK_CORRECTION(tmhr, shutter, spectra, int_time, mode="dark_interpolate", darkExtend=2, lightExtend=2, countOffset=0, lightThr=10, darkThr=5, fillValue=-1):

    if shutter[0] == 0:
        darkL = np.array([], dtype=np.int32)
        darkR = np.array([0], dtype=np.int32)
    else:
        darkR = np.array([], dtype=np.int32)
        darkL = np.array([0], dtype=np.int32)

    darkL0 = np.squeeze(np.argwhere((shutter[1:]-shutter[:-1]) ==  1)) + 1
    darkL  = np.hstack((darkL, darkL0))

    darkR0 = np.squeeze(np.argwhere((shutter[1:]-shutter[:-1]) == -1)) + 1
    darkR  = np.hstack((darkR, darkR0))

    if shutter[-1] == 0:
        darkL = np.hstack((darkL, shutter.size))
    else:
        darkR = np.hstack((darkR, shutter.size))

    if darkL.size != darkR.size:
        exit('Error [DARK_CORRECTION]: cannot find correct number of dark cycles.')
    else:
        if darkL.size == 1:
            print('Warning [DARK_CORRECTION]: only one dark cycle is detected.')

    if mode == 'dark_interpolate':

        if darkL.size < 2:
            exit('Error [DARK_CORRECTION]: cannot perform \'dark_interpolate\' with less than two dark cycles, try \'dark_mean\'.')

    dark_offset  = np.zeros(spectra.shape, dtype=np.float64)
    Nrecord, Nchannel, Nsensor = spectra.shape


    spectra_corr = spectra.copy() + countOffset
    dark_std     = np.zeros(spectra.shape, dtype=np.float64)

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
            exit('Error [DARK_CORR]: darkL and darkR are wrong.')

        for i in range(darkL.size-1):
            if darkR[i] < darkL[i]:
                exit('Error [DARK_CORR]: darkL > darkR.')

            darkLL = darkL[i] + darkExtend
            darkLR = darkR[i] - darkExtend
            darkRL = darkL[i+1] + darkExtend
            darkRR = darkR[i+1] - darkExtend

            if i == 0:
                shutter[:darkLL] = fillValue  # omit the data before the first dark cycle

            lightL = darkR[i]   + lightExtend
            lightR = darkL[i+1] - lightExtend

            if lightR-lightL>lightThr and darkLR-darkLL>darkThr and darkRR-darkRL>darkThr:

                shutter[darkL[i]:darkLL] = fillValue
                shutter[darkLR:darkR[i]] = fillValue
                shutter[darkR[i]:lightL] = fillValue
                shutter[lightR:darkL[i+1]] = fillValue
                shutter[darkL[i+1]:darkRL] = fillValue
                shutter[darkRR:darkR[i+1]] = fillValue

                int_dark  = np.append(int_time[darkLL:darkLR], int_time[darkRL:darkRR]).mean()
                int_light = int_time[lightL:lightR].mean()

                if np.abs(int_dark - int_light) > 0.0001:
                    shutter[lightL:lightR] = fillValue
                else:
                    interp_x  = np.append(tmhr[darkLL:darkLR], tmhr[darkRL:darkRR])
                    if i==darkL.size-2:
                        target_x  = tmhr[darkL[i]:darkR[i+1]]
                    else:
                        target_x  = tmhr[darkL[i]:darkL[i+1]]

                    for ichan in range(Nchannel):
                        for isen in range(Nsensor):
                            interp_y = np.append(spectra[darkLL:darkLR,ichan,isen], spectra[darkRL:darkRR,ichan,isen])
                            slope, intercept, r_value, p_value, std_err  = stats.linregress(interp_x, interp_y)
                            if i==darkL.size-2:
                                dark_offset[darkL[i]:darkR[i+1], ichan, isen] = target_x*slope + intercept
                                spectra_corr[darkL[i]:darkR[i+1], ichan, isen] -= dark_offset[darkL[i]:darkR[i+1], ichan, isen]
                                dark_std[darkL[i]:darkR[i+1], ichan, isen] = np.std(interp_y)
                            else:
                                dark_offset[darkL[i]:darkL[i+1], ichan, isen] = target_x*slope + intercept
                                spectra_corr[darkL[i]:darkL[i+1], ichan, isen] -= dark_offset[darkL[i]:darkL[i+1], ichan, isen]
                                dark_std[darkL[i]:darkL[i+1], ichan, isen] = np.std(interp_y)

            else:
                shutter[darkL[i]:darkR[i+1]] = fillValue

        shutter[darkRR:] = fillValue  # omit the data after the last dark cycle

        return shutter, spectra_corr, dark_offset, dark_std

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

            shutter[darkL[i]:darkLL] = fillValue
            shutter[darkLR:darkR[i]] = fillValue
            shutter[darkR[i]:lightL] = fillValue
            shutter[lightR:darkL[i+1]] = fillValue

            int_dark  = int_time[darkLL:darkLR].mean()
            int_light = int_time[lightL:lightR].mean()
            if np.abs(int_dark - int_light) > 0.0001:
                shutter[lightL:lightR] = fillValue
                exit('Error [READ_SKS.DARK_CORR]: inconsistent integration time.')
            else:
                for itmhr in range(darkLR, lightR):
                    for isen in range(Nsensor):
                        dark_offset0 = np.mean(spectra[darkLL:darkLR, :, isen], axis=0)
                        dark_offset[itmhr, :, isen] = dark_offset0
                spectra_corr[lightL:lightR,:,:] -= dark_offset[lightL:lightR,:,:]

    elif mode == -4:
        print('Message [DARK_CORR]: Not implemented...')






if __name__ == '__main__':

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator


    fname = '/Users/hoch4240/Chen/work/00_reuse/SSFR-util/CU-SSFR/Belana/data/20180315/1324/zenith/RB/s40_80i200_375/cal/20170314_spc00001.SKS'
    comment, spectra, shutter, int_time, temp, jday_ARINC, jday_cRIO, qual_flag, iterN = READ_CU_SSFR(fname, verbose=False)

    # shutter, spectra_corr, dark_offset, dark_std = DARK_CORRECTION((jday_ARINC-int(jday_ARINC[0]))*24.0, shutter, spectra, int_time)

    # figure settings
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(jday_ARINC, int_time[:, 0])
    # ax1.scatter(jday_ARINC, int_time[:, 2])
    # ax1.legend(loc='best', fontsize=k12, framealpha=0.4)
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
