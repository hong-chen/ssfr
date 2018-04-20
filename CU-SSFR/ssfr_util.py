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

    by Hong Chen (me@hongchen.cz), Sebastian Schmidt (sebastian.schmidt@lasp.colorado.edu)
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

    f.close()

    return comment, spectra, shutter, int_time, temp, jday_ARINC, jday_cRIO, qual_flag, iterN






class CU_SSFR:

    def __init__(self, fnames, Ndata=600, whichTime='arinc', timeOffset=0.0, dark_corr_mode='dark_interpolate'):

        '''
        Description:
        fnames    : list of SSFR files to read
        Ndata     : pre-defined number of data records, any number larger than the "number of data records per file" will work
        whichTime : "ARINC" or "cRIO"
        timeOffset: time offset in [seconds]
        '''

        if type(fnames) is not list:
            exit('Error [CU_SSFR]: input variable should be \'list\'.')
        if len(fnames) == 0:
            exit('Error [CU_SSFR]: input \'list\' is empty.')

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
            comment0, spectra0, shutter0, int_time0, temp0, jday_ARINC0, jday_cRIO0, qual_flag0, iterN0 = READ_CU_SSFR(fname, verbose=False)

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



    def COUNT2FLUX(self, cal, wvl_zen_join=900.0, wvl_nad_join=900.0):

        """
        Convert digital count to flux (irradiance)
        """

        logic_zen_si = (cal.wvl_zen_si <= wvl_zen_join)
        logic_zen_in = (cal.wvl_zen_in >= wvl_zen_join)
        n_zen_si = logic_zen_si.sum()
        n_zen_in = logic_zen_in.sum()
        n_zen    = n_zen_si + n_zen_in
        self.wvl_zen = np.append(cal.wvl_zen_si[logic_zen_si], cal.wvl_zen_in[logic_zen_in][::-1])

        logic_nad_si = (cal.wvl_nad_si <= wvl_nad_join)
        logic_nad_in = (cal.wvl_nad_in >= wvl_nad_join)
        n_nad_si = logic_nad_si.sum()
        n_nad_in = logic_nad_in.sum()
        n_nad    = n_nad_si + n_nad_in
        self.wvl_nad = np.append(cal.wvl_nad_si[logic_nad_si], cal.wvl_nad_in[logic_nad_in][::-1])

        self.spectra_flux_zen = np.zeros((self.tmhr.size, n_zen), dtype=np.float64)
        self.spectra_flux_nad = np.zeros((self.tmhr.size, n_nad), dtype=np.float64)

        for i in range(self.tmhr.size):
            self.spectra_flux_zen[i, :n_zen_si] =  self.spectra_dark_corr[i, logic_zen_si, 0]/float(self.int_time[i, 0])/cal.secondary_response_zen_si[logic_zen_si]
            self.spectra_flux_zen[i, n_zen_si:] = (self.spectra_dark_corr[i, logic_zen_in, 1]/float(self.int_time[i, 1])/cal.secondary_response_zen_in[logic_zen_in])[::-1]
            self.spectra_flux_nad[i, :n_nad_si] =  self.spectra_dark_corr[i, logic_nad_si, 2]/float(self.int_time[i, 2])/cal.secondary_response_nad_si[logic_nad_si]
            self.spectra_flux_nad[i, n_nad_si:] = (self.spectra_dark_corr[i, logic_nad_in, 3]/float(self.int_time[i, 3])/cal.secondary_response_nad_in[logic_nad_in])[::-1]







def DARK_CORRECTION(tmhr0, shutter0, spectra0, mode="dark_interpolate", darkExtend=2, lightExtend=2, lightThr=10, darkThr=5, fillValue=-99999, verbose=False):

    tmhr              = tmhr0.copy()
    shutter           = shutter0.copy()
    spectra           = spectra0.copy()
    Nrecord, Nchannel = spectra.shape

    spectra_dark_corr      = np.zeros_like(spectra)
    spectra_dark_corr[...] = fillValue

    # only dark or light cycle present
    if np.unique(shutter).size == 1:

        if mode.lower() != 'mean':
            print('Warning [DARK_CORRECTION]: only one light/dark cycle is detected, \'%s\' is not supported, return fill value.' % mode)
            return spectra_dark_corr
        else:
            if np.unique(shutter)[0] == 0:
                if verbose:
                    print('Warning [DARK_CORRECTION]: only one light cycle is detected.')
                mean = np.mean(spectra[lightExtend:-lightExtend, :], axis=0)
                spectra_dark_corr = np.tile(mean, spectra.shape[0]).reshape(spectra.shape)
                return spectra_dark_corr
            elif np.unique(shutter)[0] == 1:
                if verbose:
                    print('Warning [DARK_CORRECTION]: only one dark cycle is detected.')
                mean = np.mean(spectra[darkExtend:-darkExtend, :], axis=0)
                spectra_dark_corr = np.tile(mean, spectra.shape[0]).reshape(spectra.shape)
                return spectra_dark_corr
            else:
                print('Exit [DARK_CORRECTION]: cannot interpret shutter status.')

    # both dark and light cycles present
    else:

        dark_offset            = np.zeros_like(spectra)
        dark_std               = np.zeros_like(spectra)
        dark_offset[...]       = fillValue
        dark_std[...]          = fillValue

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

        # ??????????????????????????????????????????????????????????????????????????????
        # this part might need more work
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
            exit('Error [DARK_CORRECTION]: darkL and darkR do not match.')
        # ??????????????????????????????????????????????????????????????????????????????

        if darkL.size != darkR.size:
            exit('Error [DARK_CORRECTION]: the number of dark cycles is incorrect.')

        if mode.lower() == 'dark_interpolate':

            shutter[:darkL[0]+darkExtend] = fillValue  # omit the data before the first dark cycle

            for i in range(darkL.size-1):

                if darkR[i] < darkL[i]:
                    exit('Error [DARK_CORRECTION]: darkR[%d]=%d is smaller than darkL[%d]=%d.' % (i,darkR[i],i,darkL[i]))

                darkLL = darkL[i]   + darkExtend
                darkLR = darkR[i]   - darkExtend
                darkRL = darkL[i+1] + darkExtend
                darkRR = darkR[i+1] - darkExtend
                lightL = darkR[i]   + lightExtend
                lightR = darkL[i+1] - lightExtend

                shutter[darkL[i]:darkLL] = fillValue
                shutter[darkLR:darkR[i]] = fillValue
                shutter[darkR[i]:lightL] = fillValue
                shutter[lightR:darkL[i+1]] = fillValue
                shutter[darkL[i+1]:darkRL] = fillValue
                shutter[darkRR:darkR[i+1]] = fillValue

                if lightR-lightL>lightThr and darkLR-darkLL>darkThr and darkRR-darkRL>darkThr:

                    interp_x = np.append(tmhr[darkLL:darkLR], tmhr[darkRL:darkRR])
                    target_x = tmhr[darkL[i]:darkL[i+1]]

                    for iChan in range(Nchannel):
                        interp_y = np.append(spectra[darkLL:darkLR, iChan], spectra[darkRL:darkRR, iChan])
                        slope, intercept, r_value, p_value, std_err  = stats.linregress(interp_x, interp_y)
                        dark_offset[darkL[i]:darkL[i+1], iChan] = target_x*slope + intercept
                        dark_std[darkL[i]:darkL[i+1], iChan]    = np.std(interp_y)
                        spectra_dark_corr[darkL[i]:darkL[i+1], iChan] = spectra[darkL[i]:darkL[i+1], iChan] - dark_offset[darkL[i]:darkL[i+1], iChan]

                else:
                    shutter[darkL[i]:darkR[i+1]] = fillValue

            shutter[darkRR:] = fillValue  # omit the data after the last dark cycle
            spectra_dark_corr[shutter==fillValue, :] = fillValue

            return spectra_dark_corr

        else:
            exit('Error [DARK_CORRECTION]: \'%s\' has not been implemented yet.' % mode)






class CALIBRATION_CU_SSFR:

    def __init__(self, config):

        self.config = config
        self.CAL_WAVELENGTH()
        self.CAL_PRIMARY_RESPONSE(config)
        self.CAL_TRANSFER(config)
        self.CAL_SECONDARY_RESPONSE(config)

        # self.CAL_ANGULAR_RESPONSE(config)


    def CAL_WAVELENGTH(self):

        self.chanNum = 256
        xChan = np.arange(self.chanNum)

        self.coef_zen_si = np.array([301.946,  3.31877,  0.00037585,  -1.76779e-6, 0])
        self.coef_zen_in = np.array([2202.33, -4.35275, -0.00269498,   3.84968e-6, -2.33845e-8])

        self.coef_nad_si = np.array([302.818,  3.31912,  0.000343831, -1.81135e-6, 0])
        self.coef_nad_in = np.array([2210.29,  -4.5998,  0.00102444,  -1.60349e-5, 1.29122e-8])

        self.wvl_zen_si = self.coef_zen_si[0] + self.coef_zen_si[1]*xChan + self.coef_zen_si[2]*xChan**2 + self.coef_zen_si[3]*xChan**3 + self.coef_zen_si[4]*xChan**4
        self.wvl_zen_in = self.coef_zen_in[0] + self.coef_zen_in[1]*xChan + self.coef_zen_in[2]*xChan**2 + self.coef_zen_in[3]*xChan**3 + self.coef_zen_in[4]*xChan**4

        self.wvl_nad_si = self.coef_nad_si[0] + self.coef_nad_si[1]*xChan + self.coef_nad_si[2]*xChan**2 + self.coef_nad_si[3]*xChan**3 + self.coef_nad_si[4]*xChan**4
        self.wvl_nad_in = self.coef_nad_in[0] + self.coef_nad_in[1]*xChan + self.coef_nad_in[2]*xChan**2 + self.coef_nad_in[3]*xChan**3 + self.coef_nad_in[4]*xChan**4


    def CAL_PRIMARY_RESPONSE(self, config, lampTag='f-1324', fdirLamp='/Users/hoch4240/Chen/other/data/aux_ssfr'):

        # read in calibrated lamp data and interpolated at SSFR wavelengths
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.fnameLamp = '%s/%s.dat' % (fdirLamp, lampTag)
        if not os.path.exists(self.fnameLamp):
            exit('Error [CALIBRATION_CU_SSFR.CAL_PRIMARY_RESPONSE]: cannot locate lamp standards for %s.' % lampTag.title())

        data = np.loadtxt(self.fnameLamp)
        data_wvl  = data[:, 0]
        if lampTag == 'f-1324':
            data_flux = data[:, 1]*10000.0
        elif lampTag == '506c':
            data_flux = data[:, 1]*0.01

        lampStd_zen_si = np.interp(self.wvl_zen_si, data_wvl, data_flux)
        lampStd_zen_in = np.interp(self.wvl_zen_in, data_wvl, data_flux)
        lampStd_nad_si = np.interp(self.wvl_nad_si, data_wvl, data_flux)
        lampStd_nad_in = np.interp(self.wvl_nad_in, data_wvl, data_flux)
        # ---------------------------------------------------------------------------
        # so far we have (W m^-2 nm^-1 as a function of wavelength)

        # for zenith
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.primary_response_zen_si = {}
        for intTime in config['int_time_primary_zen_si']:
            if config['fname_primary_zen_cal'] != None and config['fname_primary_zen_dark'] != None:
                try:
                    ssfr = CU_SSFR([config['fname_primary_zen_cal']], dark_corr_mode='mean')
                    spectra_zen_si_l = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 0]-intTime)<0.00001, :, 0], axis=0)
                    ssfr = CU_SSFR([config['fname_primary_zen_dark']], dark_corr_mode='mean')
                    spectra_zen_si_d = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 0]-intTime)<0.00001, :, 0], axis=0)
                    spectra_zen_si = spectra_zen_si_l - spectra_zen_si_d
                    spectra_zen_si[spectra_zen_si<=0.0] = np.nan
                    self.primary_response_zen_si[intTime] = spectra_zen_si / intTime / lampStd_zen_si
                except:
                    self.primary_response_zen_si[intTime] = np.repeat(np.nan, self.chanNum)
            else:
                self.primary_response_zen_si[intTime] = np.repeat(np.nan, self.chanNum)

        self.primary_response_zen_in = {}
        for intTime in config['int_time_primary_zen_in']:
            if config['fname_primary_zen_cal'] != None and config['fname_primary_zen_dark'] != None:
                try:
                    ssfr = CU_SSFR([config['fname_primary_zen_cal']], dark_corr_mode='mean')
                    spectra_zen_in_l = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 1]-intTime)<0.00001, :, 1], axis=0)
                    ssfr = CU_SSFR([config['fname_primary_zen_dark']], dark_corr_mode='mean')
                    spectra_zen_in_d = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 1]-intTime)<0.00001, :, 1], axis=0)
                    spectra_zen_in = spectra_zen_in_l - spectra_zen_in_d
                    spectra_zen_in[spectra_zen_in<=0.0] = np.nan
                    self.primary_response_zen_in[intTime] = spectra_zen_in / intTime / lampStd_zen_in
                except:
                    self.primary_response_zen_in[intTime] = np.repeat(np.nan, self.chanNum)
            else:
                self.primary_response_zen_in[intTime] = np.repeat(np.nan, self.chanNum)
        # ---------------------------------------------------------------------------

        # for nadir
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.primary_response_nad_si = {}
        for intTime in config['int_time_primary_nad_si']:
            if config['fname_primary_nad_cal'] != None and config['fname_primary_nad_dark'] != None:
                try:
                    ssfr = CU_SSFR([config['fname_primary_nad_cal']], dark_corr_mode='mean')
                    spectra_nad_si_l = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 2]-intTime)<0.00001, :, 2], axis=0)
                    ssfr = CU_SSFR([config['fname_primary_nad_dark']], dark_corr_mode='mean')
                    spectra_nad_si_d = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 2]-intTime)<0.00001, :, 2], axis=0)
                    spectra_nad_si = spectra_nad_si_l - spectra_nad_si_d
                    spectra_nad_si[spectra_nad_si<=0.0] = np.nan
                    self.primary_response_nad_si[intTime] = spectra_nad_si / intTime / lampStd_nad_si
                except:
                    self.primary_response_nad_si[intTime] = np.repeat(np.nan, self.chanNum)
            else:
                self.primary_response_nad_si[intTime] = np.repeat(np.nan, self.chanNum)

        self.primary_response_nad_in = {}
        for intTime in config['int_time_primary_nad_in']:
            if config['fname_primary_nad_cal'] != None and config['fname_primary_nad_dark'] != None:
                try:
                    ssfr = CU_SSFR([config['fname_primary_nad_cal']], dark_corr_mode='mean')
                    spectra_nad_in_l = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 3]-intTime)<0.00001, :, 3], axis=0)
                    ssfr = CU_SSFR([config['fname_primary_nad_dark']], dark_corr_mode='mean')
                    spectra_nad_in_d = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 3]-intTime)<0.00001, :, 3], axis=0)
                    spectra_nad_in = spectra_nad_in_l - spectra_nad_in_d
                    spectra_nad_in[spectra_nad_in<=0.0] = np.nan
                    self.primary_response_nad_in[intTime] = spectra_nad_in / intTime / lampStd_nad_in
                except:
                    self.primary_response_nad_in[intTime] = np.repeat(np.nan, self.chanNum)
        else:
            self.primary_response_nad_in[intTime] = np.repeat(np.nan, self.chanNum)
        # ---------------------------------------------------------------------------


    def CAL_TRANSFER(self, config):

        # for zenith
        if config['fname_transfer_zen_cal'] != None and config['fname_transfer_zen_dark'] != None:
            ssfr = CU_SSFR([config['fname_transfer_zen_cal']], dark_corr_mode='mean')
            spectra_zen_si_l = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 0]-config['int_time_transfer_zen_si'])<0.00001, :, 0], axis=0)
            spectra_zen_in_l = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 1]-config['int_time_transfer_zen_in'])<0.00001, :, 1], axis=0)

            ssfr = CU_SSFR([config['fname_transfer_zen_dark']], dark_corr_mode='mean')
            spectra_zen_si_d = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 0]-config['int_time_transfer_zen_si'])<0.00001, :, 0], axis=0)
            spectra_zen_in_d = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 1]-config['int_time_transfer_zen_in'])<0.00001, :, 1], axis=0)

            spectra_zen_si = spectra_zen_si_l - spectra_zen_si_d
            # spectra_zen_si[spectra_zen_si<=0.0] = 0.00000001
            spectra_zen_si[spectra_zen_si<=0.0] = np.nan

            spectra_zen_in = spectra_zen_in_l - spectra_zen_in_d
            # spectra_zen_in[spectra_zen_in<=0.0] = 0.00000001
            spectra_zen_in[spectra_zen_in<=0.0] = np.nan

            self.field_lamp_zen_si = spectra_zen_si / config['int_time_transfer_zen_si'] / self.primary_response_zen_si
            self.field_lamp_zen_in = spectra_zen_in / config['int_time_transfer_zen_in'] / self.primary_response_zen_in
        else:
            print('Warning [CALIBRATION_CU_SSFR.CAL_TRANSFER]: cannot locate data files for calculating zenith transfer, return NaN.')
            self.field_lamp_zen_si = np.repeat(np.nan, self.chanNum)
            self.field_lamp_zen_in = np.repeat(np.nan, self.chanNum)


        # for nadir
        if config['fname_transfer_nad_cal'] != None and config['fname_transfer_nad_dark'] != None:
            ssfr = CU_SSFR([config['fname_transfer_nad_cal']], dark_corr_mode='mean')
            spectra_nad_si_l = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 2]-config['int_time_transfer_nad_si'])<0.00001, :, 2], axis=0)
            spectra_nad_in_l = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 3]-config['int_time_transfer_nad_in'])<0.00001, :, 3], axis=0)

            ssfr = CU_SSFR([config['fname_transfer_nad_dark']], dark_corr_mode='mean')
            spectra_nad_si_d = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 2]-config['int_time_transfer_nad_si'])<0.00001, :, 2], axis=0)
            spectra_nad_in_d = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 3]-config['int_time_transfer_nad_in'])<0.00001, :, 3], axis=0)

            spectra_nad_si = spectra_nad_si_l - spectra_nad_si_d
            # spectra_nad_si[spectra_nad_si<=0.0] = 0.00000001
            spectra_nad_si[spectra_nad_si<=0.0] = np.nan

            spectra_nad_in = spectra_nad_in_l - spectra_nad_in_d
            # spectra_nad_in[spectra_nad_in<=0.0] = 0.00000001
            spectra_nad_in[spectra_nad_in<=0.0] = np.nan

            self.field_lamp_nad_si = spectra_nad_si / config['int_time_transfer_nad_si'] / self.primary_response_nad_si
            self.field_lamp_nad_in = spectra_nad_in / config['int_time_transfer_nad_in'] / self.primary_response_nad_in
        else:
            print('Warning [CALIBRATION_CU_SSFR.CAL_TRANSFER]: cannot locate data files for calculating nadir transfer, return NaN.')
            self.field_lamp_nad_si = np.repeat(np.nan, self.chanNum)
            self.field_lamp_nad_in = np.repeat(np.nan, self.chanNum)


    def CAL_SECONDARY_RESPONSE(self, config):

        # for zenith
        if config['fname_secondary_zen_cal'] != None and config['fname_secondary_zen_dark'] != None:
            ssfr = CU_SSFR([config['fname_secondary_zen_cal']], dark_corr_mode='mean')
            spectra_zen_si_l = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 0]-config['int_time_secondary_zen_si'])<0.00001, :, 0], axis=0)
            spectra_zen_in_l = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 1]-config['int_time_secondary_zen_in'])<0.00001, :, 1], axis=0)

            ssfr = CU_SSFR([config['fname_secondary_zen_dark']], dark_corr_mode='mean')
            spectra_zen_si_d = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 0]-config['int_time_secondary_zen_si'])<0.00001, :, 0], axis=0)
            spectra_zen_in_d = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 1]-config['int_time_secondary_zen_in'])<0.00001, :, 1], axis=0)

            spectra_zen_si = spectra_zen_si_l - spectra_zen_si_d
            # spectra_zen_si[spectra_zen_si<=0.0] = 0.00000001
            spectra_zen_si[spectra_zen_si<=0.0] = np.nan

            spectra_zen_in = spectra_zen_in_l - spectra_zen_in_d
            # spectra_zen_in[spectra_zen_in<=0.0] = 0.00000001
            spectra_zen_in[spectra_zen_in<=0.0] = np.nan

            self.secondary_response_zen_si = spectra_zen_si / config['int_time_secondary_zen_si'] / self.field_lamp_zen_si
            self.secondary_response_zen_in = spectra_zen_in / config['int_time_secondary_zen_in'] / self.field_lamp_zen_in
        else:
            print('Warning [CALIBRATION_CU_SSFR.CAL_PRIMARY_RESPONSE]: cannot locate data files for calculating zenith secondary response, return NaN.')
            self.secondary_response_zen_si = np.repeat(np.nan, self.chanNum)
            self.secondary_response_zen_in = np.repeat(np.nan, self.chanNum)


        # for nadir
        if config['fname_secondary_nad_cal'] != None and config['fname_secondary_nad_dark'] != None:
            ssfr = CU_SSFR([config['fname_secondary_nad_cal']], dark_corr_mode='mean')
            spectra_nad_si_l = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 2]-config['int_time_secondary_nad_si'])<0.00001, :, 2], axis=0)
            spectra_nad_in_l = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 3]-config['int_time_secondary_nad_in'])<0.00001, :, 3], axis=0)

            ssfr = CU_SSFR([config['fname_secondary_nad_dark']], dark_corr_mode='mean')
            spectra_nad_si_d = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 2]-config['int_time_secondary_nad_si'])<0.00001, :, 2], axis=0)
            spectra_nad_in_d = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 3]-config['int_time_secondary_nad_in'])<0.00001, :, 3], axis=0)

            spectra_nad_si = spectra_nad_si_l - spectra_nad_si_d
            # spectra_nad_si[spectra_nad_si<=0.0] = 0.00000001
            spectra_nad_si[spectra_nad_si<=0.0] = np.nan

            spectra_nad_in = spectra_nad_in_l - spectra_nad_in_d
            # spectra_nad_in[spectra_nad_in<=0.0] = 0.00000001
            spectra_nad_in[spectra_nad_in<=0.0] = np.nan

            self.secondary_response_nad_si = spectra_nad_si / config['int_time_secondary_nad_si'] / self.field_lamp_nad_si
            self.secondary_response_nad_in = spectra_nad_in / config['int_time_secondary_nad_in'] / self.field_lamp_nad_in
        else:
            print('Warning [CALIBRATION_CU_SSFR.CAL_PRIMARY_RESPONSE]: cannot locate data files for calculating nadir secondary response, return NaN.')
            self.secondary_response_nad_si = np.repeat(np.nan, self.chanNum)
            self.secondary_response_nad_in = np.repeat(np.nan, self.chanNum)


    def CAL_ANGULAR_RESPONSE(self, config):

        print('under development')







def PLOT():
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    rcParams['font.size'] = 18
    fig = plt.figure(figsize=(7, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(xChan, self.wvl_zen_si, label='Silicon', c='red')
    ax1.scatter(xChan, self.wvl_zen_in, label='InGaAs', c='blue')
    # ax1.scatter(xChan, self.wvl_nad_si, label='Silicon', c='red')
    # ax1.scatter(xChan, self.wvl_nad_in, label='InGaAs', c='blue')
    ax1.set_xlim((0, 255))
    ax1.set_ylim((250, 2250))
    ax1.set_xlabel('Channel Number')
    ax1.set_ylabel('Wavelength [nm]')
    # ax1.set_title('Nadir')
    ax1.set_title('Zenith')
    ax1.legend(loc='upper right', framealpha=0.4, fontsize=18)
    plt.savefig('wvl_zenith.png')
    plt.show()
    exit()
    # ---------------------------------------------------------------------

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(5, 3))
    ax1 = fig.add_subplot(111)
    # ax1.scatter(self.wvl_zen_si, lampStd_zen_si)
    # ax1.scatter(self.wvl_zen_in, lampStd_zen_in)
    ax1.scatter(self.wvl_nad_si, lampStd_nad_si)
    # ax1.scatter(self.wvl_nad_in, lampStd_nad_in)
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
    # ax1.set_xlim((200, 2600))
    # ax1.xaxis.set_major_locator(FixedLocator(np.arange(200, 2601, 400)))
    # ax1.set_ylim((0.0, 0.25))
    # ax1.set_title('Zenith Silicon')
    # ax1.set_title('Zenith InGaAs')
    ax1.set_title('Nadir Silicon')
    # ax1.set_title('Nadir InGaAs')
    # ax1.legend(loc='upper right', fontsize=10, framealpha=0.4)
    # plt.savefig('std_zen_si.png')
    # plt.savefig('std_zen_in.png')
    plt.savefig('std_nad_si.png')
    # plt.savefig('std_nad_in.png')
    plt.show()
    exit()
    # ---------------------------------------------------------------------







if __name__ == '__main__':

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

    import matplotlib as mpl
    from matplotlib import rcParams
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator

    config = {
              'fname_primary_zen_cal'   : '/Users/hoch4240/Desktop/SSFR_cal/Alvin/20180320/1324/zenith/108692-1/si45_90i250_400/cal/20180320_spc00001.SKS',
              'fname_primary_zen_dark'  : '/Users/hoch4240/Desktop/SSFR_cal/Alvin/20180320/1324/zenith/108692-1/si45_90i250_400/dark/20180320_spc00001.SKS',
              'fname_primary_nad_cal'   : '/Users/hoch4240/Desktop/SSFR_cal/Alvin/20180320/1324/nadir/2008-4/si45_90i250_400/cal/20180320_spc00001.SKS',
              'fname_primary_nad_dark'  : '/Users/hoch4240/Desktop/SSFR_cal/Alvin/20180320/1324/nadir/2008-4/si45_90i250_400/dark/20180320_spc00001.SKS',
              'fname_transfer_zen_cal'  : '/Users/hoch4240/Desktop/SSFR_cal/Alvin/20180320/150C/zenith/108692-1/si45_90i250_400/cal/20180320_spc00001.SKS',
              'fname_transfer_zen_dark' : '/Users/hoch4240/Desktop/SSFR_cal/Alvin/20180320/150C/zenith/108692-1/si45_90i250_400/dark/20180320_spc00001.SKS',
              'fname_transfer_nad_cal'  : '/Users/hoch4240/Desktop/SSFR_cal/Alvin/20180320/150C/nadir/2008-4/si45_90i250_400/cal/20180320_spc00001.SKS',
              'fname_transfer_nad_dark' : '/Users/hoch4240/Desktop/SSFR_cal/Alvin/20180320/150C/nadir/2008-4/si45_90i250_400/dark/20180320_spc00001.SKS',
              'fname_secondary_zen_cal' : None,
              'fname_secondary_zen_dark': None,
              'fname_secondary_nad_cal' : '/Users/hoch4240/Chen/work/00_reuse/SSFR-util/CU-SSFR/Belana/data/20180313/150C/nadir/s45_90i250_400/cal/20180313_spc00002.SKS',
              'fname_secondary_nad_dark': '/Users/hoch4240/Chen/work/00_reuse/SSFR-util/CU-SSFR/Belana/data/20180313/150C/nadir/s45_90i250_400/dark/20180313_spc00001.SKS',

              'int_time_primary_zen_si'  : [45, 75],
              'int_time_primary_zen_in'  : [250, 375],
              'int_time_primary_nad_si'  : [45, 90],
              'int_time_primary_nad_in'  : [250, 375],
              'int_time_transfer_zen_si' : 45,
              'int_time_transfer_zen_in' : 250,
              'int_time_transfer_nad_si' : 45,
              'int_time_transfer_nad_in' : 250,
              'int_time_secondary_zen_si': 45,
              'int_time_secondary_zen_in': 250,
              'int_time_secondary_nad_si': 45,
              'int_time_secondary_nad_in': 250
             }

    cal = CALIBRATION_CU_SSFR(config)



    fnames = sorted(glob.glob('/Users/hoch4240/Chen/work/00_reuse/SSFR-util/CU-SSFR/Belana/data/20180313/data/*.SKS'))
    ssfr   = CU_SSFR(fnames)
    print(ssfr.int_time_info)
    ssfr.COUNT2FLUX(cal)



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    # ax1.plot(ssfr.wvl_nad, ssfr.spectra_flux_nad[1000, :])
    # ax1.plot(ssfr.wvl_nad, ssfr.spectra_flux_nad[1001, :])
    # ax1.plot(cal.wvl_nad_si, ssfr.spectra_dark_corr[1000, :, 2])
    # ax1.plot(cal.wvl_nad_in, ssfr.spectra_dark_corr[1000, :, 3])
    # ax1.plot(cal.wvl_nad_si, ssfr.spectra_dark_corr[1001, :, 2])
    # ax1.plot(cal.wvl_nad_in, ssfr.spectra_dark_corr[1001, :, 3])

    ax1.scatter(ssfr.tmhr, ssfr.spectra_flux_nad[:, 100])
    # ax1.set_xlim(())
    # ax1.set_ylim(())
    # ax1.legend(loc='upper right', fontsize=10, framealpha=0.4)
    # plt.savefig('test.png')
    plt.show()
    exit()
    # ---------------------------------------------------------------------
