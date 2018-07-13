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


    def COUNT2RADIATION(self, cal, wvl_zen_join=900.0, wvl_nad_join=900.0, whichRadiation={'zenith':'radiance', 'nadir':'irradiance'}, wvlRange=[350, 2100]):

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
            if whichRadiation['zenith'] == 'radiance':
                self.spectra_zen[i, :n_zen_si] =  self.spectra_dark_corr[i, logic_zen_si, 0]/float(self.int_time[i, 0])/(np.pi * cal.primary_response_zen_si[90][logic_zen_si])
                self.spectra_zen[i, n_zen_si:] = (self.spectra_dark_corr[i, logic_zen_in, 1]/float(self.int_time[i, 1])/(np.pi * cal.primary_response_zen_in[375][logic_zen_in]))[::-1]
            elif whichRadiation['zenith'] == 'irradiance':
                self.spectra_zen[i, :n_zen_si] =  self.spectra_dark_corr[i, logic_zen_si, 0]/float(self.int_time[i, 0])/cal.secondary_response_zen_si[90][logic_zen_si]
                self.spectra_zen[i, n_zen_si:] = (self.spectra_dark_corr[i, logic_zen_in, 1]/float(self.int_time[i, 1])/cal.secondary_response_zen_in[375][logic_zen_in])[::-1]

            if whichRadiation['nadir'] == 'radiance':
                self.spectra_nad[i, :n_nad_si] =  self.spectra_dark_corr[i, logic_nad_si, 2]/float(self.int_time[i, 2])/(np.pi * cal.primary_response_nad_si[90][logic_nad_si])
                self.spectra_nad[i, n_nad_si:] = (self.spectra_dark_corr[i, logic_nad_in, 3]/float(self.int_time[i, 3])/(np.pi * cal.primary_response_nad_in[375][logic_nad_in]))[::-1]
            elif whichRadiation['nadir'] == 'irradiance':
                self.spectra_nad[i, :n_nad_si] =  self.spectra_dark_corr[i, logic_nad_si, 2]/float(self.int_time[i, 2])/cal.secondary_response_nad_si[90][logic_nad_si]
                self.spectra_nad[i, n_nad_si:] = (self.spectra_dark_corr[i, logic_nad_in, 3]/float(self.int_time[i, 3])/cal.secondary_response_nad_in[375][logic_nad_in])[::-1]







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
        self.CAL_PRIMARY_RESPONSE(self.config)
        self.CAL_TRANSFER(self.config)
        # for key in self.config.keys():
        #     print(key, self.config[key])
        # self.CAL_SECONDARY_RESPONSE(config)
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


    def CAL_PRIMARY_RESPONSE(self, config, lampTag='f-1324', fdirLamp='aux'):

        # read in calibrated lamp data and interpolated at SSFR wavelengths
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.fnameLamp = '%s/%s.dat' % (os.path.abspath(fdirLamp), lampTag)
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
        # Silicon
        print('Primary [Zenith Silicon]: processing primary response...')
        self.primary_response_zen_si = {}
        iSen = 0
        try:
            ssfr_l = CU_SSFR([config['fname_primary_zen_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_primary_zen_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_primary_zen_si'] = intTimes_l
            else:
                exit('Primary [Zenith Silicon]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_primary_zen_si']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.primary_response_zen_si[intTime] = spectra / intTime / lampStd_zen_si
        except:
            print('Primary [Zenith Silicon]: Cannot read calibration files.')
            self.primary_response_zen_si[-1] = np.repeat(np.nan, self.chanNum)


        # InGaAs
        self.primary_response_zen_in = {}
        iSen = 1
        try:
            ssfr_l = CU_SSFR([config['fname_primary_zen_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_primary_zen_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_primary_zen_in'] = intTimes_l
            else:
                exit('Primary [Zenith InGaAs]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_primary_zen_in']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.primary_response_zen_in[intTime] = spectra / intTime / lampStd_zen_in
        except:
            print('Primary [Zenith InGaAs]: Cannot read calibration files.')
            self.primary_response_zen_in[-1] = np.repeat(np.nan, self.chanNum)
        # ---------------------------------------------------------------------------



        # for nadir
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Silicon
        self.primary_response_nad_si = {}
        iSen = 2
        try:
            ssfr_l = CU_SSFR([config['fname_primary_nad_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_primary_nad_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_primary_nad_si'] = intTimes_l
            else:
                exit('Primary [Nadir Silicon]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_primary_nad_si']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.primary_response_nad_si[intTime] = spectra / intTime / lampStd_nad_si
        except:
            print('Primary [Nadir Silicon]: Cannot read calibration files.')
            self.primary_response_nad_si[-1] = np.repeat(np.nan, self.chanNum)


        # InGaAs
        self.primary_response_nad_in = {}
        iSen = 3
        try:
            ssfr_l = CU_SSFR([config['fname_primary_nad_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_primary_nad_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_primary_nad_in'] = intTimes_l
            else:
                exit('Primary [Nadir InGaAs]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_primary_nad_in']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.primary_response_nad_in[intTime] = spectra / intTime / lampStd_nad_in
        except:
            print('Primary [Nadir InGaAs]: Cannot read calibration files.')
            self.primary_response_nad_in[-1] = np.repeat(np.nan, self.chanNum)
        # ---------------------------------------------------------------------------


    def CAL_TRANSFER(self, config):

        # for zenith
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Silicon
        self.field_lamp_zen_si = {}
        iSen = 0
        try:
            ssfr_l = CU_SSFR([config['fname_transfer_zen_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_transfer_zen_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_transfer_zen_si'] = intTimes_l
            else:
                exit('Transfer [Zenith Silicon]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_transfer_zen_si']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.field_lamp_zen_si[intTime] = spectra / intTime / self.primary_response_zen_si[intTime]
        except:
            print('Transfer [Zenith Silicon]: Cannot read calibration files.')
            self.field_lamp_zen_si[-1] = np.repeat(np.nan, self.chanNum)


        # InGaAs
        self.field_lamp_zen_in = {}
        iSen = 1
        try:
            ssfr_l = CU_SSFR([config['fname_transfer_zen_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_transfer_zen_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_transfer_zen_in'] = intTimes_l
            else:
                exit('Transfer [Zenith InGaAs]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_transfer_zen_in']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.field_lamp_zen_in[intTime] = spectra / intTime / self.primary_response_zen_in[intTime]
        except:
            print('Transfer [Zenith InGaAs]: Cannot read calibration files.')
            self.field_lamp_zen_in[-1] = np.repeat(np.nan, self.chanNum)
        # ---------------------------------------------------------------------------



        # for nadir
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Silicon
        self.field_lamp_nad_si = {}
        iSen = 2
        try:
            ssfr_l = CU_SSFR([config['fname_transfer_nad_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_transfer_nad_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_transfer_nad_si'] = intTimes_l
            else:
                exit('Transfer [Nadir Silicon]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_transfer_nad_si']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.field_lamp_nad_si[intTime] = spectra / intTime / self.primary_response_nad_si[intTime]
        except:
            print('Transfer [Nadir Silicon]: Cannot read calibration files.')
            self.field_lamp_nad_si[-1] = np.repeat(np.nan, self.chanNum)


        # InGaAs
        self.field_lamp_nad_in = {}
        iSen = 3
        try:
            ssfr_l = CU_SSFR([config['fname_transfer_nad_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_transfer_nad_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_transfer_nad_in'] = intTimes_l
            else:
                exit('Transfer [Nadir InGaAs]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_transfer_nad_in']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.field_lamp_nad_in[intTime] = spectra / intTime / self.primary_response_nad_in[intTime]
        except:
            print('Transfer [Nadir InGaAs]: Cannot read calibration files.')
            self.field_lamp_nad_in[-1] = np.repeat(np.nan, self.chanNum)
        # ---------------------------------------------------------------------------


    def CAL_SECONDARY_RESPONSE(self, config):

        # for zenith
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.secondary_response_zen_si = {}
        for intTime in config['int_time_secondary_zen_si']:
            try:
                ssfr = CU_SSFR([config['fname_secondary_zen_cal']], dark_corr_mode='mean')
                spectra_zen_si_l = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 0]-intTime)<0.00001, :, 0], axis=0)
                ssfr = CU_SSFR([config['fname_secondary_zen_dark']], dark_corr_mode='mean')
                spectra_zen_si_d = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 0]-intTime)<0.00001, :, 0], axis=0)
                spectra_zen_si = spectra_zen_si_l - spectra_zen_si_d
                spectra_zen_si[spectra_zen_si<=0.0] = np.nan
                self.secondary_response_zen_si[intTime] = spectra_zen_si / intTime / self.field_lamp_zen_si[intTime]
            except:
                self.secondary_response_zen_si[intTime] = np.repeat(np.nan, self.chanNum)

        self.secondary_response_zen_in = {}
        for intTime in config['int_time_secondary_zen_in']:
            try:
                ssfr = CU_SSFR([config['fname_secondary_zen_cal']], dark_corr_mode='mean')
                spectra_zen_in_l = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 1]-intTime)<0.00001, :, 1], axis=0)
                ssfr = CU_SSFR([config['fname_secondary_zen_dark']], dark_corr_mode='mean')
                spectra_zen_in_d = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 1]-intTime)<0.00001, :, 1], axis=0)
                spectra_zen_in = spectra_zen_in_l - spectra_zen_in_d
                spectra_zen_in[spectra_zen_in<=0.0] = np.nan
                self.secondary_response_zen_in[intTime] = spectra_zen_in / intTime / self.field_lamp_zen_in[intTime]
            except:
                self.secondary_response_zen_in[intTime] = np.repeat(np.nan, self.chanNum)

        # for nadir
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.secondary_response_nad_si = {}
        for intTime in config['int_time_secondary_nad_si']:
            try:
                ssfr = CU_SSFR([config['fname_secondary_nad_cal']], dark_corr_mode='mean')
                spectra_nad_si_l = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 2]-intTime)<0.00001, :, 2], axis=0)
                ssfr = CU_SSFR([config['fname_secondary_nad_dark']], dark_corr_mode='mean')
                spectra_nad_si_d = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 2]-intTime)<0.00001, :, 2], axis=0)
                spectra_nad_si = spectra_nad_si_l - spectra_nad_si_d
                spectra_nad_si[spectra_nad_si<=0.0] = np.nan
                self.secondary_response_nad_si[intTime] = spectra_nad_si / intTime / self.field_lamp_nad_si[intTime]
            except:
                self.secondary_response_nad_si[intTime] = np.repeat(np.nan, self.chanNum)

        self.secondary_response_nad_in = {}
        for intTime in config['int_time_secondary_nad_in']:
            try:
                ssfr = CU_SSFR([config['fname_secondary_nad_cal']], dark_corr_mode='mean')
                spectra_nad_in_l = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 3]-intTime)<0.00001, :, 3], axis=0)
                ssfr = CU_SSFR([config['fname_secondary_nad_dark']], dark_corr_mode='mean')
                spectra_nad_in_d = np.mean(ssfr.spectra_dark_corr[np.abs(ssfr.int_time[:, 3]-intTime)<0.00001, :, 3], axis=0)
                spectra_nad_in = spectra_nad_in_l - spectra_nad_in_d
                spectra_nad_in[spectra_nad_in<=0.0] = np.nan
                self.secondary_response_nad_in[intTime] = spectra_nad_in / intTime / self.field_lamp_nad_in[intTime]
            except:
                self.secondary_response_nad_in[intTime] = np.repeat(np.nan, self.chanNum)
        # ---------------------------------------------------------------------------


    def CAL_ANGULAR_RESPONSE(self, config):

        print('under development')







def QUICKLOOK_TIME_SERIES(ssfr, wavelengths, tag='nadir'):

    tag = tag.lower()
    rad = ssfr.whichRadiation[tag]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(111)

    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(wavelengths)))

    for i, wavelength in enumerate(wavelengths):
        if tag == 'nadir':
            index = np.argmin(np.abs(ssfr.wvl_nad-wavelength))
            ax1.scatter(ssfr.tmhr, ssfr.spectra_nad[:, index], c=colors[i, ...], s=3, label='%.2f nm' % ssfr.wvl_nad[index])
        elif tag == 'zenith':
            index = np.argmin(np.abs(ssfr.wvl_zen-wavelength))
            ax1.scatter(ssfr.tmhr, ssfr.spectra_zen[:, index], c=colors[i, ...], s=3, label='%.2f nm' % ssfr.wvl_nad[index])

    ax1.set_ylim(bottom=0.0)
    ax1.set_title('%s %s Time Series' % (tag.title(), ssfr.whichRadiation[tag].title()))
    ax1.set_xlabel('Time [hour]')
    if rad == 'radiance':
        ax1.set_ylabel('Radiance [$\mathrm{W m^{-2} nm^{-1} sr^{-1}}$]')
    elif rad == 'irradiance':
        ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
    ax1.legend(loc='upper left', fontsize=16, framealpha=0.4, scatterpoints=3, markerscale=3)
    plt.savefig('time_series_%s.png' % tag)
    plt.show()
    # ---------------------------------------------------------------------







def QUICKLOOK_SPECTRA(ssfr, tmhrRange, tag='nadir'):

    tag = tag.lower()
    rad = ssfr.whichRadiation[tag]

    indices = np.where((ssfr.tmhr>=tmhrRange[0]) & (ssfr.tmhr<=tmhrRange[1]))[0]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(111)

    colors = plt.cm.jet(np.linspace(0.0, 1.0, indices.size))

    for i, index in enumerate(indices):
        if tag == 'nadir':
            ax1.scatter(ssfr.wvl_nad, ssfr.spectra_nad[index, :], c=colors[i, ...], s=2)
        elif tag == 'zenith':
            ax1.scatter(ssfr.wvl_zen, ssfr.spectra_zen[index, :], c=colors[i, ...], s=2)

    ax1.set_ylim(bottom=0.0)
    ax1.set_title('%s %s Spectra [%.2f, %.2f]' % (tag.title(), ssfr.whichRadiation[tag].title(), tmhrRange[0], tmhrRange[1]))
    ax1.set_xlabel('Wavelength [nm]')
    if rad == 'radiance':
        ax1.set_ylabel('Radiance [$\mathrm{W m^{-2} nm^{-1} sr^{-1}}$]')
    elif rad == 'irradiance':
        ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
    plt.savefig('spectra_%s.png' % tag)
    plt.show()
    # ---------------------------------------------------------------------







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






def PLOT_PRIMARY_RESPONSE_20180711():

    from ssfr_config import config_20180711_a1, config_20180711_a2, config_20180711_a3, config_20180711_b1, config_20180711_b2, config_20180711_b3

    markers     = ['D', '*']
    markersizes = [10, 4]
    linestyles = ['-', '--']
    colors     = ['red', 'blue', 'green']
    linewidths = [1.0, 1.0]
    alphas     = [1.0, 1.0]

    cal_a1 = CALIBRATION_CU_SSFR(config_20180711_a1)
    cal_a2 = CALIBRATION_CU_SSFR(config_20180711_a2)
    cal_a3 = CALIBRATION_CU_SSFR(config_20180711_a3)

    cals_a = [cal_a1, cal_a2, cal_a3]

    cal_b1 = CALIBRATION_CU_SSFR(config_20180711_b1)
    cal_b2 = CALIBRATION_CU_SSFR(config_20180711_b2)
    cal_b3 = CALIBRATION_CU_SSFR(config_20180711_b3)

    cals_b = [cal_b1, cal_b2, cal_b3]

    cals = [cals_a, cals_b]
    labels  = [['SSIM1 S45-90/I250-375', 'SSIM1 S45-45/I250-250', 'SSIM1 S90-90/I375-375'], ['10862 S45-90/I250-375', '10862 S45-45/I250-250', '10862 S90-90/I375-375']]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(111)
    for i, cals0 in enumerate(cals):
        for j, cal0 in enumerate(cals0):
            label0 = labels[i][j]

            # if 'SSIM1' in label0:
            intTimes_si = list(cal0.primary_response_zen_si.keys())
            intTimes_in = list(cal0.primary_response_zen_in.keys())

            for k in range(len(intTimes_si)):

                label = '%s (S%dI%d)' % (label0, intTimes_si[k], intTimes_in[k])

                if k==0:
                    ax1.plot(cal0.wvl_zen_si, cal0.primary_response_zen_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                    ax1.plot(cal0.wvl_zen_in, cal0.primary_response_zen_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                if k==1:
                    ax1.plot(cal0.wvl_zen_si, cal0.primary_response_zen_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')
                    ax1.plot(cal0.wvl_zen_in, cal0.primary_response_zen_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')


    ax1.legend(loc='upper right', fontsize=14, framealpha=0.4)

    ax1.set_title('Primary Response (Zenith 20180711)')
    ax1.set_xlim((250, 2250))
    ax1.set_ylim((0, 600))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Primary Response')
    plt.savefig('pri_resp_20180711.png', bbox_inches='tight')
    plt.show()
    # ---------------------------------------------------------------------






def PLOT_TRANSFER_20180711():

    from ssfr_config import config_20180711_a1, config_20180711_a2, config_20180711_a3, config_20180711_b1, config_20180711_b2, config_20180711_b3

    markers     = ['D', '*']
    markersizes = [10, 4]
    linestyles = ['-', '--']
    colors     = ['red', 'blue', 'green']
    linewidths = [1.0, 1.0]
    alphas     = [1.0, 1.0]

    cal_a1 = CALIBRATION_CU_SSFR(config_20180711_a1)
    cal_a2 = CALIBRATION_CU_SSFR(config_20180711_a2)
    cal_a3 = CALIBRATION_CU_SSFR(config_20180711_a3)

    cals_a = [cal_a1, cal_a2, cal_a3]

    cal_b1 = CALIBRATION_CU_SSFR(config_20180711_b1)
    cal_b2 = CALIBRATION_CU_SSFR(config_20180711_b2)
    cal_b3 = CALIBRATION_CU_SSFR(config_20180711_b3)

    cals_b = [cal_b1, cal_b2, cal_b3]

    cals = [cals_a, cals_b]
    labels  = [['SSIM1 S45-90/I250-375', 'SSIM1 S45-45/I250-250', 'SSIM1 S90-90/I375-375'], ['10862 S45-90/I250-375', '10862 S45-45/I250-250', '10862 S90-90/I375-375']]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(111)
    for i, cals0 in enumerate(cals):
        for j, cal0 in enumerate(cals0):
            label0 = labels[i][j]

            # if 'SSIM1' in label0:
            intTimes_si = list(cal0.field_lamp_zen_si.keys())
            intTimes_in = list(cal0.field_lamp_zen_in.keys())

            for k in range(len(intTimes_si)):

                label = '%s (S%dI%d)' % (label0, intTimes_si[k], intTimes_in[k])

                if k==0:
                    ax1.plot(cal0.wvl_zen_si, cal0.field_lamp_zen_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                    ax1.plot(cal0.wvl_zen_in, cal0.field_lamp_zen_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                if k==1:
                    ax1.plot(cal0.wvl_zen_si, cal0.field_lamp_zen_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')
                    ax1.plot(cal0.wvl_zen_in, cal0.field_lamp_zen_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')


    ax1.legend(loc='upper right', fontsize=14, framealpha=0.4)

    ax1.set_title('Field Calibrator (Zenith 20180711)')
    ax1.set_xlim((250, 2250))
    ax1.set_ylim((0, 0.4))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
    plt.savefig('transfer_20180711.png', bbox_inches='tight')
    plt.show()
    # ---------------------------------------------------------------------






def PLOT_PRIMARY_RESPONSE_20180712():

    from ssfr_config import config_20180712_a1, config_20180712_a2, config_20180712_b1

    markers     = ['D', '*']
    markersizes = [10, 4]
    linestyles = ['-', '--']
    colors     = ['red', 'blue', 'green']
    linewidths = [1.0, 1.0]
    alphas     = [1.0, 1.0]

    cal_a1 = CALIBRATION_CU_SSFR(config_20180712_a1)
    cal_a2 = CALIBRATION_CU_SSFR(config_20180712_a2)
    cal_a3 = CALIBRATION_CU_SSFR(config_20180712_b1)

    cals = [[cal_a1, cal_a2, cal_a3]]
    labels  = [['2008-04 S45-90/I250-375', '2008-04 S90-150/I250-375', 'L2008-2 S45-90/I250-375']]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(111)
    for i, cals0 in enumerate(cals):
        for j, cal0 in enumerate(cals0):
            label0 = labels[i][j]

            intTimes_si = list(cal0.primary_response_nad_si.keys())
            intTimes_in = list(cal0.primary_response_nad_in.keys())

            for k in range(len(intTimes_si)):

                label = '%s (S%dI%d)' % (label0, intTimes_si[k], intTimes_in[k])

                if k==0:
                    ax1.plot(cal0.wvl_nad_si, cal0.primary_response_nad_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                    ax1.plot(cal0.wvl_nad_in, cal0.primary_response_nad_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                if k==1:
                    ax1.plot(cal0.wvl_nad_si, cal0.primary_response_nad_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')
                    ax1.plot(cal0.wvl_nad_in, cal0.primary_response_nad_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')


    ax1.legend(loc='upper right', fontsize=14, framealpha=0.4)

    ax1.set_title('Primary Response (Nadir 20180712)')
    ax1.set_xlim((250, 2250))
    ax1.set_ylim((0, 600))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Primary Response')
    plt.savefig('pri_resp_20180712.png', bbox_inches='tight')
    plt.show()
    # ---------------------------------------------------------------------






def PLOT_TRANSFER_20180712():

    from ssfr_config import config_20180712_a1, config_20180712_a2, config_20180712_b1

    markers     = ['D', '*']
    markersizes = [10, 4]
    linestyles = ['-', '--']
    colors     = ['red', 'blue', 'green']
    linewidths = [1.0, 1.0]
    alphas     = [1.0, 1.0]

    cal_a1 = CALIBRATION_CU_SSFR(config_20180712_a1)
    cal_a2 = CALIBRATION_CU_SSFR(config_20180712_a2)
    cal_a3 = CALIBRATION_CU_SSFR(config_20180712_b1)

    cals = [[cal_a1, cal_a2, cal_a3]]
    labels  = [['2008-04 S45-90/I250-375', '2008-04 S90-150/I250-375', 'L2008-2 S45-90/I250-375']]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(111)
    for i, cals0 in enumerate(cals):
        for j, cal0 in enumerate(cals0):
            label0 = labels[i][j]

            intTimes_si = list(cal0.field_lamp_nad_si.keys())
            intTimes_in = list(cal0.field_lamp_nad_in.keys())

            for k in range(len(intTimes_si)):

                label = '%s (S%dI%d)' % (label0, intTimes_si[k], intTimes_in[k])

                if k==0:
                    ax1.plot(cal0.wvl_nad_si, cal0.field_lamp_nad_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                    ax1.plot(cal0.wvl_nad_in, cal0.field_lamp_nad_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                if k==1:
                    ax1.plot(cal0.wvl_nad_si, cal0.field_lamp_nad_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')
                    ax1.plot(cal0.wvl_nad_in, cal0.field_lamp_nad_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')


    ax1.legend(loc='upper right', fontsize=14, framealpha=0.4)

    ax1.set_title('Field Calibrator (Nadir 20180712)')
    ax1.set_xlim((250, 2250))
    ax1.set_ylim((0, 0.4))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
    plt.savefig('transfer_20180712.png', bbox_inches='tight')
    plt.show()
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



    # PLOT_PRIMARY_RESPONSE_20180711()
    # PLOT_TRANSFER_20180711()

    # PLOT_PRIMARY_RESPONSE_20180712()
    PLOT_TRANSFER_20180712()
    exit()









    date = datetime.datetime(2018, 5, 3)
    # read in data
    # ==============================================================
    fdir   = '/Users/hoch4240/Desktop/SSFR/Alvin/%s/data' % date.strftime('%Y%m%d')
    # ==============================================================
    fnames = sorted(glob.glob('%s/*.SKS' % fdir))
    ssfr   = CU_SSFR(fnames)
    # ==============================================================
    whichRadiation = {'zenith':'irradiance', 'nadir':'irradiance'}
    # ==============================================================
    ssfr.COUNT2RADIATION(cal, whichRadiation=whichRadiation)

    f = h5py.File('%s_Alvin.h5' % date.strftime('%Y%m%d'), 'w')
    f['spectra_zen'] = ssfr.spectra_zen
    f['spectra_nad'] = ssfr.spectra_nad
    f['wvl_zen'] = ssfr.wvl_zen
    f['wvl_nad'] = ssfr.wvl_nad
    f['tmhr']    = ssfr.tmhr
    f['temp']    = ssfr.temp
    f.close()
    exit()

    wavelengths = [600.0, 1260.0]
    QUICKLOOK_TIME_SERIES(ssfr, wavelengths, tag='nadir')

    tmhrRange = [4.0, 4.1]
    QUICKLOOK_SPECTRA(ssfr, tmhrRange, tag='nadir')
