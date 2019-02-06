import os
import glob
import h5py
import struct
import numpy as np
import datetime
from scipy import stats



def IF_FILE_EXISTS(fname, exitTag=True):

    if not os.path.exists(fname):
        if exitTag=True:
            exit("Error [IF_FILE_EXISTS]: cannot find '{fname}'".format(fname=fname))
        else
            print("Alert [IF_FILE_EXISTS]: cannot find '{fname}'".format(fname=fname))



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

    IF_FILE_EXISTS(fname, exitTag=True)

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

        if not isinstance(fnames, list):
            exit("Error [CU_SSFR]: input variable 'fnames' should be a Python list.")

        if len(fnames) == 0:
            exit("Error [CU_SSFR]: input variable 'fnames' is empty.")

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



if __name__ == '__main__':

    pass
