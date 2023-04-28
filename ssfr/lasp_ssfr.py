import os
import sys
import glob
import struct
import warnings
import numpy as np
import datetime

import ssfr




__all__ = [
        'get_ssfr_wavelength',
        'read_ssfr_raw',
        'read_ssfr',
        ]




def get_ssfr_wavelength(chanNum=256):

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

def read_ssfr_raw(
        fname,
        headLen=148,
        dataLen=2276,
        verbose=False
        ):

    '''
    Reader code for Solar Spectral Flux Radiometer at LASP of University of Colorado Bouder (LASP-SSFR).

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
    data0 = read_ssfr_raw(fname, verbose=False)

    data0 contains the following variables:

    comment  (str)        [N/A]    : comment in header
    spectra  (numpy array)[N/A]    : counts of Silicon and InGaAs for both zenith and nadir
    shutter  (numpy array)[N/A]    : shutter status (1:closed(dark), 0:open(light))
    int_time (numpy array)[ms]     : integration time of Silicon and InGaAs for both zenith and nadir
    temp (numpy array)    [Celsius]: temperature variables
    jday_ARINC (numpy array)[day]  : julian days (w.r.t 0001-01-01) of aircraft nagivation system
    jday_cRIO(numpy array)[day]    : julian days (w.r.t 0001-01-01) of SSFR Inertial Navigation System (INS)
    qual_flag(numpy array)[N/A]    : quality flag(1:good, 0:bad)
    iterN (numpy array)   [N/A]    : number of data record
    '''

    ssfr.util.if_file_exists(fname, exitTag=True)

    fileSize = os.path.getsize(fname)
    if fileSize > headLen:
        iterN   = (fileSize-headLen) // dataLen
        residual = (fileSize-headLen) %  dataLen
        if residual != 0:
            msg = '\nWarning [read_ssfr_raw]: <%s> contains unreadable data, omit the last data record...' % fname
            warnings.warn(msg)
    else:
        msg = '\nError [read_ssfr_raw]: <%s> has invalid file size.' % fname
        raise OSError(msg)

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
        print('#/--------------------------------------------------------------\#')
        print('Comments in <%s>...' % fname.split('/')[-1])
        print(comment)
        print('#\--------------------------------------------------------------/#')

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

    data_ = {
             'comment': comment,
             'spectra': spectra,
             'shutter': shutter,
            'int_time': int_time,
                'temp': temp,
                'jday': jday_ARINC,
          'jday_ARINC': jday_ARINC,
           'jday_cRIO': jday_cRIO,
           'qual_flag': qual_flag,
               'iterN': iterN,
            }

    return data_

class read_ssfr:

    def __init__(
            self,
            fnames,
            Ndata=600,
            whichTime='arinc',
            time_offset=0.0,
            process=True,
            dark_corr_mode='interp',
            verbose=False,
            ):

        '''
        Description:
        fnames    : list of SSFR files to read
        Ndata     : pre-defined number of data records, any number larger than the "number of data records per file" will work
        whichTime : "ARINC" or "cRIO"
        time_offset: time offset in [seconds]
        '''

        # input check
        #/----------------------------------------------------------------------------\#
        if not isinstance(fnames, list):
            msg = '\nError [read_ssfr]: Input variable <fnames> should be a Python list.'
            raise OSError(msg)

        if len(fnames) == 0:
            msg = 'Error [read_ssfr]: input variable <fnames> is empty.'
            raise OSError(msg)
        #\----------------------------------------------------------------------------/#


        # read in all the data
        # after the following process, the object will contain
        #   self.data_raw['ssfr_tag']
        #   self.data_raw['fnames']
        #   self.data_raw['comment']
        #   self.data_raw['spectra']
        #   self.data_raw['shutter']
        #   self.data_raw['int_time']
        #   self.data_raw['temp']
        #   self.data_raw['jday_ARINC']
        #   self.data_raw['jday_cRIO']
        #   self.data_raw['qual_flag']
        #   self.data_raw['jday']
        #   self.data_raw['tmhr']
        #   self.data_raw['jday_corr']
        #   self.data_raw['tmhr_corr']
        #/----------------------------------------------------------------------------\#
        self.data_raw = {}
        self.data_raw['ssfr_tag'] = 'CU LASP SSFR'
        self.data_raw['fnames']   = fnames

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

            data0 = read_ssfr_raw(fname, verbose=verbose)

            Nend = data0['iterN'] + Nstart

            comment.append(data0['comment'])
            spectra[Nstart:Nend, ...]    = data0['spectra']
            shutter[Nstart:Nend, ...]    = data0['shutter']
            int_time[Nstart:Nend, ...]   = data0['int_time']
            temp[Nstart:Nend, ...]       = data0['temp']
            jday_ARINC[Nstart:Nend, ...] = data0['jday_ARINC']
            jday_cRIO[Nstart:Nend, ...]  = data0['jday_cRIO']
            qual_flag[Nstart:Nend, ...]  = data0['qual_flag']

            Nstart = Nend

        self.data_raw['comment']    = comment
        self.data_raw['spectra']    = spectra[:Nend, ...]
        self.data_raw['shutter']    = shutter[:Nend, ...]
        self.data_raw['int_time']   = int_time[:Nend, ...]
        self.data_raw['temp']       = temp[:Nend, ...]
        self.data_raw['jday_a'] = jday_ARINC[:Nend, ...]
        self.data_raw['jday_c']  = jday_cRIO[:Nend, ...]
        self.data_raw['qual_flag']  = qual_flag[:Nend, ...]

        if whichTime.lower() == 'arinc':
            self.data_raw['jday'] = self.data_raw['jday_a'].copy()
        elif whichTime.lower() == 'crio':
            self.data_raw['jday'] = self.data_raw['jday_c'].copy()
        self.data_raw['tmhr'] = (self.data_raw['jday'] - int(self.data_raw['jday'][0])) * 24.0

        self.data_raw['jday_corr'] = self.data_raw['jday'] + float(time_offset)/86400.0
        self.data_raw['tmhr_corr'] = self.data_raw['tmhr'] + float(time_offset)/3600.0
        #\----------------------------------------------------------------------------/#


        # process data
        #/----------------------------------------------------------------------------\#
        if process:
            self.process_data(dark_corr_mode=dark_corr_mode)
        #\----------------------------------------------------------------------------/#

    def process_data(
            self,
            dark_corr_mode='interp',
            fill_value=np.nan,
            ):

        int_time_ = np.unique(self.data_raw['int_time'], axis=0)
        Nt, Np = int_time_.shape
        self.Ndata = Nt

        # split data for different pairs of integration times
        # after the following process, the object will contain
        #   self.data0, self.data1 ...
        #
        #   with the same data variables contained in self.data_raw but with additions of
        #   self.data0['int_time_info']
        #   self.data0['shutter_dark_corr'], where -1 is data excluded during dark correction
        #   self.data0['spectra_dark_corr']
        #/----------------------------------------------------------------------------\#
        for it in range(Nt):

            data = {}
            data_name = 'data%d' % it

            int_time_info = {
                    'Zenith Silicon': int_time_[it, 0],
                     'Zenith InGaAs': int_time_[it, 1],
                     'Nadir Silicon': int_time_[it, 2],
                      'Nadir InGaAs': int_time_[it, 3],
                    }

            # split data by integration times
            #/----------------------------------------------------------------------------\#
            logic = self.data_raw['int_time'][:, 0] == int_time_[it, 0]
            for vname in self.data_raw.keys():
                if vname in ['ssfr_tag', 'fnames', 'comment']:
                    data[vname] = self.data_raw[vname]
                else:
                    data[vname] = self.data_raw[vname][logic, ...]
            data['int_time_info'] = int_time_info
            #\----------------------------------------------------------------------------/#

            # dark correction (light-dark)
            # variable name: self.spectra_dark_corr
            #/----------------------------------------------------------------------------\#
            spectra_dark_corr = data['spectra'].copy()
            spectra_dark_corr[...] = fill_value
            for ip in range(Np):
                shutter_dark_corr, spectra_dark_corr[:, :, ip] = ssfr.corr.dark_corr(data['tmhr'], data['shutter'], data['spectra'][:, :, ip], mode=dark_corr_mode, fillValue=fill_value)

            data['shutter_dark_corr'] = shutter_dark_corr
            data['spectra_dark_corr'] = spectra_dark_corr
            #\----------------------------------------------------------------------------/#

            setattr(self, data_name, data)
        #\----------------------------------------------------------------------------/#

    def count_to_radiation(self, cal, wvl_zen_join=900.0, wvl_nad_join=900.0, whichRadiation={'zenith':'radiance', 'nadir':'irradiance'}, wvlRange=[350, 2100]):

        """
        Convert digital count to radiation (radiance or irradiance)
        """

        msg = 'Error [read_ssfr]: Under development ...'
        raise OSError(msg)

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




if __name__ == '__main__':

    pass
