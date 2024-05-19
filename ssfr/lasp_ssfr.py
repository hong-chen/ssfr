import os
import sys
import glob
import struct
import warnings
import numpy as np
import datetime

import ssfr




__all__ = [
        'get_ssfr_wvl',
        'read_ssfr_raw',
        'read_ssfr',
        ]




def get_ssfr_wvl(
        which_ssfr,
        Nchan=256,
        ):

    which_ssfr = which_ssfr.lower()
    if which_ssfr not in ['lasp|ssfr-a', 'lasp|ssfr-b']:
        msg = 'Error [get_ssfr_wvl]: <which_ssfr> can only be <lasp|ssfr-a> or <lasp|ssfr-b>.'
        raise OSError(msg)

    wvls = {
            'zen|si': ssfr.cal.cal_wvl(ssfr.cal.get_wvl_coef('%s|zen|si' % which_ssfr), Nchan=Nchan),
            'zen|in': ssfr.cal.cal_wvl(ssfr.cal.get_wvl_coef('%s|zen|in' % which_ssfr), Nchan=Nchan),
            'nad|si': ssfr.cal.cal_wvl(ssfr.cal.get_wvl_coef('%s|nad|si' % which_ssfr), Nchan=Nchan),
            'nad|in': ssfr.cal.cal_wvl(ssfr.cal.get_wvl_coef('%s|nad|in' % which_ssfr), Nchan=Nchan),
            }

    return wvls




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
        count_raw
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
    count_raw  (numpy array)[N/A]  : counts of Silicon and InGaAs for both zenith and nadir
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

    count_raw = np.zeros((iterN, 256, 4), dtype=np.float64) # raw counts (Ntime, Nchan, Nspec)
    shutter    = np.zeros(iterN          , dtype=np.int32  ) # shutter status (1:closed, 0:open)
    int_time   = np.zeros((iterN, 4)     , dtype=np.float64) # integration time [ms]
    temp       = np.zeros((iterN, 11)    , dtype=np.float64) # temperature
    qual_flag  = np.ones(iterN           , dtype=np.int32)   # quality flag (1:good, 0:bad)
    jday_ARINC = np.zeros(iterN          , dtype=np.float64) # ARINC time (aircraft time, e.g., from P-3)
    jday_cRIO  = np.zeros(iterN          , dtype=np.float64) # cRIO time (SSFR computer time)

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
        # ----------------          below repeat for zen_si, nad_si, zen_in, nad_in          ----------------------------
        # l2Bl: int_time[l], shutter[B], EOS[B], null[l]
        # 257h: raw counts(257)
        # ---------------------------------------------------------------------------------------------------------------
        data     = struct.unpack('<d9ld9ll11dl2Bl257hl2Bl257hl2Bl257hlBBl257h', dataRec)

        dataHead = data[:32]
        dataSpec = np.transpose(np.array(data[32:]).reshape((4, 261)))[:, [0, 2, 1, 3]]
        # [0, 2, 1, 3]: change order from 'zen_si, nad_si, zen_in, nad_in' to 'zen_si, zen_in, nad_si, nad_in'
        # transpose: change shape from (4, 261) to (261, 4)

        shutter_logic = (np.unique(dataSpec[1, :]).size != 1)
        eos_logic     = any(dataSpec[2, :] != 1)
        null_logic    = any(dataSpec[3, :] != 257)
        order_logic   = not np.array_equal(dataSpec[4, :], np.array([0, 2, 1, 3]))

        if any([shutter_logic, eos_logic, null_logic, order_logic]):
            qual_flag[i] = 0

        count_raw[i, :, :]  = dataSpec[5:, :]
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
           'count_raw': count_raw,
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

    ID = 'CU LASP SSFR'
    Nchan = 256
    Ntemp = 11
    Nspec = 4

    def __init__(
            self,
            fnames,
            Ndata=2000,
            which_time='arinc',
            process=True,
            dark_corr_mode='interp',
            dark_extend=2,
            light_extend=2,
            which_ssfr=None,
            wvl_s=350.0,
            wvl_e=2200.0,
            wvl_j=950.0,
            verbose=False,
            ):

        '''
        Description:
        fnames      : list of SSFR files to read
        Ndata=      : pre-defined number of data records (any number larger than the "number of data records per file" will work); default=600
        which_time=  : "ARINC" or "cRIO"; default='arinc'
        process=    : whether or not process data, e.g., dark correction; default=True
        dark_corr_mode=: dark correction mode, can be 'interp' or 'mean'; default='interp'
        verbose=    : verbose tag; default=False
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
        #   self.data_raw['info']['ssfr_tag']
        #   self.data_raw['info']['fnames']
        #   self.data_raw['info']['comment']
        #   self.data_raw['info']['Ndata']
        #   self.data_raw['count_raw']
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

        self.data_raw['info'] = {}
        self.data_raw['info']['ssfr_tag'] = '%s' % (self.ID)
        self.data_raw['info']['fnames']   = fnames

        Nx         = Ndata * len(fnames)
        comment    = []
        count_raw  = np.zeros((Nx, self.Nchan, self.Nspec), dtype=np.float64)
        shutter    = np.zeros(Nx                          , dtype=np.int32  )
        int_time   = np.zeros((Nx, self.Nspec)            , dtype=np.float64)
        temp       = np.zeros((Nx, self.Ntemp)            , dtype=np.float64)
        qual_flag  = np.zeros(Nx                          , dtype=np.int32)
        jday_ARINC = np.zeros(Nx                          , dtype=np.float64)
        jday_cRIO  = np.zeros(Nx                          , dtype=np.float64)

        Nstart = 0
        for fname in fnames:

            data0 = read_ssfr_raw(fname, verbose=verbose)

            Nend = data0['iterN'] + Nstart

            comment.append(data0['comment'])
            count_raw[Nstart:Nend, ...]    = data0['count_raw']
            shutter[Nstart:Nend, ...]    = data0['shutter']
            int_time[Nstart:Nend, ...]   = data0['int_time']
            temp[Nstart:Nend, ...]       = data0['temp']
            jday_ARINC[Nstart:Nend, ...] = data0['jday_ARINC']
            jday_cRIO[Nstart:Nend, ...]  = data0['jday_cRIO']
            qual_flag[Nstart:Nend, ...]  = data0['qual_flag']

            Nstart = Nend

        self.data_raw['count_raw']    = count_raw[:Nend, ...]
        self.data_raw['shutter']    = shutter[:Nend, ...]
        self.data_raw['int_time']   = int_time[:Nend, ...]
        self.data_raw['temp']       = temp[:Nend, ...]
        self.data_raw['jday_a'] = jday_ARINC[:Nend, ...]
        self.data_raw['jday_c']  = jday_cRIO[:Nend, ...]
        self.data_raw['qual_flag']  = qual_flag[:Nend, ...]
        self.data_raw['info']['comment'] = comment
        self.data_raw['info']['Ndata'] = self.data_raw['shutter'].size

        if which_time.lower() == 'arinc':
            self.data_raw['jday'] = self.data_raw['jday_a'].copy()
        elif which_time.lower() == 'crio':
            self.data_raw['jday'] = self.data_raw['jday_c'].copy()
        self.data_raw['tmhr'] = (self.data_raw['jday'] - int(self.data_raw['jday'][0])) * 24.0
        #\----------------------------------------------------------------------------/#

        # process data
        #/----------------------------------------------------------------------------\#
        if process:
            self.dset_check()
            self.dark_corr(dark_corr_mode=dark_corr_mode, dark_extend=dark_extend, light_extend=light_extend)
            if which_ssfr is not None:
                self.wvl_join(which_ssfr, wvl_start=wvl_s, wvl_end=wvl_e, wvl_join=wvl_j)
        #\----------------------------------------------------------------------------/#

    def dset_check(
            self,
            ):

        self.data_raw['dset_num'] = np.zeros(self.data_raw['jday'].size, dtype=np.int32)

        int_time_ = np.unique(self.data_raw['int_time'], axis=0)
        self.Ndset, _ = int_time_.shape
        print('\nMessage [read_ssfr]:\nTotal of %d sets of integration times were found:' % self.Ndset)

        #/----------------------------------------------------------------------------\#
        for idset in range(self.Ndset):

            # seperate data by integration times
            #/----------------------------------------------------------------------------\#
            logic = (self.data_raw['int_time'][:, 0] == int_time_[idset, 0]) & \
                    (self.data_raw['int_time'][:, 1] == int_time_[idset, 1]) & \
                    (self.data_raw['int_time'][:, 2] == int_time_[idset, 2]) & \
                    (self.data_raw['int_time'][:, 3] == int_time_[idset, 3])

            self.data_raw['dset_num'][logic] = idset
            #\----------------------------------------------------------------------------/#

            dset_name = 'dset%d' % idset
            print('  %s (%5d samples): zen|si=%3dms, zen|in=%3dms, nad|si=%3dms, nad|in=%3dms' % (dset_name, logic.sum(), *int_time_[idset]))
        #\----------------------------------------------------------------------------/#

    def dark_corr(
            self,
            dark_corr_mode='interp',
            dark_extend=2,
            light_extend=2,
            fill_value=np.nan,
            ):

        spec_info = {
                0: 'zen|si',
                1: 'zen|in',
                2: 'nad|si',
                3: 'nad|in',
                }

        shutter_dark_corr = np.zeros_like(self.data_raw['shutter'])
        shutter_dark_corr[...] = -2
        count_dark_corr  = np.zeros_like(self.data_raw['count_raw'])
        count_dark_corr[...] = fill_value

        fail_list = []

        for ispec in range(self.Nspec):
            int_time = np.unique(self.data_raw['int_time'][:, ispec])
            for int_time0 in int_time:
                logic = (self.data_raw['int_time'][:, ispec]==int_time0)
                logic_light = (logic & ((self.data_raw['shutter'][:]==0)))
                logic_dark  = (logic & ((self.data_raw['shutter'][:]==1)))

                if logic_dark.sum() > 0:

                    shutter_dark_corr[logic], count_dark_corr[logic, :, ispec] = \
                            ssfr.corr.dark_corr(
                            self.data_raw['tmhr'][logic],
                            self.data_raw['shutter'][logic],
                            self.data_raw['count_raw'][logic, :, ispec],
                            mode=dark_corr_mode,
                            darkExtend=dark_extend,
                            lightExtend=light_extend,
                            fillValue=fill_value
                            )

                else:

                    msg = '\nWarning [read_ssfr]: cannot find corresponding darks for %s=%3dms at\n    %s' % (spec_info[ispec], int_time0, np.where(logic_light)[0])
                    warnings.warn(msg)
                    fail_list.append([ispec, int_time0, logic_light])

        # a fallback process when no darks are found for corresponding integration times
        #/----------------------------------------------------------------------------\#
        for item in fail_list:

            ispec, int_time0, logic_light = item

            logic = (shutter_dark_corr == 1)
            count_base = -2**15
            darks = (self.data_raw['count_raw'][logic, :, ispec]-count_base) / (self.data_raw['int_time'][logic, np.newaxis, ispec]) * int_time0 + count_base
            dark_mean = np.mean(darks, axis=0)

            shutter_dark_corr[logic_light] = 0
            count_dark_corr[logic_light, :, ispec] = self.data_raw['count_raw'][logic_light, :, ispec] - dark_mean[np.newaxis, :]
            msg = '\nWarning [read_ssfr]: using average darks for %s=%3dms (no darks) at\n    %s' % (spec_info[ispec], int_time0, np.where(logic_light)[0])
            warnings.warn(msg)
        #\----------------------------------------------------------------------------/#

        self.data_raw['shutter_dark-corr'] = shutter_dark_corr
        self.data_raw['count_dark-corr'] = count_dark_corr
        self.data_raw['count_per_ms_dark-corr'] = count_dark_corr / self.data_raw['int_time'][:, np.newaxis, :]

        # dark_per_ms = count_per_ms[self.data_raw['shutter']==]

        # dark correction (light minus dark)
        #/----------------------------------------------------------------------------\#
        # count_per_ms = (self.data_raw['count_raw']+2**15) / (self.data_raw['int_time'][:, np.newaxis, :])
        # count_per_ms_dark_corr = count_per_ms.copy()
        # count_per_ms_dark_corr[...] = fill_value
        # for ip in range(self.Nspec):
        #     shutter_dark_corr, count_per_ms_dark_corr[:, :, ip] = ssfr.corr.dark_corr(self.data_raw['tmhr'], self.data_raw['shutter'], count_per_ms[:, :, ip], mode=dark_corr_mode, darkExtend=dark_extend, lightExtend=light_extend, fillValue=fill_value)

        # self.data_raw['shutter_dark-corr'] = shutter_dark_corr
        # self.data_raw['count_per_ms_dark-corr'] = count_per_ms_dark_corr
        #\----------------------------------------------------------------------------/#

        # self.dset0['shutter_dark-corr'], where -1 is data excluded during dark correction
        # self.dset0['count_dark-corr']
        #/----------------------------------------------------------------------------\#
        # for idset in range(self.Ndset):

        #     dset = getattr(self, 'dset%d' %idset)

        #     # dark correction (light minus dark)
        #     #/----------------------------------------------------------------------------\#
        #     count_dark_corr = dset['count_raw'].copy()
        #     count_dark_corr[...] = fill_value
        #     for ip in range(self.Nspec):
        #         shutter_dark_corr, count_dark_corr[:, :, ip] = ssfr.corr.dark_corr(dset['tmhr'], dset['shutter'], dset['count_raw'][:, :, ip], mode=dark_corr_mode, darkExtend=dark_extend, lightExtend=light_extend, fillValue=fill_value)

        #     dset['shutter_dark-corr'] = shutter_dark_corr
        #     dset['count_dark-corr'] = count_dark_corr
        #     #\----------------------------------------------------------------------------/#
        #\----------------------------------------------------------------------------/#

    def wvl_join(
            self,
            which_ssfr,
            wvl_start=350.0,
            wvl_end=2200.0,
            wvl_join=950.0,
            ):

        wvls = get_ssfr_wvl(which_ssfr)
        self.data_raw['wvl_zen_si'] = wvls['zen|si']
        self.data_raw['wvl_zen_in'] = wvls['zen|in']
        self.data_raw['wvl_nad_si'] = wvls['nad|si']
        self.data_raw['wvl_nad_in'] = wvls['nad|in']

        # zenith wavelength
        #/----------------------------------------------------------------------------\#
        logic_zen_si = (wvls['zen|si'] >= wvl_start) & (wvls['zen|si'] <= wvl_join)
        logic_zen_in = (wvls['zen|in'] >  wvl_join)  & (wvls['zen|in'] <= wvl_end)

        wvl_zen = np.concatenate((wvls['zen|si'][logic_zen_si], wvls['zen|in'][logic_zen_in]))

        indices_sort_zen = np.argsort(wvl_zen)
        wvl_zen = wvl_zen[indices_sort_zen]
        #\----------------------------------------------------------------------------/#

        # nadir wavelength
        #/----------------------------------------------------------------------------\#
        logic_nad_si = (wvls['nad|si'] >= wvl_start) & (wvls['nad|si'] <= wvl_join)
        logic_nad_in = (wvls['nad|in'] >  wvl_join)  & (wvls['nad|in'] <= wvl_end)

        wvl_nad = np.concatenate((wvls['nad|si'][logic_nad_si], wvls['nad|in'][logic_nad_in]))

        indices_sort_nad = np.argsort(wvl_nad)
        wvl_nad = wvl_nad[indices_sort_nad]
        #\----------------------------------------------------------------------------/#

        # processing data (unit counts: [counts/ms])
        #/----------------------------------------------------------------------------\#
        counts_zen = np.hstack((self.data_raw['count_per_ms_dark-corr'][:, logic_zen_si, 0], self.data_raw['count_per_ms_dark-corr'][:, logic_zen_in, 1]))
        counts_nad = np.hstack((self.data_raw['count_per_ms_dark-corr'][:, logic_nad_si, 2], self.data_raw['count_per_ms_dark-corr'][:, logic_nad_in, 3]))

        counts_zen = counts_zen[:, indices_sort_zen]
        counts_nad = counts_nad[:, indices_sort_nad]

        self.data_corr = {}
        self.data_corr['wvl_zen'] = wvl_zen
        self.data_corr['cnt_zen'] = counts_zen
        self.data_corr['wvl_nad'] = wvl_nad
        self.data_corr['cnt_nad'] = counts_nad
        #\----------------------------------------------------------------------------/#




if __name__ == '__main__':

    pass
