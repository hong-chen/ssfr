import os
import glob
import h5py
import struct
import numpy as np
import datetime
from scipy.io import readsav
from scipy import stats

def READ_SKS_V2(fname, headLen=148, dataLen=2276, verbose=False):

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
    #}}}

class READ_SKS:

    def __init__(self, fnames, Ndata=600, config=None):

        if type(fnames) is not list:
            exit('Error [READ_SKS]: input variable "fnames" should be in list type.')
        Nx         = Ndata * len(fnames)
        comment    = []
        spectra    = np.zeros((Nx, 256, 4), dtype=np.float64) # spectra
        shutter    = np.zeros(Nx          , dtype=np.int32  ) # shutter status (1:closed, 0:open)
        int_time   = np.zeros((Nx, 4)     , dtype=np.float64) # integration time [ms]
        temp       = np.zeros((Nx, 11)    , dtype=np.float64) # temperature
        qual_flag  = np.zeros(Nx          , dtype=np.int32)
        jday_NSF   = np.zeros(Nx          , dtype=np.float64)
        jday_cRIO  = np.zeros(Nx          , dtype=np.float64)

        Nstart = 0
        for fname in fnames:
            comment0, spectra0, shutter0, int_time0, temp0, jday_NSF0, jday_cRIO0, qual_flag0, iterN0 = READ_SKS_V2(fname)
            comment.append(comment0)

            Nend = iterN0 + Nstart

            spectra[Nstart:Nend, ...]    = spectra0
            shutter[Nstart:Nend, ...]    = shutter0
            int_time[Nstart:Nend, ...]   = int_time0
            temp[Nstart:Nend, ...]       = temp0
            jday_NSF[Nstart:Nend, ...]   = jday_NSF0
            jday_cRIO[Nstart:Nend, ...]  = jday_cRIO0
            qual_flag[Nstart:Nend, ...]  = qual_flag0

            Nstart = Nend

        if config != None:
            self.config = config

        self.comment    = comment
        self.spectra    = spectra[:Nend, ...]
        self.shutter    = shutter[:Nend, ...]
        self.int_time   = int_time[:Nend, ...]
        self.temp       = temp[:Nend, ...]
        self.jday_NSF   = jday_NSF[:Nend, ...]
        self.jday_cRIO  = jday_cRIO[:Nend, ...]
        self.qual_flag  = qual_flag[:Nend, ...]
        self.shutter_ori= self.shutter.copy()

        self.jday = self.jday_NSF.copy()
        #self.jday = self.jday_cRIO[0] + 0.5/86400.0 * np.arange(self.jday_cRIO.size)
        self.tmhr = (self.jday - int(self.jday[0])) * 24.0
        self.tmhr_corr = self.tmhr.copy()

        self.DARK_CORR()

        #self.spectra_nlin_corr = self.spectra_dark_corr.copy()
        #fname_nlin ='/argus/home/chen/work/02_arise/06_cal/aux/20141121.sav'
        #Nsen       = 1
        #self.NLIN_CORR(fname_nlin, Nsen)
        #fname_nlin ='/argus/home/chen/work/02_arise/06_cal/aux/20141119.sav'
        #Nsen       = 3
        #self.NLIN_CORR(fname_nlin, Nsen)

    def DARK_CORR(self, mode=-1, darkExtend=2, lightExtend=2, countOffset=0, lightThr=10, darkThr=5, fillValue=10):

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
                    self.shutter[:darkLL] = fillValue  # omit the data before the first dark cycle

                lightL = darkR[i]   + lightExtend
                lightR = darkL[i+1] - lightExtend

                if lightR-lightL>lightThr and darkLR-darkLL>darkThr and darkRR-darkRL>darkThr:

                    self.shutter[darkL[i]:darkLL] = fillValue
                    self.shutter[darkLR:darkR[i]] = fillValue
                    self.shutter[darkR[i]:lightL] = fillValue
                    self.shutter[lightR:darkL[i+1]] = fillValue
                    self.shutter[darkL[i+1]:darkRL] = fillValue
                    self.shutter[darkRR:darkR[i+1]] = fillValue

                    int_dark  = np.append(self.int_time[darkLL:darkLR], self.int_time[darkRL:darkRR]).mean()
                    int_light = self.int_time[lightL:lightR].mean()

                    if np.abs(int_dark - int_light) > 0.0001:
                        self.shutter[lightL:lightR] = fillValue
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
                    self.shutter[darkL[i]:darkR[i+1]] = fillValue

            self.shutter[darkRR:] = fillValue  # omit the data after the last dark cycle

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

                self.shutter[darkL[i]:darkLL] = fillValue
                self.shutter[darkLR:darkR[i]] = fillValue
                self.shutter[darkR[i]:lightL] = fillValue
                self.shutter[lightR:darkL[i+1]] = fillValue

                int_dark  = self.int_time[darkLL:darkLR].mean()
                int_light = self.int_time[lightL:lightR].mean()
                if np.abs(int_dark - int_light) > 0.0001:
                    self.shutter[lightL:lightR] = fillValue
                    exit('Error [READ_SKS.DARK_CORR]: inconsistent integration time.')
                else:
                    for itmhr in range(darkLR, lightR):
                        for isen in range(Nsensor):
                            dark_offset0 = np.mean(self.spectra[darkLL:darkLR, :, isen], axis=0)
                            self.dark_offset[itmhr, :, isen] = dark_offset0
                    self.spectra_dark_corr[lightL:lightR,:,:] -= self.dark_offset[lightL:lightR,:,:]

        elif mode == -4:
            print('Message [DARK_CORR]: Not implemented...')

    def NLIN_CORR(self, fname_nlin, Nsen):
        #{{{
        int_time0 = np.mean(self.int_time[:, Nsen])
        f_nlin = readsav(fname_nlin)

        if abs(f_nlin.iin_-int_time0)>1.0e-5:
            exit('Error [READ_SKS]: Integration time do not match.')

        for iwvl in range(256):
            xx0   = self.spectra_nlin_corr[:,iwvl,Nsen].copy()
            xx    = np.zeros_like(xx0)
            yy    = np.zeros_like(xx0)
            self.spectra_nlin_corr[:,iwvl,Nsen] = np.nan
            logic_xx     = (xx0>-100)
            print('++++++++++++++++++++++++++++++++++++++++++++++++')
            print('range', f_nlin.mxm_[0,iwvl]*f_nlin.in_[iwvl], f_nlin.mxm_[1,iwvl]*f_nlin.in_[iwvl])
            print('good', logic_xx.sum(), xx0.size)
            xx0[logic_xx] = xx0[logic_xx]/f_nlin.in_[iwvl]

            if (f_nlin.bad_[1,iwvl]<1.0) and (f_nlin.mxm_[0,iwvl]>=1.0e-3):

                #+ data in range (0, minimum)
                yy_e = 0.0
                for ideg in range(f_nlin.gr_):
                    yy_e += f_nlin.res2_[ideg,iwvl]*f_nlin.mxm_[0,iwvl]**ideg
                slope = yy_e/f_nlin.mxm_[0,iwvl]
                logic_xx     = (xx0>-100) & (xx0<f_nlin.mxm_[0,iwvl])
                print('0-min', logic_xx.sum(), xx0.size)
                print('data', xx0[logic_xx])
                xx[xx<0]     = 0.0
                xx[logic_xx] = xx0[logic_xx]
                yy[logic_xx] = xx[logic_xx]*slope

                self.spectra_nlin_corr[logic_xx,iwvl,Nsen] = yy[logic_xx]*f_nlin.in_[iwvl]
                #-

                #+ data in range [minimum, maximum]
                logic_xx     = (xx0>=f_nlin.mxm_[0,iwvl]) & (xx0<=f_nlin.mxm_[1,iwvl])
                xx[logic_xx] = xx0[logic_xx]
                print('min-max', logic_xx.sum(), xx0.size)
                print('------------------------------------------------')
                for ideg in range(f_nlin.gr_):
                    yy[logic_xx] += f_nlin.res2_[ideg, iwvl]*xx[logic_xx]**ideg

                self.spectra_nlin_corr[logic_xx,iwvl,Nsen] = yy[logic_xx]*f_nlin.in_[iwvl]
                #-

        #}}}

def SSFR_FRAME(statements, Nchan=100, nameTag=None, testMode=False):

    # extract input arguments
    data_sks, index = statements

    # compute date and time at index "index"
    jday0 = data_sks.jday[index]
    dtime0= datetime.datetime.fromordinal(int(jday0))+datetime.timedelta(hours=data_sks.tmhr[index])
    if nameTag == None:
        nameTag = dtime0.strftime('%Y%m%d')

    # general settings for the plot
    rcParams['font.size'] = 10
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['axes.titlepad']   = 4

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('CU SSFR %s' % dtime0.strftime('%Y-%m-%d %H:%M:%S'), fontsize=24, color='black')

    gs_temp_ts = gridspec.GridSpec(2, 1)

    # +
    # plot temperatures in time series
    indices = [0, 1, 2, 3, 7]
    labels  = ['Ambient', 'InGaAs Body (Zen.)', 'InGaAs Body (Nad.)', 'Plate', 'PTC5K']
    data_temp = data_sks.temp[:, indices]
    unit = 2 * (((np.nanmax(data_temp)-np.nanmin(data_temp))/5) // 2 + 1)
    ylim_min = unit * ((np.nanmin(data_temp))//unit - 1)
    ylim_max = unit * ((np.nanmax(data_temp))//unit + 2)
    if ((ylim_max-np.nanmax(data_temp))<(np.nanmin(data_temp)-ylim_min)):
        ylim_max += unit

    ax_temp_ts_1 = fig.add_subplot(gs_temp_ts[0, :])

    patches = []

    colors = plt.cm.jet(np.linspace(0.0, 1.0, data_temp.shape[-1]))
    for ii in range(data_temp.shape[-1]):
        ax_temp_ts_1.scatter(data_sks.tmhr, data_temp[:, ii], lw=0, marker='o', facecolor=colors[ii, :], edgecolor='none', s=10, alpha=1.0, zorder=1)
        patches.append(mpatches.Patch(color=colors[ii, :] , label=labels[ii]))

    tmhr0    = (data_sks.tmhr[:-1]+data_sks.tmhr[1:])/2.0
    shutter0 = np.interp(tmhr0, data_sks.tmhr, data_sks.shutter)
    tmhr0    = np.append(tmhr0, tmhr0[-1]+1.0/3600.0)
    shutter0 = np.append(shutter0, shutter0[-1])
    tmhr0    = np.append(tmhr0[0]-1.0/3600.0, tmhr0)
    shutter0 = np.append(shutter0[0], shutter0)
    ax_temp_ts_1.fill_between(tmhr0, ylim_min, ylim_max, where=(shutter0<5.6)&(shutter0>0.9), facecolor='k', alpha=0.2, zorder=0)
    ax_temp_ts_1.fill_between(tmhr0, ylim_min, ylim_max, where=(shutter0>4.9)               , facecolor='r', alpha=0.2, zorder=0)
    ax_temp_ts_1.axvline(data_sks.tmhr[index], color='k', ls=':')

    ax_temp_ts_1.set_title('Temperature', fontsize='large', y=1.4)
    ax_temp_ts_1.set_ylim((ylim_min, ylim_max))
    ax_temp_ts_1.xaxis.set_major_locator(FixedLocator(np.array([data_sks.tmhr[index]-0.0166666666, data_sks.tmhr[index], data_sks.tmhr[index]+0.0166666666])))
    ax_temp_ts_1.set_xlim([data_sks.tmhr[index]-0.0166666666, data_sks.tmhr[index]+0.0166666667])
    ax_temp_ts_1.set_xticklabels([])
    ax_temp_ts_1.spines['bottom'].set_visible(False)
    ax_temp_ts_1.xaxis.tick_top()
    ax_temp_ts_1.tick_params(labeltop='off')

    # -

    # +
    # plot temperatures in time series
    indices = [5, 6]
    labels  = ['InGaAs TEC (Zen.)', 'InGaAs TEC (Nad.)']
    colors  = ['k', 'gray']
    data_temp = data_sks.temp[:, indices]
    unit = 2 * (((np.nanmax(data_temp)-np.nanmin(data_temp))/5) // 2 + 1)
    ylim_min = unit * ((np.nanmin(data_temp))//unit - 1)
    ylim_max = unit * ((np.nanmax(data_temp))//unit + 2)
    if ((ylim_max-np.nanmax(data_temp))<(np.nanmin(data_temp)-ylim_min)):
        ylim_max += unit

    ax_temp_ts_2 = fig.add_subplot(gs_temp_ts[1, :])
    ax_temp_ts_2.spines['top'].set_visible(False)

    for ii in range(len(labels)):
        ax_temp_ts_2.scatter(data_sks.tmhr, data_temp[:, ii], lw=0, marker='o', facecolor=colors[ii], edgecolor='none', s=10, alpha=1.0, zorder=1)
        patches.append(mpatches.Patch(color=colors[ii] , label=labels[ii]))

    tmhr0    = (data_sks.tmhr[:-1]+data_sks.tmhr[1:])/2.0
    shutter0 = np.interp(tmhr0, data_sks.tmhr, data_sks.shutter)
    tmhr0    = np.append(tmhr0, tmhr0[-1]+1.0/3600.0)
    shutter0 = np.append(shutter0, shutter0[-1])
    tmhr0    = np.append(tmhr0[0]-1.0/3600.0, tmhr0)
    shutter0 = np.append(shutter0[0], shutter0)
    ax_temp_ts_2.fill_between(tmhr0, ylim_min, ylim_max, where=(shutter0<5.6)&(shutter0>0.9), facecolor='k', alpha=0.2, zorder=0)
    ax_temp_ts_2.fill_between(tmhr0, ylim_min, ylim_max, where=(shutter0>4.9)               , facecolor='r', alpha=0.2, zorder=0)
    ax_temp_ts_2.axvline(data_sks.tmhr[index], color='k', ls=':')

    ax_temp_ts_2.set_ylim((ylim_min, ylim_max))
    ax_temp_ts_2.xaxis.set_major_locator(FixedLocator(np.array([data_sks.tmhr[index]-0.0166666666, data_sks.tmhr[index], data_sks.tmhr[index]+0.0166666666])))
    ax_temp_ts_2.set_xlim([data_sks.tmhr[index]-0.0166666666, data_sks.tmhr[index]+0.0166666667])
    ax_temp_ts_2.set_xticklabels([])
    # -
    ax_temp_ts_1.legend(handles=patches, bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=len(patches), mode="expand", borderaxespad=0., frameon=False, handletextpad=0.2, fontsize=9)
    gs_temp_ts.update(bottom=0.7, top=0.85, hspace=0.1)

    d = 0.003
    kwargs = dict(transform=ax_temp_ts_1.transAxes, color='k', clip_on=False, lw=1.0)
    ax_temp_ts_1.plot((-d, +d), (-d, +d), **kwargs)
    ax_temp_ts_1.plot((1-d, 1+d), (-d, +d), **kwargs)

    kwargs = dict(transform=ax_temp_ts_2.transAxes, color='k', clip_on=False, lw=1.0)
    ax_temp_ts_2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax_temp_ts_2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)


    gs_counts_ts = gridspec.GridSpec(3, 4)
    # +
    # plot counts for one channel ("Nchan") in time series
    logic = (data_sks.shutter==0) | (data_sks.shutter==1)

    unit = 20 * ((((data_sks.spectra_dark_corr[logic, Nchan, :]).max()-(data_sks.spectra_dark_corr[logic, Nchan, :]).min())/10) // 20 + 1)
    ylim_min = unit * (((data_sks.spectra_dark_corr[logic, Nchan, :]).min())//unit)
    ylim_max = unit * (((data_sks.spectra_dark_corr[logic, Nchan, :]).max())//unit + 1)
    if ((ylim_max-(data_sks.spectra_dark_corr[logic, Nchan, :]).max())<((data_sks.spectra_dark_corr[logic, Nchan, :]).min()-ylim_min)):
        ylim_max += unit

    ax_counts_ts = fig.add_subplot(gs_counts_ts[0, :])
    ax_counts_ts.scatter(data_sks.tmhr[logic], data_sks.spectra_dark_corr[:, Nchan, 0][logic], lw=0, marker='o', facecolor='Red'          , edgecolor='none', s=10, alpha=1.0, zorder=1)
    ax_counts_ts.scatter(data_sks.tmhr[logic], data_sks.spectra_dark_corr[:, Nchan, 1][logic], lw=0, marker='o', facecolor='Salmon'       , edgecolor='none', s=10, alpha=1.0, zorder=2)
    ax_counts_ts.scatter(data_sks.tmhr[logic], data_sks.spectra_dark_corr[:, Nchan, 2][logic], lw=0, marker='o', facecolor='Blue'         , edgecolor='none', s=10, alpha=1.0, zorder=3)
    ax_counts_ts.scatter(data_sks.tmhr[logic], data_sks.spectra_dark_corr[:, Nchan, 3][logic], lw=0, marker='o', facecolor='SkyBlue'      , edgecolor='none', s=10, alpha=1.0, zorder=4)

    tmhr0    = (data_sks.tmhr[:-1]+data_sks.tmhr[1:])/2.0
    shutter0 = np.interp(tmhr0, data_sks.tmhr, data_sks.shutter)
    tmhr0    = np.append(tmhr0, tmhr0[-1]+1.0/3600.0)
    shutter0 = np.append(shutter0, shutter0[-1])
    tmhr0    = np.append(tmhr0[0]-1.0/3600.0, tmhr0)
    shutter0 = np.append(shutter0[0], shutter0)
    ax_counts_ts.fill_between(tmhr0, ylim_min, ylim_max, where=(shutter0<5.6)&(shutter0>0.9), facecolor='k', alpha=0.2, zorder=0)
    ax_counts_ts.fill_between(tmhr0, ylim_min, ylim_max, where=(shutter0>4.9)               , facecolor='r', alpha=0.2, zorder=0)
    ax_counts_ts.axvline(data_sks.tmhr[index], color='k', ls=':')

    ax_counts_ts.set_title('Channel %d' % Nchan, fontsize='large')
    ax_counts_ts.set_ylim((ylim_min, ylim_max))
    ax_counts_ts.xaxis.set_major_locator(FixedLocator(np.array([data_sks.tmhr[index]-0.0166666666, data_sks.tmhr[index], data_sks.tmhr[index]+0.0166666666])))
    ax_counts_ts.set_xlim([data_sks.tmhr[index]-0.0166666666, data_sks.tmhr[index]+0.0166666667])
    ax_counts_ts.set_xticklabels(['UTC-60s', 'UTC(%f)' % data_sks.tmhr[index], 'UTC+60s'])
    # -

    # +
    # plot spectral dark counts
    ax_dark_zen_si = fig.add_subplot(gs_counts_ts[1, 0])
    ax_dark_zen_in = fig.add_subplot(gs_counts_ts[1, 1])
    ax_dark_nad_si = fig.add_subplot(gs_counts_ts[1, 2])
    ax_dark_nad_in = fig.add_subplot(gs_counts_ts[1, 3])

    logic = (data_sks.shutter==0) | (data_sks.shutter==1)

    ylims_min = np.zeros(4)
    ylims_max = np.zeros(4)
    for ii in range(4):
        unit = 20 * ((((data_sks.dark_offset[logic, :, ii]).max()-(data_sks.dark_offset[logic, :, ii]).min())/10) // 20 + 1)
        ylims_min[ii] = unit * (np.min(data_sks.dark_offset[logic, :, ii])//unit)
        ylims_max[ii] = unit * (np.max(data_sks.dark_offset[logic, :, ii])//unit + 1)
        if ((ylims_max[ii]-(data_sks.dark_offset[logic, :, ii]).max())<((data_sks.dark_offset[logic, :, ii]).min()-ylims_min[ii])):
            ylims_max[ii] += unit

    xx = np.arange(256) + 1
    if logic[index]:

        ax_dark_zen_si.plot(xx, data_sks.dark_offset[index, :, 0], color='k', zorder=0)
        ax_dark_zen_si.axvline(100, ls='--', color='gray', lw=1.0)
        if data_sks.shutter[index] == 1:
            ax_dark_zen_si.scatter(xx, data_sks.spectra[index, :, 0], c='b', s=3, zorder=1)
        ax_dark_zen_si.set_xlim((1, 256))
        ax_dark_zen_si.set_ylim([ylims_min[0], ylims_max[0]])
        ax_dark_zen_si.set_title('Dark of Zen. Si.', color='Red')
        ax_dark_zen_si.set_xticklabels([])
        ax_dark_zen_si.tick_params(axis='both', which='major', labelsize=6)

        ax_dark_zen_in.plot(xx, data_sks.dark_offset[index, :, 1], color='k', zorder=0)
        ax_dark_zen_in.axvline(100, ls='--', color='gray', lw=1.0)
        if data_sks.shutter[index] == 1:
            ax_dark_zen_in.scatter(xx, data_sks.spectra[index, :, 1], c='b', s=3, zorder=1)
        ax_dark_zen_in.set_xlim((1, 256))
        ax_dark_zen_in.set_ylim([ylims_min[1], ylims_max[1]])
        ax_dark_zen_in.set_title('Dark of Zen. In.', color='Salmon')
        ax_dark_zen_in.set_xticklabels([])
        ax_dark_zen_in.tick_params(axis='both', which='major', labelsize=6)

        ax_dark_nad_si.plot(xx, data_sks.dark_offset[index, :, 2], color='k', zorder=0)
        ax_dark_nad_si.axvline(100, ls='--', color='gray', lw=1.0)
        if data_sks.shutter[index] == 1:
            ax_dark_nad_si.scatter(xx, data_sks.spectra[index, :, 2], c='b', s=3, zorder=1)
        ax_dark_nad_si.set_xlim((1, 256))
        ax_dark_nad_si.set_ylim([ylims_min[2], ylims_max[2]])
        ax_dark_nad_si.set_title('Dark of Nad. Si.', color='Blue')
        ax_dark_nad_si.set_xticklabels([])
        ax_dark_nad_si.tick_params(axis='both', which='major', labelsize=6)

        ax_dark_nad_in.plot(xx, data_sks.dark_offset[index, :, 3], color='k', zorder=0)
        ax_dark_nad_in.axvline(100, ls='--', color='gray', lw=1.0)
        if data_sks.shutter[index] == 1:
            ax_dark_nad_in.scatter(xx, data_sks.spectra[index, :, 3], c='b', s=3, zorder=1)
        ax_dark_nad_in.set_xlim((1, 256))
        ax_dark_nad_in.set_ylim([ylims_min[3], ylims_max[3]])
        ax_dark_nad_in.set_title('Dark of Nad. In.', color='SkyBlue')
        ax_dark_nad_in.set_xticklabels([])
        ax_dark_nad_in.tick_params(axis='both', which='major', labelsize=6)
    else:
        ax_dark_zen_si.axis('off')
        ax_dark_zen_in.axis('off')
        ax_dark_nad_si.axis('off')
        ax_dark_nad_in.axis('off')
    # -

    # +
    # plot spectral dark-corrected counts
    ax_corr_zen_si = fig.add_subplot(gs_counts_ts[2, 0])
    ax_corr_zen_in = fig.add_subplot(gs_counts_ts[2, 1])
    ax_corr_nad_si = fig.add_subplot(gs_counts_ts[2, 2])
    ax_corr_nad_in = fig.add_subplot(gs_counts_ts[2, 3])

    logic_0 = (data_sks.shutter==0)
    logic_1 = (data_sks.shutter==1)

    ylims_min_0 = np.zeros(4)
    ylims_max_0 = np.zeros(4)
    ylims_min_1 = np.zeros(4)
    ylims_max_1 = np.zeros(4)
    for ii in range(4):
        unit_0 = 20 * ((((data_sks.spectra_dark_corr[logic_0, :, ii]).max()-(data_sks.spectra_dark_corr[logic_0, :, ii]).min())/10) // 20 + 1)
        ylims_min_0[ii] = unit_0 * (np.min(data_sks.spectra_dark_corr[logic_0, :, ii])//unit_0)
        ylims_max_0[ii] = unit_0 * (np.max(data_sks.spectra_dark_corr[logic_0, :, ii])//unit_0 + 1)
        if ((ylims_max_0[ii]-(data_sks.spectra_dark_corr[logic_0, :, ii]).max())<((data_sks.spectra_dark_corr[logic_0, :, ii]).min()-ylims_min_0[ii])):
            ylims_max_0[ii] += unit_0

        unit_1 = 20 * ((((data_sks.spectra_dark_corr[logic_1, :, ii]).max()-(data_sks.spectra_dark_corr[logic_1, :, ii]).min())/10) // 20 + 1)
        ylims_min_1[ii] = unit_1 * (np.min(data_sks.spectra_dark_corr[logic_1, :, ii])//unit_1)
        ylims_max_1[ii] = unit_1 * (np.max(data_sks.spectra_dark_corr[logic_1, :, ii])//unit_1 + 1)
        if ((ylims_max_1[ii]-(data_sks.spectra_dark_corr[logic_1, :, ii]).max())<((data_sks.spectra_dark_corr[logic_1, :, ii]).min()-ylims_min_1[ii])):
            ylims_max_1[ii] += unit_1

    xx = np.arange(256) + 1
    if logic[index]:

        ax_corr_zen_si.plot(xx, data_sks.spectra_dark_corr[index, :, 0], color='k', zorder=0)
        ax_corr_zen_si.axvline(100, ls='--', color='gray', lw=1.0)
        if data_sks.shutter[index] == 1:
            ax_corr_zen_si.scatter(xx, data_sks.dark_std[index, :, 0], c='b', s=3, zorder=1)
        ax_corr_zen_si.set_xlim([1, 256])
        ax_corr_zen_si.set_title('Dark-Corr. Zen. Si. (%d ms)' % (data_sks.int_time[index, 0]), color='Red')
        ax_corr_zen_si.tick_params(axis='both', which='major', labelsize=6)

        ax_corr_zen_in.plot(xx, data_sks.spectra_dark_corr[index, :, 1], color='k', zorder=0)
        ax_corr_zen_in.axvline(100, ls='--', color='gray', lw=1.0)
        if data_sks.shutter[index] == 1:
            ax_corr_zen_in.scatter(xx, data_sks.dark_std[index, :, 1], c='b', s=3, zorder=1)
        ax_corr_zen_in.set_xlim([1, 256])
        ax_corr_zen_in.set_title('Dark-Corr. Zen. In. (%d ms)' % (data_sks.int_time[index, 1]), color='Salmon')
        ax_corr_zen_in.tick_params(axis='both', which='major', labelsize=6)

        ax_corr_nad_si.plot(xx, data_sks.spectra_dark_corr[index, :, 2], color='k', zorder=0)
        ax_corr_nad_si.axvline(100, ls='--', color='gray', lw=1.0)
        if data_sks.shutter[index] == 1:
            ax_corr_nad_si.scatter(xx, data_sks.dark_std[index, :, 2], c='b', s=3, zorder=1)
        ax_corr_nad_si.set_xlim([1, 256])
        ax_corr_nad_si.set_title('Dark-Corr. Nad. Si. (%d ms)' % (data_sks.int_time[index, 2]), color='Blue')
        ax_corr_nad_si.tick_params(axis='both', which='major', labelsize=6)

        ax_corr_nad_in.plot(xx, data_sks.spectra_dark_corr[index, :, 3], color='k', zorder=0)
        ax_corr_nad_in.axvline(100, ls='--', color='gray', lw=1.0)
        if data_sks.shutter[index] == 1:
            ax_corr_nad_in.scatter(xx, data_sks.dark_std[index, :, 3], c='b', s=3, zorder=1)
        ax_corr_nad_in.set_xlim([1, 256])
        ax_corr_nad_in.set_title('Dark-Corr.  Nad. In. (%d ms)' % (data_sks.int_time[index, 3]), color='SkyBlue')
        ax_corr_nad_in.tick_params(axis='both', which='major', labelsize=6)
    else:
        ax_corr_zen_si.axis('off')
        ax_corr_zen_in.axis('off')
        ax_corr_nad_si.axis('off')
        ax_corr_nad_in.axis('off')

    if data_sks.shutter[index] == 0:
        ax_corr_zen_si.set_ylim([ylims_min_0[0], ylims_max_0[0]])
        ax_corr_zen_in.set_ylim([ylims_min_0[1], ylims_max_0[1]])
        ax_corr_nad_si.set_ylim([ylims_min_0[2], ylims_max_0[2]])
        ax_corr_nad_in.set_ylim([ylims_min_0[3], ylims_max_0[3]])
    elif data_sks.shutter[index] == 1:
        ax_corr_zen_si.set_ylim([ylims_min_1[0], ylims_max_1[0]])
        ax_corr_zen_in.set_ylim([ylims_min_1[1], ylims_max_1[1]])
        ax_corr_nad_si.set_ylim([ylims_min_1[2], ylims_max_1[2]])
        ax_corr_nad_in.set_ylim([ylims_min_1[3], ylims_max_1[3]])

    # -
    # gs_counts_ts.tight_layout(fig, rect=[0.0, 0.0, 1.0, 0.6])
    gs_counts_ts.update(bottom=0.0, top=0.65)
    gs_counts_ts.update(wspace=0.3, hspace=0.3)

    # left = min(gs_temp_ts.left, gs_counts_ts.left)
    # right= max(gs_temp_ts.right,gs_counts_ts.right)
    # gs_temp_ts.update(left=left, right=right)
    # gs_counts_ts.update(left=left, right=right)

    # ax1.get_yaxis().get_major_formatter().set_scientific(False)
    # ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    # ax2.get_yaxis().get_major_formatter().set_scientific(False)
    # ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    # ax3.get_yaxis().get_major_formatter().set_scientific(False)
    # ax3.get_yaxis().get_major_formatter().set_useOffset(False)
    # ax4.get_yaxis().get_major_formatter().set_scientific(False)
    # ax4.get_yaxis().get_major_formatter().set_useOffset(False)

    if testMode:
        plt.show()
        exit()
    else:
        plt.savefig('%s_%6.6d.png' % (nameTag, index), transparent=False, bbox_inches=None)
        plt.close(fig)

def SSFR_VIDEO(data_sks, ncpu=6, fdir_graph='graph/tmp', fname_vid='test.mp4'):

    import multiprocessing as mp

    fdir_graph = os.path.abspath(fdir_graph)
    fname_vid = os.path.abspath(fname_vid)

    if not os.path.exists(fdir_graph):
        os.system('mkdir -p %s' % fdir_graph)
    if len(glob.glob('%s/*.png' % fdir_graph)) > 0:
        os.system('find %s -name "*.png" | xargs rm -f' % fdir_graph)

    os.chdir(fdir_graph)

    indice = np.arange(data_sks.shutter.size)
    inits  = [data_sks]*data_sks.shutter.size

    pool = mp.Pool(processes=ncpu)
    pool.outputs = pool.map(SSFR_FRAME, zip(inits, indice))
    pool.close()
    pool.join()

    command = 'ffmpeg -y -framerate 10 -pattern_type glob -i \'%s/*.png\' -vf scale=2010:-1 -c:v libx264 -pix_fmt yuv420p %s' % (fdir_graph, fname_vid)
    print(command)
    os.system(command)


if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib.ticker import FixedLocator, MaxNLocator
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import matplotlib.patches as mpatches

    # fnames = sorted(glob.glob('data/20180104/sat_test_s600i600/*.SKS'))
    fnames = sorted(glob.glob('data/20180104/cal_test_amesLC/*.SKS'))
    data_sks = READ_SKS(fnames)

    # SSFR_FRAME([data_sks, 130], testMode=True)
    SSFR_VIDEO(data_sks)
