import os
import sys
import glob
import struct
import datetime
import multiprocessing as mp
import h5py
from pyhdf.SD import SD, SDC
import numpy as np
from scipy import interpolate
from scipy.io import readsav
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
# import cartopy.crs as ccrs



__all__ = ['cg4']



def read_cg4_cfg(fname):

    with open(fname) as f:

        head = []
        body = []

        for line in f:
            line = line.partition('#')[0].rstrip()
            if len(line) > 0:
                line_data = line.split()
                head.append(line_data[0])

                for i in range(1, len(line_data)):
                    try:
                        line_data[i] = float(line_data[i])
                    except ValueError:
                        line_data[i] = line_data[i]

                if len(line_data) > 2:
                    body.append(line_data[1:])
                else:
                    body.append(line_data[1])

    cfg_dict = dict(zip(head, body))

    return cfg_dict



def read_cg4_raw(fname, headLen=0):

    """



    Notes:
    Here's what's from IDL
    data    = {time1: lonarr(2), cnt:lonarr(1),$
               status1:bytarr(1), pad1:bytarr(3), serial1:lonarr(1),sys_serial1:intarr(1),version1:bytarr(1),gain1:bytarr(1),$
               voltage1:lonarr(1),temperature1:lonarr(1),reartemp1:lonarr(1),systemp1:lonarr(1),caltemp1:lonarr(1),$
               status2:bytarr(1), pad2:bytarr(3), serial2:lonarr(1),sys_serial2:intarr(1),version2:bytarr(1),gain2:bytarr(1),$
               voltage2:lonarr(1),temperature2:lonarr(1),reartemp2:lonarr(1),systemp2:lonarr(1),caltemp2:lonarr(1)}

    dataRecFmt = '<2l1l1B3B1l1h1B1B1l1l1l1l1l1B3B1l1h1B1B1l1l1l1l1l'
    dataLen    = struct.calcsize(dataRecFmt)
    dataRec    = f.read(dataLen)
    data       = struct.unpack(dataRecFmt, dataRec)

    There are total of 29 data records in data, the corresponding indices in data

    time1(0,1) , cnt(2),
    status1(3) , pad1(4,5,6)   , serial1(7) , sys_serial1(8) , version1(9) , gain1(10), voltage1(11), temperature1(12), reartemp1(13), systemp1(14), caltemp1(15),
    status2(16), pad2(17,18,19), serial2(20), sys_serial2(21), version2(22), gain2(23), voltage2(24), temperature2(25), reartemp2(26), systemp2(27), caltemp2(28)

    IDL                  Python
    '800000'XL           int('800000', 16)
    """


    dataRecFmt = '<2l1l1B3B1l1h1B1B1l1l1l1l1l1B3B1l1h1B1B1l1l1l1l1l'
    dataLen    = struct.calcsize(dataRecFmt)

    fileSize = os.path.getsize(fname)
    if fileSize > headLen:
        iterN   = (fileSize-headLen) // dataLen
        residual = (fileSize-headLen) %  dataLen
        if residual != 0:
            print('Warning [read_cg4_raw]: %s contains unreadable data, omit the last data record...' % fname)
    else:
        exit('Error   [read_cg4_raw]: %s has invalid file size.' % fname)

    julian_sec    = np.zeros(iterN, dtype=np.float64)  # start from 1970-01-01 00:00:00
    vol_zen       = np.zeros(iterN, dtype=np.float64)
    vol_nad       = np.zeros(iterN, dtype=np.float64)
    temp_zen      = np.zeros(iterN, dtype=np.float64)
    temp_nad      = np.zeros(iterN, dtype=np.float64)
    temp_rear_zen = np.zeros(iterN, dtype=np.float64)
    temp_rear_nad = np.zeros(iterN, dtype=np.float64)
    temp_sys_zen  = np.zeros(iterN, dtype=np.float64)
    temp_sys_nad  = np.zeros(iterN, dtype=np.float64)

    const = int('800000', 16)
    with open(fname, 'rb') as f:

        for i in range(iterN):
            dataRec = f.read(dataLen)

            data = struct.unpack(dataRecFmt, dataRec)

            factor1 = 1.25 / float(const) / float(data[10])
            factor2 = 1.25 / float(const) / float(data[23])

            vol_zen[i] = (data[11] - const) * factor1
            vol_nad[i] = (data[24] - const) * factor2

            temp_zen[i] = (data[12] - const) * factor1
            temp_nad[i] = (data[25] - const) * factor2

            temp_rear_zen[i] = (data[13] - const) * factor1
            temp_rear_nad[i] = (data[26] - const) * factor2

            temp_sys_zen[i] =(data[14] - const) * factor1
            temp_sys_zen[i] =(data[27] - const) * factor2

            julian_sec[i] = data[0]

    return julian_sec, vol_zen, vol_nad, temp_zen, temp_nad, temp_rear_zen, temp_rear_nad, temp_sys_zen, temp_sys_nad, iterN



class cg4:

    """
    Read CG4 data

    Input:
        fnames: Python list of CG4 file paths (string type)
    """


    def __init__(self, fnames, fname_cfg, dtime, Ndata=600):

        if type(fnames) is not list:
            exit('Error   [read_cg4]: input variable should be \'list\'.')
        if len(fnames) == 0:
            exit('Error   [read_cg4]: input \'list\' is empty.')

        Nf = Ndata * len(fnames)

        julian_sec    = np.zeros(Ndata*Nf, dtype=np.float64)  # start from 1970-01-01 00:00:00
        vol_zen       = np.zeros(Ndata*Nf, dtype=np.float64)
        vol_nad       = np.zeros(Ndata*Nf, dtype=np.float64)
        temp_zen      = np.zeros(Ndata*Nf, dtype=np.float64)
        temp_nad      = np.zeros(Ndata*Nf, dtype=np.float64)
        temp_rear_zen = np.zeros(Ndata*Nf, dtype=np.float64)
        temp_rear_nad = np.zeros(Ndata*Nf, dtype=np.float64)
        temp_sys_zen  = np.zeros(Ndata*Nf, dtype=np.float64)
        temp_sys_nad  = np.zeros(Ndata*Nf, dtype=np.float64)

        Nstart = 0
        for fname in fnames:

            julian_sec0, vol_zen0, vol_nad0, temp_zen0, temp_nad0, temp_rear_zen0, temp_rear_nad0, temp_sys_zen0, temp_sys_nad0, iterN = read_cg4_raw(fname)

            Nend = Nstart + iterN

            julian_sec[Nstart:Nend]    = julian_sec0
            vol_zen[Nstart:Nend]       = vol_zen0
            vol_nad[Nstart:Nend]       = vol_nad0
            temp_zen[Nstart:Nend]      = temp_zen0
            temp_nad[Nstart:Nend]      = temp_nad0
            temp_rear_zen[Nstart:Nend] = temp_rear_zen0
            temp_rear_nad[Nstart:Nend] = temp_rear_nad0
            temp_sys_zen[Nstart:Nend]  = temp_sys_zen0
            temp_sys_nad[Nstart:Nend]  = temp_sys_nad0

            Nstart = Nend

        julian_sec         = julian_sec[:Nend]
        self.tmhr          = (julian_sec-((dtime-datetime.datetime(1970, 1, 1)).days)*86400) / 3600.0
        self.vol_zen       = vol_zen[:Nend]
        self.vol_nad       = vol_nad[:Nend]
        self.temp_zen      = temp_zen[:Nend]
        self.temp_nad      = temp_nad[:Nend]
        self.temp_rear_zen = temp_rear_zen[:Nend]
        self.temp_rear_nad = temp_rear_nad[:Nend]
        self.temp_sys_zen  = temp_sys_zen[:Nend]
        self.temp_sys_nad  = temp_sys_nad[:Nend]

        self.calibrate(fname_cfg)

        self.filter()

        self.save_h5('CG4_%s.h5' % dtime.strftime('%Y%m%d'))

        self.plot('CG4_QL_%s.png' % dtime.strftime('%Y%m%d'))


    def calibrate(self, fname_cfg, nad_id=20618, zen_id=20592):

        self.zen = {}
        self.nad = {}

        cfg = read_cg4_cfg(fname_cfg)

        for i in range(len(cfg['cg4cal'])//4):
            if nad_id == cfg['cg4cal'][i*4]:
                flux_coef_nad = [cfg['cg4cal'][i*4+1], cfg['cg4cal'][i*4+2], cfg['cg4cal'][i*4+3]]
                temp_coef_nad = [cfg['cg4tem'][i*7+1], cfg['cg4tem'][i*7+2], cfg['cg4tem'][i*7+3], cfg['cg4tem'][i*7+4], cfg['cg4tem'][i*7+5], cfg['cg4tem'][i*7+6]]
            if zen_id == cfg['cg4cal'][i*4]:
                flux_coef_zen = [cfg['cg4cal'][i*4+1], cfg['cg4cal'][i*4+2], cfg['cg4cal'][i*4+3]]
                temp_coef_zen = [cfg['cg4tem'][i*7+1], cfg['cg4tem'][i*7+2], cfg['cg4tem'][i*7+3], cfg['cg4tem'][i*7+4], cfg['cg4tem'][i*7+5], cfg['cg4tem'][i*7+6]]

        sigma = 5.6704e-8
        self.nad['T']      = temp_coef_nad[0] + temp_coef_nad[1]*1.0e3*self.temp_nad
        self.nad['T_rear'] = temp_coef_nad[2] + temp_coef_nad[3]*1.0e3*self.temp_rear_nad
        self.nad['T_sys']  = temp_coef_nad[4] + temp_coef_nad[5]*1.0e3*self.temp_sys_nad
        self.nad['F_net']  = flux_coef_nad[0] + self.vol_nad*1.0e6*flux_coef_nad[1]
        self.nad['F']      = self.nad['F_net']+ flux_coef_nad[2]*sigma*(self.nad['T']+273.15)**4.0

        self.zen['T']      = temp_coef_zen[0] + temp_coef_zen[1]*1.0e3*self.temp_zen
        self.zen['T_rear'] = temp_coef_zen[2] + temp_coef_zen[3]*1.0e3*self.temp_rear_zen
        self.zen['T_sys']  = temp_coef_zen[4] + temp_coef_zen[5]*1.0e3*self.temp_sys_zen
        self.zen['F_net']  = flux_coef_zen[0] + self.vol_zen*1.0e6*flux_coef_zen[1]
        self.zen['F']      = self.zen['F_net']+ flux_coef_zen[2]*sigma*(self.zen['T']+273.15)**4.0


    def filter(self):

        # logic = (self.nad['T']>0.0) & (self.zen['T']>0.0) & (self.tmhr>0.0) & (self.tmhr<24.0)
        # logic = (self.tmhr>0.0) & (self.tmhr<24.0)
        # logic = np.repeat(True, self.nad['T'].size)
        # logic = (self.tmhr>0.0) & (self.tmhr<24.0) & (self.nad['F']<1000.0) & (self.zen['F']<1000.0)
        logic = (self.nad['T']>-200.0) & (self.zen['T']>-200.0) & (self.tmhr>0.0) & (self.tmhr<24.0) & (self.nad['F']>-500.0) & (self.nad['F']<1000.0) &  (self.zen['F']>-500.0) & (self.zen['F']<1000.0)


        for key in self.nad.keys():
            self.nad[key] = self.nad[key][logic]
        for key in self.zen.keys():
            self.zen[key] = self.zen[key][logic]

        self.tmhr = self.tmhr[logic]


    def save_h5(self, fname):

        f  = h5py.File(fname, 'w')
        f['tmhr'] = self.tmhr
        g1 = f.create_group('nad')
        for key in self.nad.keys():
            g1[key] = self.nad[key]
        g2 = f.create_group('zen')
        for key in self.zen.keys():
            g2[key] = self.zen[key]
        f.close()


    def plot(self, fname):

        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        ax1.scatter(self.tmhr, self.nad['F'], c='blue', s=6)
        ax1.scatter(self.tmhr, self.zen['F'], c='red' , s=6)
        ax1.set_xlim((0, 24))
        ax1.set_xlabel('Time [Hour]')
        ax1.set_ylabel('Flux [$\mathrm{W m^{-2}}$]')

        ax2 = fig.add_subplot(212)
        ax2.scatter(self.tmhr, self.nad['T'], c='blue', s=6)
        ax2.scatter(self.tmhr, self.zen['T']     , c='red' , s=6)
        ax2.set_xlim((0, 24))
        ax2.set_xlabel('Time [Hour]')
        ax2.set_ylabel('Temperature [$^\circ$C]')
        patches_legend = [
                    mpatches.Patch(color='blue' , label='Nadir'),
                    mpatches.Patch(color='red'  , label='Zenith')
                    ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)

        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)



if __name__ == '__main__':

    fdir       = 'data/20170813'

    fnames_cg4 = sorted(glob.glob('%s/*.CG4' % fdir))
    fname_cfg  = 'cg4_20181224.cfg'

    cg4 = read_cg4(fnames_cg4, fname_cfg, datetime.datetime(2017, 8, 13))
