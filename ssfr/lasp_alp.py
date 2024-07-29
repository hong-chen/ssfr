import os
import sys
import struct
import datetime
import warnings
import h5py
import numpy as np

import ssfr

__all__ = [
            'read_alp_raw',
            'read_alp',
        ]


def read_alp_raw(fname, vnames=None, dataLen=248, verbose=False):

    vnames_dict  = {                    \
                    'Computer_Hour':0,  \
                  'Computer_Minute':1,  \
                  'Computer_Second':2,  \
                         'GPS_Time':3,  \
                         'GPS_Week':4,  \
                   'Velocity_North':5,  \
                    'Velocity_East':6,  \
                      'Velocity_Up':7,  \
                   'Span_CPT_Pitch':8,  \
                    'Span_CPT_Roll':9,  \
                         'Latitude':10, \
                        'Longitude':11, \
                           'Height':12, \
                  'Span_CPT_Status':13, \
                      'Motor_Pitch':14, \
                       'Motor_Roll':15, \
         'Inclinometer_Temperature':16, \
           'Motor_Roll_Temperature':17, \
          'Motor_Pitch_Temperature':18, \
                'Stage_Temperature':19, \
                'Relative_Humidity':20, \
              'Chassis_Temperature':21, \
                   'System_Voltage':22, \
                       'ARINC_Roll':23, \
                      'ARINC_Pitch':24, \
        'Inclinometer_Roll_Voltage':25, \
       'Inclinometer_Pitch_Voltage':26, \
                'Inclinometer_Roll':27, \
               'Inclinometer_Pitch':28, \
                   'Reference_Roll':29, \
                  'Reference_Pitch':30  \
                  }


    fileSize = os.path.getsize(fname)
    if fileSize > dataLen:
        iterN    = fileSize // dataLen
        residual = fileSize %  dataLen
        if residual != 0:
            msg = '\nWarning [read_alp_raw]: <%s> has invalid data size.' % fname
            warnings.warn(msg)
    else:
        msg = '\nWarning [read_alp_raw]: \'%s\' has invalid file size.' % fname
        warnings.warn(msg)
        iterN = 0
        if vnames == None:
            dataAll = np.zeros((iterN, len(vnames_dict)), dtype=np.float64)
        else:
            dataAll = np.zeros((iterN, len(vnames)), dtype=np.float64)
        return dataAll

    if vnames == None:
        dataAll = np.zeros((iterN, len(vnames_dict)), dtype=np.float64)
    else:
        dataAll = np.zeros((iterN, len(vnames)), dtype=np.float64)
        vnames_indice = []
        for vname in vnames:
            vnames_indice.append(vnames_dict[vname])

    f = open(fname, 'rb')
    if verbose:
        print('# //--------------------------------------------------------------------------\\ #')
        print('    Reading <%s> ...' % fname.split('/')[-1])

    # read data record
    for i in range(iterN):
        dataRec = f.read(dataLen)
        data    = struct.unpack('<31d', dataRec)
        if vnames == None:
            dataAll[i,:] = np.array(data, dtype=np.float64)
        else:
            dataAll[i,:] = np.array(data, dtype=np.float64)[vnames_indice]

    if verbose:
        print(dataAll[:, 3].min()/3600.0, dataAll[:, 3].max()/3600.0)
        print('# \\--------------------------------------------------------------------------// #')

    return dataAll


class read_alp:

    ID = 'CU LASP ALP'

    def __init__(
            self,
            fnames,
            date=None,
            tmhr_range=None,
            Ndata=15000,
            time_offset=0.0,
            verbose=ssfr.common.karg['verbose'],
            ):

        if len(fnames) == 0:
            msg = '\nError [read_alp]: No files are found in <fnames>.'
            raise OSError(msg)

        self.verbose = verbose

        Nfile = len(fnames)
        if self.verbose:
            msg = '\nMessage [read_alp]: Processing %s files (Total of %d):' % (self.ID, Nfile)
            print(msg)


        # variable names
        # /--------------------------------------------------------------------------\ #
        self.vnames_dict = {
                  'GPS_Time': 'gps_time',  \
            'Span_CPT_Pitch': 'ang_pit_s', \
             'Span_CPT_Roll': 'ang_rol_s', \
               'Motor_Pitch': 'ang_pit_m', \
                'Motor_Roll': 'ang_rol_m', \
                 'Longitude': 'lon', \
                  'Latitude': 'lat', \
                    'Height': 'alt', \
               'ARINC_Pitch': 'ang_pit_a', \
                'ARINC_Roll': 'ang_rol_a', \
        'Inclinometer_Pitch': 'ang_pit_i', \
         'Inclinometer_Roll': 'ang_rol_i', \
         'Relative_Humidity': 'rh', \
         }

        self.vnames = list(self.vnames_dict.keys())
        # \--------------------------------------------------------------------------/ #


        # read raw data
        # /--------------------------------------------------------------------------\ #
        Nx         = Ndata * len(fnames)
        dataAll    = np.zeros((Nx, len(self.vnames)), dtype=np.float64)

        Nstart = 0
        for i, fname in enumerate(fnames):

            if self.verbose:
                msg = '    reading %3d/%3d <%s> ...' % (i+1, Nfile, fname)
                print(msg)

            dataAll0 = read_alp_raw(fname, vnames=self.vnames, dataLen=248, verbose=False)
            Nend = Nstart + dataAll0.shape[0]
            dataAll[Nstart:Nend, ...]  = dataAll0
            Nstart = Nend

        dataAll   = dataAll[:Nend, ...]
        # \--------------------------------------------------------------------------/ #


        self.data_raw = {}
        self.data_raw['info'] = {}
        self.data_raw['info']['alp_tag'] = '%s' % (self.ID)
        self.data_raw['info']['fnames']  = fnames

        # retrieve time in hour (tmhr)
        # notes: after starting the ALP system, the GPS will need to go through initialization,
        #    during the initialization, the data of time, longitude, latitude etc. are bad values,
        #    the following code is to retrieve correct time from data that contains bad values
        # /--------------------------------------------------------------------------\ #
        day = (dataAll[:, self.vnames.index('GPS_Time')] + time_offset) / 86400.0
        day_int = np.int_(day)
        day_unique, counts = np.unique(day_int[day_int>0], return_counts=True)
        dayRef = day_unique[np.argmax(counts)]
        tmhr = (day-dayRef) * 24.0
        # \--------------------------------------------------------------------------/ #
        self.data_raw['tmhr'] = tmhr # time in hour

        for vname in self.vnames:
            self.data_raw[self.vnames_dict[vname]] = dataAll[:, self.vnames.index(vname)]

        self.data_raw['ang_hed'] = ssfr.util.cal_heading(self.data_raw['lon'], self.data_raw['lat'])

        if date is not None:
            self.data_raw['jday'] = ssfr.util.dtime_to_jday(date) + self.data_raw['tmhr']/24.0

        if self.verbose and ('jday' in self.data_raw.keys()):
            dtime_s0 = ssfr.util.jday_to_dtime(self.data_raw['jday'][0]).strftime('%Y-%m-%d %H:%M:%S')
            dtime_e0 = ssfr.util.jday_to_dtime(self.data_raw['jday'][-1]).strftime('%Y-%m-%d %H:%M:%S')
            msg = '\nMessage [read_alp]: Data processing complete (%s to %s).' % (dtime_s0, dtime_e0)
            print(msg)


    def save_h5(self, fname):

        f = h5py.File(fname, 'w')

        for vname in self.data_raw.keys():
            if vname not in ['info']:
                f.create_dataset(vname, data=self.data_raw[vname], compression='gzip', compression_opts=9, chunks=True)

        f.close()


if __name__ == '__main__':

    pass
