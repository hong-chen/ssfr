import os
import sys
import struct
import datetime
import h5py
import numpy as np
from ssfr.util import load_h5, cal_heading, cal_julian_day

__all__ = [
            'read_alp_raw',
            'read_alp',
        ]


def read_alp_raw(fname, vnames=None, dataLen=248, verbose=False):

    fileSize = os.path.getsize(fname)
    if fileSize > dataLen:
        iterN    = fileSize // dataLen
        residual = fileSize %  dataLen
        if residual != 0:
            print('Warning [read_alp_raw]: \'%s\' has invalid data size.' % fname)
    else:
        sys.exit('Error   [read_alp_raw]: \'%s\' has invalid file size.' % fname)

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

    if vnames == None:
        dataAll = np.zeros((iterN, len(vnames_dict)), dtype=np.float64)
    else:
        dataAll = np.zeros((iterN, len(vnames)), dtype=np.float64)
        vnames_indice = []
        for vname in vnames:
            vnames_indice.append(vnames_dict[vname])

    f = open(fname, 'rb')
    if verbose:
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Reading %s...' % fname.split('/')[-1])

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
        print('--------------------------------------------------')

    return dataAll

class read_alp:

    def __init__(self, fnames, date=None, tmhr_range=None, Ndata=15000, time_add_offset=0.0, verbose=False):

        if len(fnames) == 0:
            msg = '\nError [read_alp]: No files are found in <fnames>.'
            raise OSError(msg)

        vnames = ['GPS_Time',
                  'Span_CPT_Pitch',
                  'Span_CPT_Roll',
                  'Motor_Pitch',
                  'Motor_Roll',
                  'Longitude',
                  'Latitude',
                  'Height',
                  'ARINC_Pitch',
                  'ARINC_Roll',
                  'Inclinometer_Pitch',
                  'Inclinometer_Roll']

        Nx         = Ndata * len(fnames)
        dataAll    = np.zeros((Nx, len(vnames)), dtype=np.float64)

        Nstart = 0
        for fname in fnames:
            dataAll0 = read_alp_raw(fname, vnames=vnames, dataLen=248, verbose=False)
            Nend = Nstart + dataAll0.shape[0]
            dataAll[Nstart:Nend, ...]  = dataAll0
            Nstart = Nend

        dataAll   = dataAll[:Nend, ...]

        index_tmhr= 0
        tmhr = (dataAll[:, index_tmhr] + time_add_offset) / 3600.0
        tmhr_int = np.int_(tmhr)
        tmhr_unique, counts = np.unique(tmhr_int, return_counts=True)
        tmhrRef = tmhr_unique[np.argmax(counts)]
        while tmhrRef > 24.0:
            tmhr    -= 24.0
            tmhrRef -= 24.0

        index_lon = 5
        lon = dataAll[:, index_lon]; lon[lon<0.0] += 360.0
        index_lat = 6
        lat = dataAll[:, index_lat]
        logic = (tmhr>0.0) & (lon>0.0) & (lon<360.0) & (lat>-90.0) & (lat<90.0)
        dataAll   = dataAll[logic, :]

        self.tmhr        = tmhr[logic]    # time in hour
        self.lon         = dataAll[:, 5]  # longitude
        self.lat         = dataAll[:, 6]  # latitude
        self.alt         = dataAll[:, 7]  # altitude
        self.ang_pit_a   = dataAll[:, 8]  # ARINC pitch angle
        self.ang_rol_a   = dataAll[:, 9]  # ARINC roll angle
        self.ang_pit_s   = dataAll[:, 1]  # SPAN CPT pitch angle
        self.ang_rol_s   = dataAll[:, 2]  # SPAN CPT roll angle
        self.ang_pit_m   = dataAll[:, 3]  # motor pitch angle
        self.ang_rol_m   = dataAll[:, 4]  # motor roll angle
        self.ang_pit_i   = dataAll[:, 10] # inclinometer pitch angle
        self.ang_rol_i   = dataAll[:, 11] # inclinometer roll angle

        self.ang_hed     = cal_heading(self.lon, self.lat)
        if date is not None:
            self.jday = cal_julian_day(date, self.tmhr)

    def save_h5(self, fname):

        f = h5py.File(fname, 'w')

        f['tmhr']      = self.tmhr
        f['ang_hed']   = self.ang_hed
        f['ang_pit_s'] = self.ang_pit_s
        f['ang_rol_s'] = self.ang_rol_s
        f['ang_pit_m'] = self.ang_pit_m
        f['ang_rol_m'] = self.ang_rol_m
        f['ang_pit_a'] = self.ang_pit_a
        f['ang_rol_a'] = self.ang_rol_a
        f['ang_pit_i'] = self.ang_pit_i
        f['ang_rol_i'] = self.ang_rol_i
        f['lon']       = self.lon
        f['lat']       = self.lat
        f['alt']       = self.alt

        if hasattr(self, 'jday'):
            f['jday'] = self.jday

        f.close()


if __name__ == '__main__':

    pass
