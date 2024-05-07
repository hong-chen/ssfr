# under development
import os
import sys
import struct
import datetime
import h5py
import numpy as np
import pysolar

__all__ = [
            'read_cu_alp',
            'cu_alp',
            'read_cu_alp_v1',
            'cu_alp_v1',
            'read_cu_alp_v2',
            'cu_alp_v2',
            'load_h5',
            'cal_heading',
        ]



def cal_julian_day(date, tmhr):

    julian_day = np.zeros_like(tmhr, dtype=np.float64)

    for i in range(tmhr.size):
        tmhr0 = tmhr[i]
        julian_day[i] = (date - datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0 + tmhr0/24.0

    return julian_day

def cal_heading(lon, lat):

    dx = lat[1:]-lat[:-1]
    dy = lon[1:]-lon[:-1]

    heading = np.rad2deg(np.arctan2(dy, dx))
    heading = np.append(heading[0], heading) % 360.0

    return heading

def load_h5(fname):

    f = h5py.File(fname, 'r')
    data = {}
    for key in f.keys():
        data[key] = f[key][...]
    f.close()

    return data



def read_cu_alp(fname, vnames=None, dataLen=248, verbose=False):

    fileSize = os.path.getsize(fname)
    if fileSize > dataLen:
        iterN    = fileSize // dataLen
        residual = fileSize %  dataLen
        if residual != 0:
            print('Warning [read_cu_alp]: \'%s\' has invalid data size.' % fname)
    else:
        sys.exit('Error   [read_cu_alp]: \'%s\' has invalid file size.' % fname)

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

class cu_alp:

    def __init__(self, fnames, date=None, tmhr_range=None, Ndata=15000, time_add_offset=0.0, verbose=False):

        if len(fnames) == 0:
            sys.exit('Error   [cu_alp]: No files are found in \'fnames\'.')

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
            dataAll0 = read_cu_alp(fname, vnames=vnames, dataLen=248, verbose=False)
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



def read_cu_alp_v2(fname, vnames=None, dataLen=232, verbose=False):

    fileSize = os.path.getsize(fname)
    if fileSize > dataLen:
        iterN    = fileSize // dataLen
        residual = fileSize %  dataLen
        if residual != 0:
            print('Warning [read_cu_alp_v2]: \'%s\' has invalid data size.' % fname)
    else:
        sys.exit('Error   [read_cu_alp_v2]: \'%s\' has invalid file size.' % fname)

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
               'Inclinometer_Pitch':28
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
        data    = struct.unpack('<29d', dataRec)
        if vnames == None:
            dataAll[i,:] = np.array(data, dtype=np.float64)
        else:
            dataAll[i,:] = np.array(data, dtype=np.float64)[vnames_indice]

    if verbose:
        print(dataAll[:, 3].min()/3600.0, dataAll[:, 3].max()/3600.0)
        print('--------------------------------------------------')

    return dataAll

class cu_alp_v2:

    def __init__(self, fnames, date=None, tmhr_range=None, Ndata=15000, time_add_offset=0.0, verbose=False):

        if len(fnames) == 0:
            sys.exit('Error   [cu_alp_v2]: No files are found in \'fnames\'.')

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
            dataAll0 = read_cu_alp_v2(fname, vnames=vnames, dataLen=232, verbose=False)
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



def read_cu_alp_v1(fname, vnames=None, dataLen=126, verbose=False):

    fileSize = os.path.getsize(fname)
    if fileSize > dataLen:
        iterN    = fileSize // dataLen
        residual = fileSize%  dataLen
        if residual != 0:
            print('Warning [read_cu_alp_v1]: \'%s\' has invalid data size.' % fname)
    else:
        print('Error   [read_cu_alp_v1]: \'%s\' has invalid file size.' % fname)
        return None

    vnames_dict = {'mini_pit':0,
                   'mini_rol':1,
                 'GPSSeconds':2,
                    'GPSWeek':3,
                        'Lat':4,
                        'Lon':5,
                        'Alt':6,
                        'Rol':7,
                        'Pit':8,
                        'Azi':9,
             'North_velocity':10,
              'East_velocity':11,
                'Up_velocity':12,
                  'motor_Rol':13,
                  'motor_Pit':14,
                 'temp_stage':15,
               'temp_c_pitch':16,
                 'temp_c_rol':17,
               'temp_m_pitch':18,
                 'temp_m_rol':19,
                      'acc_x':20,
                      'acc_y':21,
                      'acc_z':22,
          'relative_humidity':23,
        'chassis_temperature':24,
             'system_voltage':25,
      'INS_solution_inactive':26,
               'INS_aligning':27,
      'INS_solution_not_good':28,
          'INS_solution_good':29,
      'INS_bad_GPS_agreement':30,
     'INS_alignment_complete':31,
               'power_status':32,
            'SPAN_CPT_status':33,
                'file_status':34,
               'pitch_status':35,
                'roll_status':36,
                'FPGA_status':37,
                  'ARINC_rol':38,
                  'ARINC_pit':39,
                  'data_flag':40,
            'mini_INS_status':41}

    if vnames == None:
        dataAll = np.zeros((iterN, len(vnames_dict)), dtype=np.float64)
        print('Warning [read_cu_alp_v1]: Some variables should be integer type, please change accordingly.')
    else:
        dataAll = np.zeros((iterN, len(vnames)), dtype=np.float64)
        vnames_indice = []
        for vname in vnames:
            if vnames_dict[vname] in [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 41]:
                print('Warning [read_cu_alp_v1]: %s should be integer type, please change accordingly.' % vname)
            vnames_indice.append(vnames_dict[vname])

    f = open(fname, 'rb')
    if verbose:
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Reading %s...' % fname.split('/')[-1])

    # read data record
    for i in range(iterN):
        dataRec = f.read(dataLen)
        data    = struct.unpack('<26f12B2f2B', dataRec)
        if vnames == None:
            dataAll[i,:] = np.array(data, dtype=np.float64)
        else:
            dataAll[i,:] = np.array(data, dtype=np.float64)[vnames_indice]

    if verbose:
        print(dataAll[:, 0].min()/86400.0, dataAll[:, 0].max()/86400.0)
        print('--------------------------------------------------')

    return dataAll

class cu_alp_v1:

    def __init__(self, fnames, date=None, tmhr_range=None, Ndata=15000, time_add_offset=0.0, verbose=False):

        if len(fnames) == 0:
            sys.exit('Error   [cu_alp_v1]: No files are found in \'fnames\'.')

        vnames=['GPSSeconds',
                'GPSWeek',
                'Pit',
                'Rol',
                'motor_Pit',
                'motor_Rol',
                'Azi',
                'Lon',
                'Lat',
                'Alt',
                'ARINC_pit',
                'ARINC_rol']

        Nx         = Ndata * len(fnames)
        dataAll    = np.zeros((Nx, len(vnames)), dtype=np.float64)

        Nstart = 0
        for fname in fnames:
            dataAll0 = read_cu_alp_v1(fname, vnames=vnames, dataLen=126, verbose=False)
            Nend = Nstart + dataAll0.shape[0]
            print(dataAll0.shape)
            dataAll[Nstart:Nend, ...]  = dataAll0
            Nstart = Nend

        dataAll   = dataAll[:Nend, ...]

        date0   = datetime.datetime(1980, 1, 6)
        jday0   = (date0   -(datetime.datetime(1, 1, 1))).days + 1.0

        jday = (dataAll[:, 0]+time_add_offset)/86400.0 + 7.0* dataAll[:, 1] + jday0
        jday_int = np.int_(jday)
        jday_unique, counts = np.unique(jday_int, return_counts=True)
        if date is None:
            jdayRef = jday_unique[np.argmax(counts)]
        else:
            jdayRef = (date - (datetime.datetime(1, 1, 1))).days + 1.0
        tmhr    = (jday-jdayRef) * 24.0

        index_lon = 7
        lon = dataAll[:, index_lon]; lon[lon<0.0] += 360.0
        index_lat = 8
        lat = dataAll[:, index_lat]
        logic = (tmhr>0.0) & (lon>0.0) & (lon<360.0) & (lat>-90.0) & (lat<90.0)
        dataAll   = dataAll[logic, :]

        self.jday        = jday[logic]    # julian day
        self.tmhr        = tmhr[logic]    # time in hour
        self.lon         = dataAll[:, 7]  # longitude
        self.lat         = dataAll[:, 8]  # latitude
        self.alt         = dataAll[:, 9]  # altitude
        self.ang_pit_a   = dataAll[:, 10] # ARINC pitch angle
        self.ang_rol_a   = dataAll[:, 11] # ARINC roll angle
        self.ang_pit_s   = dataAll[:, 2]  # SPAN CPT pitch angle
        self.ang_rol_s   = dataAll[:, 3]  # SPAN CPT roll angle
        self.ang_pit_m   = dataAll[:, 4]  # motor pitch angle
        self.ang_rol_m   = dataAll[:, 5]  # motor roll angle
        self.ang_hed     = dataAll[:, 6]  # SPAN CPT heading angle

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
        f['lon']       = self.lon
        f['lat']       = self.lat
        f['alt']       = self.alt

        if hasattr(self, 'jday'):
            f['jday'] = self.jday

        f.close()




class cu_alp_old:

    def __init__(self, date, fdir='data/ssfr', tmhr_range=None, Ndata=15000, config=None, secOffset=16.0):

        date_s = date.strftime('%Y%m%d')
        fnames = sorted(glob.glob('%s/%s/plt/*.plt2' % (fdir, date_s)))

        if type(fnames) is not list:
            exit('Error   [READ_PLT2_SSFR]: input variable "fnames" should be in list type.')
        else:
            if len(fnames) == 0:
                exit('Error   [READ_PLT2_SSFR]: input variable "fnames" have no files.')

        vnames=['GPSSeconds', 'GPSWeek', 'Pit', 'Rol', 'motor_Pit', 'motor_Rol', 'Azi', 'Lon', 'Lat', 'Alt', 'ARINC_pit', 'ARINC_rol']
        Nx         = Ndata * len(fnames)
        dataAll    = np.zeros((Nx, len(vnames)), dtype=np.float64)

        Nstart = 0
        for fname in fnames:
            dataAll0 = READ_PLT2_ONE_V1(fname, vnames=vnames, dataLen=126, verbose=False)
            try:
                Nend = Nstart + dataAll0.shape[0]
                dataAll[Nstart:Nend, ...]  = dataAll0
                Nstart = Nend
            except:
                pass

        if config != None:
            self.config = config

        dataAll   = dataAll[:Nend, ...]

        date0   = datetime.datetime(1980, 1, 6)
        jday0   = (date0   -(datetime.datetime(1, 1, 1))).days + 1.0

        jday = dataAll[:, 0]/86400.0 + 7.0* dataAll[:, 1] + jday0
        jdayRef = (date -(datetime.datetime(1, 1, 1))).days + 1.0

        tmhr0 = (jday-jdayRef)*24.0
        if tmhr_range != None:
            logic = (tmhr0>=tmhr_range[0]) & (tmhr0<=tmhr_range[1])
        else:
            logic = (tmhr0>=0.0) & (tmhr0<=48.0)

        self.secOffset = secOffset
        self.jday_corr = (jday - secOffset/86400.0)[logic]
        self.tmhr_corr = (tmhr0 - secOffset/3600.0)[logic]

        self.ang_pit     = dataAll[:, 2][logic] # light collector pitch/roll angle
        self.ang_rol     = dataAll[:, 3][logic]
        self.ang_pit_m   = dataAll[:, 4][logic] # motor pitch/roll angle
        self.ang_rol_m   = dataAll[:, 5][logic]
        self.ang_head360 = dataAll[:, 6][logic] # heading angle
        self.lon         = dataAll[:, 7][logic]
        self.lon[self.lon<0.0] += 360.0
        self.lat         = dataAll[:, 8][logic]
        self.alt         = dataAll[:, 9][logic]
        self.ang_pit_a   = dataAll[:, 10][logic]# aircraft pitch/roll angle
        self.ang_rol_a   = dataAll[:, 11][logic]

        self.ang_head    = self.ang_head360.copy()
        self.ang_head[self.ang_head>180.0] -= 360.0



if __name__ == '__main__':

    pass
