import os
import glob
import h5py
import struct
import numpy as np
import datetime
from scipy.io import readsav
from scipy import stats
import pysolar

def READ_PLT3_ONE_V1(fname, vnames=None, dataLen=248, verbose=False):

    fileSize = os.path.getsize(fname)
    if fileSize > dataLen:
        iterN    = fileSize // dataLen
        residual = fileSize %  dataLen
        if residual != 0:
            print('Warning [READ_PLT3_ONE_V1]: %s has invalid data size.' % fname)
    else:
        exit('Error [READ_PLT3_ONE_V1]: %s has invalid file size.' % fname)

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

class READ_PLT3:

    def __init__(self, fnames, Ndata=15000, secOffset=0.0):

        if type(fnames) is not list:
            exit('Error [READ_PLT3]: input variable "fnames" should be in list type.')

        vnames=['GPS_Time', 'Span_CPT_Pitch', 'Span_CPT_Roll', 'Motor_Pitch', 'Motor_Roll', 'Longitude', 'Latitude', 'Height', 'ARINC_Pitch', 'ARINC_Roll', 'Inclinometer_Pitch', 'Inclinometer_Roll']
        Nx         = Ndata * len(fnames)
        dataAll    = np.zeros((Nx, len(vnames)), dtype=np.float64)

        Nstart = 0
        for fname in fnames:
            dataAll0 = READ_PLT3_ONE_V1(fname, vnames=vnames, dataLen=248, verbose=False)
            Nend = Nstart + dataAll0.shape[0]
            dataAll[Nstart:Nend, ...]  = dataAll0
            Nstart = Nend

        dataAll   = dataAll[:Nend, ...]

        self.tmhr = dataAll[:, 0]/3600.0

        self.secOffset = secOffset
        self.tmhr_corr = self.tmhr - secOffset/3600.0

        self.ang_pit     = dataAll[:, 1] # light collector pitch/roll angle
        self.ang_rol     = dataAll[:, 2]
        self.ang_pit_m   = dataAll[:, 3] # motor pitch/roll angle
        self.ang_rol_m   = dataAll[:, 4]
        self.lon         = dataAll[:, 5]
        self.lat         = dataAll[:, 6]
        self.alt         = dataAll[:, 7]
        self.ang_pit_a   = dataAll[:, 8] # aircraft pitch/roll angle
        self.ang_rol_a   = dataAll[:, 9]
        self.ang_pit_i   = dataAll[:, 10]# accelerometer pitch/roll
        self.ang_rol_i   = dataAll[:, 11]

        logic = (self.tmhr>1.0) & (self.lon>=-180.0) & (self.lon<=360.0) & (self.lat>=-90.0) & (self.lat<=90.0)
        self.tmhr        = self.tmhr[logic]
        self.tmhr_corr   = self.tmhr_corr[logic]
        while self.tmhr[0] > 24.0:
            self.tmhr -= 24.0
            self.tmhr_corr -=24.0

        self.ang_pit     = self.ang_pit[logic]
        self.ang_rol     = self.ang_rol[logic]
        self.ang_pit_m   = self.ang_pit_m[logic]
        self.ang_rol_m   = self.ang_rol_m[logic]
        self.lon         = self.lon[logic]
        self.lat         = self.lat[logic]
        self.alt         = self.alt[logic]
        self.ang_pit_a   = self.ang_pit_a[logic]
        self.ang_rol_a   = self.ang_rol_a[logic]
        self.ang_pit_i   = self.ang_pit_i[logic]
        self.ang_rol_i   = self.ang_rol_i[logic]

def CAL_SOLAR_ANGLES(julian_day, longitude, latitude, altitude):

    dateRef = datetime.datetime(1, 1, 1)
    jdayRef = 1.0

    sza = np.zeros_like(julian_day)
    saa = np.zeros_like(julian_day)

    for i, jday in enumerate(julian_day):

        dtime_i = (dateRef + datetime.timedelta(days=jday-jdayRef)).replace(tzinfo=datetime.timezone.utc)

        sza_i = 90.0 - pysolar.solar.get_altitude(latitude[i], longitude[i], dtime_i, elevation=altitude[i])
        if sza_i < 0.0 or sza_i > 90.0:
            sza_i = np.nan
        sza[i] = sza_i

        saa_i = pysolar.solar.get_azimuth(latitude[i], longitude[i], dtime_i, elevation=altitude[i])
        if saa_i >= 0.0:
            if 0.0<=saa_i<=180.0:
                saa_i = 180.0 - saa_i
            elif 180.0<saa_i<=360.0:
                saa_i = 540.0 - saa_i
            else:
                saa_i = np.nan
        elif saa_i < 0.0:
            if -180.0<=saa_i<0.0:
                saa_i = -saa_i + 180.0
            elif -360.0<=saa_i<-180.0:
                saa_i = -saa_i - 180.0
            else:
                saa_i = np.nan
        saa[i] = saa_i

    return sza, saa

class READ_ICT_HSK:

    def __init__(self, fname):

        f = open(fname, 'r')
        firstLine = f.readline()
        skip_header = int(firstLine.split(',')[0])

        vnames = []
        units  = []
        for i in range(7):
            f.readline()
        vname0, unit0 = f.readline().split(',')
        vnames.append(vname0.strip())
        units.append(unit0.strip())
        Nvar = int(f.readline())
        for i in range(2):
            f.readline()
        for i in range(Nvar):
            vname0, unit0 = f.readline().split(',')
            vnames.append(vname0.strip())
            units.append(unit0.strip())
        f.close()

        data = np.genfromtxt(fname, skip_header=skip_header, delimiter=',')

        self.data = {}

        for i, vname in enumerate(vnames):
            print(vname)
            self.data[vname] = data[:, i]

if __name__ == '__main__':


    import matplotlib as mpl
    #mpl.use('Agg')
    from matplotlib.ticker import FixedLocator, MaxNLocator
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    fname = '/Users/hoch4240/Chen/work/07_ORACLES-2/cal/data/p3/20170815/Hskping_P3_20170815_R0.ict'
    hsk = READ_ICT_HSK(fname)

    fnames = sorted(glob.glob('/Users/hoch4240/Google Drive/CU LASP/ORACLES/Data/ORACLES 2017/p3/20170815/ALP/*.plt3'))
    alp = READ_PLT3(fnames)

    # figure settings
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(hsk.data['Start_UTC']/3600.0, hsk.data['MSL_GPS_Altitude']/1000.0, label='Aircraft', s=1)
    # ax1.scatter(hsk.data['Start_UTC']/3600.0, hsk.data['True_Heading'], label='Aircraft', s=1)
    ax1.scatter(alp.tmhr_corr, alp.alt/1000.0, label='ALP', s=1, c='r')
    # ax1.scatter(alp.tmhr_corr, alp.ang_hed, label='ALP', s=1, c='r')
    # ax1.legend(loc='best', fontsize=12, framealpha=0.4)
    plt.show()
    exit()

    julian_day = np.linspace(736554.0, 736555.0, 100)
    longitude  = np.repeat(0.332889, 100)
    latitude   = np.repeat(6.741197, 100)
    altitude   = np.repeat(0.0, 100)

    sza, saa = CAL_SOLAR_ANGLES(julian_day, longitude, latitude, altitude)
    # figure settings
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter((julian_day-736554.0)*24.0, sza)
    # ax1.legend(loc='best', fontsize=12, framealpha=0.4)
    plt.show()
    exit()

    fnames = sorted(glob.glob('/Users/hoch4240/Google Drive/CU LASP/ORACLES/Data/ORACLES 2017/p3/20170812/ALP/*.plt3'))
    # fnames = sorted(glob.glob('/Users/hoch4240/Google Drive/CU LASP/ORACLES/Integration/2017/Test Flights/data/20170717/platform/*.plt3'))
    # fnames = sorted(glob.glob('/Users/hoch4240/Google Drive/CU LASP/ORACLES/Integration/2017/Test Flights/data/20170717/platform/*.plt3'))
    plat    = READ_PLT3(fnames)


    # figure settings
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    # ax1.scatter(plat.tmhr, plat.ang_rol, s=0.1, c='r')
    ax1.scatter(plat.tmhr, plat.ang_rol_m+0.3, s=0.1, c='b')
    # ax1.scatter(plat.lon, plat.lat, s=0.1, c='r')
    plt.show()

    exit()
