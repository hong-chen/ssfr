import os
import glob
import h5py
import struct
import numpy as np
import datetime
from scipy.io import readsav
from scipy import stats
import pysolar

# ++++++++++++++++++++++++    for .plt3 files   ++++++++++++++++++++++++++++++
def READ_PLT3_ONE_V1(fname, vnames=None, dataLen=248, verbose=False):

    fileSize = os.path.getsize(fname)
    if fileSize > dataLen:
        iterN    = fileSize // dataLen
        residual = fileSize%  dataLen
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
                            'Pitch':8,  \
                             'Roll':9,  \
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

        # vnames_dict  = {                    \
        #                 'Computer_Hour':0,  \
        #               'Computer_Minute':1,  \
        #               'Computer_Second':2,  \
        #                      'GPS_Time':3,  \
        #                      'GPS_Week':4,  \
        #                'Velocity_North':5,  \
        #                 'Velocity_East':6,  \
        #                   'Velocity_Up':7,  \
        #                         'Pitch':8,  \
        #                          'Roll':9,  \
        #                      'Latitude':10, \
        #                     'Longitude':11, \
        #                        'Height':12, \
        #               'Span_CPT_Status':13, \
        #                   'Motor_Pitch':14, \
        #                    'Motor_Roll':15, \
        #      'Inclinometer_Temperature':16, \
        #        'Motor_Roll_Temperature':17, \
        #       'Motor_Pitch_Temperature':18, \
        #             'Stage_Temperature':19, \
        #             'Relative_Humidity':20, \
        #           'Chassis_Temperature':21, \
        #                'System_Voltage':22, \
        #                    'ARINC_Roll':23, \
        #                   'ARINC_Pitch':24, \
        #     'Inclinometer_Roll_Voltage':25, \
        #    'Inclinometer_Pitch_Voltage':26, \
        #             'Inclinometer_Roll':27, \
        #            'Inclinometer_Pitch':28, \
        #                'Reference_Roll':29, \
        #               'Reference_Pitch':30  \
        #               }

        vnames=['GPS_Time', 'Pitch', 'Roll', 'Motor_Pitch', 'Motor_Roll', 'Longitude', 'Latitude', 'Height', 'ARINC_Pitch', 'ARINC_Roll', 'Inclinometer_Pitch', 'Inclinometer_Roll']
        Nx         = Ndata * len(fnames)
        dataAll    = np.zeros((Nx, len(vnames)), dtype=np.float64)

        Nstart = 0
        for fname in fnames:
            dataAll0 = READ_PLT3_ONE_V1(fname, vnames=vnames, dataLen=248, verbose=False)
            print(dataAll0.shape)
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

# ----------------------------------------------------------------------------

if __name__ == '__main__':

    import matplotlib as mpl
    #mpl.use('Agg')
    from matplotlib.ticker import FixedLocator, MaxNLocator
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    fnames = sorted(glob.glob('/Users/hoch4240/Google Drive/CU LASP/ORACLES/Integration/2017/Test Flights/data/20170716/platform/*.plt3'))
    # fnames = sorted(glob.glob('/Users/hoch4240/Google Drive/CU LASP/ORACLES/Integration/2017/Test Flights/data/20170717/platform/*.plt3'))
    plat    = READ_PLT3(fnames)

    # figure settings
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(np.arange(plat.tmhr.size), plat.tmhr, s=1, c='k')
    plt.savefig('test.png')
    plt.show()

    exit()
