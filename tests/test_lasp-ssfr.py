import os
import sys
import glob
import datetime
import h5py
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
import numpy as np
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.image as mpl_img
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cartopy.crs as ccrs
# mpl.use('Agg')

import ssfr

def cdata_rad_resp(
        fdir_lab,
        fdir_field=None,
        plot=True,
        intTime={'si':60, 'in':300},
        field_lamp_tag='150',
        ):

    dirs = get_sub_dir(fdir_lab, full=False)

    if len(dirs) != 2:
        sys.exit('Error [cdata_rad_resp_camp2ex]: Incomplete lab radiometric calibration dataset.')

    if field_lamp_tag in dirs[0]:
        index_tra = 0
        index_pri = 1
    elif field_lamp_tag in dirs[1]:
        index_tra = 1
        index_pri = 0
    else:
        sys.exit('Error [cdata_rad_resp_camp2ex]: Cannot locate lab radiometric calibration for field lamp.')

    fdir_tra = '%s/%s' % (fdir_lab, dirs[index_tra])
    fdir_pri = '%s/%s' % (fdir_lab, dirs[index_pri])

    if fdir_field is None:
        fdir_field = fdir_tra
    else:
        fdir_field = get_sub_dir(fdir_field, full=True, contains=[field_lamp_tag])[0]

    filename_tag0 = '%s/%s_%s' % (os.path.dirname(fdir_lab), os.path.basename(fdir_lab).replace('_', '-'), os.path.basename(os.path.dirname(fdir_field)).replace('_', '-'))

    lc_all = get_sub_dir(fdir_field, full=False, contains=['zenith', 'nadir'])

    for lc in lc_all:

        fnames_pri = {'dark':'%s/%s/s%di%d/dark/spc00000.OSA2' % (fdir_pri, lc, intTime['si'], intTime['in']),\
                      'cal' :'%s/%s/s%di%d/cal/spc00000.OSA2'  % (fdir_pri, lc, intTime['si'], intTime['in'])}
        fnames_tra = {'dark':'%s/%s/s%di%d/dark/spc00000.OSA2' % (fdir_tra, lc, intTime['si'], intTime['in']),\
                      'cal' :'%s/%s/s%di%d/cal/spc00000.OSA2'  % (fdir_tra, lc, intTime['si'], intTime['in'])}
        fnames_sec = {'dark':'%s/%s/s%di%d/dark/spc00000.OSA2' % (fdir_field, lc, intTime['si'], intTime['in']),\
                      'cal' :'%s/%s/s%di%d/cal/spc00000.OSA2'  % (fdir_field, lc, intTime['si'], intTime['in'])}

        which = lc.split('_')[0]
        filename_tag = '%s_%s' % (filename_tag0, lc.replace('_', '-'))
        pri_lamp_tag = 'f-%s' % (os.path.basename(fdir_pri)).lower()
        fname_cal = cdata_rad_resp(fnames_pri=fnames_pri, fnames_tra=fnames_tra, fnames_sec=fnames_sec, filename_tag=filename_tag, which=which, wvl_join=950.0, wvl_start=350.0, wvl_end=2200.0, intTime=intTime, pri_lamp_tag=pri_lamp_tag)

        if plot:
            plot_rad_resp_camp2ex(fname_cal)


def test():

    fnames = {
            'dark':'data/20221208_ssfr-lasp_pri-cal/20221208_CALIBRATION_75_150/20221208_spc00001.SKS',\
            'cal' :'data/20221208_ssfr-lasp_pri-cal/20221208_CALIBRATION_75_150/20221208_spc00002.SKS',
            }
    int_time = {'si':75, 'in':150}
    # ssfr.vis.plot_ssfr_raw(fnames['dark'])
    # ssfr.vis.plot_ssfr_raw(fnames['cal'])

    # fnames = {
    #         'dark':'data/20221208_ssfr-lasp_pri-cal/20221208_CALIBRATION_250_500/20221208_spc00001.SKS',\
    #         'cal' :'data/20221208_ssfr-lasp_pri-cal/20221208_CALIBRATION_250_500/20221208_spc00002.SKS',
    #         }
    # int_time = {'si':250, 'in':500}

    ssfr.vis.plot_ssfr_raw(fnames['dark'])
    ssfr.vis.plot_ssfr_raw(fnames['cal'])
    sys.exit()


    # resp = ssfr.cal.cdata_rad_resp(
    #         fnames,
    #         which_ssfr='lasp',
    #         which_lc='zenith',
    #         int_time=int_time
    #         )

    data0 = ssfr.lasp_ssfr.read_ssfr_raw(fnames['cal'], verbose=False)

    data1 = ssfr.lasp_ssfr.read_ssfr_raw(fnames['dark'], verbose=False)

    # print(data0['shutter'])
    # print(data1['shutter'])
    # print(data0['jday_cRIO'])
    # print(data1['jday_cRIO'])
    print(ssfr.util.jday_to_dtime(data0['jday_cRIO'][-1]))
    print(ssfr.util.jday_to_dtime(data0['jday_ARINC'][-1]))
    # print(data1['jday_cRIO'])
    # print(np.unique(data0['int_time'][:, 0]))
    # print(np.unique(data0['int_time'][:, 1]))
    # print(np.unique(data0['int_time'][:, 2]))
    # print(np.unique(data0['int_time'][:, 3]))

    # print(np.unique(data1['int_time'][:, 0]))
    # print(np.unique(data1['int_time'][:, 1]))
    # print(np.unique(data1['int_time'][:, 2]))
    # print(np.unique(data1['int_time'][:, 3]))

    pass


if __name__ == '__main__':

    test()
