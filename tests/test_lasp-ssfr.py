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

    fnames0 = {
            'dark':'data/20221208_ssfr-lasp_pri-cal/20221208_CALIBRATION_75_150/20221208_spc00001.SKS',\
            'cal' :'data/20221208_ssfr-lasp_pri-cal/20221208_CALIBRATION_75_150/20221208_spc00002.SKS',
            }
    int_time0 = {'si':75, 'in':150}

    fnames1 = {
            'dark':'data/20221208_ssfr-lasp_pri-cal/20221208_CALIBRATION_250_500/20221208_spc00001.SKS',\
            'cal' :'data/20221208_ssfr-lasp_pri-cal/20221208_CALIBRATION_250_500/20221208_spc00002.SKS',
            }
    int_time1 = {'si':250, 'in':500}

    ssfr.vis.quicklook_ssfr_raw(fnames0['dark'], extra_tag='INT-TIME-075-150_')
    ssfr.vis.quicklook_ssfr_raw(fnames0['cal'], extra_tag='INT-TIME-075-150_')
    ssfr.vis.quicklook_ssfr_raw(fnames1['dark'], extra_tag='INT-TIME-250-500_')
    ssfr.vis.quicklook_ssfr_raw(fnames1['cal'], extra_tag='INT-TIME-250-500_')

    data0_cal  = ssfr.lasp_ssfr.read_ssfr_raw(fnames0['cal'], verbose=False)
    data0_dark = ssfr.lasp_ssfr.read_ssfr_raw(fnames0['dark'], verbose=False)

    data1_cal  = ssfr.lasp_ssfr.read_ssfr_raw(fnames1['cal'], verbose=False)
    data1_dark = ssfr.lasp_ssfr.read_ssfr_raw(fnames1['dark'], verbose=False)


if __name__ == '__main__':

    test()
