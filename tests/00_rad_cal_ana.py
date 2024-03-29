import os
import sys
import glob
import datetime
import warnings
import h5py
import numpy as np
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




def rad_cal(
        fdir_pri,
        fdir_tra,
        fdir_sec=None
        ):

    # get calibration files of primary
    #/----------------------------------------------------------------------------\#
    date_cal_s_pri, ssfr_tag_pri, lc_tag_pri, cal_tag_pri, lamp_tag_pri, si_int_tag_pri, in_int_tag_pri = os.path.basename(fdir_pri).split('_')
    fnames_pri_ = sorted(glob.glob('%s/*.SKS' % (fdir_pri)))
    fnames_pri = [fnames_pri_[-1]]
    if len(fnames_pri) > 1:
        msg = '\nWarning [rad_cal]: find more than one file for "%s", selected "%s" ...' % (fdir_pri, fnames_pri[0])
        warnings.warn(msg)
    #\----------------------------------------------------------------------------/#


    # get calibration files of transfer
    #/----------------------------------------------------------------------------\#
    date_cal_s_tra, ssfr_tag_tra, lc_tag_tra, cal_tag_tra, lamp_tag_tra, si_int_tag_tra, in_int_tag_tra = os.path.basename(fdir_tra).split('_')
    fnames_tra_ = sorted(glob.glob('%s/*.SKS' % (fdir_tra)))
    fnames_tra = [fnames_tra_[-1]]
    if len(fnames_tra) > 1:
        msg = '\nWarning [rad_cal]: find more than one file for "%s", selected "%s" ...' % (fdir_tra, fnames_tra[0])
        warnings.warn(msg)
    #\----------------------------------------------------------------------------/#


    # placeholder for calibration files of transfer
    #/----------------------------------------------------------------------------\#
    if fdir_sec is None:
        fdir_sec = fdir_tra
    date_cal_s_sec, ssfr_tag_sec, lc_tag_sec, cal_tag_sec, lamp_tag_sec, si_int_tag_sec, in_int_tag_sec = os.path.basename(fdir_sec).split('_')
    fnames_sec_ = sorted(glob.glob('%s/*.SKS' % (fdir_sec)))
    fnames_sec = [fnames_sec_[-1]]
    if len(fnames_sec) > 1:
        msg = '\nWarning [rad_cal]: find more than one file for "%s", selected "%s" ...' % (fdir_sec, fnames_sec[0])
        warnings.warn(msg)
    #\----------------------------------------------------------------------------/#

    if (ssfr_tag_pri==ssfr_tag_tra):
        ssfr_tag = ssfr_tag_pri
    if (lc_tag_pri==lc_tag_tra):
        lc_tag = lc_tag_pri

    date_today_s = datetime.datetime.now().strftime('%Y-%m-%d')

    ssfr_ = ssfr.lasp_ssfr.read_ssfr(fnames_pri)

    for i in range(ssfr_.Ndset):
        dset_tag = 'dset%d' % i
        dset_ = getattr(ssfr_, dset_tag)
        int_time = dset_['info']['int_time']

        cal_tag = '%s_%s|%s_%s|%s_%s' % (date_cal_s_pri, lamp_tag_pri, date_cal_s_tra, lamp_tag_tra, date_cal_s_sec, lamp_tag_sec)
        filename_tag = '%s|%s|%s' % (cal_tag, date_today_s, dset_tag)

        ssfr.cal.cdata_rad_resp(fnames_pri=fnames_pri, fnames_tra=fnames_tra, fnames_sec=fnames_sec, which_ssfr='lasp|%s' % ssfr_tag, which_lc=lc_tag, int_time=int_time, which_lamp=lamp_tag_pri, filename_tag=filename_tag, verbose=False)


def main_calibration():

    """
    Notes:
        irradiance setup:
            SSFR-A (Alvin)
              - nadir : LC6 + stainless steel cased fiber
              - zenith: LC4 + black plastic cased fiber
    """

    # radiometric calibration
    #/----------------------------------------------------------------------------\#
    fdirs_pri = sorted(glob.glob('/argus/field/arcsix/cal/rad-cal/*SSFR-A*pri-cal*si-080-120_in-250-350*'))
    for fdir_pri in fdirs_pri:
        date_cal_s_pri, ssfr_tag_pri, lc_tag_pri, cal_tag_pri, lamp_tag_pri, si_int_tag_pri, in_int_tag_pri = os.path.basename(fdir_pri).split('_')

        date_cal_pri = datetime.datetime.strptime(date_cal_s_pri, '%Y-%m-%d')

        if date_cal_pri >= datetime.datetime(2024, 3, 1):
            fdirs_tra = sorted(glob.glob('/argus/field/arcsix/cal/rad-cal/*%s*%s*transfer*%s*%s*' % (ssfr_tag_pri, lc_tag_pri, si_int_tag_pri, in_int_tag_pri)))

            if len(fdirs_tra) >= 1:
                for fdir_tra in fdirs_tra:
                    print('='*50)
                    print(fdir_pri)
                    print(fdir_tra)
                    rad_cal(fdir_pri, fdir_tra)
                    print('='*50)
                    print()
    #\----------------------------------------------------------------------------/#


def field_lamp_150c_consis_check():

    fnames = sorted(glob.glob('*150c*si-080*.h5'))

    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(fnames)))

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        fig.suptitle('Field Lamp 150c', fontsize=24)

        ax1 = fig.add_subplot(111)

        for i, fname in enumerate(fnames):
            tags = os.path.basename(fname).replace('.h5', '').split('|')
            f = h5py.File(fname, 'r')
            wvl = f['wvl'][...]
            transfer = f['transfer'][...]
            f.close()

            ax1.scatter(wvl, transfer, s=6, lw=0.0, color=colors[i, ...], label='%s|%s|%s|%s' % (tags[7], tags[8], tags[1], tags[0]))

        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Irradiance [$W m^{-2} nm^{-1}$]')
        ax1.set_xlim((350, 2150))
        ax1.set_ylim((0, 0.35))
        plt.legend(fontsize=10)


        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % (_metadata['Function']), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


def field_lamp_150e_consis_check():

    fnames = sorted(glob.glob('*150e*si-080*.h5'))

    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(fnames)))

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        fig.suptitle('Field Lamp 150e', fontsize=24)

        ax1 = fig.add_subplot(111)

        for i, fname in enumerate(fnames):
            tags = os.path.basename(fname).replace('.h5', '').split('|')
            f = h5py.File(fname, 'r')
            wvl = f['wvl'][...]
            transfer = f['transfer'][...]
            f.close()

            ax1.scatter(wvl, transfer, s=6, lw=0.0, color=colors[i, ...], label='%s|%s|%s|%s' % (tags[7], tags[8], tags[1], tags[0]))

        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Irradiance [$W m^{-2} nm^{-1}$]')
        ax1.set_xlim((350, 2150))
        ax1.set_ylim((0, 0.35))
        plt.legend(fontsize=10)


        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % (_metadata['Function']), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


if __name__ == '__main__':

    # fig_belana_darks_si()
    # fig_alvin_darks_si()
    # fnames = sorted(glob.glob('data/*cos-resp*.h5'))
    # for fname in [fnames[-1]]:
    #     fig_cos_resp(fname)

    # main_calibration()

    # for fname in fnames:
    #     field_lamp_consis_check(fname)

    # field_lamp_150c_consis_check()
    field_lamp_150e_consis_check()
    pass
