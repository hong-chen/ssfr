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
        fdir_sec=None,
        spec_reverse=False,
        ):

    # get calibration files of primary
    #/----------------------------------------------------------------------------\#
    # date_cal_s_pri, ssfr_tag_pri, lc_tag_pri, cal_tag_pri, lamp_tag_pri, si_int_tag_pri, in_int_tag_pri = os.path.basename(fdir_pri).split('_')
    tags_pri = os.path.basename(fdir_pri).split('_')
    fnames_pri_ = sorted(glob.glob('%s/*.SKS' % (fdir_pri)))
    fnames_pri = [fnames_pri_[-1]]
    if len(fnames_pri) > 1:
        msg = '\nWarning [rad_cal]: find more than one file for "%s", selected "%s" ...' % (fdir_pri, fnames_pri[0])
        warnings.warn(msg)
    #\----------------------------------------------------------------------------/#


    # get calibration files of transfer
    #/----------------------------------------------------------------------------\#
    # date_cal_s_tra, ssfr_tag_tra, lc_tag_tra, cal_tag_tra, lamp_tag_tra, si_int_tag_tra, in_int_tag_tra = os.path.basename(fdir_tra).split('_')
    tags_tra = os.path.basename(fdir_tra).split('_')
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
    # date_cal_s_sec, ssfr_tag_sec, lc_tag_sec, cal_tag_sec, lamp_tag_sec, si_int_tag_sec, in_int_tag_sec = os.path.basename(fdir_sec).split('_')
    tags_sec = os.path.basename(fdir_sec).split('_')
    fnames_sec_ = sorted(glob.glob('%s/*.SKS' % (fdir_sec)))
    fnames_sec = [fnames_sec_[-1]]
    if len(fnames_sec) > 1:
        msg = '\nWarning [rad_cal]: find more than one file for "%s", selected "%s" ...' % (fdir_sec, fnames_sec[0])
        warnings.warn(msg)
    #\----------------------------------------------------------------------------/#

    if (tags_pri[1]==tags_tra[1]):
        ssfr_tag = tags_pri[1]
    if (tags_pri[2]==tags_tra[2]):
        lc_tag = tags_pri[2]

    date_today_s = datetime.datetime.now().strftime('%Y-%m-%d')

    ssfr_ = ssfr.lasp_ssfr.read_ssfr(fnames_pri)

    for i in range(ssfr_.Ndset):
        dset_tag = 'dset%d' % i
        dset_ = getattr(ssfr_, dset_tag)
        int_time = dset_['info']['int_time']

        if len(tags_pri) == 7:
            cal_tag = '%s_%s' % (tags_pri[0], tags_pri[4])
        elif len(tags_pri) == 8:
            cal_tag = '%s_%s_%s' % (tags_pri[0], tags_pri[4], tags_pri[7])

        if len(tags_tra) == 7:
            cal_tag = '%s|%s_%s' % (cal_tag, tags_tra[0], tags_tra[4])
        elif len(tags_tra) == 8:
            cal_tag = '%s|%s_%s_%s' % (cal_tag, tags_tra[0], tags_tra[4], tags_tra[7])

        filename_tag = '%s|%s|%s' % (cal_tag, date_today_s, dset_tag)

        ssfr.cal.cdata_rad_resp(fnames_pri=fnames_pri, fnames_tra=fnames_tra, fnames_sec=fnames_sec, which_ssfr='lasp|%s' % ssfr_tag, which_lc=lc_tag, int_time=int_time, which_lamp=tags_pri[4], filename_tag=filename_tag, verbose=True, spec_reverse=spec_reverse)


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


def field_lamp_150c_consis_check(int_si=120):

    fnames = sorted(glob.glob('*150c*si-%3.3d*.h5' % int_si))

    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(fnames)))

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))

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
        fig.suptitle('Field Lamp 150c (%s|%s)' % (tags[-2], tags[-1]), fontsize=24)


        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_%s_%s.png' % (_metadata['Function'], tags[-2], tags[-1]), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


def field_lamp_150e_consis_check(int_si=120):

    fnames = sorted(glob.glob('*150e*si-%3.3d*.h5' % int_si))

    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(fnames)))

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))

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
        fig.suptitle('Field Lamp 150c (%s|%s)' % (tags[-2], tags[-1]), fontsize=24)


        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_%s_%s.png' % (_metadata['Function'], tags[-2], tags[-1]), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#



def main_calibration_20240329():

    """
    Notes:
        irradiance setup:
            SSFR-A (Alvin)
              - nadir : LC6 + stainless steel cased fiber
              - zenith: LC4 + black plastic cased fiber
    """

    # radiometric calibration
    #/----------------------------------------------------------------------------\#
    fdirs_pri = sorted(glob.glob('/argus/field/arcsix/cal/rad-cal/2024-03-29*SSFR-A*pri-cal*si-080-120_in-250-350*'))
    for fdir_pri in fdirs_pri:

        tags_pri = os.path.basename(fdir_pri).split('_')

        date_cal_pri = datetime.datetime.strptime(tags_pri[0], '%Y-%m-%d')

        if date_cal_pri == datetime.datetime(2024, 3, 29):
            fdirs_tra = sorted(glob.glob('/argus/field/arcsix/cal/rad-cal/2024-03-29*%s*%s*transfer*%s*%s*' % (tags_pri[1], tags_pri[2], tags_pri[5], tags_pri[6])))

            if len(fdirs_tra) >= 1:
                for fdir_tra in fdirs_tra:
                    print('='*50)
                    print(fdir_pri)
                    print(fdir_tra)
                    if (fdir_tra.split('_')[-1][:4] != 'spec'):
                        rad_cal(fdir_pri, fdir_tra, spec_reverse=False)
                    else:
                        rad_cal(fdir_pri, fdir_tra, spec_reverse=True)
                    print('='*50)
                    print()
    #\----------------------------------------------------------------------------/#


def field_lamp_150c_consis_check_20240329(int_si=120):

    fnames = sorted(glob.glob('2024-03-29*150c*si-%3.3d*.h5' % int_si))

    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(fnames)))

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))

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
        plt.legend(fontsize=6)
        fig.suptitle('Field Lamp 150c (%s|%s)' % (tags[-2], tags[-1]), fontsize=24)


        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_%s_%s.png' % (_metadata['Function'], tags[-2], tags[-1]), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


def field_lamp_150e_consis_check_20240329(int_si=120):

    fnames = sorted(glob.glob('2024-03-29*150e*si-%3.3d*.h5' % int_si))

    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(fnames)))

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))

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
        plt.legend(fontsize=6)
        fig.suptitle('Field Lamp 150e (%s|%s)' % (tags[-2], tags[-1]), fontsize=24)


        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_%s_%s.png' % (_metadata['Function'], tags[-2], tags[-1]), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


def field_calibration_check(
        ssfr_tag='ssfr-a',
        lc_tag='zen',
        int_time={'si':80.0, 'in':250.0},
        ):


    tag = '%s|%s|si-%3.3d|in-%3.3d' % (ssfr_tag, lc_tag, int_time['si'], int_time['in'])
    fnames = sorted(glob.glob('../examples/data/arcsix/cal/rad-cal/*pituffik*%s*.h5' % tag))

    # colors = plt.cm.jet(np.linspace(0.0, 1.0, len(fnames)))

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(16, 8))

        ax1 = fig.add_subplot(111)

        for i, fname in enumerate(fnames):
            tags = os.path.basename(fname).replace('.h5', '').split('|')
            f = h5py.File(fname, 'r')
            wvl = f['wvl'][...]
            sec_resp = f['sec_resp'][...]
            pri_resp = f['pri_resp'][...]
            f.close()

            if i == 0:
                ax1.plot(wvl, pri_resp, marker='o', markersize=2, lw=0.5, label='%s (Primary)' % tags[0])

            ax1.plot(wvl, sec_resp, marker='o', markersize=2, lw=0.5, label='%s (Secondary)' % tags[2])

        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Irradiance [$W m^{-2} nm^{-1}$]')
        ax1.set_xlim((200, 2400))
        if lc_tag == 'zen':
            ax1.set_ylim((0, 300))
        else:
            ax1.set_ylim((0, 600))
        plt.legend(fontsize=12)
        fig.suptitle('Field Calibration (%s|%s|%s|%s)' % (ssfr_tag.upper(), lc_tag.upper(), tags[-2].upper(), tags[-1].upper()), fontsize=24)


        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_%s_%s_%s_%s.png' % (_metadata['Function'], ssfr_tag, lc_tag, tags[-2], tags[-1]), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


if __name__ == '__main__':

    # fig_belana_darks_si()
    # fig_alvin_darks_si()
    # fnames = sorted(glob.glob('data/*cos-resp*.h5'))
    # for fname in [fnames[-1]]:
    #     fig_cos_resp(fname)

    # main_calibration()

    # field_lamp_150c_consis_check(int_si=80)
    # field_lamp_150e_consis_check(int_si=80)
    # field_lamp_150c_consis_check(int_si=120)
    # field_lamp_150e_consis_check(int_si=120)


    # main_calibration_20240329()
    # field_lamp_150c_consis_check_20240329(int_si=80)
    # field_lamp_150e_consis_check_20240329(int_si=80)
    # field_lamp_150c_consis_check_20240329(int_si=120)
    # field_lamp_150e_consis_check_20240329(int_si=120)

    for lc_tag in ['zen', 'nad']:
        for int_time in [{'si':80, 'in':250}, {'si':120, 'in':350}]:
            field_calibration_check(lc_tag=lc_tag, int_time=int_time)
    pass
