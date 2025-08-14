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
mpl.use('Agg')

import ssfr




def rad_cal(
        fdir_pri,
        fdir_tra,
        fdir_sec=None,
        spec_reverse=False,
        fdir_out=None
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
    print(vars(ssfr_).keys())
    for i in range(ssfr_.Ndset):
        dset_tag = 'dset%d' % i
        # dset_ = getattr(ssfr_, dset_tag)
        int_time = ssfr_.dset_info[dset_tag]

        if len(tags_pri) == 7:
            cal_tag = '%s_%s' % (tags_pri[0], tags_pri[4])
        elif len(tags_pri) == 8:
            cal_tag = '%s_%s_%s' % (tags_pri[0], tags_pri[4], tags_pri[7])

        if len(tags_tra) == 7:
            cal_tag = '%s|%s_%s' % (cal_tag, tags_tra[0], tags_tra[4])
        elif len(tags_tra) == 8:
            cal_tag = '%s|%s_%s_%s' % (cal_tag, tags_tra[0], tags_tra[4], tags_tra[7])

        filename_tag = '%s|%s|%s' % (cal_tag, date_today_s, dset_tag)

        ssfr.cal.cdata_rad_resp(fnames_pri=fnames_pri, fnames_tra=fnames_tra, fnames_sec=fnames_sec, which_ssfr='lasp|%s' % ssfr_tag, which_lc=lc_tag, int_time=int_time, which_lamp=tags_pri[4], filename_tag=filename_tag, verbose=True, spec_reverse=spec_reverse, fdir_out=fdir_out)


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


def main_calibration_20250812(fdir_out):

    """
    Notes:
        irradiance setup:
            SSFR-A (Alvin)
              - nadir : LC6 + stainless steel cased fiber
              - zenith: LC4 + black plastic cased fiber
    """

    # radiometric calibration
    #/----------------------------------------------------------------------------\#
    fdirs_pri = sorted(glob.glob('data/rad-cal/2025-08-12/2025-08-12*SSFR-A*pri-cal*'))
    print(fdirs_pri)
    for fdir_pri in fdirs_pri:

        tags_pri = os.path.basename(fdir_pri).split('_')

        date_cal_pri = datetime.datetime.strptime(tags_pri[0], '%Y-%m-%d')

        if date_cal_pri == datetime.datetime(2025, 8, 12):
            fdirs_tra = sorted(glob.glob('data/rad-cal/2025-08-12/2025-08-12*%s*%s*transfer*%s*%s*' % (tags_pri[1], tags_pri[2], tags_pri[5], tags_pri[6])))
            print(fdirs_tra)

            if len(fdirs_tra) >= 1:
                for fdir_tra in fdirs_tra:
                    print('='*50)
                    print(fdir_pri)
                    print(fdir_tra)
                    if (fdir_tra.split('_')[-1][:4] != 'spec'):
                        rad_cal(fdir_pri, fdir_tra, spec_reverse=False, fdir_out=fdir_out)
                    else:
                        rad_cal(fdir_pri, fdir_tra, spec_reverse=True, fdir_out=fdir_out)
                    print('='*50)
                    print()

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
        ssfr_tag='ssfr-b',
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
            ax1.set_ylim((0, 600))
        else:
            ax1.set_ylim((0, 800))
        plt.legend(fontsize=12)
        # fig.suptitle('Field Calibration (%s|%s|%s|%s)' % (ssfr_tag.upper(), lc_tag.upper(), tags[-2].upper(), tags[-1].upper()), fontsize=24)


        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        # fig.savefig('%s_%s_%s_%s_%s.png' % (_metadata['Function'], ssfr_tag, lc_tag, tags[-2], tags[-1]), bbox_inches='tight', metadata=_metadata)
        fig.savefig('%s_%s_%s.png' % (_metadata['Function'], ssfr_tag, lc_tag), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#

def plot_radiometric_response_20250812(
        ssfr_tag='ssfr-a',
        lc_tag='zen',
        ):

    # tag = '%s|%s|si-%3.3d|in-%3.3d' % (ssfr_tag, lc_tag, int_time['si'], int_time['in'])
    # fnames = sorted(glob.glob('processed/2025-08-12/*%s*.h5' % tag))
    fnames = sorted(glob.glob('processed/2025-08-12/*.h5'))
    #. exampe fname: 2025-08-12_lamp-1324_postdeploymentresurgery|2025-08-12_lamp-150c_postdeploymentresurgery|2025-08-12_lamp-150c_postdeploymentresurgery|2025-08-14_processed-for-arcsix|rad-resp|lasp|ssfr-a|zen|si-240|in-350.h5

    # colors = plt.cm.jet(np.linspace(0.0, 1.0, len(fnames)))

    # figure
    #/----------------------------------------------------------------------------\#
    colors = ['violet', 'blue', 'red', 'teal', 'orange', 'navy']
    colors = plt.cm.rainbow(np.linspace(0.0, 1.0, len(fnames)))
    print(len(fnames))
    plt.close('all')
    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(111)

    count = 0
    for i, fname in enumerate(fnames):
        if ('fiber180' in fname) or (ssfr_tag not in fname):
            continue

        tags = os.path.basename(fname).replace('.h5', '').split('|')
        f = h5py.File(fname, 'r')
        wvl = f['wvl'][...]
        sec_resp = f['sec_resp'][...]
        pri_resp = f['pri_resp'][...]
        tra_resp = f['transfer'][...]
        print(fname, np.nanmean(pri_resp[wvl > 2000]), np.nanmean(tra_resp[wvl > 2000]))
        f.close()

        if count == 0:
            ax1.plot(wvl, pri_resp, marker='o', markersize=2, lw=0.5, label='%s | %s | %s (Primary)' % (tags[0], tags[-2], tags[-1]), color=colors[i])

        # ax1.plot(wvl, tra_resp, marker='d', markersize=5, ls='--', lw=0.5, label='%s | %s | %s (Transfer)' % (tags[0], tags[-2], tags[-1]), color=colors[i])
        ax1.plot(wvl, sec_resp, marker='^', markersize=5, ls='--', lw=0.5, label='%s | %s | %s (Secondary)' % (tags[0], tags[-2], tags[-1]), color=colors[i])

        count += 1

    ax1.set_title('Radiometric Response (%s|%s)' % (ssfr_tag.upper(), lc_tag.upper()), fontsize=20)
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Response [$counts/(W m^{-2} nm^{-1} s)$]')
    ax1.set_xlim((200, 2400))
    if lc_tag == 'zen':
        ax1.set_ylim((0, 250))
    else:
        ax1.set_ylim((0, 800))
    plt.legend(fontsize=12)
    # fig.suptitle('Field Calibration (%s|%s|%s|%s)' % (ssfr_tag.upper(), lc_tag.upper(), tags[-2].upper(), tags[-1].upper()), fontsize=24)


    # save figure
    #/--------------------------------------------------------------\#
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    # fig.savefig('%s_%s_%s_%s_%s.png' % (_metadata['Function'], ssfr_tag, lc_tag, tags[-2], tags[-1]), bbox_inches='tight', metadata=_metadata)
    fig.savefig('%s_%s_%s.png' % (_metadata['Function'], ssfr_tag, lc_tag), bbox_inches='tight', metadata=_metadata)
    #\--------------------------------------------------------------/#
    #/----------------------------------------------------------------------------\#


def plot_radiometric_response_4panel_20250812(
        ssfr_tag='ssfr-a',
        lc_tag='zen',
        ):
    """
    Create a 4-panel plot of radiometric responses with different spectral regions:
    - Top left: Full spectrum (200-2400 nm)
    - Top right: Near UV region (300-400 nm)
    - Bottom left: VIS region (400-700 nm)
    - Bottom right: NIR/SWIR region (700-2400 nm)
    """

    fnames = sorted(glob.glob('processed/2025-08-12/*.h5'))

    if not fnames:
        print("No H5 files found in processed/2025-08-12/")
        return

    # Create 2x2 subplot layout
    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Radiometric Response Analysis ({ssfr_tag.upper()}|{lc_tag.upper()})',
                 fontsize=18, fontweight='bold')

    # Define spectral regions
    regions = {
        'full': {'xlim': (200, 2400), 'title': 'Full Spectrum', 'ax': axes[0,0]},
        'near_uv': {'xlim': (300, 400), 'title': 'Near UV Region', 'ax': axes[0,1]},
        'vis': {'xlim': (400, 700), 'title': 'VIS Region', 'ax': axes[1,0]},
        'nir_swir': {'xlim': (700, 2400), 'title': 'NIR/SWIR Region', 'ax': axes[1,1]}
    }

    # Generate colors for different files
    colors = plt.cm.rainbow(np.linspace(0.0, 1.0, len(fnames)))

    print(f"Found {len(fnames)} files")

    # Process each file
    count = 0
    for i, fname in enumerate(fnames):
        # Skip fiber180 files and files not matching the instrument
        if ('fiber180' in fname) or (ssfr_tag not in fname):
            continue

        try:
            tags = os.path.basename(fname).replace('.h5', '').split('|')

            with h5py.File(fname, 'r') as f:
                wvl = f['wvl'][...]
                sec_resp = f['sec_resp'][...]
                pri_resp = f['pri_resp'][...]
                tra_resp = f['transfer'][...]

                # Extract integration times from filename for labeling
                si_time = 'unknown'
                in_time = 'unknown'
                for tag in tags:
                    if 'si-' in tag:
                        si_time = tag
                    elif 'in-' in tag:
                        in_time = tag

                label_primary = f'{si_time}|{in_time} (Primary)'
                label_secondary = f'{si_time}|{in_time} (Secondary)'

                print(f"Processing: {os.path.basename(fname)}")
                print(f"  SWIR mean (>2000nm): Pri={np.nanmean(pri_resp[wvl > 2000]):.3f}, "
                      f"Tra={np.nanmean(tra_resp[wvl > 2000]):.3f}")

                # Plot on all panels
                for region_name, region_info in regions.items():
                    ax = region_info['ax']
                    xlim = region_info['xlim']

                    # Mask data for this spectral region
                    mask = (wvl >= xlim[0]) & (wvl <= xlim[1])

                    if np.any(mask):
                        wvl_region = wvl[mask]
                        pri_region = pri_resp[mask]
                        sec_region = sec_resp[mask]

                        # Plot primary response (only for first file since they are all the same to avoid clutter)
                        if count == 0:
                            ax.plot(wvl_region, pri_region, marker='o', markersize=2,
                                   linewidth=1.0, label=label_primary,
                                   color=colors[i], alpha=0.8)

                        # Plot secondary response for all files
                        ax.plot(wvl_region, sec_region, marker='^', markersize=3,
                               linestyle='--', linewidth=1.0, label=label_secondary,
                               color=colors[i], alpha=0.8)

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

        count += 1

    # Customize each panel
    for region_name, region_info in regions.items():
        ax = region_info['ax']
        xlim = region_info['xlim']
        title = region_info['title']

        ax.set_title(f'{title} ({xlim[0]}-{xlim[1]} nm)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Wavelength [nm]', fontsize=12)
        ax.set_ylabel('Response [$counts/(W m^{-2} nm^{-1} s)$]', fontsize=12)
        ax.set_xlim(xlim)

        # Set appropriate y-limits based on spectral region and light collector
        if region_name == 'full':
            if lc_tag == 'zen':
                ax.set_ylim((0, 250))
            else:
                ax.set_ylim((0, 800))
        elif region_name == 'near_uv':
            if lc_tag == 'zen':
                ax.set_ylim((0, 200))  # UV typically has lower response
            else:
                ax.set_ylim((0, 150))
        elif region_name == 'vis':
            if lc_tag == 'zen':
                ax.set_ylim((0, 250))
            else:
                ax.set_ylim((0, 800))
        elif region_name == 'nir_swir':
            if lc_tag == 'zen':
                ax.set_ylim((0, 250))  # NIR/SWIR medium response
            else:
                ax.set_ylim((0, 600))

        # Add Si/InGaAs transition line for relevant regions
        if xlim[0] <= 950 <= xlim[1]:
            ax.axvline(950, color='black', linestyle=':', alpha=0.7, linewidth=1.5,
                      label='Si/InGaAs transition')

        # Add spectral band boundaries for full spectrum
        # if region_name == 'full':
            # ax.axvline(400, color='purple', linestyle=':', alpha=0.5, linewidth=1, label='Near UV')
            # ax.axvline(700, color='orange', linestyle=':', alpha=0.5, linewidth=1, label='VIS/NIR')
            # ax.axvline(1400, color='brown', linestyle=':', alpha=0.5, linewidth=1, label='NIR/SWIR')

            # Add region labels
            # ax.text(300, ax.get_ylim()[1]*0.9, 'Near UV', fontsize=10, ha='center',
            #        bbox=dict(boxstyle='round', facecolor='purple', alpha=0.3))
            # ax.text(550, ax.get_ylim()[1]*0.9, 'VIS', fontsize=10, ha='center',
            #        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
            # ax.text(1050, ax.get_ylim()[1]*0.9, 'NIR', fontsize=10, ha='center',
            #        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
            # ax.text(1800, ax.get_ylim()[1]*0.9, 'SWIR', fontsize=10, ha='center',
            #        bbox=dict(boxstyle='round', facecolor='brown', alpha=0.3))

        # Add specific features for each region
        # elif region_name == 'vis':
        #     # Highlight important visible wavelengths
        #     ax.axvline(555, color='green', linestyle=':', alpha=0.5, linewidth=1, label='Green peak')
        # elif region_name == 'nir_swir':
        #     # Add important atmospheric windows
        #     ax.axvspan(700, 1300, alpha=0.1, color='green', label='Atmospheric window')
        #     ax.axvspan(1400, 1800, alpha=0.1, color='red', label='H₂O absorption')

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    # Add overall statistics and information
    # stats_text = "Analyzed {count} configurations (excluding fiber180)\n"
    # stats_text += "Instrument: {ssfr_tag.upper()}, Light Collector: {lc_tag.upper()}\n"
    # stats_text += "Integration times: Multiple configurations shown"

    # fig.text(0.02, 0.02, stats_text, fontsize=10,
    #          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Save figure
    plt.tight_layout()
    fig.subplots_adjust(top=0.93, bottom=0.1)

    _metadata = {
        'Computer': os.uname()[1],
        'Script': os.path.abspath(__file__),
        'Function': sys._getframe().f_code.co_name,
        'Date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    filename = f'{_metadata["Function"]}_{ssfr_tag}_{lc_tag}.png'
    fig.savefig(filename, bbox_inches='tight', metadata=_metadata, dpi=300)
    print(f"Saved: {filename}")

    plt.show()


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

    # Generate calibration data
    fdir_out = 'processed/2025-08-12/'
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)
    # main_calibration_20250812(fdir_out=fdir_out)

    # Generate all visualization plots
    plot_radiometric_response_20250812()
    plot_radiometric_response_4panel_20250812()

    # field_lamp_150c_consis_check_20240329(int_si=80)
    # field_lamp_150e_consis_check_20240329(int_si=80)
    # field_lamp_150c_consis_check_20240329(int_si=120)
    # field_lamp_150e_consis_check_20240329(int_si=120)

    # for lc_tag in ['zen', 'nad']:
    #     for int_time in [{'si':80, 'in':250}, {'si':120, 'in':350}]:
    #         field_calibration_check(lc_tag=lc_tag, int_time=int_time)
    pass
