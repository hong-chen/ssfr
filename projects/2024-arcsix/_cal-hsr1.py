import os
import sys
import glob
import datetime
import copy
import multiprocessing as mp
from collections import OrderedDict
# from tqdm import tqdm
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


def hsr1_rad_cal_20250903():

    fname_raw = 'data/arcsix/cal/rad-cal/2025-09-03_HSR1-A_pri-cal_lamp-wood_int-040_gain-050_premission/SpectrometerCalibration SN03 v2.txt'
    data_raw = np.genfromtxt(fname_raw, skip_header=3, delimiter='\t', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9))

    fname_new = 'data/arcsix/cal/rad-cal/2025-09-03_HSR1-A_pri-cal_lamp-1324_int-040_gain-050_postdeployment/SpectrometerCalibrationSN1324v1.txt'
    data_new = np.genfromtxt(fname_new, skip_header=3, delimiter='\t', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9))

    for i in range(1, 9):

        # figure
        #╭────────────────────────────────────────────────────────────────────────────╮#
        plot = True
        if plot:
            plt.close('all')
            fig = plt.figure(figsize=(8, 6))
            # plot1
            #╭──────────────────────────────────────────────────────────────╮#
            ax1 = fig.add_subplot(111)
            ax1.scatter(data_raw[:, 0], data_raw[:, i], s=4, c='k', lw=0.0)
            ax1.scatter(data_new[:, 0], data_new[:, i], s=4, c='r', lw=0.0)
            ax1.set_xlabel('Wavelength [nm]')
            ax1.set_ylabel('Response')
            ax1.set_title(f'HSR1-A Sensor {i}')
            #╰──────────────────────────────────────────────────────────────╯#
            patches_legend = [
                              mpatches.Patch(color='black' , label='Pre-mission'), \
                              mpatches.Patch(color='red'   , label='Post-mission'), \
                             ]
            ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)
            # save figure
            #╭──────────────────────────────────────────────────────────────╮#
            fig.subplots_adjust(hspace=0.35, wspace=0.35)
            _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
            fname_fig = f"{_metadata_['Function']}_{i}.png"
            plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
            #╰──────────────────────────────────────────────────────────────╯#
            plt.close(fig)
            plt.clf()
        #╰────────────────────────────────────────────────────────────────────────────╯#

def hsr1_rad_cal_raw_20250903():

    fname_raw = 'data/arcsix/cal/rad-cal/2025-09-03_HSR1-A_pri-cal_lamp-wood_int-040_gain-050_premission/SpectrometerCalibration SN03 v2.txt'
    data_raw = np.genfromtxt(fname_raw, skip_header=3, delimiter='\t', usecols=(2, 3, 4, 5, 6, 7, 8, 9))

    fname_new = 'data/arcsix/cal/rad-cal/2025-09-03_HSR1-A_pri-cal_lamp-1324_int-040_gain-050_postdeployment/SpectrometerCalibrationSN1324v1.txt'
    data_new = np.genfromtxt(fname_new, skip_header=3, delimiter='\t', usecols=(2, 3, 4, 5, 6, 7, 8, 9))

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        cs = ax1.imshow(data_raw.T, origin='lower', cmap='jet', zorder=0, aspect='auto', extent=[300, 1100, 1, 8], interpolation='none') #, extent=extent, vmin=0.0, vmax=0.5)

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        # cbar.set_label('', rotation=270, labelpad=4.0)
        # cbar.set_ticks([])
        # cax.axis('off')
        # ax1.scatter(data_raw_x, data_raw_y, s=2, c='k', lw=0.0)
        # ax1.set_xlim((0, 1))
        # ax1.set_ylim((0, 1))
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Sensor #')
        ax1.set_title('Pre-mission')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = f"{_metadata_['Function']}.png"
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def hsr1_rad_cal_new_20250903():

    fname_raw = 'data/arcsix/cal/rad-cal/2025-09-03_HSR1-A_pri-cal_lamp-wood_int-040_gain-050_premission/SpectrometerCalibration SN03 v2.txt'
    data_raw = np.genfromtxt(fname_raw, skip_header=3, delimiter='\t', usecols=(2, 3, 4, 5, 6, 7, 8, 9))

    fname_new = 'data/arcsix/cal/rad-cal/2025-09-03_HSR1-A_pri-cal_lamp-1324_int-040_gain-050_postdeployment/SpectrometerCalibrationSN1324v1.txt'
    data_new = np.genfromtxt(fname_new, skip_header=3, delimiter='\t', usecols=(2, 3, 4, 5, 6, 7, 8, 9))

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        cs = ax1.imshow(data_new.T, origin='lower', cmap='jet', zorder=0, aspect='auto', extent=[300, 1100, 1, 8], interpolation='none') #, extent=extent, vmin=0.0, vmax=0.5)
        # ax1.scatter(data_raw_x, data_raw_y, s=2, c='k', lw=0.0)
        # ax1.set_xlim((0, 1))
        # ax1.set_ylim((0, 1))
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Sensor #')
        ax1.set_title('Post-mission')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #╰──────────────────────────────────────────────────────────────╯#
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', '5%', pad='3%')
        cbar = fig.colorbar(cs, cax=cax)
        # cbar.set_label('', rotation=270, labelpad=4.0)
        # cbar.set_ticks([])
        # cax.axis('off')
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = f"{_metadata_['Function']}.png"
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def hsr1_flux_compare_20250905():

    # lamp data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname = f'{ssfr.common.fdir_data}/lamp/f-506c.dat'
    data_lamp_ = np.loadtxt(fname)
    logic_lamp = (data_lamp_[:, 0]>=300.0) & (data_lamp_[:, 0]<=1100.0)
    data_lamp0 = {
            'wvl': data_lamp_[logic_lamp, 0],
            'flux': data_lamp_[logic_lamp, 1]*0.01,
            }
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # lamp data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname = f'{ssfr.common.fdir_data}/lamp/f-1324.dat'
    data_lamp_ = np.loadtxt(fname)
    logic_lamp = (data_lamp_[:, 0]>=300.0) & (data_lamp_[:, 0]<=1100.0)
    data_lamp1 = {
            'wvl': data_lamp_[logic_lamp, 0],
            'flux': data_lamp_[logic_lamp, 1]*1.0e4,
            }
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # new configuration with 40/50 (integration time/gains) settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_tot = 'data/20250905_HSR1_cal-test/2025-09-05_0905lamp507calfile/Total.txt'
    data_hsr1_ = ssfr.lasp_hsr.read_hsr1(fname=fname_tot)

    data_hsr1_new0 = {
            'wvl': data_hsr1_.data['wvl'],
            'flux': np.nanmean(data_hsr1_.data['flux'], axis=0),
            'flux_std': np.nanstd(data_hsr1_.data['flux'], axis=0),
            }
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # new configuration with 40/50 (integration time/gains) settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_tot = 'data/20250905_HSR1_cal-test/2025-09-05_0903calfile/Total.txt'
    data_hsr1_ = ssfr.lasp_hsr.read_hsr1(fname=fname_tot)

    data_hsr1_new1 = {
            'wvl': data_hsr1_.data['wvl'],
            'flux': np.nanmean(data_hsr1_.data['flux'], axis=0),
            'flux_std': np.nanstd(data_hsr1_.data['flux'], axis=0),
            }
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # old configuration with 40/50 (integration time/gains) settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_tot = 'data/20250905_HSR1_cal-test/2025-09-05_oldSN03v2calfile_G50I40/Total.txt'
    data_hsr1_ = ssfr.lasp_hsr.read_hsr1(fname=fname_tot)

    data_hsr1_old0 = {
            'wvl': data_hsr1_.data['wvl'],
            'flux': np.nanmean(data_hsr1_.data['flux'], axis=0),
            'flux_std': np.nanstd(data_hsr1_.data['flux'], axis=0),
            }
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # old configuration with 50/50 (integration time/gains) settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_tot = 'data/20250905_HSR1_cal-test/2025-09-05_oldSN03v2calfile_G50I50/Total.txt'
    data_hsr1_ = ssfr.lasp_hsr.read_hsr1(fname=fname_tot)

    data_hsr1_old1 = {
            'wvl': data_hsr1_.data['wvl'],
            'flux': np.nanmean(data_hsr1_.data['flux'], axis=0),
            'flux_std': np.nanstd(data_hsr1_.data['flux'], axis=0),
            }
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # old configuration with 40/50 (integration time/gains) settings
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_tot = 'data/20250905_HSR1_cal-test/2025-09-05_calSN03G50I40/Total.txt'
    data_hsr1_ = ssfr.lasp_hsr.read_hsr1(fname=fname_tot)

    data_hsr1_old2 = {
            'wvl': data_hsr1_.data['wvl'],
            'flux': np.nanmean(data_hsr1_.data['flux'], axis=0),
            'flux_std': np.nanstd(data_hsr1_.data['flux'], axis=0),
            }
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        # fig.suptitle('Figure')
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        ax1.plot(data_lamp0['wvl'], data_lamp0['flux'], color='gray', lw=2.0)
        ax1.plot(data_lamp1['wvl'], data_lamp1['flux'], color='k', lw=2.0)

        ax1.fill_between(data_hsr1_new0['wvl'], data_hsr1_new0['flux']-data_hsr1_new0['flux_std'], data_hsr1_new0['flux']+data_hsr1_new0['flux_std'], color='red', lw=0.0, alpha=0.1)
        ax1.plot(data_hsr1_new0['wvl'], data_hsr1_new0['flux'], color='r', lw=1.5)

        ax1.fill_between(data_hsr1_new1['wvl'], data_hsr1_new1['flux']-data_hsr1_new1['flux_std'], data_hsr1_new1['flux']+data_hsr1_new1['flux_std'], color='orange', lw=0.0, alpha=0.1)
        ax1.plot(data_hsr1_new1['wvl'], data_hsr1_new1['flux'], color='orange', lw=1.5)

        ax1.fill_between(data_hsr1_old0['wvl'], data_hsr1_old0['flux']-data_hsr1_old0['flux_std'], data_hsr1_old0['flux']+data_hsr1_old0['flux_std'], color='blue', lw=0.0, alpha=0.1)
        ax1.plot(data_hsr1_old0['wvl'], data_hsr1_old0['flux'], color='b', lw=1.5)

        ax1.fill_between(data_hsr1_old1['wvl'], data_hsr1_old1['flux']-data_hsr1_old1['flux_std'], data_hsr1_old1['flux']+data_hsr1_old1['flux_std'], color='purple', lw=0.0, alpha=0.1)
        ax1.plot(data_hsr1_old1['wvl'], data_hsr1_old1['flux'], color='purple', lw=1.5)

        ax1.fill_between(data_hsr1_old2['wvl'], data_hsr1_old2['flux']-data_hsr1_old2['flux_std'], data_hsr1_old2['flux']+data_hsr1_old2['flux_std'], color='green', lw=0.0, alpha=0.1)
        ax1.plot(data_hsr1_old2['wvl'], data_hsr1_old2['flux'], color='green', lw=1.5)

        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Irradiance [$\\mathrm{W m^{-2} nm^{-1}}$]')
        # ax1.set_xlim((0, 1))
        # ax1.set_ylim((0, 1))
        # ax1.set_xlabel('X')
        # ax1.set_ylabel('Y')
        # ax1.set_title('Plot1')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #╰──────────────────────────────────────────────────────────────╯#
        patches_legend = [
                          mpatches.Patch(color='black', label='Lamp data (1324)'), \
                          mpatches.Patch(color='gray' , label='Lamp data (506c)'), \
                          mpatches.Patch(color='red'   , label='New use 506c (I40|G50)'), \
                          mpatches.Patch(color='orange', label='New use 1324 (I40|G50)'), \
                          mpatches.Patch(color='blue'  , label='Old use general (I40|G50)'), \
                          mpatches.Patch(color='purple' , label='Old use general (I50|G50)'), \
                          mpatches.Patch(color='green' , label='Old use specific (I40|G50)'), \
                         ]
        # ax1.legend(handles=patches_legend, bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=len(patches_legend), mode="expand", borderaxespad=0., frameon=False, handletextpad=0.2, fontsize=14)
        ax1.legend(handles=patches_legend, loc='upper left', fontsize=16)

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = f"{_metadata_['Function']}.png"
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def hsr1_rad_cal_20251027():
    fname_raw = 'data/arcsix/cal/rad-cal/2025-10-26_HSR1-B_pri-cal_lamp-wood_hdr_premission/SN01 SpectrometerCalibration 2023-07-10 x10 Std.txt'
    data_raw = np.genfromtxt(fname_raw, skip_header=4, delimiter='\t', usecols=(1, 2, 3, 4, 5, 6, 7, 8))

    fname_new = 'data/arcsix/cal/rad-cal/2025-10-27_HSR1-B_pri-cal_lamp-1324_hdr_postmission/hsr1-b_2025-10-27_lamp-1324_int-18_gain-50_hdr_n10.txt'
    data_new = np.genfromtxt(fname_new, skip_header=4, delimiter='\t', usecols=(1, 2, 3, 4, 5, 6, 7, 8))

    fname_new2 = 'data/arcsix/cal/rad-cal/2025-10-27_HSR1-B_pri-cal_lamp-1324_int-040_gain-050_postmission/hsr1-b_2025-10-27_lamp-1324_int-40_gain-50_n10.txt'
    data_new2 = np.genfromtxt(fname_new2, skip_header=4, delimiter='\t', usecols=(1, 2, 3, 4, 5, 6, 7, 8))
    
    fname_new3 = 'data/arcsix/cal/rad-cal/2025-10-27_HSR1-B_pri-cal_lamp-1324_int-040_gain-050_postmission/hsr1-b_2025-10-27_lamp-1324_int-18_gain-50_n10.txt'
    data_new3 = np.genfromtxt(fname_new3, skip_header=4, delimiter='\t', usecols=(1, 2, 3, 4, 5, 6, 7, 8))

    fname_new4 = 'data/arcsix/cal/rad-cal/2025-10-27_HSR1-B_pri-cal_lamp-1324_int-040_gain-050_postmission/hsr1-b_2025-10-27_lamp-1324_int-100_gain-50_n10.txt'
    data_new4 = np.genfromtxt(fname_new4, skip_header=4, delimiter='\t', usecols=(1, 2, 3, 4, 5, 6, 7, 8))

    fname_new5 = 'data/arcsix/cal/rad-cal/2025-10-27_HSR1-B_pri-cal_lamp-1324_int-040_gain-050_postmission/hsr1-b_2025-10-27_lamp-1324_int-1000_gain-50_n10.txt'
    data_new5 = np.genfromtxt(fname_new5, skip_header=4, delimiter='\t', usecols=(1, 2, 3, 4, 5, 6, 7, 8))

    for i in range(1, 8):

        # figure
        #╭────────────────────────────────────────────────────────────────────────────╮#
        plot = True
        if plot:
            plt.close('all')
            fig = plt.figure(figsize=(8, 6))
            # plot1
            #╭──────────────────────────────────────────────────────────────╮#
            ax1 = fig.add_subplot(111)
            ax1.plot(data_raw[:, 0], data_raw[:, i], c='blue')
            ax1.plot(data_new[:, 0], data_new[:, i], c='red')
            ax1.plot(data_new2[:, 0], data_new2[:, i], c='orange')
            ax1.plot(data_new3[:, 0], data_new3[:, i], c='green', ls='--')
            ax1.plot(data_new4[:, 0], data_new4[:, i], c='lightgreen', ls='--')
            ax1.plot(data_new5[:, 0], data_new5[:, i], c='gold', ls='--')
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax1.set_yscale('log')
            ax1.set_ylim((0.1, None))
            ax1.set_xlabel('Wavelength [nm]')
            ax1.set_ylabel('Response')
            ax1.set_title(f'HSR1-A Sensor {i}')
            #╰──────────────────────────────────────────────────────────────╯#
            patches_legend = [
                              mpatches.Patch(color='blue' , label='Pre-mission (HDR)'), \
                              mpatches.Patch(color='red'   , label='Post-mission (HDR)'), \
                              mpatches.Patch(color='orange'  , label='Post-mission (I=40, G=50)'), \
                                mpatches.Patch(color='green'  , label='Post-mission (I=18, G=50)'), \
                                mpatches.Patch(color='lightgreen'  , label='Post-mission (I=100, G=50)'), \
                                mpatches.Patch(color='gold'  , label='Post-mission (I=1000, G=50)'), \
                             ]
            ax1.legend(handles=patches_legend, loc='upper left', fontsize=16)
            # save figure
            #╭──────────────────────────────────────────────────────────────╮#
            fig.subplots_adjust(hspace=0.35, wspace=0.35)
            _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
            fname_fig = f"{_metadata_['Function']}_{i}.png"
            plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
            #╰──────────────────────────────────────────────────────────────╯#
            plt.close(fig)
            plt.clf()
        #╰────────────────────────────────────────────────────────────────────────────╯#

def hsr1_flux_compare_20251027(
        fname_tot_orig=None, \
        fname_tot_new=None, \
        fname_tot_new_int=None
):

    # lamp data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # fname = f'{ssfr.common.fdir_data}/lamp/f-506c.dat'
    # data_lamp_ = np.loadtxt(fname)
    # logic_lamp = (data_lamp_[:, 0]>=300.0) & (data_lamp_[:, 0]<=1100.0)
    # data_lamp0 = {
    #         'wvl': data_lamp_[logic_lamp, 0],
    #         'flux': data_lamp_[logic_lamp, 1]*0.01,
    #         }
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # lamp data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname = f'{ssfr.common.fdir_data}/lamp/f-1324.dat'
    data_lamp_ = np.loadtxt(fname)
    logic_lamp = (data_lamp_[:, 0]>=300.0) & (data_lamp_[:, 0]<=1100.0)
    data_lamp1 = {
            'wvl': data_lamp_[logic_lamp, 0],
            'flux': data_lamp_[logic_lamp, 1]*1.0e4,
            }
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # measurements with the original calibration file (HDR)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # fname_tot_orig = '/Users/kehi6101/Downloads/HSR1b_cal_20251027/2025-10-27_int-40_cal-orig/Total.txt'
    if fname_tot_orig is not None:
        data_hsr1_ = ssfr.lasp_hsr.read_hsr1(fname=fname_tot_orig)

        data_hsr1_old0 = {
                'wvl': data_hsr1_.data['wvl'],
                'flux': np.nanmean(data_hsr1_.data['flux'], axis=0),
                'flux_std': np.nanstd(data_hsr1_.data['flux'], axis=0),
                }
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # measurements with the updated calibration file (HDR)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # fname_tot_new = '/Users/kehi6101/Downloads/HSR1b_cal_20251027/2025-10-27_int-40_cal-new-hdr/Total.txt'
    if fname_tot_new is not None:
        data_hsr1_ = ssfr.lasp_hsr.read_hsr1(fname=fname_tot_new)

        data_hsr1_new0 = {
                'wvl': data_hsr1_.data['wvl'],
                'flux': np.nanmean(data_hsr1_.data['flux'], axis=0),
                'flux_std': np.nanstd(data_hsr1_.data['flux'], axis=0),
                }
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # Measurements with the updated calibration file (integration time/gains = 40/50)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # fname_tot_new_int = '/Users/kehi6101/Downloads/HSR1b_cal_20251027/2025-10-27_int-40_cal-new-int-40/Total.txt'
    if fname_tot_new_int is not None:
        data_hsr1_ = ssfr.lasp_hsr.read_hsr1(fname=fname_tot_new_int)

        data_hsr1_new1 = {
                'wvl': data_hsr1_.data['wvl'],
                'flux': np.nanmean(data_hsr1_.data['flux'], axis=0),
                'flux_std': np.nanstd(data_hsr1_.data['flux'], axis=0),
                }
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(12, 10))
        # fig.suptitle('Figure')
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(211)
        # ax1.plot(data_lamp0['wvl'], data_lamp0['flux'], color='gray', lw=2.0)
        ax1.plot(data_lamp1['wvl'], data_lamp1['flux'], color='k', lw=2.0)

        if fname_tot_orig is not None:
            ax1.fill_between(data_hsr1_old0['wvl'], data_hsr1_old0['flux']-data_hsr1_old0['flux_std'], data_hsr1_old0['flux']+data_hsr1_old0['flux_std'], color='blue', lw=0.0, alpha=0.1)
            ax1.plot(data_hsr1_old0['wvl'], data_hsr1_old0['flux'], color='blue', lw=1.5)

        if fname_tot_new is not None:
            ax1.fill_between(data_hsr1_new0['wvl'], data_hsr1_new0['flux']-data_hsr1_new0['flux_std'], data_hsr1_new0['flux']+data_hsr1_new0['flux_std'], color='red', lw=0.0, alpha=0.1)
            ax1.plot(data_hsr1_new0['wvl'], data_hsr1_new0['flux'], color='red', lw=1.5)

        if fname_tot_new_int is not None:
            ax1.fill_between(data_hsr1_new1['wvl'], data_hsr1_new1['flux']-data_hsr1_new1['flux_std'], data_hsr1_new1['flux']+data_hsr1_new1['flux_std'], color='orange', lw=0.0, alpha=0.1)
            ax1.plot(data_hsr1_new1['wvl'], data_hsr1_new1['flux'], color='orange', lw=1.5)

        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Irradiance [$\\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.set_xlim((300, 1100))
        ax1.set_ylim((0, None))
        # ax1.set_xlabel('X')
        # ax1.set_ylabel('Y')
        # ax1.set_title('Plot1')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #╰──────────────────────────────────────────────────────────────╯#
        color_list = ['black', 'gray', 'blue', 'red', 'orange']
        patches_legend = [
                          mpatches.Patch(color='black', label='Lamp data (1324)'), \
                        #   mpatches.Patch(color='gray' , label='Lamp data (506c)'), \
                        #   mpatches.Patch(color='blue'   , label='Original cal (HDR)'), \
                        #   mpatches.Patch(color='red', label='New Cal (HDR)'), \
                        #   mpatches.Patch(color='orange'  , label='New Cal (I40|G50)'), \
                         ]
        if fname_tot_orig is not None:
            patches_legend.append(mpatches.Patch(color='blue', label='Lamp measured with Original Cal'))
        if fname_tot_new is not None:
            patches_legend.append(mpatches.Patch(color='red', label='Lamp measured with New Cal'))
        if fname_tot_new_int is not None:
            patches_legend.append(mpatches.Patch(color='orange', label='Lamp measured with New Cal (I40|G50)'))
        # ax1.legend(handles=patches_legend, bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=len(patches_legend), mode="expand", borderaxespad=0., frameon=False, handletextpad=0.2, fontsize=14)
        ax1.legend(handles=patches_legend, loc='upper left', fontsize=16)

        if fname_tot_orig is not None:
            data_lamp1_flux = np.interp(data_hsr1_old0['wvl'], data_lamp1['wvl'], data_lamp1['flux'])

            wvl = data_hsr1_old0['wvl']
            ratio = data_lamp1_flux/data_hsr1_old0['flux']

            x_ = wvl[(wvl > 400) & (wvl < 800)]
            y_ = ratio[(wvl > 400) & (wvl < 800)]
            x, y = x_[~np.isnan(y_)], y_[~np.isnan(y_)]
            coefs = np.polyfit(x, y, 2)
            poly = np.poly1d(coefs)

            print('2nd order polynomial fit coefficients: %.3e, %.3e, %.4f' % (coefs[0], coefs[1], coefs[2]))

            wvl_1_st = 450
            wvl_1_en = 550
            wvl_2_st = 650
            wvl_2_en = 750

            ratio_1 = np.nanmean(ratio[(wvl > wvl_1_st) & (wvl < wvl_1_en)])
            ratio_2 = np.nanmean(ratio[(wvl > wvl_2_st) & (wvl < wvl_2_en)])
            poly_1 = np.nanmean(poly(wvl[(wvl > wvl_1_st) & (wvl < wvl_1_en)]))
            poly_2 = np.nanmean(poly(wvl[(wvl > wvl_2_st) & (wvl < wvl_2_en)]))

            ax2 = fig.add_subplot(212)
            ax2.plot(wvl, ratio, color='red', lw=1.5, label='Lamp truth / Lamp measured with Original Cal')
            ax2.plot(x, poly(x), color='blue', lw=1.5, ls='--', label='2nd order polynomial fit (%.3e, %.3e, %.4f)' % (coefs[0], coefs[1], coefs[2]))
            ax2.axhline(1.0, color='k', lw=0.5, ls='--')
            ax2.fill_between(wvl, 0.5, 1.5, where=(wvl > 400) & (wvl < 800), color='gray', alpha=0.1)
            ax2.text(0.98, 0.9, 'Data Spectral factor (%d-%d vs %d-%d): %.3f' % (wvl_1_st, wvl_1_en, wvl_2_st, wvl_2_en, ratio_1/ratio_2), transform=ax2.transAxes, fontsize=14, ha='right')
            ax2.text(0.98, 0.8, 'Fitted Spectral factor (%d-%d vs %d-%d): %.3f' % (wvl_1_st, wvl_1_en, wvl_2_st, wvl_2_en, poly_1/poly_2), transform=ax2.transAxes, fontsize=14, ha='right')
            ax2.set_xlabel('Wavelength [nm]')
            ax2.set_ylabel('Ratio')
            ax2.set_ylim((0.75, 1.25))
            ax2.set_xlim((300, 1100))
            ax2.legend(loc='lower right', fontsize=16)
            ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        else:
            pass

        fig.suptitle('Flux Comparison [%s]' % (fname_tot_orig.split('/')[-3] if fname_tot_orig else ''), fontsize=16)
        #╰──────────────────────────────────────────────────────────────╯#

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = f"{_metadata_['Function']}.png"
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def hsr1_rad_cal_20260331():

    fname_raw = '/Volumes/SONY_16GQX/2026-03-31_navyHSR/Calibration - Edited/Baumer 700007631221 SpectrometerCalibration 2024-03-21.txt'
    data_raw = np.genfromtxt(fname_raw, skip_header=4, delimiter='\t', usecols=(1, 2, 3, 4, 5, 6, 7, 8))

    fname_new = '/Volumes/SONY_16GQX/2026-03-31_navyHSR/Calibration - Edited/Baumer 700007631221 SpectrometerCalibration 2026-03-31_LASP.txt'
    data_new = np.genfromtxt(fname_new, skip_header=4, delimiter='\t', usecols=(1, 2, 3, 4, 5, 6, 7, 8))

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    for i in range(1, 8):
        ax1.plot(data_raw[:, 0], data_raw[:, i], c='blue', alpha=0.2)
        ax1.plot(data_new[:, 0], data_new[:, i], c='red', alpha=0.2)
    ax1.plot(data_raw[:, 0], np.nanmean(data_raw[:, 1:8], axis=1), c='blue', label=fname_raw.split(' ')[-1].split('.')[0])
    ax1.plot(data_new[:, 0], np.nanmean(data_new[:, 1:8], axis=1), c='red', label=fname_new.split(' ')[-1].split('.')[0])
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_yscale('log')
    ax1.set_xlim((300, 1100))
    ax1.set_ylim((0.1, None))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Response')
    ax1.set_title('Navy HSR1 Response')
    ax1.legend(loc='upper left', fontsize=16)

    # save figure
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    fname_fig = f"{_metadata_['Function']}.png"
    plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
    plt.show()
    plt.close(fig)
    plt.clf()

def hsr1_flux_compare_20260331():

    # lamp data
    fname = f'{ssfr.common.fdir_data}/lamp/f-1324.dat'
    data_lamp_ = np.loadtxt(fname)
    logic_lamp = (data_lamp_[:, 0]>=300.0) & (data_lamp_[:, 0]<=1100.0)
    data_lamp = {
            'wvl': data_lamp_[logic_lamp, 0],
            'flux': data_lamp_[logic_lamp, 1]*1.0e4,
            }
    # measurements with the old calibration file
    fname_tot = '/Volumes/SONY_16GQX/2026-03-31_navyHSR/2026-03-31_cal-old/Total.txt'
    data_hsr1_ = ssfr.lasp_hsr.read_hsr1(fname=fname_tot)
    data_hsr1_old = {
            'wvl': data_hsr1_.data['wvl'],
            'flux': np.nanmean(data_hsr1_.data['flux'], axis=0),
            'flux_std': np.nanstd(data_hsr1_.data['flux'], axis=0),
            }
    # measurements with the new calibration file
    fname_tot = '/Volumes/SONY_16GQX/2026-03-31_navyHSR/2026-03-31_cal-new/Total.txt'
    data_hsr1_ = ssfr.lasp_hsr.read_hsr1(fname=fname_tot)
    data_hsr1_new = {
            'wvl': data_hsr1_.data['wvl'],
            'flux': np.nanmean(data_hsr1_.data['flux'], axis=0),
            'flux_std': np.nanstd(data_hsr1_.data['flux'], axis=0),
            }
    
    # figure
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(211)
    ax1.plot(data_lamp['wvl'], data_lamp['flux'], color='k', lw=2.0, label='Lamp data (1324)')
    ax1.fill_between(data_hsr1_old['wvl'], data_hsr1_old['flux']-data_hsr1_old['flux_std'], data_hsr1_old['flux']+data_hsr1_old['flux_std'], color='blue', lw=0.0, alpha=0.1)
    ax1.plot(data_hsr1_old['wvl'], data_hsr1_old['flux'], color='blue', lw=1.5, label='Old Cal')
    ax1.fill_between(data_hsr1_new['wvl'], data_hsr1_new['flux']-data_hsr1_new['flux_std'], data_hsr1_new['flux']+data_hsr1_new['flux_std'], color='red', lw=0.0, alpha=0.1)
    ax1.plot(data_hsr1_new['wvl'], data_hsr1_new['flux'], color='red', lw=1.5, label='New Cal')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Irradiance [$\\mathrm{W m^{-2} nm^{-1}}$]')
    ax1.set_ylim((0, None))
    ax1.set_title('Navy HSR1 Flux Comparison')
    ax1.legend(loc='upper left', fontsize=16)

    data_lamp_flux = np.interp(data_hsr1_new['wvl'], data_lamp['wvl'], data_lamp['flux'])
    ax2 = fig.add_subplot(212)
    ax2.plot(data_hsr1_new['wvl'], data_hsr1_new['flux']/data_lamp_flux, color='red', lw=1.5, label='New Cal / Lamp truth')
    ax2.plot(data_hsr1_new['wvl'], data_hsr1_old['flux']/data_lamp_flux, color='blue', lw=1.5, label='Old Cal / Lamp truth')
    ax2.axhline(1.0, color='k', lw=0.5, ls='--')
    ax2.set_xlabel('Wavelength [nm]')
    ax2.set_ylabel('Ratio (Lamp / New Cal)')
    ax2.set_ylim((0.8, 1.2))
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper left', fontsize=16)
    # save figure
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    fname_fig = f"{_metadata_['Function']}.png"
    plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
    plt.show()
    plt.close(fig)
    plt.clf()


if __name__ == '__main__':

    # hsr1_rad_cal_raw_20250903()
    # hsr1_rad_cal_new_20250903()
    # hsr1_flux_compare_20250905()

    # hsr1_rad_cal_20251027()
    # hsr1_flux_compare_20251027()
    hsr1_flux_compare_20251027(
        fname_tot_orig='/Users/kehi6101/Downloads/HSR1a_test_20251022/2025-10-22_old_cal_I250/Total.txt', \
    )
    # hsr1_flux_compare_20251027(
    #     fname_tot_orig='/Users/kehi6101/Downloads/HSR1b_cal_20251027/2025-10-27_int-40_cal-orig/Total.txt', \
    # )
    # hsr1_flux_compare_20251027(
    #     fname_tot_orig='/Users/kehi6101/Downloads/HSR1b_cal_20251027/2025-10-27_int-40_cal-orig/Total.txt', \
    #     fname_tot_new='/Users/kehi6101/Downloads/HSR1b_cal_20251027/2025-10-27_int-40_cal-new-hdr/Total.txt', \
    #     fname_tot_new_int='/Users/kehi6101/Downloads/HSR1b_cal_20251027/2025-10-27_int-40_cal-new-int-40/Total.txt'
    # )

    # hsr1_rad_cal_20260331()
    # hsr1_flux_compare_20260331()

    pass
