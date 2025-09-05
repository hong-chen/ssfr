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
            fname_fig = f'{_metadata_['Function']}_{i}.png'
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
        fname_fig = f'{_metadata_['Function']}.png'
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
        fname_fig = f'{_metadata_['Function']}.png'
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#


if __name__ == '__main__':

    hsr1_rad_cal_raw_20250903()
    # hsr1_rad_cal_new_20250903()
    pass
