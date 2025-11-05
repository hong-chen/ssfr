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

def main():
    
    parent_dir = '/Users/yuch8913/programming/ssfr_arcsix/ssfr/data/arcsix/cal/rad-cal/2025-11-04_hsr1'
    file = f'{parent_dir}/2025-11-04_hdr_new_hdr_cal_w_camhdr_camera_hdr/Total.txt'
    hdr_new_hdr_cal_w_camhdr_data = ssfr.lasp_hsr.read_hsr1(file)
    

    # file = f'{parent_dir}/2025-11-04_hdr_new_hdr_cal_wo_camhdr_camera_no_hdr/Total.txt'
    # hdr_new_hdr_cal_wo_camhdr_data = ssfr.lasp_hsr.read_hsr1(file)
    
    file = f'{parent_dir}/2025-11-04_hdr_old_cal_camera_hdr/Total.txt'
    hdr_old_cal_data = ssfr.lasp_hsr.read_hsr1(file)
    
    file = f'{parent_dir}/2025-11-04_i40g50_new_hdr_cal_w_camhdr_camera_hdr/Total.txt'
    i40g50_new_hdr_data = ssfr.lasp_hsr.read_hsr1(file)
    
    file = f'{parent_dir}/2025-11-04_i40g50_new_i40g50_cal_wo_camhdr_camera_no_hdr/Total.txt'
    i40g50_new_i40g50_cal_wo_camhdr_data = ssfr.lasp_hsr.read_hsr1(file)
    
    file = f'{parent_dir}/2025-11-04_i40g50_old_cal_camera_hdr/Total.txt'
    i40g50_old_cal_data = ssfr.lasp_hsr.read_hsr1(file)
    
    file = f'{parent_dir}/2025-10-27_int-40_cal-orig/Total.txt'
    i40_1027_cal_orig_data = ssfr.lasp_hsr.read_hsr1(file)
    
    file = f'{parent_dir}/2025-10-27_int-40_cal-new-hdr/Total.txt'
    i40_1027_cal_new_hdr_data = ssfr.lasp_hsr.read_hsr1(file)
    
    file = f'{parent_dir}/2025-10-27_int-40_cal-new-int-40/Total.txt'
    i40_1027_cal_new_i40_data = ssfr.lasp_hsr.read_hsr1(file)
    
    
    lamp_file = '/Users/yuch8913/programming/ssfr_arcsix/ssfr/data/arcsix/cal/rad-cal/2025-11-04_hsr1/hsr1_lamp_file_1324.txt'
    lamp_data = np.loadtxt(lamp_file, skiprows=1)
    lamp_wvl = lamp_data[:, 0]
    lamp_flux = lamp_data[:, 1]
    
    hsr1_wvl = hdr_new_hdr_cal_w_camhdr_data.data['wvl']
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(lamp_wvl, lamp_flux, label='Lamp file', color='k', linewidth=2)
    ax.plot(hsr1_wvl, hdr_new_hdr_cal_w_camhdr_data.data['flux'].mean(axis=0), label='HDR 11/04 (11/04 cal HDR)', linestyle='-', markersize=4)
    ax.plot(hsr1_wvl, hdr_old_cal_data.data['flux'].mean(axis=0), label='HDR 11/04 (old cal HDR)', linestyle='--', markersize=4)
    ax.plot(hsr1_wvl, i40g50_new_hdr_data.data['flux'].mean(axis=0), label='I40G50 11/04 (11/04 cal HDR)', linestyle='-.', markersize=4)
    ax.plot(hsr1_wvl, i40g50_old_cal_data.data['flux'].mean(axis=0), label='I40G50 11/04 (old cal HDR)', linestyle=':', markersize=4)
    ax.plot(hsr1_wvl, i40g50_new_i40g50_cal_wo_camhdr_data.data['flux'].mean(axis=0), label='I40G50 11/04 (11/04 cal I40G50)', linestyle='-', markersize=4, color='C4')
    ax.plot(hsr1_wvl, i40_1027_cal_orig_data.data['flux'].mean(axis=0), label='I40G50 10/27 (old cal)', linestyle='--', markersize=4, color='C5')
    ax.plot(hsr1_wvl, i40_1027_cal_new_hdr_data.data['flux'].mean(axis=0), label='I40G50 10/27 (10/27 cal HDR)', linestyle='-.', markersize=4, color='C6')
    ax.plot(hsr1_wvl, i40_1027_cal_new_i40_data.data['flux'].mean(axis=0), label='I40G50 10/27 (10/27 cal I40G50)', linestyle=':', markersize=4, color='C7')
    
    
    ax.set_xlabel('Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Flux (arb. units)', fontsize=14)
    ax.set_title('HSR1 Calibration Data Comparison', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid()
    fig.tight_layout()
    fig.savefig('hsr1_calibration_comparison.png', dpi=300)
    plt.show()    

if __name__ == '__main__':

    main()
