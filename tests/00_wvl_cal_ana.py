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




def p_to_lambda(p_, C0, C1, C2, C3, C4):
    return C0 + C1*p_ + C2*p_**2 + C3*p_**3 + C4*p_**4

def p_to_lambda_3rd(p_, C0, C1, C2, C3):
    return C0 + C1*p_ + C2*p_**2 + C3*p_**3 

def main():
    wvl_cal_dir = '/Users/yuch8913/programming/ssfr_arcsix/ssfr/output/wvl-cal'
    
    # glob txt files
    txt_files = sorted(glob.glob(f'{wvl_cal_dir}/*.txt'))
    print(f'Found {len(txt_files)} txt files in {wvl_cal_dir}')
    
    # coef_array = np.zeros((len(txt_files), 5))
    wvl_array = np.zeros((len(txt_files), 256))
    intru_list = []
    
    for i, txt_file in enumerate(txt_files):
        instrument_tag = os.path.basename(txt_file).split('|')[1]
        spec_tag = os.path.basename(txt_file).split('|')[2]
        channel_tag = os.path.basename(txt_file).split('_')[-1].replace('.txt', '')
        
        final_tag = f'{instrument_tag}_{spec_tag}_{channel_tag}'
        
        # read txt file
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            # skip first line
            line = lines[1:]
            # '# 2216.7556556418813 -4.549476053175344 0.0005245120000055728 -1.3997217788607562e-05 1.0386522862932913e-08\n'
            line = line[0].replace('#', '').strip()
            # split by space
            line = line.split(' ')
            # convert to float
            line = [float(i) for i in line]
            line = np.array(line)
        
        intru_list.append(final_tag)
        # coef_array[i, :] = line
        
        print(f'{final_tag}: {line}')
        
        if line[-1] == 0 or line.shape[0] == 4:
            wvl_array[i, :] = p_to_lambda_3rd(np.arange(256), *line)
        else:
            wvl_array[i, :] = p_to_lambda(np.arange(256), *line)
            
        print(f'Wavelength range: {wvl_array[i, 0]:.2f} - {wvl_array[i, -1]:.2f} nm')
            
    # plot the wavelength solution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    intru_list = np.array(intru_list)
    si_select = ['Si' in intru for intru in intru_list]
    in_select = ['In' in intru for intru in intru_list]
    
    for i, intru_tag in enumerate(intru_list[si_select]):
        ax1.plot(np.arange(256), wvl_array[si_select, :][i], label=intru_tag.replace('_Si', ''))
    ax1.set_title('Wavelength - Si')
    ax1.set_xlabel('Pixel')
    ax1.set_ylabel('Wavelength (nm)')
    ax1.legend()
    
    for i, intru_tag in enumerate(intru_list[in_select]):
        ax2.plot(np.arange(256), wvl_array[in_select, :][i], label=intru_tag.replace('_InGaAs', ''))
    ax2.set_title('Wavelength - In')
    ax2.set_xlabel('Pixel')
    ax2.set_ylabel('Wavelength (nm)')
    ax2.legend()
    
    fig.tight_layout()
    plt.show()
    fig.savefig(f'{wvl_cal_dir}/wavelength_output.png', dpi=300)
    
    
        
            




if __name__ == '__main__':

    
    main()