import os
import sys
import copy
import datetime
import warnings
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav
import matplotlib.pyplot as plt


import ssfr



__all__ = [
        'rad_resp_corr',
        ]


def rad_resp_corr(fnames_resp_zen=None,
                  fnames_resp_nad=None,
                  which_ssfr='lasp|ssfr-a',
                  int_time={'si':80.0, 'in':250.0},
                  wvl_joint=950.0,
                  wvl_joint_range=20.0,
                  wvl_range=[350.0, 2200.0],
                  verbose=True,
                  ):

    # check SSFR spectrometer
    #╭────────────────────────────────────────────────────────────────────────────╮#
    which_ssfr = which_ssfr.lower()
    which_lab  = which_ssfr.split('|')[0]
    if which_lab == 'nasa':
        import ssfr.nasa_ssfr as ssfr_toolbox
    elif which_lab == 'lasp':
        import ssfr.lasp_ssfr as ssfr_toolbox
    else:
        msg = '\nError [rad_resp_corr]: <which_ssfr=> does not support <\'%s\'> (only supports <\'nasa|ssfr-6\'> or <\'lasp|ssfr-a\'> or <\'lasp|ssfr-b\'>).' % which_ssfr
        raise ValueError(msg)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    wvls = ssfr_toolbox.get_ssfr_wvl(which_ssfr)

    wvl_start = wvl_range[0]
    wvl_end   = wvl_range[-1]
    wvl_si_zen  = wvls['zen|si']
    wvl_in_zen  = wvls['zen|in']
    wvl_si_nad  = wvls['nad|si']
    wvl_in_nad  = wvls['nad|in']
    
    if (fnames_resp_zen is None) & (fnames_resp_nad is None):
        msg = '\nError [rad_resp_corr]: cannot proceed without zenith or nadir response files.'
        raise OSError(msg)

    pri_resp_out_zen = fnames_resp_zen.replace('.h5', '|pri_resp.pkl')
    transfer_out_zen = fnames_resp_zen.replace('.h5', '|transfer.pkl')
    sec_resp_out_zen = fnames_resp_zen.replace('.h5', '|sec_resp.pkl')

    pri_resp_out_nad = fnames_resp_nad.replace('.h5', '|pri_resp.pkl')
    transfer_out_nad = fnames_resp_nad.replace('.h5', '|transfer.pkl')
    sec_resp_out_nad = fnames_resp_nad.replace('.h5', '|sec_resp.pkl')
    
    # read in all responses
    import pickle as pkl
    with open(pri_resp_out_zen, 'rb') as f:
        pri_resp_zen = pkl.load(f)
    with open(transfer_out_zen, 'rb') as f:
        transfer_zen = pkl.load(f)
    with open(sec_resp_out_zen, 'rb') as f:
        sec_resp_zen = pkl.load(f)
    with open(pri_resp_out_nad, 'rb') as f:
        pri_resp_nad = pkl.load(f)
    with open(transfer_out_nad, 'rb') as f:
        transfer_nad = pkl.load(f)
    with open(sec_resp_out_nad, 'rb') as f:
        sec_resp_nad = pkl.load(f)
    with h5py.File(fnames_resp_zen, 'r') as f:
        transfer_zen_ori = f['transfer'][:]
    with h5py.File(fnames_resp_nad, 'r') as f:
        transfer_nad_ori = f['transfer'][:]
        


    pri_resp_si_zen = pri_resp_zen['zen|si']
    transfer_si_zen = transfer_zen['zen|si']
    sec_resp_si_zen = sec_resp_zen['zen|si']
    pri_resp_si_std_zen = pri_resp_zen['zen|si_std']
    transfer_si_std_zen = transfer_zen['zen|si_std']
    sec_resp_si_std_zen = sec_resp_zen['zen|si_std']
    pri_count_si_zen = pri_resp_zen['zen|si_count']
    transfer_count_si_zen = transfer_zen['zen|si_count']
    sec_count_si_zen = sec_resp_zen['zen|si_count']
    pri_count_si_std_zen = pri_resp_zen['zen|si_count_std']
    transfer_count_si_std_zen = transfer_zen['zen|si_count_std']
    sec_count_si_std_zen = sec_resp_zen['zen|si_count_std']
    pri_resp_in_zen = pri_resp_zen['zen|in']
    transfer_in_zen = transfer_zen['zen|in']
    sec_resp_in_zen = sec_resp_zen['zen|in']
    pri_resp_in_std_zen = pri_resp_zen['zen|in_std']
    transfer_in_std_zen = transfer_zen['zen|in_std']
    sec_resp_in_std_zen = sec_resp_zen['zen|in_std']
    pri_count_in_zen = pri_resp_zen['zen|in_count']
    transfer_count_in_zen = transfer_zen['zen|in_count']
    sec_count_in_zen = sec_resp_zen['zen|in_count']
    pri_count_in_std_zen = pri_resp_zen['zen|in_count_std']
    transfer_count_in_std_zen = transfer_zen['zen|in_count_std']
    sec_count_in_std_zen = sec_resp_zen['zen|in_count_std']
    
    

    pri_resp_si_nad = pri_resp_nad['nad|si']
    transfer_si_nad = transfer_nad['nad|si']
    sec_resp_si_nad = sec_resp_nad['nad|si']
    pri_resp_si_std_nad = pri_resp_nad['nad|si_std']
    transfer_si_std_nad = transfer_nad['nad|si_std']
    sec_resp_si_std_nad = sec_resp_nad['nad|si_std']
    pri_count_si_nad = pri_resp_nad['nad|si_count']
    transfer_count_si_nad = transfer_nad['nad|si_count']
    sec_count_si_nad = sec_resp_nad['nad|si_count']
    pri_count_si_std_nad = pri_resp_nad['nad|si_count_std']
    transfer_count_si_std_nad = transfer_nad['nad|si_count_std']
    sec_count_si_std_nad = sec_resp_nad['nad|si_count_std']
    pri_resp_in_nad = pri_resp_nad['nad|in']
    transfer_in_nad = transfer_nad['nad|in']
    sec_resp_in_nad = sec_resp_nad['nad|in']
    pri_resp_in_std_nad = pri_resp_nad['nad|in_std']
    transfer_in_std_nad = transfer_nad['nad|in_std']
    sec_resp_in_std_nad = sec_resp_nad['nad|in_std']
    pri_count_in_nad = pri_resp_nad['nad|in_count']
    transfer_count_in_nad = transfer_nad['nad|in_count']
    sec_count_in_nad = sec_resp_nad['nad|in_count']
    pri_count_in_std_nad = pri_resp_nad['nad|in_count_std']
    transfer_count_in_std_nad = transfer_nad['nad|in_count_std']
    sec_count_in_std_nad = sec_resp_nad['nad|in_count_std']

        
    # (1) nad-si and nad-in joint wavelength check
    wvl_start = wvl_joint - wvl_joint_range/2.0
    wvl_end   = wvl_joint + wvl_joint_range/2.0
    wvl_interp = np.arange(wvl_start, wvl_end+0.1, 0.1)
    f_pri_resp_si_nad = interpolate.interp1d(wvl_si_nad, pri_resp_si_nad, bounds_error=False, fill_value=np.nan)
    f_transfer_si_nad = interpolate.interp1d(wvl_si_nad, transfer_si_nad, bounds_error=False, fill_value=np.nan)
    pri_resp_si_interp_nad = f_pri_resp_si_nad(wvl_interp)
    transfer_si_interp_nad = f_transfer_si_nad(wvl_interp)
    f_pri_resp_in_nad = interpolate.interp1d(wvl_in_nad, pri_resp_in_nad, bounds_error=False, fill_value=np.nan)
    f_transfer_in_nad = interpolate.interp1d(wvl_in_nad, transfer_in_nad, bounds_error=False, fill_value=np.nan)
    pri_resp_in_interp_nad = f_pri_resp_in_nad(wvl_interp)
    transfer_in_interp_nad = f_transfer_in_nad(wvl_interp)
    f_pri_resp_si_nad_count = interpolate.interp1d(wvl_si_nad, pri_count_si_nad, bounds_error=False, fill_value=np.nan)
    f_transfer_si_nad_count = interpolate.interp1d(wvl_si_nad, transfer_count_si_nad, bounds_error=False, fill_value=np.nan)
    pri_count_si_interp_nad = f_pri_resp_si_nad_count(wvl_interp)
    transfer_count_si_interp_nad = f_transfer_si_nad_count(wvl_interp)
    f_pri_resp_in_nad_count = interpolate.interp1d(wvl_in_nad, pri_count_in_nad, bounds_error=False, fill_value=np.nan)
    f_transfer_in_nad_count = interpolate.interp1d(wvl_in_nad, transfer_count_in_nad, bounds_error=False, fill_value=np.nan)
    pri_count_in_interp_nad = f_pri_resp_in_nad_count(wvl_interp)
    transfer_count_in_interp_nad = f_transfer_in_nad_count(wvl_interp)
    tranfer_si_select_nad = np.nanmean(transfer_si_interp_nad)
    tranfer_in_select_nad = np.nanmean(transfer_in_interp_nad)
    
    transfer_in_nad_ori = transfer_in_nad.copy()
    
    
    scaling_factor_in_nad = tranfer_si_select_nad / tranfer_in_select_nad
    pri_resp_in_nad = pri_resp_in_nad / scaling_factor_in_nad
    transfer_in_nad = transfer_count_in_nad / int_time['nad|in'] / pri_resp_in_nad

    count_in_std_frac_nad = transfer_count_in_std_nad / transfer_count_in_nad
    pri_resp_in_std_frac    = pri_resp_in_std_nad    / pri_resp_in_nad
    transfer_in_std_nad = np.sqrt((count_in_std_frac_nad/int_time['nad|in']/pri_resp_in_nad)**2 + (transfer_count_in_nad/int_time['nad|in']/pri_resp_in_nad*pri_resp_in_std_frac)**2) * transfer_count_in_nad

    
    # 
    plt.close('all')
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(wvl_si_nad, transfer_si_nad, 'b-', label='NAD-SI (original)')
    ax.plot(wvl_in_nad, transfer_in_nad_ori, 'r-', label='NAD-IN (original)')
    ax.plot(wvl_in_nad, transfer_in_nad, '--', color='orange', label='NAD-IN (scaled)')
    ymin, ymax = ax.get_ylim()
    ax.fill_betweenx([ymin, ymax], wvl_start, wvl_end, color='gray', alpha=0.5, label='joint region')
    ax.set_ylim(0, ymax)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transfer flux (W m$^{-2}$ nm$^{-1}$)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('rad_resp_corr_nad_si_in.png', dpi=300)
    # plt.show()
    
    # sys.exit()
    
    
    # (2) nad-si and zen-si transfer check
    wvl_start = np.min((wvl_si_nad.min(), wvl_si_zen.min()))
    wvl_end   = np.max((wvl_si_nad.max(), wvl_si_zen.max()))
    wvl_interp = np.arange(wvl_start, wvl_end+0.1, 0.1)
    f_transfer_si_zen = interpolate.interp1d(wvl_si_zen, transfer_si_zen, bounds_error=False, fill_value=np.nan)
    f_transfer_si_nad = interpolate.interp1d(wvl_si_nad, transfer_si_nad, bounds_error=False, fill_value=np.nan)
    transfer_si_interp_zen = f_transfer_si_zen(wvl_interp)
    transfer_si_interp_nad = f_transfer_si_nad(wvl_interp)
    f_transfer_si_zen_count = interpolate.interp1d(wvl_si_zen, transfer_count_si_zen, bounds_error=False, fill_value=np.nan)
    f_transfer_si_nad_count = interpolate.interp1d(wvl_si_nad, transfer_count_si_nad, bounds_error=False, fill_value=np.nan)
    transfer_count_si_interp_zen = f_transfer_si_zen_count(wvl_interp)
    transfer_count_si_interp_nad = f_transfer_si_nad_count(wvl_interp)
    tranfer_si_select_zen = transfer_si_interp_zen
    tranfer_si_select_nad = transfer_si_interp_nad
    
    scaling_factor_si = tranfer_si_select_zen/tranfer_si_select_nad
    f_scaling_zen_si = interpolate.interp1d(wvl_interp, scaling_factor_si, bounds_error=False, fill_value='extrapolate')
    scaling_factor_si_zen = f_scaling_zen_si(wvl_si_zen)
    
    transfer_si_zen_ori = transfer_si_zen.copy()
    
    pri_resp_si_zen = pri_resp_si_zen * scaling_factor_si_zen
    transfer_si_zen = transfer_count_si_zen / int_time['zen|si'] / pri_resp_si_zen
    count_si_std_frac_zen = transfer_count_si_std_zen / transfer_count_si_zen
    pri_resp_si_std_frac    = pri_resp_si_std_zen    / pri_resp_si_zen
    transfer_si_std_zen = np.sqrt((count_si_std_frac_zen/int_time['zen|si']/pri_resp_si_zen)**2 + (transfer_count_si_zen/int_time['zen|si']/pri_resp_si_zen*pri_resp_si_std_frac)**2) * transfer_count_si_zen
    
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(wvl_si_nad, transfer_si_nad, 'b-', label='NAD-SI (original)')
    ax.plot(wvl_si_zen, transfer_si_zen_ori, 'r-', label='ZEN-SI (original)')
    ax.plot(wvl_si_zen, transfer_si_zen, '--', color='orange', label='ZEN-SI (scaled)')
    ymin, ymax = ax.get_ylim()
    # plt.fill_betweenx([ymin, ymax], wvl_start, wvl_end, color='gray', alpha=0.5, label='joint region')
    ax.set_ylim(0, ymax)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transfer flux (W m$^{-2}$ nm$^{-1}$)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('rad_resp_corr_zen_si_nad_si.png', dpi=300)
    # plt.show()
    
    # sys.exit()
    
    # (3) zen-si and zen-in transfer check
    wvl_start = wvl_joint - wvl_joint_range/2.0
    wvl_end   = wvl_joint + wvl_joint_range/2.0
    wvl_interp = np.arange(wvl_start, wvl_end+0.1, 0.1)
    f_transfer_si_zen = interpolate.interp1d(wvl_si_zen, transfer_si_zen, bounds_error=False, fill_value=np.nan)
    f_transfer_in_zen = interpolate.interp1d(wvl_in_zen, transfer_in_zen, bounds_error=False, fill_value=np.nan)
    transfer_si_interp_zen = f_transfer_si_zen(wvl_interp)
    transfer_in_interp_zen = f_transfer_in_zen(wvl_interp)
    f_transfer_si_zen_count = interpolate.interp1d(wvl_si_zen, transfer_count_si_zen, bounds_error=False, fill_value=np.nan)
    f_transfer_in_zen_count = interpolate.interp1d(wvl_in_zen, transfer_count_in_zen, bounds_error=False, fill_value=np.nan)
    transfer_count_si_interp_zen = f_transfer_si_zen_count(wvl_interp)
    transfer_count_in_interp_zen = f_transfer_in_zen_count(wvl_interp)
    tranfer_si_select_zen = np.nanmean(transfer_si_interp_zen)
    tranfer_in_select_zen = np.nanmean(transfer_in_interp_zen)
    
    transfer_in_zen_ori = transfer_in_zen.copy()
    
    scaling_factor_in_zen = tranfer_si_select_zen / tranfer_in_select_zen
    pri_resp_in_zen = pri_resp_in_zen / scaling_factor_in_zen
    transfer_in_zen = transfer_count_in_zen / int_time['zen|in'] / pri_resp_in_zen
    count_in_std_frac_zen = transfer_count_in_std_zen / transfer_count_in_zen
    pri_resp_in_std_frac    = pri_resp_in_std_zen    / pri_resp_in_zen
    transfer_in_std_zen = np.sqrt((count_in_std_frac_zen/int_time['zen|in']/pri_resp_in_zen)**2 + (transfer_count_in_zen/int_time['zen|in']/pri_resp_in_zen*pri_resp_in_std_frac)**2) * transfer_count_in_zen
    
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(wvl_si_zen, transfer_si_zen_ori, '-', color='cyan', label='ZEN-SI (original)')
    ax.plot(wvl_si_zen, transfer_si_zen, '--', color='blue', label='ZEN-SI (scaled)')
    ax.plot(wvl_in_zen, transfer_in_zen_ori, 'r-', label='ZEN-IN (original)')
    ax.plot(wvl_in_zen, transfer_in_zen, '--', color='orange', label='ZEN-IN (scaled)')
    ymin, ymax = ax.get_ylim()
    ax.fill_betweenx([ymin, ymax], wvl_start, wvl_end, color='gray', alpha=0.5, label='joint region')
    ax.set_ylim(0, ymax)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transfer flux (W m$^{-2}$ nm$^{-1}$)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('rad_resp_corr_zen_si_zen_in.png', dpi=300)
    # plt.show()
    
    # sys.exit()
    
    # (4) secondary response update
    # assume nad-si is correct, update other three responses 
    # (4-1) nad-in
    sec_resp_in_nad_ori = sec_resp_in_nad.copy()
    sec_resp_in_nad = sec_count_in_nad / int_time['nad|in'] / transfer_in_nad
    count_in_std_frac_nad = sec_count_in_std_nad / sec_count_in_nad
    transfer_in_std_frac    = transfer_in_std_nad    / transfer_in_nad
    sec_resp_in_std_nad = np.sqrt((count_in_std_frac_nad/int_time['nad|in']/transfer_in_nad)**2 + (sec_count_in_nad/int_time['nad|in']/transfer_in_nad*transfer_in_std_frac)**2) * sec_count_in_nad
    # (4-2) zen-si
    sec_resp_si_zen_ori = sec_resp_si_zen.copy()
    sec_resp_si_zen = sec_count_si_zen / int_time['zen|si'] / transfer_si_zen
    count_si_std_frac_zen = sec_count_si_std_zen / sec_count_si_zen
    transfer_si_std_frac    = transfer_si_std_zen    / transfer_si_zen
    sec_resp_si_std_zen = np.sqrt((count_si_std_frac_zen/int_time['zen|si']/transfer_si_zen)**2 + (sec_count_si_zen/int_time['zen|si']/transfer_si_zen*transfer_si_std_frac)**2) * sec_count_si_zen
    # (4-3) zen-in
    sec_resp_in_zen_ori = sec_resp_in_zen.copy()
    sec_resp_in_zen = sec_count_in_zen / int_time['zen|in'] / transfer_in_zen
    count_in_std_frac_zen = sec_count_in_std_zen / sec_count_in_zen
    transfer_in_std_frac    = transfer_in_std_zen    / transfer_in_zen
    sec_resp_in_std_zen = np.sqrt((count_in_std_frac_zen/int_time['zen|in']/transfer_in_zen)**2 + (sec_count_in_zen/int_time['zen|in']/transfer_in_zen*transfer_in_std_frac)**2) * sec_count_in_zen
    
    plt.close('all')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))
    ax1.plot(wvl_si_nad, sec_resp_si_nad, 'b-', label='NAD-si (original)')
    ax1.set_title('NAD-si')
    
    ax2.plot(wvl_in_nad, sec_resp_in_nad_ori, 'b-', label='NAD-in (original)')
    ax2.plot(wvl_in_nad, sec_resp_in_nad, 'r--', label='NAD-in (updated)')
    ax2.set_title('NAD-in')
    
    ax3.plot(wvl_si_zen, sec_resp_si_zen_ori, 'b-', label='ZEN-si (original)')
    ax3.plot(wvl_si_zen, sec_resp_si_zen, 'r--', label='ZEN-si (updated)')
    ax3.set_title('ZEN-si')
    
    ax4.plot(wvl_in_zen, sec_resp_in_zen_ori, 'b-', label='ZEN-in (original)')
    ax4.plot(wvl_in_zen, sec_resp_in_zen, 'r--', label='ZEN-in (updated)')
    ax4.set_title('ZEN-in')
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Secondary response (count/energy)')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
    fig.tight_layout()
    fig.savefig('rad_resp_corr_sec_resp_update.png', dpi=300)
    # plt.show()
    # sys.exit()
    
    
    new_pri_resp_zen = {
                        'zen|si': pri_resp_si_zen,
                        'zen|in': pri_resp_in_zen,
                        'zen|si_std': pri_resp_si_std_zen,
                        'zen|in_std': pri_resp_in_std_zen,
                        'zen|si_count': pri_count_si_zen,
                        'zen|in_count': pri_count_in_zen,
                        'zen|si_count_std': pri_count_si_std_zen,
                        'zen|in_count_std': pri_count_in_std_zen,
                        }
    new_transfer_zen = {
                        'zen|si': transfer_si_zen,
                        'zen|in': transfer_in_zen,
                        'zen|si_std': transfer_si_std_zen,
                        'zen|in_std': transfer_in_std_zen,
                        'zen|si_count': transfer_count_si_zen,
                        'zen|in_count': transfer_count_in_zen,
                        'zen|si_count_std': transfer_count_si_std_zen,
                        'zen|in_count_std': transfer_count_in_std_zen,
                        }
    new_sec_resp_zen = {
                        'zen|si': sec_resp_si_zen,
                        'zen|in': sec_resp_in_zen,
                        'zen|si_std': sec_resp_si_std_zen,
                        'zen|in_std': sec_resp_in_std_zen,
                        'zen|si_count': sec_count_si_zen,
                        'zen|in_count': sec_count_in_zen,
                        'zen|si_count_std': sec_count_si_std_zen,
                        'zen|in_count_std': sec_count_in_std_zen,
                        }

    new_pri_resp_nad = {
                        'nad|si': pri_resp_si_nad,
                        'nad|in': pri_resp_in_nad,
                        'nad|si_std': pri_resp_si_std_nad,
                        'nad|in_std': pri_resp_in_std_nad,
                        'nad|si_count': pri_count_si_nad,
                        'nad|in_count': pri_count_in_nad,
                        'nad|si_count_std': pri_count_si_std_nad,
                        'nad|in_count_std': pri_count_in_std_nad,
                        }
    new_transfer_nad = {
                        'nad|si': transfer_si_nad,
                        'nad|in': transfer_in_nad,
                        'nad|si_std': transfer_si_std_nad,
                        'nad|in_std': transfer_in_std_nad,
                        'nad|si_count': transfer_count_si_nad,
                        'nad|in_count': transfer_count_in_nad,
                        'nad|si_count_std': transfer_count_si_std_nad,
                        'nad|in_count_std': transfer_count_in_std_nad,
                        }
    new_sec_resp_nad = {
                        'nad|si': sec_resp_si_nad,
                        'nad|in': sec_resp_in_nad,
                        'nad|si_std': sec_resp_si_std_nad,
                        'nad|in_std': sec_resp_in_std_nad,
                        'nad|si_count': sec_count_si_nad,
                        'nad|in_count': sec_count_in_nad,
                        'nad|si_count_std': sec_count_si_std_nad,
                        'nad|in_count_std': sec_count_in_std_nad,
                        }
    
    # save file for zen
    si_tag = 'zen|si'
    in_tag = 'zen|in'
    pri_resp  = new_pri_resp_zen
    transfer  = new_transfer_zen
    sec_resp  = new_sec_resp_zen
    wvl_start = wvl_range[0]
    wvl_end   = wvl_range[-1]
    logic_si  = (wvls[si_tag] >= wvl_start)  & (wvls[si_tag] <= wvl_joint)
    logic_in  = (wvls[in_tag] >  wvl_joint)  & (wvls[in_tag] <= wvl_end)

    wvl_data      = np.concatenate((wvls[si_tag][logic_si], wvls[in_tag][logic_in]))
    pri_resp_data = np.concatenate((pri_resp[si_tag][logic_si], pri_resp[in_tag][logic_in]))
    transfer_data = np.concatenate((transfer[si_tag][logic_si], transfer[in_tag][logic_in]))
    sec_resp_data = np.concatenate((sec_resp[si_tag][logic_si], sec_resp[in_tag][logic_in]))
    pri_count_data = np.concatenate((pri_resp[si_tag+'_count'][logic_si], pri_resp[in_tag+'_count'][logic_in]))
    transfer_count_data = np.concatenate((transfer[si_tag+'_count'][logic_si], transfer[in_tag+'_count'][logic_in]))
    sec_count_data = np.concatenate((sec_resp[si_tag+'_count'][logic_si], sec_resp[in_tag+'_count'][logic_in]))
    
    pri_resp_data_std = np.concatenate((pri_resp[si_tag+'_std'][logic_si], pri_resp[in_tag+'_std'][logic_in]))
    transfer_data_std = np.concatenate((transfer[si_tag+'_std'][logic_si], transfer[in_tag+'_std'][logic_in]))
    sec_resp_data_std = np.concatenate((sec_resp[si_tag+'_std'][logic_si], sec_resp[in_tag+'_std'][logic_in]))
    pri_count_data_std = np.concatenate((pri_resp[si_tag+'_count_std'][logic_si], pri_resp[in_tag+'_count_std'][logic_in]))
    transfer_count_data_std = np.concatenate((transfer[si_tag+'_count_std'][logic_si], transfer[in_tag+'_count_std'][logic_in]))
    sec_count_data_std = np.concatenate((sec_resp[si_tag+'_count_std'][logic_si], sec_resp[in_tag+'_count_std'][logic_in]))

    indices_sort = np.argsort(wvl_data)
    wvl_      = wvl_data[indices_sort]
    pri_resp_ = pri_resp_data[indices_sort]
    transfer_ = transfer_data[indices_sort]
    sec_resp_ = sec_resp_data[indices_sort]
    pri_resp_std_ = pri_resp_data_std[indices_sort]
    transfer_std_ = transfer_data_std[indices_sort]
    sec_resp_std_ = sec_resp_data_std[indices_sort]
    pri_count_ = pri_count_data[indices_sort]
    transfer_count_ = transfer_count_data[indices_sort]
    sec_count_ = sec_count_data[indices_sort]
    pri_count_std_ = pri_count_data_std[indices_sort]
    transfer_count_std_ = transfer_count_data_std[indices_sort]
    sec_count_std_ = sec_count_data_std[indices_sort]
    wvl_zen_ = wvl_.copy()
    transfer_zen_ = transfer_.copy()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    

    
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_out = fnames_resp_zen.replace('.h5', '|corr.h5')

    f = h5py.File(fname_out, 'w')
    f['wvl']       = wvl_
    f['pri_resp']  = pri_resp_
    f['transfer']  = transfer_
    f['sec_resp']  = sec_resp_
    f['pri_resp_std']  = pri_resp_std_
    f['transfer_std']  = transfer_std_
    f['sec_resp_std']  = sec_resp_std_
    f['pri_count'] = pri_count_
    f['transfer_count'] = transfer_count_
    f['sec_count'] = sec_count_
    f['pri_count_std'] = pri_count_std_
    f['transfer_count_std'] = transfer_count_std_
    f['sec_count_std'] = sec_count_std_

    # raw data
    #╭────────────────────────────────────────────────╮#
    g = f.create_group('raw')
    g_si = g.create_group('si')
    g_si['wvl'] = wvls[si_tag]
    g_si['pri_resp'] = pri_resp[si_tag]
    g_si['transfer'] = transfer[si_tag]
    g_si['sec_resp'] = sec_resp[si_tag]
    g_si['pri_resp_std'] = pri_resp[si_tag+'_std']
    g_si['transfer_std'] = transfer[si_tag+'_std']
    g_si['sec_resp_std'] = sec_resp[si_tag+'_std']
    g_si['pri_count'] = pri_resp[si_tag+'_count']
    g_si['transfer_count'] = transfer[si_tag+'_count']
    g_si['sec_count'] = sec_resp[si_tag+'_count']
    g_si['pri_count_std'] = pri_resp[si_tag+'_count_std']
    g_si['transfer_count_std'] = transfer[si_tag+'_count_std']
    g_si['sec_count_std'] = sec_resp[si_tag+'_count_std']

    g_in = g.create_group('in')
    g_in['wvl'] = wvls[in_tag]
    g_in['pri_resp'] = pri_resp[in_tag]
    g_in['transfer'] = transfer[in_tag]
    g_in['sec_resp'] = sec_resp[in_tag]
    g_in['pri_resp_std'] = pri_resp[in_tag+'_std']
    g_in['transfer_std'] = transfer[in_tag+'_std']
    g_in['sec_resp_std'] = sec_resp[in_tag+'_std']
    g_in['pri_count'] = pri_resp[in_tag+'_count']
    g_in['transfer_count'] = transfer[in_tag+'_count']
    g_in['sec_count'] = sec_resp[in_tag+'_count']
    g_in['pri_count_std'] = pri_resp[in_tag+'_count_std']
    g_in['transfer_count_std'] = transfer[in_tag+'_count_std']
    g_in['sec_count_std'] = sec_resp[in_tag+'_count_std']
    #╰────────────────────────────────────────────────╯#

    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#
    
    
    # save file for nad
    si_tag = 'nad|si'
    in_tag = 'nad|in'
    pri_resp  = new_pri_resp_nad
    transfer  = new_transfer_nad
    sec_resp  = new_sec_resp_nad
    wvl_start = wvl_range[0]
    wvl_end   = wvl_range[-1]
    logic_si  = (wvls[si_tag] >= wvl_start)  & (wvls[si_tag] <= wvl_joint)
    logic_in  = (wvls[in_tag] >  wvl_joint)  & (wvls[in_tag] <= wvl_end)

    wvl_data      = np.concatenate((wvls[si_tag][logic_si], wvls[in_tag][logic_in]))
    pri_resp_data = np.concatenate((pri_resp[si_tag][logic_si], pri_resp[in_tag][logic_in]))
    transfer_data = np.concatenate((transfer[si_tag][logic_si], transfer[in_tag][logic_in]))
    sec_resp_data = np.concatenate((sec_resp[si_tag][logic_si], sec_resp[in_tag][logic_in]))
    pri_count_data = np.concatenate((pri_resp[si_tag+'_count'][logic_si], pri_resp[in_tag+'_count'][logic_in]))
    transfer_count_data = np.concatenate((transfer[si_tag+'_count'][logic_si], transfer[in_tag+'_count'][logic_in]))
    sec_count_data = np.concatenate((sec_resp[si_tag+'_count'][logic_si], sec_resp[in_tag+'_count'][logic_in]))
    
    pri_resp_data_std = np.concatenate((pri_resp[si_tag+'_std'][logic_si], pri_resp[in_tag+'_std'][logic_in]))
    transfer_data_std = np.concatenate((transfer[si_tag+'_std'][logic_si], transfer[in_tag+'_std'][logic_in]))
    sec_resp_data_std = np.concatenate((sec_resp[si_tag+'_std'][logic_si], sec_resp[in_tag+'_std'][logic_in]))
    pri_count_data_std = np.concatenate((pri_resp[si_tag+'_count_std'][logic_si], pri_resp[in_tag+'_count_std'][logic_in]))
    transfer_count_data_std = np.concatenate((transfer[si_tag+'_count_std'][logic_si], transfer[in_tag+'_count_std'][logic_in]))
    sec_count_data_std = np.concatenate((sec_resp[si_tag+'_count_std'][logic_si], sec_resp[in_tag+'_count_std'][logic_in]))

    indices_sort = np.argsort(wvl_data)
    wvl_      = wvl_data[indices_sort]
    pri_resp_ = pri_resp_data[indices_sort]
    transfer_ = transfer_data[indices_sort]
    sec_resp_ = sec_resp_data[indices_sort]
    pri_resp_std_ = pri_resp_data_std[indices_sort]
    transfer_std_ = transfer_data_std[indices_sort]
    sec_resp_std_ = sec_resp_data_std[indices_sort]
    pri_count_ = pri_count_data[indices_sort]
    transfer_count_ = transfer_count_data[indices_sort]
    sec_count_ = sec_count_data[indices_sort]
    pri_count_std_ = pri_count_data_std[indices_sort]
    transfer_count_std_ = transfer_count_data_std[indices_sort]
    sec_count_std_ = sec_count_data_std[indices_sort]
    wvl_nad_ = wvl_.copy()
    transfer_nad_ = transfer_.copy()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_out = fnames_resp_nad.replace('.h5', '|corr.h5')

    f = h5py.File(fname_out, 'w')
    f['wvl']       = wvl_
    f['pri_resp']  = pri_resp_
    f['transfer']  = transfer_
    f['sec_resp']  = sec_resp_
    f['pri_resp_std']  = pri_resp_std_
    f['transfer_std']  = transfer_std_
    f['sec_resp_std']  = sec_resp_std_
    f['pri_count'] = pri_count_
    f['transfer_count'] = transfer_count_
    f['sec_count'] = sec_count_
    f['pri_count_std'] = pri_count_std_
    f['transfer_count_std'] = transfer_count_std_
    f['sec_count_std'] = sec_count_std_

    # raw data
    #╭────────────────────────────────────────────────╮#
    g = f.create_group('raw')
    g_si = g.create_group('si')
    g_si['wvl'] = wvls[si_tag]
    g_si['pri_resp'] = pri_resp[si_tag]
    g_si['transfer'] = transfer[si_tag]
    g_si['sec_resp'] = sec_resp[si_tag]
    g_si['pri_resp_std'] = pri_resp[si_tag+'_std']
    g_si['transfer_std'] = transfer[si_tag+'_std']
    g_si['sec_resp_std'] = sec_resp[si_tag+'_std']
    g_si['pri_count'] = pri_resp[si_tag+'_count']
    g_si['transfer_count'] = transfer[si_tag+'_count']
    g_si['sec_count'] = sec_resp[si_tag+'_count']
    g_si['pri_count_std'] = pri_resp[si_tag+'_count_std']
    g_si['transfer_count_std'] = transfer[si_tag+'_count_std']
    g_si['sec_count_std'] = sec_resp[si_tag+'_count_std']

    g_in = g.create_group('in')
    g_in['wvl'] = wvls[in_tag]
    g_in['pri_resp'] = pri_resp[in_tag]
    g_in['transfer'] = transfer[in_tag]
    g_in['sec_resp'] = sec_resp[in_tag]
    g_in['pri_resp_std'] = pri_resp[in_tag+'_std']
    g_in['transfer_std'] = transfer[in_tag+'_std']
    g_in['sec_resp_std'] = sec_resp[in_tag+'_std']
    g_in['pri_count'] = pri_resp[in_tag+'_count']
    g_in['transfer_count'] = transfer[in_tag+'_count']
    g_in['sec_count'] = sec_resp[in_tag+'_count']
    g_in['pri_count_std'] = pri_resp[in_tag+'_count_std']
    g_in['transfer_count_std'] = transfer[in_tag+'_count_std']
    g_in['sec_count_std'] = sec_resp[in_tag+'_count_std']
    #╰────────────────────────────────────────────────╯#

    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    plt.close('all')
    f_transfer_nad_ori = interpolate.interp1d(wvl_nad_, transfer_nad_ori, bounds_error=False, fill_value=np.nan)
    transfer_nad_ori_interp = f_transfer_nad_ori(wvl_zen_)
    transfer_ratio_ori = transfer_nad_ori_interp / transfer_zen_ori
    
    f_transfer_nad_corr = interpolate.interp1d(wvl_nad_, transfer_nad_, bounds_error=False, fill_value=np.nan)
    transfer_nad_corr_interp = f_transfer_nad_corr(wvl_zen_)
    transfer_ratio_corr = transfer_nad_corr_interp / transfer_zen_
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    ax1.plot(wvl_zen_, transfer_zen_ori, 'r-', label='original')
    ax1.plot(wvl_zen_, transfer_zen_, 'b--', label='scaled')
    ax1.set_title('ZEN transfer')
    
    ax2.plot(wvl_zen_, transfer_nad_ori_interp, 'r-', label='original')
    ax2.plot(wvl_zen_, transfer_nad_corr_interp, 'b--', label='scaled')
    ax2.set_title('NAD transfer (interpolated to ZEN wvl)')
    
    
    
    ax3.plot(wvl_zen_, transfer_nad_ori/transfer_zen_ori, 
            '-', color='cyan', label='original')
    ax3.plot(wvl_zen_, transfer_nad_corr_interp/transfer_zen_, 
            '--', color='blue', label='scaled')
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Transfer flux (W m$^{-2}$ nm$^{-1}$)')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Transfer ratio (NAD/Zen)')
    # set ax4 invisible
    ax4.axis('off')
    fig.tight_layout()
    fig.savefig('rad_resp_corr_transfer_check.png', dpi=300)
    # plt.show()
    # sys.exit()


if __name__ == '__main__':

    pass
