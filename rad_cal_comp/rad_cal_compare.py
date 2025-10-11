"""
Code for comparing the SSFR effective counts of different measurements.
"""

import os
import sys
import glob
import datetime
import warnings
import importlib
from collections import OrderedDict
from tqdm import tqdm
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# mpl.use('Agg')



import ssfr


def plot_response(
        which_ssfr='lasp|ssrr-a',
        which_lc='nad',
        si_integration_time='*',
        fdir='../',
        ):

    # 2025-02-18_lamp-1324_post|2024-03-29_lamp-150c_after-pri|2024-03-29_lamp-150c_after-pri|2025-09-11_processed-for-arcsix|rad-resp|lasp|ssfr-a|nad|si-120|in-350.h5
    search_path_nad = os.path.join(fdir, '*|*processed-for-arcsix|rad-resp|%s|%s|si-%s|in-*0*|corr.h5' % (which_ssfr, which_lc, si_integration_time))
    fnames_nad = sorted(glob.glob(search_path_nad))
    pri_files_nad = [os.path.basename(fname).split('|')[0] for fname in fnames_nad]
    transfer_files_nad = [os.path.basename(fname).split('|')[1] for fname in fnames_nad]
    sec_files_nad = [os.path.basename(fname).split('|')[2] for fname in fnames_nad]
    integration_time_nad = [int(os.path.basename(fname).split('|')[-2].split('-')[-1]) for fname in fnames_nad]
    pri_set_nad = [pri_files_nad[i]+'|'+str(integration_time_nad[i]) for i in range(len(pri_files_nad))]
    transfer_nad = [transfer_files_nad[i]+'|'+str(integration_time_nad[i]) for i in range(len(transfer_files_nad))]
    sec_nad = [sec_files_nad[i]+'|'+str(integration_time_nad[i]) for i in range(len(sec_files_nad))]
    pri_transfer_set_nad = [pri_files_nad[i]+'|'+transfer_files_nad[i]+'|'+str(integration_time_nad[i]) for i in range(len(pri_set_nad))]
    transfer_sec_set_nad = [transfer_files_nad[i]+'|'+sec_files_nad[i]+'|'+str(integration_time_nad[i]) for i in range(len(transfer_files_nad))]
    if not fnames_nad:
        raise OSError('No file found for pattern: %s' % search_path_nad)

    color_list_nad = plt.cm.rainbow(np.linspace(0, 1, len(fnames_nad)))
    
    # plot primary response
    plt.close('all')
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(111)
    color_list_nad = plt.cm.rainbow(np.linspace(0, 1, len(set(pri_set_nad))))
    color_list_nad_ind = 0
    pri_set_nad_plot = []
    for i in range(len(fnames_nad)):
        fname = fnames_nad[i]
        if pri_set_nad[i] in pri_set_nad_plot:
            continue
        pri_set_nad_plot.append(pri_set_nad[i])
        color = color_list_nad[color_list_nad_ind]
        color_list_nad_ind += 1
        # Optionally parse params from filename if needed
        # params = parse_fname(fname)
        f = h5py.File(fname, 'r')
        wvl = f['wvl'][...]
        resp = f['pri_resp'][...]
        resp_std = f['pri_resp_std'][...]
        wvl_si = f['raw/si/wvl'][...]
        respil_si = f['raw/si/pri_resp'][...]
        respil_si_std = f['raw/si/pri_resp_std'][...]
        wvl_in = f['raw/in/wvl'][...]
        respil_in = f['raw/in/pri_resp'][...]
        respil_in_std = f['raw/in/pri_resp_std'][...]
        f.close()
        label = pri_set_nad[i]
        ax1.plot(wvl, resp, lw=1.0, color=color, label=label)
        ax1.fill_between(wvl, resp - resp_std, resp + resp_std, color=color, alpha=0.3)
        ax1.plot(wvl_si, respil_si, lw=1.0, ls='--', color=color)
        ax1.fill_between(wvl_si, respil_si - respil_si_std, respil_si + respil_si_std, color=color, alpha=0.3)
        ax1.plot(wvl_in, respil_in, lw=1.0, ls=':', color=color)
        ax1.fill_between(wvl_in, respil_in - respil_in_std, respil_in + respil_in_std, color=color, alpha=0.3)

    ax1.set_xlim(350.0, 2200.0)
    ax1.set_ylim(0.0, None)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Response (counts / (W m$^{-2}$ nm$^{-1}$ $\\cdot$ s))')
    ax1.set_title('%s (%s)' % (which_ssfr.upper(), which_lc.upper()))
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=10)
    
    

    fname_fig = '%s_%s_si_%s_pri.png' % (which_ssfr, which_lc, si_integration_time)
    fig.savefig(fname_fig, bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)
    
    
    plt.close('all')
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(111)
    color_list_nad = plt.cm.rainbow(np.linspace(0, 1, len(set(pri_set_nad))))
    color_list_nad_ind = 0
    pri_set_nad_plot = []
    for i in range(len(fnames_nad)):
        fname = fnames_nad[i]
        if pri_set_nad[i] in pri_set_nad_plot:
            continue
        pri_set_nad_plot.append(pri_set_nad[i])
        color = color_list_nad[color_list_nad_ind]
        
        # Optionally parse params from filename if needed
        # params = parse_fname(fname)
        f = h5py.File(fname, 'r')
        wvl = f['wvl'][...]
        resp = f['pri_resp'][...]
        resp_std = f['pri_resp_std'][...]
        wvl_si = f['raw/si/wvl'][...]
        respil_si = f['raw/si/pri_resp'][...]
        respil_si_std = f['raw/si/pri_resp_std'][...]
        wvl_in = f['raw/in/wvl'][...]
        respil_in = f['raw/in/pri_resp'][...]
        respil_in_std = f['raw/in/pri_resp_std'][...]
        f.close()
        
        
        if color_list_nad_ind == 0:
            label_ref = pri_set_nad[i]
            resp_ref = resp.copy()
            respil_si_ref = respil_si.copy()
            respil_in_ref = respil_in.copy()
        
        label = pri_set_nad[i]+'/'+label_ref
        
        ax1.plot(wvl, resp/resp_ref, lw=1.0, color=color, label=label)
        # ax1.fill_between(wvl, resp - resp_std, resp + resp_std, color=color, alpha=0.3)
        ax1.plot(wvl_si, respil_si/respil_si_ref, lw=1.0, ls='--', color=color)
        # ax1.fill_between(wvl_si, respil_si - respil_si_std, respil_si + respil_si_std, color=color, alpha=0.3)
        ax1.plot(wvl_in, respil_in/respil_in_ref, lw=1.0, ls=':', color=color)
        # ax1.fill_between(wvl_in, respil_in - respil_in_std, respil_in + respil_in_std, color=color, alpha=0.3)
        
        color_list_nad_ind += 1

    ax1.set_xlim(350.0, 2200.0)
    ax1.set_ylim(0.0, None)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Relative Response (to %s)' % label_ref)
    ax1.set_title('%s (%s)' % (which_ssfr.upper(), which_lc.upper()))
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=10)


    fname_fig = '%s_%s_si_%s_pri_relative.png' % (which_ssfr, which_lc, si_integration_time)
    fig.savefig(fname_fig, bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)
    
    # plot primary count
    plt.close('all')
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(111)
    color_list_nad = plt.cm.rainbow(np.linspace(0, 1, len(set(pri_set_nad))))
    color_list_nad_ind = 0
    pri_set_nad_plot = []
    for i in range(len(fnames_nad)):
        fname = fnames_nad[i]
        if pri_set_nad[i] in pri_set_nad_plot:
            continue
        pri_set_nad_plot.append(pri_set_nad[i])
        color = color_list_nad[color_list_nad_ind]
        color_list_nad_ind += 1
        # Optionally parse params from filename if needed
        # params = parse_fname(fname)
        f = h5py.File(fname, 'r')
        wvl = f['wvl'][...]
        resp = f['pri_count'][...]
        resp_std = f['pri_count_std'][...]
        f.close()
        label = pri_set_nad[i]
        ax1.plot(wvl, resp, lw=1.0, color=color, label=label)
        ax1.fill_between(wvl, resp - resp_std, resp + resp_std, color=color, alpha=0.3)

    ax1.set_xlim(350.0, 2200.0)
    ax1.set_ylim(0.0, None)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Counts')
    ax1.set_title('%s (%s)' % (which_ssfr.upper(), which_lc.upper()))
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=10)

    fname_fig = '%s_%s_si_ %s_pri_counts.png' % (which_ssfr, which_lc, si_integration_time)
    fig.savefig(fname_fig, bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)
    
    
    # plot transfer flux
    plt.close('all')
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(111)
    color_list_nad = plt.cm.rainbow(np.linspace(0, 1, len(set(pri_transfer_set_nad))))
    color_list_nad_ind = 0
    pri_transfer_set_nad_plot = []
    for i in range(len(fnames_nad)):
        fname = fnames_nad[i]
        if pri_transfer_set_nad[i] in pri_transfer_set_nad_plot:
            continue
        pri_transfer_set_nad_plot.append(pri_transfer_set_nad[i])
        color = color_list_nad[color_list_nad_ind]
        color_list_nad_ind += 1
        # Optionally parse params from filename if needed
        # params = parse_fname(fname)
        f = h5py.File(fname, 'r')
        wvl = f['wvl'][...]
        resp = f['transfer'][...]
        resp_std = f['transfer_std'][...]
        f.close()
        label = pri_transfer_set_nad[i]
        ax1.plot(wvl, resp, lw=1.0, color=color, label=label)
        ax1.fill_between(wvl, resp - resp_std, resp + resp_std, color=color, alpha=0.3)

    ax1.set_xlim(350.0, 2200.0)
    ax1.set_ylim(0.0, None)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Transfer flux (W m$^{-2}$ nm$^{-1}$)')
    ax1.set_title('%s (%s)' % (which_ssfr.upper(), which_lc.upper()))
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=10)

    fname_fig = '%s_%s_si_%s_transfer.png' % (which_ssfr, which_lc, si_integration_time)
    fig.savefig(fname_fig, bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)
    
    # plot transfer counts
    plt.close('all')
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(111)
    color_list_nad = plt.cm.rainbow(np.linspace(0, 1, len(set(transfer_nad))))
    color_list_nad_ind = 0
    transfer_nad_plot = []
    for i in range(len(fnames_nad)):
        fname = fnames_nad[i]
        if transfer_nad[i] in transfer_nad_plot:
            continue
        transfer_nad_plot.append(transfer_nad[i])
        color = color_list_nad[color_list_nad_ind]
        color_list_nad_ind += 1
        # Optionally parse params from filename if needed
        # params = parse_fname(fname)
        f = h5py.File(fname, 'r')
        wvl = f['wvl'][...]
        resp = f['transfer_count'][...]
        resp_std = f['transfer_count_std'][...]
        f.close()
        label = transfer_nad[i]
        ax1.plot(wvl, resp, lw=1.0, color=color, label=label)
        ax1.fill_between(wvl, resp - resp_std, resp + resp_std, color=color, alpha=0.3)

    ax1.set_xlim(350.0, 2200.0)
    ax1.set_ylim(0.0, None)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Transfer Counts')
    ax1.set_title('%s (%s)' % (which_ssfr.upper(), which_lc.upper()))
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=10)

    fname_fig = '%s_%s_si_%s_transfer_counts.png' % (which_ssfr, which_lc, si_integration_time)
    fig.savefig(fname_fig, bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)
    
    
    # plot transfer and secondary counts
    plt.close('all')
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(111)
    color_list_nad = plt.cm.rainbow(np.linspace(0, 1, len(set(transfer_nad+sec_nad))))
    color_list_nad_ind = 0
    transfer_sec_nad_plot = []
    for i in range(len(fnames_nad)):
        fname = fnames_nad[i]
        
        
        # print("transfer_nad[i]=", transfer_nad[i], transfer_nad[i] in transfer_sec_nad_plot)
        if transfer_nad[i] in transfer_sec_nad_plot:
            continue
        transfer_sec_nad_plot.append(transfer_nad[i])
        color = color_list_nad[color_list_nad_ind]
        # Optionally parse params from filename if needed
        # params = parse_fname(fname)
        f = h5py.File(fname, 'r')
        wvl = f['wvl'][...]
        resp = f['transfer_count'][...]
        resp_std = f['transfer_count_std'][...]
        f.close()
        label = transfer_nad[i]
        ax1.plot(wvl, resp, lw=1.0, color=color, label=label)
        ax1.fill_between(wvl, resp - resp_std, resp + resp_std, color=color, alpha=0.3)
        color_list_nad_ind += 1
    
    for i in range(len(fnames_nad)):
        fname = fnames_nad[i]    
        # print("sec_nad[i]=", sec_nad[i], sec_nad[i] in transfer_sec_nad_plot)
        if sec_nad[i] in transfer_sec_nad_plot:
            continue
        transfer_sec_nad_plot.append(sec_nad[i])
        color = color_list_nad[color_list_nad_ind]
        # Optionally parse params from filename if needed
        # params = parse_fname(fname)
        f = h5py.File(fname, 'r')
        wvl = f['wvl'][...]
        sec_count = f['sec_count'][...]
        sec_count_std = f['sec_count_std'][...]
        f.close()
        label = sec_nad[i]
        ax1.plot(wvl, sec_count, lw=1.0, color=color, label=label)
        ax1.fill_between(wvl, sec_count - sec_count_std, sec_count + sec_count_std, color=color, alpha=0.3)
        color_list_nad_ind += 1
        

    ax1.set_xlim(350.0, 2200.0)
    ax1.set_ylim(0.0, None)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Counts')
    ax1.set_title('%s (%s)' % (which_ssfr.upper(), which_lc.upper()))
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=10)

    fname_fig = '%s_%s_si_%s_transfer_sec_counts.png' % (which_ssfr, which_lc, si_integration_time)
    fig.savefig(fname_fig, bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)



#╰────────────────────────────────────────────────────────────────────────────╯#


if __name__ == '__main__':


    # post-mission SSRR calibration (nadir)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # main_ssrr_rad_cal_all(which_ssrr='lasp|ssrr-a')
    # main_ssrr_rad_cal_all(which_ssrr='lasp|ssrr-b')
    plot_response(which_ssfr='lasp|ssfr-a', which_lc='nad', si_integration_time='080', fdir='../',)
    # plot_response(which_ssfr='lasp|ssfr-a', which_lc='nad', si_integration_time='*', fdir='.',)
    plot_response(which_ssfr='lasp|ssfr-a', which_lc='zen', si_integration_time='080', fdir='../',)
    # plot_response(which_ssfr='lasp|ssfr-a', which_lc='zen', si_integration_time='120', fdir='.',)
    # plot_response(which_ssfr='lasp|ssfr-a', which_lc='zen', si_integration_time='*', fdir='.',)
    # plot_response(which_ssfr='lasp|ssrr-b', which_lc='nad', fdir='.',)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    pass
