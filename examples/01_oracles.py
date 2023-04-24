import os
import sys
import glob
import datetime
import multiprocessing as mp
from collections import OrderedDict
from tqdm import tqdm
import h5py
from pyhdf.SD import SD, SDC
import numpy as np
from scipy import interpolate
from scipy.io import readsav
from scipy.optimize import curve_fit
from scipy.ndimage import shift
from scipy.stats import pearsonr
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from ssfr import nasa_ssfr, load_h5, read_ict, write_ict, read_iwg, cal_solar_angles, quicklook_bokeh_ssfr_and_spns, prh2za, muslope
from ssfr.cal import cdata_cos_resp
from ssfr.cal import cdata_rad_resp
from ssfr.corr import cos_corr



# plots
# ================================================================================================
def plot_cos_resp_oracles(fname):

    f         = h5py.File(fname, 'r')
    wvl       = f['wvl'][...]
    cos_resp  = f['cos_resp'][...][::10, :]
    poly_coef = f['poly_coef'][...][::10, :]
    mu        = f['mu'][...][::10]
    f.close()

    logic    = (wvl>400.0) & (wvl<2000.0)
    wvl      = wvl[logic]
    cos_resp = cos_resp[:, logic]

    colors1 = plt.cm.jet(np.linspace(0.0, 1.0, wvl.size))
    colors2 = plt.cm.jet(np.linspace(0.0, 1.0, mu.size))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for i in range(wvl.size):
        ax1.plot(mu, cos_resp[:, i], color=colors1[i, ...], lw=0.5)
    ax1.set_xlim((0.0, 1.0))
    ax1.set_ylim((0.0, 1.2))
    ax1.set_xlabel('$\mathrm{cos(\\theta)}$')
    ax1.set_ylabel('Cosine Response')

    for i in range(mu.size):
        f = np.poly1d(poly_coef[i, :])
        ax2.plot(wvl, f(wvl), color=colors2[i, ...], lw=0.5)
    ax2.set_xlim((300.0, 2100.0))
    ax2.set_xlabel('Wavelength [nm]')
    ax2.set_ylabel('Cosine Response')

    plt.savefig(fname.replace('h5', 'png'), bbox_inches='tight')
    plt.close(fig)
    # ---------------------------------------------------------------------

def plot_rad_resp_oracles(fname):

    f         = h5py.File(fname, 'r')
    wvl       = f['wvl'][...]
    pri_resp  = f['pri_resp'][...]
    transfer  = f['transfer'][...]
    sec_resp  = f['sec_resp'][...]
    f.close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111)
    ax1.plot(wvl, pri_resp, lw=5.5, c='k', alpha=0.7)
    ax1.plot(wvl, sec_resp, lw=1.5, c='b')

    ax2 = ax1.twinx()
    ax2.plot(wvl, transfer, lw=1.5, c='r')
    ax2.set_ylim((0.0, 0.3))

    ax1.set_xlim((300.0, 2100.0))
    ax1.set_ylim((0.0, 300.0))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Radiometric Response')

    patches_legend = [
                mpatches.Patch(color='black' , label='Primary'),
                mpatches.Patch(color='red'   , label='Transfer'),
                mpatches.Patch(color='blue'  , label='Secondary')
                ]
    ax2.legend(handles=patches_legend, loc='upper right', fontsize=14)

    plt.savefig(fname.replace('h5', 'png'), bbox_inches='tight')
    plt.close(fig)
    # ---------------------------------------------------------------------

def plot_rad_cal_oracles_time_series(which='nadir', wvl0=500.0):

    fdir = '/argus/field/oracles/2018/p3/calibration/rad-cal'

    dirs = get_sub_dir(fdir, full=False, contains=['pre', 'post'])

    colors = {
            '20171106_pre' : 'red',
            '20190423_post': 'blue'
            }

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)

    for dir0 in dirs:
        fnames = sorted(glob.glob('%s/%s*%s*.h5' % (fdir, dir0.replace('_', '-'), which)))

        x = np.array([])
        y = np.array([])
        x_label = []
        for i, fname in enumerate(fnames):
            filename = os.path.basename(fname)
            info = filename.split('.')[0].split('_')
            label_pri = info[0]
            label_field = info[1]
            date_field_s = label_field.split('-')[0]
            f = h5py.File(fname, 'r')
            wvl = f['wvl'][...]
            resp = f['sec_resp'][...]
            f.close()
            index = np.argmin(np.abs(wvl-wvl0))

            date_field = (datetime.datetime.strptime(date_field_s, '%Y%m%d'))
            jday0      = (date_field-datetime.datetime(2018, 1, 1)).days + 1.0
            x_label.append(date_field.strftime('%m-%d'))

            x = np.append(x, i)
            y = np.append(y, resp[index])

        if 'pre' in dir0:
            f = h5py.File(fnames[0], 'r')
            resp = f['pri_resp'][...]
            f.close()
            ax1.plot(x[0]-1, resp[index], marker='*', markersize=20, color=colors[dir0])
            ax1.plot([x[0]-1, x[0]], [resp[index], y[0]], color=colors[dir0])

        elif 'post' in dir0:
            f = h5py.File(fnames[0], 'r')
            resp = f['pri_resp'][...]
            f.close()
            ax1.plot(x[-1]+1, resp[index], marker='*', markersize=20, color=colors[dir0])
            ax1.plot([x[-1], x[-1]+1], [y[-1], resp[index]], color=colors[dir0])

        ax1.plot(x, y, marker='o', markersize=6, color=colors[dir0])

    patches_legend = [
                mpatches.Patch(color='red'   , label='20171106 Pre (Lamp 1324)'),
                mpatches.Patch(color='blue'  , label='20190423 Post (Lamp 1324)'),
                ]
    ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)

    ax1.xaxis.set_major_locator(FixedLocator(x))
    ax1.set_xticklabels(x_label, rotation=45, ha='center')
    # ax1.set_ylim((100, 300))
    ax1.set_ylim((50, 250))
    ax1.set_ylabel('Response')
    ax1.set_title('Secondary Response of %s at %.4fnm (ORACLES 2018)' % (which.title(), wvl[index]))
    plt.savefig('rad_cal_time_series_%s_%.4fnm.png' % (which, wvl[index]), bbox_inches='tight')
    plt.close(fig)
    # ---------------------------------------------------------------------

def plot_diffuse_ratio():
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(7, 4))
    ax1 = fig.add_subplot(111)
    ax1.scatter(ssfr_v0.zen_wvl, diff_ratio0[5000, :], s=12, c='k', alpha=0.05, zorder=0)
    ax1.scatter(x, y, s=3 , c='b', alpha=0.6 , zorder=0)
    ax1.plot(ssfr_v0.zen_wvl, diff_ratio[5000, :], lw=1.0, c='r', zorder=1)
    ax1.text(2000, 0.5, '$y=%.3f(\\frac{x}{500.0})^{%.3f} + %.3f$' % (popt[0], popt[1], popt[2]), color='red', ha='right')

    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Ratio')
    ax1.set_ylim((0.0, 1.0))

    ax1.set_title('SPN-S %s' % date_s)

    patches_legend = [
                mpatches.Patch(color='black', label='SPN-S Ratio (All)'),
                mpatches.Patch(color='blue' , label='SPN-S Ratio (Filtered)'),
                mpatches.Patch(color='red'  , label='SSFR Ratio (Fitted)')
                ]
    ax1.legend(handles=patches_legend, loc='upper right', fontsize=10)
    plt.savefig('%s_%5.5d.png' % (date_s, i), bbox_inches='tight')
    plt.close(fig)
    # ---------------------------------------------------------------------

def plot_spns_v0(fname, wvl0=532.0):

    data = load_h5(fname)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)

    index = np.argmin(np.abs(wvl0-data['dif_wvl']))
    ax1.scatter(data['dif_tmhr'], data['dif_flux'][:, index], color='red', s=40, lw=0.0)

    index = np.argmin(np.abs(wvl0-data['tot_wvl']))
    ax1.scatter(data['tot_tmhr'], data['tot_flux'][:, index], color='blue', s=40, lw=0.0)

    ax1.set_xlabel('Time [Hour]')
    ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
    ax1.set_title('SPN-S at %.2f nm' % data['tot_wvl'][index])
    plt.savefig(os.path.basename(fname).replace('.h5', '.png'), bbox_inches='tight')
    plt.close(fig)
    # ---------------------------------------------------------------------

def plot_flux(date, tmhr_range=[-5, 8]):

    date_s = date.strftime('%Y%m%d')

    data1 = load_h5('data/SSFR_%s_V0.h5' % date_s)
    data2 = load_h5('data/SSFR_%s_IWG.h5' % date_s)

    logic1 = (data1['tmhr']>=tmhr_range[0]) & (data1['tmhr']<=tmhr_range[1])
    logic2 = (data2['tmhr']>=tmhr_range[0]) & (data2['tmhr']<=tmhr_range[1])

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(data1['tmhr'][logic1], data1['shutter'][logic1])
    ax1.scatter(data2['tmhr'][logic2], data2['zen_flux'][:, 100][logic2])
    ax1.set_xlim(tmhr_range)

    plt.show()
    exit()
    # ---------------------------------------------------------------------


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    index_zen = np.argmin(np.abs(data1['zen_wvl']-500.0))
    index_nad = np.argmin(np.abs(data1['nad_wvl']-500.0))

    ax1.scatter(data1['tmhr'], data1['zen_flux'][:, index_zen], c='red', s=10, lw=0.0, alpha=0.6)
    ax1.scatter(data1['tmhr'], data1['nad_flux'][:, index_nad], c='red', s=10, lw=0.0, alpha=0.6)

    ax1.scatter(data2['tmhr'], data2['zen_flux'][:, index_zen], c='black', s=4, lw=0.0)
    ax1.scatter(data2['tmhr'], data2['nad_flux'][:, index_nad], c='black', s=4, lw=0.0)

    ax1.set_xlim((12, 20))
    ax1.set_ylim((0.0, 3.0))
    ax1.set_xlabel('Time [Hour]')
    ax1.set_ylabel('Flux [$\mathrm{W m^{-2} nm^{-1}}$]')
    ax1.set_title('500 nm')
    ax1.axvline(tmhr0, color='k', lw=1.0, ls='-')

    ax0 = ax1.twinx()
    ax0.plot(data2['tmhr'], np.cos(np.deg2rad(data2['sza'])), c='r', lw=0.6)
    ax0.set_ylabel('$\mathrm{cos(\\theta)}$', rotation=270, labelpad=18, color='r')
    ax0.set_ylim((0.5, 1.0))

    patches_legend = [
                mpatches.Patch(color='red'    , label='Before Cosine Correction'),
                mpatches.Patch(color='black'   , label='After Cosine Correction'),
                ]

    ax0.legend(handles=patches_legend, loc='upper right', fontsize=12)

    ax2   = fig.add_subplot(212)
    index = np.argmin(np.abs(data1['tmhr']-tmhr0))
    ax2.plot(data1['zen_wvl'], data1['zen_flux'][index, :], color='red', lw=2.5, alpha=0.6)
    ax2.plot(data1['nad_wvl'], data1['nad_flux'][index, :], color='red', lw=2.5, alpha=0.6)
    ax2.plot(data2['zen_wvl'], data2['zen_flux'][index, :], color='black', lw=1.5)
    ax2.plot(data2['nad_wvl'], data2['nad_flux'][index, :], color='black', lw=1.5)

    ax2.set_xlim((300, 2100))
    ax2.set_ylim((0.0, 2.0))
    ax2.set_xlabel('Wavelength [nm]')
    ax2.set_ylabel('Flux [$\mathrm{W m^{-2} nm^{-1}}$]')

    plt.savefig('flux.png', bbox_inches='tight')
    plt.show()
    # ---------------------------------------------------------------------

def plot_angles(date_s):

    # date_s = fdir_ssfr.split('/')[-1]

    fname_alp = 'data/ALP_%s_V0.h5' % date_s
    data_alp  = load_h5(fname_alp)

    fname_ssfr0 = 'data/SSFR_%s_V0.h5' % date_s
    data_ssfr0  = load_h5(fname_ssfr0)

    angles = {}
    angles['pitch']        = np.interp(data_ssfr0['tmhr'], data_alp['tmhr'], data_alp['ang_pit'])
    angles['roll']         = np.interp(data_ssfr0['tmhr'], data_alp['tmhr'], data_alp['ang_rol'])
    angles['pitch_motor']  = np.interp(data_ssfr0['tmhr'], data_alp['tmhr'], data_alp['ang_pit_m'])
    angles['roll_motor']   = np.interp(data_ssfr0['tmhr'], data_alp['tmhr'], data_alp['ang_rol_m'])
    angles['heading']      = np.interp(data_ssfr0['tmhr'], data_alp['tmhr'], data_alp['ang_hed'])
    angles['pitch_offset'] = 0.0
    angles['roll_offset']  = 0.0

    lon = np.interp(data_ssfr0['tmhr'], data_alp['tmhr'], data_alp['lon'])
    lat = np.interp(data_ssfr0['tmhr'], data_alp['tmhr'], data_alp['lat'])
    alt = np.interp(data_ssfr0['tmhr'], data_alp['tmhr'], data_alp['alt'])
    angles['solar_zenith'], angles['solar_azimuth'] = cal_solar_angles(data_ssfr0['jday'], lon, lat, alt)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    ax1.scatter(data_ssfr0['tmhr'], angles['pitch']      , c='b'      , s=2)
    ax1.scatter(data_ssfr0['tmhr'], angles['pitch_motor'], c='cyan'   , s=2)
    ax1.scatter(data_ssfr0['tmhr'], angles['roll']       , c='r'      , s=2)
    ax1.scatter(data_ssfr0['tmhr'], angles['roll_motor'] , c='magenta', s=2)
    ax1.set_xlim((12, 20))
    ax1.set_xlabel('Time [Hour]')
    ax1.set_ylabel('Pitch/Roll Angles')
    patches_legend = [
                mpatches.Patch(color='blue'    , label='Pitch'),
                mpatches.Patch(color='cyan'    , label='Pitch Motor'),
                mpatches.Patch(color='red'     , label='Roll'),
                mpatches.Patch(color='magenta' , label='Roll Motor')
                ]
    ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)


    ax2 = fig.add_subplot(212)
    ax2.scatter(data_ssfr0['tmhr'], angles['solar_zenith']  , c='r', s=2)
    ax2.scatter(data_ssfr0['tmhr'], angles['solar_azimuth'] , c='b', s=2)
    ax2.set_xlim((12, 20))
    ax2.set_xlabel('Time [Hour]')
    ax2.set_ylabel('Solar Zenith/Azimuth Angles')
    patches_legend = [
                mpatches.Patch(color='red'   , label='Solar Zenith'),
                mpatches.Patch(color='blue'  , label='Solar Azimuth')
                ]
    ax2.legend(handles=patches_legend, loc='upper right', fontsize=12)


    plt.savefig('angles.png', bbox_inches='tight')
    plt.show()
    # ---------------------------------------------------------------------

def plot_spiral(date, tmhr_range=[4.5222, 5.0611], wvl=532.0):

    date_s     = date.strftime('%Y%m%d')
    fname_ssfr = 'data/SSFR_%s_IWG.h5'% date_s
    fname_iwg  = 'data/IWG_%s_V0.h5' % date_s

    ssfr0 = load_h5(fname_ssfr)
    nav0  = load_h5(fname_iwg)

    logic = (ssfr0['tmhr']>tmhr_range[0]) & (ssfr0['tmhr']<tmhr_range[1]) & (np.abs(nav0['pitch_angle'])<=3.0) & (np.abs(nav0['roll_angle'])<=3.0)

    index_wvl0 = np.argmin(np.abs(ssfr0['zen_wvl']-wvl))
    f_up      = ssfr0['nad_flux'][logic, index_wvl0]
    f_dn      = ssfr0['zen_flux'][logic, index_wvl0]
    f_net     = f_dn-f_up

    index_wvl1 = np.argmin(np.abs(ssfr0['spns_wvl']-wvl))
    f_dn_dif  = ssfr0['spns_dif_flux'][logic, index_wvl1]
    f_dn_tot  = ssfr0['spns_tot_flux'][logic, index_wvl1]

    alt       = ssfr0['alt'][logic]/1000.0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(4, 8))
    ax1 = fig.add_subplot(111)
    ax1.scatter(f_up, alt, s=5, c='r')
    ax1.scatter(f_dn, alt, s=5, c='b')
    ax1.scatter(f_net, alt, s=5, c='k')
    ax1.scatter(f_dn_dif, alt, s=5, c='g')
    ax1.scatter(f_dn_tot, alt, s=5, c='cyan')
    ax1.set_xlim((0.0, 2.5))
    ax1.set_xlabel('Flux [$\mathrm{W m^{-2} nm^{-1}}$]')
    ax1.set_ylabel('Altitude [km]')
    ax1.set_title('Spiral on %s [%.4f, %.4f]' % (date_s, tmhr_range[0], tmhr_range[1]))

    # patches_legend = [
    #             mpatches.Patch(color='red'   , label='SSFR $F_\\uparrow$ %.2f nm' % (ssfr0['zen_wvl'][index_wvl0])),
    #             mpatches.Patch(color='blue'  , label='SSFR $F_\downarrow$ %.2f nm' % (ssfr0['zen_wvl'][index_wvl0])),
    #             mpatches.Patch(color='black' , label='SSFR $F_{net}$ %.2f nm' % (ssfr0['zen_wvl'][index_wvl0])),
    #             mpatches.Patch(color='green' , label='SPN-S $F_\downarrow^{dif}$ %.2f nm' % (ssfr0['spns_wvl'][index_wvl1])),
    #             mpatches.Patch(color='cyan'  , label='SPN-S $F_\downarrow^{tot}$  %.2f nm' % (ssfr0['spns_wvl'][index_wvl1]))
    #             ]
    patches_legend = [
                mpatches.Patch(color='red'   , label='SSFR $F_\\uparrow$'),
                mpatches.Patch(color='blue'  , label='SSFR $F_\downarrow$'),
                mpatches.Patch(color='black' , label='SSFR $F_{net}$'),
                mpatches.Patch(color='green' , label='SPN-S $F_\downarrow^{dif}$'),
                mpatches.Patch(color='cyan'  , label='SPN-S $F_\downarrow^{tot}$m')
                ]
    ax1.legend(handles=patches_legend, loc='upper center', fontsize=10)
    # ax1.legend(handles=patches_legend, loc='lower right', fontsize=10)

    plt.savefig('%s_spiral.png' % date_s, bbox_inches='tight')
    plt.show()
    # ---------------------------------------------------------------------
# ================================================================================================




# sub functions
# ================================================================================================
def interp(x, x0, y0, mode='linear'):

    f = interpolate.interp1d(x0, y0, bounds_error=False, kind=mode)

    return f(x)

def get_sub_dir(fdir, full=False, contains=[], verbose=False):

    fdirs = []

    for fdir0 in os.listdir(fdir):
        if len(contains) > 0:
            if os.path.isdir(os.path.join(fdir, fdir0)) and any([(string in fdir0) for string in contains]):
                if full:
                    fdirs.append('%s/%s' % (fdir, fdir0))
                else:
                    fdirs.append(fdir0)
        else:
            if os.path.isdir(os.path.join(fdir, fdir0)):
                if full:
                    fdirs.append('%s/%s' % (fdir, fdir0))
                else:
                    fdirs.append(fdir0)
    return sorted(fdirs)

def get_file(fdir, full=False, contains=[], verbose=False):

    fnames = []

    for filename0 in os.listdir(fdir):
        if len(contains) > 0:
            if os.path.isfile(os.path.join(fdir, filename0)) and all([(string in filename0) for string in contains]):
                if full:
                    fnames.append('%s/%s' % (fdir, filename0))
                else:
                    fnames.append(filename0)
        else:
            if os.path.isdir(os.path.join(fdir, filename0)):
                if full:
                    fnames.append('%s/%s' % (fdir, filename0))
                else:
                    fnames.append(filename0)

    Nfile = len(fnames)
    if Nfile == 0:
        sys.exit('Error   [get_file]: Cannot find file.')
    elif Nfile > 1:
        if verbose:
            print('Warning [get_file]: Find more %d files that matches.' % Nfile)
        return fnames
    elif Nfile == 1:
        fname = fnames[0]
        return fname

def dtime_to_jday(dtime):

    jday = (dtime - datetime.datetime(1, 1, 1)).total_seconds()/86400.0 + 1.0

    return jday

def jday_to_dtime(jday):

    dtime = datetime.datetime(1, 1, 1) + dt.timedelta(seconds=(jday-1)*86400.0)

    return dtime



def cal_time_offset(x_ref, x_target, fill_value=np.nan, offset_range=[-900, 900]):

    logic = (np.logical_not(np.isnan(x_ref))) & (np.logical_not(np.isnan(x_target)))
    x_ref    = x_ref[logic]
    x_target = x_target[logic]
    x_ref    = x_ref/x_ref.max()
    x_target = x_target/x_target.max()

    offsets = np.arange(offset_range[0], offset_range[1], dtype=np.float64)
    cross_corr = np.zeros_like(offsets)
    for i, offset in enumerate(offsets):
        x0 = shift(x_target, offset, cval=fill_value)
        logic = (np.logical_not(np.isnan(x0))) & (np.logical_not(np.isnan(x_ref)))
        coef = pearsonr(x_ref[logic], x0[logic])
        cross_corr[i] = coef[0]

    time_offset = offsets[np.argmax(cross_corr)]

    return time_offset



def func_diff_ratio(x, a, b, c):

    return a * (x/500.0)**(b) + c

def fit_diff_ratio(wavelength, ratio):

    popt, pcov = curve_fit(func_diff_ratio, wavelength, ratio, maxfev=1000000, bounds=(np.array([0.0, -np.inf, 0.0]), np.array([np.inf, 0.0, np.inf])))

    return popt, pcov

def cal_diff_ratio_from_spns(fname_ssfr, fname_spns, wvl_s=400.0, wvl_e=750.0, Nmin=20):

    data_ssfr  = load_h5(fname_ssfr)
    data_spns  = load_h5(fname_spns)

    dif_flux = np.zeros_like(data_ssfr['zen_cnt']); dif_flux[...] = np.nan
    tot_flux = np.zeros_like(data_ssfr['zen_cnt']); tot_flux[...] = np.nan
    for i in range(data_ssfr['tmhr'].size):
        dif_flux[i, :] = interp(data_ssfr['zen_wvl'], data_spns['wvl'], data_spns['dif_flux'][i, :])
        tot_flux[i, :] = interp(data_ssfr['zen_wvl'], data_spns['wvl'], data_spns['tot_flux'][i, :])


    diff_ratio0 = dif_flux / tot_flux
    diff_ratio  = np.zeros_like(data_ssfr['zen_cnt']); diff_ratio[...] = np.nan
    coefs       = np.zeros((data_ssfr['tmhr'].size, 3)); coefs[...] = np.nan
    qual_flag   = np.repeat(0, data_ssfr['tmhr'].size)

    for i in tqdm(range(diff_ratio.shape[0])):

        logic = (diff_ratio0[i, :]>=0.0) & (diff_ratio0[i, :]<=1.0) & (data_ssfr['zen_wvl']>=wvl_s) & (data_ssfr['zen_wvl']<=wvl_e)
        if logic.sum() > Nmin:

            x = data_ssfr['zen_wvl'][logic]
            y = diff_ratio0[i, logic]
            popt, pcov = fit_diff_ratio(x, y)

            diff_ratio[i, :] = func_diff_ratio(data_ssfr['zen_wvl'], *popt)
            diff_ratio[i, diff_ratio[i, :]>1.0] = 1.0
            diff_ratio[i, diff_ratio[i, :]<0.0] = 0.0

            coefs[i, :] = popt
            qual_flag[i] = 1

    for i in range(diff_ratio.shape[1]):
        logic_nan = np.isnan(diff_ratio[:, i])
        logic     = np.logical_not(logic_nan)

        f_interp  = interpolate.interp1d(data_ssfr['tmhr'][logic], diff_ratio[:, i][logic], bounds_error=None, fill_value='extrapolate')
        diff_ratio[logic_nan, i] = f_interp(data_ssfr['tmhr'][logic_nan])
        diff_ratio[diff_ratio[:, i]>1.0, i] = 1.0
        diff_ratio[diff_ratio[:, i]<0.0, i] = 0.0

    data_ratio = {}
    data_ratio['diff_ratio_x']         = data_ssfr['zen_wvl']
    data_ratio['diff_ratio_coef']      = coefs
    data_ratio['diff_ratio_qual_flag'] = qual_flag
    data_ratio['diff_ratio_ori']       = diff_ratio0
    data_ratio['diff_ratio']           = diff_ratio

    return data_ratio



def get_ang_cal_oracles(date, fdir_cal, tags=[]):

    if len(tags) == 0:
        tags = ['.h5']
        if date < datetime.datetime(2018, 1, 1):
            tags += ['pre']
        else:
            tags += ['post']

    # During ORACLES, the zenith light collector (LC1) was found condensation
    # we replaced it with CUZ1 after August 30th, 2018
    if date < datetime.datetime(2018, 1, 1):
        zen_tags = tags + ['CUZ1']
    else:
        zen_tags = tags + ['LC1']


    nad_tags = tags + ['LC2']

    zen_tags += ['20190611']
    nad_tags += ['20190423']

    fnames_ang_cal = {}

    fnames_zen = get_file(fdir_cal, full=True, contains=zen_tags)
    if type(fnames_zen) is list:
        sys.exit('Error   [get_ang_cal_oracles]: Found more than one angular calibration file for zenith.')
    elif type(fnames_zen) is str:
        fnames_ang_cal['zenith'] = fnames_zen

    fnames_nad = get_file(fdir_cal, full=True, contains=nad_tags)
    if type(fnames_nad) is list:
        sys.exit('Error   [get_ang_cal_oracles]: Found more than one angular calibration file for nadir.')
    elif type(fnames_nad) is str:
        fnames_ang_cal['nadir'] = fnames_nad

    return fnames_ang_cal

def get_rad_cal_oracles(date, fdir_cal, tags=[]):

    if len(tags) == 0:
        tags = ['.h5']
        if date < datetime.datetime(2018, 1, 1):
            tags += ['pre']
        else:
            tags += ['post']

    # During ORACLES
    if date < datetime.datetime(2018, 1, 1):
        zen_tags = tags + ['CUZ1']
    else:
        zen_tags = tags + ['LC1']

    nad_tags = tags + ['LC2']

    zen_tags += ['20190128']
    nad_tags += ['20190128']

    jday0 = dtime_to_jday(date)
    fnames_rad_cal = {}

    fnames_zen = get_file(fdir_cal, full=True, contains=zen_tags)
    if type(fnames_zen) is list:

        jdays = np.zeros(len(fnames_zen), dtype=np.float64)
        for i, fname in enumerate(fnames_zen):
            date0 = datetime.datetime.strptime(os.path.basename(fname).split('_')[1][:8], '%Y%m%d')
            jdays[i] = dtime_to_jday(date0)
        index = np.argmin(np.abs(jdays-jday0))
        fnames_rad_cal['zenith'] = fnames_zen[index]

    elif type(fnames_zen) is str:
        fnames_rad_cal['zenith'] = fnames_zen

    fnames_nad = get_file(fdir_cal, full=True, contains=nad_tags)
    if type(fnames_nad) is list:

        jdays = np.zeros(len(fnames_nad), dtype=np.float64)
        for i, fname in enumerate(fnames_nad):
            date0 = datetime.datetime.strptime(os.path.basename(fname).split('_')[1][:8], '%Y%m%d')
            jdays[i] = dtime_to_jday(date0)
        index = np.argmin(np.abs(jdays-jday0))
        fnames_rad_cal['nadir'] = fnames_nad[index]

    elif type(fnames_nad) is str:
        fnames_rad_cal['nadir'] = fnames_nad

    return fnames_rad_cal



def cal_diff_532(date,
                 fdir_processed,
                 fdir_cal, pitch_angle, roll_angle):

    date_s = date.strftime('%Y%m%d')

    fname_hsk = get_file(fdir_processed, full=True, contains=['hsk_%s' % date_s])
    data_hsk = load_h5(fname_hsk)

    fname_ssfr_v0 = get_file(fdir_processed, full=True, contains=['ssfr', 'v0', date_s])
    ssfr_v0 = nasa_ssfr(fnames=[], fname_v0=fname_ssfr_v0)

    fname_ssfr_aux = get_file(fdir_processed, full=True, contains=['ssfr', 'aux', date_s])
    ssfr_aux = load_h5(fname_ssfr_aux)

    # calculate cosine correction factors
    # ?? what angles to use for pitch/roll motor, offset??
    # ========================================================================================
    angles = {}
    angles['solar_zenith']  = ssfr_aux['sza']
    angles['solar_azimuth'] = ssfr_aux['saa']
    angles['pitch']         = interp(ssfr_v0.tmhr, data_hsk['tmhr'], data_hsk['pitch_angle'])
    angles['roll']          = interp(ssfr_v0.tmhr, data_hsk['tmhr'], data_hsk['roll_angle'])
    angles['heading']       = interp(ssfr_v0.tmhr, data_hsk['tmhr'], data_hsk['true_heading'])
    angles['pitch_motor']   = np.repeat(pitch_angle, ssfr_v0.tmhr.size)
    angles['roll_motor']    = np.repeat(roll_angle, ssfr_v0.tmhr.size)
    angles['pitch_offset']  = 0.0
    angles['roll_offset']   = 0.0

    fdir_ang_cal = '%s/ang-cal' % fdir_cal
    fnames_ang_cal = get_ang_cal_oracles(date, fdir_ang_cal)
    factors = cos_corr(fnames_ang_cal, angles, diff_ratio=ssfr_aux['diff_ratio'])

    # apply cosine correction
    ssfr_v0.zen_cnt = ssfr_v0.zen_cnt*factors['zenith']
    ssfr_v0.nad_cnt = ssfr_v0.nad_cnt*factors['nadir']
    # ========================================================================================

    # primary transfer calibration
    # ?? how to determine which calibration file to use ??
    # ========================================================================================
    fdir_rad_cal = '%s/rad-cal' % fdir_cal
    fnames_rad_cal = get_rad_cal_oracles(date, fdir_rad_cal)
    ssfr_v0.cal_flux(fnames_rad_cal)
    # ========================================================================================

    zen_flux = np.zeros((data_hsk['tmhr'].size, ssfr_v0.zen_wvl.size), dtype=np.float64)
    for i in range(ssfr_v0.zen_wvl.size):
        zen_flux[:, i] = interp(data_hsk['tmhr'], ssfr_v0.tmhr, ssfr_v0.zen_flux[:, i])

    nad_flux = np.zeros((data_hsk['tmhr'].size, ssfr_v0.nad_wvl.size), dtype=np.float64)
    for i in range(ssfr_v0.nad_wvl.size):
        nad_flux[:, i] = interp(data_hsk['tmhr'], ssfr_v0.tmhr, ssfr_v0.nad_flux[:, i])
    # ========================================================================================

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fname_spns = get_file(fdir_processed, full=True, contains=['spns', 'v0', date_s])
    data_spns  = load_h5(fname_spns)

    index_spns = np.argmin(np.abs(532.0-data_spns['tot_wvl']))
    zen_flux_spns = interp(data_hsk['tmhr'], data_spns['tot_tmhr'], data_spns['tot_flux'][:, index_spns])

    index_ssfr = np.argmin(np.abs(532.0-ssfr_v0.zen_wvl))
    zen_flux_ssfr = zen_flux[:, index_ssfr]

    logic = np.logical_not(np.isnan(zen_flux_ssfr)) & (np.logical_not(np.isnan(zen_flux_spns)))

    diff = np.abs(zen_flux_spns[logic]-zen_flux_ssfr[logic])

    return diff.sum()
    # ============================================================================
# ================================================================================================





# calibrations
# ================================================================================================
def cdata_cos_resp_oracles(
        fdir,
        angles = np.array([ 0.0,  5.0,  10.0,  15.0,  20.0,  25.0,  30.0,  40.0,  50.0,  60.0,  70.0,  80.0,  90.0, \
                            0.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0, -90.0, 0.0]),
        plot=True,
        intTime={'si':60, 'in':300}
        ):

    date_s = os.path.basename(fdir).split('_')[0]

    lc_all = [dir0 for dir0 in os.listdir(fdir) if os.path.isdir(os.path.join(fdir, dir0)) and ('nadir' in dir0.lower() or 'zenith' in dir0.lower())]

    for lc in lc_all:

        fnames = sorted(glob.glob('%s/%s/s%di%d/pos/*.OSA2' % (fdir, lc, intTime['si'], intTime['in']))) + \
                 sorted(glob.glob('%s/%s/s%di%d/neg/*.OSA2' % (fdir, lc, intTime['si'], intTime['in'])))

        fnames_cal = OrderedDict()
        for i, fname in enumerate(fnames):
            fnames_cal[fname] = angles[i]

        if 'nadir' in lc.lower():
            which = 'nadir'
        elif 'zenith' in lc.lower():
            which = 'zenith'

        filename_tag = '%s_%s' % (fdir, lc.replace('_', '-'))

        fname_cal = cdata_cos_resp(fnames_cal, filename_tag=filename_tag, which=which, Nchan=256, wvl_join=950.0, wvl_start=350.0, wvl_end=2100.0, intTime=intTime)

        if plot:
            plot_cos_resp_oracles(fname_cal)

def cdata_rad_resp_oracles(
        fdir_lab,
        fdir_field=None,
        plot=True,
        intTime={'si':60, 'in':300},
        field_lamp_tag='150',
        ):

    dirs = get_sub_dir(fdir_lab, full=False)

    if len(dirs) != 2:
        sys.exit('Error [cdata_rad_resp_oracles]: Incomplete lab radiometric calibration dataset.')

    if field_lamp_tag in dirs[0]:
        index_tra = 0
        index_pri = 1
    elif field_lamp_tag in dirs[1]:
        index_tra = 1
        index_pri = 0
    else:
        sys.exit('Error [cdata_rad_resp_oracles]: Cannot locate lab radiometric calibration for field lamp.')

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
        fname_cal = cdata_rad_resp(fnames_pri=fnames_pri, fnames_tra=fnames_tra, fnames_sec=fnames_sec, filename_tag=filename_tag, which=which, wvl_join=950.0, wvl_start=350.0, wvl_end=2100.0, intTime=intTime, pri_lamp_tag=pri_lamp_tag)

        if plot:
            plot_rad_resp_oracles(fname_cal)

def cdata_cal(fdir_data='/argus/field/oracles/2018', platform='p3', run=True):

    # ORACLES 2018

    # Angular Calibrations
    # =====================================================================================
    if run:
        fdir0 = '%s/%s/calibration/ang-cal' % (fdir_data, platform)
        fdirs = ['%s/%s' % (fdir0, dir0) for dir0 in os.listdir(fdir0) if os.path.isdir(os.path.join(fdir0, dir0))]
        for fdir in fdirs:
            cdata_cos_resp_oracles(fdir)
    # =====================================================================================

    # Radiometric Calibrations
    # =====================================================================================
    if run:
        fdir0 = '%s/%s/calibration/rad-cal' % (fdir_data, platform)
        fdirs_lab   = get_sub_dir(fdir0, full=True, contains=['pre', 'post'])
        fdirs_field = get_sub_dir(fdir0, full=True, contains=['field'])
        for fdir_lab in fdirs_lab:
            for fdir_field in fdirs_field:
                cdata_rad_resp_oracles(fdir_lab, fdir_field=fdir_field)
    # =====================================================================================
# ================================================================================================





# data processing
# ================================================================================================
def cdata_hsk_v0_hsk(date, fdir_out, fdir_data='/argus/field/oracles', run=True):

    date_s = date.strftime('%Y%m%d')

    fdir_raw  = '%s/hsk' % fdir_data
    fname  = get_file(fdir_raw, full=True, contains=[date_s])

    data = read_ict(fname)

    fname_hsk   = '%s/hsk_%s_v0-hsk.h5' % (fdir_out, date_s)
    if run:
        f = h5py.File(fname_hsk, 'w')
        f.attrs['description'] = 'v0-hsk: raw data from %s' % fname
        for vname in data.keys():
            f[vname] = data[vname]['data']

        f['jday'] = dtime_to_jday(date) + data['tmhr']['data']/24.0
        f.close()

    return fname_hsk




def cdata_alp_v0_raw(date, fdir_processed, fdir_data='/argus/field/oracles', run=True):

    date_s = date.strftime('%Y%m%d')

    fdir_raw  = '%s/raw/alp' % (fdir_data)
    fnames_alp_raw = sorted(glob.glob('%s/*.plt3' % fdir_raw))

    fname_alp      = '%s/alp_%s_v0-raw.h5' % (fdir_processed, date_s)
    if run:

        if date.year == 2016:
            from alp import cu_alp_v2 as cu_alp
        else:
            from alp import cu_alp

        # create ALP raw data
        # ============================================================================
        alp0 = cu_alp(fnames_alp_raw, date=date)
        alp0.save_h5(fname_alp)
        # ============================================================================

    return fname_alp

def cdata_alp_v1_hsk(date, fdir_processed, fdir_data='/argus/field/oracles', run=True):

    date_s = date.strftime('%Y%m%d')

    fname_hsk = get_file(fdir_processed, full=True, contains=['hsk_%s' % date_s, 'v0-hsk', date_s])
    data_hsk = load_h5(fname_hsk)

    fname_alp = get_file(fdir_processed, full=True, contains=['alp', 'v0-raw', date_s])
    data_alp  = load_h5(fname_alp)

    time_offset = cal_time_offset(data_hsk['gps_altitude'], interp(data_hsk['tmhr'], data_alp['tmhr'], data_alp['alt']))

    fname_alp = '%s/alp_%s_v1-hsk.h5' % (fdir_processed, date_s)
    if run:

        f = h5py.File(fname_alp, 'w')
        f.attrs['description'] = 'v1-hsk: raw data interpolated to HSK time frame; time offset (seconds) was calculated.'

        f['tmhr']        = data_hsk['tmhr'] + time_offset/3600.0
        f['tmhr_ori']    = data_hsk['tmhr']
        f['time_offset'] = time_offset
        for vname in data_alp.keys():
            if vname not in ['tmhr', 'jday']:
                f[vname] = interp(data_hsk['tmhr'], data_alp['tmhr'], data_alp[vname])
        f.close()

    return fname_alp



def cdata_ssfr_v0_raw(date, fdir_processed, fdir_data='/argus/field/oracles', run=True):

    """
    version 0: counts after dark correction
    """

    date_s = date.strftime('%Y%m%d')

    fdir_raw  = '%s/raw/ssfr' % (fdir_data)

    # create SSFR V0 data (counts after dark correction)
    # ============================================================================
    fnames_ssfr = sorted(glob.glob('%s/*.OSA2' % fdir_raw))
    ssfr0 = nasa_ssfr(fnames_ssfr, date_ref=date)
    ssfr0.pre_process(wvl_join=950.0, wvl_start=350.0, wvl_end=2100.0, intTime={'si':60, 'in':300})

    fname_ssfr_v0 = '%s/ssfr_%s_v0-raw.h5' % (fdir_processed, date_s)
    if run:
        f = h5py.File(fname_ssfr_v0, 'w')
        f.attrs['description'] = 'v0-raw: Raw data (counts); dark correction was applied'
        f['tmhr']    = ssfr0.tmhr
        f['jday']    = ssfr0.jday
        f['shutter'] = ssfr0.shutter
        f['zen_wvl'] = ssfr0.zen_wvl
        f['nad_wvl'] = ssfr0.nad_wvl
        f['zen_cnt'] = ssfr0.zen_cnt
        f['nad_cnt'] = ssfr0.nad_cnt
        f['zen_int_time'] = ssfr0.zen_int_time
        f['nad_int_time'] = ssfr0.nad_int_time
        f.close()

    return fname_ssfr_v0
    # ============================================================================

def cdata_ssfr_v1_hsk(date, fdir_processed, fdir_data='/argus/field/oracles', run=True):

    date_s = date.strftime('%Y%m%d')

    fname_hsk = get_file(fdir_processed, full=True, contains=['hsk_%s' % date_s, 'v0-hsk', date_s])
    data_hsk = load_h5(fname_hsk)

    fname_ssfr_v0 = get_file(fdir_processed, full=True, contains=['ssfr', 'v0-raw', date_s])
    ssfr_v0 = load_h5(fname_ssfr_v0)

    zen_cnt  = np.zeros((data_hsk['tmhr'].size, ssfr_v0['zen_wvl'].size), dtype=np.float64)
    for i in range(ssfr_v0['zen_wvl'].size):
        zen_cnt[:, i] = interp(data_hsk['tmhr'], ssfr_v0['tmhr'], ssfr_v0['zen_cnt'][:, i])

    nad_cnt = np.zeros((data_hsk['tmhr'].size, ssfr_v0['nad_wvl'].size), dtype=np.float64)
    for i in range(ssfr_v0['nad_wvl'].size):
        nad_cnt[:, i] = interp(data_hsk['tmhr'], ssfr_v0['tmhr'], ssfr_v0['nad_cnt'][:, i])

    shutter      = interp(data_hsk['tmhr'], ssfr_v0['tmhr'], ssfr_v0['shutter'], mode='nearest')

    fname_ssfr = '%s/ssfr_%s_v1-hsk.h5' % (fdir_processed, date_s)
    if run:
        f = h5py.File(fname_ssfr, 'w')
        f.attrs['description'] = 'hsk: Raw data (counts) interpolated to HSK time frame; dark correction was applied'
        f['tmhr']    = data_hsk['tmhr']
        f['jday']    = dtime_to_jday(date) + data_hsk['tmhr']/24.0
        f['zen_wvl'] = ssfr_v0['zen_wvl']
        f['nad_wvl'] = ssfr_v0['nad_wvl']
        f['zen_cnt'] = zen_cnt
        f['nad_cnt'] = nad_cnt
        f['shutter']      = shutter
        f['zen_int_time'] = ssfr_v0['zen_int_time']
        f['nad_int_time'] = ssfr_v0['nad_int_time']
        f.close()

    return fname_ssfr



def cdata_spns_v0_raw(date, fdir_processed, fdir_data='/argus/field/oracles', run=True):

    """
    V0: only read in the raw datasets into HDF5
    """

    import spn

    date_s = date.strftime('%Y%m%d')

    fdir_raw  = '%s/raw/spns' % (fdir_data)

    fnames = spn.get_all_files(fdir_raw)

    for fname in fnames:
        filename = os.path.basename(fname)
        if filename == 'Diffuse.txt':
            f = spn.spn_s(fname, date_ref=date, quiet=True)
            try:
                dif_jday = np.concatenate((dif_jday, f.jday))
                dif_f    = np.concatenate((dif_f, f.flux))
                dif_tmhr = np.concatenate((dif_tmhr, f.tmhr))
            except:
                dif_jday = f.jday
                dif_f    = f.flux
                dif_wvl  = f.wavelength
                dif_tmhr = f.tmhr

        elif filename == 'Total.txt':
            f = spn.spn_s(fname, date_ref=date, quiet=True)
            try:
                tot_jday = np.append(tot_jday, f.jday)
                tot_f    = np.concatenate((tot_f, f.flux))
                tot_tmhr = np.concatenate((tot_tmhr, f.tmhr))
            except:
                tot_jday = f.jday
                tot_f    = f.flux
                tot_wvl  = f.wavelength
                tot_tmhr = f.tmhr

    fname_spns     = '%s/spns_%s_v0-raw.h5' % (fdir_processed, date_s)
    if run:
        f = h5py.File(fname_spns, 'w')
        f.attrs['description'] = 'v0-raw: Raw data'
        f['dif_tmhr'] = dif_tmhr
        f['dif_jday'] = dif_jday
        f['dif_flux'] = dif_f
        f['dif_wvl']  = dif_wvl
        f['tot_tmhr'] = tot_tmhr
        f['tot_jday'] = tot_jday
        f['tot_flux'] = tot_f
        f['tot_wvl']  = tot_wvl
        f.close()

    return fname_spns

def cdata_spns_v1_hsk(date, fdir_processed, fdir_data='/argus/field/oracles', run=True):

    date_s = date.strftime('%Y%m%d')

    fname_hsk = get_file(fdir_processed, full=True, contains=['hsk_%s' % date_s, 'v0-hsk', date_s])
    data_hsk = load_h5(fname_hsk)

    fname_spns = get_file(fdir_processed, full=True, contains=['spns', 'v0-raw', date_s])
    data_spns  = load_h5(fname_spns)

    wvl = data_spns['dif_wvl']
    dif_flux = np.zeros((data_hsk['tmhr'].size, wvl.size), dtype=np.float64)
    tot_flux = np.zeros((data_hsk['tmhr'].size, wvl.size), dtype=np.float64)
    for i in range(wvl.size):
        dif_flux[:, i] = interp(data_hsk['tmhr'], data_spns['dif_tmhr'], data_spns['dif_flux'][:, i])
        tot_flux[:, i] = interp(data_hsk['tmhr'], data_spns['tot_tmhr'], data_spns['tot_flux'][:, i])

    # calculate time offset
    # ====================================================================
    # fname_ssfr = get_file(fdir_processed, full=True, contains=['ssfr', 'v1-hsk', date_s])
    # data_ssfr  = load_h5(fname_ssfr)

    # wvl0 = 532.0
    # data_ref = data_ssfr['zen_cnt'][:, np.argmin(np.abs(data_ssfr['zen_wvl']-wvl0))]
    # data_tar = tot_flux[:, np.argmin(np.abs(wvl-wvl0))]

    # time_offset = cal_time_offset(data_ref, data_tar)

    # print(time_offset)
    # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # fig = plt.figure(figsize=(8, 6))
    # ax1 = fig.add_subplot(111)
    # ax1.scatter(data_ssfr['tmhr'], data_ref/np.nanmax(data_ref), c='k', s=10, lw=0)
    # ax1.scatter(data_ssfr['tmhr'], data_tar/np.nanmax(data_tar), c='r', s=10, lw=0)
    # ax1.scatter(data_ssfr['tmhr']-1.0, data_tar/np.nanmax(data_tar), c='g', s=10, lw=0)
    # # ax1.set_xlim(())
    # # ax1.set_ylim(())
    # # ax1.set_xlabel('')
    # # ax1.set_ylabel('')
    # # ax1.set_title('')
    # # ax1.legend(loc='upper right', fontsize=12, framealpha=0.4)
    # # plt.savefig('test.png')
    # plt.show()
    # exit()
    # ====================================================================
    time_offset = -3600.0

    fname_spns = '%s/spns_%s_v1-hsk.h5' % (fdir_processed, date_s)
    if run:

        f = h5py.File(fname_spns, 'w')
        f.attrs['description'] = 'v1-hsk: Raw data interpolated to HSK time frame.'
        f['time_offset'] = time_offset
        f['tmhr_ori']    = data_hsk['tmhr']
        f['tmhr']        = data_hsk['tmhr'] + time_offset/3600.0
        f['jday']        = data_hsk['jday'] + time_offset/86400.0
        f['wvl']         = wvl
        f['dif_flux']    = dif_flux
        f['tot_flux']    = tot_flux
        f.close()

    return fname_spns



def cdata_aux_v0_hsk(date, fdir_processed, diff_ratio_method=None, run=True):

    date_s = date.strftime('%Y%m%d')

    fname_ssfr = get_file(fdir_processed, full=True, contains=['ssfr', 'v1-hsk', date_s])

    # calculate diffuse-to-global ratio for SSFR
    # ========================================================================================
    if diff_ratio_method is None:
        data_ssfr  = load_h5(fname_ssfr)
        diff_ratio = np.zero_like(data_ssfr['zen_cnt'])
    else:
        if diff_ratio_method.lower() == 'spns':
            fname_spns = get_file(fdir_processed, full=True, contains=['spns', 'v1-hsk', date_s])
            data_ratio = cal_diff_ratio_from_spns(fname_ssfr, fname_spns)
        elif diff_ratio_method.lower() == 'rtm':
            sys.exit('Error   [cdata_aux_v0_hsk]: \'%s\' has not been implemented.' % diff_ratio_method)
    # ========================================================================================

    # calculate solar angles
    # ========================================================================================
    fname_hsk = get_file(fdir_processed, full=True, contains=['hsk_%s' % date_s, 'v0-hsk', date_s])
    data_hsk  = load_h5(fname_hsk)
    sza, saa = cal_solar_angles(data_hsk['jday'], data_hsk['longitude'], data_hsk['latitude'], data_hsk['gps_altitude'])
    # ========================================================================================

    if run:
        fname = '%s/aux_%s_v0-hsk.h5' % (fdir_processed, date_s)
        f = h5py.File(fname, 'w')
        f['jday'] = data_hsk['jday']
        f['tmhr'] = data_hsk['tmhr']
        f['alt']  = data_hsk['gps_altitude']
        f['lon']  = data_hsk['longitude']
        f['lat']  = data_hsk['latitude']

        f['sza']  = sza
        f['saa']  = saa

        if diff_ratio_method is None:
            f['diff_ratio'] = diff_ratio
        else:
            if diff_ratio_method.lower() == 'spns':
                f['diff_ratio_x']         = data_ratio['diff_ratio_x']
                f['diff_ratio_coef']      = data_ratio['diff_ratio_coef']
                f['diff_ratio_qual_flag'] = data_ratio['diff_ratio_qual_flag']
                f['diff_ratio_ori']       = data_ratio['diff_ratio_ori']
                f['diff_ratio']           = data_ratio['diff_ratio']
            elif diff_ratio_method.lower() == 'rtm':
                sys.exit('Error   [cdata_aux_v0_hsk]: \'%s\' has not been implemented.' % diff_ratio_method)

        f.close()

    return fname



def cdata_ssfr_v2_att(date,
                   fdir_processed,
                   fdir_cal,
                   pitch_offset=0.0,
                   roll_offset=0.0,
                   version_info='R0: initial science data release'
                   ):

    date_s = date.strftime('%Y%m%d')

    fname_hsk = get_file(fdir_processed, full=True, contains=['hsk_%s' % date_s, 'v0-hsk', date_s])
    data_hsk = load_h5(fname_hsk)

    fname_aux = get_file(fdir_processed, full=True, contains=['aux_%s' % date_s, 'v0-hsk', date_s])
    data_aux = load_h5(fname_aux)

    fname_ssfr_v1 = get_file(fdir_processed, full=True, contains=['ssfr', 'v1-hsk', date_s])
    ssfr_v1 = nasa_ssfr(fnames=[], fname_raw=fname_ssfr_v1)


    # calculate cosine correction factors
    # need modification
    # ========================================================================================
    angles = {}
    angles['solar_zenith']  = data_aux['sza']
    angles['solar_azimuth'] = data_aux['saa']
    if date not in [datetime.datetime(2018, 10, 2), datetime.datetime(2018, 10, 3)]:
        fname_alp = get_file(fdir_processed, full=True, contains=['alp', 'v1-hsk', date_s])
        data_alp = load_h5(fname_alp)
        angles['pitch']        = data_alp['ang_pit_s']
        angles['roll']         = data_alp['ang_rol_s']
        angles['heading']      = data_hsk['true_heading']
        angles['pitch_motor']  = data_alp['ang_pit_m']
        angles['roll_motor']   = data_alp['ang_rol_m']
        angles['pitch_motor'][np.isnan(angles['pitch_motor'])] = 0.0
        angles['roll_motor'][np.isnan(angles['roll_motor'])]   = 0.0
        angles['pitch_offset']  = pitch_offset
        angles['roll_offset']   = roll_offset

    else:

        angles['pitch']         = data_hsk['pitch_angle']
        angles['roll']          = data_hsk['roll_angle']
        angles['heading']       = data_hsk['true_heading']
        angles['pitch_motor']   = np.repeat(0.0, data_hsk['tmhr'].size)
        angles['roll_motor']    = np.repeat(0.0, data_hsk['tmhr'].size)
        angles['pitch_offset']  = pitch_offset
        angles['roll_offset']   = roll_offset


    # apply cosine correction
    fdir_ang_cal = '%s/ang-cal' % fdir_cal
    fnames_ang_cal = get_ang_cal_oracles(date, fdir_ang_cal)
    factors = cos_corr(fnames_ang_cal, angles, diff_ratio=data_aux['diff_ratio'])

    ssfr_v1.zen_cnt = ssfr_v1.zen_cnt*factors['zenith']
    ssfr_v1.nad_cnt = ssfr_v1.nad_cnt*factors['nadir']
    # ========================================================================================

    # primary transfer calibration
    # ========================================================================================
    fdir_rad_cal = '%s/rad-cal' % fdir_cal
    fnames_rad_cal = get_rad_cal_oracles(date, fdir_rad_cal)
    ssfr_v1.cal_flux(fnames_rad_cal)
    # ========================================================================================

    # comments for the data
    # ========================================================================================
    comments_list = []
    comments_list.append('Bandwidth of Silicon channels (wavelength < 950nm) as defined by the FWHM: 6 nm')
    comments_list.append('Bandwidth of InGaAs channels (wavelength > 950nm) as defined by the FWHM: 12 nm')
    comments_list.append('Pitch angle offset: %.1f degree' % pitch_offset)
    comments_list.append('Roll angle offset: %.1f degree' % roll_offset)

    for key in fnames_rad_cal.keys():
        comments_list.append('Radiometric calibration file (%s): %s' % (key, os.path.basename(fnames_rad_cal[key])))
    for key in fnames_ang_cal.keys():
        comments_list.append('Angular calibration file (%s): %s' % (key, os.path.basename(fnames_ang_cal[key])))
    comments = '\n'.join(comments_list)

    print(date_s)
    print(comments)
    print()
    # ========================================================================================


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fname_spns = get_file(fdir_processed, full=True, contains=['spns', 'v1-hsk', date_s])
    data_spns  = load_h5(fname_spns)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    index = np.argmin(np.abs(532.0-ssfr_v1.zen_wvl))
    ax1.scatter(data_hsk['tmhr'], ssfr_v1.zen_flux[:, index], s=10, lw=0, c='r')

    index = np.argmin(np.abs(532.0-data_spns['wvl']))
    ax1.scatter(data_spns['tmhr'], data_spns['tot_flux'][:, index], s=10, lw=0, c='b')

    # ax1.scatter(ssfr_v1.zen_wvl, ssfr_v1.zen_flux[5000, :], s=10, lw=0, c='r')
    # ax1.scatter(data_spns['wvl'], data_spns['tot_flux'][5000, :], s=10, lw=0, c='b')

    patches_legend = [
                mpatches.Patch(color='red'   , label='SSFR (532nm)'),
                mpatches.Patch(color='blue'  , label='SPN-S (532nm)')
                ]
    ax1.legend(handles=patches_legend, bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=len(patches_legend), mode="expand", borderaxespad=0., frameon=False, handletextpad=0.2, fontsize=16)

    ax1.set_xlabel('Time [Hour]')
    ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
    plt.savefig('%s_532nm.png' % date_s, bbox_inches='tight')
    plt.close(fig)
    # ---------------------------------------------------------------------


    fname_ssfr = '%s/ssfr_%s_v2-att.h5' % (fdir_processed, date_s)
    f = h5py.File(fname_ssfr, 'w')

    dset = f.create_dataset('comments', data=comments)
    dset.attrs['description'] = 'comments on the data'

    dset = f.create_dataset('info', data=version_info)
    dset.attrs['description'] = 'information on the version'

    dset = f.create_dataset('utc', data=data_hsk['tmhr'])
    dset.attrs['description'] = 'universal time (numbers above 24 are for the next day)'
    dset.attrs['unit'] = 'decimal hour'

    dset = f.create_dataset('altitude', data=data_hsk['gps_altitude'])
    dset.attrs['description'] = 'altitude above sea level (GPS altitude)'
    dset.attrs['unit'] = 'meter'

    dset = f.create_dataset('longitude', data=data_hsk['longitude'])
    dset.attrs['description'] = 'longitude'
    dset.attrs['unit'] = 'degree'

    dset = f.create_dataset('latitude', data=data_hsk['latitude'])
    dset.attrs['description'] = 'latitude'
    dset.attrs['unit'] = 'degree'

    dset = f.create_dataset('zen_wvl', data=ssfr_v1.zen_wvl)
    dset.attrs['description'] = 'center wavelengths of zenith channels (bandwidth see info)'
    dset.attrs['unit'] = 'nm'

    dset = f.create_dataset('nad_wvl', data=ssfr_v1.nad_wvl)
    dset.attrs['description'] = 'center wavelengths of nadir channels (bandwidth see info)'
    dset.attrs['unit'] = 'nm'

    dset = f.create_dataset('zen_flux', data=ssfr_v1.zen_flux)
    dset.attrs['description'] = 'downwelling shortwave spectral irradiance'
    dset.attrs['unit'] = 'W / m2 / nm'

    dset = f.create_dataset('nad_flux', data=ssfr_v1.nad_flux)
    dset.attrs['description'] = 'upwelling shortwave spectral irradiance'
    dset.attrs['unit'] = 'W / m2 / nm'

    dset = f.create_dataset('pitch', data=angles['pitch'])
    dset.attrs['description'] = 'aircraft pitch angle (positive values indicate nose up)'
    dset.attrs['unit'] = 'degree'

    dset = f.create_dataset('roll', data=angles['roll'])
    dset.attrs['description'] = 'aircraft roll angle (positive values indicate right wing down)'
    dset.attrs['unit'] = 'degree'

    dset = f.create_dataset('heading', data=angles['heading'])
    dset.attrs['description'] = 'aircraft heading angle (positive values clockwise, w.r.t north)'
    dset.attrs['unit'] = 'degree'

    dset = f.create_dataset('sza', data=data_aux['sza'])
    dset.attrs['description'] = 'solar zenith angle'
    dset.attrs['unit'] = 'degree'

    dset = f.create_dataset('saa', data=data_aux['saa'])
    dset.attrs['description'] = 'solar azimuth angle'
    dset.attrs['unit'] = 'degree'
    f.close()

    return fname_ssfr
    # ============================================================================




def cdata_ssfr_ict(date,
              fdir_processed,
              wvls = np.array([415, 440, 500, 550, 675, 870, 990, 1020, 1064, 1250, 1650, 2100]),
              version_info = 'R0: initial science data release'
              ):

    date_s = date.strftime('%Y%m%d')

    fname_ssfr = get_file(fdir_processed, full=True, contains=['ssfr_%s_hsk' % date_s])
    data0  = load_h5(fname_ssfr)

    vars_list = ['Time_start'] +\
                ['DN%d' % wvl for wvl in wvls] +\
                ['UP%d' % wvl for wvl in wvls] +\
                ['LON', 'LAT', 'ALT', 'PITCH', 'ROLL', 'HEADING', 'SZA']

    data = OrderedDict()
    for var in vars_list:
        data[var] = {'data':None, 'unit':None, 'description':None}

    for var in data.keys():
        if 'time' in var.lower() or 'utc' in var.lower():
            data[var]['unit'] = 'second'
            data[var]['description'] = '%s, %s, %s' % (var, data[var]['unit'], var)
            data[var]['data'] = data0['utc']*3600.0
        elif 'dn' in var.lower():
            index= np.argmin(np.abs(data0['zen_wvl']-float(var.replace('DN', ''))))
            data[var]['unit'] = 'W m^-2 nm^-1'
            data[var]['description'] = '%s, %s, Rad_IrradianceDownwellingGlobal_InSitu_SC, Downward Shortwave Irradiance at %.4f nm' % (var, data[var]['unit'], data0['zen_wvl'][index])
            data[var]['data'] = data0['zen_flux'][:, index]
            data[var]['data'][data[var]['data']<0.0] = np.nan
        elif 'up' in var.lower():
            index= np.argmin(np.abs(data0['nad_wvl']-float(var.replace('UP', ''))))
            data[var]['unit'] = 'W m^-2 nm^-1'
            data[var]['description'] = '%s, %s, Rad_IrradianceUpwelling_Insitu_SC, Upward Shortwave Irradiance at %.4f nm' % (var, data[var]['unit'], data0['nad_wvl'][index])
            data[var]['data'] = data0['nad_flux'][:, index]
            data[var]['data'][data[var]['data']<0.0] = np.nan
        elif 'lon' in var.lower():
            data[var]['unit'] = 'degree'
            data[var]['description'] = '%s, %s, Platform_Longitude_InSitu_None, Longitude' % (var, data[var]['unit'])
            data[var]['data'] = data0['longitude']
        elif 'lat' in var.lower():
            data[var]['unit'] = 'degree'
            data[var]['description'] = '%s, %s, Platform_Latitude_InSitu_None, Latitude' % (var, data[var]['unit'])
            data[var]['data'] = data0['latitude']
        elif 'alt' in var.lower():
            data[var]['unit'] = 'meter'
            data[var]['description'] = '%s, %s, Platform_AltitudeMSL_InSitu_None, GPS Altitude' % (var, data[var]['unit'])
            data[var]['data'] = data0['altitude']
        elif 'pitch' in var.lower():
            data[var]['unit'] = 'degree'
            data[var]['description'] = '%s, %s, Platform_PitchAngle_InSitu_None, Aircraft Pitch Angle (positive values indicate nose up)' % (var, data[var]['unit'])
            data[var]['data'] = data0['pitch']
        elif 'roll' in var.lower():
            data[var]['unit'] = 'degree'
            data[var]['description'] = '%s, %s, Platform_RollAngle_InSitu_None, Aircraft Roll Angle (positive values indicate right wing down)' % (var, data[var]['unit'])
            data[var]['data'] = data0['roll']
        elif 'heading' in var.lower():
            data[var]['unit'] = 'degree'
            data[var]['description'] = '%s, %s, Platform_HeadingTrue_InSitu_None, Aircraft Heading Angle (positive values clockwise, w.r.t. north)' % (var, data[var]['unit'])
            data[var]['data'] = data0['heading']
        elif 'sza' in var.lower():
            data[var]['unit'] = 'degree'
            data[var]['description'] = '%s, %s, Met_SolarZenithAngle_Insitu_None, Solar Zenith Angle (w.r.t. Earth)' % (var, data[var]['unit'])
            data[var]['data'] = data0['sza']


    version_info_list = version_info.split(':')
    comments = {
            'PI_CONTACT_INFO': 'Address: University of Colorado Boulder, LASP, 3665 Discovery Drive, Boulder, CO 80303; E-mail: Sebastian.Schmidt@lasp.colorado.edu; Phone: (303)492-6423',
            'PLATFORM': 'P3',
            'LOCATION': 'N/A',
            'ASSOCIATED_DATA': 'N/A',
            'INSTRUMENT_INFO': 'SSFR (Solar Spectral Flux Radiometer, 350-2150 nm)',
            'DATA_INFO': 'Reported are only selected wavelengths (SSFR), pitch/roll from leveling platform INS or aircraft, lat/lon/alt/heading from aircraft, sza calculated from time/lon/lat.',
            'UNCERTAINTY': 'Nominal SSFR uncertainty (shortwave): nadir: 5% zenith: 7%',
            'ULOD_FLAG': '-7777',
            'ULOD_VALUE': 'N/A',
            'LLOD_FLAG': '-8888',
            'LLOD_VALUE': 'N/A',
            'DM_CONTACT_INFO': 'N/A',
            'PROJECT_INFO': 'ORACLES experiment out of Sao Tome, September - October 2018',
            'STIPULATIONS_ON_USE': 'This is the initial release of the ORACLES-2018 field dataset. We strongly recommend that you consult the PI, both for updates to the data set, and for the proper and most recent interpretation of the data for specific science use.',
            'OTHER_COMMENTS': 'The full SSFR spectra from 350-2150 nm is also available in the data archive.\n%s' % str(data0['comments']),
            'REVISION': version_info_list[0],
            version_info_list[0]: version_info_list[1]
            }

    fname = 'ORACLES-SSFR_P3B_%s_%s.ict' % (date_s, version_info_list[0])
    fname = write_ict(date, data, fname, comments=comments, mission_info='ORACLES 2018')
    return fname
    # ============================================================================

def cdata_main(fdir, fdir_archive='/argus/field/oracles/2017/p3/archive'):

    date_s    = os.path.basename(fdir).split('_')[0]
    date      = datetime.datetime.strptime(date_s, '%Y%m%d')

    fdir_processed = '%s/processed' % fdir
    if os.path.exists(fdir_processed):
        os.system('rm -rf %s' % fdir_processed)
    if not os.path.exists(fdir_processed):
        os.makedirs(fdir_processed)

    # 1 - HSK (Aircraft housekeeping file) - Done
    fname_hsk_h5   = cdata_hsk_v0_hsk(date, fdir_processed, fdir_data=fdir_archive, run=True)

    # 2 - ALP (Active Leveling Platform)- Done
    # if date not in [datetime.datetime(2018, 10, 2), datetime.datetime(2018, 10, 3)]:
    #     fname_alp_raw  = cdata_alp_v0_raw(date, fdir_processed, fdir_data=fdir, run=True)
    #     fname_alp_hsk  = cdata_alp_v1_hsk(date, fdir_processed, fdir_data=fdir, run=True)

    # 3 - SSFR (Solar Spectral Flux Radiometer)
    # fname_ssfr_raw = cdata_ssfr_v0_raw(date, fdir_processed, fdir_data=fdir, run=True)
    # fname_ssfr_hsk = cdata_ssfr_v1_hsk(date, fdir_processed, fdir_data=fdir, run=True)

    # 4 - SPN-S (Spectral SPN) - Done
    # fname_spns_raw = cdata_spns_v0_raw(date, fdir_processed, fdir_data=fdir, run=True)
    # fname_spns_hsk = cdata_spns_v1_hsk(date, fdir_processed, fdir_data=fdir, run=True)

    # 5 - Auxiliary Data - Done
    # fname_aux_hsk = cdata_aux_v0_hsk(date, fdir_processed, diff_ratio_method='spns')

    # 6 - SSFR (Solar Spectral Flux Radiometer)
    # fdir_cal = '/argus/field/oracles/2018/p3/calibration'
    # fname_ssfr_att = cdata_ssfr_v2_att(date, fdir_processed, fdir_cal, pitch_offset=0.0, roll_offset=0.0, version_info='R0: initial science data release')


    # version_info = 'R0: initial science data release'

    # fname_ssfr_hsk = cdata_ssfr_hsk(date, fdir_processed, fdir_cal, version_info=version_info)
    # command = 'cp "%s" "ORACLES-SSFR_P3B_%s_%s.h5"' % (fname_ssfr_hsk, date_s, version_info.split(':')[0])
    # os.system(command)

    # fname_ict = cdata_ssfr_ict(date, fdir_processed, version_info=version_info)
# ================================================================================================

def copy_cfg(fdir, fdir_archive='/argus/field/oracles/2018/p3/archive'):

    date_s    = os.path.basename(fdir).split('_')[0]
    date      = datetime.datetime.strptime(date_s, '%Y%m%d')

    command = 'cp /argus/field/oracles/oracles_code_archive_20200504/SSFR_ORACLES_2018/p3/%s/%s.cfg ./' % (date_s, date_s)
    os.system(command)




if __name__ == '__main__':

    # process calibration data (angular and radiometric calibration)
    # ================================================================================================
    # cdata_cal(fdir_data='/argus/field/oracles/2018')

    # plot_rad_cal_oracles_time_series(which='zenith-LC1')
    # plot_rad_cal_oracles_time_series(which='zenith-CUZ1')
    # plot_rad_cal_oracles_time_series(which='nadir-LC2')
    # ================================================================================================

    # process data
    # ================================================================================================
    # ORACLES-3 (2018)
    # fdirs = get_sub_dir('/argus/field/oracles/2018/p3', full=True, contains=['tr', 'rf'])
    # for fdir in fdirs:
    #     cdata_main(fdir)
    #     copy_cfg(fdir)

    # ORACLES-2 (2017)
    # fdir_archive = '/argus/field/oracles/2017/p3/archive'
    # fdirs = get_sub_dir('/argus/field/oracles/2017/p3', full=True, contains=['tr', 'rf'])
    # for fdir in fdirs:
    #     cdata_main(fdir, fdir_archive=fdir_archive)
    #     copy_cfg(fdir)

    # ORACLES-1 (2016)
    fdir_archive = '/argus/field/oracles/2016/p3/archive'
    fdirs = get_sub_dir('/argus/field/oracles/2016/p3', full=True, contains=['tr', 'rf'])
    for fdir in fdirs:
        cdata_main(fdir, fdir_archive=fdir_archive)
    #     copy_cfg(fdir)
    # ================================================================================================
