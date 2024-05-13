import os
import sys
import glob
import datetime
import multiprocessing as mp
from collections import OrderedDict
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav

import ssfr.util
import ssfr.corr



__all__ = ['cal_ang_resp', 'cdata_ang_resp']



def cal_ang_resp(
        fnames,
        which_ssfr='lasp|ssfr-a',
        which_lc='zen',
        int_time={'si':60, 'in':300},
        Nchan=256
        ):

    # check SSFR spectrometer
    #/----------------------------------------------------------------------------\#
    which_ssfr = which_ssfr.lower()
    which_lab  = which_ssfr.split('|')[0]
    if which_lab == 'nasa':
        import ssfr.nasa_ssfr as ssfr_toolbox
    elif which_lab == 'lasp':
        import ssfr.lasp_ssfr as ssfr_toolbox
    else:
        msg = '\nError [cal_ang_resp]: <which_ssfr=> does not support <\'%s\'> (only supports <\'nasa|ssfr-6\'> or <\'lasp|ssfr-a\'> or <\'lasp|ssfr-b\'>).' % which_ssfr
        raise ValueError(msg)
    #\----------------------------------------------------------------------------/#


    # check light collector
    #/----------------------------------------------------------------------------\#
    which_lc = which_lc.lower()
    if (which_lc in ['zenith', 'zen', 'z']) | ('zen' in which_lc):
        which_lc = 'zen'
        index_si = 0
        index_in = 1
    elif (which_lc in ['nadir', 'nad', 'n']) | ('nad' in which_lc):
        which_lc = 'nad'
        index_si = 2
        index_in = 3
    else:
        msg = '\nError [cal_ang_resp]: <which_lc=> does not support <\'%s\'> (only supports <\'zenith, zen, z\'> or <\'nadir, nad, n\'>).' % which_lc
        raise ValueError(msg)
    #\----------------------------------------------------------------------------/#


    # si/in tag
    #/----------------------------------------------------------------------------\#
    si_tag = '%s|si' % which_lc
    in_tag = '%s|in' % which_lc

    if si_tag not in int_time.keys():
        int_time[si_tag] = int_time.pop('si')

    if in_tag not in int_time.keys():
        int_time[in_tag] = int_time.pop('in')
    #\----------------------------------------------------------------------------/#

    Nfile = len(fnames)

    counts_si   = np.zeros((Nfile, Nchan), dtype=np.float64)
    counts_in   = np.zeros((Nfile, Nchan), dtype=np.float64)
    for i, fname in enumerate(fnames.keys()):

        ssfr0 = ssfr_toolbox.read_ssfr([fname])

        logic_si = (np.abs(ssfr0.data_raw['int_time'][:, index_si]-int_time[si_tag])<0.00001)
        logic_in = (np.abs(ssfr0.data_raw['int_time'][:, index_in]-int_time[in_tag])<0.00001)

        shutter, counts = ssfr.corr.dark_corr(ssfr0.data_raw['tmhr'][logic_si], ssfr0.data_raw['shutter'][logic_si], ssfr0.data_raw['spectra'][logic_si, :, index_si], mode='interp')
        logic  = (shutter==0)
        counts_si[i, :] = np.nanmean(counts[logic, :], axis=0)

        shutter, counts = ssfr.corr.dark_corr(ssfr0.data_raw['tmhr'][logic_in], ssfr0.data_raw['shutter'][logic_in], ssfr0.data_raw['spectra'][logic_in, :, index_in], mode='interp')
        logic  = (shutter==0)
        counts_in[i, :] = np.nanmean(counts[logic, :], axis=0)

    ang_resp = {
            si_tag: counts_si/(np.tile(counts_si[0, :], Nfile).reshape(Nfile, -1)),
            in_tag: counts_in/(np.tile(counts_in[0, :], Nfile).reshape(Nfile, -1)),
            }

    return ang_resp



def cdata_ang_resp(
        fnames,
        filename_tag=None,
        which_ssfr='lasp|ssfr-a',
        which_lc='zen',
        Nchan=256,
        wvl_joint=950.0,
        wvl_range=[350.0, 2200.0],
        int_time={'si':60, 'in':300},
        verbose=True
        ):

    # check SSFR spectrometer
    #/----------------------------------------------------------------------------\#
    which_ssfr = which_ssfr.lower()
    which_lab  = which_ssfr.split('|')[0]
    if which_lab == 'nasa':
        import ssfr.nasa_ssfr as ssfr_toolbox
    elif which_lab == 'lasp':
        import ssfr.lasp_ssfr as ssfr_toolbox
    else:
        msg = '\nError [cdata_ang_resp]: <which_ssfr=> does not support <\'%s\'> (only supports <\'nasa|ssfr-6\'> or <\'lasp|ssfr-a\'> or <\'lasp|ssfr-b\'>).' % which_ssfr
        raise ValueError(msg)
    #\----------------------------------------------------------------------------/#


    # check light collector
    #/----------------------------------------------------------------------------\#
    which_lc = which_lc.lower()
    if (which_lc in ['zenith', 'zen', 'z']) | ('zen' in which_lc):
        which_lc = 'zen'
        index_si = 0
        index_in = 1
    elif (which_lc in ['nadir', 'nad', 'n']) | ('nad' in which_lc):
        which_lc = 'nad'
        index_si = 2
        index_in = 3
    else:
        msg = '\nError [cdata_ang_resp]: <which_lc=> does not support <\'%s\'> (only supports <\'zenith, zen, z\'> or <\'nadir, nad, n\'>).' % which_lc
        raise ValueError(msg)
    #\----------------------------------------------------------------------------/#


    # si/in tag
    #/----------------------------------------------------------------------------\#
    si_tag = '%s|si' % which_lc
    in_tag = '%s|in' % which_lc

    if si_tag not in int_time.keys():
        int_time[si_tag] = int_time.pop('si')

    if in_tag not in int_time.keys():
        int_time[in_tag] = int_time.pop('in')
    #\----------------------------------------------------------------------------/#


    # get cosine response (aka angular response)
    #/----------------------------------------------------------------------------\#
    ang_resp_ = cal_ang_resp(fnames, which_ssfr=which_ssfr, which_lc=which_lc, Nchan=Nchan, int_time=int_time)
    #\----------------------------------------------------------------------------/#


    # average the data from two different directions
    #/----------------------------------------------------------------------------\#
    angles = np.array([fnames[fname] for fname in fnames.keys()])
    ang_mu = np.cos(np.deg2rad(angles))

    ang_mu0 = np.sort(np.unique(ang_mu))[::-1]
    Nmu0    = ang_mu0.size

    ang_resp0 = {
            si_tag: np.zeros((Nmu0, Nchan), dtype=np.float64),
            in_tag: np.zeros((Nmu0, Nchan), dtype=np.float64)
            }

    ang_resp_std0 = {
            si_tag: np.zeros((Nmu0, Nchan), dtype=np.float64),
            in_tag: np.zeros((Nmu0, Nchan), dtype=np.float64)
            }

    for i, mu0 in enumerate(ang_mu0):
        indices = np.where(ang_mu==mu0)[0]
        if indices.size in [2, 3]:
            ang_resp0[si_tag][i, :] = np.nanmean(ang_resp_[si_tag][indices, :], axis=0)
            ang_resp0[in_tag][i, :] = np.nanmean(ang_resp_[in_tag][indices, :], axis=0)
            ang_resp_std0[si_tag][i, :] = np.nanstd(ang_resp_[si_tag][indices, :], axis=0)
            ang_resp_std0[in_tag][i, :] = np.nanstd(ang_resp_[in_tag][indices, :], axis=0)
    #\----------------------------------------------------------------------------/#


    # gridding the data
    #/----------------------------------------------------------------------------\#
    ang_mu_all   = np.linspace(0.0, 1.0, 1001)
    Nmu_all      = ang_mu_all.size
    ang_resp_all = {
            si_tag: np.zeros((Nmu_all, Nchan), dtype=np.float64),
            in_tag: np.zeros((Nmu_all, Nchan), dtype=np.float64)
            }
    ang_resp_std_all = {
            si_tag: np.zeros((Nmu_all, Nchan), dtype=np.float64),
            in_tag: np.zeros((Nmu_all, Nchan), dtype=np.float64)
            }

    for i in range(Nchan):

        f = interpolate.interp1d(ang_mu0, ang_resp0[si_tag][:, i], fill_value='extrapolate')
        ang_resp_all[si_tag][:, i] = f(ang_mu_all)

        f = interpolate.interp1d(ang_mu0, ang_resp0[in_tag][:, i], fill_value='extrapolate')
        ang_resp_all[in_tag][:, i] = f(ang_mu_all)

        f = interpolate.interp1d(ang_mu0, ang_resp_std0[si_tag][:, i], fill_value='extrapolate')
        ang_resp_std_all[si_tag][:, i] = f(ang_mu_all)

        f = interpolate.interp1d(ang_mu0, ang_resp_std0[in_tag][:, i], fill_value='extrapolate')
        ang_resp_std_all[in_tag][:, i] = f(ang_mu_all)
    #\----------------------------------------------------------------------------/#


    # wavelength fitting
    #/----------------------------------------------------------------------------\#
    wvls = ssfr_toolbox.get_ssfr_wvl(which_ssfr)

    wvl_start = wvl_range[0]
    wvl_end   = wvl_range[-1]
    logic_si  = (wvls[si_tag] >= wvl_start)  & (wvls[si_tag] <= wvl_joint)
    logic_in  = (wvls[in_tag] >  wvl_joint)  & (wvls[in_tag] <= wvl_end)

    wvl_data      = np.concatenate((wvls[si_tag][logic_si], wvls[in_tag][logic_in]))
    ang_resp_data = np.concatenate((ang_resp_all[si_tag][:, logic_si], ang_resp_all[in_tag][:, logic_in]), axis=1)

    indices_sort = np.argsort(wvl_data)
    wvl          = wvl_data[indices_sort]
    ang_resp = ang_resp_data[:, indices_sort]

    ang_resp_int = np.zeros(wvl.size, dtype=np.float64)
    for i in range(wvl.size):
        ang_resp_int[i] = np.trapz(ang_resp[:, i], x=ang_mu_all)

    logic = (wvl>=400.0) & (wvl<=2000.0)
    order = 4
    coef  = np.zeros((Nmu_all, order+1), dtype=np.float64)
    for i in range(Nmu_all):
        coef[i, :] = np.polyfit(wvl[logic], ang_resp[i, :][logic], order)
    #\----------------------------------------------------------------------------/#


    # save file
    #/----------------------------------------------------------------------------\#
    if filename_tag is not None:
        fname_out = '%s|ang-resp|%s|%s|si-%3.3d|in-%3.3d.h5' % (filename_tag, which_ssfr, which_lc, int_time[si_tag], int_time[in_tag])
    else:
        fname_out = 'ang-resp|%s|%s|si-%3.3d|in-%3.3d.h5' % (which_ssfr, which_lc, int_time[si_tag], int_time[in_tag])

    info = 'Light Collector: %s\nJoint Wavelength: %.4fnm\nStart Wavelength: %.4fnm\nEnd Wavelength: %.4fnm\nIntegration Time for Silicon Channel: %dms\nIntegration Time for InGaAs Channel: %dms\nProcessed Files:\n' % (which_lc.title(), wvl_joint, wvl_start, wvl_end, int_time[si_tag], int_time[in_tag])
    for key in fnames.keys():
        line = 'At %3d [degree] Angle: %s\n' % (fnames[key], key)
        info += line

    if verbose:
        print(info)

    f = h5py.File(fname_out, 'w')

    g = f.create_group('raw')
    g['ang'] = angles
    g['mu']  = ang_mu
    g['mu0'] = ang_mu0

    for spec_tag in ang_resp_.keys():
        g_ = g.create_group(spec_tag)
        g_['wvl'] = wvls[spec_tag]
        g_['ang_resp'] = ang_resp_[spec_tag]
        g_['ang_resp0'] = ang_resp0[spec_tag]
        g_['ang_resp_std0'] = ang_resp_std0[spec_tag]

    f['wvl']          = wvl
    f['mu']           = ang_mu_all
    f['ang_resp']     = ang_resp
    f['ang_resp_int'] = ang_resp_int
    f['poly_coef']    = coef
    f['info']         = info
    f.close()
    #\----------------------------------------------------------------------------/#


    # plot
    #/----------------------------------------------------------------------------\#
    if False:
        import matplotlib as mpl
        # mpl.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedLocator
        from matplotlib import rcParams
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches
        # import cartopy.crs as ccrs

        index = 900

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)

        ax1.fill_between(wvls[si_tag], ang_resp_all[si_tag][index, :]-ang_resp_std_all[si_tag][index, :], ang_resp_all[si_tag][index, :]+ang_resp_std_all[si_tag][index, :], facecolor='red', lw=0.0, alpha=0.3)
        ax1.scatter(wvls[si_tag], ang_resp_all[si_tag][index, :], s=6, c='red', lw=0.0)

        ax1.fill_between(wvls[in_tag], ang_resp_all[in_tag][index, :]-ang_resp_std_all[in_tag][index, :], ang_resp_all[in_tag][index, :]+ang_resp_std_all[in_tag][index, :], facecolor='blue', lw=0.0, alpha=0.3)
        ax1.scatter(wvls[in_tag], ang_resp_all[in_tag][index, :], s=6, c='blue', lw=0.0)

        ax1.scatter(wvl, ang_resp[index, :], s=3, c='k', lw=0.0)

        wvl_new = np.linspace(wvl.min(), wvl.max(), 1000)
        p0 = np.poly1d(coef[index, :])
        ax1.plot(wvl_new, p0(wvl_new), lw=3.0, alpha=0.6, color='g')

        ax1.axvline(wvl_joint, ls='--', color='gray')

        ax1.set_title('%s\n$\mu$=%.4f (%.2f$^\circ$)' % (os.path.basename(fname_out), ang_mu_all[index], np.rad2deg(np.arccos(ang_mu_all[index]))))

        ax1.set_ylim((0.85, 0.95))
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Response')

        plt.savefig('ang_resp_mu-%06.4f.png' % ang_mu_all[index], bbox_inches='tight')
    #\----------------------------------------------------------------------------/#


    return fname_out



if __name__ == '__main__':

    pass
