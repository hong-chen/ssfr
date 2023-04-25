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

# from ssfr.util import lasp_ssfr, nasa_ssfr, get_nasa_ssfr_wavelength
# from ssfr.corr import dark_corr
import ssfr.util
import ssfr.corr



__all__ = ['cal_cos_resp', 'cdata_cos_resp', 'load_cos_resp_h5']



def cal_cos_resp(
        fnames,
        which_ssfr='lasp',
        which_lc='zenith',
        int_time={'si':60, 'in':300},
        Nchan=256
        ):

    which_lc = which_lc.lower()
    if which_lc == 'zenith':
        index_si = 0
        index_in = 1
    elif which_lc == 'nadir':
        index_si = 2
        index_in = 3

    Nfile = len(fnames)

    counts_si   = np.zeros((Nfile, Nchan), dtype=np.float64)
    counts_in   = np.zeros((Nfile, Nchan), dtype=np.float64)
    for i, fname in enumerate(fnames.keys()):

        ssfr0 = ssfr.util.nasa_ssfr([fname])

        logic_si = (np.abs(ssfr0.int_time[:, index_si]-int_time['si'])<0.00001)
        logic_in = (np.abs(ssfr0.int_time[:, index_in]-int_time['in'])<0.00001)

        shutter, counts = ssfr.corr.dark_corr(ssfr0.tmhr[logic_si], ssfr0.shutter[logic_si], ssfr0.spectra[logic_si, :, index_si], mode='mean')
        logic  = (logic_si) & (shutter==0)
        counts_si[i, :] = np.mean(counts[logic, :], axis=0)

        shutter, counts = ssfr.corr.dark_corr(ssfr0.tmhr[logic_in], ssfr0.shutter[logic_in], ssfr0.spectra[logic_in, :, index_in])
        logic  = (logic_in) & (shutter==0)
        counts_in[i, :] = np.mean(counts[logic, :], axis=0)

    cos_resp = {
            'si': counts_si/(np.tile(counts_si[0, :], Nfile).reshape(Nfile, -1)),
            'in': counts_in/(np.tile(counts_in[0, :], Nfile).reshape(Nfile, -1)),
            }

    return cos_resp



def cdata_cos_resp(
        fnames,
        filename_tag=None,
        which_lc='zenith',
        Nchan=256,
        wvl_joint=950.0,
        wvl_start=350.0,
        wvl_end=2200.0,
        int_time={'si':60, 'in':300},
        verbose=True
        ):

    which_lc = which_lc.lower()

    cos_resp = cal_cos_resp(fnames, which_lc=which_lc, Nchan=Nchan, int_time=int_time)

    angles = np.array([fnames[fname] for fname in fnames.keys()])
    cos_mu = np.cos(np.deg2rad(angles))

    cos_mu0 = np.unique(cos_mu)
    Nmu0    = cos_mu0.size

    cos_resp0 = {'si': np.zeros((Nmu0, Nchan), dtype=np.float64), 'in': np.zeros((Nmu0, Nchan), dtype=np.float64)}
    cos_resp_std0 = {'si': np.zeros((Nmu0, Nchan), dtype=np.float64), 'in': np.zeros((Nmu0, Nchan), dtype=np.float64)}
    for i, mu in enumerate(cos_mu0):
        indices = np.where(cos_mu==mu)[0]
        cos_resp0['si'][i, :] = np.mean(cos_resp['si'][indices, :], axis=0)
        cos_resp0['in'][i, :] = np.mean(cos_resp['in'][indices, :], axis=0)
        cos_resp_std0['si'][i, :] = np.std(cos_resp['si'][indices, :], axis=0)
        cos_resp_std0['in'][i, :] = np.std(cos_resp['in'][indices, :], axis=0)

    cos_mu_all   = np.linspace(0.0, 1.0, 1001)
    Nmu_all      = cos_mu_all.size
    cos_resp_all = {'si': np.zeros((Nmu_all, Nchan), dtype=np.float64), 'in': np.zeros((Nmu_all, Nchan), dtype=np.float64)}
    cos_resp_std_all = {'si': np.zeros((Nmu_all, Nchan), dtype=np.float64), 'in': np.zeros((Nmu_all, Nchan), dtype=np.float64)}

    for i in range(Nchan):

        f = interpolate.interp1d(cos_mu0, cos_resp0['si'][:, i], fill_value='extrapolate')
        cos_resp_all['si'][:, i] = f(cos_mu_all)

        f = interpolate.interp1d(cos_mu0, cos_resp0['in'][:, i], fill_value='extrapolate')
        cos_resp_all['in'][:, i] = f(cos_mu_all)

        f = interpolate.interp1d(cos_mu0, cos_resp_std0['si'][:, i], fill_value='extrapolate')
        cos_resp_std_all['si'][:, i] = f(cos_mu_all)

        f = interpolate.interp1d(cos_mu0, cos_resp_std0['in'][:, i], fill_value='extrapolate')
        cos_resp_std_all['in'][:, i] = f(cos_mu_all)

    wvls = ssfr.util.get_nasa_ssfr_wavelength()

    logic_si = (wvls['%s_si' % which_lc] >= wvl_start) & (wvls['%s_si' % which_lc] <= wvl_joint)
    logic_in = (wvls['%s_in' % which_lc] >  wvl_joint)  & (wvls['%s_in' % which_lc] <= wvl_end)

    wvl_data      = np.concatenate((wvls['%s_si' % which_lc][logic_si], wvls['%s_in' % which_lc][logic_in]))

    cos_resp_data = np.hstack((cos_resp_all['si'][:, logic_si], cos_resp_all['in'][:, logic_in]))
    cos_resp_std_data = np.hstack((cos_resp_std_all['si'][:, logic_si], cos_resp_std_all['in'][:, logic_in]))

    indices_sort = np.argsort(wvl_data)
    wvl      = wvl_data[indices_sort]
    cos_resp = cos_resp_data[:, indices_sort]

    cos_resp_int = np.zeros(wvl.size, dtype=np.float64)
    for i in range(wvl.size):
        cos_resp_int[i] = np.trapz(cos_resp[:, i], x=cos_mu_all)

    logic = (wvl>=400.0) & (wvl<=2000.0)
    order = 4
    coef  = np.zeros((Nmu_all, order+1), dtype=np.float64)
    for i in range(Nmu_all):
        coef[i, :] = np.polyfit(wvl[logic], cos_resp[i, :][logic], order)

    if filename_tag is not None:
        fname_out = '%s_cos-resp_s%3.3di%3.3d.h5' % (filename_tag, int_time['si'], int_time['in'])
    else:
        fname_out = 'cos-resp_s%3.3di%3.3d.h5' % (int_time['si'], int_time['in'])

    # =============================================================================
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

        ax1.fill_between(wvls['%s_si' % which_lc], cos_resp_all['si'][index, :]-cos_resp_std_all['si'][index, :], cos_resp_all['si'][index, :]+cos_resp_std_all['si'][index, :], facecolor='red', lw=0.0, alpha=0.3)
        ax1.scatter(wvls['%s_si' % which_lc], cos_resp_all['si'][index, :], s=6, c='red', lw=0.0)

        ax1.fill_between(wvls['%s_in' % which_lc], cos_resp_all['in'][index, :]-cos_resp_std_all['in'][index, :], cos_resp_all['in'][index, :]+cos_resp_std_all['in'][index, :], facecolor='blue', lw=0.0, alpha=0.3)
        ax1.scatter(wvls['%s_in' % which_lc], cos_resp_all['in'][index, :], s=6, c='blue', lw=0.0)

        ax1.scatter(wvl, cos_resp[index, :], s=3, c='k', lw=0.0)

        wvl_new = np.linspace(wvl.min(), wvl.max(), 1000)
        p0 = np.poly1d(coef[index, :])
        ax1.plot(wvl_new, p0(wvl_new), lw=3.0, alpha=0.6, color='g')

        ax1.axvline(wvl_joint, ls='--', color='gray')

        ax1.set_title('%s\n$\mu$=%.4f (%.2f$^\circ$)' % (os.path.basename(fname_out), cos_mu_all[index], np.rad2deg(np.arccos(cos_mu_all[index]))))

        ax1.set_ylim((0.85, 0.95))
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Response')

        plt.savefig('cos_resp_mu-%06.4f.png' % cos_mu_all[index], bbox_inches='tight')
        plt.show()
        exit()
    # =============================================================================

    info = 'Light Collector: %s\nJoint Wavelength: %.4fnm\nStart Wavelength: %.4fnm\nEnd Wavelength: %.4fnm\nIntegration Time for Silicon Channel: %dms\nIntegration Time for InGaAs Channel: %dms\nProcessed Files:\n' % (which_lc.title(), wvl_joint, wvl_start, wvl_end, int_time['si'], int_time['in'])
    for key in fnames.keys():
        line = 'At %3d [degree] Angle: %s\n' % (fnames[key], key)
        info += line

    if verbose:
        print(info)

    f = h5py.File(fname_out, 'w')
    f['wvl']          = wvl
    f['mu']           = cos_mu_all
    f['cos_resp']     = cos_resp
    f['cos_resp_int'] = cos_resp_int
    f['poly_coef']    = coef
    f['info']         = info
    f.close()

    return fname_out



def load_cos_resp_h5(fname):

    resp = {}
    f = h5py.File(fname, 'r')
    for key in f.keys():
        resp[key] = f[key][...]
    f.close()

    return resp



if __name__ == '__main__':

    pass
