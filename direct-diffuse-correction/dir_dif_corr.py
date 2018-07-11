import os
import sys
import glob
import datetime
import multiprocessing as mp
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import cartopy.crs as ccrs


def CDATA_DIFFUSE_RATIO(date, altitude=np.arange(0.0, 10.1, 0.1), solar_zenith_angle=np.arange(0.0, 90.1, 0.1), wavelength=np.arange(300.0, 4001.0, 50.0)):

    from lrt_util import lrt_cfg, cld_cfg, aer_cfg
    from lrt_util import LRT_V2_INIT, LRT_RUN_MP, LRT_READ_UVSPEC

    for altitude0 in [7.0]:
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        for solar_zenith_angle0 in [0.0, 40.0, 80.0, 89.0]:
            inits = []
            for wavelength0 in wavelength:

                input_file  = 'data/LRT_input_%4.4d.txt' % wavelength0
                output_file = 'data/LRT_output_%4.4d.txt' % wavelength0
                init = LRT_V2_INIT(input_file=input_file, output_file=output_file, date=date, surface_albedo=0.03, solar_zenith_angle=solar_zenith_angle0, wavelength=wavelength0, output_altitude=7.0, lrt_cfg=lrt_cfg, cld_cfg=None, aer_cfg=None)

                inits.append(init)

            LRT_RUN_MP(inits)
            data = LRT_READ_UVSPEC(inits)

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            ax1.plot(wavelength, data.f_down_diffuse/data.f_down, label='SZA=%.1f' % (solar_zenith_angle0))
            # ax1.set_xlim(())
        ax1.set_ylim((0.0, 1.0))
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Diffuse/Total Ratio')
        ax1.legend(loc='upper right', fontsize=16, framealpha=0.4)
        plt.savefig('test.png')
        plt.show()
        exit()
            # ---------------------------------------------------------------------


def CAL_SPEC_DIFF_RATIO_FROM_SPN1(tmhr, wvl, f_dn_diff, f_dn, diff_ratio_spn):

    if f_dn_diff.ndim != 2 or f_dn.ndim != 2:
        exit('Error [CAL_SPEC_DIFF_RATIO_FROM_SPN1]: wrong dimension of diffuse/global downwelling fluxes.')

    # spectral diffuse fraction from model calculations
    diff_ratio_mod = f_dn_diff / f_dn
    diff_ratio_mod[np.isnan(diff_ratio_mod)] = 0.0

    # "clear" fraction
    f      = (1.0-diff_ratio_spn) / (np.trapz((f_dn*(1.0-diff_ratio_mod)), x=wvl, axis=1)/np.trapz(f_dn, x=wvl, axis=1))
    f_spec = np.repeat(f, wvl.size).reshape((-1, wvl.size))

    # spectral diffuse fraction
    diff_ratio_spec = diff_ratio_mod*f_spec + (1.0-f_spec)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    tmhrs = [21.3223, 22.0592, 22.5467]
    colors = ['red', 'blue', 'green']

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Zenith Spectral Diffuse Ratio on 2014-09-11 (Above Cloud)')
    ax1 = fig.add_subplot(211)
    ax1.scatter(tmhr, diff_ratio_spn, label='Diffuse Ratio (SPN1)', color='gray')
    ax1.scatter(tmhr, f, label='"Clear" Fraction (f)', color='k')
    ax1.set_ylim((0.0, 1.0))
    ax1.set_xlabel('Time [Hour]')
    for i, tmhr0 in enumerate(tmhrs):
        ax1.axvline(tmhr0, color=colors[i], ls='--', lw=2.0)

    plt.legend()


    ax2 = fig.add_subplot(212)
    for i, tmhr0 in enumerate(tmhrs):
        index = np.argmin(np.abs(tmhr-tmhr0))
        ax2.scatter(wvl, diff_ratio_spec[index, :], c=colors[i])
        ax2.scatter(wvl, diff_ratio_mod[index, :], c=colors[i], alpha=0.2)

    ax2.set_ylim((0.0, 1.0))
    ax2.set_xlabel('Wavelength [nm]')
    ax2.set_ylabel('Diffuse Ratio')
    # ax1.legend(loc='upper right', fontsize=10, framealpha=0.4)
    plt.savefig('diff_ratio_spec.png')
    plt.show()
    exit()
    # ---------------------------------------------------------------------




if __name__ == '__main__':

    # date = datetime.datetime(2017, 8, 13)
    # CDATA_DIFFUSE_RATIO(date)

    f = h5py.File('data/bbr_info_20140911.h5', 'r')
    diff_ratio_spn = f['diff_ratio'][...]
    f.close()

    f = h5py.File('data/lrt_info_20140911.h5', 'r')
    tmhr      = f['tmhr'][...]
    f_dn_diff = f['f_dn_diff'][...]
    f_dn      = f['f_dn'][...]
    wvl       = f['wvl'][...]
    f.close()

    CAL_SPEC_DIFF_RATIO_FROM_SPN1(tmhr, wvl, f_dn_diff, f_dn, diff_ratio_spn)
