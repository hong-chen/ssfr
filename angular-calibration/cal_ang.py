import os
import sys
import glob
import datetime
import multiprocessing as mp
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav


def CDATA_COS_RESP(fnames, ang, which='nadir', intTimes_si=[45.0, 90.0], intTimes_in=[250.0, 375.0]):

    from ssfr_util import CU_SSFR

    if which.lower() == 'zenith':
        index_si = 0
        index_in = 1
    elif which.lower() == 'nadir':
        index_si = 2
        index_in = 3

    cos = np.cos(np.deg2rad(ang))

    cos_resp_si = {}
    cos_resp_in = {}

    for j in range(len(intTimes_si)):
        intTime_si = intTimes_si[j]
        intTime_in = intTimes_in[j]
        counts_si = np.zeros((256, ang.size), dtype=np.float64)
        counts_in = np.zeros((256, ang.size), dtype=np.float64)
        for i, fname in enumerate(fnames):
            ssfr            = CU_SSFR(fname, dark_corr_mode='dark_interpolate')
            counts_si[:, i] = np.mean(ssfr.spectra_dark_corr[(np.abs(ssfr.int_time[:, index_si]-intTime_si)<0.00001)&(ssfr.shutter==0), :, 0], axis=0)
            counts_in[:, i] = np.mean(ssfr.spectra_dark_corr[(np.abs(ssfr.int_time[:, index_in]-intTime_in)<0.00001)&(ssfr.shutter==0), :, 0], axis=0)

        cos_resp_si[intTime_si] =  counts_si/counts_si[:, 0]
        cos_resp_in[intTime_in] =  counts_in/counts_in[:, 0]









def PLOT():

    if True:
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(111)
        ax1.set_ylim([0, 8000])
        for index in [0, Npos+1, -1]:
            #ax1.plot(np.arange(256), counts_si[:, index], color='r', lw=0.0, alpha=0.2, marker='o', markersize=10)
            #ax1.plot(np.arange(256), counts_in[:, index], color='b', lw=0.0, alpha=0.2, marker='o', markersize=10)
            # ax1.plot(np.arange(256), counts_si[:, index], color='r', lw=0.2, alpha=0.4)
            ax1.plot(np.arange(256), counts_si[:, index], lw=0.2, alpha=0.4)
            ax1.plot(np.arange(256), counts_in[:, index], lw=0.2, alpha=0.4)

        plt.savefig('angle0_%s.svg' % tag)

    cos_resp_si = np.zeros(ang.size, dtype=np.float64)

    cos_resp_si[:Npos] = counts_si[refChan, :Npos]/counts_si[refChan, 0]
    cos_resp_si[Npos:] = counts_si[refChan, Npos:]/counts_si[refChan, 0]
    # cos_resp_si[Npos:] = counts_si[refChan, Npos:]/counts_si[refChan, Npos+1]
    # cos_resp_si[Npos:] = counts_si[refChan, Npos:]/counts_si[refChan, -1]

    cos_resp_in = np.zeros(ang.size, dtype=np.float64)
    cos_resp_in[:Npos] = counts_in[refChan, :Npos]/counts_in[refChan, 0]
    cos_resp_in[Npos:] = counts_in[refChan, Npos:]/counts_in[refChan, 0]
    # cos_resp_in[Npos:] = counts_in[refChan, Npos:]/counts_in[refChan, Npos+1]
    # cos_resp_in[Npos:] = counts_in[refChan, Npos:]/counts_in[refChan, -1]

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(cos[:Npos], cos_resp_si[:Npos], marker='o', color='r', markersize=8, markeredgecolor='none', alpha=0.4, lw=0.0, label='Positive Angle')
    ax1.plot(cos[Npos:], cos_resp_si[Npos:], marker='o', color='b', markersize=8, markeredgecolor='none', alpha=0.4, lw=0.0, label='Negative Angle')
    ax1.plot([0, 1], [0, 1], ls='--', lw=2.0)
    ax1.set_title('%s Silicon (channel %d)' % (tag, refChan+1))
    ax1.set_xlabel('cos(ang)')
    ax1.set_ylabel('response')
    ax1.legend(framealpha=0.3, fontsize=12, loc='upper left')
    plt.savefig('cos_resp_%s_si_%3.3d.png' % (tag, refChan+1))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(cos[:Npos], cos_resp_in[:Npos], marker='o', color='r', markersize=8, markeredgecolor='none', alpha=0.4, lw=0.0, label='Positive Angle')
    ax1.plot(cos[Npos:], cos_resp_in[Npos:], marker='o', color='b', markersize=8, markeredgecolor='none', alpha=0.4, lw=0.0, label='Negative Angle')
    ax1.plot([0, 1], [0, 1], ls='--', lw=2.0)
    ax1.set_title('%s InGaAs (channel %d)' % (tag, refChan+1))
    ax1.set_xlabel('cos(ang)')
    ax1.set_ylabel('response')
    ax1.legend(framealpha=0.3, fontsize=12, loc='upper left')
    plt.savefig('cos_resp_%s_in_%3.3d.png' % (tag, refChan+1))
    plt.close(fig)




if __name__ == '__main__':
    import matplotlib as mpl
    from matplotlib.ticker import FixedLocator, MaxNLocator
    import matplotlib.pyplot as plt
    from matplotlib import rcParams


    ang = np.array([ 0.0,  3.0,  6.0,  9.0,  12.0,  15.0,  18.0,  21.0,  24.0,  27.0,  30.0,  35.0,  40.0,  45.0,  50.0,  60.0,  70.0,  80.0,  90.0, \
                     0.0, -3.0, -6.0, -9.0, -12.0, -15.0, -18.0, -21.0, -24.0, -27.0, -30.0, -35.0, -40.0, -45.0, -50.0, -60.0, -70.0, -80.0, -90.0, 0.0])

    CDATA_COS_RESP(ang, 100, tag='zenith')
    #for refChan in range(256):
        #CDATA_COS_RESP(ang, refChan, tag='nadir')
        #CDATA_COS_RESP(ang, refChan, tag='zenith')
    exit()


    fdir_zenith_s60i300  = '/argus/home/chen/work/06_oracles/04_cal/data/20161222/dist/zenith/s60i300'
    fdir_zenith_s100i300 = '/argus/home/chen/work/06_oracles/04_cal/data/20161222/dist/zenith/s100i300'
    fdir_nadir_s30i100   = '/argus/home/chen/work/06_oracles/04_cal/data/20161222/dist/nadir/s30i100'
    fdir_nadir_s60i200   = '/argus/home/chen/work/06_oracles/04_cal/data/20161222/dist/nadir/s60i200'
    fdir_nadir_s100i300  = '/argus/home/chen/work/06_oracles/04_cal/data/20161222/dist/nadir/s100i300'
    #CDATA_NLIN(dist, tag='nadir')
    #CDATA_NLIN(dist, tag='zenith')
    #exit()

    dist = np.array([50, 45, 50, 40, 50, 35, 50, 55, 50, 60, 50, 65, 50, 70, 50])
    wvl_in, coef_zenith_s60i300  = CDATA_NLIN(dist, tag='zenith', fdir=fdir_zenith_s60i300)
    wvl_in, coef_zenith_s100i300 = CDATA_NLIN(dist, tag='zenith', fdir=fdir_zenith_s100i300)
    wvl_in, coef_nadir_s30i100   = CDATA_NLIN(dist, tag='nadir',  fdir=fdir_nadir_s30i100)
    wvl_in, coef_nadir_s60i200   = CDATA_NLIN(dist, tag='nadir',  fdir=fdir_nadir_s60i200)
    wvl_in, coef_nadir_s100i300  = CDATA_NLIN(dist, tag='nadir',  fdir=fdir_nadir_s100i300)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot(wvl_in, coef_zenith_s60i300[:, -1], label='zenith s60i300')
    ax1.plot(wvl_in, coef_zenith_s100i300[:, -1],label='zenith s100i300')
    ax1.plot(wvl_in, coef_nadir_s30i100[:, -1],label='nadir s30i100')
    ax1.plot(wvl_in, coef_nadir_s60i200[:, -1],label='nadir s60i200')
    ax1.plot(wvl_in, coef_nadir_s100i300[:, -1],label='nadir s100i300')
    #ax1.axhline(0.0, color='k', ls=':')
    ax1.set_ylim([-0.2, 0.2])
    ax1.set_title('Polynomial Coefficient')
    ax1.set_xlabel('Wavelength [nm]')
    plt.legend(fontsize=10, framealpha=0.2, loc='upper left')
    plt.savefig('coef.png')
    plt.show()
