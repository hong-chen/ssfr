import os
import sys
import glob
import datetime
import multiprocessing as mp
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav

from ssfr_util.util import CU_SSFR



def CAL_COS_RESP(fnames, which='nadir', intTimes_si=[45.0, 90.0], intTimes_in=[250.0, 375.0]):

    from ssfr_util import CU_SSFR

    if which.lower() == 'zenith':
        index_si = 0
        index_in = 1
    elif which.lower() == 'nadir':
        index_si = 2
        index_in = 3


    Nfile = len(fnames)

    cos_resp_si = {}
    cos_resp_in = {}

    for j in range(len(intTimes_si)):

        intTime_si = intTimes_si[j]
        intTime_in = intTimes_in[j]
        counts_si = np.zeros((256, Nfile), dtype=np.float64)
        counts_in = np.zeros((256, Nfile), dtype=np.float64)

        for i, fname in enumerate(fnames):

            ssfr            = CU_SSFR([fname], dark_corr_mode='dark_interpolate')
            logic_si = (np.abs(ssfr.int_time[:, index_si]-intTime_si)<0.00001)&(ssfr.shutter==0)
            logic_in = (np.abs(ssfr.int_time[:, index_in]-intTime_in)<0.00001)&(ssfr.shutter==0)
            logic_si[:10]  = False; logic_si[-10:] = False
            logic_in[:10]  = False; logic_in[-10:] = False
            counts_si[:, i] = np.mean(ssfr.spectra_dark_corr[logic_si, :, index_si], axis=0)
            counts_in[:, i] = np.mean(ssfr.spectra_dark_corr[logic_in, :, index_in], axis=0)


        cos_resp_si[intTime_si] =  counts_si/(np.repeat(counts_si[:, 0], Nfile).reshape((-1, Nfile)))
        cos_resp_in[intTime_in] =  counts_in/(np.repeat(counts_in[:, 0], Nfile).reshape((-1, Nfile)))

    return cos_resp_si, cos_resp_in



def PLOT_COS_RESP(fnames, ang, which='nadir', cable='L2008-2', intTimes_si=[45.0, 90.0], intTimes_in=[250.0, 375.0]):

    cos_resp_si, cos_resp_in = CAL_COS_RESP(fnames, which=which, intTimes_si=intTimes_si, intTimes_in=intTimes_in)

    cos = np.cos(np.deg2rad(ang))
    logic_pos = (ang>=0)
    logic_neg = (ang<0)

    refChan = 100

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)

    for i, intTime in enumerate(list(cos_resp_si.keys())):
        if i == 0:
            ax1.plot(cos[logic_pos], cos_resp_si[intTime][refChan, logic_pos], marker='o', color='r', markersize=3, markeredgecolor='none', alpha=0.7, lw=0.0, label='Positive Angle (S%d)' % intTime)
            ax1.plot(cos[logic_neg], cos_resp_si[intTime][refChan, logic_neg], marker='o', color='b', markersize=3, markeredgecolor='none', alpha=0.7, lw=0.0, label='Negative Angle (S%d)' % intTime)
        elif i == 1:
            ax1.plot(cos[logic_pos], cos_resp_si[intTime][refChan, logic_pos], marker='o', markeredgewidth=1.5, markeredgecolor='r', markersize=8, markerfacecolor='none', alpha=0.8, lw=0.8, label='Positive Angle (S%d)' % intTime)
            ax1.plot(cos[logic_neg], cos_resp_si[intTime][refChan, logic_neg], marker='o', markeredgewidth=1.5, markeredgecolor='b', markersize=8, markerfacecolor='none', alpha=0.8, lw=0.8, label='Negative Angle (S%d)' % intTime)

    ax1.plot([0, 1], [0, 1], ls='--', lw=2.0)
    ax1.set_xlim((0.0, 1.0))
    ax1.set_ylim((0.0, 1.2))
    ax1.set_title('%s Silicon (%s Channel %d)' % (which, cable, refChan+1))
    ax1.set_xlabel('$\cos(ang)$')
    ax1.set_ylabel('Response')
    ax1.legend(framealpha=0.3, fontsize=12, loc='upper left')
    plt.savefig('cos_resp_%s_si_%3.3d.png' % (which, refChan+1))
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)

    for i, intTime in enumerate(list(cos_resp_in.keys())):
        if i == 0:
            ax1.plot(cos[logic_pos], cos_resp_in[intTime][refChan, logic_pos], marker='o', color='r', markersize=3, markeredgecolor='none', alpha=0.7, lw=0.0, label='Positive Angle (I%d)' % intTime)
            ax1.plot(cos[logic_neg], cos_resp_in[intTime][refChan, logic_neg], marker='o', color='b', markersize=3, markeredgecolor='none', alpha=0.7, lw=0.0, label='Negative Angle (I%d)' % intTime)
        elif i == 1:
            ax1.plot(cos[logic_pos], cos_resp_in[intTime][refChan, logic_pos], marker='o', markeredgewidth=1.5, markeredgecolor='r', markersize=8, markerfacecolor='none', alpha=0.8, lw=0.8, label='Positive Angle (I%d)' % intTime)
            ax1.plot(cos[logic_neg], cos_resp_in[intTime][refChan, logic_neg], marker='o', markeredgewidth=1.5, markeredgecolor='b', markersize=8, markerfacecolor='none', alpha=0.8, lw=0.8, label='Negative Angle (I%d)' % intTime)

    ax1.plot([0, 1], [0, 1], ls='--', lw=2.0)
    ax1.set_xlim((0.0, 1.0))
    ax1.set_ylim((0.0, 1.2))
    ax1.set_title('%s InGaAs (%s Channel %d)' % (which, cable, refChan+1))
    ax1.set_xlabel('$\cos(ang)$')
    ax1.set_ylabel('Response')
    ax1.legend(framealpha=0.3, fontsize=12, loc='upper left')
    plt.savefig('cos_resp_%s_in_%3.3d.png' % (which, refChan+1))
    plt.close(fig)






if __name__ == '__main__':

    import matplotlib as mpl
    from matplotlib.ticker import FixedLocator, MaxNLocator
    import matplotlib.pyplot as plt
    from matplotlib import rcParams


    # ang = np.array([ 0.0,  3.0,  6.0,  9.0,  12.0,  15.0,  18.0,  21.0,  24.0,  27.0,  30.0,  35.0,  40.0,  45.0,  50.0,  60.0,  70.0,  80.0,  90.0, \
    #                  0.0, -3.0, -6.0, -9.0, -12.0, -15.0, -18.0, -21.0, -24.0, -27.0, -30.0, -35.0, -40.0, -45.0, -50.0, -60.0, -70.0, -80.0, -90.0, 0.0])

    ang = np.array([ 0.0,  5.0,  10.0,  15.0,  20.0,  25.0,  30.0,  40.0,  50.0,  60.0,  70.0,  80.0,  90.0, \
                     0.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0, -90.0, 0.0])

    # fnames = sorted(glob.glob('/Users/hoch4240/Chen/mygit/SSFR-util/CU-SSFR/Alvin/data/post_cals/20180712/Alvin/508/nadir/LCN-A-01/L2008-2/s45_90i250_375/pos/*.SKS')) + sorted(glob.glob('/Users/hoch4240/Chen/mygit/SSFR-util/CU-SSFR/Alvin/data/post_cals/20180712/Alvin/508/nadir/LCN-A-01/L2008-2/s45_90i250_375/neg/*.SKS'))
    # PLOT_COS_RESP(fnames, ang, which='Nadir', intTimes_si=[45.0, 90.0], intTimes_in=[250.0, 375.0], cable='L2008-2')

    # fnames = sorted(glob.glob('/Users/hoch4240/Chen/mygit/SSFR-util/CU-SSFR/Alvin/data/post_cals/20180712/Alvin/508/zenith/LCN-A-02/SSIM1/s45_90i250_375/pos/*.SKS')) + sorted(glob.glob('/Users/hoch4240/Chen/mygit/SSFR-util/CU-SSFR/Alvin/data/post_cals/20180712/Alvin/508/zenith/LCN-A-02/SSIM1/s45_90i250_375/neg/*.SKS'))
    # PLOT_COS_RESP(fnames, ang, which='Zenith', intTimes_si=[45.0, 90.0], intTimes_in=[250.0, 375.0], cable='SSIM1')

    fnames = sorted(glob.glob('/Users/hoch4240/Chen/work/00_reuse/SSFR-util/CU-SSFR/Belana/data/post_cals/20180723/Belana/508/nadir/LCN-B/111080/s45_90i250_375/pos/*.SKS')) + sorted(glob.glob('/Users/hoch4240/Chen/work/00_reuse/SSFR-util/CU-SSFR/Belana/data/post_cals/20180723/Belana/508/nadir/LCN-B/111080/s45_90i250_375/neg/*.SKS'))
    PLOT_COS_RESP(fnames, ang, which='Nadir', intTimes_si=[45.0, 90.0], intTimes_in=[250.0, 375.0], cable='111080')

    # fnames = sorted(glob.glob('/Users/hoch4240/Chen/mygit/SSFR-util/CU-SSFR/Alvin/data/post_cals/20180712/Alvin/508/zenith/LCN-A-02/SSIM1/s45_90i250_375/pos/*.SKS')) + sorted(glob.glob('/Users/hoch4240/Chen/mygit/SSFR-util/CU-SSFR/Alvin/data/post_cals/20180712/Alvin/508/zenith/LCN-A-02/SSIM1/s45_90i250_375/neg/*.SKS'))
    # PLOT_COS_RESP(fnames, ang, which='Zenith', intTimes_si=[45.0, 90.0], intTimes_in=[250.0, 375.0], cable='SSIM1')

    exit()