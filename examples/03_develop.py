import os
import sys
import glob
import datetime
import multiprocessing as mp
from collections import OrderedDict
import h5py
from pyhdf.SD import SD, SDC
import numpy as np
from scipy import interpolate
from scipy.io import readsav
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from ssfr import nasa_ssfr, load_h5, read_ict, read_iwg, cal_solar_angles, quicklook_bokeh, cal_solar_flux
from ssfr.cal import cdata_cos_resp
from ssfr.cal import cdata_rad_resp
from ssfr.corr import cos_corr

def time_series():

    fdir = '/Users/hoch4240/Desktop/SSFR/20191210/LED_test/LED_warm_up'
    fnames_ssfr = sorted(glob.glob('%s/*.OSA2' % fdir))

    date   = datetime.datetime(2019, 12, 10)
    date_s = date.strftime('%Y%m%d')

    ssfr0 = nasa_ssfr(fnames_ssfr, date_ref=datetime.datetime(2019, 12, 10))
    ssfr0.pre_process(wvl_join=950.0, wvl_start=350.0, wvl_end=2200.0, intTime={'si':60, 'in':300})

    f = h5py.File('SSFR_%s_V0.h5' % date_s, 'w')
    f['tmhr']    = ssfr0.tmhr
    f['jday']    = ssfr0.jday
    f['shutter'] = ssfr0.shutter
    f['zen_wvl'] = ssfr0.zen_wvl
    f['nad_wvl'] = ssfr0.nad_wvl
    f['zen_cnt'] = ssfr0.zen_cnt
    f['nad_cnt'] = ssfr0.nad_cnt
    f.close()
    # ============================================================================


def plot_time_series():

    data = load_h5('SSFR_20191210_V0.h5')

    # wvls = [431.787, 446.854, 457.991, 479.609, 577.872, 688.583]
    wvls = [446.854, 479.609, 577.872, 688.583]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(211)
    colors = plt.cm.jet(np.linspace(0.0, 1.0, data['tmhr'].size))
    for i in range(data['tmhr'].size):
        ax1.plot(data['zen_wvl'], data['zen_cnt'][i, :], lw=0.1, color=colors[i, ...])

    ax1.set_ylim([0, 2500])
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Counts')

    ax2 = fig.add_subplot(212)
    for wvl0 in wvls:
        index = np.argmin(np.abs(wvl0-data['zen_wvl']))
        ax1.axvline(data['zen_wvl'][index], ls='-', lw=1.5)
        ax2.plot(data['tmhr'], data['zen_cnt'][:, index], lw=0.8, label='%.2f nm' % data['zen_wvl'][index])
    ax2.set_ylim([0, 2500])
    ax2.set_xlabel('Time [Hour]')
    ax2.set_ylabel('Counts')
    ax2.grid(color='gray')
    ax2.legend(loc='upper right', fontsize=12)


    # ax1.set_title('')
    # ax1.legend(loc='upper right', fontsize=12, framealpha=0.4)
    plt.savefig('time_series.png', bbox_inches='tight')
    plt.show()
    exit()
    # ---------------------------------------------------------------------


def cdata_cos_resp_led_bulb(
        channel,
        angles = np.array([ 0.0,  5.0,  10.0,  15.0,  20.0,  25.0,  30.0,  40.0,  50.0,  60.0,  70.0,  80.0,  90.0, 0.0]),
        plot=True
        ):

    fdir = '/Users/hoch4240/Desktop/SSFR/20191210/LED_test/zenith/LC1/cosine/pos'
    fnames = sorted(glob.glob('%s/*.OSA2' % (fdir)))

    fnames_cal = OrderedDict()
    for i, fname in enumerate(fnames):
        fnames_cal[fname] = angles[i]

    filename_tag = 'LC1_LED'

    cos_mu, cos_resp = cdata_cos_resp(fnames_cal, filename_tag=filename_tag, which='zenith', Nchan=256, wvl_join=950.0, wvl_start=350.0, wvl_end=2200.0, intTime={'si':60, 'in':300})

    fdir0 = '/Users/hoch4240/Google Drive/CU LASP/CAMP2Ex/Calibrations/post-calibration/ang-cal/20191202/cosine/F508/zenith/LC1/s60i300'
    fnames0 = sorted(glob.glob('%s/pos/*.OSA2' % fdir0)) + [sorted(glob.glob('%s/neg/*.OSA2' % fdir0))[0]]
    fnames_cal0 = OrderedDict()
    for i, fname in enumerate(fnames0):
        fnames_cal0[fname] = angles[i]

    filename_tag0 = 'F508'
    cos_mu0, cos_resp0 = cdata_cos_resp(fnames_cal0, filename_tag=filename_tag0, which='zenith', Nchan=256, wvl_join=950.0, wvl_start=350.0, wvl_end=2200.0, intTime={'si':60, 'in':300})

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(cos_mu, cos_resp['si'][:, channel], marker='o', color='k', markersize=4)
    ax1.plot(cos_mu0, cos_resp0['si'][:, channel], marker='o', color='r', markersize=4)
    ax1.set_xlim((0.0, 1.0))
    ax1.set_ylim((0.0, 1.2))
    ax1.set_xlabel('cos($\\theta$)')
    ax1.set_ylabel('Cosine Response')
    ax1.set_title('Silicon Channel %d' % channel)

    patches_legend = [
                mpatches.Patch(color='black' , label='LED Bulb'),
                mpatches.Patch(color='red'   , label='F508'),
                ]
    # ax1.legend(handles=patches_legend, bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=len(patches_legend), mode="expand", borderaxespad=0., frameon=False, handletextpad=0.2, fontsize=16)
    ax1.legend(handles=patches_legend, loc='upper left', fontsize=14)

    plt.savefig('cos_resp_%3.3d.png' % channel, bbox_inches='tight')
    plt.close(fig)
    # ---------------------------------------------------------------------



if __name__ == '__main__':

    # time_series()
    # plot_time_series()
    for channel in [20, 60, 100, 150, 240]:
        cdata_cos_resp_led_bulb(channel)
