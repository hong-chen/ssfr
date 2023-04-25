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



__all__ = ['cdata_nlin_coef']



def cdata_nlin_coef(dist, fnames,
        wvl_ref=480, tag='nadir',
        fname_std='/Users/hoch4240/Chen/other/data/aux_ssfr/f-1324.dat'):

    NChan = 256
    NFile = dist.size

    xxChan = np.arange(NChan)

    if tag == 'zenith':
        coef_si = np.array([303.087, 3.30588, 4.09568e-04, -1.63269e-06, 0])
        coef_in = np.array([2213.37, -4.46844, -0.00111879, -2.76593e-06, -1.57883e-08])
        iSi     = 0
        iIn     = 1
    elif tag == 'nadir':
        coef_si = np.array([302.255, 3.30977, 4.38733e-04, -1.90935e-06, 0])
        coef_in = np.array([2225.74, -4.37926, -0.00220588, 2.80201e-06, -2.2624e-08])
        iSi     = 2
        iIn     = 3

    wvl_si  = coef_si[0] + coef_si[1]*xxChan + coef_si[2]*xxChan**2 + coef_si[3]*xxChan**3 + coef_si[4]*xxChan**4
    wvl_in  = coef_in[0] + coef_in[1]*xxChan + coef_in[2]*xxChan**2 + coef_in[3]*xxChan**3 + coef_in[4]*xxChan**4

    if len(fnames) != NFile:
        exit('Error [CDATA_NLIN]: the number of input .sks files are inconsistent with NFile.')

    intTime_si  = np.zeros( NFile,            dtype=np.float64)
    intTime_in  = np.zeros( NFile,            dtype=np.float64)
    data2fit_si = np.zeros((NFile, NChan, 2), dtype=np.float64) # index 0: mean, index 1: standard deviation
    data2fit_in = np.zeros((NFile, NChan, 2), dtype=np.float64) # index 0: mean, index 1: standard deviation
    dark_offset_si = np.zeros((NFile, NChan), dtype=np.float64)
    dark_offset_in = np.zeros((NFile, NChan), dtype=np.float64)

    for i, fname in enumerate(fnames):
        data= READ_NASA_SSFR([fname])
        logic = (data.shutter==0)

        intTime_si[i]        = np.mean(data.int_time[logic, iSi])
        intTime_in[i]        = np.mean(data.int_time[logic, iIn])
        data2fit_si[i, :, 0] = np.mean(data.spectra_dark_corr[logic, :, iSi], axis=0)
        data2fit_si[i, :, 1] = np.std( data.spectra_dark_corr[logic, :, iSi], axis=0)
        data2fit_in[i, :, 0] = np.mean(data.spectra_dark_corr[logic, :, iIn], axis=0)
        data2fit_in[i, :, 1] = np.std( data.spectra_dark_corr[logic, :, iIn], axis=0)

        dark_offset_si[i, :]    = np.mean(data.dark_offset[logic, :, iSi], axis=0)
        dark_offset_in[i, :]    = np.mean(data.dark_offset[logic, :, iIn], axis=0)

    # read standard (50cm) lamp file
    # data_std[:, 0]: wavelength in [nm]
    # data_std[:, 1]: irradiance in [???]
    data_std = np.loadtxt(fname_std)
    resp_si_interp = np.interp(wvl_si, data_std[:, 0], data_std[:, 1]*10000.0)
    resp_in_interp = np.interp(wvl_in, data_std[:, 0], data_std[:, 1]*10000.0)

    indexCal = 0
    resp_si = None
    resp_in = None
    if resp_si is None and resp_in is None:
        std_si  = data2fit_si[indexCal, :, 0]
        std_in  = data2fit_in[indexCal, :, 0]
        resp_si = std_si / intTime_si[indexCal] / resp_si_interp
        resp_in = std_in / intTime_in[indexCal] / resp_in_interp

    plt_logic = True
    if plt_logic:

        rcParams['font.size'] = 12
        rcParams['xtick.direction'] = 'in'
        rcParams['ytick.direction'] = 'in'
        fig = plt.figure(figsize=(16, 5))
        gs  = gridspec.GridSpec(1, 3)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[0, 2])
        ax1.plot(wvl_si, data2fit_si[indexCal, :, 0], color='k')
        ax1.axvline(wvl_ref, color='grey', ls='--')
        ax1.axhline(0, color='k', ls=':')
        ax1.set_title('Silicon')
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Dark Corrected Counts')
        ax1.set_xlim([200, 1200])
        ax1.set_ylim([0, 2000])
        ax1.ticklabel_format(style='sci',axis='y', scilimits=(-3,4))

        ax2.plot(wvl_in, data2fit_in[indexCal, :, 0], color='k')
        ax2.axhline(0, color='k', ls=':')
        ax2.set_title('InGaAs')
        ax2.set_xlabel('Wavelength [nm]')
        ax2.set_ylabel('Dark Corrected Counts')
        ax2.set_xlim([500, 2500])
        ax2.ticklabel_format(style='sci',axis='y', scilimits=(-3,4))
        ax2.set_ylim([0, 10000])

        ax3.plot(data_std[:, 0], data_std[:, 1]*0.01, label='Lamp', zorder=0)
        ax3.plot(wvl_si, data2fit_si[indexCal, :, 0]/intTime_si[indexCal]/resp_si, color='red', alpha=0.4, zorder=1)
        ax3.fill_between(wvl_si, (data2fit_si[indexCal, :, 0]-data2fit_si[indexCal, :, 1])/intTime_si[indexCal]/resp_si, (data2fit_si[indexCal, :, 0]+data2fit_si[indexCal, :, 1])/intTime_si[indexCal]/resp_si, facecolor='r', alpha=0.4, lw=0, label='Cal Si', zorder=1)

        ax3.plot(wvl_in, data2fit_in[indexCal, :, 0]/intTime_in[indexCal]/resp_in, color='blue', alpha=0.4, zorder=1)
        ax3.fill_between(wvl_in, (data2fit_in[indexCal, :, 0]-data2fit_in[indexCal, :, 1])/intTime_in[indexCal]/resp_in, (data2fit_in[indexCal, :, 0]+data2fit_in[indexCal, :, 1])/intTime_in[indexCal]/resp_in, facecolor='b', alpha=0.4, lw=0, label='Cal In', zorder=1)
        ax3.axhline(0, color='k', ls=':')
        ax3.set_title('Lamp Spectrum')
        ax3.set_xlabel('Wavelength [nm]')
        ax3.set_ylabel('Irradiance [$\mathrm{W m^2 nm^{-1}}$]')
        ax3.set_xlim([350, 2200])
        ax3.set_ylim([0, 0.3])
        ax3.legend(loc='upper right', fontsize=11, framealpha=0.2)
        plt.savefig('fig0.png')


    plt_logic = True
    if plt_logic:
        for indexCal in range(NFile):

            rcParams['font.size'] = 12
            rcParams['xtick.direction'] = 'in'
            rcParams['ytick.direction'] = 'in'
            fig = plt.figure(figsize=(16, 5))
            gs  = gridspec.GridSpec(1, 3)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])
            ax3 = plt.subplot(gs[0, 2])

            ax1.plot(wvl_si, data2fit_si[indexCal, :, 0]/intTime_si[indexCal]/resp_si, color='red', alpha=0.6, zorder=1)
            ax1.fill_between(wvl_si, (data2fit_si[indexCal, :, 0]-data2fit_si[indexCal, :, 1])/intTime_si[indexCal]/resp_si, (data2fit_si[indexCal, :, 0]+data2fit_si[indexCal, :, 1])/intTime_si[indexCal]/resp_si, facecolor='r', alpha=0.4, lw=0, label='Cal Si', zorder=1)
            ax1.plot(wvl_in, data2fit_in[indexCal, :, 0]/intTime_in[indexCal]/resp_in, color='blue', alpha=0.6, zorder=1)
            ax1.fill_between(wvl_in, (data2fit_in[indexCal, :, 0]-data2fit_in[indexCal, :, 1])/intTime_in[indexCal]/resp_in, (data2fit_in[indexCal, :, 0]+data2fit_in[indexCal, :, 1])/intTime_in[indexCal]/resp_in, facecolor='b', alpha=0.4, lw=0, label='Cal In', zorder=1)
            ax1.axvline(wvl_ref, color='grey', ls='--')
            ax1.set_title('Spectrum (Distance=%.1fcm)' % dist[indexCal])
            ax1.set_xlabel('Wavelength [nm]')
            ax1.set_ylabel('Irradiance [$\mathrm{W m^2 nm^{-1}}$]')
            ax1.set_xlim([350, 2200])
            ax1.set_ylim([0, 0.5])

            ax2.plot(wvl_in, data2fit_in[indexCal, :, 0], color='b', alpha=0.6)
            ax2.plot(wvl_in, dark_offset_in[indexCal, :], color='k', alpha=0.6)
            ax2.set_title('InGaAs')
            ax2.set_xlabel('Wavelength [nm]')
            ax2.set_ylabel('Dark Corrected Counts')
            ax2.set_xlim([500, 2500])
            ax2.set_ylim([0, 20000])
            ax2.ticklabel_format(style='sci',axis='y', scilimits=(-3,4))

            ax3.plot(wvl_si, data2fit_si[indexCal, :, 0]/intTime_si[indexCal]/resp_si/resp_si_interp, color='r', alpha=0.6, zorder=1)
            ax3.fill_between(wvl_si, (data2fit_si[indexCal, :, 0]-data2fit_si[indexCal, :, 1])/intTime_si[indexCal]/resp_si/resp_si_interp, (data2fit_si[indexCal, :, 0]+data2fit_si[indexCal, :, 1])/intTime_si[indexCal]/resp_si/resp_si_interp, facecolor='r', alpha=0.4, lw=0, label='Si', zorder=1)
            ax3.plot(wvl_in, data2fit_in[indexCal, :, 0]/intTime_in[indexCal]/resp_in/resp_in_interp, color='b', alpha=0.6, zorder=2)
            ax3.axvline(wvl_ref, color='grey', ls=':')
            ax3.axhline(1.0, color='k', zorder=0)
            ax3.axhline((50.0/dist[indexCal])**2, color='g', ls='--', alpha=0.6)
            ax3.axhline((50.0/dist[indexCal])**2 * 1.03, color='g', ls='--', alpha=0.6, zorder=0)
            ax3.axhline((50.0/dist[indexCal])**2 * 0.97, color='g', ls='--', alpha=0.6, zorder=0)
            #ax3.set_title('Lamp Spectrum')
            ax3.set_xlabel('Wavelength [nm]')
            ax3.set_ylabel('Ratio [Measurement/Lamp]')
            ax3.set_xlim([350, 1600])
            ylim_mid = np.round((50.0/dist[indexCal])**2, decimals=1)
            ax3.set_ylim([ylim_mid-0.4, ylim_mid+0.4])
            plt.savefig('fig1_dist%2.2d.png' % indexCal)
            plt.close(fig)

    poly_degree = 3
    coef_linear = np.zeros((NChan, 2), dtype=np.float64)
    coef_poly   = np.zeros((NChan, poly_degree+1), dtype=np.float64)
    nlin_deg    = np.zeros((NChan, 2), dtype=np.float64)
    x_minmax    = np.zeros((NChan, 2), dtype=np.float64)
    index_wvl_ref = np.argmin(np.abs(wvl_si-wvl_ref))
    for iChan in range(NChan):
        interp_x0 = data2fit_in[:, iChan, 0] / std_in[iChan]
        interp_y0 = data2fit_si[:, index_wvl_ref, 0] / std_si[index_wvl_ref]
        index_sort = np.argsort(interp_x0)
        interp_x = interp_x0[index_sort]
        interp_y = interp_y0[index_sort]

        coef_linear[iChan, 1], coef_linear[iChan, 0], r_value, p_value, std_err = stats.linregress(interp_x, interp_y)
        coef_poly[iChan, ::-1] = np.polyfit(interp_x, interp_y, poly_degree)

        x_minmax[iChan, 0] = np.min(interp_x)
        x_minmax[iChan, 1] = np.max(interp_x)

        lin_y  = coef_linear[iChan, 0] + coef_linear[iChan, 1]*interp_x
        nlin_y = np.zeros_like(lin_y)
        for exp in range(poly_degree+1):
            nlin_y += coef_poly[iChan, exp]*interp_x**exp

        nlin_deg[iChan, 0] = np.max(np.abs(( lin_y/(data2fit_si[:, index_wvl_ref, 0]/std_si[index_wvl_ref])-1.0)*100.0))
        nlin_deg[iChan, 1] = np.max(np.abs((nlin_y/(data2fit_si[:, index_wvl_ref, 0]/std_si[index_wvl_ref])-1.0)*100.0))

    plt_logic = True
    if plt_logic:
        for indexCal in range(NChan):

            rcParams['font.size'] = 12
            rcParams['xtick.direction'] = 'in'
            rcParams['ytick.direction'] = 'in'
            fig = plt.figure(figsize=(11, 5))
            gs  = gridspec.GridSpec(1, 2)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])

            ax1.scatter(data2fit_si[:, index_wvl_ref, 0]/std_si[index_wvl_ref], data2fit_in[:, indexCal, 0]/std_in[indexCal], edgecolor='k', facecolor='none', alpha=0.6, s=60, zorder=1, marker='s')

            xerr = (data2fit_si[:, index_wvl_ref, 1])/std_si[index_wvl_ref]
            yerr = (data2fit_in[:, indexCal     , 1])/std_in[indexCal     ]
            ax1.errorbar(data2fit_si[:, index_wvl_ref, 0]/std_si[index_wvl_ref], data2fit_in[:, indexCal, 0]/std_in[indexCal], xerr=xerr, yerr=yerr, fmt='none')

            ax1.plot([1.0, 1.0], [0.8, 1.2], color='g')
            ax1.plot([0.8, 1.2], [1.0, 1.0], color='g')
            ax1.plot([-10, 10], [-10, 10], color='k', ls=':')

            xx = np.linspace(-10, 10, 1000)
            lin_y    = coef_linear[indexCal, 0] + coef_linear[indexCal, 1]*xx
            nlin_y = np.zeros_like(lin_y)
            for exp in range(poly_degree+1):
                nlin_y += coef_poly[indexCal, exp]*xx**exp
            ax1.plot( lin_y, xx, color='r', alpha=0.6)
            ax1.plot(nlin_y, xx, color='b', alpha=0.6)

            ax1.set_title('Channel %d' % (indexCal+1))
            ax1.set_xlabel('Silicon/Silicon_Cal_50cm')
            ax1.set_ylabel('InGaAs/InGaAs_Cal_50cm')
            ax1.set_xlim([0.5, 2.5])
            ax1.set_ylim([0.5, 2.5])

            ax2.axvline(1.0, color='k', ls='--')
            ax2.axhline( 0.3, color='g', ls=':')
            ax2.axhline(-0.3, color='g', ls=':')
            xx = data2fit_in[:, indexCal, 0] / std_in[indexCal]
            lin_y    = coef_linear[indexCal, 0] + coef_linear[indexCal, 1]*xx
            nlin_y = np.zeros_like(lin_y)
            for exp in range(poly_degree+1):
                nlin_y += coef_poly[indexCal, exp]*xx**exp

            ax2.scatter(xx, (    xx/(data2fit_si[:, index_wvl_ref, 0]/std_si[index_wvl_ref])-1.0)*100.0, c='k', s=50, alpha=0.4, label='residual from 1:1')
            ax2.scatter(xx, ( lin_y/(data2fit_si[:, index_wvl_ref, 0]/std_si[index_wvl_ref])-1.0)*100.0, c='r', s=50, alpha=0.9, label='residual from lin')
            ax2.scatter(xx, (nlin_y/(data2fit_si[:, index_wvl_ref, 0]/std_si[index_wvl_ref])-1.0)*100.0, c='b', s=50, alpha=0.9, label='residual from nlin')

            yy = data2fit_si[:, index_wvl_ref, 0] / std_si[index_wvl_ref]
            ax2.scatter(xx, yy, edgecolor='k', facecolor='none', alpha=0.6, s=60, zorder=1, marker='s')

            xerr = (data2fit_in[:, indexCal, 1]/std_in[indexCal])
            xx = (data2fit_in[:, indexCal, 0]/std_in[indexCal])
            yy = (nlin_y/(data2fit_si[:,index_wvl_ref,0]/std_si[index_wvl_ref])-1.0)*100.0
            ax2.errorbar(xx, yy, xerr=xerr, yerr=0, fmt='none')

            ax2.set_xlim([0.5, 2.5])
            ax2.legend(loc='lower right', fontsize=8, framealpha=0.3)
            ax2.set_xlabel('InGaAs/InGaAs_Cal_50cm')
            ax2.set_ylabel('Non-linearity Degree [%]')
            ax2.set_title('Wavelength %.2f nm' % wvl_in[indexCal])
            plt.savefig('fig2_chan%3.3d.png' % (indexCal+1))
            plt.close(fig)

    exit()


    plt_logic = False
    if plt_logic:
        rcParams['font.size'] = 12
        rcParams['xtick.direction'] = 'in'
        rcParams['ytick.direction'] = 'in'
        fig = plt.figure(figsize=(11, 5))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.plot(wvl_in, coef_poly[:, -1])
        ax1.axhline(0.0, color='red', lw=0.8, ls=':')
        ax1.set_ylabel('Quadratic Term')
        ax1.set_ylim([-0.1, 0.1])

        #ax2.plot(wvl_in, nlin_deg[:, 1])
        #ax2.set_ylim([162, 164])
        #offset = np.zeros((NFile, NChan), dtype=np.float64)
        #for i in range(NFile):
            #xx = dist[i]
            #for exp in range(poly_degree+1):
                #nlin_y += coef_poly[:, exp]*xx**exp

        #offset = np.array([])
        #for indexCal in range(NChan):
            #xx = data2fit_in[:, indexCal, 0] / std_in[indexCal]
            #nlin_y = np.zeros_like(lin_y)
            #for exp in range(poly_degree+1):
                #nlin_y += coef_poly[indexCal, exp]*xx**exp
            #offset = np.append(offset, nliny-)


        ax2.set_ylabel('Offset')
        ax2.set_xlabel('Wavelength [nm]')
        plt.savefig('quadratic.png')


def CDATA_NLIN(dist, fnames,
        wvl_ref=480, tag='nadir',
        fname_std='/Users/hoch4240/Chen/other/data/aux_ssfr/f-1324.dat'):

    NChan = 256
    NFile = dist.size

    xxChan = np.arange(NChan)

    if tag == 'zenith':
        coef_si = np.array([303.087, 3.30588, 4.09568e-04, -1.63269e-06, 0])
        coef_in = np.array([2213.37, -4.46844, -0.00111879, -2.76593e-06, -1.57883e-08])
        iSi     = 0
        iIn     = 1
    elif tag == 'nadir':
        coef_si = np.array([302.255, 3.30977, 4.38733e-04, -1.90935e-06, 0])
        coef_in = np.array([2225.74, -4.37926, -0.00220588, 2.80201e-06, -2.2624e-08])
        iSi     = 2
        iIn     = 3

    wvl_si  = coef_si[0] + coef_si[1]*xxChan + coef_si[2]*xxChan**2 + coef_si[3]*xxChan**3 + coef_si[4]*xxChan**4
    wvl_in  = coef_in[0] + coef_in[1]*xxChan + coef_in[2]*xxChan**2 + coef_in[3]*xxChan**3 + coef_in[4]*xxChan**4

    if len(fnames) != NFile:
        exit('Error [CDATA_NLIN]: the number of input .sks files are inconsistent with NFile.')

    intTime_si  = np.zeros( NFile,            dtype=np.float64)
    intTime_in  = np.zeros( NFile,            dtype=np.float64)
    data2fit_si = np.zeros((NFile, NChan, 2), dtype=np.float64) # index 0: mean, index 1: standard deviation
    data2fit_in = np.zeros((NFile, NChan, 2), dtype=np.float64) # index 0: mean, index 1: standard deviation
    dark_offset_si = np.zeros((NFile, NChan), dtype=np.float64)
    dark_offset_in = np.zeros((NFile, NChan), dtype=np.float64)

    for i, fname in enumerate(fnames):
        data= READ_NASA_SSFR([fname])
        logic = (data.shutter==0)

        intTime_si[i]        = np.mean(data.int_time[logic, iSi])
        intTime_in[i]        = np.mean(data.int_time[logic, iIn])
        data2fit_si[i, :, 0] = np.mean(data.spectra_dark_corr[logic, :, iSi], axis=0)
        data2fit_si[i, :, 1] = np.std( data.spectra_dark_corr[logic, :, iSi], axis=0)
        data2fit_in[i, :, 0] = np.mean(data.spectra_dark_corr[logic, :, iIn], axis=0)
        data2fit_in[i, :, 1] = np.std( data.spectra_dark_corr[logic, :, iIn], axis=0)

        dark_offset_si[i, :]    = np.mean(data.dark_offset[logic, :, iSi], axis=0)
        dark_offset_in[i, :]    = np.mean(data.dark_offset[logic, :, iIn], axis=0)

    # read standard (50cm) lamp file
    # data_std[:, 0]: wavelength in [nm]
    # data_std[:, 1]: irradiance in [???]
    data_std = np.loadtxt(fname_std)
    resp_si_interp = np.interp(wvl_si, data_std[:, 0], data_std[:, 1]*10000.0)
    resp_in_interp = np.interp(wvl_in, data_std[:, 0], data_std[:, 1]*10000.0)

    indexCal = 0
    resp_si = None
    resp_in = None
    if resp_si is None and resp_in is None:
        std_si  = data2fit_si[indexCal, :, 0]
        std_in  = data2fit_in[indexCal, :, 0]
        resp_si = std_si / intTime_si[indexCal] / resp_si_interp
        resp_in = std_in / intTime_in[indexCal] / resp_in_interp

    plt_logic = True
    if plt_logic:

        rcParams['font.size'] = 12
        rcParams['xtick.direction'] = 'in'
        rcParams['ytick.direction'] = 'in'
        fig = plt.figure(figsize=(16, 5))
        gs  = gridspec.GridSpec(1, 3)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[0, 2])
        ax1.plot(wvl_si, data2fit_si[indexCal, :, 0], color='k')
        ax1.axvline(wvl_ref, color='grey', ls='--')
        ax1.axhline(0, color='k', ls=':')
        ax1.set_title('Silicon')
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Dark Corrected Counts')
        ax1.set_xlim([200, 1200])
        ax1.set_ylim([0, 2000])
        ax1.ticklabel_format(style='sci',axis='y', scilimits=(-3,4))

        ax2.plot(wvl_in, data2fit_in[indexCal, :, 0], color='k')
        ax2.axhline(0, color='k', ls=':')
        ax2.set_title('InGaAs')
        ax2.set_xlabel('Wavelength [nm]')
        ax2.set_ylabel('Dark Corrected Counts')
        ax2.set_xlim([500, 2500])
        ax2.ticklabel_format(style='sci',axis='y', scilimits=(-3,4))
        ax2.set_ylim([0, 10000])

        ax3.plot(data_std[:, 0], data_std[:, 1]*0.01, label='Lamp', zorder=0)
        ax3.plot(wvl_si, data2fit_si[indexCal, :, 0]/intTime_si[indexCal]/resp_si, color='red', alpha=0.4, zorder=1)
        ax3.fill_between(wvl_si, (data2fit_si[indexCal, :, 0]-data2fit_si[indexCal, :, 1])/intTime_si[indexCal]/resp_si, (data2fit_si[indexCal, :, 0]+data2fit_si[indexCal, :, 1])/intTime_si[indexCal]/resp_si, facecolor='r', alpha=0.4, lw=0, label='Cal Si', zorder=1)

        ax3.plot(wvl_in, data2fit_in[indexCal, :, 0]/intTime_in[indexCal]/resp_in, color='blue', alpha=0.4, zorder=1)
        ax3.fill_between(wvl_in, (data2fit_in[indexCal, :, 0]-data2fit_in[indexCal, :, 1])/intTime_in[indexCal]/resp_in, (data2fit_in[indexCal, :, 0]+data2fit_in[indexCal, :, 1])/intTime_in[indexCal]/resp_in, facecolor='b', alpha=0.4, lw=0, label='Cal In', zorder=1)
        ax3.axhline(0, color='k', ls=':')
        ax3.set_title('Lamp Spectrum')
        ax3.set_xlabel('Wavelength [nm]')
        ax3.set_ylabel('Irradiance [$\mathrm{W m^2 nm^{-1}}$]')
        ax3.set_xlim([350, 2200])
        ax3.set_ylim([0, 0.3])
        ax3.legend(loc='upper right', fontsize=11, framealpha=0.2)
        plt.savefig('fig0.png')


    plt_logic = True
    if plt_logic:
        for indexCal in range(NFile):

            rcParams['font.size'] = 12
            rcParams['xtick.direction'] = 'in'
            rcParams['ytick.direction'] = 'in'
            fig = plt.figure(figsize=(16, 5))
            gs  = gridspec.GridSpec(1, 3)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])
            ax3 = plt.subplot(gs[0, 2])

            ax1.plot(wvl_si, data2fit_si[indexCal, :, 0]/intTime_si[indexCal]/resp_si, color='red', alpha=0.6, zorder=1)
            ax1.fill_between(wvl_si, (data2fit_si[indexCal, :, 0]-data2fit_si[indexCal, :, 1])/intTime_si[indexCal]/resp_si, (data2fit_si[indexCal, :, 0]+data2fit_si[indexCal, :, 1])/intTime_si[indexCal]/resp_si, facecolor='r', alpha=0.4, lw=0, label='Cal Si', zorder=1)
            ax1.plot(wvl_in, data2fit_in[indexCal, :, 0]/intTime_in[indexCal]/resp_in, color='blue', alpha=0.6, zorder=1)
            ax1.fill_between(wvl_in, (data2fit_in[indexCal, :, 0]-data2fit_in[indexCal, :, 1])/intTime_in[indexCal]/resp_in, (data2fit_in[indexCal, :, 0]+data2fit_in[indexCal, :, 1])/intTime_in[indexCal]/resp_in, facecolor='b', alpha=0.4, lw=0, label='Cal In', zorder=1)
            ax1.axvline(wvl_ref, color='grey', ls='--')
            ax1.set_title('Spectrum (Distance=%.1fcm)' % dist[indexCal])
            ax1.set_xlabel('Wavelength [nm]')
            ax1.set_ylabel('Irradiance [$\mathrm{W m^2 nm^{-1}}$]')
            ax1.set_xlim([350, 2200])
            ax1.set_ylim([0, 0.5])

            ax2.plot(wvl_in, data2fit_in[indexCal, :, 0], color='b', alpha=0.6)
            ax2.plot(wvl_in, dark_offset_in[indexCal, :], color='k', alpha=0.6)
            ax2.set_title('InGaAs')
            ax2.set_xlabel('Wavelength [nm]')
            ax2.set_ylabel('Dark Corrected Counts')
            ax2.set_xlim([500, 2500])
            ax2.set_ylim([0, 20000])
            ax2.ticklabel_format(style='sci',axis='y', scilimits=(-3,4))

            ax3.plot(wvl_si, data2fit_si[indexCal, :, 0]/intTime_si[indexCal]/resp_si/resp_si_interp, color='r', alpha=0.6, zorder=1)
            ax3.fill_between(wvl_si, (data2fit_si[indexCal, :, 0]-data2fit_si[indexCal, :, 1])/intTime_si[indexCal]/resp_si/resp_si_interp, (data2fit_si[indexCal, :, 0]+data2fit_si[indexCal, :, 1])/intTime_si[indexCal]/resp_si/resp_si_interp, facecolor='r', alpha=0.4, lw=0, label='Si', zorder=1)
            ax3.plot(wvl_in, data2fit_in[indexCal, :, 0]/intTime_in[indexCal]/resp_in/resp_in_interp, color='b', alpha=0.6, zorder=2)
            ax3.axvline(wvl_ref, color='grey', ls=':')
            ax3.axhline(1.0, color='k', zorder=0)
            ax3.axhline((50.0/dist[indexCal])**2, color='g', ls='--', alpha=0.6)
            ax3.axhline((50.0/dist[indexCal])**2 * 1.03, color='g', ls='--', alpha=0.6, zorder=0)
            ax3.axhline((50.0/dist[indexCal])**2 * 0.97, color='g', ls='--', alpha=0.6, zorder=0)
            #ax3.set_title('Lamp Spectrum')
            ax3.set_xlabel('Wavelength [nm]')
            ax3.set_ylabel('Ratio [Measurement/Lamp]')
            ax3.set_xlim([350, 1600])
            ylim_mid = np.round((50.0/dist[indexCal])**2, decimals=1)
            ax3.set_ylim([ylim_mid-0.4, ylim_mid+0.4])
            plt.savefig('fig1_dist%2.2d.png' % indexCal)
            plt.close(fig)

    poly_degree = 3
    coef_linear = np.zeros((NChan, 2), dtype=np.float64)
    coef_poly   = np.zeros((NChan, poly_degree+1), dtype=np.float64)
    nlin_deg    = np.zeros((NChan, 2), dtype=np.float64)
    x_minmax    = np.zeros((NChan, 2), dtype=np.float64)
    index_wvl_ref = np.argmin(np.abs(wvl_si-wvl_ref))
    for iChan in range(NChan):
        interp_x0 = data2fit_in[:, iChan, 0] / std_in[iChan]
        interp_y0 = data2fit_si[:, index_wvl_ref, 0] / std_si[index_wvl_ref]
        index_sort = np.argsort(interp_x0)
        interp_x = interp_x0[index_sort]
        interp_y = interp_y0[index_sort]

        coef_linear[iChan, 1], coef_linear[iChan, 0], r_value, p_value, std_err = stats.linregress(interp_x, interp_y)
        coef_poly[iChan, ::-1] = np.polyfit(interp_x, interp_y, poly_degree)

        x_minmax[iChan, 0] = np.min(interp_x)
        x_minmax[iChan, 1] = np.max(interp_x)

        lin_y  = coef_linear[iChan, 0] + coef_linear[iChan, 1]*interp_x
        nlin_y = np.zeros_like(lin_y)
        for exp in range(poly_degree+1):
            nlin_y += coef_poly[iChan, exp]*interp_x**exp

        nlin_deg[iChan, 0] = np.max(np.abs(( lin_y/(data2fit_si[:, index_wvl_ref, 0]/std_si[index_wvl_ref])-1.0)*100.0))
        nlin_deg[iChan, 1] = np.max(np.abs((nlin_y/(data2fit_si[:, index_wvl_ref, 0]/std_si[index_wvl_ref])-1.0)*100.0))

    plt_logic = True
    if plt_logic:
        for indexCal in range(NChan):

            rcParams['font.size'] = 12
            rcParams['xtick.direction'] = 'in'
            rcParams['ytick.direction'] = 'in'
            fig = plt.figure(figsize=(11, 5))
            gs  = gridspec.GridSpec(1, 2)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])

            ax1.scatter(data2fit_si[:, index_wvl_ref, 0]/std_si[index_wvl_ref], data2fit_in[:, indexCal, 0]/std_in[indexCal], edgecolor='k', facecolor='none', alpha=0.6, s=60, zorder=1, marker='s')

            xerr = (data2fit_si[:, index_wvl_ref, 1])/std_si[index_wvl_ref]
            yerr = (data2fit_in[:, indexCal     , 1])/std_in[indexCal     ]
            ax1.errorbar(data2fit_si[:, index_wvl_ref, 0]/std_si[index_wvl_ref], data2fit_in[:, indexCal, 0]/std_in[indexCal], xerr=xerr, yerr=yerr, fmt='none')

            ax1.plot([1.0, 1.0], [0.8, 1.2], color='g')
            ax1.plot([0.8, 1.2], [1.0, 1.0], color='g')
            ax1.plot([-10, 10], [-10, 10], color='k', ls=':')

            xx = np.linspace(-10, 10, 1000)
            lin_y    = coef_linear[indexCal, 0] + coef_linear[indexCal, 1]*xx
            nlin_y = np.zeros_like(lin_y)
            for exp in range(poly_degree+1):
                nlin_y += coef_poly[indexCal, exp]*xx**exp
            ax1.plot( lin_y, xx, color='r', alpha=0.6)
            ax1.plot(nlin_y, xx, color='b', alpha=0.6)

            ax1.set_title('Channel %d' % (indexCal+1))
            ax1.set_xlabel('Silicon/Silicon_Cal_50cm')
            ax1.set_ylabel('InGaAs/InGaAs_Cal_50cm')
            ax1.set_xlim([0.5, 2.5])
            ax1.set_ylim([0.5, 2.5])

            ax2.axvline(1.0, color='k', ls='--')
            ax2.axhline( 0.3, color='g', ls=':')
            ax2.axhline(-0.3, color='g', ls=':')
            xx = data2fit_in[:, indexCal, 0] / std_in[indexCal]
            lin_y    = coef_linear[indexCal, 0] + coef_linear[indexCal, 1]*xx
            nlin_y = np.zeros_like(lin_y)
            for exp in range(poly_degree+1):
                nlin_y += coef_poly[indexCal, exp]*xx**exp

            ax2.scatter(xx, (    xx/(data2fit_si[:, index_wvl_ref, 0]/std_si[index_wvl_ref])-1.0)*100.0, c='k', s=50, alpha=0.4, label='residual from 1:1')
            ax2.scatter(xx, ( lin_y/(data2fit_si[:, index_wvl_ref, 0]/std_si[index_wvl_ref])-1.0)*100.0, c='r', s=50, alpha=0.9, label='residual from lin')
            ax2.scatter(xx, (nlin_y/(data2fit_si[:, index_wvl_ref, 0]/std_si[index_wvl_ref])-1.0)*100.0, c='b', s=50, alpha=0.9, label='residual from nlin')

            yy = data2fit_si[:, index_wvl_ref, 0] / std_si[index_wvl_ref]
            ax2.scatter(xx, yy, edgecolor='k', facecolor='none', alpha=0.6, s=60, zorder=1, marker='s')

            xerr = (data2fit_in[:, indexCal, 1]/std_in[indexCal])
            xx = (data2fit_in[:, indexCal, 0]/std_in[indexCal])
            yy = (nlin_y/(data2fit_si[:,index_wvl_ref,0]/std_si[index_wvl_ref])-1.0)*100.0
            ax2.errorbar(xx, yy, xerr=xerr, yerr=0, fmt='none')

            ax2.set_xlim([0.5, 2.5])
            ax2.legend(loc='lower right', fontsize=8, framealpha=0.3)
            ax2.set_xlabel('InGaAs/InGaAs_Cal_50cm')
            ax2.set_ylabel('Non-linearity Degree [%]')
            ax2.set_title('Wavelength %.2f nm' % wvl_in[indexCal])
            plt.savefig('fig2_chan%3.3d.png' % (indexCal+1))
            plt.close(fig)

    exit()


    plt_logic = False
    if plt_logic:
        rcParams['font.size'] = 12
        rcParams['xtick.direction'] = 'in'
        rcParams['ytick.direction'] = 'in'
        fig = plt.figure(figsize=(11, 5))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.plot(wvl_in, coef_poly[:, -1])
        ax1.axhline(0.0, color='red', lw=0.8, ls=':')
        ax1.set_ylabel('Quadratic Term')
        ax1.set_ylim([-0.1, 0.1])

        #ax2.plot(wvl_in, nlin_deg[:, 1])
        #ax2.set_ylim([162, 164])
        #offset = np.zeros((NFile, NChan), dtype=np.float64)
        #for i in range(NFile):
            #xx = dist[i]
            #for exp in range(poly_degree+1):
                #nlin_y += coef_poly[:, exp]*xx**exp

        #offset = np.array([])
        #for indexCal in range(NChan):
            #xx = data2fit_in[:, indexCal, 0] / std_in[indexCal]
            #nlin_y = np.zeros_like(lin_y)
            #for exp in range(poly_degree+1):
                #nlin_y += coef_poly[indexCal, exp]*xx**exp
            #offset = np.append(offset, nliny-)


        ax2.set_ylabel('Offset')
        ax2.set_xlabel('Wavelength [nm]')
        plt.savefig('quadratic.png')


if __name__ == '__main__':

    cdata_nlin_coef()
