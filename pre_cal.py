import os
import sys
import glob
import datetime
import multiprocessing as mp
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav

class CALIBRATION_CU_SSFR:

    def __init__(self, dataSSFR):

        self.CAL_WAVELENGTH()
        self.CAL_PRIMARY_RESPONSE(dataSSFR)

    def CAL_WAVELENGTH(self):

        self.chanNum = 256
        xChan = np.arange(self.chanNum)

        self.coef_zen_si = np.array([301.946,  3.31877,  0.00037585,  -1.76779e-6, 0])
        self.coef_zen_in = np.array([2202.33, -4.35275, -0.00269498,   3.84968e-6, -2.33845e-8])

        self.coef_nad_si = np.array([302.818,  3.31912,  0.000343831, -1.81135e-6, 0])
        self.coef_nad_in = np.array([2210.29,  -4.5998,  0.00102444,  -1.60349e-5, 1.29122e-8])

        self.wvl_zen_si = self.coef_zen_si[0] + self.coef_zen_si[1]*xChan + self.coef_zen_si[2]*xChan**2 + self.coef_zen_si[3]*xChan**3 + self.coef_zen_si[4]*xChan**4
        self.wvl_zen_in = self.coef_zen_in[0] + self.coef_zen_in[1]*xChan + self.coef_zen_in[2]*xChan**2 + self.coef_zen_in[3]*xChan**3 + self.coef_zen_in[4]*xChan**4

        self.wvl_nad_si = self.coef_nad_si[0] + self.coef_nad_si[1]*xChan + self.coef_nad_si[2]*xChan**2 + self.coef_nad_si[3]*xChan**3 + self.coef_nad_si[4]*xChan**4
        self.wvl_nad_in = self.coef_nad_in[0] + self.coef_nad_in[1]*xChan + self.coef_nad_in[2]*xChan**2 + self.coef_nad_in[3]*xChan**3 + self.coef_nad_in[4]*xChan**4

    def CAL_PRIMARY_RESPONSE(self, dataSSFR, lampTag='f-1324', fdirLamp='/Users/hoch4240/Chen/other/data/aux_ssfr'):

        self.fnameLamp = '%s/%s.dat' % (fdirLamp, lampTag)

        if not os.path.exists(self.fnameLamp):
            exit('Error [CALIBRATION_CU_SSFR.CAL_PRIMARY_RESPONSE]: cannot locate lamp standards for %s.' % lampTag.title())

        data = np.loadtxt(self.fnameLamp)
        data_wvl  = data[:, 0]

        if lampTag == 'f-1324':
            data_flux = data[:, 1]*10000.0
        elif lampTag == '506c':
            data_flux = data[:, 1]*0.01

        lampStd_zen_si = np.interp(self.wvl_zen_si, data_wvl, data_flux)
        lampStd_zen_in = np.interp(self.wvl_zen_in, data_wvl, data_flux)
        lampStd_nad_si = np.interp(self.wvl_nad_si, data_wvl, data_flux)
        lampStd_nad_in = np.interp(self.wvl_nad_in, data_wvl, data_flux)


        counts_zen_si = dataSSFR.spectra_dark_corr[dataSSFR.shutter==0, :, 0]
        counts_zen_in = dataSSFR.spectra_dark_corr[dataSSFR.shutter==0, :, 1]
        counts_nad_si = dataSSFR.spectra_dark_corr[dataSSFR.shutter==0, :, 2]
        counts_nad_in = dataSSFR.spectra_dark_corr[dataSSFR.shutter==0, :, 3]

        count_zen_si = np.mean(counts_zen_si, axis=0); count_std_zen_si = np.std(counts_zen_si, axis=0)
        count_zen_in = np.mean(counts_zen_in, axis=0); count_std_zen_in = np.std(counts_zen_in, axis=0)
        count_nad_si = np.mean(counts_nad_si, axis=0); count_std_nad_si = np.std(counts_nad_si, axis=0)
        count_nad_in = np.mean(counts_nad_in, axis=0); count_std_nad_in = np.std(counts_nad_in, axis=0)

        logicTmp = np.repeat(False, dataSSFR.shutter.size)
        logicTmp[-100:] = True
        logicTmp = (logicTmp)&(dataSSFR.shutter==1)
        count_zen_si = np.mean(dataSSFR.spectra[(dataSSFR.shutter==0), :, 0], axis=0) - np.mean(dataSSFR.spectra[logicTmp, :, 0], axis=0); count_std_zen_si = np.std(counts_zen_si, axis=0)
        count_zen_in = np.mean(dataSSFR.spectra[(dataSSFR.shutter==0), :, 1], axis=0) - np.mean(dataSSFR.spectra[logicTmp, :, 1], axis=0); count_std_zen_in = np.std(counts_zen_in, axis=0)
        count_nad_si = np.mean(dataSSFR.spectra[(dataSSFR.shutter==0), :, 2], axis=0) - np.mean(dataSSFR.spectra[logicTmp, :, 2], axis=0); count_std_nad_si = np.std(counts_nad_si, axis=0)
        count_nad_in = np.mean(dataSSFR.spectra[(dataSSFR.shutter==0), :, 3], axis=0) - np.mean(dataSSFR.spectra[logicTmp, :, 3], axis=0); count_std_nad_in = np.std(counts_nad_in, axis=0)

        intTime_zen_si = dataSSFR.int_time[dataSSFR.shutter==0, 0].mean()
        intTime_zen_in = dataSSFR.int_time[dataSSFR.shutter==0, 1].mean()
        intTime_nad_si = dataSSFR.int_time[dataSSFR.shutter==0, 2].mean()
        intTime_nad_in = dataSSFR.int_time[dataSSFR.shutter==0, 3].mean()

        self.resp_zen_si = count_zen_si / intTime_zen_si / lampStd_zen_si
        self.resp_zen_in = count_zen_in / intTime_zen_in / lampStd_zen_in
        self.resp_nad_si = count_nad_si / intTime_nad_si / lampStd_nad_si
        self.resp_nad_in = count_nad_in / intTime_nad_in / lampStd_nad_in

        self.resp_std_zen_si = count_std_zen_si / intTime_zen_si / lampStd_zen_si
        self.resp_std_zen_in = count_std_zen_in / intTime_zen_in / lampStd_zen_in
        self.resp_std_nad_si = count_std_nad_si / intTime_nad_si / lampStd_nad_si
        self.resp_std_nad_in = count_std_nad_in / intTime_nad_in / lampStd_nad_in

if __name__ == '__main__':

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator
    from matplotlib import rcParams
    import cartopy.crs as ccrs

    from vid_ssfr import READ_SKS

    fnames = sorted(glob.glob('data/20180112/pcal_1324/*.SKS'))
    dataSSFR = READ_SKS(fnames)

    cal = CALIBRATION_CU_SSFR(dataSSFR)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    rcParams['font.size'] = 14
    fig = plt.figure(figsize=(12, 6))

    ax_zen_si = fig.add_subplot(221)
    ax_zen_in = fig.add_subplot(222)
    ax_nad_si = fig.add_subplot(221)
    ax_nad_in = fig.add_subplot(222)

    ax_zen_si.fill_between(cal.wvl_zen_si, cal.resp_zen_si-cal.resp_std_zen_si, cal.resp_zen_si+cal.resp_std_zen_si, color='r', alpha=0.3, lw=0.0, zorder=0)
    ax_zen_si.plot(cal.wvl_zen_si, cal.resp_zen_si, lw=1.0, color='r')
    ax_zen_si.axhline(0.0, color='k', ls='-')
    ax_zen_si.grid(color='gray', ls='--')
    ax_zen_si.set_title('Zenith Silicon Response Function')
    ax_zen_si.set_xlabel('Wavelength [nm]')
    ax_zen_si.set_ylabel('Response [$\mathrm{\\frac{Counts \cdot ms^{-1}}{W m^{-2} nm^{-1}}}$]')
    ax_zen_si.set_xlim((300, 1200))
    ax_zen_si.set_ylim((-200, 400))

    ax_zen_in.fill_between(cal.wvl_zen_in, cal.resp_zen_in-cal.resp_std_zen_in, cal.resp_zen_in+cal.resp_std_zen_in, color='r', alpha=0.3, lw=0.0, zorder=0)
    ax_zen_in.plot(cal.wvl_zen_in, cal.resp_zen_in, lw=1.0, color='r')
    ax_zen_in.axhline(0.0, color='k', ls='-')
    ax_zen_in.grid(color='gray', ls='--')
    ax_zen_in.set_title('Zenith InGaAs Response Function')
    ax_zen_in.set_xlabel('Wavelength [nm]')
    ax_zen_in.set_ylabel('Response [$\mathrm{\\frac{Counts \cdot ms^{-1}}{W m^{-2} nm^{-1}}}$]')
    ax_zen_in.set_xlim((800, 2300))
    ax_zen_in.set_ylim((-100, 600))

    plt.savefig('20180112_resp.png')
    plt.show()
    exit()
    # ---------------------------------------------------------------------
