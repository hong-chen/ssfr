import os
import sys
import glob
import datetime
import multiprocessing as mp
import struct
from scipy import stats
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav
from pre_ssfr import READ_NASA_SSFR_V1, READ_CU_SSFR_V2


class CALIBRATION_NASA_SSFR:

    def __init__(self, fdir_primary, fdir_transfer, fdir_secondary):

        self.CAL_WAVELENGTH()
        self.CAL_PRIMARY_RESPONSE(fdir_primary)
        self.CAL_TRANSFER(fdir_transfer)
        self.CAL_SECONDARY_RESPONSE(fdir_secondary)


    def CAL_WAVELENGTH(self):

        self.chanNum = 256
        xChan = np.arange(self.chanNum)

        self.coef_zen_si = np.array([303.087, 3.30588, 4.09568e-04, -1.63269e-06, 0])
        self.coef_zen_in = np.array([2213.37, -4.46844, -0.00111879, -2.76593e-06, -1.57883e-08])

        self.coef_nad_si = np.array([302.255, 3.30977, 4.38733e-04, -1.90935e-06, 0])
        self.coef_nad_in = np.array([2225.74, -4.37926, -0.00220588, 2.80201e-06, -2.2624e-08])

        self.wvl_zen_si = self.coef_zen_si[0] + self.coef_zen_si[1]*xChan + self.coef_zen_si[2]*xChan**2 + self.coef_zen_si[3]*xChan**3 + self.coef_zen_si[4]*xChan**4
        self.wvl_zen_in = self.coef_zen_in[0] + self.coef_zen_in[1]*xChan + self.coef_zen_in[2]*xChan**2 + self.coef_zen_in[3]*xChan**3 + self.coef_zen_in[4]*xChan**4

        self.wvl_nad_si = self.coef_nad_si[0] + self.coef_nad_si[1]*xChan + self.coef_nad_si[2]*xChan**2 + self.coef_nad_si[3]*xChan**3 + self.coef_nad_si[4]*xChan**4
        self.wvl_nad_in = self.coef_nad_in[0] + self.coef_nad_in[1]*xChan + self.coef_nad_in[2]*xChan**2 + self.coef_nad_in[3]*xChan**3 + self.coef_nad_in[4]*xChan**4


    def CAL_PRIMARY_RESPONSE(self, fdir_primary, int_time_si=60, int_time_in=300, lampTag='f-1324', fdirLamp='/Users/hoch4240/Chen/other/data/aux_ssfr'):

        # read in calibrated lamp data and interpolated at SSFR wavelengths
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
        # ---------------------------------------------------------------------------

        # so far we have (W m^-2 nm^-1 as a function of wavelength)

        # read in SSFR data collected during primary calibration
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # for zenith
        fname_pattern = '%s/zenith/s%di%d/cal/*.OSA2' % (fdir_primary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_PRIMARY_RESPONSE]: cannot find data for zenith cal.')
        fname = fnames[0]
        spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR_V1(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_zen_si_l = spectra_mean[:, 0]
        spectra_zen_in_l = spectra_mean[:, 1]

        fname_pattern = '%s/zenith/s%di%d/dark/*.OSA2' % (fdir_primary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_PRIMARY_RESPONSE]: cannot find data for zenith dark.')
        fname = fnames[0]
        spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR_V1(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_zen_si_d = spectra_mean[:, 0]
        spectra_zen_in_d = spectra_mean[:, 1]

        spectra_zen_si = spectra_zen_si_l - spectra_zen_si_d
        spectra_zen_si[spectra_zen_si<=0.0] = 0.00000001

        spectra_zen_in = spectra_zen_in_l - spectra_zen_in_d
        spectra_zen_in[spectra_zen_in<=0.0] = 0.00000001

        # self.primary_response_zen_si = lampStd_zen_si / spectra_zen_si / int_time_si
        # self.primary_response_zen_in = lampStd_zen_in / spectra_zen_in / int_time_in
        self.primary_response_zen_si = spectra_zen_si / int_time_si / lampStd_zen_si
        self.primary_response_zen_in = spectra_zen_in / int_time_in / lampStd_zen_in


        # for nadir
        fname_pattern = '%s/nadir/s%di%d/cal/*.OSA2' % (fdir_primary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_PRIMARY_RESPONSE]: cannot find data for nadir cal.')
        fname = fnames[0]
        spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR_V1(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_nad_si_l = spectra_mean[:, 2]
        spectra_nad_in_l = spectra_mean[:, 3]

        fname_pattern = '%s/nadir/s%di%d/dark/*.OSA2' % (fdir_primary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_PRIMARY_RESPONSE]: cannot find data for nadir dark.')
        fname = fnames[0]
        spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR_V1(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_nad_si_d = spectra_mean[:, 2]
        spectra_nad_in_d = spectra_mean[:, 3]

        spectra_nad_si = spectra_nad_si_l - spectra_nad_si_d
        spectra_nad_si[spectra_nad_si<=0.0] = 0.00000001

        spectra_nad_in = spectra_nad_in_l - spectra_nad_in_d
        spectra_nad_in[spectra_nad_in<=0.0] = 0.00000001

        # self.primary_response_nad_si = lampStd_nad_si / spectra_nad_si / int_time_si
        # self.primary_response_nad_in = lampStd_nad_in / spectra_nad_in / int_time_in
        self.primary_response_nad_si = spectra_nad_si / int_time_si / lampStd_nad_si
        self.primary_response_nad_in = spectra_nad_in / int_time_in / lampStd_nad_in
        # ---------------------------------------------------------------------------


    def CAL_TRANSFER(self, fdir_transfer, int_time_si=60, int_time_in=300, lampTag='150C'):

        # read in SSFR data collected during transfer transfer
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # for zenith
        fname_pattern = '%s/zenith/s%di%d/cal/*.OSA2' % (fdir_transfer, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_TRANSFER]: cannot find data for zenith cal.')
        fname = fnames[0]
        spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR_V1(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_zen_si_l = spectra_mean[:, 0]
        spectra_zen_in_l = spectra_mean[:, 1]

        fname_pattern = '%s/zenith/s%di%d/dark/*.OSA2' % (fdir_transfer, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_TRANSFER]: cannot find data for zenith dark.')
        fname = fnames[0]
        spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR_V1(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_zen_si_d = spectra_mean[:, 0]
        spectra_zen_in_d = spectra_mean[:, 1]

        spectra_zen_si = spectra_zen_si_l - spectra_zen_si_d
        spectra_zen_si[spectra_zen_si<=0.0] = 0.00000001

        spectra_zen_in = spectra_zen_in_l - spectra_zen_in_d
        spectra_zen_in[spectra_zen_in<=0.0] = 0.00000001

        self.field_lamp_zen_si = spectra_zen_si / int_time_si / self.primary_response_zen_si
        self.field_lamp_zen_in = spectra_zen_in / int_time_in / self.primary_response_zen_in


        # for nadir
        fname_pattern = '%s/nadir/s%di%d/cal/*.OSA2' % (fdir_transfer, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_TRANSFER]: cannot find data for nadir cal.')
        fname = fnames[0]
        spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR_V1(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_nad_si_l = spectra_mean[:, 2]
        spectra_nad_in_l = spectra_mean[:, 3]

        fname_pattern = '%s/nadir/s%di%d/dark/*.OSA2' % (fdir_transfer, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_TRANSFER]: cannot find data for nadir dark.')
        fname = fnames[0]
        spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR_V1(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_nad_si_d = spectra_mean[:, 2]
        spectra_nad_in_d = spectra_mean[:, 3]

        spectra_nad_si = spectra_nad_si_l - spectra_nad_si_d
        spectra_nad_si[spectra_nad_si<=0.0] = 0.00000001

        spectra_nad_in = spectra_nad_in_l - spectra_nad_in_d
        spectra_nad_in[spectra_nad_in<=0.0] = 0.00000001

        self.field_lamp_nad_si = spectra_nad_si / int_time_si / self.primary_response_nad_si
        self.field_lamp_nad_in = spectra_nad_in / int_time_in / self.primary_response_nad_in
        # ---------------------------------------------------------------------------


    def CAL_SECONDARY_RESPONSE(self, fdir_secondary, int_time_si=60, int_time_in=300, lampTag='150C'):

        # read in SSFR data collected during field calibration
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # for zenith
        fname_pattern = '%s/zenith/s%di%d/cal/*.OSA2' % (fdir_secondary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_SECONDARY_RESPONSE]: cannot find data for zenith cal.')
        fname = fnames[0]
        spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR_V1(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_zen_si_l = spectra_mean[:, 0]
        spectra_zen_in_l = spectra_mean[:, 1]

        fname_pattern = '%s/zenith/s%di%d/dark/*.OSA2' % (fdir_secondary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_SECONDARY_RESPONSE]: cannot find data for zenith dark.')
        fname = fnames[0]
        spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR_V1(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_zen_si_d = spectra_mean[:, 0]
        spectra_zen_in_d = spectra_mean[:, 1]

        spectra_zen_si = spectra_zen_si_l - spectra_zen_si_d
        spectra_zen_si[spectra_zen_si<=0.0] = 0.00000001

        spectra_zen_in = spectra_zen_in_l - spectra_zen_in_d
        spectra_zen_in[spectra_zen_in<=0.0] = 0.00000001

        self.secondary_response_zen_si = spectra_zen_si / int_time_si / self.field_lamp_zen_si
        self.secondary_response_zen_in = spectra_zen_in / int_time_in / self.field_lamp_zen_in


        # for nadir
        fname_pattern = '%s/nadir/s%di%d/cal/*.OSA2' % (fdir_secondary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_SECONDARY_RESPONSE]: cannot find data for nadir cal.')
        fname = fnames[0]
        spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR_V1(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_nad_si_l = spectra_mean[:, 2]
        spectra_nad_in_l = spectra_mean[:, 3]

        fname_pattern = '%s/nadir/s%di%d/dark/*.OSA2' % (fdir_secondary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_SECONDARY_RESPONSE]: cannot find data for nadir dark.')
        fname = fnames[0]
        spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR_V1(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_nad_si_d = spectra_mean[:, 2]
        spectra_nad_in_d = spectra_mean[:, 3]

        spectra_nad_si = spectra_nad_si_l - spectra_nad_si_d
        spectra_nad_si[spectra_nad_si<=0.0] = 0.00000001

        spectra_nad_in = spectra_nad_in_l - spectra_nad_in_d
        spectra_nad_in[spectra_nad_in<=0.0] = 0.00000001

        self.secondary_response_nad_si = spectra_nad_si / int_time_si / self.field_lamp_nad_si
        self.secondary_response_nad_in = spectra_nad_in / int_time_in / self.field_lamp_nad_in
        # ---------------------------------------------------------------------------








class CALIBRATION_CU_SSFR:

    def __init__(self, fdir_primary, fdir_transfer, fdir_secondary):

        self.CAL_WAVELENGTH()
        self.CAL_PRIMARY_RESPONSE(fdir_primary)
        self.CAL_TRANSFER(fdir_transfer)
        self.CAL_SECONDARY_RESPONSE(fdir_secondary)


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


    def CAL_PRIMARY_RESPONSE(self, fdir_primary, int_time_si=60, int_time_in=300, lampTag='f-1324', fdirLamp='/Users/hoch4240/Chen/other/data/aux_ssfr'):

        # read in calibrated lamp data and interpolated at SSFR wavelengths
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
        # ---------------------------------------------------------------------------

        # so far we have (W m^-2 nm^-1 as a function of wavelength)

        # read in SSFR data collected during primary calibration
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # for zenith
        fname_pattern = '%s/zenith/s%di%d/cal/*.SKS' % (fdir_primary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_PRIMARY_RESPONSE]: cannot find data for zenith cal.')
        fname = fnames[0]
        comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR_V2(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_zen_si_l = spectra_mean[:, 0]
        spectra_zen_in_l = spectra_mean[:, 1]

        fname_pattern = '%s/zenith/s%di%d/dark/*.SKS' % (fdir_primary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_PRIMARY_RESPONSE]: cannot find data for zenith dark.')
        fname = fnames[0]
        comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR_V2(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_zen_si_d = spectra_mean[:, 0]
        spectra_zen_in_d = spectra_mean[:, 1]

        spectra_zen_si = spectra_zen_si_l - spectra_zen_si_d
        spectra_zen_si[spectra_zen_si<=0.0] = 0.00000001

        spectra_zen_in = spectra_zen_in_l - spectra_zen_in_d
        spectra_zen_in[spectra_zen_in<=0.0] = 0.00000001

        self.primary_response_zen_si = spectra_zen_si / int_time_si / lampStd_zen_si
        self.primary_response_zen_in = spectra_zen_in / int_time_in / lampStd_zen_in


        # for nadir
        fname_pattern = '%s/nadir/s%di%d/cal/*.SKS' % (fdir_primary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_PRIMARY_RESPONSE]: cannot find data for nadir cal.')
        fname = fnames[0]
        comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR_V2(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_nad_si_l = spectra_mean[:, 2]
        spectra_nad_in_l = spectra_mean[:, 3]

        fname_pattern = '%s/nadir/s%di%d/dark/*.SKS' % (fdir_primary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_PRIMARY_RESPONSE]: cannot find data for nadir dark.')
        fname = fnames[0]
        comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR_V2(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_nad_si_d = spectra_mean[:, 2]
        spectra_nad_in_d = spectra_mean[:, 3]

        spectra_nad_si = spectra_nad_si_l - spectra_nad_si_d
        spectra_nad_si[spectra_nad_si<=0.0] = 0.00000001

        spectra_nad_in = spectra_nad_in_l - spectra_nad_in_d
        spectra_nad_in[spectra_nad_in<=0.0] = 0.00000001

        self.primary_response_nad_si = spectra_nad_si / int_time_si / lampStd_nad_si
        self.primary_response_nad_in = spectra_nad_in / int_time_in / lampStd_nad_in
        # ---------------------------------------------------------------------------


    def CAL_TRANSFER(self, fdir_transfer, int_time_si=60, int_time_in=300, lampTag='150C'):

        # read in SSFR data collected during transfer transfer
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # for zenith
        fname_pattern = '%s/zenith/s%di%d/cal/*.SKS' % (fdir_transfer, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_TRANSFER]: cannot find data for zenith cal.')
        fname = fnames[0]
        comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR_V2(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_zen_si_l = spectra_mean[:, 0]
        spectra_zen_in_l = spectra_mean[:, 1]

        fname_pattern = '%s/zenith/s%di%d/dark/*.SKS' % (fdir_transfer, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_TRANSFER]: cannot find data for zenith dark.')
        fname = fnames[0]
        comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR_V2(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_zen_si_d = spectra_mean[:, 0]
        spectra_zen_in_d = spectra_mean[:, 1]

        spectra_zen_si = spectra_zen_si_l - spectra_zen_si_d
        spectra_zen_si[spectra_zen_si<=0.0] = 0.00000001

        spectra_zen_in = spectra_zen_in_l - spectra_zen_in_d
        spectra_zen_in[spectra_zen_in<=0.0] = 0.00000001

        self.field_lamp_zen_si = spectra_zen_si / int_time_si / self.primary_response_zen_si
        self.field_lamp_zen_in = spectra_zen_in / int_time_in / self.primary_response_zen_in


        # for nadir
        fname_pattern = '%s/nadir/s%di%d/cal/*.SKS' % (fdir_transfer, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_TRANSFER]: cannot find data for nadir cal.')
        fname = fnames[0]
        comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR_V2(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_nad_si_l = spectra_mean[:, 2]
        spectra_nad_in_l = spectra_mean[:, 3]

        fname_pattern = '%s/nadir/s%di%d/dark/*.SKS' % (fdir_transfer, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_TRANSFER]: cannot find data for nadir dark.')
        fname = fnames[0]
        comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR_V2(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_nad_si_d = spectra_mean[:, 2]
        spectra_nad_in_d = spectra_mean[:, 3]

        spectra_nad_si = spectra_nad_si_l - spectra_nad_si_d
        spectra_nad_si[spectra_nad_si<=0.0] = 0.00000001

        spectra_nad_in = spectra_nad_in_l - spectra_nad_in_d
        spectra_nad_in[spectra_nad_in<=0.0] = 0.00000001

        self.field_lamp_nad_si = spectra_nad_si / int_time_si / self.primary_response_nad_si
        self.field_lamp_nad_in = spectra_nad_in / int_time_in / self.primary_response_nad_in
        # ---------------------------------------------------------------------------


    def CAL_SECONDARY_RESPONSE(self, fdir_secondary, int_time_si=60, int_time_in=300, lampTag='150C'):

        # read in SSFR data collected during field calibration
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # for zenith
        fname_pattern = '%s/zenith/s%di%d/cal/*.SKS' % (fdir_secondary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_SECONDARY_RESPONSE]: cannot find data for zenith cal.')
        fname = fnames[0]
        comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR_V2(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_zen_si_l = spectra_mean[:, 0]
        spectra_zen_in_l = spectra_mean[:, 1]

        fname_pattern = '%s/zenith/s%di%d/dark/*.SKS' % (fdir_secondary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_SECONDARY_RESPONSE]: cannot find data for zenith dark.')
        fname = fnames[0]
        comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR_V2(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_zen_si_d = spectra_mean[:, 0]
        spectra_zen_in_d = spectra_mean[:, 1]

        spectra_zen_si = spectra_zen_si_l - spectra_zen_si_d
        spectra_zen_si[spectra_zen_si<=0.0] = 0.00000001

        spectra_zen_in = spectra_zen_in_l - spectra_zen_in_d
        spectra_zen_in[spectra_zen_in<=0.0] = 0.00000001

        self.secondary_response_zen_si = spectra_zen_si / int_time_si / self.field_lamp_zen_si
        self.secondary_response_zen_in = spectra_zen_in / int_time_in / self.field_lamp_zen_in


        # for nadir
        fname_pattern = '%s/nadir/s%di%d/cal/*.SKS' % (fdir_secondary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_SECONDARY_RESPONSE]: cannot find data for nadir cal.')
        fname = fnames[0]
        comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR_V2(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_nad_si_l = spectra_mean[:, 2]
        spectra_nad_in_l = spectra_mean[:, 3]

        fname_pattern = '%s/nadir/s%di%d/dark/*.SKS' % (fdir_secondary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_SECONDARY_RESPONSE]: cannot find data for nadir dark.')
        fname = fnames[0]
        comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR_V2(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_nad_si_d = spectra_mean[:, 2]
        spectra_nad_in_d = spectra_mean[:, 3]

        spectra_nad_si = spectra_nad_si_l - spectra_nad_si_d
        spectra_nad_si[spectra_nad_si<=0.0] = 0.00000001

        spectra_nad_in = spectra_nad_in_l - spectra_nad_in_d
        spectra_nad_in[spectra_nad_in<=0.0] = 0.00000001

        self.secondary_response_nad_si = spectra_nad_si / int_time_si / self.field_lamp_nad_si
        self.secondary_response_nad_in = spectra_nad_in / int_time_in / self.field_lamp_nad_in
        # ---------------------------------------------------------------------------








class CALIBRATION_CU_SSFR_20180221:

    def __init__(self, fdir_secondary):

        self.CAL_WAVELENGTH()
        self.CAL_SECONDARY_RESPONSE(fdir_secondary)


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


    def CAL_SECONDARY_RESPONSE(self, fdir_secondary, int_time_si=45, int_time_in=250, lampTag='150C'):

        fdir_primary0    = '/Users/hoch4240/Chen/work/00_reuse/SSFR-util/data/ssfr6/20171106/1324'
        fdir_transfer0   = '/Users/hoch4240/Chen/work/00_reuse/SSFR-util/data/ssfr6/20171106/150C'
        fdir_secondary0  = '/Users/hoch4240/Chen/work/00_reuse/SSFR-util/data/ssfr6/20180221/150C'
        nasa_cal = CALIBRATION_NASA_SSFR(fdir_primary0, fdir_transfer0, fdir_secondary0)

        self.field_lamp_zen_si = np.interp(self.wvl_zen_si, nasa_cal.wvl_zen_si, nasa_cal.field_lamp_zen_si)
        self.field_lamp_zen_in = np.interp(self.wvl_zen_in, nasa_cal.wvl_zen_in[::-1], nasa_cal.field_lamp_zen_in[::-1])
        self.field_lamp_nad_si = np.interp(self.wvl_nad_si, nasa_cal.wvl_nad_si, nasa_cal.field_lamp_nad_si)
        self.field_lamp_nad_in = np.interp(self.wvl_nad_in, nasa_cal.wvl_nad_in[::-1], nasa_cal.field_lamp_nad_in[::-1])

        # this is not a universal reader
        # read in SSFR data collected during field calibration
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # for zenith
        fname_pattern = '%s/zen/*.SKS' % (fdir_secondary)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_SECONDARY_RESPONSE]: cannot find data for zenith cal.')
        fname = fnames[0]

        comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR_V2(fname)
        shutter[:2] = -1; shutter[28:32] = -1; shutter[128:132] = -1; shutter[178:182] = -1; shutter[198:] = -1

        logic_l = (shutter==0)
        logic_d = (shutter==1)

        spectra_mean_l = np.mean(spectra[logic_l, :, :], axis=0)
        spectra_mean_d = np.mean(spectra[logic_d, :, :], axis=0)

        spectra_zen_si = spectra_mean_l[:, 0] - spectra_mean_d[:, 0]
        spectra_zen_in = spectra_mean_l[:, 1] - spectra_mean_d[:, 1]

        spectra_zen_si[spectra_zen_si<=0.0] = 0.00000001
        spectra_zen_in[spectra_zen_in<=0.0] = 0.00000001

        self.secondary_response_zen_si = spectra_zen_si / int_time_si / self.field_lamp_zen_si
        self.secondary_response_zen_in = spectra_zen_in / int_time_in / self.field_lamp_zen_in

        # for nadir
        fname_pattern = '%s/nad/*.SKS' % (fdir_secondary)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_SECONDARY_RESPONSE]: cannot find data for nadir cal.')
        fname = fnames[0]

        comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN = READ_CU_SSFR_V2(fname)
        shutter[:2] = -1; shutter[48:52] = -1; shutter[148:152] = -1; shutter[198:] = -1

        logic_l = (shutter==0)
        logic_d = (shutter==1)

        spectra_mean_l = np.mean(spectra[logic_l, :, :], axis=0)
        spectra_mean_d = np.mean(spectra[logic_d, :, :], axis=0)

        spectra_nad_si = spectra_mean_l[:, 2] - spectra_mean_d[:, 2]
        spectra_nad_in = spectra_mean_l[:, 3] - spectra_mean_d[:, 3]

        spectra_nad_si[spectra_nad_si<=0.0] = 0.00000001
        spectra_nad_in[spectra_nad_in<=0.0] = 0.00000001

        self.secondary_response_nad_si = spectra_nad_si / int_time_si / self.field_lamp_nad_si
        self.secondary_response_nad_in = spectra_nad_in / int_time_in / self.field_lamp_nad_in
        # ---------------------------------------------------------------------------







class CALIBRATION_CU_SSFR_20180228:

    def __init__(self, fdir_secondary):

        self.CAL_WAVELENGTH()
        self.CAL_SECONDARY_RESPONSE(fdir_secondary)


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


    def CAL_SECONDARY_RESPONSE(self, fdir_secondary, int_time_si=45, int_time_in=250, lampTag='150C'):

        fdir_primary0    = '/Users/hoch4240/Chen/work/00_reuse/SSFR-util/data/ssfr6/20171106/1324'
        fdir_transfer0   = '/Users/hoch4240/Chen/work/00_reuse/SSFR-util/data/ssfr6/20171106/150C'
        fdir_secondary0  = '/Users/hoch4240/Chen/work/00_reuse/SSFR-util/data/ssfr6/20180221/150C'
        nasa_cal = CALIBRATION_NASA_SSFR(fdir_primary0, fdir_transfer0, fdir_secondary0)

        self.field_lamp_zen_si = np.interp(self.wvl_zen_si, nasa_cal.wvl_zen_si, nasa_cal.field_lamp_zen_si)
        self.field_lamp_zen_in = np.interp(self.wvl_zen_in, nasa_cal.wvl_zen_in[::-1], nasa_cal.field_lamp_zen_in[::-1])
        self.field_lamp_nad_si = np.interp(self.wvl_nad_si, nasa_cal.wvl_nad_si, nasa_cal.field_lamp_nad_si)
        self.field_lamp_nad_in = np.interp(self.wvl_nad_in, nasa_cal.wvl_nad_in[::-1], nasa_cal.field_lamp_nad_in[::-1])

        # for zenith
        fname_pattern = '%s/zenith/s%di%d/cal/*.OSA2' % (fdir_secondary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_SECONDARY_RESPONSE]: cannot find data for zenith cal.')
        fname = fnames[0]
        spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR_V1(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_zen_si_l = spectra_mean[:, 0]
        spectra_zen_in_l = spectra_mean[:, 1]

        fname_pattern = '%s/zenith/s%di%d/dark/*.OSA2' % (fdir_secondary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_SECONDARY_RESPONSE]: cannot find data for zenith dark.')
        fname = fnames[0]
        spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR_V1(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_zen_si_d = spectra_mean[:, 0]
        spectra_zen_in_d = spectra_mean[:, 1]

        spectra_zen_si = spectra_zen_si_l - spectra_zen_si_d
        spectra_zen_si[spectra_zen_si<=0.0] = 0.00000001

        spectra_zen_in = spectra_zen_in_l - spectra_zen_in_d
        spectra_zen_in[spectra_zen_in<=0.0] = 0.00000001

        self.secondary_response_zen_si = spectra_zen_si / int_time_si / self.field_lamp_zen_si
        self.secondary_response_zen_in = spectra_zen_in / int_time_in / self.field_lamp_zen_in


        # for nadir
        fname_pattern = '%s/nadir/s%di%d/cal/*.OSA2' % (fdir_secondary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_SECONDARY_RESPONSE]: cannot find data for nadir cal.')
        fname = fnames[0]
        spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR_V1(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_nad_si_l = spectra_mean[:, 2]
        spectra_nad_in_l = spectra_mean[:, 3]

        fname_pattern = '%s/nadir/s%di%d/dark/*.OSA2' % (fdir_secondary, int_time_si, int_time_in)
        fnames  = glob.glob(fname_pattern)
        if len(fnames) != 1:
            exit('Error [CAL_SECONDARY_RESPONSE]: cannot find data for nadir dark.')
        fname = fnames[0]
        spectra, shutter, int_time, temp, jday, qual_flag, iterN = READ_NASA_SSFR_V1(fname)
        spectra_mean = np.mean(spectra[2:-2, :, :], axis=0)
        spectra_nad_si_d = spectra_mean[:, 2]
        spectra_nad_in_d = spectra_mean[:, 3]

        spectra_nad_si = spectra_nad_si_l - spectra_nad_si_d
        spectra_nad_si[spectra_nad_si<=0.0] = 0.00000001

        spectra_nad_in = spectra_nad_in_l - spectra_nad_in_d
        spectra_nad_in[spectra_nad_in<=0.0] = 0.00000001

        self.secondary_response_nad_si = spectra_nad_si / int_time_si / self.field_lamp_nad_si
        self.secondary_response_nad_in = spectra_nad_in / int_time_in / self.field_lamp_nad_in
        # ---------------------------------------------------------------------------


def READ_SKS_V2(fname, headLen=148, dataLen=2276, verbose=False):

    fileSize = os.path.getsize(fname)
    if fileSize > headLen:
        iterN   = (fileSize-headLen) // dataLen
        residual = (fileSize-headLen) %  dataLen
        if residual != 0:
            print('Warning [READ_SKS_V2]: %s has invalid data size.' % fname)
    else:
        exit('Error [READ_SKS_V2]: %s has invalid file size.' % fname)

    spectra    = np.zeros((iterN, 256, 4), dtype=np.float64) # spectra
    shutter    = np.zeros(iterN          , dtype=np.int32  ) # shutter status (1:closed, 0:open)
    int_time   = np.zeros((iterN, 4)     , dtype=np.float64) # integration time [ms]
    temp       = np.zeros((iterN, 11)    , dtype=np.float64) # temperature
    qual_flag  = np.ones(iterN           , dtype=np.int32)   # quality flag (1:good, 0:bad)
    jday_NSF   = np.zeros(iterN          , dtype=np.float64)
    jday_cRIO  = np.zeros(iterN          , dtype=np.float64)

    f           = open(fname, 'rb')
    # read head
    headRec   = f.read(headLen)
    head      = struct.unpack('<B144s3B', headRec)
    if head[0] != 144:
        f.seek(0)
    else:
        comment = head[1]

    if verbose:
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Comments in %s...' % fname.split('/')[-1])
        print(comment)
        print('--------------------------------------------------')

    # read data record
    for i in range(iterN):
        dataRec = f.read(dataLen)
        # ---------------------------------------------------------------------------------------------------------------
        # d9l: frac_second[d] , second[l] , minute[l] , hour[l] , day[l] , month[l] , year[l] , dow[l] , doy[l] , DST[l]
        # d9l: frac_second0[d], second0[l], minute0[l], hour0[l], day0[l], month0[l], year0[l], dow0[l], doy0[l], DST0[l]
        # l9d: null[l], temp(9)[9d]
        # --------------------------          below repeat for sz, sn, iz, in          ----------------------------------
        # l2Bl: int_time[l], shutter[B], EOS[B], null[l]
        # 257h: spectra(257)
        # ---------------------------------------------------------------------------------------------------------------
        #data     = struct.unpack('<d9ld9ll9dl2Bl257hl2Bl257hl2Bl257hlBBl257h', dataRec)
        data     = struct.unpack('<d9ld9ll11dl2Bl257hl2Bl257hl2Bl257hlBBl257h', dataRec)

        dataHead = data[:32]
        dataSpec = np.transpose(np.array(data[32:]).reshape((4, 261)))[:, [0, 2, 1, 3]]
        # [0, 2, 1, 3]: change order from 'sz, sn, iz, in' to 'sz, iz, sn, in'
        # transpose: change shape from (4, 261) to (261, 4)

        shutter_logic = (np.unique(dataSpec[1, :]).size != 1)
        eos_logic     = any(dataSpec[2, :] != 1)
        null_logic    = any(dataSpec[3, :] != 257)
        order_logic   = not np.array_equal(dataSpec[4, :], np.array([0, 2, 1, 3]))
        if any([shutter_logic, eos_logic, null_logic, order_logic]):
            qual_flag[i] = 0

        if True:
            spectra[i, :, :]  = dataSpec[5:, :]
            shutter[i]        = dataSpec[1, 0]
            int_time[i, :]    = dataSpec[0, :]
            temp[i, :]        = dataHead[21:]

            dtime          = datetime.datetime(dataHead[6] , dataHead[5] , dataHead[4] , dataHead[3] , dataHead[2] , dataHead[1] , int(round(dataHead[0]*1000000.0)))
            dtime0         = datetime.datetime(dataHead[16], dataHead[15], dataHead[14], dataHead[13], dataHead[12], dataHead[11], int(round(dataHead[10]*1000000.0)))

            # calculate the proleptic Gregorian ordinal of the date
            jday_NSF[i]    = (dtime  - datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0
            jday_cRIO[i]   = (dtime0 - datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0

    return comment, spectra, shutter, int_time, temp, jday_NSF, jday_cRIO, qual_flag, iterN

class READ_SKS:

    def __init__(self, fnames, Ndata=600, secOffset=0.0, config=None):

        if type(fnames) is not list:
            exit('Error [READ_SKS]: input variable "fnames" should be in list type.')
        Nx         = Ndata * len(fnames)
        comment    = []
        spectra    = np.zeros((Nx, 256, 4), dtype=np.float64) # spectra
        shutter    = np.zeros(Nx          , dtype=np.int32  ) # shutter status (1:closed, 0:open)
        int_time   = np.zeros((Nx, 4)     , dtype=np.float64) # integration time [ms]
        temp       = np.zeros((Nx, 11)    , dtype=np.float64) # temperature
        qual_flag  = np.zeros(Nx          , dtype=np.int32)
        jday_NSF   = np.zeros(Nx          , dtype=np.float64)
        jday_cRIO  = np.zeros(Nx          , dtype=np.float64)

        Nstart = 0
        for fname in fnames:
            comment0, spectra0, shutter0, int_time0, temp0, jday_NSF0, jday_cRIO0, qual_flag0, iterN0 = READ_SKS_V2(fname)
            comment.append(comment0)

            Nend = iterN0 + Nstart

            spectra[Nstart:Nend, ...]    = spectra0
            shutter[Nstart:Nend, ...]    = shutter0
            int_time[Nstart:Nend, ...]   = int_time0
            temp[Nstart:Nend, ...]       = temp0
            jday_NSF[Nstart:Nend, ...]   = jday_NSF0
            jday_cRIO[Nstart:Nend, ...]  = jday_cRIO0
            qual_flag[Nstart:Nend, ...]  = qual_flag0

            Nstart = Nend

        if config != None:
            self.config = config

        self.comment    = comment
        self.spectra    = spectra[:Nend, ...]
        self.shutter    = shutter[:Nend, ...]
        self.int_time   = int_time[:Nend, ...]
        self.temp       = temp[:Nend, ...]
        self.jday_NSF   = jday_NSF[:Nend, ...]
        self.jday_cRIO  = jday_cRIO[:Nend, ...]
        self.qual_flag  = qual_flag[:Nend, ...]
        self.shutter_ori= self.shutter.copy()

        self.jday = self.jday_NSF.copy()
        #self.jday = self.jday_cRIO[0] + 0.5/86400.0 * np.arange(self.jday_cRIO.size)
        self.tmhr = (self.jday - int(self.jday[0])) * 24.0
        self.tmhr_corr = self.tmhr.copy() - secOffset/3600.0

        self.DARK_CORR()
        self.COUNT2FLUX(self.spectra_dark_corr)

        # f = h5py.File('20180228_cu.h5', 'w')
        # f['spectra_flux_zen'] = self.spectra_flux_zen
        # f['spectra_flux_nad'] = self.spectra_flux_nad
        # f['wvl_zen'] = self.wvl_zen
        # f['wvl_nad'] = self.wvl_nad
        # f['shutter'] = self.shutter
        # f['tmhr']    = self.tmhr_corr
        # f.close()

        f = h5py.File('20180228_CU_20180228.h5', 'w')
        f['spectra_flux_zen'] = self.spectra_flux_zen
        f['spectra_flux_nad'] = self.spectra_flux_nad
        f['wvl_zen'] = self.wvl_zen
        f['wvl_nad'] = self.wvl_nad
        f['shutter'] = self.shutter
        f['tmhr']    = self.tmhr_corr
        f.close()


    def DARK_CORR(self, mode=-1, darkExtend=2, lightExtend=2, countOffset=0, lightThr=10, darkThr=5, fillValue=10):

        if self.shutter[0] == 0:
            darkL = np.array([], dtype=np.int32)
            darkR = np.array([0], dtype=np.int32)
        else:
            darkR = np.array([], dtype=np.int32)
            darkL = np.array([0], dtype=np.int32)

        darkL0 = np.squeeze(np.argwhere((self.shutter[1:]-self.shutter[:-1]) ==  1)) + 1
        darkL  = np.hstack((darkL, darkL0))

        darkR0 = np.squeeze(np.argwhere((self.shutter[1:]-self.shutter[:-1]) == -1)) + 1
        darkR  = np.hstack((darkR, darkR0))

        if self.shutter[-1] == 0:
            darkL = np.hstack((darkL, self.shutter.size))
        else:
            darkR = np.hstack((darkR, self.shutter.size))

        self.spectra_dark_corr = self.spectra.copy() + countOffset
        self.dark_offset       = np.zeros(self.spectra.shape, dtype=np.float64)
        self.dark_std          = np.zeros(self.spectra.shape, dtype=np.float64)

        Nrecord, Nchannel, Nsensor = self.spectra.shape
        if mode == -1:
            if darkL.size-darkR.size==0:
                if darkL[0]>darkR[0] and darkL[-1]>darkR[-1]:
                    darkL = darkL[:-1]
                    darkR = darkR[1:]
            elif darkL.size-darkR.size==1:
                if darkL[0]>darkR[0] and darkL[-1]<darkR[-1]:
                    darkL = darkL[1:]
                elif darkL[0]<darkR[0] and darkL[-1]>darkR[-1]:
                    darkL = darkL[:-1]
            elif darkR.size-darkL.size==1:
                if darkL[0]>darkR[0] and darkL[-1]<darkR[-1]:
                    darkR = darkR[1:]
                elif darkL[0]<darkR[0] and darkL[-1]>darkR[-1]:
                    darkR = darkR[:-1]
            else:
                exit('Error [READ_SKS.DARK_CORR]: darkL and darkR are wrong.')

            for i in range(darkL.size-1):
                if darkR[i] < darkL[i]:
                    exit('Error [READ_SKS.DARK_CORR]: darkL > darkR.')

                darkLL = darkL[i] + darkExtend
                darkLR = darkR[i] - darkExtend
                darkRL = darkL[i+1] + darkExtend
                darkRR = darkR[i+1] - darkExtend

                if i == 0:
                    self.shutter[:darkLL] = fillValue  # omit the data before the first dark cycle

                lightL = darkR[i]   + lightExtend
                lightR = darkL[i+1] - lightExtend

                if lightR-lightL>lightThr and darkLR-darkLL>darkThr and darkRR-darkRL>darkThr:

                    self.shutter[darkL[i]:darkLL] = fillValue
                    self.shutter[darkLR:darkR[i]] = fillValue
                    self.shutter[darkR[i]:lightL] = fillValue
                    self.shutter[lightR:darkL[i+1]] = fillValue
                    self.shutter[darkL[i+1]:darkRL] = fillValue
                    self.shutter[darkRR:darkR[i+1]] = fillValue

                    int_dark  = np.append(self.int_time[darkLL:darkLR], self.int_time[darkRL:darkRR]).mean()
                    int_light = self.int_time[lightL:lightR].mean()

                    if np.abs(int_dark - int_light) > 0.0001:
                        self.shutter[lightL:lightR] = fillValue
                    else:
                        interp_x  = np.append(self.tmhr_corr[darkLL:darkLR], self.tmhr_corr[darkRL:darkRR])
                        if i==darkL.size-2:
                            target_x  = self.tmhr_corr[darkL[i]:darkR[i+1]]
                        else:
                            target_x  = self.tmhr_corr[darkL[i]:darkL[i+1]]

                        for ichan in range(Nchannel):
                            for isen in range(Nsensor):
                                interp_y = np.append(self.spectra[darkLL:darkLR,ichan,isen], self.spectra[darkRL:darkRR,ichan,isen])
                                slope, intercept, r_value, p_value, std_err  = stats.linregress(interp_x, interp_y)
                                if i==darkL.size-2:
                                    self.dark_offset[darkL[i]:darkR[i+1], ichan, isen] = target_x*slope + intercept
                                    self.spectra_dark_corr[darkL[i]:darkR[i+1], ichan, isen] -= self.dark_offset[darkL[i]:darkR[i+1], ichan, isen]
                                    self.dark_std[darkL[i]:darkR[i+1], ichan, isen] = np.std(interp_y)
                                else:
                                    self.dark_offset[darkL[i]:darkL[i+1], ichan, isen] = target_x*slope + intercept
                                    self.spectra_dark_corr[darkL[i]:darkL[i+1], ichan, isen] -= self.dark_offset[darkL[i]:darkL[i+1], ichan, isen]
                                    self.dark_std[darkL[i]:darkL[i+1], ichan, isen] = np.std(interp_y)

                else:
                    self.shutter[darkL[i]:darkR[i+1]] = fillValue

            self.shutter[darkRR:] = fillValue  # omit the data after the last dark cycle

        elif mode == -2:
            print('Message [DARK_CORR]: Not implemented...')

        elif mode == -3:

            #if darkL.size-darkR.size==0:
                #if darkL[0]>darkR[0] and darkL[-1]>darkR[-1]:
                    #darkL = darkL[:-1]
                    #darkR = darkR[1:]
            #elif darkL.size-darkR.size==1:
                #if darkL[0]>darkR[0] and darkL[-1]<darkR[-1]:
                    #darkL = darkL[1:]
                #elif darkL[0]<darkR[0] and darkL[-1]>darkR[-1]:
                    #darkL = darkL[:-1]
            #elif darkR.size-darkL.size==1:
                #if darkL[0]>darkR[0] and darkL[-1]<darkR[-1]:
                    #darkR = darkR[1:]
                #elif darkL[0]<darkR[0] and darkL[-1]>darkR[-1]:
                    #darkR = darkR[:-1]
            #else:
                #exit('Error [READ_SKS.DARK_CORR]: darkL and darkR are wrong.')

            for i in range(darkR.size):
                darkLL = darkL[i]   + darkExtend
                darkLR = darkR[i]   - darkExtend
                lightL = darkR[i]   + lightExtend
                lightR = darkL[i+1] - lightExtend

                self.shutter[darkL[i]:darkLL] = fillValue
                self.shutter[darkLR:darkR[i]] = fillValue
                self.shutter[darkR[i]:lightL] = fillValue
                self.shutter[lightR:darkL[i+1]] = fillValue

                int_dark  = self.int_time[darkLL:darkLR].mean()
                int_light = self.int_time[lightL:lightR].mean()
                if np.abs(int_dark - int_light) > 0.0001:
                    self.shutter[lightL:lightR] = fillValue
                    exit('Error [READ_SKS.DARK_CORR]: inconsistent integration time.')
                else:
                    for itmhr in range(darkLR, lightR):
                        for isen in range(Nsensor):
                            dark_offset0 = np.mean(self.spectra[darkLL:darkLR, :, isen], axis=0)
                            self.dark_offset[itmhr, :, isen] = dark_offset0
                    self.spectra_dark_corr[lightL:lightR,:,:] -= self.dark_offset[lightL:lightR,:,:]

        elif mode == -4:
            print('Message [DARK_CORR]: Not implemented...')


    def NLIN_CORR(self, fname_nlin, Nsen):
        #{{{
        int_time0 = np.mean(self.int_time[:, Nsen])
        f_nlin = readsav(fname_nlin)

        if abs(f_nlin.iin_-int_time0)>1.0e-5:
            exit('Error [READ_SKS]: Integration time do not match.')

        for iwvl in range(256):
            xx0   = self.spectra_nlin_corr[:,iwvl,Nsen].copy()
            xx    = np.zeros_like(xx0)
            yy    = np.zeros_like(xx0)
            self.spectra_nlin_corr[:,iwvl,Nsen] = np.nan
            logic_xx     = (xx0>-100)
            print('++++++++++++++++++++++++++++++++++++++++++++++++')
            print('range', f_nlin.mxm_[0,iwvl]*f_nlin.in_[iwvl], f_nlin.mxm_[1,iwvl]*f_nlin.in_[iwvl])
            print('good', logic_xx.sum(), xx0.size)
            xx0[logic_xx] = xx0[logic_xx]/f_nlin.in_[iwvl]

            if (f_nlin.bad_[1,iwvl]<1.0) and (f_nlin.mxm_[0,iwvl]>=1.0e-3):

                #+ data in range (0, minimum)
                yy_e = 0.0
                for ideg in range(f_nlin.gr_):
                    yy_e += f_nlin.res2_[ideg,iwvl]*f_nlin.mxm_[0,iwvl]**ideg
                slope = yy_e/f_nlin.mxm_[0,iwvl]
                logic_xx     = (xx0>-100) & (xx0<f_nlin.mxm_[0,iwvl])
                print('0-min', logic_xx.sum(), xx0.size)
                print('data', xx0[logic_xx])
                xx[xx<0]     = 0.0
                xx[logic_xx] = xx0[logic_xx]
                yy[logic_xx] = xx[logic_xx]*slope

                self.spectra_nlin_corr[logic_xx,iwvl,Nsen] = yy[logic_xx]*f_nlin.in_[iwvl]
                #-

                #+ data in range [minimum, maximum]
                logic_xx     = (xx0>=f_nlin.mxm_[0,iwvl]) & (xx0<=f_nlin.mxm_[1,iwvl])
                xx[logic_xx] = xx0[logic_xx]
                print('min-max', logic_xx.sum(), xx0.size)
                print('------------------------------------------------')
                for ideg in range(f_nlin.gr_):
                    yy[logic_xx] += f_nlin.res2_[ideg, iwvl]*xx[logic_xx]**ideg

                self.spectra_nlin_corr[logic_xx,iwvl,Nsen] = yy[logic_xx]*f_nlin.in_[iwvl]
                #-

        #}}}


    def COUNT2FLUX(self, countIn, wvl_zen_join=950.0, wvl_nad_join=950.0):

        """
        Convert digital count to flux (irradiance)
        """

        # fdir_secondary = '/Users/hoch4240/Chen/work/00_reuse/SSFR-util/data/20180221/cal'
        # cal = CALIBRATION_CU_SSFR_20180221(fdir_secondary)
        fdir_secondary = '/Users/hoch4240/Chen/work/00_reuse/SSFR-util/data/20180228/150C'
        cal = CALIBRATION_CU_SSFR_20180228(fdir_secondary)

        logic_zen_si = (cal.wvl_zen_si <= wvl_zen_join)
        logic_zen_in = (cal.wvl_zen_in >= wvl_zen_join)
        n_zen_si = logic_zen_si.sum()
        n_zen_in = logic_zen_in.sum()
        n_zen    = n_zen_si + n_zen_in
        self.wvl_zen = np.append(cal.wvl_zen_si[logic_zen_si], cal.wvl_zen_in[logic_zen_in][::-1])

        logic_nad_si = (cal.wvl_nad_si <= wvl_nad_join)
        logic_nad_in = (cal.wvl_nad_in >= wvl_nad_join)
        n_nad_si = logic_nad_si.sum()
        n_nad_in = logic_nad_in.sum()
        n_nad    = n_nad_si + n_nad_in
        self.wvl_nad = np.append(cal.wvl_nad_si[logic_nad_si], cal.wvl_nad_in[logic_nad_in][::-1])

        self.spectra_flux_zen = np.zeros((self.tmhr_corr.size, n_zen), dtype=np.float64)
        self.spectra_flux_nad = np.zeros((self.tmhr_corr.size, n_nad), dtype=np.float64)

        for i in range(self.tmhr_corr.size):
            if self.shutter[i] == 0:
                self.spectra_flux_zen[i, :n_zen_si] =  countIn[i, logic_zen_si, 0]/float(self.int_time[i, 0])/cal.secondary_response_zen_si[logic_zen_si]
                self.spectra_flux_zen[i, n_zen_si:] = (countIn[i, logic_zen_in, 1]/float(self.int_time[i, 1])/cal.secondary_response_zen_in[logic_zen_in])[::-1]
                self.spectra_flux_nad[i, :n_nad_si] =  countIn[i, logic_nad_si, 2]/float(self.int_time[i, 2])/cal.secondary_response_nad_si[logic_nad_si]
                self.spectra_flux_nad[i, n_nad_si:] = (countIn[i, logic_nad_in, 3]/float(self.int_time[i, 3])/cal.secondary_response_nad_in[logic_nad_in])[::-1]
            else:
                self.spectra_flux_zen[i, :] = -1.0
                self.spectra_flux_nad[i, :] = -1.0

        self.spectra_flux_zen[self.spectra_flux_zen<0.0] = np.nan
        self.spectra_flux_nad[self.spectra_flux_nad<0.0] = np.nan







def PLOT():
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







def PLOT_TIME_SERIES(wvl0):

    f = h5py.File('20180228_CU_20180221.h5', 'r')
    spectra_flux_zen = f['spectra_flux_zen'][...]
    spectra_flux_nad = f['spectra_flux_nad'][...]
    wvl_zen          = f['wvl_zen'][...]
    wvl_nad          = f['wvl_nad'][...]
    shutter          = f['shutter'][...]
    tmhr             = f['tmhr'][...]
    f.close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)

    index = np.argmin(np.abs(wvl_zen-wvl0))
    ax1.scatter(tmhr, spectra_flux_zen[:, index], edgecolor='red', marker='v', s=80, facecolor='none', label='CU SSFR Zenith', lw=0.8, alpha=0.3)
    index = np.argmin(np.abs(wvl_nad-wvl0))
    ax1.scatter(tmhr, spectra_flux_nad[:, index], edgecolor='magenta', marker='^', s=80, facecolor='none', label='CU SSFR Nadir', lw=0.8, alpha=0.3)

    f_nasa0 = readsav('test_20180221_nasa.out')
    index = np.argmin(np.abs(f_nasa0.zenlambda-wvl0))
    ax1.scatter(f_nasa0.tmhrs, f_nasa0.zspectra[:, index], edgecolor='blue', marker='v', s=80, facecolor='none', label='NASA SSFR Zenith', lw=0.8, alpha=0.3)
    index = np.argmin(np.abs(f_nasa0.nadlambda-wvl0))
    ax1.scatter(f_nasa0.tmhrs, f_nasa0.nspectra[:, index], edgecolor='cyan', marker='^', s=80, facecolor='none', label='NASA SSFR Nadir', lw=0.8, alpha=0.3)

    f_nasa = readsav('test_20180221_nasa_corr.out')
    index = np.argmin(np.abs(f_nasa.zenlambda-wvl0))
    ax1.scatter(f_nasa0.tmhrs, f_nasa.zenspectra[:, index], edgecolor='green', marker='v', s=80, facecolor='none', label='NASA SSFR Zenith (CORR)', lw=0.8, alpha=0.3)
    index = np.argmin(np.abs(f_nasa.nadlambda-wvl0))
    ax1.scatter(f_nasa0.tmhrs, f_nasa.nadspectra[:, index], edgecolor='lightgreen', marker='^', s=80, facecolor='none', label='NASA SSFR Nadir (CORR)', lw=0.8, alpha=0.3)

    ax1.set_title('SSFR at %dnm' % wvl0)
    ax1.set_xlabel('Time [hour]')
    ax1.set_ylabel('Flux [$\mathrm{W m^{-2} nm^{-1}}$]')

    # ax1.set_ylim((0.8, 1.2))

    plt.show()
    exit()
    # ---------------------------------------------------------------------







def PLOT_SPECTRA(tmhr_range):

    f = h5py.File('20180228_CU_20180221.h5', 'r')
    spectra_flux_zen = f['spectra_flux_zen'][...]
    spectra_flux_nad = f['spectra_flux_nad'][...]
    wvl_zen          = f['wvl_zen'][...]
    wvl_nad          = f['wvl_nad'][...]
    shutter          = f['shutter'][...]
    tmhr             = f['tmhr'][...]
    f.close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)

    logic = (tmhr>=tmhr_range[0])&(tmhr<=tmhr_range[1])
    for i in range(logic.sum()):
        ax1.scatter(wvl_zen, spectra_flux_zen[logic, :][i, :], c='red', s=1, label='CU SSFR Zenith', alpha=0.3)
    for i in range(logic.sum()):
        ax1.scatter(wvl_nad, spectra_flux_nad[logic, :][i, :], c='magenta', s=1, label='CU SSFR Nadir', alpha=0.3)

    f_nasa0 = readsav('test_20180221_nasa.out')
    logic = (f_nasa0.tmhrs>=tmhr_range[0])&(f_nasa0.tmhrs<=tmhr_range[1])
    for i in range(logic.sum()):
        ax1.scatter(f_nasa0.zenlambda, f_nasa0.zspectra[logic, :][i, :], c='blue', s=1, label='NASA SSFR Zenith', alpha=0.3)
    for i in range(logic.sum()):
        ax1.scatter(f_nasa0.nadlambda, f_nasa0.nspectra[logic, :][i, :], c='cyan', s=1, label='NASA SSFR Nadir', alpha=0.3)

    f_nasa = readsav('test_20180221_nasa_corr.out')

    logic = (f_nasa0.tmhrs>=tmhr_range[0])&(f_nasa0.tmhrs<=tmhr_range[1])
    for i in range(logic.sum()):
        ax1.scatter(f_nasa0.zenlambda, f_nasa.zenspectra[logic, :][i, :], c='green', s=1, label='NASA SSFR Zenith (CORR)', alpha=0.3)
    for i in range(logic.sum()):
        ax1.scatter(f_nasa0.nadlambda, f_nasa.nadspectra[logic, :][i, :], c='lightgreen', s=1, label='NASA SSFR Nadir (CORR)', alpha=0.3)

    ax1.set_title('SSFR from %.4f to %.4f' % (tmhr_range[0], tmhr_range[1]))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Flux [$\mathrm{W m^{-2} nm^{-1}}$]')

    ax1.set_ylim((0, 1.3))

    plt.show()
    exit()
    # ---------------------------------------------------------------------





def PLOT_SPECTRA_RATIO(tmhr_range):

    # f = h5py.File('test_20180221_cu.h5', 'r')
    f = h5py.File('20180228_CU_20180221.h5', 'r')
    spectra_flux_zen = f['spectra_flux_zen'][...]
    spectra_flux_nad = f['spectra_flux_nad'][...]
    wvl_zen          = f['wvl_zen'][...]
    wvl_nad          = f['wvl_nad'][...]
    shutter          = f['shutter'][...]
    tmhr             = f['tmhr'][...]
    f.close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)

    logic = (tmhr>=tmhr_range[0])&(tmhr<=tmhr_range[1])
    for i in range(logic.sum()):
        spectra_flux_zen0 = spectra_flux_zen[logic, :][i, :]
        spectra_flux_nad0 = np.interp(wvl_zen, wvl_nad, spectra_flux_nad[logic, :][i, :])
        ax1.scatter(wvl_zen, spectra_flux_nad0/spectra_flux_zen0, c='red', s=1, label='CU SSFR', alpha=0.3)

    f_nasa0 = readsav('test_20180221_nasa.out')
    logic = (f_nasa0.tmhrs>=tmhr_range[0])&(f_nasa0.tmhrs<=tmhr_range[1])
    for i in range(logic.sum()):
        spectra_flux_zen0 = f_nasa0.zspectra[logic, :][i, :]
        spectra_flux_nad0 = np.interp(f_nasa0.zenlambda, f_nasa0.nadlambda,  f_nasa0.nspectra[logic, :][i, :])
        ax1.scatter(f_nasa0.zenlambda, spectra_flux_nad0/spectra_flux_zen0, c='blue', s=1, label='NASA SSFR', alpha=0.3)

    f_nasa = readsav('test_20180221_nasa_corr.out')

    logic = (f_nasa0.tmhrs>=tmhr_range[0])&(f_nasa0.tmhrs<=tmhr_range[1])
    for i in range(logic.sum()):
        spectra_flux_zen0 = f_nasa.zenspectra[logic, :][i, :]
        spectra_flux_nad0 = np.interp(f_nasa.zenlambda, f_nasa.nadlambda,  f_nasa.nadspectra[logic, :][i, :])
        ax1.scatter(f_nasa.zenlambda, spectra_flux_nad0/spectra_flux_zen0, c='green', s=1, label='NASA SSFR (CORR)', alpha=0.3)

    ax1.set_title('SSFR from %.4f to %.4f' % (tmhr_range[0], tmhr_range[1]))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Ratio [Nadir/Zenith]')

    ax1.set_ylim((0.8, 1.2))
    plt.savefig('ssfr_ratio.png')

    plt.show()
    exit()
    # ---------------------------------------------------------------------




if __name__ == '__main__':

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator
    from matplotlib import rcParams
    import cartopy.crs as ccrs

    # fdir_primary    = '/Users/hoch4240/Chen/work/00_reuse/SSFR-util/data/ssfr6/20171106/1324'
    # fdir_transfer   = '/Users/hoch4240/Chen/work/00_reuse/SSFR-util/data/ssfr6/20171106/150C'
    # fdir_secondary  = '/Users/hoch4240/Chen/work/00_reuse/SSFR-util/data/ssfr6/20180221/150C'
    # nasa_cal = CALIBRATION_NASA_SSFR(fdir_primary, fdir_transfer, fdir_secondary)




    # fnames = sorted(glob.glob('/Users/hoch4240/Chen/work/00_reuse/SSFR-util/data/20180228/data/*.SKS'))
    # f_sks = READ_SKS(fnames)
    # exit()





    PLOT_TIME_SERIES(500.0)
    # PLOT_TIME_SERIES(1600.0)
    # PLOT_SPECTRA([19.2, 19.3])
    # PLOT_SPECTRA_RATIO([19.2, 19.3])


    exit()
