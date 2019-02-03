import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav



class CALIBRATION_CU_SSFR:

    def __init__(self, config):

        self.config = config

        self.CAL_WAVELENGTH()
        self.CAL_PRIMARY_RESPONSE(self.config)
        self.CAL_TRANSFER(self.config)
        self.CAL_SECONDARY_RESPONSE(config)
        # self.CAL_ANGULAR_RESPONSE(config)

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

    def CAL_PRIMARY_RESPONSE(self, config, lampTag='f-1324', fdirLamp='aux'):

        # read in calibrated lamp data and interpolated at SSFR wavelengths
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.fnameLamp = '%s/%s.dat' % (os.path.abspath(fdirLamp), lampTag)
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



        # for zenith
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Silicon
        print('Primary [Zenith Silicon]: processing primary response...')
        self.primary_response_zen_si = {}
        iSen = 0
        try:
            ssfr_l = CU_SSFR([config['fname_primary_zen_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_primary_zen_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_primary_zen_si'] = intTimes_l
            else:
                exit('Primary [Zenith Silicon]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_primary_zen_si']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.primary_response_zen_si[intTime] = spectra / intTime / lampStd_zen_si
        except:
            print('Primary [Zenith Silicon]: Cannot read calibration files.')
            self.primary_response_zen_si[-1] = np.repeat(np.nan, self.chanNum)


        # InGaAs
        self.primary_response_zen_in = {}
        iSen = 1
        try:
            ssfr_l = CU_SSFR([config['fname_primary_zen_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_primary_zen_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_primary_zen_in'] = intTimes_l
            else:
                exit('Primary [Zenith InGaAs]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_primary_zen_in']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.primary_response_zen_in[intTime] = spectra / intTime / lampStd_zen_in
        except:
            print('Primary [Zenith InGaAs]: Cannot read calibration files.')
            self.primary_response_zen_in[-1] = np.repeat(np.nan, self.chanNum)
        # ---------------------------------------------------------------------------



        # for nadir
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Silicon
        self.primary_response_nad_si = {}
        iSen = 2
        try:
            ssfr_l = CU_SSFR([config['fname_primary_nad_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_primary_nad_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_primary_nad_si'] = intTimes_l
            else:
                exit('Primary [Nadir Silicon]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_primary_nad_si']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.primary_response_nad_si[intTime] = spectra / intTime / lampStd_nad_si
        except:
            print('Primary [Nadir Silicon]: Cannot read calibration files.')
            self.primary_response_nad_si[-1] = np.repeat(np.nan, self.chanNum)


        # InGaAs
        self.primary_response_nad_in = {}
        iSen = 3
        try:
            ssfr_l = CU_SSFR([config['fname_primary_nad_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_primary_nad_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_primary_nad_in'] = intTimes_l
            else:
                exit('Primary [Nadir InGaAs]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_primary_nad_in']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.primary_response_nad_in[intTime] = spectra / intTime / lampStd_nad_in
        except:
            print('Primary [Nadir InGaAs]: Cannot read calibration files.')
            self.primary_response_nad_in[-1] = np.repeat(np.nan, self.chanNum)
        # ---------------------------------------------------------------------------

    def CAL_TRANSFER(self, config):

        # for zenith
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Silicon
        self.field_lamp_zen_si = {}
        iSen = 0
        try:
            ssfr_l = CU_SSFR([config['fname_transfer_zen_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_transfer_zen_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_transfer_zen_si'] = intTimes_l
            else:
                exit('Transfer [Zenith Silicon]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_transfer_zen_si']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.field_lamp_zen_si[intTime] = spectra / intTime / self.primary_response_zen_si[intTime]
        except:
            print('Transfer [Zenith Silicon]: Cannot read calibration files.')
            self.field_lamp_zen_si[-1] = np.repeat(np.nan, self.chanNum)


        # InGaAs
        self.field_lamp_zen_in = {}
        iSen = 1
        try:
            ssfr_l = CU_SSFR([config['fname_transfer_zen_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_transfer_zen_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_transfer_zen_in'] = intTimes_l
            else:
                exit('Transfer [Zenith InGaAs]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_transfer_zen_in']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.field_lamp_zen_in[intTime] = spectra / intTime / self.primary_response_zen_in[intTime]
        except:
            print('Transfer [Zenith InGaAs]: Cannot read calibration files.')
            self.field_lamp_zen_in[-1] = np.repeat(np.nan, self.chanNum)
        # ---------------------------------------------------------------------------



        # for nadir
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Silicon
        self.field_lamp_nad_si = {}
        iSen = 2
        try:
            ssfr_l = CU_SSFR([config['fname_transfer_nad_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_transfer_nad_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_transfer_nad_si'] = intTimes_l
            else:
                exit('Transfer [Nadir Silicon]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_transfer_nad_si']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.field_lamp_nad_si[intTime] = spectra / intTime / self.primary_response_nad_si[intTime]
        except:
            print('Transfer [Nadir Silicon]: Cannot read calibration files.')
            self.field_lamp_nad_si[-1] = np.repeat(np.nan, self.chanNum)


        # InGaAs
        self.field_lamp_nad_in = {}
        iSen = 3
        try:
            ssfr_l = CU_SSFR([config['fname_transfer_nad_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_transfer_nad_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_transfer_nad_in'] = intTimes_l
            else:
                exit('Transfer [Nadir InGaAs]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_transfer_nad_in']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.field_lamp_nad_in[intTime] = spectra / intTime / self.primary_response_nad_in[intTime]
        except:
            print('Transfer [Nadir InGaAs]: Cannot read calibration files.')
            self.field_lamp_nad_in[-1] = np.repeat(np.nan, self.chanNum)
        # ---------------------------------------------------------------------------

    def CAL_SECONDARY_RESPONSE(self, config):

        # for zenith
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Silicon
        self.secondary_response_zen_si = {}
        iSen = 0
        try:
            ssfr_l = CU_SSFR([config['fname_secondary_zen_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_secondary_zen_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_secondary_zen_si'] = intTimes_l
            else:
                exit('Secondary [Zenith Silicon]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_secondary_zen_si']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.secondary_response_zen_si[intTime] = spectra / intTime / self.field_lamp_zen_si[intTime]
        except:
            print('Secondary [Zenith Silicon]: Cannot read calibration files.')
            self.secondary_response_zen_si[-1] = np.repeat(np.nan, self.chanNum)


        # InGaAs
        self.secondary_response_zen_in = {}
        iSen = 1
        try:
            ssfr_l = CU_SSFR([config['fname_secondary_zen_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_secondary_zen_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_secondary_zen_in'] = intTimes_l
            else:
                exit('Secondary [Zenith InGaAs]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_secondary_zen_in']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.secondary_response_zen_in[intTime] = spectra / intTime / self.field_lamp_zen_in[intTime]
        except:
            print('Secondary [Zenith InGaAs]: Cannot read calibration files.')
            self.secondary_response_zen_in[-1] = np.repeat(np.nan, self.chanNum)
        # ---------------------------------------------------------------------------



        # for nadir
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Silicon
        self.secondary_response_nad_si = {}
        iSen = 2
        try:
            ssfr_l = CU_SSFR([config['fname_secondary_nad_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_secondary_nad_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_secondary_nad_si'] = intTimes_l
            else:
                exit('Secondary [Nadir Silicon]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_secondary_nad_si']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.secondary_response_nad_si[intTime] = spectra / intTime / self.field_lamp_nad_si[intTime]
        except:
            print('Secondary [Nadir Silicon]: Cannot read calibration files.')
            self.secondary_response_nad_si[-1] = np.repeat(np.nan, self.chanNum)


        # InGaAs
        self.secondary_response_nad_in = {}
        iSen = 3
        try:
            ssfr_l = CU_SSFR([config['fname_secondary_nad_cal']] , dark_corr_mode='mean')
            ssfr_d = CU_SSFR([config['fname_secondary_nad_dark']], dark_corr_mode='mean')

            intTimes_l = list(np.unique(ssfr_l.int_time[:, iSen]))
            intTimes_d = list(np.unique(ssfr_d.int_time[:, iSen]))

            if intTimes_l == intTimes_d:
                config['int_time_secondary_nad_in'] = intTimes_l
            else:
                exit('Secondary [Nadir InGaAs]: dark and light integration times doesn\'t match.')

            for intTime in config['int_time_secondary_nad_in']:

                spectra_l = np.mean(ssfr_l.spectra_dark_corr[np.abs(ssfr_l.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)
                spectra_d = np.mean(ssfr_d.spectra_dark_corr[np.abs(ssfr_d.int_time[:, iSen]-intTime)<0.00001, :, iSen], axis=0)

                spectra = spectra_l - spectra_d
                spectra[spectra<=0.0] = np.nan
                self.secondary_response_nad_in[intTime] = spectra / intTime / self.field_lamp_nad_in[intTime]
        except:
            print('Secondary [Nadir InGaAs]: Cannot read calibration files.')
            self.secondary_response_nad_in[-1] = np.repeat(np.nan, self.chanNum)
        # ---------------------------------------------------------------------------

    def CAL_ANGULAR_RESPONSE(self, config):

        print('under development')




if __name__ == '__main__':
    pass
