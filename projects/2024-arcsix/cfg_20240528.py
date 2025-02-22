import datetime

common = {
        'date': datetime.datetime(2024, 5, 28),
        'mission': 'ARCSIX-1',
        'platform': 'P3B',
        'operator': 'Vikas Nataraja',
        'tag': 'Science Flight #1',
        'comments': 'Clear-sky spiral',
        }

ssfr1 = {
        'fdir_data': '',
        'fname_cal': '',

        'which_ssfr': 'lasp|ssfr-a',

        # zenith wavelength setting
        'wvl_s_zen': 350.0,  # beginning/first wavelength [nm] of the selected wavelength range
        'wvl_e_zen': 2000.0, # ending/last wavelength [nm] of the selected wavelength range
        'wvl_j_zen': 950.0,  # joinder wavelength within the overlapping wavelength coverage between Silicon and InGaAs spectrometers

        # nadir wavelength setting
        'wvl_s_nad': 350.0,  # beginning/first wavelength [nm] of the selected wavelength range
        'wvl_e_nad': 2000.0, # ending/last wavelength [nm] of the selected wavelength range
        'wvl_j_nad': 950.0,  # joinder wavelength within the overlapping wavelength coverage between Silicon and InGaAs spectrometers

        # time offset [seconds]
        'time_offset': 0.0,

        # number of data points to be excluded at the beginning and end of a dark cycle (due to slow shutter closing/opening glitch)
        'dark_extend': 1,

        # number of data points to be excluded at the beginning and end of a light cycle (due to slow shutter closing/opening glitch)
        'light_extend': 1,

        # dark correction mode: `interp`, linear interpolation using two adjacent dark cycles
        #   also available in `mean`, which uses the average to represent darks
        #   generally, `interp` is preferred
        'dark_corr_mode': 'interp',

        # minimum number of darks to achieve valid dark correction
        'dark_threshold': 5,

        # minimum number of lights to achieve valid dark correction
        'light_threshold': 10,
        }

alp = {
        'time_offset': -17.19,

        'ang_pit_offset': 0.0,

        'ang_rol_offset': 0.0,
        }
