import datetime
import ssfr

# parameters that need frequent change
#╭────────────────────────────────────────────────────────────────────────────╮#
date = datetime.datetime(2024, 5, 28)
operator = 'Vikas Nataraja'
mission = 'arcsix'
platform = 'p3'
comments = 'Clear-sky spiral'

hsk_tag  = 'hsk'

hsr1_tag = 'hsr1-a'

ssfr_tag = 'ssfr-a'

ssrr_tag = 'ssfr-b'

alp_tag  = 'alp'

fdir_cal = 'data/%s/cal' % mission
fdir_out = 'data/%s/processed' % mission
#╰────────────────────────────────────────────────────────────────────────────╯#

common = {
        'date': date,
        'date_s': date.strftime('%Y%m%d'),
        'date_s_': date.strftime('%Y-%m-%d'),
        'mission': mission.lower(),
        'platform': platform.lower(),
        'operator': operator,
        'comments': comments,
        }

hsk = {
        'tag': hsk_tag.lower(),
        'fdir': 'data/arcsix/2024/p3/aux/hsk',
        }

ssfr = {
        'tag': ssfr_tag.lower(),

        'fdir_data': 'data/arcsix/2024/p3/20240528_sci-flt-01/raw/ssfr-a',

        'fname_rad_cal': 'data/arcsix/cal/rad-cal/2024-03-29_lamp-506|2024-03-29_lamp-150e_spec-zen|2024-04-01|dset1|rad-resp|lasp|ssfr-a|zen|si-120|in-350.h5',

        'which_ssfr': 'lasp|%s' % ssfr_tag.lower(),

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

ssrr = {
        'tag': ssrr_tag.lower(),

        'fdir_data': 'data/arcsix/2024/p3/20240528_sci-flt-01/raw/ssfr-b',

        'fname_rad_cal': 'data/arcsix/cal/rad-cal/2024-03-29_lamp-506|2024-03-29_lamp-150e_spec-zen|2024-04-01|dset1|rad-resp|lasp|ssfr-a|zen|si-120|in-350.h5',

        'which_ssfr': 'lasp|%s' % ssrr_tag.lower(),

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

hsr1 = {
        'tag': hsr1_tag.lower(),
        }

alp = {
        'tag': 'alp',

        'fdir_data': 'data/%s/%4.4d/%s/20240528_sci-flt-01/raw/alp' % (common['mission'], date.year, common['platform']),

        'time_offset': -17.19,

        'ang_pit_offset': 0.0,

        'ang_rol_offset': 0.0,
        }
