import os

__all__ = ['fdir_data']

fdir_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

karg = {
        'verbose': True,
        }

ssfr_default = {
        'which_ssfr': 'lasp',
          'which_lc': 'zenith',
         'wvl_joint': 950.0,
         'wvl_range': [350.0, 2200.0],
          'int_time': {'si':60, 'in':300},
        }

serial_number = {
        'lasp|ssfr-a|zen|si': 'n/a',
        'lasp|ssfr-a|zen|in': '044832', # found malfunction and was replaced with Zenith InGaAs spectrometer from NASA Ames's SSFR
        'lasp|ssfr-a|nad|si': 'n/a',
        'lasp|ssfr-a|nad|in': 'n/a',
        'lasp|ssfr-b|zen|si': 'n/a',
        'lasp|ssfr-b|zen|in': 'n/a',
        'lasp|ssfr-b|nad|si': 'n/a',
        'lasp|ssfr-b|nad|in': 'n/a',
        'nasa|ssfr-6|zen|si': '033161',
        'nasa|ssfr-6|zen|in': 'n/a',
        'nasa|ssfr-6|nad|si': '045924',
        'nasa|ssfr-6|nad|in': '044829',
        }
