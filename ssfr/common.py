import os

__all__ = ['fdir_data']

fdir_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

input_list = {
        'which_ssfr': 'lasp',
          'which_lc': 'zenith',
         'wvl_joint': 950.0,
         'wvl_range': [350.0, 2200.0],
          'int_time': {'si':60, 'in':300},
        }

serial_number = {
        'lasp|ssfr-a|zenith|silicon': 'n/a',
        'lasp|ssfr-a|zenith|ingaas' : '044832', # found broken and replaced with Zenith InGaAs spectrometer from NASA Ames's SSFR
        'lasp|ssfr-a|nadir|silicon' : 'n/a',
        'lasp|ssfr-a|nadir|ingaas'  : 'n/a',
        'lasp|ssfr-b|zenith|silicon': 'n/a',
        'lasp|ssfr-b|zenith|ingaas' : 'n/a',
        'lasp|ssfr-b|nadir|silicon' : 'n/a',
        'lasp|ssfr-b|nadir|ingaas'  : 'n/a',
        'nasa|ssfr|zenith|silicon': '033161',
        'nasa|ssfr|zenith|ingaas' : '044832',
        'nasa|ssfr|nadir|silicon' : '045924',
        'nasa|ssfr|nadir|ingaas'  : '044829',
        }
