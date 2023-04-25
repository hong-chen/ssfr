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
