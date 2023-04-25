import os

__all__ = ['fdir_data']

fdir_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

input_list = {
        'which_ssfr': 'nasa',
        'wvl_joint': 750.0,
        }
