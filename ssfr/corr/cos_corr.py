import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav

import ssfr



__all__ = ['cos_corr']



def cos_corr(fnames,
             angles,
             diff_ratio=None):

    """
    Calculate the correction factors based on the spectral angular response.
    The `dc` factor is first calculated based on the `pitch`, `roll`, `pitch_motor`, `roll_motor`, `heading`, `pitch_offset`, `roll_offset`.
    Based on the `dc` factor, the angular response is selected and then the spectral response is calculated from the polynomial coefficients.

    Input:
        `fnames`: Python dictionary, angular calibration file processed through `ssfr.cal.cdata_cos_resp`
            ['zenith']
            ['nadir']

        `angles`: Python dictionary
            ['pitch']
            ['roll']
            ['heading']
            ['pitch_motor']
            ['roll_motor']
            ['pitch_offset']
            ['roll_offset']
            ['solar_zenith']
            ['solar_azimuth']

        `diff_ratio`: None or numpy array
            if is a numpy array, it should be a 2D array with dimension of [tmhr, wvl]

    Output:
        `corr_factors`: Python dictionary
            ['zenith']
            ['nadir']
    """

    corr_factors = angles.copy()

    iza, iaa = ssfr.prh2za(angles['pitch']-angles['pitch_motor']-angles['pitch_offset'], angles['roll']-angles['roll_motor']-angles['roll_offset'], angles['heading'])
    dc       = ssfr.muslope(angles['solar_zenith'], angles['solar_azimuth'], iza, iaa)
    corr_factors['sensor_zenith']  = iza
    corr_factors['sensor_azimuth'] = iza
    corr_factors['cosine_zenith']  = dc


    indices  = np.int_(np.round(dc, decimals=3)*1000.0)
    indices[indices>1000] = 1000
    indices[indices<0]    = 0


    # zenith
    # ==============================================================================================
    # if `diff_ratio` is provided,
    #     in addition to `factors_dir`, calculate `factors_dif`
    #     factors = factors_dir*(1-diff_ratio) + factors_dif*diff_ratio
    # if `diff_ratio` is not provided,
    #     factors = factors_dir
    cos_resp     = ssfr.cal.load_cos_resp_h5(fnames['zenith'])
    wvl          = cos_resp['wvl']
    factors_dir  = np.zeros((dc.size, wvl.size), dtype=np.float64)
    for i, index in enumerate(indices):
        f = np.poly1d(cos_resp['poly_coef'][index, :])
        resp = f(wvl)
        resp[resp<0.0000001] = 0.0000001
        factors_dir[i, :] = cos_resp['mu'][index] / resp

    if diff_ratio is None:
        corr_factors['zenith'] = factors_dir
    else:
        factors_dif  = np.zeros((dc.size, wvl.size), dtype=np.float64)
        for i in range(wvl.size):
            factors_dif[:, i] = 0.5 / cos_resp['cos_resp_int'][i]
        corr_factors['zenith'] = factors_dif*diff_ratio + factors_dir*(1.0-diff_ratio)
    # ==============================================================================================

    # nadir
    # ==============================================================================================
    # use `cos_resp_int` for diffuse correction
    cos_resp     = ssfr.cal.load_cos_resp_h5(fnames['nadir'])
    wvl          = cos_resp['wvl']
    factors_dif  = np.zeros((dc.size, wvl.size), dtype=np.float64)
    for i in range(wvl.size):
        factors_dif[:, i] = 0.5 / cos_resp['cos_resp_int'][i]
    corr_factors['nadir'] = factors_dif
    # ==============================================================================================

    return corr_factors



if __name__ == '__main__':

    pass
