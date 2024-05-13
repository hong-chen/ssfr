import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav

import ssfr



__all__ = ['att_corr']



def att_corr(fnames,
             angles,
             diff_ratio=None):

    """
    Calculate the correction factors based on the spectral angular response.
    The `dc` factor is first calculated based on the `ang_pit`, `ang_rol`, `ang_pit_m`, `ang_rol_m`, `ang_hed`, `ang_pit_offset`, `ang_rol_offset`.
    Based on the `dc` factor, the angular response is selected and then the spectral response is calculated from the polynomial coefficients.

    Input:
        `fnames`: Python dictionary, angular calibration file processed through `ssfr.cal.cdata_ang_resp`
            ['zen']
            ['nad']

        `angles`: Python dictionary
            ['ang_pit']
            ['ang_rol']
            ['ang_hed']
            ['ang_pit_m']
            ['ang_rol_m']
            ['ang_pit_offset']
            ['ang_rol_offset']
            ['sza']
            ['saa']

        `diff_ratio`: None or numpy array
            if is a numpy array, it should be a 2D array with dimension of [tmhr, wvl]

    Output:
        `corr_factors`: Python dictionary
            ['zen']
            ['nad']
    """

    corr_factors = angles.copy()

    iza, iaa = ssfr.util.prh2za(angles['ang_pit']-angles['ang_pit_m']-angles['ang_pit_offset'], angles['ang_rol']-angles['ang_rol_m']-angles['ang_rol_offset'], angles['ang_hed'])
    dc       = ssfr.util.muslope(angles['sza'], angles['saa'], iza, iaa)
    corr_factors['iza'] = iza # sensor zenith
    corr_factors['iaa'] = iza # sensor azimuth
    corr_factors['dc']  = dc  # cosine of relative zenith (zenith between sun and sensor)


    indices  = np.int_(np.round(dc, decimals=3)*1000.0)
    indices[indices>1000] = 1000
    indices[indices<0]    = 0

    # zenith
    #/----------------------------------------------------------------------------\#
    # zenith attitude correction contains two parts:
    #    1) correction for diffuse radiation (e.g., rayleigh scattering by atmospheric gases) -
    #       this will use angularly integrated angular response
    #    2) correction for direct radiation -
    #       this will use angular response at given angle (angular response is measured in the lab)
    #       *Notes: attitude correction is also done in the spectral space via polynomial fitting
    # if `diff_ratio` is provided,
    #     in addition to `factors_dir`, calculate `factors_dif`
    #     factors = factors_dir*(1-diff_ratio) + factors_dif*diff_ratio
    # if `diff_ratio` is not provided,
    #     factors = factors_dir
    ang_resp     = ssfr.util.load_h5(fnames['zen'])
    wvl          = ang_resp['wvl']
    factors_dir  = np.zeros((dc.size, wvl.size), dtype=np.float64)
    for i, index in enumerate(indices):
        f = np.poly1d(ang_resp['poly_coef'][index, :])
        resp = f(wvl)
        resp[resp<1e-8] = 1e-8
        factors_dir[i, :] = ang_resp['mu'][index] / resp

    if diff_ratio is None:
        corr_factors['zen'] = factors_dir
    else:
        factors_dif  = np.zeros((dc.size, wvl.size), dtype=np.float64)
        for i in range(wvl.size):
            factors_dif[:, i] = 0.5 / ang_resp['ang_resp_int'][i]
        corr_factors['zen'] = factors_dif*diff_ratio + factors_dir*(1.0-diff_ratio)
    #\----------------------------------------------------------------------------/#

    # nadir
    #/----------------------------------------------------------------------------\#
    # use `ang_resp_int` for diffuse correction
    ang_resp     = ssfr.util.load_h5(fnames['nad'])
    wvl          = ang_resp['wvl']
    factors_dif  = np.zeros((dc.size, wvl.size), dtype=np.float64)
    for i in range(wvl.size):
        factors_dif[:, i] = 0.5 / ang_resp['ang_resp_int'][i]
    corr_factors['nad'] = factors_dif
    #\----------------------------------------------------------------------------/#

    return corr_factors



if __name__ == '__main__':

    pass
