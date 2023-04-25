import os
import sys
import datetime
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav

import ssfr

from ssfr.util import nasa_ssfr, lasp_ssfr, get_nasa_ssfr_wavelength, cal_weighted_flux
from ssfr.corr import dark_corr


__all__ = ['cal_rad_resp', 'cdata_rad_resp', 'load_rad_resp_h5']



def cal_rad_resp(
        fnames,
        resp=None,
        which_ssfr='nasa',
        which_lc='zenith',
        pri_lamp_tag='f-1324',
        int_time={'si':60, 'in':300}
        ):

    which_lc = which_lc.lower()
    if which_lc == 'zenith':
        index_si = 0
        index_in = 1
    if which_lc == 'nadir':
        index_si = 2
        index_in = 3

    if resp is None:

        # read in calibrated lamp data and interpolated at SSFR wavelengths
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        fnameLamp = '%s/%s.dat' % (ssfr.common.fdir_data, pri_lamp_tag)
        if not os.path.exists(fnameLamp):
            sys.exit('Error [cal_rad_resp]: Cannot locate lamp standards for \'%s\'.' % pri_lamp_tag.title())

        data      = np.loadtxt(fnameLamp)
        data_wvl  = data[:, 0]
        if pri_lamp_tag == 'f-506c':
            data_flux = data[:, 1]*0.01
        else:
            data_flux = data[:, 1]*10000.0

        wvls   = get_nasa_ssfr_wavelength()
        wvl_si = wvls['%s_si' % which_lc]
        wvl_in = wvls['%s_in' % which_lc]

        lampStd_si = np.zeros_like(wvl_si)
        for i in range(lampStd_si.size):
            lampStd_si[i] = cal_weighted_flux(wvl_si[i], data_wvl, data_flux, slit_func_file='%s/vis_0.1nm_s.dat' % ssfr.common.fdir_data)

        lampStd_in = np.zeros_like(wvl_in)
        for i in range(lampStd_in.size):
            lampStd_in[i] = cal_weighted_flux(wvl_in[i], data_wvl, data_flux, slit_func_file='%s/nir_0.1nm_s.dat' % ssfr.common.fdir_data)

        resp = {'si':lampStd_si,
                'in':lampStd_in}
        # ---------------------------------------------------------------------------
        # so far we have (W m^-2 nm^-1 as a function of wavelength)


    ssfr_l = nasa_ssfr([fnames['cal']])
    ssfr_d = nasa_ssfr([fnames['dark']])

    # Silicon
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    counts_l  = ssfr_l.spectra[:, :, index_si]
    counts_d  = ssfr_d.spectra[:, :, index_si]

    logic_l   = (np.abs(ssfr_l.int_time[:, index_si]-int_time['si'])<0.00001)
    spectra_l = np.mean(counts_l[logic_l, :], axis=0)

    logic_d   = (np.abs(ssfr_d.int_time[:, index_si]-int_time['si'])<0.00001)
    spectra_d = np.mean(counts_d[logic_d, :], axis=0)

    spectra   = spectra_l - spectra_d
    spectra[spectra<=0.0] = np.nan
    rad_resp_si = spectra / int_time['si'] / resp['si']
    # ---------------------------------------------------------------------------

    # InGaAs
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    counts_l  = ssfr_l.spectra[:, :, index_in]
    counts_d  = ssfr_d.spectra[:, :, index_in]

    logic_l   = (np.abs(ssfr_l.int_time[:, index_in]-int_time['in'])<0.00001)
    spectra_l = np.mean(counts_l[logic_l, :], axis=0)

    logic_d   = (np.abs(ssfr_d.int_time[:, index_in]-int_time['in'])<0.00001)
    spectra_d = np.mean(counts_d[logic_d, :], axis=0)

    spectra   = spectra_l - spectra_d
    spectra[spectra<=0.0] = np.nan
    rad_resp_in = spectra / int_time['in'] / resp['in']
    # ---------------------------------------------------------------------------

    rad_resp = {'si':rad_resp_si,
                'in':rad_resp_in}

    return rad_resp



def cdata_rad_resp(
        fnames_pri=None,
        fnames_tra=None,
        fnames_sec=None,
        filename_tag=None,
        pri_lamp_tag='f-1324',
        which_lc='zenith',
        which_ssfr='nasa',
        wvl_joint=950.0,
        wvl_start=350.0,
        wvl_end=2200.0,
        int_time={'si':60, 'in':300}
        ):

    which_lc = which_lc.lower()

    if fnames_pri is not None:
        pri_resp = cal_rad_resp(fnames_pri, resp=None, which_lc=which_lc, int_time=int_time, pri_lamp_tag=pri_lamp_tag)
    else:
        sys.exit('Error [cdata_rad_resp]: Cannot proceed without primary calibration files.')

    if fnames_tra is not None:
        transfer = cal_rad_resp(fnames_tra, resp=pri_resp, which_lc=which_lc, int_time=int_time, pri_lamp_tag=pri_lamp_tag)
    else:
        sys.exit('Error [cdata_rad_resp]: Cannot proceed without transfer calibration files.')

    if fnames_sec is not None:
        sec_resp = cal_rad_resp(fnames_sec, resp=transfer, which_lc=which_lc, int_time=int_time, pri_lamp_tag=pri_lamp_tag)
    else:
        print('Warning [cdata_rad_resp]: Secondary/field calibration files are not available, use transfer calibration files for secondary/field calibration ...')
        sec_resp = cal_rad_resp(fnames_tra, transfer, which_lc=which_lc, int_time=int_time, pri_lamp_tag=pri_lamp_tag)

    wvls = get_nasa_ssfr_wavelength()

    logic_si = (wvls['%s_si' % which_lc] >= wvl_start) & (wvls['%s_si' % which_lc] <= wvl_joint)
    logic_in = (wvls['%s_in' % which_lc] >  wvl_joint)  & (wvls['%s_in' % which_lc] <= wvl_end)

    wvl_data      = np.concatenate((wvls['%s_si' % which_lc][logic_si], wvls['%s_in' % which_lc][logic_in]))
    pri_resp_data = np.hstack((pri_resp['si'][logic_si], pri_resp['in'][logic_in]))
    transfer_data = np.hstack((transfer['si'][logic_si], transfer['in'][logic_in]))
    sec_resp_data = np.hstack((sec_resp['si'][logic_si], sec_resp['in'][logic_in]))

    indices_sort = np.argsort(wvl_data)
    wvl      = wvl_data[indices_sort]
    pri_resp = pri_resp_data[indices_sort]
    transfer = transfer_data[indices_sort]
    sec_resp = sec_resp_data[indices_sort]

    if filename_tag is not None:
        fname_out = '%s_rad-resp_s%3.3di%3.3d.h5' % (filename_tag, int_time['si'], int_time['in'])
    else:
        fname_out = 'rad-resp_s%3.3di%3.3d.h5' % (int_time['si'], int_time['in'])

    f = h5py.File(fname_out, 'w')
    f['wvl']       = wvl
    f['pri_resp']  = pri_resp
    f['transfer']  = transfer
    f['sec_resp']  = sec_resp
    f.close()

    return fname_out



def load_rad_resp_h5(fname):

    resp = {}
    f = h5py.File(fname, 'r')
    for key in f.keys():
        resp[key] = f[key][...]
    f.close()

    return resp



if __name__ == '__main__':

    pass
