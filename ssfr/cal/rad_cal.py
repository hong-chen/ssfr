import os
import sys
import datetime
import warnings
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav


import ssfr



__all__ = [
        'cal_rad_resp',
        'cdata_rad_resp',
        ]



def cal_rad_resp(
        fnames,
        resp=None,
        which_ssfr='lasp|ssfr-a',
        which_lc='zen',
        which_lamp='f-1324',
        int_time={'si':80.0, 'in':250.0},
        ):


    # check SSFR spectrometer
    #/----------------------------------------------------------------------------\#
    which_ssfr = which_ssfr.lower()
    which_lab  = which_ssfr.split('|')[0]
    if which_lab == 'nasa':
        import ssfr.nasa_ssfr as ssfr_toolbox
    elif which_lab == 'lasp':
        import ssfr.lasp_ssfr as ssfr_toolbox
    else:
        msg = '\nError [cal_rad_resp]: <which_ssfr=> does not support <\'%s\'> (only supports <\'nasa|ssfr-6\'> or <\'lasp|ssfr-a\'> or <\'lasp|ssfr-b\'>).' % which_ssfr
        raise ValueError(msg)
    print(which_ssfr)
    print(which_lab)
    #\----------------------------------------------------------------------------/#


    # check light collector
    #/----------------------------------------------------------------------------\#
    which_lc = which_lc.lower()
    if which_lc in ['zenith', 'zen', 'z']:
        index_si = 0
        index_in = 1
    elif which_lc in ['nadir', 'nad', 'n']:
        index_si = 2
        index_in = 3
    else:
        msg = '\nError [cal_rad_resp]: <which_lc=> does not support <\'%s\'> (only supports <\'zenith, zen, z\'> or <\'nadir, nad, n\'>).' % which_lc
        raise ValueError(msg)
    print(which_lc)
    print(index_si, index_in)
    #\----------------------------------------------------------------------------/#


    # si/in tag
    #/----------------------------------------------------------------------------\#
    si_tag = '%s|si' % which_lc
    in_tag = '%s|in' % which_lc

    if si_tag not in int_time.keys():
        int_time[si_tag] = int_time.pop('si')

    if in_tag not in int_time.keys():
        int_time[in_tag] = int_time.pop('in')
    #\----------------------------------------------------------------------------/#


    # get radiometric response
    # by default (resp=None), this function will perform primary radiometric calibration
    #/----------------------------------------------------------------------------\#
    if resp is None:

        # check lamp
        #/--------------------------------------------------------------\#
        which_lamp = which_lamp.lower()

        if (which_lamp[:4] == 'f-50') or (which_lamp[-3:-1] == '50'):
            which_lamp = 'f-506c'
        elif (which_lamp[-4:] == '1324'):
            which_lamp = 'f-1324'

        print(which_lamp)
        #\--------------------------------------------------------------/#

        # read in calibrated lamp data and interpolated/integrated at SSFR wavelengths/slits
        #/--------------------------------------------------------------\#
        fname_lamp = '%s/lamp/%s.dat' % (ssfr.common.fdir_data, which_lamp)
        if not os.path.exists(fname_lamp):
            msg = '\nError [cal_rad_resp]: Cannot locate calibration file for lamp <%s>.' % which_lamp
            raise OSError(msg)

        print(fname_lamp)

        data      = np.loadtxt(fname_lamp)
        data_wvl  = data[:, 0]
        if which_lamp == 'f-506c':
            data_flux = data[:, 1]*0.01      # W m^-2 nm^-1
        else:
            data_flux = data[:, 1]*10000.0   # W m^-2 nm^-1

        # !!!!!!!!!! this will change !!!!!!!!!!!!!!
        # currently we are developing wavelength calibration,
        # the wavelength calculation process will be modified.
        #/--------------------------------------------------------------\#
        wvls = ssfr_toolbox.get_ssfr_wvl(which_ssfr)
        wvl_si = wvls[si_tag]
        wvl_in = wvls[in_tag]
        print(wvls.keys())
        #\--------------------------------------------------------------/#

        lamp_std_si = np.zeros_like(wvl_si)
        for i in range(lamp_std_si.size):
            lamp_std_si[i] = ssfr.util.cal_weighted_flux(wvl_si[i], data_wvl, data_flux, slit_func_file='%s/slit/vis_0.1nm_s.dat' % ssfr.common.fdir_data)

        lamp_std_in = np.zeros_like(wvl_in)
        for i in range(lamp_std_in.size):
            lamp_std_in[i] = ssfr.util.cal_weighted_flux(wvl_in[i], data_wvl, data_flux, slit_func_file='%s/slit/nir_0.1nm_s.dat' % ssfr.common.fdir_data)

        # lamp_std_si = np.interp(wvl_si, data_wvl, data_flux)
        # lamp_std_in = np.interp(wvl_in, data_wvl, data_flux)

        resp = {
                si_tag: lamp_std_si,
                in_tag: lamp_std_in
               }
    #\----------------------------------------------------------------------------/#


    # read raw data
    #/----------------------------------------------------------------------------\#
    ssfr_obj = ssfr_toolbox.read_ssfr(fnames, dark_corr_mode='interp', dark_extend=5, light_extend=5)

    spectra_si = None
    spectra_in = None
    for i in range(ssfr_obj.Ndset):

        dset_name = 'dset%d' % i
        data = getattr(ssfr_obj, dset_name)

        if abs(data['info']['int_time'][si_tag] - int_time[si_tag]) < 0.00001:
            spectra_si = np.nanmean(data['spectra_dark-corr'][:, :, index_si], axis=0)

        if abs(data['info']['int_time'][in_tag] - int_time[in_tag]) < 0.00001:
            spectra_in = np.nanmean(data['spectra_dark-corr'][:, :, index_in], axis=0)
    #\----------------------------------------------------------------------------/#


    # Silicon
    # some placeholder ideas:
    # if nan is detected (e.g., spectra_si smaller than 0.0), one can use
    # interpolation to fill in the nan values
    #/----------------------------------------------------------------------------\#
    if spectra_si is not None:
        spectra_si[spectra_si<=0.0] = np.nan
        rad_resp_si = spectra_si / int_time[si_tag] / resp[si_tag]
    else:
        rad_resp_si = None
    #\----------------------------------------------------------------------------/#


    # InGaAs
    #/----------------------------------------------------------------------------\#
    if spectra_in is not None:
        spectra_in[spectra_in<=0.0] = np.nan
        rad_resp_in = spectra_in / int_time[in_tag] / resp[in_tag]
    else:
        rad_resp_in = None
    #\----------------------------------------------------------------------------/#


    # response output
    #/----------------------------------------------------------------------------\#
    rad_resp = {
               si_tag: rad_resp_si,
               in_tag: rad_resp_in
               }

    return rad_resp
    #\----------------------------------------------------------------------------/#



def cdata_rad_resp(
        fnames_pri=None,
        fnames_tra=None,
        fnames_sec=None,
        filename_tag=None,
        which_ssfr='lasp|ssfr-a',
        which_lc='zen',
        which_lamp='f-1324',
        wvl_joint=950.0,
        wvl_range=[350.0, 2200.0],
        ):

    # check SSFR spectrometer
    #/----------------------------------------------------------------------------\#
    which_ssfr = which_ssfr.lower()
    which_lab  = which_ssfr.split('|')[0]
    if which_lab == 'nasa':
        import ssfr.nasa_ssfr as ssfr_toolbox
    elif which_lab == 'lasp':
        import ssfr.lasp_ssfr as ssfr_toolbox
    else:
        msg = '\nError [cdata_rad_resp]: <which_ssfr=> does not support <\'%s\'> (only supports <\'nasa|ssfr-6\'> or <\'lasp|ssfr-a\'> or <\'lasp|ssfr-b\'>).' % which_ssfr
        raise ValueError(msg)
    #\----------------------------------------------------------------------------/#


    # check light collector
    #/----------------------------------------------------------------------------\#
    which_lc = which_lc.lower()
    #\----------------------------------------------------------------------------/#

    if fnames_pri is not None:
        pri_resp = cal_rad_resp(
                fnames_pri,
                resp=None,
                which_ssfr=which_ssfr,
                which_lc=which_lc,
                which_lamp=which_lamp,
                )
    else:
        msg = '\nError [cdata_rad_resp]: Cannot proceed without primary calibration files.'
        raise OSError(msg)

    if fnames_tra is not None:
        transfer = cal_rad_resp(
                fnames_tra,
                resp=pri_resp,
                which_ssfr=which_ssfr,
                which_lc=which_lc,
                which_lamp=which_lamp,
                )
    else:
        msg = '\nError [cdata_rad_resp]: Cannot proceed without transfer calibration files.'
        raise OSError(msg)

    if fnames_sec is not None:
        sec_resp = cal_rad_resp(
                fnames_sec,
                resp=transfer,
                which_ssfr=which_ssfr,
                which_lc=which_lc,
                which_lamp=which_lamp,
                )
    else:
        msg = '\nWarning [cdata_rad_resp]: Secondary/field calibration files are not available, use transfer calibration files for secondary/field calibration ...'
        warnings.warn(msg)
        sec_resp = cal_rad_resp(
                fnames_tra,
                resp=transfer,
                which_ssfr=which_ssfr,
                which_lc=which_lc,
                which_lamp=which_lamp,
                )

    wvls = ssfr_toolbox.get_ssfr_wvl()

    wvl_start = wvl_range[0]
    wvl_end   = wvl_range[-1]
    logic_si = (wvls['%s_si' % which_lc] >= wvl_start) & (wvls['%s_si' % which_lc] <= wvl_joint)
    logic_in = (wvls['%s_in' % which_lc] >  wvl_joint) & (wvls['%s_in' % which_lc] <= wvl_end)

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
        fname_out = '%s_rad-resp_%s|si-%3.3d-%s|in-%3.3d.h5' % (filename_tag, which_lc, int_time['si'], which_lc, int_time['in'])
    else:
        fname_out = 'rad-resp_%s|si-%3.3d-%s|in-%3.3d.h5' % (which_lc, int_time['si'], which_lc, int_time['in'])

    f = h5py.File(fname_out, 'w')
    f['wvl']       = wvl
    f['pri_resp']  = pri_resp
    f['transfer']  = transfer
    f['sec_resp']  = sec_resp
    f.close()

    return fname_out



if __name__ == '__main__':

    pass
