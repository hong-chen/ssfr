import os
import sys
import copy
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
        spec_reverse=False,
        which_lamp='f-1324',
        int_time={'si':80.0, 'in':250.0},
        dark_extend=2,
        light_extend=2,
        verbose=True,
        ):


    # check SSFR spectrometer
    #╭────────────────────────────────────────────────────────────────────────────╮#
    which_ssfr = which_ssfr.lower()
    which_lab  = which_ssfr.split('|')[0]
    if which_lab == 'nasa':
        import ssfr.nasa_ssfr as ssfr_toolbox
    elif which_lab == 'lasp':
        import ssfr.lasp_ssfr as ssfr_toolbox
    else:
        msg = '\nError [cal_rad_resp]: <which_ssfr=> does not support <\'%s\'> (only supports <\'nasa|ssfr-6\'> or <\'lasp|ssfr-a\'> or <\'lasp|ssfr-b\'>).' % which_ssfr
        raise ValueError(msg)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # check light collector
    #╭────────────────────────────────────────────────────────────────────────────╮#
    which_lc = which_lc.lower()
    if (which_lc in ['zenith', 'zen', 'z']) | ('zen' in which_lc):
        which_lc = 'zen'
        if not spec_reverse:
            which_spec = 'zen'
            index_si = 0
            index_in = 1
        else:
            which_spec = 'nad'
            index_si = 2
            index_in = 3
    elif (which_lc in ['nadir', 'nad', 'n']) | ('nad' in which_lc):
        which_lc = 'nad'
        if not spec_reverse:
            which_spec = 'nad'
            index_si = 2
            index_in = 3
        else:
            which_spec = 'zen'
            index_si = 0
            index_in = 1
    else:
        msg = '\nError [cal_rad_resp]: <which_lc=> does not support <\'%s\'> (only supports <\'zenith, zen, z\'> or <\'nadir, nad, n\'>).' % which_lc
        raise ValueError(msg)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # si/in tag
    #╭────────────────────────────────────────────────────────────────────────────╮#
    si_tag = '%s|si' % which_spec
    in_tag = '%s|in' % which_spec

    if si_tag not in int_time.keys():
        int_time[si_tag] = int_time.pop('si')

    if in_tag not in int_time.keys():
        int_time[in_tag] = int_time.pop('in')
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # print message
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if verbose:
        if resp is None:
            msg = '\nMessage [cal_rad_resp]: processing primary response for <%s|%s|SI-%3.3d|IN-%3.3d> ...' % (which_ssfr.upper(), which_lc.upper(), int_time[si_tag], int_time[in_tag])
        else:
            msg = '\nMessage [cal_rad_resp]: processing transfer/secondary response for <%s|%s|SI-%3.3d|IN-%3.3d> ...' % (which_ssfr.upper(), which_lc.upper(), int_time[si_tag], int_time[in_tag])
        print(msg)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # get radiometric response
    # by default (resp=None), this function will perform primary radiometric calibration
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if resp is None:

        # check lamp
        #╭──────────────────────────────────────────────────────────────╮#
        which_lamp = which_lamp.lower()

        if (which_lamp[:4] == 'f-50') or (which_lamp[-3:-1] == '50') or (('50' in which_lamp) and ('150' not in which_lamp)):
            which_lamp = 'f-506c'
        elif (which_lamp[-4:] == '1324') or ('1324' in which_lamp):
            which_lamp = 'f-1324'
        #╰──────────────────────────────────────────────────────────────╯#


        # read in calibrated lamp data and interpolated/integrated at SSFR wavelengths/slits
        #╭──────────────────────────────────────────────────────────────╮#
        fname_lamp = '%s/lamp/%s.dat' % (ssfr.common.fdir_data, which_lamp)
        if not os.path.exists(fname_lamp):
            msg = '\nError [cal_rad_resp]: cannot locate calibration file for lamp <%s>.' % which_lamp
            raise OSError(msg)

        if verbose:
            msg = '\nMessage [cal_rad_resp]: using calibrated lamp <%s> with lamp file at \n  <%s>...' % (which_lamp, fname_lamp)
            print(msg)

        data      = np.loadtxt(fname_lamp)
        data_wvl  = data[:, 0]
        if which_lamp == 'f-506c':
            data_flux = data[:, 1]*0.01      # W m^-2 nm^-1
        else:
            data_flux = data[:, 1]*10000.0   # W m^-2 nm^-1
        #╰──────────────────────────────────────────────────────────────╯#


        # get ssfr wavelength for two spectrometers
        #╭──────────────────────────────────────────────────────────────╮#
        wvls = ssfr_toolbox.get_ssfr_wvl(which_ssfr)
        wvl_si = wvls[si_tag]
        wvl_in = wvls[in_tag]
        #╰──────────────────────────────────────────────────────────────╯#


        # use SSFR slit functions to get flux from lamp file
        # the other option is to interpolate the lamp file at SSFR wavelength, which is commented out
        #╭──────────────────────────────────────────────────────────────╮#
        # lamp_nist_si = np.zeros_like(wvl_si)
        # for i in range(lamp_nist_si.size):
        #     lamp_nist_si[i] = ssfr.util.cal_weighted_flux(wvl_si[i], data_wvl, data_flux, slit_func_file='%s/slit/vis_0.1nm_s.dat' % ssfr.common.fdir_data)

        # lamp_nist_in = np.zeros_like(wvl_in)
        # for i in range(lamp_nist_in.size):
        #     lamp_nist_in[i] = ssfr.util.cal_weighted_flux(wvl_in[i], data_wvl, data_flux, slit_func_file='%s/slit/nir_0.1nm_s.dat' % ssfr.common.fdir_data)

        lamp_nist_si = np.interp(wvl_si, data_wvl, data_flux)
        lamp_nist_in = np.interp(wvl_in, data_wvl, data_flux)
        #╰──────────────────────────────────────────────────────────────╯#

        resp = {
                si_tag: lamp_nist_si,
                in_tag: lamp_nist_in
               }
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # read raw data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    try:
        ssfr0 = ssfr_toolbox.read_ssfr(fnames, dark_extend=dark_extend, light_extend=light_extend, verbose=False)

        # integration time fallback
        # in case the data does not contain measurement with given integration time
        #╭──────────────────────────────────────────────────────────────╮#
        int_time_si_diff = np.zeros(ssfr0.Ndset, dtype=np.float64)
        int_time_in_diff = np.zeros(ssfr0.Ndset, dtype=np.float64)
        for idset, dset_tag in enumerate(ssfr0.dset_info.keys()):
            int_time_si_diff[idset] = (ssfr0.dset_info[dset_tag][si_tag]-int_time[si_tag])
            int_time_in_diff[idset] = (ssfr0.dset_info[dset_tag][in_tag]-int_time[in_tag])

        idset_si = np.argmin(np.abs(int_time_si_diff))
        idset_in = np.argmin(np.abs(int_time_in_diff))

        int_time_new = copy.deepcopy(int_time)
        if int_time_si_diff[idset_si] != 0.0:
            int_time_si_new = int_time_si_diff[idset_si]+int_time[si_tag]
            msg = '\nWarning [cal_rad_resp]: Cannot find given integration time for <%s=%dms>, fallback to <%s=%dms>' % (si_tag, int_time[si_tag], si_tag, int_time_si_new)
            warnings.warn(msg)
            int_time_new[si_tag] = int_time_si_new

        if int_time_in_diff[idset_in] != 0.0:
            int_time_in_new = int_time_in_diff[idset_in]+int_time[in_tag]
            msg = '\nWarning [cal_rad_resp]: Cannot find given integration time for <%s=%dms>, fallback to <%s=%dms>' % (in_tag, int_time[in_tag], in_tag, int_time_in_new)
            warnings.warn(msg)
            int_time_new[in_tag] = int_time_in_new
        #╰──────────────────────────────────────────────────────────────╯#

        logic_si = (np.abs(ssfr0.data_raw['int_time'][:, index_si]-int_time_new[si_tag])<0.00001)
        logic_in = (np.abs(ssfr0.data_raw['int_time'][:, index_in]-int_time_new[in_tag])<0.00001)

        shutter, counts = ssfr.corr.dark_corr(ssfr0.data_raw['tmhr'][logic_si], ssfr0.data_raw['shutter'][logic_si], ssfr0.data_raw['count_raw'][logic_si, :, index_si], mode='interp', dark_extend=dark_extend, light_extend=light_extend)
        logic  = (shutter==0)
        logic_nan = (np.sum(np.isnan(counts), axis=-1)) > 0
        spectra_si     = np.nanmean(counts[logic, :], axis=0)
        spectra_si_std = np.nanstd(counts[logic, :], axis=0)

        shutter, counts = ssfr.corr.dark_corr(ssfr0.data_raw['tmhr'][logic_in], ssfr0.data_raw['shutter'][logic_in], ssfr0.data_raw['count_raw'][logic_in, :, index_in], mode='interp', dark_extend=dark_extend, light_extend=light_extend)
        logic  = (shutter==0)
        spectra_in     = np.nanmean(counts[logic, :], axis=0)
        spectra_in_std = np.nanstd(counts[logic, :], axis=0)

    except Exception as error:

        print(error)
        msg = '\nWarning [rad_cal_resp]: cannot process the data, set parameters to <None>.'
        warnings.warn(msg)
        spectra_si     = None
        spectra_si_std = None
        spectra_in     = None
        spectra_in_std = None
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # Silicon
    # some placeholder ideas:
    # if nan is detected (e.g., spectra_si smaller than 0.0), one can use
    # interpolation to fill in the nan values
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if spectra_si is not None:
        spectra_si[spectra_si<=0.0] = np.nan
        rad_resp_si = spectra_si / int_time_new[si_tag] / resp[si_tag]

        spectra_si_std[spectra_si_std<=0.0] = np.nan
        rad_resp_si_std = spectra_si_std / int_time_new[si_tag] / resp[si_tag]
    else:
        rad_resp_si     = None
        rad_resp_si_std = None
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # InGaAs
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if spectra_in is not None:
        spectra_in[spectra_in<=0.0] = np.nan
        rad_resp_in = spectra_in / int_time_new[in_tag] / resp[in_tag]

        spectra_in_std[spectra_in_std<=0.0] = np.nan
        rad_resp_in_std = spectra_in_std / int_time_new[in_tag] / resp[in_tag]
    else:
        rad_resp_in     = None
        rad_resp_in_std = None
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # response output
    #╭────────────────────────────────────────────────────────────────────────────╮#
    rad_resp = {
               si_tag: rad_resp_si,
               in_tag: rad_resp_in
               }

    return rad_resp
    #╰────────────────────────────────────────────────────────────────────────────╯#



def cdata_rad_resp(
        fnames_pri=None,
        fnames_tra=None,
        fnames_sec=None,
        filename_tag=None,
        which_ssfr='lasp|ssfr-a',
        which_lc='zen',
        spec_reverse=False,
        which_lamp='f-1324',
        wvl_joint=950.0,
        wvl_range=[350.0, 2200.0],
        int_time={'si':80.0, 'in':250.0},
        verbose=True,
        ):

    # check SSFR spectrometer
    #╭────────────────────────────────────────────────────────────────────────────╮#
    which_ssfr = which_ssfr.lower()
    which_lab  = which_ssfr.split('|')[0]
    if which_lab == 'nasa':
        import ssfr.nasa_ssfr as ssfr_toolbox
    elif which_lab == 'lasp':
        import ssfr.lasp_ssfr as ssfr_toolbox
    else:
        msg = '\nError [cdata_rad_resp]: <which_ssfr=> does not support <\'%s\'> (only supports <\'nasa|ssfr-6\'> or <\'lasp|ssfr-a\'> or <\'lasp|ssfr-b\'>).' % which_ssfr
        raise ValueError(msg)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # check light collector
    #╭────────────────────────────────────────────────────────────────────────────╮#
    which_lc = which_lc.lower()
    if (which_lc in ['zenith', 'zen', 'z']) | ('zen' in which_lc):
        which_lc = 'zen'
        if not spec_reverse:
            which_spec = 'zen'
        else:
            which_spec = 'nad'
    elif (which_lc in ['nadir', 'nad', 'n']) | ('nad' in which_lc):
        which_lc = 'nad'
        if not spec_reverse:
            which_spec = 'nad'
        else:
            which_spec = 'zen'
    else:
        msg = '\nError [cdata_cos_resp]: <which_lc=> does not support <\'%s\'> (only supports <\'zenith, zen, z\'> or <\'nadir, nad, n\'>).' % which_lc
        raise ValueError(msg)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # si/in tag
    #╭────────────────────────────────────────────────────────────────────────────╮#
    si_tag = '%s|si' % which_spec
    in_tag = '%s|in' % which_spec

    if si_tag not in int_time.keys():
        int_time[si_tag] = int_time.pop('si')

    if in_tag not in int_time.keys():
        int_time[in_tag] = int_time.pop('in')
    #╰────────────────────────────────────────────────────────────────────────────╯#

    if fnames_pri is not None:
        pri_resp = cal_rad_resp(
                fnames_pri,
                resp=None,
                which_ssfr=which_ssfr,
                which_lc=which_lc,
                spec_reverse=spec_reverse,
                which_lamp=which_lamp,
                int_time=int_time,
                verbose=verbose,
                )
    else:
        msg = '\nError [cdata_rad_resp]: cannot proceed without primary calibration files.'
        raise OSError(msg)

    if fnames_tra is not None:
        transfer = cal_rad_resp(
                fnames_tra,
                resp=pri_resp,
                which_ssfr=which_ssfr,
                which_lc=which_lc,
                spec_reverse=spec_reverse,
                which_lamp=which_lamp,
                int_time=int_time,
                verbose=verbose,
                )
    else:
        msg = '\nError [cdata_rad_resp]: cannot proceed without transfer calibration files.'
        raise OSError(msg)

    if fnames_sec is not None:
        sec_resp = cal_rad_resp(
                fnames_sec,
                resp=transfer,
                which_ssfr=which_ssfr,
                which_lc=which_lc,
                spec_reverse=spec_reverse,
                which_lamp=which_lamp,
                int_time=int_time,
                verbose=verbose,
                )
    else:
        msg = '\nWarning [cdata_rad_resp]: secondary/field calibration files are not available, use transfer calibration files for secondary/field calibration ...'
        warnings.warn(msg)
        sec_resp = cal_rad_resp(
                fnames_tra,
                resp=transfer,
                which_ssfr=which_ssfr,
                which_lc=which_lc,
                spec_reverse=spec_reverse,
                which_lamp=which_lamp,
                int_time=int_time,
                verbose=verbose,
                )

    # wavelength
    #╭────────────────────────────────────────────────────────────────────────────╮#
    wvls = ssfr_toolbox.get_ssfr_wvl(which_ssfr)

    wvl_start = wvl_range[0]
    wvl_end   = wvl_range[-1]
    logic_si  = (wvls[si_tag] >= wvl_start)  & (wvls[si_tag] <= wvl_joint)
    logic_in  = (wvls[in_tag] >  wvl_joint)  & (wvls[in_tag] <= wvl_end)

    wvl_data      = np.concatenate((wvls[si_tag][logic_si], wvls[in_tag][logic_in]))
    pri_resp_data = np.concatenate((pri_resp[si_tag][logic_si], pri_resp[in_tag][logic_in]))
    transfer_data = np.concatenate((transfer[si_tag][logic_si], transfer[in_tag][logic_in]))
    sec_resp_data = np.concatenate((sec_resp[si_tag][logic_si], sec_resp[in_tag][logic_in]))

    indices_sort = np.argsort(wvl_data)
    wvl_      = wvl_data[indices_sort]
    pri_resp_ = pri_resp_data[indices_sort]
    transfer_ = transfer_data[indices_sort]
    sec_resp_ = sec_resp_data[indices_sort]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # save file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if filename_tag is not None:
        fname_out = '%s|rad-resp|%s|%s|si-%3.3d|in-%3.3d.h5' % (filename_tag, which_ssfr, which_spec, int_time[si_tag], int_time[in_tag])
    else:
        fname_out = 'rad-resp|%s|%s|si-%3.3d|in-%3.3d.h5' % (which_ssfr, which_spec, int_time[si_tag], int_time[in_tag])

    f = h5py.File(fname_out, 'w')
    f['wvl']       = wvl_
    f['pri_resp']  = pri_resp_
    f['transfer']  = transfer_
    f['sec_resp']  = sec_resp_

    # raw data
    #╭────────────────────────────────────────────────╮#
    g = f.create_group('raw')
    g_si = g.create_group('si')
    g_si['wvl'] = wvls[si_tag]
    g_si['pri_resp'] = pri_resp[si_tag]
    g_si['transfer'] = transfer[si_tag]
    g_si['sec_resp'] = sec_resp[si_tag]

    g_in = g.create_group('in')
    g_in['wvl'] = wvls[in_tag]
    g_in['pri_resp'] = pri_resp[in_tag]
    g_in['transfer'] = transfer[in_tag]
    g_in['sec_resp'] = sec_resp[in_tag]
    #╰────────────────────────────────────────────────╯#

    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return fname_out



if __name__ == '__main__':

    pass
