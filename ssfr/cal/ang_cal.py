import os
import sys
import glob
import datetime
import multiprocessing as mp
from collections import OrderedDict
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav

import ssfr.util
import ssfr.corr



__all__ = ['cal_ang_resp', 'cdata_ang_resp']



def cal_ang_resp(
        fnames,
        which_ssfr='lasp|ssfr-a',
        which_lc='zen',
        int_time={'si':60, 'in':300},
        Nchan=256
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
        msg = '\nError [cal_ang_resp]: <which_ssfr=> does not support <\'%s\'> (only supports <\'nasa|ssfr-6\'> or <\'lasp|ssfr-a\'> or <\'lasp|ssfr-b\'>).' % which_ssfr
        raise ValueError(msg)
    #\----------------------------------------------------------------------------/#


    # check light collector
    #/----------------------------------------------------------------------------\#
    which_lc = which_lc.lower()
    if (which_lc in ['zenith', 'zen', 'z']) | ('zen' in which_lc):
        which_lc = 'zen'
        index_si = 0
        index_in = 1
    elif (which_lc in ['nadir', 'nad', 'n']) | ('nad' in which_lc):
        which_lc = 'nad'
        index_si = 2
        index_in = 3
    else:
        msg = '\nError [cal_ang_resp]: <which_lc=> does not support <\'%s\'> (only supports <\'zenith, zen, z\'> or <\'nadir, nad, n\'>).' % which_lc
        raise ValueError(msg)
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

    Nfile = len(fnames)

    counts_si   = np.zeros((Nfile, Nchan), dtype=np.float64)
    counts_in   = np.zeros((Nfile, Nchan), dtype=np.float64)
    for i, fname in enumerate(fnames.keys()):

        ssfr0 = ssfr_toolbox.read_ssfr([fname])

        logic_si = (np.abs(ssfr0.data_raw['int_time'][:, index_si]-int_time[si_tag])<0.00001)
        logic_in = (np.abs(ssfr0.data_raw['int_time'][:, index_in]-int_time[in_tag])<0.00001)

        shutter, counts = ssfr.corr.dark_corr(ssfr0.data_raw['tmhr'][logic_si], ssfr0.data_raw['shutter'][logic_si], ssfr0.data_raw['count_raw'][logic_si, :, index_si], mode='interp')
        logic  = (shutter==0)
        counts_si[i, :] = np.nanmean(counts[logic, :], axis=0)

        shutter, counts = ssfr.corr.dark_corr(ssfr0.data_raw['tmhr'][logic_in], ssfr0.data_raw['shutter'][logic_in], ssfr0.data_raw['count_raw'][logic_in, :, index_in], mode='interp')
        logic  = (shutter==0)
        counts_in[i, :] = np.nanmean(counts[logic, :], axis=0)

    ang_resp = {
            si_tag: counts_si/(np.tile(counts_si[0, :], Nfile).reshape(Nfile, -1)),
            in_tag: counts_in/(np.tile(counts_in[0, :], Nfile).reshape(Nfile, -1)),
            }

    return ang_resp



def cdata_ang_resp(
        fnames,
        fdir_out=None,
        filename_tag=None,
        which_ssfr='lasp|ssfr-a',
        which_lc='zen',
        Nchan=256,
        wvl_joint=950.0,
        wvl_range=[350.0, 2200.0],
        int_time={'si':60, 'in':300},
        calibrated_by='Yu-Wen Chen, Ken Hirata',
        processed_by='Ken Hirata',
        verbose=True
        ):

    """
    Process angular response calibration data for SSFR spectrometers.
    This function processes raw angular response measurements from SSFR (Solar Spectral
    Flux Radiometer) instruments to generate calibrated angular response functions. It
    combines data from both Silicon (Si) and InGaAs (In) detector channels, performs
    interpolation, and fits polynomial models to the spectral response.

    Args:
    ----
        fnames (dict): Dictionary mapping file paths to measurement angles in degrees.
                        Format: {'/path/to/file.h5': angle_in_degrees}
        fdir_out (str, optional): Output directory path. If None, saves to current directory.
        filename_tag (str, optional): Prefix tag for output filename. If None, no prefix added.
        which_ssfr (str): SSFR instrument identifier. Supports 'nasa|ssfr-6', 'lasp|ssfr-a',
                            or 'lasp|ssfr-b'. Default is 'lasp|ssfr-a'.
        which_lc (str): Light collector specification. Accepts 'zenith'/'zen'/'z' or
                        'nadir'/'nad'/'n'. Default is 'zen'.
        Nchan (int): Number of spectral channels. Default is 256.
        wvl_joint (float): Joint wavelength in nm separating Si and InGaAs channels.
                            Default is 950.0 nm.
        wvl_range (list): Two-element list [start_wvl, end_wvl] defining wavelength range
                            in nm. Default is [350.0, 2200.0].
        int_time (dict): Integration times in ms for each channel. Format: {'si': ms, 'in': ms}.
                        Default is {'si': 60, 'in': 300}.
        verbose (bool): If True, prints processing information. Default is True.

    Returns:
    -------
        str: Path to the output HDF5 file containing processed angular response data.

    Output HDF5 File Structure:
        Global Attributes:
            - title: File description
            - instrument: SSFR instrument identifier
            - light_collector: Light collector type
            - creation_date: Processing timestamp
            - joint_wavelength_nm: Wavelength separating Si/InGaAs channels
            - wavelength_range_nm: Processing wavelength range
            - integration_time_si_ms/integration_time_in_ms: Channel integration times
            - polynomial_order: Order of fitted polynomials
            - description: File content description
        Datasets:
            /wvl: Combined wavelength array from both channels [nm]
            /mu: Interpolated cosine grid (1001 points from 0 to 1)
            /ang_resp: Angular response on interpolated grid (mu, wavelength)
            /ang_resp_int: Hemispherically integrated angular response (wavelength,)
            /poly_coef: Polynomial coefficients for response vs wavelength (mu, order+1)
            /info: Human-readable processing information string
        /raw/ Group - Raw measurement data:
            /raw/ang: Original measurement angles [degrees]
            /raw/mu: Cosine of measurement angles
            /raw/mu0: Unique cosine values after averaging
            /raw/{si_tag|in_tag}/ Subgroups for each channel:
                /raw/{channel}/wvl: Channel wavelength array [nm]
                /raw/{channel}/ang_resp: Raw angular response (angle, wavelength)
                /raw/{channel}/ang_resp0: Averaged response for unique angles
                /raw/{channel}/ang_resp_std0: Standard deviation of averaged response
    Notes:
        - Angular response is normalized to nadir (0 degrees) response
        - Data from duplicate angle measurements are averaged
        - Polynomial fitting uses wavelengths between 400-2000 nm with 4th order
        - Si channel covers wavelengths up to wvl_joint, InGaAs covers above wvl_joint
        - All angular response values are dimensionless ratios
    """

    # check SSFR spectrometer
    #/----------------------------------------------------------------------------\#
    which_ssfr = which_ssfr.lower()
    which_lab  = which_ssfr.split('|')[0]
    if which_lab == 'nasa':
        import ssfr.nasa_ssfr as ssfr_toolbox
    elif which_lab == 'lasp':
        import ssfr.lasp_ssfr as ssfr_toolbox
    else:
        msg = '\nError [cdata_ang_resp]: <which_ssfr=> does not support <\'%s\'> (only supports <\'nasa|ssfr-6\'> or <\'lasp|ssfr-a\'> or <\'lasp|ssfr-b\'>).' % which_ssfr
        raise ValueError(msg)
    #\----------------------------------------------------------------------------/#


    # check light collector
    #/----------------------------------------------------------------------------\#
    which_lc = which_lc.lower()
    if (which_lc in ['zenith', 'zen', 'z']) | ('zen' in which_lc):
        which_lc = 'zen'
        index_si = 0
        index_in = 1
    elif (which_lc in ['nadir', 'nad', 'n']) | ('nad' in which_lc):
        which_lc = 'nad'
        index_si = 2
        index_in = 3
    else:
        msg = '\nError [cdata_ang_resp]: <which_lc=> does not support <\'%s\'> (only supports <\'zenith, zen, z\'> or <\'nadir, nad, n\'>).' % which_lc
        raise ValueError(msg)
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


    # get cosine response (aka angular response)
    #/----------------------------------------------------------------------------\#
    ang_resp_ = cal_ang_resp(fnames, which_ssfr=which_ssfr, which_lc=which_lc, Nchan=Nchan, int_time=int_time)
    #\----------------------------------------------------------------------------/#


    # average the data from two different directions
    #/----------------------------------------------------------------------------\#
    angles = np.array([fnames[fname] for fname in fnames.keys()])
    ang_mu = np.cos(np.deg2rad(angles))

    ang_mu0 = np.sort(np.unique(ang_mu))[::-1]
    Nmu0    = ang_mu0.size

    ang_resp0 = {
            si_tag: np.zeros((Nmu0, Nchan), dtype=np.float64),
            in_tag: np.zeros((Nmu0, Nchan), dtype=np.float64)
            }

    ang_resp_std0 = {
            si_tag: np.zeros((Nmu0, Nchan), dtype=np.float64),
            in_tag: np.zeros((Nmu0, Nchan), dtype=np.float64)
            }

    for i, mu0 in enumerate(ang_mu0):
        indices = np.where(ang_mu==mu0)[0]
        if indices.size >= 2:
            ang_resp0[si_tag][i, :] = np.nanmean(ang_resp_[si_tag][indices, :], axis=0)
            ang_resp0[in_tag][i, :] = np.nanmean(ang_resp_[in_tag][indices, :], axis=0)
            ang_resp_std0[si_tag][i, :] = np.nanstd(ang_resp_[si_tag][indices, :], axis=0)
            ang_resp_std0[in_tag][i, :] = np.nanstd(ang_resp_[in_tag][indices, :], axis=0)
        else:
            ang_resp0[si_tag][i, :] = ang_resp_[si_tag][indices[0], :]
            ang_resp0[in_tag][i, :] = ang_resp_[in_tag][indices[0], :]
            ang_resp_std0[si_tag][i, :] = np.nan
            ang_resp_std0[in_tag][i, :] = np.nan
    #\----------------------------------------------------------------------------/#


    # gridding the data
    #/----------------------------------------------------------------------------\#
    ang_mu_all   = np.linspace(0.0, 1.0, 1001)
    Nmu_all      = ang_mu_all.size
    ang_resp_all = {
            si_tag: np.zeros((Nmu_all, Nchan), dtype=np.float64),
            in_tag: np.zeros((Nmu_all, Nchan), dtype=np.float64)
            }
    ang_resp_std_all = {
            si_tag: np.zeros((Nmu_all, Nchan), dtype=np.float64),
            in_tag: np.zeros((Nmu_all, Nchan), dtype=np.float64)
            }

    for i in range(Nchan):

        f = interpolate.interp1d(ang_mu0, ang_resp0[si_tag][:, i], fill_value='extrapolate')
        ang_resp_all[si_tag][:, i] = f(ang_mu_all)

        f = interpolate.interp1d(ang_mu0, ang_resp0[in_tag][:, i], fill_value='extrapolate')
        ang_resp_all[in_tag][:, i] = f(ang_mu_all)

        f = interpolate.interp1d(ang_mu0, ang_resp_std0[si_tag][:, i], fill_value='extrapolate')
        ang_resp_std_all[si_tag][:, i] = f(ang_mu_all)

        f = interpolate.interp1d(ang_mu0, ang_resp_std0[in_tag][:, i], fill_value='extrapolate')
        ang_resp_std_all[in_tag][:, i] = f(ang_mu_all)
    #\----------------------------------------------------------------------------/#


    # wavelength fitting
    #/----------------------------------------------------------------------------\#
    wvls = ssfr_toolbox.get_ssfr_wvl(which_ssfr)

    wvl_start = wvl_range[0]
    wvl_end   = wvl_range[-1]
    logic_si  = (wvls[si_tag] >= wvl_start)  & (wvls[si_tag] <= wvl_joint)
    logic_in  = (wvls[in_tag] >  wvl_joint)  & (wvls[in_tag] <= wvl_end)

    wvl_data      = np.concatenate((wvls[si_tag][logic_si], wvls[in_tag][logic_in]))
    ang_resp_data = np.concatenate((ang_resp_all[si_tag][:, logic_si], ang_resp_all[in_tag][:, logic_in]), axis=1)

    indices_sort = np.argsort(wvl_data)
    wvl          = wvl_data[indices_sort]
    ang_resp = ang_resp_data[:, indices_sort]

    ang_resp_int = np.zeros(wvl.size, dtype=np.float64)
    for i in range(wvl.size):
        ang_resp_int[i] = np.trapz(ang_resp[:, i], x=ang_mu_all)

    logic = (wvl>=400.0) & (wvl<=2000.0)
    order = 4
    coef  = np.zeros((Nmu_all, order+1), dtype=np.float64)
    for i in range(Nmu_all):
        coef[i, :] = np.polyfit(wvl[logic], ang_resp[i, :][logic], order)
    #\----------------------------------------------------------------------------/#


    # save file
    #/----------------------------------------------------------------------------\#
    if filename_tag is not None:
        fname_out = '%s|ang-resp|%s|%s|si-%3.3d|in-%3.3d.h5' % (filename_tag, which_ssfr, which_lc, int_time[si_tag], int_time[in_tag])
    else:
        fname_out = 'ang-resp|%s|%s|si-%3.3d|in-%3.3d.h5' % (which_ssfr, which_lc, int_time[si_tag], int_time[in_tag])

    # add output directory
    if fdir_out is not None:
        if not os.path.exists(fdir_out):
            os.makedirs(fdir_out)
        fname_out = os.path.join(fdir_out, fname_out)

    info = 'Light Collector: %s\nJoint Wavelength: %.4fnm\nStart Wavelength: %.4fnm\nEnd Wavelength: %.4fnm\nIntegration Time for Silicon Channel: %dms\nIntegration Time for InGaAs Channel: %dms\nProcessed Files:\n' % (which_lc.title(), wvl_joint, wvl_start, wvl_end, int_time[si_tag], int_time[in_tag])
    for key in fnames.keys():
        line = 'At %3d [degree] Angle: %s\n' % (fnames[key], key)
        info += line

    if verbose:
        print(info)

    # create HDF5 file and add metadata
    #/----------------------------------------------------------------------------\#
    f = h5py.File(fname_out, 'w')

    # Add global attributes for file metadata
    f.attrs['title'] = 'SSFR Angular Response Calibration Data'
    f.attrs['instrument'] = which_ssfr
    f.attrs['light_collector'] = which_lc
    f.attrs['joint_wavelength_nm'] = wvl_joint
    f.attrs['wavelength_range_nm'] = [wvl_start, wvl_end]
    f.attrs['integration_time_si_ms'] = int_time[si_tag]
    f.attrs['integration_time_in_ms'] = int_time[in_tag]
    f.attrs['polynomial_order'] = order
    f.attrs['description'] = 'Angular response calibration data for SSFR spectrometer including raw measurements and processed products'
    f.attrs['processing_team'] = processed_by
    f.attrs['calibration_team'] = calibrated_by
    f.attrs['created_on'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    # Raw data group with comprehensive metadata
    g = f.create_group('raw')
    g.attrs['description'] = 'Raw angular response measurements and associated angles'

    # Raw angles and cosines
    ang_dset = g.create_dataset('ang', data=angles)
    ang_dset.attrs['description'] = 'Measurement angles for angular response calibration'
    ang_dset.attrs['units'] = 'degrees'
    ang_dset.attrs['long_name'] = 'Incident angles relative to instrument normal'

    mu_dset = g.create_dataset('mu', data=ang_mu)
    mu_dset.attrs['description'] = 'Cosine of measurement angles (mu = cos(theta))'
    mu_dset.attrs['units'] = 'dimensionless'
    mu_dset.attrs['long_name'] = 'Cosine of incident angles'
    mu_dset.attrs['valid_range'] = [0.0, 1.0]

    mu0_dset = g.create_dataset('mu0', data=ang_mu0)
    mu0_dset.attrs['description'] = 'Unique cosine values after averaging duplicate measurements'
    mu0_dset.attrs['units'] = 'dimensionless'
    mu0_dset.attrs['long_name'] = 'Unique cosine of incident angles'
    mu0_dset.attrs['valid_range'] = [0.0, 1.0]

    # Raw spectral data for each channel
    for spec_tag in ang_resp_.keys():
        g_ = g.create_group(spec_tag)
        g_.attrs['description'] = f'Raw data for {spec_tag} spectrometer channel'
        g_.attrs['integration_time_ms'] = int_time[spec_tag]

        wvl_dset = g_.create_dataset('wvl', data=wvls[spec_tag])
        wvl_dset.attrs['description'] = f'Wavelength array for {spec_tag} channel'
        wvl_dset.attrs['units'] = 'nm'
        wvl_dset.attrs['long_name'] = 'Wavelength'
        wvl_dset.attrs['standard_name'] = 'radiation_wavelength'

        resp_dset = g_.create_dataset('ang_resp', data=ang_resp_[spec_tag])
        resp_dset.attrs['description'] = 'Raw angular response normalized to nadir (0 degrees)'
        resp_dset.attrs['units'] = 'dimensionless'
        resp_dset.attrs['long_name'] = 'Angular response function'
        resp_dset.attrs['dimensions'] = '(angle, wavelength)'
        resp_dset.attrs['normalization'] = 'Normalized to response at 0 degrees'

        resp0_dset = g_.create_dataset('ang_resp0', data=ang_resp0[spec_tag])
        resp0_dset.attrs['description'] = 'Angular response averaged over duplicate angle measurements'
        resp0_dset.attrs['units'] = 'dimensionless'
        resp0_dset.attrs['long_name'] = 'Averaged angular response function'
        resp0_dset.attrs['dimensions'] = '(unique_angle, wavelength)'
        resp0_dset.attrs['normalization'] = 'Normalized to response at 0 degrees'

        std_dset = g_.create_dataset('ang_resp_std0', data=ang_resp_std0[spec_tag])
        std_dset.attrs['description'] = 'Standard deviation of angular response for duplicate measurements'
        std_dset.attrs['units'] = 'dimensionless'
        std_dset.attrs['long_name'] = 'Angular response standard deviation'
        std_dset.attrs['dimensions'] = '(unique_angle, wavelength)'

    # Processed data with enhanced metadata
    wvl_dset = f.create_dataset('wvl', data=wvl)
    wvl_dset.attrs['description'] = 'Combined wavelength array from both Si and InGaAs channels'
    wvl_dset.attrs['units'] = 'nm'
    wvl_dset.attrs['long_name'] = 'Wavelength'
    wvl_dset.attrs['standard_name'] = 'radiation_wavelength'
    wvl_dset.attrs['joint_wavelength_nm'] = wvl_joint
    wvl_dset.attrs['wavelength_range_nm'] = [wvl.min(), wvl.max()]

    mu_all_dset = f.create_dataset('mu', data=ang_mu_all)
    mu_all_dset.attrs['description'] = 'Interpolated cosine grid from 0 to 1 with 1001 points'
    mu_all_dset.attrs['units'] = 'dimensionless'
    mu_all_dset.attrs['long_name'] = 'Cosine of incident angles (interpolated grid)'
    mu_all_dset.attrs['valid_range'] = [0.0, 1.0]
    mu_all_dset.attrs['grid_points'] = ang_mu_all.size

    ang_resp_dset = f.create_dataset('ang_resp', data=ang_resp)
    ang_resp_dset.attrs['description'] = 'Angular response interpolated onto regular cosine grid'
    ang_resp_dset.attrs['units'] = 'dimensionless'
    ang_resp_dset.attrs['long_name'] = 'Angular response function (interpolated)'
    ang_resp_dset.attrs['dimensions'] = '(cosine_angle, wavelength)'
    ang_resp_dset.attrs['normalization'] = 'Normalized to response at 0 degrees (mu=1)'
    ang_resp_dset.attrs['interpolation_method'] = 'linear'

    ang_resp_int_dset = f.create_dataset('ang_resp_int', data=ang_resp_int)
    ang_resp_int_dset.attrs['description'] = 'Integrated angular response over all angles (hemispherical integral)'
    ang_resp_int_dset.attrs['units'] = 'dimensionless'
    ang_resp_int_dset.attrs['long_name'] = 'Hemispherically integrated angular response'
    ang_resp_int_dset.attrs['dimensions'] = '(wavelength,)'
    ang_resp_int_dset.attrs['integration_method'] = 'trapezoidal'
    ang_resp_int_dset.attrs['integration_variable'] = 'cosine of zenith angle'

    poly_dset = f.create_dataset('poly_coef', data=coef)
    poly_dset.attrs['description'] = f'Polynomial coefficients (order {order}) for angular response vs wavelength'
    poly_dset.attrs['units'] = 'various'
    poly_dset.attrs['long_name'] = 'Polynomial coefficients for angular response'
    poly_dset.attrs['dimensions'] = '(cosine_angle, polynomial_coefficient)'
    poly_dset.attrs['polynomial_order'] = order
    poly_dset.attrs['wavelength_fit_range_nm'] = [400.0, 2000.0]
    poly_dset.attrs['coefficient_order'] = 'highest to lowest order (numpy polyfit convention)'

    info_dset = f.create_dataset('info', data=info)
    info_dset.attrs['description'] = 'Human-readable processing information and file manifest'
    info_dset.attrs['content'] = 'Processing parameters and list of input files'

    f.close()
    #\----------------------------------------------------------------------------/#


    # plot
    #/----------------------------------------------------------------------------\#
    if False:
        import matplotlib as mpl
        # mpl.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedLocator
        from matplotlib import rcParams
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches
        # import cartopy.crs as ccrs

        index = 900

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)

        ax1.fill_between(wvls[si_tag], ang_resp_all[si_tag][index, :]-ang_resp_std_all[si_tag][index, :], ang_resp_all[si_tag][index, :]+ang_resp_std_all[si_tag][index, :], facecolor='red', lw=0.0, alpha=0.3)
        ax1.scatter(wvls[si_tag], ang_resp_all[si_tag][index, :], s=6, c='red', lw=0.0)

        ax1.fill_between(wvls[in_tag], ang_resp_all[in_tag][index, :]-ang_resp_std_all[in_tag][index, :], ang_resp_all[in_tag][index, :]+ang_resp_std_all[in_tag][index, :], facecolor='blue', lw=0.0, alpha=0.3)
        ax1.scatter(wvls[in_tag], ang_resp_all[in_tag][index, :], s=6, c='blue', lw=0.0)

        ax1.scatter(wvl, ang_resp[index, :], s=3, c='k', lw=0.0)

        wvl_new = np.linspace(wvl.min(), wvl.max(), 1000)
        p0 = np.poly1d(coef[index, :])
        ax1.plot(wvl_new, p0(wvl_new), lw=3.0, alpha=0.6, color='g')

        ax1.axvline(wvl_joint, ls='--', color='gray')

        ax1.set_title('%s\n$\\mu$=%.4f (%.2f$^\\circ$)' % (os.path.basename(fname_out), ang_mu_all[index], np.rad2deg(np.arccos(ang_mu_all[index]))))

        ax1.set_ylim((0.85, 0.95))
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Response')

        plt.savefig('ang_resp_mu-%06.4f.png' % ang_mu_all[index], bbox_inches='tight')
    #\----------------------------------------------------------------------------/#


    return fname_out



if __name__ == '__main__':

    pass
