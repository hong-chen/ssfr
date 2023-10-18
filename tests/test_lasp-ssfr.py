import os
import sys
import glob
import datetime
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.image as mpl_img
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cartopy.crs as ccrs
# mpl.use('Agg')


import ssfr


_mission_   = 'arcsix'
_spns_      = 'spns-b'
_ssfr_      = 'ssfr-b'
_fdir_data_ = '/argus/pre-mission/%s' % _mission_
_fdir_hsk_  = '%s/raw/hsk'
_fdir_ssfr_ = '%s/raw/%s' % (_fdir_data_, _ssfr_)
_fdir_spns_ = '%s/raw/%s' % (_fdir_data_, _spns_)
_fdir_v0_   = '%s/processed'  % _fdir_data_
_fdir_v1_   = '%s/processed'  % _fdir_data_
_fdir_v2_   = '%s/processed'  % _fdir_data_



# old
# ================================================================================================
def cdata_alp_v0(date, fdir_processed, fdir_data='/argus/field/camp2ex', run=True):

    date_s = date.strftime('%Y%m%d')

    fdir_raw  = '%s/raw/alp' % (fdir_data)
    fnames_alp_raw = sorted(glob.glob('%s/*.plt3' % fdir_raw))

    fname_alp      = '%s/alp_%s_v0.h5' % (fdir_processed, date_s)
    if run:

        if date.year == 2016:
            from alp import cu_alp_v2 as cu_alp
        else:
            from alp import cu_alp

        # create ALP V0 data
        # ============================================================================
        alp0 = cu_alp(fnames_alp_raw, date=date)
        alp0.save_h5(fname_alp)
        # ============================================================================

    return fname_alp

def cdata_ssfr_aux(date, fdir_processed, run=True):

    date_s = date.strftime('%Y%m%d')

    fname_ssfr_v0 = get_file(fdir_processed, full=True, contains=['ssfr', 'v0', date_s])
    ssfr_v0 = nasa_ssfr(fnames=[], fname_v0=fname_ssfr_v0)

    # calculate diffuse-to-global ratio for SSFR
    # ========================================================================================
    fname_spns = get_file(fdir_processed, full=True, contains=['spns', 'v0', date_s])
    data_spns  = load_h5(fname_spns)

    dif_flux0 = data_spns['dif_flux']
    dif_tmhr0 = data_spns['dif_tmhr']
    dif_wvl0  = data_spns['dif_wvl']

    dif_flux1 = np.zeros((ssfr_v0.tmhr.size, dif_wvl0.size), dtype=np.float64); dif_flux1[...] = np.nan
    for i in range(dif_wvl0.size):
        dif_flux1[:, i] = interp(ssfr_v0.tmhr, dif_tmhr0, dif_flux0[:, i])

    dif_flux = np.zeros_like(ssfr_v0.zen_cnt); dif_flux[...] = np.nan
    for i in range(ssfr_v0.tmhr.size):
        dif_flux[i, :] = interp(ssfr_v0.zen_wvl, dif_wvl0, dif_flux1[i, :])


    tot_flux0 = data_spns['tot_flux']
    tot_tmhr0 = data_spns['tot_tmhr']
    tot_wvl0  = data_spns['tot_wvl']

    tot_flux1 = np.zeros((ssfr_v0.tmhr.size, tot_wvl0.size), dtype=np.float64); tot_flux1[...] = np.nan
    for i in range(dif_wvl0.size):
        tot_flux1[:, i] = interp(ssfr_v0.tmhr, tot_tmhr0, tot_flux0[:, i])

    tot_flux = np.zeros_like(ssfr_v0.zen_cnt); tot_flux[...] = np.nan
    for i in range(ssfr_v0.tmhr.size):
        tot_flux[i, :] = interp(ssfr_v0.zen_wvl, tot_wvl0, tot_flux1[i, :])

    diff_ratio0 = dif_flux / tot_flux
    diff_ratio  = np.zeros_like(ssfr_v0.zen_cnt)  ; diff_ratio[...] = np.nan
    coefs       = np.zeros((ssfr_v0.tmhr.size, 3)); coefs[...] = np.nan
    qual_flag   = np.repeat(0, ssfr_v0.tmhr.size)

    for i in tqdm(range(diff_ratio.shape[0])):

        logic = (diff_ratio0[i, :]>=0.0) & (diff_ratio0[i, :]<=1.0) & (ssfr_v0.zen_wvl>=400.0) & (ssfr_v0.zen_wvl<=750.0)
        if logic.sum() > 20:

            x = ssfr_v0.zen_wvl[logic]
            y = diff_ratio0[i, logic]
            popt, pcov = fit_diff_ratio(x, y)

            diff_ratio[i, :] = func_diff_ratio(ssfr_v0.zen_wvl, *popt)
            diff_ratio[i, diff_ratio[i, :]>1.0] = 1.0
            diff_ratio[i, diff_ratio[i, :]<0.0] = 0.0

            coefs[i, :] = popt
            qual_flag[i] = 1

    print(np.isnan(diff_ratio).sum())

    for i in range(diff_ratio.shape[1]):
        logic_nan = np.isnan(diff_ratio[:, i])
        logic     = np.logical_not(logic_nan)

        f_interp  = interpolate.interp1d(ssfr_v0.tmhr[logic], diff_ratio[:, i][logic], bounds_error=None, fill_value='extrapolate')
        diff_ratio[logic_nan, i] = f_interp(ssfr_v0.tmhr[logic_nan])
        diff_ratio[diff_ratio[:, i]>1.0, i] = 1.0
        diff_ratio[diff_ratio[:, i]<0.0, i] = 0.0

    print(np.isnan(diff_ratio).sum())
    # ========================================================================================


    # calculate solar angles
    # ========================================================================================
    fname_hsk = get_file(fdir_processed, full=True, contains=['hsk_%s' % date_s])
    data_hsk  = load_h5(fname_hsk)

    lon = interp(ssfr_v0.tmhr, data_hsk['tmhr'], data_hsk['longitude'])
    lat = interp(ssfr_v0.tmhr, data_hsk['tmhr'], data_hsk['latitude'])
    alt = interp(ssfr_v0.tmhr, data_hsk['tmhr'], data_hsk['gps_altitude'])
    sza, saa = cal_solar_angles(ssfr_v0.jday, lon, lat, alt)
    # ========================================================================================


    if run:
        fname = '%s/ssfr_%s_aux.h5' % (fdir_processed, date_s)
        f = h5py.File(fname, 'w')
        f['tmhr'] = ssfr_v0.tmhr
        f['alt']  = alt
        f['lon']  = lon
        f['lat']  = lat
        f['sza']  = sza
        f['saa']  = saa
        f['diff_ratio_x']         = ssfr_v0.zen_wvl
        f['diff_ratio_coef']      = coefs
        f['diff_ratio_qual_flag'] = qual_flag
        f['diff_ratio']           = diff_ratio
        f['diff_ratio_ori']       = diff_ratio0
        f.close()

    return fname

def cdata_ssfr_hsk(date,
                   fdir_processed,
                   fdir_cal,
                   pitch_angle=0.0,
                   roll_angle=0.0,
                   version_info='R0: initial science data release'
                   ):

    date_s = date.strftime('%Y%m%d')

    fname_hsk = get_file(fdir_processed, full=True, contains=['hsk_%s' % date_s])
    data_hsk = load_h5(fname_hsk)

    fname_ssfr_v0 = get_file(fdir_processed, full=True, contains=['ssfr', 'v0', date_s])
    ssfr_v0 = nasa_ssfr(fnames=[], fname_v0=fname_ssfr_v0)

    fname_ssfr_aux = get_file(fdir_processed, full=True, contains=['ssfr', 'aux', date_s])
    ssfr_aux = load_h5(fname_ssfr_aux)


    # pitch_angles = np.arange(-5.0, 5.1, 2.0)
    # roll_angles  = np.arange(-5.0, 5.1, 2.0)
    # diffs = np.zeros((pitch_angles.size, roll_angles.size))
    # for i in range(pitch_angles.size):
    #     pitch_angle = pitch_angles[i]
    #     for j in range(roll_angles.size):
    #         roll_angle = roll_angles[j]
    #         diffs[i, j] = cal_diff_532(date, fdir_processed, fdir_cal, pitch_angle, roll_angle)
    #         print(i, j, pitch_angle, roll_angle, diffs[i, j])

    # fig = plt.figure(figsize=(7, 6))
    # ax1 = fig.add_subplot(111)
    # ax1.imshow(diffs, cmap='jet', origin='lower', aspect='auto')
    # # ax1.set_xlim(())
    # # ax1.set_ylim(())
    # # ax1.set_xlabel('')
    # # ax1.set_ylabel('')
    # # ax1.set_title('')
    # # ax1.legend(loc='upper right', fontsize=12, framealpha=0.4)
    # # plt.savefig('test.png')
    # plt.show()
    # exit()

    # calculate cosine correction factors
    # ?? what angles to use for pitch/roll motor, offset??
    # ========================================================================================
    angles = {}
    angles['solar_zenith']  = ssfr_aux['sza']
    angles['solar_azimuth'] = ssfr_aux['saa']
    if date < datetime.datetime(2019, 8, 24):
        fname_alp = get_file(fdir_processed, full=True, contains=['alp_%s_v0' % date_s])
        data_alp = load_h5(fname_alp)
        angles['pitch']        = interp(ssfr_v0.tmhr, data_alp['tmhr'], data_alp['ang_pit_s'])
        angles['roll']         = interp(ssfr_v0.tmhr, data_alp['tmhr'], data_alp['ang_rol_s'])
        angles['heading']      = interp(ssfr_v0.tmhr, data_hsk['tmhr'], data_hsk['true_heading'])
        angles['pitch_motor']  = interp(ssfr_v0.tmhr, data_alp['tmhr'], data_alp['ang_pit_m'])
        angles['roll_motor']   = interp(ssfr_v0.tmhr, data_alp['tmhr'], data_alp['ang_rol_m'])
        angles['pitch_motor'][np.isnan(angles['pitch_motor'])] = 0.0
        angles['roll_motor'][np.isnan(angles['roll_motor'])]   = 0.0
        angles['pitch_offset']  = pitch_angle
        angles['roll_offset']   = roll_angle

    else:

        angles['pitch']         = interp(ssfr_v0.tmhr, data_hsk['tmhr'], data_hsk['pitch_angle'])
        angles['roll']          = interp(ssfr_v0.tmhr, data_hsk['tmhr'], data_hsk['roll_angle'])
        angles['heading']       = interp(ssfr_v0.tmhr, data_hsk['tmhr'], data_hsk['true_heading'])
        angles['pitch_motor']   = np.repeat(0.0, ssfr_v0.tmhr.size)
        angles['roll_motor']    = np.repeat(0.0, ssfr_v0.tmhr.size)
        angles['pitch_offset']  = pitch_angle
        angles['roll_offset']   = roll_angle


    fdir_ang_cal = '%s/ang-cal' % fdir_cal
    fnames_ang_cal = get_ang_cal_camp2ex(date, fdir_ang_cal)
    factors = cos_corr(fnames_ang_cal, angles, diff_ratio=ssfr_aux['diff_ratio'])

    # apply cosine correction
    ssfr_v0.zen_cnt = ssfr_v0.zen_cnt*factors['zenith']
    ssfr_v0.nad_cnt = ssfr_v0.nad_cnt*factors['nadir']
    # ========================================================================================

    # primary transfer calibration
    # ?? how to determine which calibration file to use ??
    # ========================================================================================
    fdir_rad_cal = '%s/rad-cal' % fdir_cal
    fnames_rad_cal = get_rad_cal_camp2ex(date, fdir_rad_cal)
    ssfr_v0.cal_flux(fnames_rad_cal)
    # ========================================================================================

    zen_flux = np.zeros((data_hsk['tmhr'].size, ssfr_v0.zen_wvl.size), dtype=np.float64)
    for i in range(ssfr_v0.zen_wvl.size):
        zen_flux[:, i] = interp(data_hsk['tmhr'], ssfr_v0.tmhr, ssfr_v0.zen_flux[:, i])

    nad_flux = np.zeros((data_hsk['tmhr'].size, ssfr_v0.nad_wvl.size), dtype=np.float64)
    for i in range(ssfr_v0.nad_wvl.size):
        nad_flux[:, i] = interp(data_hsk['tmhr'], ssfr_v0.tmhr, ssfr_v0.nad_flux[:, i])

    sza      = interp(data_hsk['tmhr'], ssfr_aux['tmhr'], ssfr_aux['sza'])
    pitch    = interp(data_hsk['tmhr'], ssfr_aux['tmhr'], angles['pitch'])
    roll     = interp(data_hsk['tmhr'], ssfr_aux['tmhr'], angles['roll'])
    heading  = interp(data_hsk['tmhr'], ssfr_aux['tmhr'], angles['heading'])

    # primary transfer calibration
    # ========================================================================================
    comments_list = []
    comments_list.append('Bandwidth of Silicon channels (wavelength < 950nm) as defined by the FWHM: 6 nm')
    comments_list.append('Bandwidth of InGaAs channels (wavelength > 950nm) as defined by the FWHM: 12 nm')
    comments_list.append('Pitch angle offset: %.1f degree' % pitch_angle)
    comments_list.append('Roll angle offset: %.1f degree' % roll_angle)

    for key in fnames_rad_cal.keys():
        comments_list.append('Radiometric calibration file (%s): %s' % (key, os.path.basename(fnames_rad_cal[key])))
    for key in fnames_ang_cal.keys():
        comments_list.append('Angular calibration file (%s): %s' % (key, os.path.basename(fnames_ang_cal[key])))
    comments = '\n'.join(comments_list)

    print(date_s)
    print(comments)
    print()
    # ========================================================================================


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fname_spns = get_file(fdir_processed, full=True, contains=['spns', 'hsk', date_s])
    data_spns  = load_h5(fname_spns)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    index = np.argmin(np.abs(532.0-ssfr_v0.zen_wvl))
    ax1.scatter(data_hsk['tmhr'], zen_flux[:, index], s=10, lw=0, c='r')

    index = np.argmin(np.abs(532.0-data_spns['wvl']))
    ax1.scatter(data_spns['tmhr'], data_spns['tot_flux'][:, index], s=10, lw=0, c='b')

    # ax1.scatter(ssfr_v0.zen_wvl, zen_flux[5000, :], s=10, lw=0, c='r')
    # ax1.scatter(data_spns['wvl'], data_spns['tot_flux'][5000, :], s=10, lw=0, c='b')

    patches_legend = [
                mpatches.Patch(color='red'   , label='SSFR (532nm)'),
                mpatches.Patch(color='blue'  , label='SPN-S (532nm)')
                ]
    ax1.legend(handles=patches_legend, bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=len(patches_legend), mode="expand", borderaxespad=0., frameon=False, handletextpad=0.2, fontsize=16)

    ax1.set_xlabel('Time [Hour]')
    ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
    plt.savefig('%s_532nm.png' % date_s, bbox_inches='tight')
    plt.close(fig)
    # ---------------------------------------------------------------------


    fname_ssfr = '%s/ssfr_%s_hsk.h5' % (fdir_processed, date_s)
    f = h5py.File(fname_ssfr, 'w')

    dset = f.create_dataset('comments', data=comments)
    dset.attrs['description'] = 'comments on the data'

    dset = f.create_dataset('info', data=version_info)
    dset.attrs['description'] = 'information on the version'

    dset = f.create_dataset('utc', data=data_hsk['tmhr'])
    dset.attrs['description'] = 'universal time (numbers above 24 are for the next day)'
    dset.attrs['unit'] = 'decimal hour'

    dset = f.create_dataset('altitude', data=data_hsk['gps_altitude'])
    dset.attrs['description'] = 'altitude above sea level (GPS altitude)'
    dset.attrs['unit'] = 'meter'

    dset = f.create_dataset('longitude', data=data_hsk['longitude'])
    dset.attrs['description'] = 'longitude'
    dset.attrs['unit'] = 'degree'

    dset = f.create_dataset('latitude', data=data_hsk['latitude'])
    dset.attrs['description'] = 'latitude'
    dset.attrs['unit'] = 'degree'

    dset = f.create_dataset('zen_wvl', data=ssfr_v0.zen_wvl)
    dset.attrs['description'] = 'center wavelengths of zenith channels (bandwidth see info)'
    dset.attrs['unit'] = 'nm'

    dset = f.create_dataset('nad_wvl', data=ssfr_v0.nad_wvl)
    dset.attrs['description'] = 'center wavelengths of nadir channels (bandwidth see info)'
    dset.attrs['unit'] = 'nm'

    dset = f.create_dataset('zen_flux', data=zen_flux)
    dset.attrs['description'] = 'downwelling shortwave spectral irradiance'
    dset.attrs['unit'] = 'W / m2 / nm'

    dset = f.create_dataset('nad_flux', data=nad_flux)
    dset.attrs['description'] = 'upwelling shortwave spectral irradiance'
    dset.attrs['unit'] = 'W / m2 / nm'

    dset = f.create_dataset('pitch', data=pitch)
    dset.attrs['description'] = 'aircraft pitch angle (positive values indicate nose up)'
    dset.attrs['unit'] = 'degree'

    dset = f.create_dataset('roll', data=roll)
    dset.attrs['description'] = 'aircraft roll angle (positive values indicate right wing down)'
    dset.attrs['unit'] = 'degree'

    dset = f.create_dataset('heading', data=heading)
    dset.attrs['description'] = 'aircraft heading angle (positive values clockwise, w.r.t north)'
    dset.attrs['unit'] = 'degree'

    dset = f.create_dataset('sza', data=sza)
    dset.attrs['description'] = 'solar zenith angle'
    dset.attrs['unit'] = 'degree'

    f.close()

    return fname_ssfr
    # ============================================================================

def cdata_ssfr_ict(date,
              fdir_processed,
              wvls = np.array([415, 440, 500, 550, 675, 870, 990, 1020, 1064, 1250, 1650, 2100]),
              version_info = 'R0: initial science data release'
              ):

    date_s = date.strftime('%Y%m%d')

    fname_ssfr = get_file(fdir_processed, full=True, contains=['ssfr_%s_hsk' % date_s])
    data0  = load_h5(fname_ssfr)

    vars_list = ['Time_start'] +\
                ['DN%d' % wvl for wvl in wvls] +\
                ['UP%d' % wvl for wvl in wvls] +\
                ['LON', 'LAT', 'ALT', 'PITCH', 'ROLL', 'HEADING', 'SZA']

    data = OrderedDict()
    for var in vars_list:
        data[var] = {'data':None, 'unit':None, 'description':None}

    for var in data.keys():
        if 'time' in var.lower() or 'utc' in var.lower():
            data[var]['unit'] = 'second'
            data[var]['description'] = '%s, %s, %s' % (var, data[var]['unit'], var)
            data[var]['data'] = data0['utc']*3600.0
        elif 'dn' in var.lower():
            index= np.argmin(np.abs(data0['zen_wvl']-float(var.replace('DN', ''))))
            data[var]['unit'] = 'W m^-2 nm^-1'
            data[var]['description'] = '%s, %s, Rad_IrradianceDownwellingGlobal_InSitu_SC, Downward Shortwave Irradiance at %.4f nm' % (var, data[var]['unit'], data0['zen_wvl'][index])
            data[var]['data'] = data0['zen_flux'][:, index]
            data[var]['data'][data[var]['data']<0.0] = np.nan
        elif 'up' in var.lower():
            index= np.argmin(np.abs(data0['nad_wvl']-float(var.replace('UP', ''))))
            data[var]['unit'] = 'W m^-2 nm^-1'
            data[var]['description'] = '%s, %s, Rad_IrradianceUpwelling_Insitu_SC, Upward Shortwave Irradiance at %.4f nm' % (var, data[var]['unit'], data0['nad_wvl'][index])
            data[var]['data'] = data0['nad_flux'][:, index]
            data[var]['data'][data[var]['data']<0.0] = np.nan
        elif 'lon' in var.lower():
            data[var]['unit'] = 'degree'
            data[var]['description'] = '%s, %s, Platform_Longitude_InSitu_None, Longitude' % (var, data[var]['unit'])
            data[var]['data'] = data0['longitude']
        elif 'lat' in var.lower():
            data[var]['unit'] = 'degree'
            data[var]['description'] = '%s, %s, Platform_Latitude_InSitu_None, Latitude' % (var, data[var]['unit'])
            data[var]['data'] = data0['latitude']
        elif 'alt' in var.lower():
            data[var]['unit'] = 'meter'
            data[var]['description'] = '%s, %s, Platform_AltitudeMSL_InSitu_None, GPS Altitude' % (var, data[var]['unit'])
            data[var]['data'] = data0['altitude']
        elif 'pitch' in var.lower():
            data[var]['unit'] = 'degree'
            data[var]['description'] = '%s, %s, Platform_PitchAngle_InSitu_None, Aircraft Pitch Angle (positive values indicate nose up)' % (var, data[var]['unit'])
            data[var]['data'] = data0['pitch']
        elif 'roll' in var.lower():
            data[var]['unit'] = 'degree'
            data[var]['description'] = '%s, %s, Platform_RollAngle_InSitu_None, Aircraft Roll Angle (positive values indicate right wing down)' % (var, data[var]['unit'])
            data[var]['data'] = data0['roll']
        elif 'heading' in var.lower():
            data[var]['unit'] = 'degree'
            data[var]['description'] = '%s, %s, Platform_HeadingTrue_InSitu_None, Aircraft Heading Angle (positive values clockwise, w.r.t. north)' % (var, data[var]['unit'])
            data[var]['data'] = data0['heading']
        elif 'sza' in var.lower():
            data[var]['unit'] = 'degree'
            data[var]['description'] = '%s, %s, Met_SolarZenithAngle_Insitu_None, Solar Zenith Angle (w.r.t. Earth)' % (var, data[var]['unit'])
            data[var]['data'] = data0['sza']


    version_info_list = version_info.split(':')
    comments = {
            'PI_CONTACT_INFO': 'Address: University of Colorado Boulder, LASP, 3665 Discovery Drive, Boulder, CO 80303; E-mail: Sebastian.Schmidt@lasp.colorado.edu; Phone: (303)492-6423',
            'PLATFORM': 'P3',
            'LOCATION': 'N/A',
            'ASSOCIATED_DATA': 'N/A',
            'INSTRUMENT_INFO': 'SSFR (Solar Spectral Flux Radiometer, 350-2150 nm)',
            'DATA_INFO': 'Reported are only selected wavelengths (SSFR), pitch/roll from leveling platform INS or aircraft, lat/lon/alt/heading from aircraft, sza calculated from time/lon/lat.',
            'UNCERTAINTY': 'Nominal SSFR uncertainty (shortwave): nadir: 5% zenith: 7%',
            'ULOD_FLAG': '-7777',
            'ULOD_VALUE': 'N/A',
            'LLOD_FLAG': '-8888',
            'LLOD_VALUE': 'N/A',
            'DM_CONTACT_INFO': 'N/A',
            'PROJECT_INFO': 'CAMP2Ex experiment out of Philippines, August - October 2019',
            'STIPULATIONS_ON_USE': 'This is the initial release of the CAMP2Ex-2019 field dataset. We strongly recommend that you consult the PI, both for updates to the data set, and for the proper and most recent interpretation of the data for specific science use.',
            'OTHER_COMMENTS': 'The full SSFR spectra from 350-2150 nm is also available in the data archive.\n%s' % str(data0['comments']),
            'REVISION': version_info_list[0],
            version_info_list[0]: version_info_list[1]
            }

    fname = 'CAMP2EX-SSFR-Partial_P3B_%s_%s.ict' % (date_s, version_info_list[0])
    fname = write_ict(date, data, fname, comments=comments, mission_info='CAMP2Ex 2019')
    return fname
    # ============================================================================

def cdata_main(fdir, fdir_archive='/argus/field/camp2ex/2019/p3/archive'):

    date_s    = os.path.basename(fdir).split('_')[0]
    date      = datetime.datetime.strptime(date_s, '%Y%m%d')

    fdir_processed = '%s/processed' % fdir
    if not os.path.exists(fdir_processed):
        os.makedirs(fdir_processed)

    # fname_hsk_h5   = cdata_hsk_h5(date, fdir_processed, fdir_data=fdir_archive, run=True)
    # fname_spns_v0  = cdata_spns_v0(date, fdir_processed, fdir_data=fdir, run=True)
    # fname_spns_hsk = cdata_spns_hsk(date, fdir_processed, fdir_data=fdir, run=True)
    # if date < datetime.datetime(2019, 8, 24):
    #     fname_alp_v0   = cdata_alp_v0(date, fdir_processed, fdir_data=fdir, run=True)
    # fname_ssfr_v0  = cdata_ssfr_v0(date, fdir_processed, fdir_data=fdir, run=True)
    # fname_ssfr_aux = cdata_ssfr_aux(date, fdir_processed)

    fdir_cal = '/argus/field/camp2ex/2019/p3/calibration'
    version_info = 'R0: initial science data release'

    # fname_ssfr_hsk = cdata_ssfr_hsk(date, fdir_processed, fdir_cal, version_info=version_info)
    # command = 'cp "%s" "CAMP2EX-SSFR_P3B_%s_%s.h5"' % (fname_ssfr_hsk, date_s, version_info.split(':')[0])
    # os.system(command)

    fname_ict = cdata_ssfr_ict(date, fdir_processed, version_info=version_info)
# ================================================================================================




# functions for processing SPN-S
#/----------------------------------------------------------------------------\#
def cdata_arcsix_hsk_v0(
        date,
        tmhr_range=[14.0, 24.0],
        fdir_data=_fdir_hsk_,
        fdir_out=_fdir_v0_,
        ):

    """
    Usually this function is for processing aricraft housekeeping file in the field (also known as cabin file).

    Now for skywatch testing, we will use a fixed longitude and latitude
    """

    # create data_hsk for skywatch
    #/----------------------------------------------------------------------------\#
    tmhr = np.arange(tmhr_range[0]*3600.0, tmhr_range[-1]*3600.0, 1.0)/3600.0
    lon0 = -105.24227862207863 # skywatch longitude
    lat0 =  40.01097849056196  # skywatch latitude
    alt0 =  4.0                # skywatch altitude
    pit0 = 0.0
    rol0 = 0.0
    hed0 = 0.0
    data_hsk = {
            'tmhr': {'data': tmhr, 'units': 'hour'},
            'long': {'data': np.repeat(lon0, tmhr.size), 'units': 'degree'},
            'lat' : {'data': np.repeat(lat0, tmhr.size), 'units': 'degree'},
            'palt': {'data': np.repeat(alt0, tmhr.size), 'units': 'meter'},
            'pitch'   : {'data': np.repeat(pit0, tmhr.size), 'units': 'degree'},
            'roll'    : {'data': np.repeat(rol0, tmhr.size), 'units': 'degree'},
            'heading' : {'data': np.repeat(hed0, tmhr.size), 'units': 'degree'},
            }
    #\----------------------------------------------------------------------------/#


    # solar geometries
    #/----------------------------------------------------------------------------\#
    jday0 = ssfr.util.dtime_to_jday(date)
    jday = jday0 + data_hsk['tmhr']['data']/24.0
    sza, saa = ssfr.util.cal_solar_angles(jday, data_hsk['long']['data'], data_hsk['lat']['data'], data_hsk['palt']['data'])
    #\----------------------------------------------------------------------------/#

    # save processed data
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/%s_HSK_%s_v0.h5' % (fdir_out, _mission_.upper(), date.strftime('%Y-%m-%d'))

    f = h5py.File(fname_h5, 'w')
    f['tmhr'] = data_hsk['tmhr']['data']
    f['lon']  = data_hsk['long']['data']
    f['lat']  = data_hsk['lat']['data']
    f['alt']  = data_hsk['palt']['data']
    f['pit']  = data_hsk['pitch']['data']
    f['rol']  = data_hsk['roll']['data']
    f['hed']  = data_hsk['heading']['data']
    f['jday'] = jday
    f['sza']  = sza
    f['saa']  = saa
    f.close()
    #\----------------------------------------------------------------------------/#

    return

def cdata_arcsix_spns_v0(
        date,
        fdir_data=_fdir_spns_,
        fdir_out=_fdir_v0_,
        ):

    """
    Process raw SPN-S data
    """

    # read spn-s raw data
    #/----------------------------------------------------------------------------\#
    fdir = '%s/%s' % (fdir_data, date.strftime('%Y-%m-%d'))

    fname_dif = sorted(glob.glob('%s/Diffuse.txt' % fdir))[0]
    data0_dif = ssfr.lasp_spn.read_spns(fname=fname_dif)

    fname_tot = sorted(glob.glob('%s/Total.txt' % fdir))[0]
    data0_tot = ssfr.lasp_spn.read_spns(fname=fname_tot)
    #/----------------------------------------------------------------------------\#

    # read wavelengths and calculate toa downwelling solar flux
    #/----------------------------------------------------------------------------\#
    flux_toa = ssfr.util.get_solar_kurudz()

    wvl_tot = data0_tot.data['wavelength']
    f_dn_sol_tot = np.zeros_like(wvl_tot)
    for i, wvl0 in enumerate(wvl_tot):
        f_dn_sol_tot[i] = ssfr.util.cal_solar_flux_toa(wvl0, flux_toa[:, 0], flux_toa[:, 1])
    #\----------------------------------------------------------------------------/#

    # save processed data
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/%s_%s_%s_v0.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), date.strftime('%Y-%m-%d'))

    f = h5py.File(fname_h5, 'w')

    g1 = f.create_group('dif')
    g1['tmhr']  = data0_dif.data['tmhr']
    g1['wvl']   = data0_dif.data['wavelength']
    g1['flux']  = data0_dif.data['flux']

    g2 = f.create_group('tot')
    g2['tmhr']  = data0_tot.data['tmhr']
    g2['wvl']   = data0_tot.data['wavelength']
    g2['flux']  = data0_tot.data['flux']
    g2['toa0']  = f_dn_sol_tot

    f.close()
    #\----------------------------------------------------------------------------/#

    return

def cdata_arcsix_spns_v1(
        date,
        time_offset=0.0,
        fdir_data=_fdir_v0_,
        fdir_out=_fdir_v1_,
        ):

    """
    Check for time offset and merge SPN-S data with aircraft data
    """

    # read hsk v0
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/%s_HSK_%s_v0.h5' % (fdir_data, _mission_.upper(), date.strftime('%Y-%m-%d'))
    f = h5py.File(fname_h5, 'r')
    jday = f['jday'][...]
    sza  = f['sza'][...]
    saa  = f['saa'][...]
    tmhr = f['tmhr'][...]
    lon  = f['lon'][...]
    lat  = f['lat'][...]
    alt  = f['alt'][...]
    pit  = f['pit'][...]
    rol  = f['rol'][...]
    hed  = f['hed'][...]
    f.close()
    #\----------------------------------------------------------------------------/#


    # read spn-s v0
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/%s_%s_%s_v0.h5' % (fdir_data, _mission_.upper(), _spns_.upper(), date.strftime('%Y-%m-%d'))
    f = h5py.File(fname_h5, 'r')
    f_dn_dif  = f['dif/flux'][...]
    wvl_dif   = f['dif/wvl'][...]
    tmhr_dif  = f['dif/tmhr'][...]

    f_dn_tot  = f['tot/flux'][...]
    wvl_tot   = f['tot/wvl'][...]
    tmhr_tot  = f['tot/tmhr'][...]
    f_dn_tot_toa0 = f['tot/toa0'][...]
    f.close()
    #/----------------------------------------------------------------------------\#


    # interpolate spn-s data to hsk time frame
    #/----------------------------------------------------------------------------\#
    flux_dif = np.zeros((tmhr.size, wvl_dif.size), dtype=np.float64)
    for i in range(wvl_dif.size):
        flux_dif[:, i] = ssfr.util.interp(tmhr, tmhr_dif, f_dn_dif[:, i])

    flux_tot = np.zeros((tmhr.size, wvl_tot.size), dtype=np.float64)
    for i in range(wvl_tot.size):
        flux_tot[:, i] = ssfr.util.interp(tmhr, tmhr_tot, f_dn_tot[:, i])
    #\----------------------------------------------------------------------------/#


    # save processed data
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/%s_%s_%s_v1.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), date.strftime('%Y-%m-%d'))

    f = h5py.File(fname_h5, 'w')

    f['jday'] = jday
    f['tmhr'] = tmhr
    f['lon']  = lon
    f['lat']  = lat
    f['alt']  = alt
    f['sza']  = sza
    f['saa']  = saa
    f['pit']  = pit
    f['rol']  = rol
    f['hed']  = hed

    g1 = f.create_group('dif')
    g1['wvl']   = wvl_dif
    g1['flux']  = flux_dif

    g2 = f.create_group('tot')
    g2['wvl']   = wvl_tot
    g2['flux']  = flux_tot
    g2['toa0']  = f_dn_tot_toa0

    f.close()
    #\----------------------------------------------------------------------------/#

    return

def cdata_arcsix_spns_v2(
        date,
        time_offset=0.0,
        fdir_data=_fdir_v1_,
        fdir_out=_fdir_v2_,
        ):

    """
    Apply attitude correction to account for pitch and roll
    """

    # read spn-s v1
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/%s_%s_%s_v1.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), date.strftime('%Y-%m-%d'))
    f = h5py.File(fname_h5, 'r')
    f_dn_dif  = f['dif/flux'][...]
    wvl_dif   = f['dif/wvl'][...]

    f_dn_tot  = f['tot/flux'][...]
    wvl_tot   = f['tot/wvl'][...]
    f_dn_toa0 = f['tot/toa0'][...]

    jday = f['jday'][...]
    tmhr = f['tmhr'][...]
    lon = f['lon'][...]
    lat = f['lat'][...]
    alt = f['alt'][...]
    sza  = f['sza'][...]
    saa  = f['saa'][...]

    pit = f['pit'][...]
    rol = f['rol'][...]
    hed = f['hed'][...]

    f.close()
    #/----------------------------------------------------------------------------\#


    # correction factor
    #/----------------------------------------------------------------------------\#
    mu = np.cos(np.deg2rad(sza))

    iza, iaa = ssfr.util.prh2za(pit, rol, hed)
    dc = ssfr.util.muslope(sza, saa, iza, iaa)

    factors = mu / dc
    #\----------------------------------------------------------------------------/#


    # attitude correction
    #/----------------------------------------------------------------------------\#
    f_dn_dir = f_dn_tot - f_dn_dif
    f_dn_dir_corr = np.zeros_like(f_dn_dir)
    f_dn_tot_corr = np.zeros_like(f_dn_tot)
    for iwvl in range(wvl_tot.size):
        f_dn_dir_corr[..., iwvl] = f_dn_dir[..., iwvl]*factors
        f_dn_tot_corr[..., iwvl] = f_dn_dir_corr[..., iwvl] + f_dn_dif[..., iwvl]
    #\----------------------------------------------------------------------------/#


    # save processed data
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/%s_%s_%s_v2.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), date.strftime('%Y-%m-%d'))

    f = h5py.File(fname_h5, 'w')

    f['jday'] = jday
    f['tmhr'] = tmhr
    f['lon']  = lon
    f['lat']  = lat
    f['alt']  = alt
    f['sza']  = sza
    f['dc']   = dc

    g1 = f.create_group('dif')
    g1['wvl']   = wvl_dif
    g1['flux']  = f_dn_dif

    g2 = f.create_group('tot')
    g2['wvl']   = wvl_tot
    g2['flux']  = f_dn_tot_corr
    g2['toa0']  = f_dn_toa0

    f.close()
    #\----------------------------------------------------------------------------/#

    ssfr.vis.quicklook_bokeh_spns(fname_h5, wvl0=None, tmhr0=None, tmhr_range=None, wvl_range=[350.0, 800.0], tmhr_step=10, wvl_step=5, description=_mission_.upper(), fname_html='%s_ql_%s_v2.html' % (_spns_, date.strftime('%Y-%m-%d')))

    return

def process_spns(date):

    cdata_arcsix_hsk_v0(date)
    cdata_arcsix_spns_v0(date)
    cdata_arcsix_spns_v1(date)
    cdata_arcsix_spns_v2(date)
#\----------------------------------------------------------------------------/#




def test_lasp_ssfr_skywatch_zenith():

    fdir = 'data/SSFR/2023-10-06'

    fnames = ssfr.util.get_all_files(fdir)

    ssfr0 = ssfr.lasp_ssfr.read_ssfr(fnames, dark_corr_mode='interp')


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.scatter(ssfr0.dset0['tmhr'], ssfr0.dset0['spectra'][:, 100, 0], s=6, c='black', lw=0.0)
        ax1.scatter(ssfr0.dset1['tmhr'], ssfr0.dset1['spectra'][:, 100, 0], s=6, c='gray' , lw=0.0)

        ax1.scatter(ssfr0.dset0['tmhr'], ssfr0.dset0['spectra'][:, 100, 1], s=6, c='red'    , lw=0.0)
        ax1.scatter(ssfr0.dset1['tmhr'], ssfr0.dset1['spectra'][:, 100, 1], s=6, c='magenta', lw=0.0)

        ax1.set_ylim((-35000, 35000))

        ax1.set_xlabel('Time [Hour]')
        ax1.set_ylabel('Counts')
        ax1.set_title('Zenith (LC6 + CAMP2Ex Zenith Cable)')
        #\--------------------------------------------------------------/#

        patches_legend = [
                         mpatches.Patch(color='black'   , label='Zenith Silicon (%d ms)' % ssfr0.dset0['info']['int_time']['zen_si']), \
                         mpatches.Patch(color='gray'    , label='Zenith Silicon (%d ms)' % ssfr0.dset1['info']['int_time']['zen_si']), \
                         mpatches.Patch(color='red'     , label='Zenith InGaAs (%d ms)' % ssfr0.dset0['info']['int_time']['zen_in']), \
                         mpatches.Patch(color='magenta' , label='Zenith InGaAs (%d ms)' % ssfr0.dset1['info']['int_time']['zen_in']), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper left', fontsize=16)

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % (_metadata['Function']), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
    #\----------------------------------------------------------------------------/#

def test_lasp_ssfr_skywatch_nadir():

    fdir = 'data/SSFR/2023-10-06'

    fnames = ssfr.util.get_all_files(fdir)

    ssfr0 = ssfr.lasp_ssfr.read_ssfr(fnames, dark_corr_mode='interp')


    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.scatter(ssfr0.dset0['tmhr'], ssfr0.dset0['spectra'][:, 100, 2], s=6, c='black', lw=0.0)
        ax1.scatter(ssfr0.dset1['tmhr'], ssfr0.dset1['spectra'][:, 100, 2], s=6, c='gray', lw=0.0)

        ax1.scatter(ssfr0.dset0['tmhr'], ssfr0.dset0['spectra'][:, 100, 3], s=6, c='red'    , lw=0.0)
        ax1.scatter(ssfr0.dset1['tmhr'], ssfr0.dset1['spectra'][:, 100, 3], s=6, c='magenta', lw=0.0)

        ax1.set_ylim((-35000, 35000))

        ax1.set_xlabel('Time [Hour]')
        ax1.set_ylabel('Counts')
        ax1.set_title('Nadir (LC4 + Stainless Steel Cable)')
        #\--------------------------------------------------------------/#

        patches_legend = [
                         mpatches.Patch(color='black', label='Nadir Silicon (%d ms)' % ssfr0.dset0['info']['int_time']['Nadir Silicon']), \
                         mpatches.Patch(color='gray' , label='Nadir Silicon (%d ms)' % ssfr0.dset1['info']['int_time']['Nadir Silicon']), \
                         mpatches.Patch(color='red'     , label='Nadir InGaAs (%d ms)' % ssfr0.dset0['info']['int_time']['Nadir InGaAs']), \
                         mpatches.Patch(color='magenta' , label='Nadir InGaAs (%d ms)' % ssfr0.dset1['info']['int_time']['Nadir InGaAs']), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper left', fontsize=16)

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % (_metadata['Function']), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
    #\----------------------------------------------------------------------------/#





# functions for processing SSFR
#/----------------------------------------------------------------------------\#
def cdata_arcsix_ssfr_v0(
        date,
        fdir_data=_fdir_ssfr_,
        fdir_out=_fdir_v0_
        ):

    """
    version 0: counts after dark correction
    """

    date_s = date.strftime('%Y-%m-%d')
    fnames = sorted(glob.glob('%s/%s/*.SKS' % (fdir_data, date_s)))

    ssfr0 = ssfr.lasp_ssfr.read_ssfr(fnames, dark_corr_mode='interp')

    # data that are useful
    #   wvl_zen
    #   cnt_zen
    #   wvl_nad
    #   cnt_nad
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/%s_%s_%s_v0.h5' % (fdir_out, _mission_.upper(), _ssfr_.upper(), date_s)
    f = h5py.File(fname_h5, 'w')

    for i in range(ssfr0.Ndset):
        dset_s = 'dset%d' % i
        data = getattr(ssfr0, dset_s)
        g = f.create_group(dset_s)
        for key in data.keys():
            if key != 'info':
                g[key] = data[key]

    f.close()
    #\----------------------------------------------------------------------------/#

    return

def process_ssfr(date):

    cdata_arcsix_ssfr_v0(date)
#\----------------------------------------------------------------------------/#



if __name__ == '__main__':

    dates = [
             # datetime.datetime(2023, 10, 11), # ssfr-a, lab diagnose
             # datetime.datetime(2023, 10, 12), # ssfr-b, skywatch test
             datetime.datetime(2023, 10, 13), # ssfr-b, skywatch test
            ]

    for date in dates:
        process_ssfr(date)
