import os
import sys
import glob
import datetime
import h5py
import numpy as np
from scipy import interpolate
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
_fdir_v0_   = 'data/processed'
_fdir_v1_   = 'data/processed'
_fdir_v2_   = 'data/processed'



# functions for processing SPNS
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

    date_s = date.strftime('%Y-%m-%d')

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
    fname_h5 = '%s/%s_HSK_%s_v0.h5' % (fdir_out, _mission_.upper(), date_s)

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

    date_s = date.strftime('%Y-%m-%d')

    # read spn-s raw data
    #/----------------------------------------------------------------------------\#
    fdir = '%s/%s' % (fdir_data, date_s)

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
    fname_h5 = '%s/%s_%s_%s_v0.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), date_s)

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

    date_s = date.strftime('%Y-%m-%d')

    # read hsk v0
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/%s_HSK_%s_v0.h5' % (fdir_data, _mission_.upper(), date_s)
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
    fname_h5 = '%s/%s_%s_%s_v0.h5' % (fdir_data, _mission_.upper(), _spns_.upper(), date_s)
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
    fname_h5 = '%s/%s_%s_%s_v1.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), date_s)

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

    date_s = date.strftime('%Y-%m-%d')

    # read spn-s v1
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/%s_%s_%s_v1.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), date_s)
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
    fname_h5 = '%s/%s_%s_%s_v2.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), date_s)

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

    ssfr.vis.quicklook_bokeh_spns(fname_h5, wvl0=None, tmhr0=None, tmhr_range=None, wvl_range=[350.0, 800.0], tmhr_step=10, wvl_step=5, description=_mission_.upper(), fname_html='%s_ql_%s_v2.html' % (_spns_, date_s))

    return

def process_spns_data(date):

    cdata_arcsix_hsk_v0(date)
    cdata_arcsix_spns_v0(date)
    cdata_arcsix_spns_v1(date)
    cdata_arcsix_spns_v2(date)
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
    #   wvl_zen [nm]
    #   cnt_zen [counts/ms]
    #   wvl_nad [nm]
    #   cnt_nad [counts/ms]
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

def cdata_arcsix_ssfr_v1(
        date,
        tmhr_offset=4.3470,
        fdir_data=_fdir_v0_,
        fdir_out=_fdir_v1_,
        ):

    """
    version 1: 1) check for time offset and merge SSFR data with aircraft housekeeping data
               2) interpolate raw SSFR data into the time frame of the housekeeping data
    """

    date_s = date.strftime('%Y-%m-%d')

    # read hsk v0
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/%s_HSK_%s_v0.h5' % (fdir_data, _mission_.upper(), date_s)
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

    # save processed data
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/%s_%s_%s_v1.h5' % (fdir_out, _mission_.upper(), _ssfr_.upper(), date_s)
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

    # read ssfr v0
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/%s_%s_%s_v0.h5' % (fdir_data, _mission_.upper(), _ssfr_.upper(), date_s)
    f_ = h5py.File(fname_h5, 'r')

    for dset_s in f_.keys():

        cnt_zen_ = f_['%s/cnt_zen' % dset_s][...]
        wvl_zen  = f_['%s/wvl_zen' % dset_s][...]
        tmhr_zen = f_['%s/tmhr'    % dset_s][...] + tmhr_offset

        cnt_nad_ = f_['%s/cnt_nad' % dset_s][...]
        wvl_nad  = f_['%s/wvl_nad' % dset_s][...]
        tmhr_nad = f_['%s/tmhr'    % dset_s][...] + tmhr_offset

        # interpolate ssfr data to hsk time frame
        #/----------------------------------------------------------------------------\#
        cnt_zen = np.zeros((tmhr.size, wvl_zen.size), dtype=np.float64)
        for i in range(wvl_zen.size):
            cnt_zen[:, i] = ssfr.util.interp(tmhr, tmhr_zen, cnt_zen_[:, i])

        cnt_nad = np.zeros((tmhr.size, wvl_nad.size), dtype=np.float64)
        for i in range(wvl_nad.size):
            cnt_nad[:, i] = ssfr.util.interp(tmhr, tmhr_nad, cnt_nad_[:, i])
        #\----------------------------------------------------------------------------/#

        g = f.create_group(dset_s)

        g['wvl_zen'] = wvl_zen
        g['cnt_zen'] = cnt_zen
        g['wvl_nad'] = wvl_nad
        g['cnt_nad'] = cnt_nad

    f_.close()
    #/----------------------------------------------------------------------------\#

    f.close()
    #\----------------------------------------------------------------------------/#

    return

def cdata_arcsix_ssfr_v2(
        date,
        fdir_data=_fdir_v1_,
        fdir_out=_fdir_v2_,
        pitch_angle=0.0,
        roll_angle=0.0,
        ):

    """
    version 1: 1) apply radiometric response to convert counts to irradiance [secondary response]
               2) apply cosine correction to correct for non-linear angular resposne (incomplete)
    """

    date_s = date.strftime('%Y-%m-%d')

    # primary transfer calibration
    #/----------------------------------------------------------------------------\#
    fname_resp_zen = '/argus/field/camp2ex/2019/p3/calibration/rad-cal/20191125-post_20191125-field*_zenith-LC1_rad-resp_s060i300.h5'
    f = h5py.File(fname_resp_zen, 'r')
    wvl_resp_zen_ = f['wvl'][...]
    pri_resp_zen_ = f['pri_resp'][...]
    transfer_zen_ = f['transfer'][...]
    sec_resp_zen_ = f['sec_resp'][...]
    f.close()

    fname_resp_nad = '/argus/field/camp2ex/2019/p3/calibration/rad-cal/20191125-post_20191125-field*_nadir-LC2_rad-resp_s060i300.h5'
    f = h5py.File(fname_resp_nad, 'r')
    wvl_resp_nad_ = f['wvl'][...]
    pri_resp_nad_ = f['pri_resp'][...]
    transfer_nad_ = f['transfer'][...]
    sec_resp_nad_ = f['sec_resp'][...]
    f.close()
    #\----------------------------------------------------------------------------/#

    fname_h5 = '%s/%s_%s_%s_v2.h5' % (fdir_out, _mission_.upper(), _ssfr_.upper(), date_s)
    f = h5py.File(fname_h5, 'w')

    fname_h5 = '%s/%s_%s_%s_v1.h5' % (fdir_data, _mission_.upper(), _ssfr_.upper(), date_s)
    f_ = h5py.File(fname_h5, 'r')
    tmhr = f_['tmhr'][...]
    for dset_s in f_.keys():

        if 'dset' in dset_s:

            # zenith
            #/--------------------------------------------------------------\#
            cnt_zen = f_['%s/cnt_zen' % dset_s][...]
            wvl_zen = f_['%s/wvl_zen' % dset_s][...]

            pri_resp_zen = np.interp(wvl_zen, wvl_resp_zen_, pri_resp_zen_)
            transfer_zen = np.interp(wvl_zen, wvl_resp_zen_, transfer_zen_)
            sec_resp_zen = np.interp(wvl_zen, wvl_resp_zen_, sec_resp_zen_)

            flux_zen = cnt_zen.copy()
            for i in range(tmhr.size):
                if np.isnan(cnt_zen[i, :]).sum() == 0:
                    flux_zen[i, :] = cnt_zen[i, :] / sec_resp_zen
            #\--------------------------------------------------------------/#

            # nadir
            #/--------------------------------------------------------------\#
            cnt_nad = f_['%s/cnt_nad' % dset_s][...]
            wvl_nad = f_['%s/wvl_nad' % dset_s][...]

            pri_resp_nad = np.interp(wvl_nad, wvl_resp_nad_, pri_resp_nad_)
            transfer_nad = np.interp(wvl_nad, wvl_resp_nad_, transfer_nad_)
            sec_resp_nad = np.interp(wvl_nad, wvl_resp_nad_, sec_resp_nad_)

            flux_nad = cnt_nad.copy()
            for i in range(tmhr.size):
                if np.isnan(cnt_nad[i, :]).sum() == 0:
                    flux_nad[i, :] = cnt_nad[i, :] / sec_resp_nad
            #\--------------------------------------------------------------/#

            g = f.create_group(dset_s)
            g['flux_zen'] = flux_zen
            g['flux_nad'] = flux_nad
            g['wvl_zen']  = wvl_zen
            g['wvl_nad']  = wvl_nad

        else:

            f[dset_s] = f_[dset_s][...]

    f_.close()

    f.close()

    # calculate cosine correction factors
    #/----------------------------------------------------------------------------\#
    # angles = {}
    # angles['solar_zenith']  = ssfr_aux['sza']
    # angles['solar_azimuth'] = ssfr_aux['saa']
    # if date < datetime.datetime(2019, 8, 24):
    #     fname_alp = get_file(fdir_processed, full=True, contains=['alp_%s_v0' % date_s])
    #     data_alp = load_h5(fname_alp)
    #     angles['pitch']        = interp(ssfr_v0.tmhr, data_alp['tmhr'], data_alp['ang_pit_s'])
    #     angles['roll']         = interp(ssfr_v0.tmhr, data_alp['tmhr'], data_alp['ang_rol_s'])
    #     angles['heading']      = interp(ssfr_v0.tmhr, data_hsk['tmhr'], data_hsk['true_heading'])
    #     angles['pitch_motor']  = interp(ssfr_v0.tmhr, data_alp['tmhr'], data_alp['ang_pit_m'])
    #     angles['roll_motor']   = interp(ssfr_v0.tmhr, data_alp['tmhr'], data_alp['ang_rol_m'])
    #     angles['pitch_motor'][np.isnan(angles['pitch_motor'])] = 0.0
    #     angles['roll_motor'][np.isnan(angles['roll_motor'])]   = 0.0
    #     angles['pitch_offset']  = pitch_angle
    #     angles['roll_offset']   = roll_angle
    # else:
    #     angles['pitch']         = interp(ssfr_v0.tmhr, data_hsk['tmhr'], data_hsk['pitch_angle'])
    #     angles['roll']          = interp(ssfr_v0.tmhr, data_hsk['tmhr'], data_hsk['roll_angle'])
    #     angles['heading']       = interp(ssfr_v0.tmhr, data_hsk['tmhr'], data_hsk['true_heading'])
    #     angles['pitch_motor']   = np.repeat(0.0, ssfr_v0.tmhr.size)
    #     angles['roll_motor']    = np.repeat(0.0, ssfr_v0.tmhr.size)
    #     angles['pitch_offset']  = pitch_angle
    #     angles['roll_offset']   = roll_angle

    # fdir_ang_cal = '%s/ang-cal' % fdir_cal
    # fnames_ang_cal = get_ang_cal_camp2ex(date, fdir_ang_cal)
    # factors = cos_corr(fnames_ang_cal, angles, diff_ratio=ssfr_aux['diff_ratio'])

    # # apply cosine correction
    # ssfr_v0.zen_cnt = ssfr_v0.zen_cnt*factors['zenith']
    # ssfr_v0.nad_cnt = ssfr_v0.nad_cnt*factors['nadir']
    #\----------------------------------------------------------------------------/#

    return

def cdata_arcsix_ssfr_hsk():

    # header
    #/----------------------------------------------------------------------------\#
    # comments_list = []
    # comments_list.append('Bandwidth of Silicon channels (wavelength < 950nm) as defined by the FWHM: 6 nm')
    # comments_list.append('Bandwidth of InGaAs channels (wavelength > 950nm) as defined by the FWHM: 12 nm')
    # comments_list.append('Pitch angle offset: %.1f degree' % pitch_angle)
    # comments_list.append('Roll angle offset: %.1f degree' % roll_angle)

    # for key in fnames_rad_cal.keys():
    #     comments_list.append('Radiometric calibration file (%s): %s' % (key, os.path.basename(fnames_rad_cal[key])))
    # for key in fnames_ang_cal.keys():
    #     comments_list.append('Angular calibration file (%s): %s' % (key, os.path.basename(fnames_ang_cal[key])))
    # comments = '\n'.join(comments_list)

    # print(date_s)
    # print(comments)
    # print()
    #\----------------------------------------------------------------------------/#

    # create hsk file for ssfr (nasa data archive)
    #/----------------------------------------------------------------------------\#
    # fname_ssfr = '%s/ssfr_%s_hsk.h5' % (fdir_processed, date_s)
    # f = h5py.File(fname_ssfr, 'w')

    # dset = f.create_dataset('comments', data=comments)
    # dset.attrs['description'] = 'comments on the data'

    # dset = f.create_dataset('info', data=version_info)
    # dset.attrs['description'] = 'information on the version'

    # dset = f.create_dataset('utc', data=data_hsk['tmhr'])
    # dset.attrs['description'] = 'universal time (numbers above 24 are for the next day)'
    # dset.attrs['unit'] = 'decimal hour'

    # dset = f.create_dataset('altitude', data=data_hsk['gps_altitude'])
    # dset.attrs['description'] = 'altitude above sea level (GPS altitude)'
    # dset.attrs['unit'] = 'meter'

    # dset = f.create_dataset('longitude', data=data_hsk['longitude'])
    # dset.attrs['description'] = 'longitude'
    # dset.attrs['unit'] = 'degree'

    # dset = f.create_dataset('latitude', data=data_hsk['latitude'])
    # dset.attrs['description'] = 'latitude'
    # dset.attrs['unit'] = 'degree'

    # dset = f.create_dataset('zen_wvl', data=ssfr_v0.zen_wvl)
    # dset.attrs['description'] = 'center wavelengths of zenith channels (bandwidth see info)'
    # dset.attrs['unit'] = 'nm'

    # dset = f.create_dataset('nad_wvl', data=ssfr_v0.nad_wvl)
    # dset.attrs['description'] = 'center wavelengths of nadir channels (bandwidth see info)'
    # dset.attrs['unit'] = 'nm'

    # dset = f.create_dataset('zen_flux', data=zen_flux)
    # dset.attrs['description'] = 'downwelling shortwave spectral irradiance'
    # dset.attrs['unit'] = 'W / m2 / nm'

    # dset = f.create_dataset('nad_flux', data=nad_flux)
    # dset.attrs['description'] = 'upwelling shortwave spectral irradiance'
    # dset.attrs['unit'] = 'W / m2 / nm'

    # dset = f.create_dataset('pitch', data=pitch)
    # dset.attrs['description'] = 'aircraft pitch angle (positive values indicate nose up)'
    # dset.attrs['unit'] = 'degree'

    # dset = f.create_dataset('roll', data=roll)
    # dset.attrs['description'] = 'aircraft roll angle (positive values indicate right wing down)'
    # dset.attrs['unit'] = 'degree'

    # dset = f.create_dataset('heading', data=heading)
    # dset.attrs['description'] = 'aircraft heading angle (positive values clockwise, w.r.t north)'
    # dset.attrs['unit'] = 'degree'

    # dset = f.create_dataset('sza', data=sza)
    # dset.attrs['description'] = 'solar zenith angle'
    # dset.attrs['unit'] = 'degree'

    # f.close()
    #\----------------------------------------------------------------------------/#

    return

def process_ssfr_data(date):

    cdata_arcsix_ssfr_v0(date)
    cdata_arcsix_ssfr_v1(date)
    cdata_arcsix_ssfr_v2(date)
#\----------------------------------------------------------------------------/#



def plot_time_series(date, wvl0=700.0):

    date_s = date.strftime('%Y-%m-%d')

    f = h5py.File('/argus/pre-mission/arcsix/processed/ARCSIX_SPNS-B_%s_v2.h5' % date_s, 'r')
    tmhr = f['tmhr'][...]
    wvl_ = f['tot/wvl'][...]
    flux_spns_tot = f['tot/flux'][...][:, np.argmin(np.abs(wvl_-wvl0))]
    f.close()

    f = h5py.File('/argus/pre-mission/arcsix/processed/ARCSIX_SSFR-B_%s_v2.h5' % date_s, 'r')
    wvl_ = f['dset0/wvl_zen'][...]
    flux_ssfr_zen0 = f['dset0/flux_zen'][...][:, np.argmin(np.abs(wvl_-wvl0))]
    wvl_ = f['dset0/wvl_nad'][...]
    flux_ssfr_nad0 = f['dset0/flux_nad'][...][:, np.argmin(np.abs(wvl_-wvl0))]

    wvl_ = f['dset1/wvl_zen'][...]
    flux_ssfr_zen1 = f['dset1/flux_zen'][...][:, np.argmin(np.abs(wvl_-wvl0))]
    wvl_ = f['dset1/wvl_nad'][...]
    flux_ssfr_nad1 = f['dset1/flux_nad'][...][:, np.argmin(np.abs(wvl_-wvl0))]
    f.close()

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.scatter(tmhr, flux_spns_tot, s=6, c='k', lw=0.0)
        ax1.scatter(tmhr, flux_ssfr_zen0, s=3, c='r', lw=0.0)
        ax1.scatter(tmhr, flux_ssfr_zen1, s=3, c='magenta', lw=0.0)
        ax1.scatter(tmhr, flux_ssfr_nad0, s=3, c='b', lw=0.0)
        ax1.scatter(tmhr, flux_ssfr_nad1, s=3, c='cyan', lw=0.0)
        ax1.set_xlabel('Time [Hour]')
        ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.set_title('Skywatch Test (Belana, %s, %d nm)' % (date_s, wvl0))
        #\--------------------------------------------------------------/#

        patches_legend = [
                          mpatches.Patch(color='black' , label='SPNS-B Total'), \
                          mpatches.Patch(color='red'    , label='SSFR-B Zenith Si080In250'), \
                          mpatches.Patch(color='magenta', label='SSFR-B Zenith Si120In350'), \
                          mpatches.Patch(color='blue'   , label='SSFR-B Nadir Si080In250'), \
                          mpatches.Patch(color='cyan'   , label='SSFR-B Nadir Si120In350'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
    #\----------------------------------------------------------------------------/#

def plot_spectra(date, tmhr0=20.830):

    date_s = date.strftime('%Y-%m-%d')

    f = h5py.File('/argus/pre-mission/arcsix/processed/ARCSIX_SPNS-B_%s_v2.h5' % date_s, 'r')
    tmhr = f['tmhr'][...]
    wvl_spns_tot  = f['tot/wvl'][...]
    flux_spns_tot = f['tot/flux'][...][np.argmin(np.abs(tmhr-tmhr0)), :]
    f.close()

    f = h5py.File('/argus/pre-mission/arcsix/processed/ARCSIX_SSFR-B_%s_v2.h5' % date_s, 'r')
    flux_ssfr_zen0 = f['dset0/flux_zen'][...][np.argmin(np.abs(tmhr-tmhr0)), :]
    flux_ssfr_nad0 = f['dset0/flux_nad'][...][np.argmin(np.abs(tmhr-tmhr0)), :]

    wvl_ssfr_zen = f['dset1/wvl_zen'][...]
    flux_ssfr_zen1 = f['dset1/flux_zen'][...][np.argmin(np.abs(tmhr-tmhr0)), :]
    wvl_ssfr_nad = f['dset1/wvl_nad'][...]
    flux_ssfr_nad1 = f['dset1/flux_nad'][...][np.argmin(np.abs(tmhr-tmhr0)), :]
    f.close()

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.scatter(wvl_spns_tot, flux_spns_tot, s=6, c='k', lw=0.0)
        ax1.scatter(wvl_ssfr_zen, flux_ssfr_zen0, s=3, c='r', lw=0.0)
        ax1.scatter(wvl_ssfr_zen, flux_ssfr_zen1, s=3, c='magenta', lw=0.0)
        ax1.scatter(wvl_ssfr_nad, flux_ssfr_nad0, s=3, c='b', lw=0.0)
        ax1.scatter(wvl_ssfr_nad, flux_ssfr_nad1, s=3, c='cyan', lw=0.0)
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.set_title('Skywatch Test (Belana, %s, %.4f Hour)' % (date_s, tmhr0))
        #\--------------------------------------------------------------/#

        patches_legend = [
                          mpatches.Patch(color='black' , label='SPNS-B Total'), \
                          mpatches.Patch(color='red'    , label='SSFR-B Zenith Si080In250'), \
                          mpatches.Patch(color='magenta', label='SSFR-B Zenith Si120In350'), \
                          mpatches.Patch(color='blue'   , label='SSFR-B Nadir Si080In250'), \
                          mpatches.Patch(color='cyan'   , label='SSFR-B Nadir Si120In350'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
    #\----------------------------------------------------------------------------/#



if __name__ == '__main__':

    dates = [
             # datetime.datetime(2023, 10, 10),
             # datetime.datetime(2023, 10, 12),
             # datetime.datetime(2023, 10, 13),
             # datetime.datetime(2023, 10, 18), # SPNS-B and SSFR-B at Skywatch
             # datetime.datetime(2023, 10, 19), # SPNS-B and SSFR-B at Skywatch
             datetime.datetime(2023, 10, 20), # SPNS-B and SSFR-B at Skywatch
            ]

    for date in dates:
        # process_spns_data(date)
        process_ssfr_data(date)
        plot_time_series(date)
        plot_spectra(date)
