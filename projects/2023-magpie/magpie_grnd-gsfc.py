"""
Code for processing SPN-S data from Jeff at GSFC

by
Hong Chen (hong.chen@lasp.colorado.edu)
"""

import os
import sys
import glob
import datetime
import warnings
from tqdm import tqdm
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav
from scipy.optimize import curve_fit
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
# mpl.use('TkAgg')


import ssfr



# parameters
#/----------------------------------------------------------------------------\#
_mission_     = 'magpie-gsfc'
_platform_    = 'ground'

_hsk_         = 'hsk'
_spns_        = 'spns'

_fdir_data_  = 'data/%s' % _mission_
_fdir_out_   = 'data/processed'


_verbose_   = True
_test_mode_ = True

_fnames_ = {}
#\----------------------------------------------------------------------------/#








# functions for processing SPNS
#/----------------------------------------------------------------------------\#
def cdata_navy_hsk_v0(
        date,
        fdir_out=_fdir_out_,
        run=True,
        ):

    """
    For processing aricraft housekeeping file

    Notes:
        The housekeeping data would require some corrections before its release by the
        data system team, we usually request the raw IWG file (similar data but with a
        slightly different data formatting) from the team right after each flight to
        facilitate our data processing in a timely manner.
    """

    date_s = date.strftime('%Y%m%d')

    # fake hsk for skywatch
    #/----------------------------------------------------------------------------\#
    tmhr = np.arange(0.0, 24.0, 60.0/3600.0)
    # lon0 = -76.85117523899086
    # lat0 = 38.99806212497037
    lon0 = -59.43
    lat0 = 13.16
    alt0 =  10.0  # building altitude
    pit0 = 0.0
    rol0 = 0.0
    hed0 = 0.0
    data_hsk = {
            'tmhr': {'data': tmhr, 'units': 'hour'},
            'lon': {'data': np.repeat(lon0, tmhr.size), 'units': 'degree'},
            'lat': {'data': np.repeat(lat0, tmhr.size), 'units': 'degree'},
            'alt': {'data': np.repeat(alt0, tmhr.size), 'units': 'meter'},
            'ang_pit': {'data': np.repeat(pit0, tmhr.size), 'units': 'degree'},
            'ang_rol': {'data': np.repeat(rol0, tmhr.size), 'units': 'degree'},
            'ang_hed': {'data': np.repeat(hed0, tmhr.size), 'units': 'degree'},
            }
    #\----------------------------------------------------------------------------/#

    fname_h5 = '%s/%s-%s_%s_%s_v0.h5' % (fdir_out, _mission_.upper(), _hsk_.upper(), _platform_.upper(), date_s)

    if run:

        # solar geometries
        #/----------------------------------------------------------------------------\#
        jday0 = ssfr.util.dtime_to_jday(date)
        jday  = jday0 + data_hsk['tmhr']['data']/24.0

        sza, saa = ssfr.util.cal_solar_angles(jday, data_hsk['lon']['data'], data_hsk['lat']['data'], data_hsk['alt']['data'])
        #\----------------------------------------------------------------------------/#

        # save processed data
        #/----------------------------------------------------------------------------\#
        f = h5py.File(fname_h5, 'w')
        for var in data_hsk.keys():
            f[var] = data_hsk[var]['data']
        f['jday'] = jday
        f['sza']  = sza
        f['saa']  = saa
        f.close()
        #\----------------------------------------------------------------------------/#

    return fname_h5

def cdata_navy_spns_v0(
        date,
        fdir_data=_fdir_data_,
        fdir_out=_fdir_out_,
        run=True,
        ):

    """
    Process raw SPN-S data
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v0.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), _platform_.upper(), date_s)

    if run:

        # read spn-s raw data
        #/----------------------------------------------------------------------------\#
        fname_dif = ssfr.util.get_all_files(fdir_data, pattern='*Diffuse.txt')[-1]
        data0_dif = ssfr.lasp_spn.read_spns(fname=fname_dif)

        fname_tot = ssfr.util.get_all_files(fdir_data, pattern='*Total.txt')[-1]
        data0_tot = ssfr.lasp_spn.read_spns(fname=fname_tot)

        msg = 'Processing %s data:\n%s\n%s\n' % (_spns_.upper(), fname_dif, fname_tot)
        print(msg)
        #/----------------------------------------------------------------------------\#

        # read wavelengths and calculate toa downwelling solar flux
        #/----------------------------------------------------------------------------\#
        flux_toa = ssfr.util.get_solar_kurudz()

        wvl_tot = data0_tot.data['wvl']
        f_dn_sol_tot = np.zeros_like(wvl_tot)
        for i, wvl0 in enumerate(wvl_tot):
            f_dn_sol_tot[i] = ssfr.util.cal_weighted_flux(wvl0, flux_toa[:, 0], flux_toa[:, 1])
        f_dn_sol_tot *= ssfr.util.cal_solar_factor(date)
        #\----------------------------------------------------------------------------/#

        f = h5py.File(fname_h5, 'w')

        g1 = f.create_group('dif')
        for key in data0_dif.data.keys():
            if key in ['tmhr', 'jday', 'wvl', 'flux']:
                dset0 = g1.create_dataset(key, data=data0_dif.data[key], compression='gzip', compression_opts=9, chunks=True)

        g2 = f.create_group('tot')
        for key in data0_tot.data.keys():
            if key in ['tmhr', 'jday', 'wvl', 'flux']:
                dset0 = g2.create_dataset(key, data=data0_tot.data[key], compression='gzip', compression_opts=9, chunks=True)
        g2['toa0'] = f_dn_sol_tot

        f.close()

    return fname_h5

def cdata_navy_spns_v1(
        date,
        fname_spns_v0,
        fname_hsk,
        fdir_out=_fdir_out_,
        # time_offset=7.0*3600.0,
        time_offset=0.0,
        run=True,
        ):

    """
    Check for time offset and merge SPN-S data with aircraft data
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v1.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), _platform_.upper(), date_s)

    if run:
        # read spn-s v0
        #/----------------------------------------------------------------------------\#
        data_spns_v0 = ssfr.util.load_h5(fname_spns_v0)
        #/----------------------------------------------------------------------------\#

        # read hsk v0
        #/----------------------------------------------------------------------------\#
        data_hsk= ssfr.util.load_h5(fname_hsk)
        #\----------------------------------------------------------------------------/#

        # interpolate spn-s data to hsk time frame
        #/----------------------------------------------------------------------------\#
        flux_dif = np.zeros((data_hsk['jday'].size, data_spns_v0['dif/wvl'].size), dtype=np.float64)
        for i in range(flux_dif.shape[-1]):
            flux_dif[:, i] = ssfr.util.interp(data_hsk['jday'], data_spns_v0['dif/jday']+time_offset/86400.0, data_spns_v0['dif/flux'][:, i])

        flux_tot = np.zeros((data_hsk['jday'].size, data_spns_v0['tot/wvl'].size), dtype=np.float64)
        for i in range(flux_tot.shape[-1]):
            flux_tot[:, i] = ssfr.util.interp(data_hsk['jday'], data_spns_v0['tot/jday']+time_offset/86400.0, data_spns_v0['tot/flux'][:, i])
        #\----------------------------------------------------------------------------/#

        f = h5py.File(fname_h5, 'w')

        for key in data_hsk.keys():
            f[key] = data_hsk[key]

        f['time_offset'] = time_offset
        f['tmhr_ori'] = data_hsk['tmhr'] - time_offset/3600.0
        f['jday_ori'] = data_hsk['jday'] - time_offset/86400.0

        g1 = f.create_group('dif')
        g1['wvl']   = data_spns_v0['dif/wvl']
        dset0 = g1.create_dataset('flux', data=flux_dif, compression='gzip', compression_opts=9, chunks=True)

        g2 = f.create_group('tot')
        g2['wvl']   = data_spns_v0['tot/wvl']
        g2['toa0']  = data_spns_v0['tot/toa0']
        dset0 = g2.create_dataset('flux', data=flux_tot, compression='gzip', compression_opts=9, chunks=True)

        f.close()

    return fname_h5

def cdata_navy_spns_v2(
        date,
        fname_spns_v1,
        fname_hsk, # interchangable with fname_alp_v1
        wvl_range=[350.0, 900.0],
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        fdir_out=_fdir_out_,
        run=True,
        ):

    """
    Apply attitude correction to account for aircraft attitude (pitch, roll, heading)
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_%s_v2.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), _platform_.upper(), date_s)

    if run:

        # read spn-s v1
        #/----------------------------------------------------------------------------\#
        data_spns_v1 = ssfr.util.load_h5(fname_spns_v1)
        #/----------------------------------------------------------------------------\#

        # read hsk v0
        #/----------------------------------------------------------------------------\#
        data_hsk = ssfr.util.load_h5(fname_hsk)
        #/----------------------------------------------------------------------------\#

        # correction factor
        #/----------------------------------------------------------------------------\#
        mu = np.cos(np.deg2rad(data_hsk['sza']))

        iza, iaa = ssfr.util.prh2za(data_hsk['ang_pit']+ang_pit_offset, data_hsk['ang_rol']+ang_rol_offset, data_hsk['ang_hed'])
        dc = ssfr.util.muslope(data_hsk['sza'], data_hsk['saa'], iza, iaa)

        factors = mu / dc
        #\----------------------------------------------------------------------------/#

        # attitude correction
        #/----------------------------------------------------------------------------\#
        f_dn_dir = data_spns_v1['tot/flux'] - data_spns_v1['dif/flux']
        f_dn_dir_corr = np.zeros_like(f_dn_dir)
        f_dn_tot_corr = np.zeros_like(f_dn_dir)
        for iwvl in range(data_spns_v1['tot/wvl'].size):
            f_dn_dir_corr[..., iwvl] = f_dn_dir[..., iwvl]*factors
            f_dn_tot_corr[..., iwvl] = f_dn_dir_corr[..., iwvl] + data_spns_v1['dif/flux'][..., iwvl]
        #\----------------------------------------------------------------------------/#

        f = h5py.File(fname_h5, 'w')

        for key in data_hsk.keys():
            f[key] = data_hsk[key]

        g0 = f.create_group('att_corr')
        g0['mu'] = mu
        g0['dc'] = dc
        g0['factors'] = factors

        if wvl_range is None:
            wvl_range = [0.0, 2200.0]

        logic_wvl_dif = (data_spns_v1['dif/wvl']>=wvl_range[0]) & (data_spns_v1['dif/wvl']<=wvl_range[1])
        logic_wvl_tot = (data_spns_v1['tot/wvl']>=wvl_range[0]) & (data_spns_v1['tot/wvl']<=wvl_range[1])

        g1 = f.create_group('dif')
        g1['wvl']   = data_spns_v1['dif/wvl'][logic_wvl_dif]
        dset0 = g1.create_dataset('flux', data=data_spns_v1['dif/flux'][:, logic_wvl_dif], compression='gzip', compression_opts=9, chunks=True)

        g2 = f.create_group('tot')
        g2['wvl']   = data_spns_v1['tot/wvl'][logic_wvl_tot]
        g2['toa0']  = data_spns_v1['tot/toa0'][logic_wvl_tot]
        dset0 = g2.create_dataset('flux', data=f_dn_tot_corr[:, logic_wvl_tot], compression='gzip', compression_opts=9, chunks=True)

        f.close()

    return fname_h5

def process_spns_data(date, run=True):

    """
    v0: raw data directly read out from the data files
    v1: data collocated/synced to aircraft nav
    v2: attitude corrected data
    """

    fdir_out = _fdir_out_
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    fdirs = ssfr.util.get_all_folders(_fdir_data_, pattern='*%4.4d*%2.2d*%2.2d' % (date.year, date.month, date.day))
    fdir_data = sorted(fdirs, key=os.path.getmtime)[-1]

    date_s = date.strftime('%Y%m%d')

    fname_hsk_v0 = cdata_navy_hsk_v0(date,
            fdir_out=fdir_out, run=run)
    fname_spns_v0 = cdata_navy_spns_v0(date, fdir_data=fdir_data,
            fdir_out=fdir_out, run=run)
    fname_spns_v1 = cdata_navy_spns_v1(date, fname_spns_v0, fname_hsk_v0,
            fdir_out=fdir_out, run=run)
    fname_spns_v2 = cdata_navy_spns_v2(date, fname_spns_v1, fname_hsk_v0,
            fdir_out=fdir_out, run=run)

    _fnames_['%s_hsk_v0' % date_s] = fname_hsk_v0
    _fnames_['%s_spns_v0' % date_s] = fname_spns_v0
    _fnames_['%s_spns_v1' % date_s] = fname_spns_v1
    _fnames_['%s_spns_v2' % date_s] = fname_spns_v2
#\----------------------------------------------------------------------------/#





# main program
#/----------------------------------------------------------------------------\#
def main_process_data(date, run=True):

    date_s = date.strftime('%Y%m%d')

    # 1. SPNS - irradiance (400nm - 900nm)
    #    - spectral downwelling diffuse
    #    - spectral downwelling global/direct (direct=global-diffuse)
    process_spns_data(date, run=True)
    ssfr.vis.quicklook_bokeh_spns(_fnames_['%s_spns_v2' % date_s], wvl0=None, tmhr0=None, tmhr_range=[0, 24], wvl_range=[350.0, 900.0], tmhr_step=1, wvl_step=5, description=_mission_.upper(), fname_html='%s-%s_%s_%s_ql.html' % (_mission_.upper(), _spns_.upper(), _platform_.upper(), date_s))
#\----------------------------------------------------------------------------/#




if __name__ == '__main__':

    # data procesing
    #/----------------------------------------------------------------------------\#
    # fdirs = sorted(glob.glob('data/%s/????-??-??' % _mission_))
    # for fdir in fdirs:
    #     date_s = os.path.basename(fdir)
    #     date = datetime.datetime.strptime(date_s, '%Y-%m-%d')
    #     main_process_data(date)
    #\----------------------------------------------------------------------------/#
    main_process_data(datetime.datetime(2024, 10, 15))

    pass
