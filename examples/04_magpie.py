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

_fdir_data_ = '/argus/field'
_fdir_v0_  = '/argus/field/magpie/2023/dhc6/processed'
_fdir_v1_  = '/argus/field/magpie/2023/dhc6/processed'



def cdata_magpie_hsk_v0(
        date,
        tmhr_range=[10.0, 24.0],
        fdir_data=_fdir_data_,
        fdir_out=_fdir_v0_,
        ):

    """
    process raw aircraft nav data
    """

    # read aircraft nav data (housekeeping file)
    #/----------------------------------------------------------------------------\#
    fname = sorted(glob.glob('%s/magpie/2023/dhc6/hsk/raw/CABIN_1hz*%s*' % (fdir_data, date.strftime('%m_%d'))))[0]
    data_hsk = ssfr.util.read_cabin(fname, tmhr_range=tmhr_range)
    data_hsk['long']['data'] = -data_hsk['long']['data']
    #\----------------------------------------------------------------------------/#

    # solar geometries
    #/----------------------------------------------------------------------------\#
    jday0 = ssfr.util.dtime_to_jday(date)
    jday = jday0 + data_hsk['tmhr']['data']/24.0
    sza, saa = ssfr.util.cal_solar_angles(jday, data_hsk['long']['data'], data_hsk['lat']['data'], data_hsk['palt']['data'])
    #\----------------------------------------------------------------------------/#

    # save processed data
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/MAGPIE_HSK_%s_v0.h5' % (fdir_out, date.strftime('%Y-%m-%d'))

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

def cdata_magpie_spns_v0(
        date,
        fdir_data=_fdir_data_,
        fdir_out=_fdir_v0_,
        ):

    """
    process raw SPN-S data
    """

    # read spn-s raw data
    #/----------------------------------------------------------------------------\#
    fdir = '%s/magpie/2023/dhc6/spn-s/raw/%s' % (fdir_data, date.strftime('%Y-%m-%d'))

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
    fname_h5 = '%s/MAGPIE_SPN-S_%s_v0.h5' % (fdir_out, date.strftime('%Y-%m-%d'))

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

def cdata_magpie_spns_v1(
        date,
        time_offset=0.0,
        fdir_data=_fdir_v0_,
        fdir_out=_fdir_v1_,
        ):

    """
    check for time offset and merge SPN-S data with aircraft data
    """

    # read hsk v0
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/MAGPIE_HSK_%s_v0.h5' % (fdir_data, date.strftime('%Y-%m-%d'))
    f = h5py.File(fname_h5, 'r')
    jday = f['jday'][...]
    sza  = f['sza'][...]
    tmhr = f['tmhr'][...]
    lon  = f['lon'][...]
    lat  = f['lat'][...]
    alt  = f['alt'][...]
    f.close()
    #\----------------------------------------------------------------------------/#


    # read spn-s v0
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/MAGPIE_SPN-S_%s_v0.h5' % (fdir_data, date.strftime('%Y-%m-%d'))
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
    fname_h5 = '%s/MAGPIE_SPN-S_%s_v1.h5' % (fdir_out, date.strftime('%Y-%m-%d'))

    f = h5py.File(fname_h5, 'w')

    f['jday'] = jday
    f['tmhr'] = tmhr
    f['lon']  = lon
    f['lat']  = lat
    f['alt']  = alt
    f['sza']  = sza

    g1 = f.create_group('dif')
    g1['wvl']   = wvl_dif
    g1['flux']  = flux_dif

    g2 = f.create_group('tot')
    g2['wvl']   = wvl_tot
    g2['flux']  = flux_tot
    g2['toa0']  = f_dn_tot_toa0

    f.close()
    #\----------------------------------------------------------------------------/#

def cdata_sat_img(
        date,
        margin=0.1,
        fdir_data=_fdir_v0_,
        ):

    # read hsk v0
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/MAGPIE_HSK_%s_v0.h5' % (fdir_data, date.strftime('%Y-%m-%d'))
    f = h5py.File(fname_h5, 'r')
    jday = f['jday'][...]
    sza  = f['sza'][...]
    tmhr = f['tmhr'][...]
    lon  = f['lon'][...]
    lat  = f['lat'][...]
    alt  = f['alt'][...]
    f.close()
    #\----------------------------------------------------------------------------/#

    # extent = [np.nanmin(lon)-margin, np.nanmax(lon)+margin, np.nanmin(lat)-margin, np.nanmax(lat)+margin]
    extent = [-60.5, -58.5, 12, 14]

    command = 'sdown --date %s --extent %s --products MODRGB MYDRGB VNPRGB VJ1RGB --fdir data/sat-img' % (date.strftime('%Y%m%d'), ' '.join([str(a) for a in extent]))

    print(command)


if __name__ == '__main__':

    dates = [
            datetime.datetime(2023, 8, 2),  \
            datetime.datetime(2023, 8, 3),  \
            datetime.datetime(2023, 8, 5),  \
            datetime.datetime(2023, 8, 13), \
            datetime.datetime(2023, 8, 14), \
            datetime.datetime(2023, 8, 15), \
        ]

    for date in dates:
        cdata_magpie_hsk_v0(date)
        cdata_magpie_spns_v0(date)
        cdata_magpie_spns_v1(date)

    for date in dates:
        date_s = date.strftime('%Y-%m-%d')
        fname = '%s/MAGPIE_SPN-S_%s_v1.h5' % (_fdir_v1_, date_s)
        ssfr.vis.quicklook_bokeh_spns(fname, wvl0=None, tmhr0=None, tmhr_range=None, wvl_range=[350.0, 800.0], tmhr_step=10, wvl_step=5, description='MAGPIE', fname_html='spns-ql_magpie_%s.html' % date_s)

    cdata_sat_img(dates[-1])
