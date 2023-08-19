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
_fdir_v2_  = '/argus/field/magpie/2023/dhc6/processed'



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

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        # cs = ax1.imshow(.T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        ax1.scatter(data0_tot.data['tmhr'], data0_tot.data['flux'][..., 200], s=6, c='k', lw=0.0)
        # ax1.hist(.ravel(), bins=100, histtype='stepfilled', alpha=0.5, color='black')
        # ax1.plot([0, 1], [0, 1], color='k', ls='--')
        # ax1.set_xlim(())
        # ax1.set_ylim(())
        # ax1.set_xlabel('')
        # ax1.set_ylabel('')
        # ax1.set_title('')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        # fig.subplots_adjust(hspace=0.3, wspace=0.3)
        # _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        # fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
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



def cdata_magpie_spns_v2(
        date,
        time_offset=0.0,
        fdir_data=_fdir_v1_,
        fdir_out=_fdir_v2_,
        ):

    """
    apply attitude correction to account for pitch and roll
    """

    # read spn-s v1
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/MAGPIE_SPN-S_%s_v1.h5' % (fdir_data, date.strftime('%Y-%m-%d'))
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
    fname_h5 = '%s/MAGPIE_SPN-S_%s_v2.h5' % (fdir_out, date.strftime('%Y-%m-%d'))

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


    if True:
        wvl0 = 532.0
        index_wvl = np.argmin(np.abs(wvl_tot-wvl0))

        plt.close('all')
        fig = plt.figure(figsize=(18, 6))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.scatter(tmhr, mu*f_dn_toa0[index_wvl], s=1, c='k', lw=0.0)
        ax1.scatter(tmhr, f_dn_tot[..., index_wvl], s=1, c='r', lw=0.0)
        ax1.scatter(tmhr, f_dn_tot_corr[..., index_wvl], s=1, c='g', lw=0.0)
        ax1.set_title('MAGPIE %s (%d nm)' % (date.strftime('%Y-%m-%d'), wvl0))
        #\--------------------------------------------------------------/#

        patches_legend = [
                          mpatches.Patch(color='black' , label='TOA (Kurudz)'), \
                          mpatches.Patch(color='red'   , label='Original (Direct)'), \
                          mpatches.Patch(color='green' , label='Attitude Corrected (Direct)'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)

        ax1.set_ylim((0.0, 2.2))
        ax1.set_xlabel('UTC Time [Hour]')
        ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_%s.png' % (_metadata['Function'], date.strftime('%Y-%m-%d')), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()

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

def cal_pit_offset(
        date,
        time_offset=0.0,
        fdir_data=_fdir_v1_,
        fdir_out=_fdir_v2_,
        ):

    """
    apply attitude correction to account for pitch and roll
    """

    # read spn-s v1
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/MAGPIE_SPN-S_%s_v1.h5' % (fdir_data, date.strftime('%Y-%m-%d'))
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


    # distance
    #/----------------------------------------------------------------------------\#
    dist = np.sqrt((lon-lon[100])**2 + (lat-lat[100])**2) * 111000.0
    logic = (dist<10) & (tmhr<16.0)
    #\----------------------------------------------------------------------------/#


    pit_offsets = np.linspace(-10.0, 10.0, 201)
    rol_offsets = np.linspace(-5.0, 5.0, 101)
    diffs = np.zeros((pit_offsets.size, rol_offsets.size), dtype=np.float64)
    mu_ = np.cos(np.deg2rad(sza[logic]))
    for i, pit_offset in enumerate(pit_offsets):
        for j, rol_offset in enumerate(rol_offsets):
            iza_, iaa_ = ssfr.util.prh2za(pit[logic]+pit_offset, rol[logic]+rol_offset, hed[logic])
            dc_ = ssfr.util.muslope(sza[logic], saa[logic], iza_, iaa_)
            diffs[i, j] = np.nanmean(mu_ - dc_)

    index_pit, index_rol = np.unravel_index(np.argmin(np.abs(diffs)), diffs.shape)
    pit_offset = pit_offsets[index_pit]
    rol_offset = rol_offsets[index_rol]
    print(date, pit_offset, rol_offset)

    # correction factor
    #/----------------------------------------------------------------------------\#
    mu = np.cos(np.deg2rad(sza))

    iza, iaa = ssfr.util.prh2za(pit+pit_offset, rol+rol_offset, hed)
    dc = ssfr.util.muslope(sza, saa, iza, iaa)

    iza0, iaa0 = ssfr.util.prh2za(pit, rol, hed)
    dc0 = ssfr.util.muslope(sza, saa, iza0, iaa0)

    # factors = mu / dc
    #\----------------------------------------------------------------------------/#

    if True:

        plt.close('all')
        fig = plt.figure(figsize=(18, 12))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(211)
        ax1.scatter(tmhr[logic], mu[logic], s=4, c='k', lw=0.0, alpha=0.7)
        ax1.scatter(tmhr[logic], dc0[logic], s=1, c='r', lw=0.0)
        ax1.scatter(tmhr[logic], dc[logic], s=1, c='g', lw=0.0)

        ax2 = fig.add_subplot(223)
        ax2.hist(mu[logic], bins=100, color='k', lw=2.0, histtype='step')
        ax2.hist(dc0[logic], bins=100, color='r', lw=1.0, histtype='step')
        ax2.hist(dc[logic], bins=100, color='g', lw=1.0, histtype='step')
        ax2.set_xlabel('$cos(SZA)$')
        ax2.set_ylabel('PDF')

        ax3 = fig.add_subplot(224)
        ax3.hist(mu[logic]-dc0[logic], bins=100, color='r', lw=1.0, histtype='step')
        ax3.hist(mu[logic]-dc[logic], bins=100, color='g', lw=1.0, histtype='step')
        ax3.axvline(0.0, color='gray', ls='--')
        ax3.set_xlabel('$cos(SZA)$')
        ax3.set_ylabel('PDF')
        #\--------------------------------------------------------------/#

        fig.suptitle('MAGPIE %s (Pitch %.1f$^\circ$, Roll %.1f$^\circ$)' % (date.strftime('%Y-%m-%d'), pit_offset, rol_offset))

        patches_legend = [
                          mpatches.Patch(color='black' , label='$cos(SZA)$'), \
                          mpatches.Patch(color='red'   , label='$cos(SZA_{ref})$'), \
                          mpatches.Patch(color='green' , label='With Offset'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)

        ax1.set_xlabel('UTC Time [Hour]')
        ax1.set_ylabel('$cos(SZA)$')

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_%s.png' % (_metadata['Function'], date.strftime('%Y-%m-%d')), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()


if __name__ == '__main__':

    dates = [
            datetime.datetime(2023, 8, 2),
            datetime.datetime(2023, 8, 3),
            datetime.datetime(2023, 8, 5),
            datetime.datetime(2023, 8, 13),
            datetime.datetime(2023, 8, 14),
            datetime.datetime(2023, 8, 15), # heavy aerosol condition
            datetime.datetime(2023, 8, 16), # data of this flight looks abnormal
            datetime.datetime(2023, 8, 18),
        ]

    for date in dates:
        cdata_magpie_hsk_v0(date)  # read aircraft raw data and calculate solar angles
        cdata_magpie_spns_v0(date) # read SPN-S raw data
        cdata_magpie_spns_v1(date) # interpolate SPN-S data to aircraft time coordinate
        cdata_magpie_spns_v2(date) # apply attitude correction (pitch, roll, and heading) to SPN-S data

    for date in dates:
        date_s = date.strftime('%Y-%m-%d')
        fname = '%s/MAGPIE_SPN-S_%s_v2.h5' % (_fdir_v1_, date_s)
        ssfr.vis.quicklook_bokeh_spns(fname, wvl0=None, tmhr0=None, tmhr_range=None, wvl_range=[350.0, 800.0], tmhr_step=10, wvl_step=5, description='MAGPIE', fname_html='spns-ql_magpie_%s_v2.html' % date_s)
