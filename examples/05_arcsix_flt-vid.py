"""
by Hong Chen (hong.chen@lasp.colorado.edu)

This code serves as an example code to reproduce 3D irradiance simulation for App. 3 in Chen et al. (2022).
Special note: due to large data volume, only partial flight track simulation is provided for illustration purpose.

The processes include:
    1) `main_run()`: pre-process aircraft and satellite data and run simulations
        a) partition flight track into mini flight track segments and collocate satellite imagery data
        b) run simulations based on satellite imagery cloud retrievals
            i) 3D mode
            ii) IPA mode

    2) `main_post()`: post-process data
        a) extract radiance observations from pre-processed data
        b) extract radiance simulations of EaR3T
        c) plot

This code has been tested under:
    1) Linux on 2023-06-27 by Hong Chen
      Operating System: Red Hat Enterprise Linux
           CPE OS Name: cpe:/o:redhat:enterprise_linux:7.7:GA:workstation
                Kernel: Linux 3.10.0-1062.9.1.el7.x86_64
          Architecture: x86-64
"""

import os
import sys
import glob
from collections import OrderedDict
import datetime
import multiprocessing as mp
import pickle
from tqdm import tqdm
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
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


import er3t

_fdir_main_ = 'data/test/arcsix'
_fdir_sat_img_ = '%s/sat-img' % _fdir_main_
_wavelength_ = 745.0

_fdir_data_ = 'data/test/arcsix'
_fdir_v0_  = 'data/test/processed'
_fdir_v1_  = 'data/test/processed'
_fdir_v2_  = 'data/test/processed'


# global variables
#/--------------------------------------------------------------\#
params = {
                    'name_tag' : os.path.relpath(__file__).replace('.py', ''),
                          'dx' : 2000.0,
                          'dy' : 2000.0,
                      'photon' : 1e7,
        }
#\--------------------------------------------------------------/#



def check_continuity(data, threshold=0.1):

    data = np.append(data[0], data)

    return (np.abs(data[1:]-data[:-1]) < threshold)

def partition_flight_track(flt_trk, jday_edges, margin_x=1.0, margin_y=1.0):

    """
    Input:
        flt_trk: Python dictionary that contains
            ['jday']: numpy array, UTC time in hour
            ['tmhr']: numpy array, UTC time in hour
            ['lon'] : numpy array, longitude
            ['lat'] : numpy array, latitude
            ['alt'] : numpy array, altitude
            ['sza'] : numpy array, solar zenith angle
            [...]   : numpy array, other data variables, e.g., 'f_up_0600'

        tmhr_interval=: float, time interval of legs to be partitioned, default=0.1
        margin_x=     : float, margin in x (longitude) direction that to be used to
                        define the rectangular box to contain cloud field, default=1.0
        margin_y=     : float, margin in y (latitude) direction that to be used to
                        define the rectangular box to contain cloud field, default=1.0


    Output:
        flt_trk_segments: Python list that contains data for each partitioned leg in Python dictionary, e.g., legs[i] contains
            [i]['jday'] : numpy array, UTC time in hour
            [i]['tmhr'] : numpy array, UTC time in hour
            [i]['lon']  : numpy array, longitude
            [i]['lat']  : numpy array, latitude
            [i]['alt']  : numpy array, altitude
            [i]['sza']  : numpy array, solar zenith angle
            [i]['jday0']: mean value
            [i]['tmhr0']: mean value
            [i]['lon0'] : mean value
            [i]['lat0'] : mean value
            [i]['alt0'] : mean value
            [i]['sza0'] : mean value
            [i][...]    : numpy array, other data variables
    """

    flt_trk_segments = []

    for i in range(jday_edges.size-1):

        logic      = (flt_trk['jday']>=jday_edges[i]) & (flt_trk['jday']<jday_edges[i+1]) & (np.logical_not(np.isnan(flt_trk['sza'])))
        if logic.sum() > 0:

            flt_trk_segment = {}

            for key in flt_trk.keys():
                flt_trk_segment[key]     = flt_trk[key][logic]
                if key in ['jday', 'tmhr', 'lon', 'lat', 'alt', 'sza']:
                    flt_trk_segment[key+'0'] = np.nanmean(flt_trk_segment[key])

            flt_trk_segment['extent'] = np.array([np.nanmin(flt_trk_segment['lon'])-margin_x, \
                                                  np.nanmax(flt_trk_segment['lon'])+margin_x, \
                                                  np.nanmin(flt_trk_segment['lat'])-margin_y, \
                                                  np.nanmax(flt_trk_segment['lat'])+margin_y])

            flt_trk_segments.append(flt_trk_segment)

    return flt_trk_segments




def cal_mca_flux(
        index,
        fname_sat,
        extent,
        solar_zenith_angle,
        cloud_top_height=None,
        fdir='tmp-data',
        wavelength=745.0,
        date=datetime.datetime.now(),
        target='flux',
        solver='3D',
        photons=params['photon'],
        Ncpu=14,
        overwrite=True,
        quiet=False
        ):

    """
    flux simulation using EaR3T for flight track based on AHI cloud retrievals
    """

    # define an atmosphere object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    levels    = np.linspace(0.0, 20.0, 21)
    fname_atm = '%s/atm_%3.3d.pk' % (fdir, index)
    fname_prof = '%s/afglus.dat' % er3t.common.fdir_data_atmmod
    atm0       = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, fname_atmmod=fname_prof, overwrite=overwrite)
    # ------------------------------------------------------------------------------------------------------

    # define an absorption object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fname_abs = '%s/abs_%3.3d.pk' % (fdir, index)
    abs0      = er3t.pre.abs.abs_16g(wavelength=wavelength, fname=fname_abs, atm_obj=atm0, overwrite=overwrite)
    # ------------------------------------------------------------------------------------------------------

    # define an cloud object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fname_cld = '%s/cld_ahi_%3.3d.pk' % (fdir, index)

    if overwrite:
        ahi0      = er3t.util.ahi_l2(fnames=[fname_sat], extent=extent, vnames=['cld_height_acha'])
        lon_2d, lat_2d, cot_2d = er3t.util.grid_by_dxdy(ahi0.data['lon']['data'], ahi0.data['lat']['data'], ahi0.data['cot']['data'], extent=extent, dx=params['dx'], dy=params['dy'])
        lon_2d, lat_2d, cer_2d = er3t.util.grid_by_dxdy(ahi0.data['lon']['data'], ahi0.data['lat']['data'], ahi0.data['cer']['data'], extent=extent, dx=params['dx'], dy=params['dy'])
        cot_2d[cot_2d>100.0] = 100.0
        cer_2d[cer_2d==0.0] = 1.0
        ahi0.data['lon_2d'] = dict(name='Gridded longitude'               , units='degrees'    , data=lon_2d)
        ahi0.data['lat_2d'] = dict(name='Gridded latitude'                , units='degrees'    , data=lat_2d)
        ahi0.data['cot_2d'] = dict(name='Gridded cloud optical thickness' , units='N/A'        , data=cot_2d)
        ahi0.data['cer_2d'] = dict(name='Gridded cloud effective radius'  , units='micro'      , data=cer_2d)

        if cloud_top_height is None:
            lon_2d, lat_2d, cth_2d = er3t.util.grid_by_dxdy(ahi0.data['lon']['data'], ahi0.data['lat']['data'], ahi0.data['cld_height_acha']['data'], extent=extent, dx=params['dx'], dy=params['dy'])
            cth_2d[cth_2d<0.0]  = 0.0; cth_2d /= 1000.0
            ahi0.data['cth_2d'] = dict(name='Gridded cloud top height', units='km', data=cth_2d)
            cloud_top_height = ahi0.data['cth_2d']['data']
        cld0 = er3t.pre.cld.cld_sat(sat_obj=ahi0, fname=fname_cld, cth=cloud_top_height, cgt=1.0, dz=(levels[1]-levels[0]), overwrite=overwrite)
    else:
        cld0 = er3t.pre.cld.cld_sat(fname=fname_cld, overwrite=overwrite)
    # ----------------------------------------------------------------------------------------------------

    # mie scattering phase function setup
    #/--------------------------------------------------------------\#
    pha0 = er3t.pre.pha.pha_mie_wc(wavelength=wavelength, overwrite=overwrite)
    sca  = er3t.rtm.mca.mca_sca(pha_obj=pha0, fname='%s/mca_sca.bin' % fdir, overwrite=overwrite)
    #\--------------------------------------------------------------/#

    # define mcarats 1d and 3d "atmosphere", can represent aersol, cloud, atmosphere
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    atm1d0  = er3t.rtm.mca.mca_atm_1d(atm_obj=atm0, abs_obj=abs0)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.mca.mca_atm_3d(cld_obj=cld0, atm_obj=atm0, pha_obj=pha0, fname='%s/mca_atm_3d.bin' % fdir, quiet=quiet, overwrite=overwrite)
    atm_3ds = [atm3d0]
    # ------------------------------------------------------------------------------------------------------

    # define mcarats object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    mca0 = er3t.rtm.mca.mcarats_ng(
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            sca=sca,
            date=date,
            weights=abs0.coef['weight']['data'],
            solar_zenith_angle=solar_zenith_angle,
            fdir='%s/ahi/%s/%3.3d' % (fdir, solver.lower(), index),
            Nrun=3,
            photons=photons,
            solver=solver,
            target=target,
            Ncpu=Ncpu,
            mp_mode='py',
            quiet=quiet,
            overwrite=overwrite
            )
    # ------------------------------------------------------------------------------------------------------

    # define mcarats output object
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    out0 = er3t.rtm.mca.mca_out_ng(fname='%s/mca-out-%s-%s_ahi_%3.3d.h5' % (fdir, target.lower(), solver.lower(), index), mca_obj=mca0, abs_obj=abs0, mode='mean', squeeze=True, quiet=quiet, overwrite=overwrite)
    # ------------------------------------------------------------------------------------------------------

    return atm0, cld0, out0

def interpolate_3d_to_flight_track(flt_trk, data_3d):

    """
    Extract radiative properties along flight track from MCARaTS outputs

    Input:
        flt_trk: Python dictionary
            ['tmhr']: UTC time in hour
            ['lon'] : longitude
            ['lat'] : latitude
            ['alt'] : altitude

        data_3d: Python dictionary
            ['lon']: longitude
            ['lat']: latitude
            ['alt']: altitude
            [...]  : other variables that contain 3D data field

    Output:
        flt_trk:
            [...]: add interpolated data from data_3d[...]
    """

    points = np.transpose(np.vstack((flt_trk['lon'], flt_trk['lat'], flt_trk['alt'])))

    lon_field = data_3d['lon']
    lat_field = data_3d['lat']
    dlon    = lon_field[1]-lon_field[0]
    dlat    = lat_field[1]-lat_field[0]
    lon_trk = flt_trk['lon']
    lat_trk = flt_trk['lat']
    indices_lon = np.int_(np.round((lon_trk-lon_field[0])/dlon, decimals=0))
    indices_lat = np.int_(np.round((lat_trk-lat_field[0])/dlat, decimals=0))

    indices_lon = np.int_(flt_trk['lon']-data_3d['lon'][0])

    for key in data_3d.keys():
        if key not in ['tmhr', 'lon', 'lat', 'alt']:
            f_interp     = RegularGridInterpolator((data_3d['lon'], data_3d['lat'], data_3d['alt']), data_3d[key], method='linear')
            flt_trk[key] = f_interp(points)

            flt_trk['%s-alt-all' % key] = data_3d[key][indices_lon, indices_lat, :]

    return flt_trk




def download_geo_sat_img(
        dtime_s,
        dtime_e=None,
        extent=[-60.5, -58.5, 12, 14],
        satellite='GOES-East',
        instrument='ABI',
        fdir_out='%s/sat_img' % _fdir_main_,
        dpi=200,
        ):

    if dtime_e is None:
        dtime_e = datetime.datetime(date_s.year, date_s.month, date_s.day, 23, 59, date_s.second)

    while dtime_s <= dtime_e:
        fname_img = er3t.dev.download_worldview_image(dtime_s, extent, satellite=satellite, instrument=instrument, fdir_out=fdir_out, dpi=dpi, run=False)
        if not os.path.exists(fname_img):
            print(fname_img)
            fname_img = er3t.dev.download_worldview_image(dtime_s, extent, satellite=satellite, instrument=instrument, fdir_out=fdir_out, dpi=dpi, run=True)
        dtime_s += datetime.timedelta(minutes=10.0)

def download_polar_sat_img(
        date,
        extent=[-60.5, -58.5, 12, 14],
        imagers=['MODIS|Aqua', 'MODIS|Terra', 'VIIRS|NOAA20', 'VIIRS|SNPP'],
        fdir_out='%s/sat_img' % _fdir_main_,
        dpi=300,
        ):

    for imager in imagers:
        instrument, satellite = imager.split('|')
        fname_img = er3t.dev.download_worldview_image(date, extent, satellite=satellite, instrument=instrument, fdir_out=fdir_out, dpi=dpi, run=False)
        if not os.path.exists(fname_img):
            fname_img = er3t.dev.download_worldview_image(date, extent, satellite=satellite, instrument=instrument, fdir_out=fdir_out, dpi=dpi, run=True)

def get_jday_sat_img(fnames):

    """
    Get UTC time in hour from the satellite file name

    Input:
        fnames: Python list, file paths of all the satellite data

    Output:
        jday: numpy array, julian day
    """

    jday = []
    for fname in fnames:
        filename = os.path.basename(fname)
        strings  = filename.split('_')
        dtime_s  = strings[2]

        dtime0 = datetime.datetime.strptime(dtime_s, '%Y-%m-%dT%H:%M:%SZ')
        jday0 = er3t.util.dtime_to_jday(dtime0)
        jday.append(jday0)

    return np.array(jday)

def get_extent(lon, lat, margin=0.2):

    logic = check_continuity(lon) & check_continuity(lat)
    lon = lon[logic]
    lat = lat[logic]

    lon_min = np.nanmin(lon)
    lon_max = np.nanmax(lon)
    lat_min = np.nanmin(lat)
    lat_max = np.nanmax(lat)

    lon_c = (lon_min+lon_max)/2.0
    lat_c = (lat_min+lat_max)/2.0

    delta_lon = lon_max - lon_min
    delta_lat = lat_max - lat_min

    delta_half = max([delta_lon, delta_lat])/2.0 + margin

    extent = [
            lon_c-delta_half, \
            lon_c+delta_half, \
            lat_c-delta_half, \
            lat_c+delta_half  \
            ]

    return extent




class flt_sim:

    def __init__(
            self,
            date=datetime.datetime.now(),
            photons=params['photon'],
            Ncpu=12,
            fdir='tmp-data/%s' % params['name_tag'],
            wavelength=None,
            flt_trks=None,
            sat_imgs=None,
            fname=None,
            overwrite=False,
            overwrite_rtm=False,
            quiet=False,
            verbose=False
            ):

        self.date      = date
        self.photons   = photons
        self.Ncpu      = Ncpu
        self.wvl       = wavelength
        self.fdir      = fdir
        self.flt_trks  = flt_trks
        self.sat_imgs  = sat_imgs
        self.overwrite = overwrite
        self.quiet     = quiet
        self.verbose   = verbose

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif (((flt_trks is not None) and (sat_imgs is not None) and (wavelength is not None)) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             (((flt_trks is not None) and (sat_imgs is not None) and (wavelength is not None)) and (fname is not None) and (not os.path.exists(fname))):

            self.run(overwrite=overwrite_rtm)
            self.dump(fname)

        elif (((flt_trks is not None) and (sat_imgs is not None) and (wavelength is not None)) and (fname is None)):

            self.run()

        else:

            sys.exit('Error   [flt_sim]: Please check if \'%s\' exists or provide \'wavelength\', \'flt_trks\', and \'sat_imgs\' to proceed.' % fname)

    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'flt_trks') and hasattr(obj, 'sat_imgs'):
                if self.verbose:
                    print('Message [flt_sim]: Loading %s ...' % fname)
                self.wvl      = obj.wvl
                self.fname    = obj.fname
                self.flt_trks = obj.flt_trks
                self.sat_imgs = obj.sat_imgs
            else:
                sys.exit('Error   [flt_sim]: File \'%s\' is not the correct pickle file to load.' % fname)

    def run(self, overwrite=True):

        N = len(self.flt_trks)

        for i in range(N):

            print('%3.3d/%3.3d' % (i, N-1))

            flt_trk = self.flt_trks[i]
            sat_img = self.sat_imgs[i]

            # atm0, cld_ahi0, mca_out_ipa0 = cal_mca_flux(i, sat_img['fname'], sat_img['extent'], flt_trk['sza0'], date=self.date, wavelength=self.wvl, solver='IPA', fdir=self.fdir, photons=self.photons, Ncpu=self.Ncpu, overwrite=overwrite, quiet=self.quiet)
            # atm0, cld_ahi0, mca_out_3d0  = cal_mca_flux(i, sat_img['fname'], sat_img['extent'], flt_trk['sza0'], date=self.date, wavelength=self.wvl, solver='3D' , fdir=self.fdir, photons=self.photons, Ncpu=self.Ncpu, overwrite=overwrite, quiet=self.quiet)

            # self.sat_imgs[i]['lon'] = cld_ahi0.lay['lon']['data']
            # self.sat_imgs[i]['lat'] = cld_ahi0.lay['lat']['data']
            # self.sat_imgs[i]['cot'] = cld_ahi0.lay['cot']['data'] # cloud optical thickness (cot) is 2D (x, y)
            # self.sat_imgs[i]['cer'] = cld_ahi0.lay['cer']['data'] # cloud effective radius (cer) is 3D (x, y, z)

            # lon_sat = self.sat_imgs[i]['lon'][:, 0]
            # lat_sat = self.sat_imgs[i]['lat'][0, :]
            # dlon    = lon_sat[1]-lon_sat[0]
            # dlat    = lat_sat[1]-lat_sat[0]
            # lon_trk = self.flt_trks[i]['lon']
            # lat_trk = self.flt_trks[i]['lat']
            # indices_lon = np.int_(np.round((lon_trk-lon_sat[0])/dlon, decimals=0))
            # indices_lat = np.int_(np.round((lat_trk-lat_sat[0])/dlat, decimals=0))
            # self.flt_trks[i]['cot'] = self.sat_imgs[i]['cot'][indices_lon, indices_lat]
            # self.flt_trks[i]['cer'] = self.sat_imgs[i]['cer'][indices_lon, indices_lat, 0]

            # if 'cth' in cld_ahi0.lay.keys():
            #     self.sat_imgs[i]['cth'] = cld_ahi0.lay['cth']['data']
            #     self.flt_trks[i]['cth'] = self.sat_imgs[i]['cth'][indices_lon, indices_lat]

            # data_3d_mca = {
            #     'lon'         : cld_ahi0.lay['lon']['data'][:, 0],
            #     'lat'         : cld_ahi0.lay['lat']['data'][0, :],
            #     'alt'         : atm0.lev['altitude']['data'],
            #     }

            # index_h = np.argmin(np.abs(atm0.lev['altitude']['data']-flt_trk['alt0']))
            # if atm0.lev['altitude']['data'][index_h] > flt_trk['alt0']:
            #     index_h -= 1
            # if index_h < 0:
            #     index_h = 0

            # for key in mca_out_3d0.data.keys():
            #     if key in ['f_down', 'f_down_diffuse', 'f_down_direct', 'f_up', 'toa']:
            #         if 'toa' not in key:
            #             vname = key.replace('_', '-') + '_mca-3d'
            #             self.sat_imgs[i][vname] = mca_out_3d0.data[key]['data'][..., index_h]
            #             data_3d_mca[vname] = mca_out_3d0.data[key]['data']
            # for key in mca_out_ipa0.data.keys():
            #     if key in ['f_down', 'f_down_diffuse', 'f_down_direct', 'f_up', 'toa']:
            #         if 'toa' not in key:
            #             vname = key.replace('_', '-') + '_mca-ipa'
            #             self.sat_imgs[i][vname] = mca_out_ipa0.data[key]['data'][..., index_h]
            #             data_3d_mca[vname] = mca_out_ipa0.data[key]['data']

            # self.flt_trks[i] = interpolate_3d_to_flight_track(flt_trk, data_3d_mca)

    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [flt_sim]: Saving object into %s ...' % fname)
            pickle.dump(self, f)




def plot_video_frame_old(statements, test=False):

    # extract arguments
    #/----------------------------------------------------------------------------\#
    flt_sim0, index_trk, index_pnt, n = statements
    #\----------------------------------------------------------------------------/#


    # general plot settings
    #/----------------------------------------------------------------------------\#
    vars_plot = OrderedDict()

    vars_plot['Total↓']   = {
            'vname':'f-down-total_spns',
            'color':'black',
            }
    vars_plot['Diffuse↓']   = {
            'vname':'f-down-diffuse_spns',
            'color':'blue',
            }
    vars_plot['Direct↓']   = {
            'vname':'f-down-direct_spns',
            'color':'red',
            }
    vars_plot['Altitude']   = {
            'vname':'alt',
            'color':'orange',
            }
    #\----------------------------------------------------------------------------/#

    colors = OrderedDict()
    colors['Total↓']  = 'black'
    colors['Diffuse↓'] = 'blue'
    colors['Direct↓']  = 'red'
    colors['Altitude']       = 'orange'
    # colors['SSFR']           = 'black'
    # colors['RTM 3D']         = 'red'
    # colors['RTM IPA']        = 'blue'
    # colors['RTM 3D Diffuse'] = 'green'
    # colors['COT']            = 'purple'

    tmhr_length  = 0.5
    tmhr_current = flt_sim0.flt_trks[index_trk]['tmhr'][index_pnt]
    tmhr_past    = tmhr_current-tmhr_length

    lon_current = flt_sim0.flt_trks[index_trk]['lon'][index_pnt]
    lat_current = flt_sim0.flt_trks[index_trk]['lat'][index_pnt]
    alt_current = flt_sim0.flt_trks[index_trk]['alt'][index_pnt]

    cot_min = 0
    cot_max = 30
    cot_cmap = mpl.cm.get_cmap('Greys_r')
    cot_norm = mpl.colors.Normalize(vmin=cot_min, vmax=cot_max)

    fig = plt.figure(figsize=(15, 5))

    gs = gridspec.GridSpec(2, 11)

    ax = fig.add_subplot(gs[:, :])
    ax.axis('off')


    ax_map = fig.add_subplot(gs[:, :4])
    divider = make_axes_locatable(ax_map)
    ax_sza = divider.append_axes('right', size='5%', pad=0.0)

    # ax_fdn = fig.add_subplot(gs[0, 5:])
    # ax_fup = fig.add_subplot(gs[1, 5:])

    ax_all = fig.add_subplot(gs[:, 5:])
    # ax_all.axis('off')
    ax_alt = ax_all.twinx()

    ax_cot = ax_all.twinx()

    fig.subplots_adjust(hspace=0.0, wspace=1.0)

    for itrk in range(index_trk+1):

        flt_trk      = flt_sim0.flt_trks[itrk]
        sat_img      = flt_sim0.sat_imgs[itrk]

        vnames_flt = flt_trk.keys()
        vnames_sat = sat_img.keys()

        if itrk == index_trk:
            alpha = 0.9

            if 'cot' in vnames_flt:
                cot0 = flt_trk['cot'][index_pnt]
                # color0 = cot_cmap(cot_norm(cot0))
                # ax_map.scatter(flt_trk['lon'][index_pnt], flt_trk['lat'][index_pnt], facecolor=color0, s=50, lw=1.0, edgecolor=colors['COT'], zorder=3)
                # ax_map.scatter(flt_trk['lon'][index_pnt], flt_trk['lat'][index_pnt], facecolor=colors['COT'], s=40, lw=0.0, zorder=3)
                # ax_map.text(flt_trk['lon'][index_pnt], flt_trk['lat'][index_pnt]-0.28, 'COT %.2f' % cot0, color=colors['COT'], ha='center', va='center', fontsize=12, zorder=4)
                # ax_cot.fill_between(flt_trk['tmhr'][:index_pnt+1], flt_trk['cot'][:index_pnt+1], facecolor=colors['COT'], alpha=0.25, lw=0.0, zorder=1)
            else:
                ax_map.scatter(flt_trk['lon'][index_pnt], flt_trk['lat'][index_pnt], c=flt_trk['alt'][index_pnt], s=40, lw=1.5, facecolor='none', zorder=3, alpha=0.7, vmin=0.0, vmax=4.0, cmap='jet')

            # if 'f-up_mca-ipa' in vnames_flt:
            #     ax_fup.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-up_mca-ipa'][:index_pnt+1], c=colors['RTM IPA'], s=4, lw=0.0, zorder=1)
            # if 'f-up_mca-3d' in vnames_flt:
            #     ax_fup.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-up_mca-3d'][:index_pnt+1] , c=colors['RTM 3D'] , s=4, lw=0.0, zorder=2)

            # if 'f-down_mca-ipa' in vnames_flt:
            #     ax_fdn.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down_mca-ipa'][:index_pnt+1], c=colors['RTM IPA'], s=4, lw=0.0, zorder=1)
            # if 'f-down_mca-3d' in vnames_flt:
            #     ax_fdn.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down_mca-3d'][:index_pnt+1] , c=colors['RTM 3D'] , s=4, lw=0.0, zorder=2)
            # if 'f-down-diffuse_mca-3d' in vnames_flt:
            #     ax_fdn.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down-diffuse_mca-3d'][:index_pnt+1], c=colors['RTM 3D Diffuse'], s=2, lw=0.0, zorder=3)

            logic_solid = (flt_trk['tmhr'][:index_pnt]>tmhr_past) & (flt_trk['tmhr'][:index_pnt]<=tmhr_current)
            logic_trans = np.logical_not(logic_solid)
            ax_map.scatter(flt_trk['lon'][:index_pnt][logic_trans], flt_trk['lat'][:index_pnt][logic_trans], c=flt_trk['alt'][:index_pnt][logic_trans], s=0.5, lw=0.0, zorder=1, vmin=0.0, vmax=4.0, cmap='jet', alpha=0.1)
            ax_map.scatter(flt_trk['lon'][:index_pnt][logic_solid], flt_trk['lat'][:index_pnt][logic_solid], c=flt_trk['alt'][:index_pnt][logic_solid], s=1  , lw=0.0, zorder=2, vmin=0.0, vmax=4.0, cmap='jet')

            # ssfr
            #/----------------------------------------------------------------------------\#
            # if 'f-up_ssfr' in vnames_flt:
            #     ax_fup.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-up_ssfr'][:index_pnt+1]   , c=colors['SSFR']   , s=4, lw=0.0, zorder=3)
            #     ax_fdn.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-up_ssfr'][:index_pnt+1]   , c=colors['SSFR']   , s=4, lw=0.0, zorder=3)
            # if 'f-down_ssfr' in vnames_flt:
            #     ax_fup.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down_ssfr'][:index_pnt+1]   , c=colors['SSFR']   , s=4, lw=0.0, zorder=5)
            #     ax_fdn.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down_ssfr'][:index_pnt+1]   , c=colors['SSFR']   , s=4, lw=0.0, zorder=5)
            #\----------------------------------------------------------------------------/#

            # spn-s
            #/----------------------------------------------------------------------------\#
            if 'f-down-diffuse_spns' in vnames_flt:
                # ax_fup.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down-diffuse_spns'][:index_pnt+1]  , c=colors['SPN-S Diffuse'] , s=2, lw=0.0, zorder=4)
                # ax_fdn.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down-diffuse_spns'][:index_pnt+1]  , c=colors['SPN-S Diffuse'] , s=2, lw=0.0, zorder=4)
                ax_all.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down-diffuse_spns'][:index_pnt+1]  , c=colors['Diffuse↓'] , s=2, lw=0.0, zorder=4)
            if 'f-down-direct_spns' in vnames_flt:
                # ax_fup.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down-direct_spns'][:index_pnt+1]  , c=colors['SPN-S Direct'] , s=2, lw=0.0, zorder=4)
                # ax_fdn.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down-direct_spns'][:index_pnt+1]  , c=colors['SPN-S Direct'] , s=2, lw=0.0, zorder=4)
                ax_all.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down-direct_spns'][:index_pnt+1]  , c=colors['Direct↓'] , s=2, lw=0.0, zorder=4)
            if 'f-down-total_spns' in vnames_flt:
                # ax_fup.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down-total_spns'][:index_pnt+1]  , c=colors['SPN-S Total'] , s=2, lw=0.0, zorder=4)
                # ax_fdn.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down-total_spns'][:index_pnt+1]  , c=colors['SPN-S Total'] , s=2, lw=0.0, zorder=4)
                ax_all.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk['f-down-total_spns'][:index_pnt+1]  , c=colors['Total↓'] , s=2, lw=0.0, zorder=4)
            #\----------------------------------------------------------------------------/#

            ax_alt.fill_between(flt_trk['tmhr'][:index_pnt+1], flt_trk['alt'][:index_pnt+1], facecolor=colors['Altitude'], alpha=0.35, lw=1.0, zorder=0)

            if ('fname_img' in vnames_sat) and ('extent_img' in vnames_sat):
                img = mpl_img.imread(sat_img['fname_img'])
                ax_map.imshow(img, extent=sat_img['extent_img'], origin='upper', aspect='equal', zorder=0)
                region = sat_img['extent_img']

            # if 'cot' in vnames_sat:
            #     ax_map.imshow(sat_img['cot'].T, extent=sat_img['extent'], cmap='Greys_r', origin='lower', vmin=cot_min, vmax=cot_max, alpha=alpha, aspect='auto', zorder=1)

        else:
            alpha = 0.4

            logic = (flt_trk['tmhr']<tmhr_current) & (flt_trk['tmhr']>=tmhr_past)

            if logic.sum() > 0:

                # if 'cot' in vnames_flt:
                #     ax_cot.fill_between(flt_trk['tmhr'], flt_trk['cot'], facecolor=colors['COT'], alpha=0.25, lw=0.0, zorder=1)

                # if 'f-up_mca-ipa' in vnames_flt:
                #     ax_fup.scatter(flt_trk['tmhr'], flt_trk['f-up_mca-ipa'], c=colors['RTM IPA'], s=4, lw=0.0, zorder=1)
                # if 'f-up_mca-3d' in vnames_flt:
                #     ax_fup.scatter(flt_trk['tmhr'], flt_trk['f-up_mca-3d'] , c=colors['RTM 3D'] , s=4, lw=0.0, zorder=2)

                # if 'f-down_mca-ipa' in vnames_flt:
                #     ax_fdn.scatter(flt_trk['tmhr'], flt_trk['f-down_mca-ipa']       , c=colors['RTM IPA']       , s=4, lw=0.0, zorder=1)
                # if 'f-down_mca-3d' in vnames_flt:
                #     ax_fdn.scatter(flt_trk['tmhr'], flt_trk['f-down_mca-3d']        , c=colors['RTM 3D']        , s=4, lw=0.0, zorder=2)
                # if 'f-down-diffuse_mca-3d' in vnames_flt:
                #     ax_fdn.scatter(flt_trk['tmhr'], flt_trk['f-down-diffuse_mca-3d'], c=colors['RTM 3D Diffuse'], s=2, lw=0.0, zorder=3)

                # ssfr
                #/----------------------------------------------------------------------------\#
                # if 'f-up_ssfr' in vnames_flt:
                #     ax_fup.scatter(flt_trk['tmhr'], flt_trk['f-up_ssfr']            , c=colors['SSFR']          , s=4, lw=0.0, zorder=3)
                #     ax_fdn.scatter(flt_trk['tmhr'], flt_trk['f-up_ssfr']            , c=colors['SSFR']          , s=4, lw=0.0, zorder=3)
                # if 'f-down_ssfr' in vnames_flt:
                #     ax_fup.scatter(flt_trk['tmhr'], flt_trk['f-down_ssfr']          , c=colors['SSFR']          , s=4, lw=0.0, zorder=5)
                #     ax_fdn.scatter(flt_trk['tmhr'], flt_trk['f-down_ssfr']          , c=colors['SSFR']          , s=4, lw=0.0, zorder=5)
                #\----------------------------------------------------------------------------/#

                # spn-s
                #/----------------------------------------------------------------------------\#
                if 'f-down-diffuse_spns' in vnames_flt:
                    # ax_fup.scatter(flt_trk['tmhr'], flt_trk['f-down-diffuse_spns']  , c=colors['SPN-S Diffuse'] , s=2, lw=0.0, zorder=4)
                    # ax_fdn.scatter(flt_trk['tmhr'], flt_trk['f-down-diffuse_spns']  , c=colors['SPN-S Diffuse'] , s=2, lw=0.0, zorder=4)
                    ax_all.scatter(flt_trk['tmhr'], flt_trk['f-down-diffuse_spns']  , c=colors['Diffuse↓'] , s=2, lw=0.0, zorder=4)
                if 'f-down-direct_spns' in vnames_flt:
                    # ax_fup.scatter(flt_trk['tmhr'], flt_trk['f-down-direct_spns']  , c=colors['SPN-S Direct'] , s=2, lw=0.0, zorder=4)
                    # ax_fdn.scatter(flt_trk['tmhr'], flt_trk['f-down-direct_spns']  , c=colors['SPN-S Direct'] , s=2, lw=0.0, zorder=4)
                    ax_all.scatter(flt_trk['tmhr'], flt_trk['f-down-direct_spns']  , c=colors['Direct↓'] , s=2, lw=0.0, zorder=4)
                if 'f-down-total_spns' in vnames_flt:
                    # ax_fup.scatter(flt_trk['tmhr'], flt_trk['f-down-total_spns']  , c=colors['SPN-S Total'] , s=2, lw=0.0, zorder=4)
                    # ax_fdn.scatter(flt_trk['tmhr'], flt_trk['f-down-total_spns']  , c=colors['SPN-S Total'] , s=2, lw=0.0, zorder=4)
                    ax_all.scatter(flt_trk['tmhr'], flt_trk['f-down-total_spns']  , c=colors['Total↓'] , s=2, lw=0.0, zorder=4)
                #\----------------------------------------------------------------------------/#

                ax_alt.fill_between(flt_trk['tmhr'], flt_trk['alt'], facecolor=colors['Altitude'], alpha=0.35, lw=1.0, zorder=0)

            logic_solid = logic.copy()
            logic_trans = np.logical_not(logic_solid)
            ax_map.scatter(flt_trk['lon'][logic_trans], flt_trk['lat'][logic_trans], c=flt_trk['alt'][logic_trans], s=0.5, lw=0.0, zorder=1, vmin=0.0, vmax=4.0, cmap='jet', alpha=0.1)
            ax_map.scatter(flt_trk['lon'][logic_solid], flt_trk['lat'][logic_solid], c=flt_trk['alt'][logic_solid], s=1  , lw=0.0, zorder=2, vmin=0.0, vmax=4.0, cmap='jet')

    # ax_fup.axvline(flt_trk['tmhr'][index_pnt], lw=1.0, color='gray')
    # ax_fdn.axvline(flt_trk['tmhr'][index_pnt], lw=1.0, color='gray')
    # ax_all.axvline(flt_trk['tmhr'][index_pnt], lw=1.5, color='gray', ls='--')

    dtime0 = datetime.datetime(1, 1, 1) + datetime.timedelta(days=flt_trk['jday'][index_pnt]-1)
    fig.suptitle('%s (SPN-S at %.2f nm)' % (dtime0.strftime('%Y-%m-%d %H:%M:%S'), flt_sim0.wvl), y=1.02, fontsize=20)

    # map plot settings
    #/----------------------------------------------------------------------------\#
    ax_map.set_xlim(region[:2])
    ax_map.set_ylim(region[2:])
    ax_map.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 180.1, 0.5)))
    ax_map.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 90.1, 0.5)))
    ax_map.set_xlabel('Longitude [$^\circ$]')
    ax_map.set_ylabel('Latitude [$^\circ$]')

    title = '%s at %s' % (flt_sim0.sat_imgs[index_trk]['imager'], er3t.util.jday_to_dtime(flt_sim0.sat_imgs[index_trk]['jday']).strftime('%H:%M'))
    time_diff = np.abs(flt_sim0.sat_imgs[index_trk]['tmhr']-tmhr_current)*3600.0
    if time_diff > 301.0:
        ax_map.set_title(title, color='red')
    else:
        ax_map.set_title(title)
    #\----------------------------------------------------------------------------/#

    # sun elevation plot settings
    #/----------------------------------------------------------------------------\#
    ax_sza.set_ylim((0.0, 90.0))
    ax_sza.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 90.1, 30.0)))
    ax_sza.axhline(90.0-flt_trk['sza'][index_pnt], lw=2.0, color='r')
    ax_sza.xaxis.set_ticks([])
    ax_sza.yaxis.tick_right()
    ax_sza.yaxis.set_label_position('right')
    ax_sza.set_ylabel('Sun Elevation [$^\circ$]', rotation=270.0, labelpad=18)
    #\----------------------------------------------------------------------------/#

    # upwelling flux plot settings
    # =======================================================================================================
    # ax_fup.set_xlim((tmhr_past-0.0000001, tmhr_current+0.0000001))
    # ax_fup.xaxis.set_major_locator(FixedLocator([tmhr_past, tmhr_current-0.5*tmhr_length, tmhr_current]))
    # ax_fup.set_xticklabels(['%.4f' % (tmhr_past), '%.4f' % (tmhr_current-0.5*tmhr_length), '%.4f' % tmhr_current])
    # ax_fup.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 1.1, 0.5)))
    # ax_fup.set_ylim((0.0, 1.5))
    # ax_fup.set_xlabel('UTC [hour]')
    # ax_fup.set_ylabel('$F_\\uparrow [\mathrm{W m^{-2} nm^{-1}}]$')

    # ax_fup.set_zorder(ax_alt.get_zorder()+1)
    # if 'cot' in vnames_flt:
    #     ax_fup.set_zorder(ax_cot.get_zorder()+2)
    # ax_fup.patch.set_visible(False)
    # =======================================================================================================

    # downwelling flux plot settings
    # =======================================================================================================
    # ax_fdn.xaxis.set_ticks([])
    # ax_fdn.set_xlim((tmhr_past-0.0000001, tmhr_current+0.0000001))
    # ax_fdn.set_ylim((1.5, 2.5))
    # ax_fdn.yaxis.set_major_locator(FixedLocator(np.arange(1.5, 2.6, 0.5)))
    # ax_fdn.set_ylabel('$F_\downarrow [\mathrm{W m^{-2} nm^{-1}}]$')
    # ax_fdn.set_zorder(ax_alt.get_zorder()+1)
    # if 'cot' in vnames_flt:
    #     ax_fdn.set_zorder(ax_cot.get_zorder()+2)
    # ax_fdn.patch.set_visible(False)
    # =======================================================================================================

    # altitude plot settings
    #/----------------------------------------------------------------------------\#
    ax_all.set_xlim((tmhr_past-0.0000001, tmhr_current+0.0000001))
    ax_all.xaxis.set_major_locator(FixedLocator([tmhr_past, tmhr_current-0.5*tmhr_length, tmhr_current]))
    ax_all.set_xticklabels(['%.4f' % (tmhr_past), '%.4f' % (tmhr_current-0.5*tmhr_length), '%.4f' % tmhr_current])
    ax_all.set_ylim((0.0, 2.0))
    ax_all.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 2.1, 0.5)))
    ax_all.set_xlabel('UTC [hour]')
    ax_all.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
    if alt_current < 1.0:
        title = 'Longitude %9.4f$^\circ$, Latitude %8.4f$^\circ$, Altitude %6.1f  m' % (lon_current, lat_current, alt_current*1000.0)
    else:
        title = 'Longitude %9.4f$^\circ$, Latitude %8.4f$^\circ$, Altitude %6.4f km' % (lon_current, lat_current, alt_current)
    ax_all.set_title(title)

    ax_alt.set_frame_on(True)
    for spine in ax_alt.spines.values():
        spine.set_visible(False)
    ax_alt.spines['right'].set_visible(True)
    ax_alt.spines['right'].set_color(colors['Altitude'])
    ax_alt.tick_params(axis='y', colors=colors['Altitude'])

    # ax_alt.xaxis.set_ticks([])
    ax_alt.set_ylim((0.0, 6.0))
    ax_alt.yaxis.tick_right()
    ax_alt.yaxis.set_label_position('right')
    ax_alt.set_ylabel('Altitude [km]', rotation=270.0, labelpad=16, color=colors['Altitude'])
    #\----------------------------------------------------------------------------/#

    # cot plot settings
    # =======================================================================================================
    ax_cot.axis('off')
    # ax_cot.spines['right'].set_position(('axes', 1.1))
    # ax_cot.set_frame_on(True)
    # for spine in ax_cot.spines.values():
    #     spine.set_visible(False)
    # ax_cot.spines['right'].set_visible(True)

    # ax_cot.set_ylim((cot_min, cot_max))
    # ax_cot.xaxis.set_ticks([])
    # ax_cot.yaxis.tick_right()
    # ax_cot.yaxis.set_label_position('right')
    # ax_cot.set_ylabel('Cloud Optical Thickness', rotation=270.0, labelpad=18)
    # =======================================================================================================

    # legend plot settings
    #/----------------------------------------------------------------------------\#
    patches_legend = []
    for key in colors.keys():
        if key.lower() != 'altitude':
            patches_legend.append(mpatches.Patch(color=colors[key], label=key))
    # ax_all.legend(handles=patches_legend, loc='upper right', fontsize=14)
    ax_all.legend(handles=patches_legend, bbox_to_anchor=(0., 0.91, 1., .102), loc=3, ncol=len(patches_legend), mode="expand", borderaxespad=0., frameon=False, handletextpad=0.2, fontsize=14)
    #\----------------------------------------------------------------------------/#

    if test:
        plt.show()
        sys.exit()
    else:
        plt.savefig('tmp-graph/%5.5d.png' % n, bbox_inches='tight')
        plt.close(fig)


def plot_video_frame(statements, test=False):

    # extract arguments
    #/----------------------------------------------------------------------------\#
    flt_sim0, index_trk, index_pnt, n = statements
    #\----------------------------------------------------------------------------/#


    # general plot settings
    #/----------------------------------------------------------------------------\#
    vars_plot = OrderedDict()

    vars_plot['Total↓']   = {
            'vname':'f-down-total_spns',
            'color':'black',
            }
    vars_plot['Diffuse↓']   = {
            'vname':'f-down-diffuse_spns',
            'color':'blue',
            }
    vars_plot['Direct↓']   = {
            'vname':'f-down-direct_spns',
            'color':'red',
            }
    vars_plot['Altitude']   = {
            'vname':'alt',
            'color':'orange',
            }
    #\----------------------------------------------------------------------------/#


    # param settings
    #/----------------------------------------------------------------------------\#
    tmhr_current = flt_sim0.flt_trks[index_trk]['tmhr'][index_pnt]
    jday_current= flt_sim0.flt_trks[index_trk]['jday'][index_pnt]
    lon_current = flt_sim0.flt_trks[index_trk]['lon'][index_pnt]
    lat_current = flt_sim0.flt_trks[index_trk]['lat'][index_pnt]
    alt_current = flt_sim0.flt_trks[index_trk]['alt'][index_pnt]
    sza_current = flt_sim0.flt_trks[index_trk]['sza'][index_pnt]
    dtime_current = er3t.util.jday_to_dtime(jday_current)

    tmhr_length  = 0.5
    tmhr_past    = tmhr_current-tmhr_length
    #\----------------------------------------------------------------------------/#


    # flight direction
    #/----------------------------------------------------------------------------\#
    alt_cmap = mpl.cm.get_cmap('jet')
    alt_norm = mpl.colors.Normalize(vmin=0.0, vmax=4.0)

    dlon = flt_sim0.sat_imgs[index_trk]['extent_img'][1] - flt_sim0.sat_imgs[index_trk]['extent_img'][0]
    Nscale = int(dlon/1.3155229999999989 * 100)

    arrow_prop = dict(
            arrowstyle='fancy,head_width=0.6,head_length=0.8',
            shrinkA=0,
            shrinkB=0,
            facecolor='red',
            edgecolor='white',
            linewidth=1.0,
            alpha=0.6,
            relpos=(0.0, 0.0),
            )
    if index_trk == 0 and index_pnt == 0:
        plot_arrow = False
    else:
        if index_pnt == 0:
            lon_before = flt_sim0.flt_trks[index_trk-1]['lon'][-1]
            lat_before = flt_sim0.flt_trks[index_trk-1]['lat'][-1]
        else:
            lon_before = flt_sim0.flt_trks[index_trk]['lon'][index_pnt-1]
            lat_before = flt_sim0.flt_trks[index_trk]['lat'][index_pnt-1]
        dx = lon_current - lon_before
        dy = lat_current - lat_before

        if np.sqrt(dx**2+dy**2)*111000.0 > 20.0:
            plot_arrow = True
            lon_point_to = lon_current + Nscale*dx
            lat_point_to = lat_current + Nscale*dy
        else:
            plot_arrow = False
    #\----------------------------------------------------------------------------/#


    # figure setup
    #/----------------------------------------------------------------------------\#
    fig = plt.figure(figsize=(16, 5))

    gs = gridspec.GridSpec(2, 14)

    ax = fig.add_subplot(gs[:, :])
    ax.axis('off')

    ax_map = fig.add_subplot(gs[:, :6])
    divider = make_axes_locatable(ax_map)
    ax_sza = divider.append_axes('right', size='5%', pad=0.0)

    ax_all = fig.add_subplot(gs[:, 7:])
    ax_alt = ax_all.twinx()

    fig.subplots_adjust(hspace=0.0, wspace=1.0)
    #\----------------------------------------------------------------------------/#

    for itrk in range(index_trk+1):

        flt_trk      = flt_sim0.flt_trks[itrk]
        sat_img      = flt_sim0.sat_imgs[itrk]

        vnames_flt = flt_trk.keys()
        vnames_sat = sat_img.keys()

        if itrk == index_trk:

            if ('fname_img' in vnames_sat) and ('extent_img' in vnames_sat):
                img = mpl_img.imread(sat_img['fname_img'])
                ax_map.imshow(img, extent=sat_img['extent_img'], origin='upper', aspect='equal', zorder=0)
                region = sat_img['extent_img']

            logic_solid = (flt_trk['tmhr'][:index_pnt]>tmhr_past) & (flt_trk['tmhr'][:index_pnt]<=tmhr_current)
            logic_trans = np.logical_not(logic_solid)
            ax_map.scatter(flt_trk['lon'][:index_pnt][logic_trans], flt_trk['lat'][:index_pnt][logic_trans], c=flt_trk['alt'][:index_pnt][logic_trans], s=0.5, lw=0.0, zorder=1, vmin=0.0, vmax=4.0, cmap='jet', alpha=0.1)
            ax_map.scatter(flt_trk['lon'][:index_pnt][logic_solid], flt_trk['lat'][:index_pnt][logic_solid], c=flt_trk['alt'][:index_pnt][logic_solid], s=1  , lw=0.0, zorder=2, vmin=0.0, vmax=4.0, cmap='jet')

            if not plot_arrow:
                ax_map.scatter(lon_current, lat_current, facecolor='none', edgecolor='white', s=40, lw=1.0, zorder=3, alpha=0.6)
                ax_map.scatter(lon_current, lat_current, c=alt_current, s=40, lw=0.0, zorder=3, alpha=0.6, vmin=0.0, vmax=4.0, cmap='jet')
            else:
                color0 = alt_cmap(alt_norm(alt_current))
                arrow_prop['facecolor'] = color0
                arrow_prop['relpos'] = (lon_current, lat_current)
                ax_map.annotate('', xy=(lon_point_to, lat_point_to), xytext=(lon_current, lat_current), arrowprops=arrow_prop, zorder=3)

            for vname_plot in vars_plot.keys():
                var_plot = vars_plot[vname_plot]
                if vname_plot.lower() == 'altitude':
                    ax_alt.fill_between(flt_trk['tmhr'][:index_pnt+1], flt_trk[var_plot['vname']][:index_pnt+1], facecolor=vars_plot[vname_plot]['color'], alpha=0.25, lw=0.0, zorder=0)
                else:
                    if var_plot['vname'] in vnames_flt:
                        ax_all.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk[var_plot['vname']][:index_pnt+1], c=vars_plot[vname_plot]['color'], s=2, lw=0.0, zorder=4)

        else:

            logic_solid = (flt_trk['tmhr']>=tmhr_past) & (flt_trk['tmhr']<tmhr_current)
            logic_trans = np.logical_not(logic_solid)
            ax_map.scatter(flt_trk['lon'][logic_trans], flt_trk['lat'][logic_trans], c=flt_trk['alt'][logic_trans], s=0.5, lw=0.0, zorder=1, vmin=0.0, vmax=4.0, cmap='jet', alpha=0.1)
            ax_map.scatter(flt_trk['lon'][logic_solid], flt_trk['lat'][logic_solid], c=flt_trk['alt'][logic_solid], s=1  , lw=0.0, zorder=2, vmin=0.0, vmax=4.0, cmap='jet')

            if logic_solid.sum() > 0:

                for vname_plot in vars_plot.keys():
                    var_plot = vars_plot[vname_plot]
                    if vname_plot.lower() == 'altitude':
                        ax_alt.fill_between(flt_trk['tmhr'], flt_trk[var_plot['vname']], facecolor=vars_plot[vname_plot]['color'], alpha=0.25, lw=0.0, zorder=0)
                    else:
                        if var_plot['vname'] in vnames_flt:
                            ax_all.scatter(flt_trk['tmhr'], flt_trk[var_plot['vname']], c=vars_plot[vname_plot]['color'] , s=2, lw=0.0, zorder=4)


    # figure settings
    #/----------------------------------------------------------------------------\#
    title_fig = '%s UTC (SPN-S at %d nm)' % (dtime_current.strftime('%Y-%m-%d %H:%M:%S'), flt_sim0.wvl)
    fig.suptitle(title_fig, y=1.02, fontsize=20)
    #\----------------------------------------------------------------------------/#


    # map plot settings
    #/----------------------------------------------------------------------------\#
    ax_map.set_xlim(region[:2])
    ax_map.set_ylim(region[2:])
    ax_map.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 180.1, 0.5)))
    ax_map.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 90.1, 0.5)))
    ax_map.set_xlabel('Longitude [$^\circ$]')
    ax_map.set_ylabel('Latitude [$^\circ$]')

    title_map = '%s at %s UTC' % (flt_sim0.sat_imgs[index_trk]['imager'], er3t.util.jday_to_dtime(flt_sim0.sat_imgs[index_trk]['jday']).strftime('%H:%M'))
    time_diff = np.abs(flt_sim0.sat_imgs[index_trk]['tmhr']-tmhr_current)*3600.0
    if time_diff > 301.0:
        ax_map.set_title(title_map, color='red')
    else:
        ax_map.set_title(title_map)
    #\----------------------------------------------------------------------------/#


    # sun elevation plot settings
    #/----------------------------------------------------------------------------\#
    ax_sza.set_ylim((0.0, 90.0))
    ax_sza.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 90.1, 30.0)))
    ax_sza.axhline(90.0-sza_current, lw=1.5, color='r')
    ax_sza.xaxis.set_ticks([])
    ax_sza.yaxis.tick_right()
    ax_sza.yaxis.set_label_position('right')
    ax_sza.set_ylabel('Sun Elevation [$^\circ$]', rotation=270.0, labelpad=18)
    #\----------------------------------------------------------------------------/#


    # altitude plot settings
    #/----------------------------------------------------------------------------\#
    ax_alt.set_ylim((0.0, 6.0))
    ax_alt.yaxis.tick_right()
    ax_alt.yaxis.set_label_position('right')
    ax_alt.set_ylabel('Altitude [km]', rotation=270.0, labelpad=18, color=vars_plot['Altitude']['color'])

    ax_alt.set_frame_on(True)
    for spine in ax_alt.spines.values():
        spine.set_visible(False)
    ax_alt.spines['right'].set_visible(True)
    ax_alt.spines['right'].set_color(vars_plot['Altitude']['color'])
    ax_alt.tick_params(axis='y', colors=vars_plot['Altitude']['color'])
    #\----------------------------------------------------------------------------/#


    # main time series plot settings
    #/----------------------------------------------------------------------------\#
    ax_all.set_xlim((tmhr_past-0.0000001, tmhr_current+0.0000001))
    ax_all.xaxis.set_major_locator(FixedLocator([tmhr_past, tmhr_current-0.5*tmhr_length, tmhr_current]))
    ax_all.set_xticklabels(['%.4f' % (tmhr_past), '%.4f' % (tmhr_current-0.5*tmhr_length), '%.4f' % tmhr_current])
    ax_all.set_xlabel('Time [hour]')

    ax_all.set_ylim((0.0, 2.0))
    ax_all.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 2.1, 0.5)))
    ax_all.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')

    if alt_current < 1.0:
        title_all = 'Longitude %9.4f$^\circ$, Latitude %8.4f$^\circ$, Altitude %6.1f  m' % (lon_current, lat_current, alt_current*1000.0)
    else:
        title_all = 'Longitude %9.4f$^\circ$, Latitude %8.4f$^\circ$, Altitude %6.4f km' % (lon_current, lat_current, alt_current)
    ax_all.set_title(title_all)

    ax_all.spines['right'].set_visible(False)
    ax_all.set_zorder(ax_alt.get_zorder()+1)
    ax_all.patch.set_visible(False)
    #\----------------------------------------------------------------------------/#


    # legend plot settings
    #/----------------------------------------------------------------------------\#
    patches_legend = []
    for key in vars_plot.keys():
        if key.lower() != 'altitude':
            patches_legend.append(mpatches.Patch(color=vars_plot[key]['color'], label=key))
    ax_all.legend(handles=patches_legend, bbox_to_anchor=(0.03, 0.89, 0.94, .102), loc=3, ncol=len(patches_legend), mode='expand', borderaxespad=0., frameon=True, handletextpad=0.2, fontsize=14)
    #\----------------------------------------------------------------------------/#


    if test:
        plt.show()
        sys.exit()
    else:
        plt.savefig('tmp-graph/%5.5d.png' % n, bbox_inches='tight')
        plt.close(fig)




def main_pre(
        date,
        wvl0=_wavelength_,
        run_rtm=False,
        ):


    # create data directory (for storing data) if the directory does not exist
    #/--------------------------------------------------------------\#
    date_s = date.strftime('%Y%m%d_flt-vid')

    fdir = os.path.abspath('%s/%s' % (_fdir_main_, date_s))
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    #\--------------------------------------------------------------/#


    # read flight data
    #/----------------------------------------------------------------------------\#
    fname_flt = '%s/MAGPIE_SPN-S_%s_v2.h5' % (_fdir_v2_, date_s)
    f_flt = h5py.File(fname_flt, 'r')
    jday   = f_flt['jday'][...]
    sza    = f_flt['sza'][...]
    lon    = f_flt['lon'][...]
    lat    = f_flt['lat'][...]

    logic0 = (~np.isnan(jday) & ~np.isinf(jday)) & \
             (~np.isnan(sza)  & ~np.isinf(sza)) & \
             check_continuity(lon) & \
             check_continuity(lat)

    jday = jday[logic0]
    sza  = sza[logic0]
    lon  = lon[logic0]
    lat  = lat[logic0]

    tmhr   = f_flt['tmhr'][...][logic0]
    alt    = f_flt['alt'][...][logic0]

    tot_flux = f_flt['tot/flux'][...][logic0, :]
    tot_wvl  = f_flt['tot/wvl'][...]
    dif_flux = f_flt['dif/flux'][...][logic0, :]
    dif_wvl  = f_flt['dif/wvl'][...]
    f_flt.close()
    #\----------------------------------------------------------------------------/#


    # download satellite imagery
    #/----------------------------------------------------------------------------\#
    extent = get_extent(lon, lat, margin=0.2)

    dtime_s = er3t.util.jday_to_dtime(jday[0])
    dtime_e = er3t.util.jday_to_dtime(jday[-1])

    if True:
        download_geo_sat_img(
            dtime_s,
            dtime_e=dtime_e,
            extent=extent,
            fdir_out=_fdir_sat_img_,
            )

        download_polar_sat_img(
            dtime_s,
            extent=extent,
            fdir_out=_fdir_sat_img_,
            )

    # get the avaiable satellite data and calculate the time in hour for each file
    fnames_sat_ = sorted(glob.glob('%s/*%sT*Z*.png' % (_fdir_sat_img_, date_s)))
    jday_sat_ = get_jday_sat_img(fnames_sat_)

    jday_sat = np.sort(np.unique(jday_sat_))

    fnames_sat = []

    for jday_sat0 in jday_sat:

        indices = np.where(jday_sat_==jday_sat0)[0]
        fname0 = sorted([fnames_sat_[index] for index in indices])[-1] # pick polar imager over geostationary imager
        fnames_sat.append(fname0)
    #\----------------------------------------------------------------------------/#


    # pre-process the aircraft and satellite data
    #/----------------------------------------------------------------------------\#
    # create a filter to remove invalid data, e.g., out of available satellite data time range,
    # invalid solar zenith angles etc.
    tmhr_interval = 10.0/60.0
    half_interval = tmhr_interval/48.0

    jday_edges = np.append(jday_sat[0]-half_interval, jday_sat[1:]-(jday_sat[1:]-jday_sat[:-1])/2.0)
    jday_edges = np.append(jday_edges, jday_sat[-1]+half_interval)

    logic = (jday>=jday_edges[0]) & (jday<=jday_edges[-1])

    # create python dictionary to store valid flight data
    flt_trk = {}
    flt_trk['jday'] = jday[logic]
    flt_trk['lon']  = lon[logic]
    flt_trk['lat']  = lat[logic]
    flt_trk['sza']  = sza[logic]
    flt_trk['tmhr'] = tmhr[logic]
    flt_trk['alt']  = alt[logic]/1000.0

    flt_trk['f-down-total_spns']   = tot_flux[logic, np.argmin(np.abs(tot_wvl-wvl0))]
    flt_trk['f-down-diffuse_spns'] = dif_flux[logic, np.argmin(np.abs(dif_wvl-wvl0))]
    flt_trk['f-down-direct_spns']  = flt_trk['f-down-total_spns'] - flt_trk['f-down-diffuse_spns']

    # partition the flight track into multiple mini flight track segments
    flt_trks = partition_flight_track(flt_trk, jday_edges, margin_x=1.0, margin_y=1.0)
    #\----------------------------------------------------------------------------/#


    # process satellite imagery
    #/----------------------------------------------------------------------------\#

    # create python dictionary to store corresponding satellite imagery data info
    sat_imgs = []
    for i in range(len(flt_trks)):
        sat_img = {}

        index0  = np.argmin(np.abs(jday_sat-flt_trks[i]['jday0']))
        sat_img['imager'] = os.path.basename(fnames_sat[index0]).split('_')[0].replace('-', ' ')
        sat_img['fname_img']  = fnames_sat[index0]
        sat_img['extent_img'] = extent
        sat_img['jday'] = jday_sat[index0]
        sat_img['tmhr'] = 24.0*(jday_sat[index0]-int(jday_sat[index0]))

        sat_imgs.append(sat_img)
    #\----------------------------------------------------------------------------/#


    # figure
    #/----------------------------------------------------------------------------\#
    if False:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        for sat_img in sat_imgs:
            ax1.axvline(sat_img['tmhr'], lw=1.5, ls='--', color='k')

        for flt_trk in flt_trks:
            ax1.scatter(flt_trk['tmhr'], flt_trk['alt'])
        # cs = ax1.imshow(.T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        # ax1.scatter(x, y, s=6, c='k', lw=0.0)
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


    # generate flt-sat combined file
    #/----------------------------------------------------------------------------\#
    fname='%s/flt_sim_%09.4fnm_%s.pk' % (_fdir_main_, wvl0, date_s)
    sim0 = flt_sim(
            date=date,
            wavelength=wvl0,
            flt_trks=flt_trks,
            sat_imgs=sat_imgs,
            fname=fname,
            overwrite=True,
            overwrite_rtm=run_rtm,
            )
    #\----------------------------------------------------------------------------/#

def main_vid(date, wvl0=_wavelength_):

    date_s = date.strftime('%Y-%m-%d')

    fdir = 'tmp-graph'
    if os.path.exists(fdir):
        os.system('rm -rf %s' % fdir)
    os.makedirs(fdir)

    fname='%s/flt_sim_%09.4fnm_%s.pk' % (_fdir_main_, wvl0, date_s)
    flt_sim0 = flt_sim(fname=fname, overwrite=False)

    Ntrk        = len(flt_sim0.flt_trks)
    indices_trk = np.array([], dtype=np.int32)
    indices_pnt = np.array([], dtype=np.int32)
    for itrk in range(Ntrk):
        indices_trk = np.append(indices_trk, np.repeat(itrk, flt_sim0.flt_trks[itrk]['tmhr'].size))
        indices_pnt = np.append(indices_pnt, np.arange(flt_sim0.flt_trks[itrk]['tmhr'].size))

    Npnt        = indices_trk.size
    indices     = np.arange(Npnt)

    interval = 10
    indices_trk = indices_trk[::interval]
    indices_pnt = indices_pnt[::interval]
    indices     = indices[::interval]

    statements = zip([flt_sim0]*indices_trk.size, indices_trk, indices_pnt, indices)


    with mp.Pool(processes=15) as pool:
        r = list(tqdm(pool.imap(plot_video_frame, statements), total=indices_trk.size))

    # make video
    os.system('ffmpeg -y -framerate 30 -pattern_type glob -i "tmp-graph/*.png" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p magpie_%s.mp4' % date_s)



def figure_arcsix_sat_img_hc(
        fname,
        tag='A2024134.1445',
        ):

    r_wvl = 650
    g_wvl = 555
    b_wvl = 470

    f = h5py.File(fname, 'r')
    lon_1d = f['%s/lon_1d' % tag][...]
    lat_1d = f['%s/lat_1d' % tag][...]

    r = f['%s/ref_%d' % (tag, r_wvl)][...]
    g = f['%s/ref_%d' % (tag, r_wvl)][...]
    b = f['%s/ref_%d' % (tag, r_wvl)][...]

    f.close()

    # lon, lat = np.meshgrid(lon_1d, lat_1d)

    # rgb = np.zeros((lat_1d.size, lon_1d.size, 3), dtype=r.dtype)
    # rgb[..., 0] = r
    # rgb[..., 1] = g
    # rgb[..., 2] = b
    # rgb = rgb[:-1, :-1, :]
    # print(lon_1d.shape)
    # print(lat_1d.shape)
    # print(rgb.shape)

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        # cs = ax1.imshow(rgb, zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        # cs = ax1.pcolormesh(lon, lat, rgb, zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        cs = ax1.pcolor(lon_1d, lat_1d, r[:-1, :-1], zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        # ax1.axis('off')



        # assume you have the following variables parsed for each imager
        #/----------------------------------------------------------------------------\#
        satellite_id = 'Aqua'      # or Terra or SNPP or VJ1
        imager_id    = 'MODIS'     # or VIIRS
        layer_name   = 'TrueColor' # or you name it for distinguishing your image product
        #\----------------------------------------------------------------------------/#

        # retrieve time
        #/----------------------------------------------------------------------------\#
        vartag = 'A2024134.1445' # this is a tag from your HDF5 data
        dtime0 = datetime.datetime.strptime(vartag[1:], '%Y%j.%H%M')
        #\----------------------------------------------------------------------------/#


        # assume you have your plotting ax <ax1>
        # add the following lines after your plotting commands
        #/----------------------------------------------------------------------------\#
        extent = [ax1.get_xlim()[0], ax1.get_xlim()[1], ax1.get_ylim()[0], ax1.get_ylim()[1]]
        ax1.axis('off')
        #\----------------------------------------------------------------------------/#


        # save figure
        #/----------------------------------------------------------------------------\#
        fname_png = '%s-%s_%s_%s_(%.2f,%.2f,%.2f,%.2f).png' % (imager_id, satellite_id, layer_name, dtime0.strftime('%Y-%m-%dT%H:%M:%SZ'), *extent)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig(fname_png, pad_inches=0.0, bbox_inches='tight', metadata=_metadata)
        #\----------------------------------------------------------------------------/#


        #\----------------------------------------------------------------------------/#

        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        # fig.subplots_adjust(hspace=0.3, wspace=0.3)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
    #\----------------------------------------------------------------------------/#


if __name__ == '__main__':


    dates = [
            datetime.datetime(2024, 5, 17), # placeholder for ARCSIX test flight
        ]

    for date in dates[::-1]:

        # prepare flight data
        #/----------------------------------------------------------------------------\#
        main_pre(date)
        #\----------------------------------------------------------------------------/#

        # generate video frames
        #/----------------------------------------------------------------------------\#
        # main_vid(date, wvl0=_wavelength_)
        #\----------------------------------------------------------------------------/#

        pass


    # test
    #/----------------------------------------------------------------------------\#
    # date = dates[-1]
    # date_s = date.strftime('%Y-%m-%d')
    # fname='%s/flt_sim_%09.4fnm_%s.pk' % (_fdir_main_, _wavelength_, date_s)
    # flt_sim0 = flt_sim(fname=fname, overwrite=False)
    # statements = (flt_sim0, 15, 0, 100)
    # plot_video_frame(statements, test=True)
    #\----------------------------------------------------------------------------/#

    pass
