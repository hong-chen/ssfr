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
from scipy import ndimage
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
mpl.use('Agg')


import er3t

_mission_      = 'ARCSIX'
_platform_     = 'p3b'

_hsk_          = 'hsk'
_alp_          = 'alp'
_spns_         = 'spns-a'
_ssfr_         = 'ssfr-a'
_cam_          = 'nac'

_fdir_main_    = 'data/test/arcsix/flt-vid'
_fdir_sat_img_ = 'data/test/arcsix/sat-img'
_wavelength_   = 745.0

_fdir_data_ = 'data/test/processed'


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

                if flt_trk['jday'].size in flt_trk[key].shape:
                    flt_trk_segment[key] = flt_trk[key][logic, ...]
                    if key in ['jday', 'tmhr', 'lon', 'lat', 'alt', 'sza']:
                        flt_trk_segment[key+'0'] = np.nanmean(flt_trk_segment[key])
                else:
                    flt_trk_segment[key] = flt_trk[key]


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
        fname_img = er3t.util.download_worldview_image(dtime_s, extent, satellite=satellite, instrument=instrument, fdir_out=fdir_out, dpi=dpi, run=False)
        if not os.path.exists(fname_img):
            fname_img = er3t.util.download_worldview_image(dtime_s, extent, satellite=satellite, instrument=instrument, fdir_out=fdir_out, dpi=dpi, run=True)
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
        fname_img = er3t.util.download_worldview_image(date, extent, satellite=satellite, instrument=instrument, fdir_out=fdir_out, dpi=dpi, run=False)
        if not os.path.exists(fname_img):
            fname_img = er3t.util.download_worldview_image(date, extent, satellite=satellite, instrument=instrument, fdir_out=fdir_out, dpi=dpi, run=True)

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

    logic = check_continuity(lon, threshold=1.0) & check_continuity(lat, threshold=1.0)
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

def get_jday_cam_img(fnames):

    """
    Get UTC time in hour from the camera file name

    Input:
        fnames: Python list, file paths of all the camera jpg data

    Output:
        jday: numpy array, julian day
    """

    jday = []
    for fname in fnames:
        filename = os.path.basename(fname)
        dtime_s = filename[:20]

        dtime0 = datetime.datetime.strptime(dtime_s, '%Y_%m_%d__%H_%M_%S')
        jday0 = er3t.util.dtime_to_jday(dtime0)
        jday.append(jday0)

    return np.array(jday)




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





def plot_video_frame(statements, test=False):

    # extract arguments
    #/----------------------------------------------------------------------------\#
    flt_sim0, index_trk, index_pnt, n = statements
    #\----------------------------------------------------------------------------/#


    # general plot settings
    #/----------------------------------------------------------------------------\#
    vars_plot = OrderedDict()

    vars_plot['SSFR↑']   = {
            'vname':'f-up_ssfr',
            'color':'red',
            }
    vars_plot['SSFR↓']   = {
            'vname':'f-down_ssfr',
            'color':'blue',
            }
    vars_plot['SPNS Total↓']   = {
            'vname':'f-down-total_spns',
            'color':'green',
            }
    vars_plot['SPNS Diffuse↓']   = {
            'vname':'f-down-diffuse_spns',
            'color':'springgreen',
            }
    vars_plot['TOA↓']   = {
            'vname':'f-down_toa',
            'color':'black',
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
    alt_norm = mpl.colors.Normalize(vmin=0.0, vmax=6.0)

    dlon = flt_sim0.sat_imgs[index_trk]['extent_img'][1] - flt_sim0.sat_imgs[index_trk]['extent_img'][0]
    Nscale = int(dlon/1.3155229999999989 * 15)

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
    fig = plt.figure(figsize=(16, 9))

    gs = gridspec.GridSpec(12, 17)

    ax = fig.add_subplot(gs[:, :])
    ax.axis('off')

    ax_map = fig.add_subplot(gs[:8, :7], aspect='auto')
    divider = make_axes_locatable(ax_map)
    ax_sza = divider.append_axes('right', size='5%', pad=0.0)

    ax_map0 = fig.add_subplot(gs[:5, 9:13])

    ax_img  = fig.add_subplot(gs[:5, 13:])
    ax_img_hist = ax_img.twinx()

    ax_nav  = fig.add_subplot(gs[:2, 7:9])

    ax_wvl  = fig.add_subplot(gs[5:8, 9:])

    ax_tms = fig.add_subplot(gs[9:, :])
    ax_alt = ax_tms.twinx()

    fig.subplots_adjust(hspace=10.0, wspace=10.0)
    #\----------------------------------------------------------------------------/#

    # for itrk in range(max([0, index_trk-3]), index_trk+1):
    for itrk in range(index_trk+1):

        flt_trk      = flt_sim0.flt_trks[itrk]
        sat_img      = flt_sim0.sat_imgs[itrk]

        vnames_flt = flt_trk.keys()
        vnames_sat = sat_img.keys()

        if itrk == index_trk:

            if ('fname_img' in vnames_sat) and ('extent_img' in vnames_sat):
                img = mpl_img.imread(sat_img['fname_img'])
                ax_map.imshow(img, extent=sat_img['extent_img'], origin='upper', aspect='auto', zorder=0)
                rect = mpatches.Rectangle((lon_current-0.25, lat_current-0.25), 0.5, 0.5, lw=1.0, ec='r', fc='none')
                ax_map.add_patch(rect)
                ax_map0.imshow(img, extent=sat_img['extent_img'], origin='upper', aspect='auto', zorder=0)
                region = sat_img['extent_img']

            if ('cam' in vnames_sat):
                ang_cam_offset = -152.0
                fname_cam = sat_img['cam'][index_pnt]
                img = mpl_img.imread(fname_cam)[210:, 550:-650, :]

                # if ('ang_hed' in vnames_flt):
                #     ang_hed0   = flt_trk['ang_hed'][index_pnt]
                # else:
                #     ang_hed0 = 0.0
                # img = ndimage.rotate(img, -ang_hed0+ang_cam_offset, reshape=False)[320:-320, 320:-320]

                img = ndimage.rotate(img, ang_cam_offset, reshape=False)
                img_plot = img.copy()
                img_plot[:, :, 0] = np.int_(img[:, :, 0]/img[:, :, 0].max()*255)
                img_plot[:, :, 1] = np.int_(img[:, :, 1]/img[:, :, 1].max()*255)
                img_plot[:, :, 2] = np.int_(img[:, :, 2]/img[:, :, 2].max()*255)
                img_plot[img_plot>=255] = 255
                img_plot = np.int_(img_plot)
                ax_img.imshow(img_plot, origin='upper', aspect='auto', zorder=0, extent=[5, 255*4.0, 0.0, 0.12])

                ax_img_hist.hist(img[:, :, 0].ravel(), bins=20, histtype='step', lw=0.5, alpha=0.9, density=True, color='r')
                ax_img_hist.hist(img[:, :, 1].ravel(), bins=20, histtype='step', lw=0.5, alpha=0.9, density=True, color='g')
                ax_img_hist.hist(img[:, :, 2].ravel(), bins=20, histtype='step', lw=0.5, alpha=0.9, density=True, color='b')
                ax_img.plot([255, 255], [0, 0.005], color='gray', lw=1.0, ls='-')

            logic_solid = (flt_trk['tmhr'][:index_pnt]>tmhr_past) & (flt_trk['tmhr'][:index_pnt]<=tmhr_current)
            logic_trans = np.logical_not(logic_solid)
            ax_map.scatter(flt_trk['lon'][:index_pnt][logic_trans], flt_trk['lat'][:index_pnt][logic_trans], c=flt_trk['alt'][:index_pnt][logic_trans], s=0.5, lw=0.0, zorder=1, vmin=0.0, vmax=6.0, cmap='jet', alpha=0.1)
            ax_map.scatter(flt_trk['lon'][:index_pnt][logic_solid], flt_trk['lat'][:index_pnt][logic_solid], c=flt_trk['alt'][:index_pnt][logic_solid], s=1  , lw=0.0, zorder=2, vmin=0.0, vmax=6.0, cmap='jet')
            ax_map0.scatter(flt_trk['lon'][:index_pnt][logic_trans], flt_trk['lat'][:index_pnt][logic_trans], c=flt_trk['alt'][:index_pnt][logic_trans], s=2.5, lw=0.0, zorder=1, vmin=0.0, vmax=6.0, cmap='jet', alpha=0.1)
            ax_map0.scatter(flt_trk['lon'][:index_pnt][logic_solid], flt_trk['lat'][:index_pnt][logic_solid], c=flt_trk['alt'][:index_pnt][logic_solid], s=4  , lw=0.0, zorder=2, vmin=0.0, vmax=6.0, cmap='jet')

            if not plot_arrow:
                ax_map.scatter(lon_current, lat_current, facecolor='none', edgecolor='white', s=60, lw=1.0, zorder=3, alpha=0.6)
                ax_map.scatter(lon_current, lat_current, c=alt_current, s=60, lw=0.0, zorder=3, alpha=0.6, vmin=0.0, vmax=6.0, cmap='jet')
                # ax_map0.scatter(lon_current, lat_current, facecolor='none', edgecolor='white', s=60, lw=1.0, zorder=3, alpha=0.6)
                # ax_map0.scatter(lon_current, lat_current, c=alt_current, s=60, lw=0.0, zorder=3, alpha=0.6, vmin=0.0, vmax=6.0, cmap='jet')
            else:
                color0 = alt_cmap(alt_norm(alt_current))
                arrow_prop['facecolor'] = color0
                arrow_prop['relpos'] = (lon_current, lat_current)
                ax_map.annotate('', xy=(lon_point_to, lat_point_to), xytext=(lon_current, lat_current), arrowprops=arrow_prop, zorder=3)
                # ax_map0.annotate('', xy=(lon_point_to, lat_point_to), xytext=(lon_current, lat_current), arrowprops=arrow_prop, zorder=3)
            ax_map0.scatter(lon_current, lat_current, facecolor='none', edgecolor='white', s=60, lw=1.0, zorder=3, alpha=0.6)
            ax_map0.scatter(lon_current, lat_current, c=alt_current, s=60, lw=0.0, zorder=3, alpha=0.6, vmin=0.0, vmax=6.0, cmap='jet')

            ax_nav.axvline(0.0, lw=0.5, color='gray', zorder=1)
            ax_nav.axhline(0.0, lw=0.5, color='gray', zorder=1)
            if ('ang_pit' in vnames_flt) and ('ang_rol' in vnames_flt):
                ang_pit0   = flt_trk['ang_pit'][index_pnt]
                ang_rol0   = flt_trk['ang_rol'][index_pnt]

                x  = np.linspace(-10.0, 10.0, 101)

                slope0  = -np.tan(np.deg2rad(ang_rol0))
                offset0 = ang_pit0
                y0 = slope0*x + offset0

                ax_nav.plot(x[15:-15], y0[15:-15], lw=1.0, color='red', zorder=1, alpha=0.6)
                ax_nav.scatter(x[50], y0[50], lw=0.0, s=40, c='red', zorder=1, alpha=0.6)

                # ax_nav.fill_between(x, y0, y2=-10.0, lw=0.0, color='orange', zorder=0, alpha=0.3)
                # ax_nav.fill_between(x, 10.0, y2=y0, lw=0.0, color='blue', zorder=0, alpha=0.3)
                ax_nav.axhspan(-10.0, 0.0, lw=0.0, color='orange', zorder=0, alpha=0.3)
                ax_nav.axhspan(0.0,  10.0, lw=0.0, color='deepskyblue', zorder=0, alpha=0.3)

                if ('ang_pit_m' in vnames_flt) and ('ang_rol_m' in vnames_flt):
                    ang_pit_m0 = flt_trk['ang_pit_m'][index_pnt]
                    ang_rol_m0 = flt_trk['ang_rol_m'][index_pnt]

                    ang_pit_offset = 4.444537204377897
                    ang_rol_offset = -0.5463839481366073

                    slope1  = -np.tan(np.deg2rad(ang_rol0-ang_rol_m0+ang_rol_offset))
                    offset1 = (ang_pit0-ang_pit_m0+ang_pit_offset)
                    y1 = slope1*x + offset1

                    ax_nav.plot(x[25:-25], y1[25:-25], lw=2.0, color='green', zorder=2, alpha=0.7)


            for vname_plot in vars_plot.keys():
                var_plot = vars_plot[vname_plot]
                if var_plot['vname'].lower() in ['alt']:
                    ax_alt.fill_between(flt_trk['tmhr'][:index_pnt+1], flt_trk[var_plot['vname']][:index_pnt+1], facecolor=vars_plot[vname_plot]['color'], alpha=0.25, lw=0.0, zorder=0)
                elif var_plot['vname'].lower() in ['f-down-total_spns', 'f-down-diffuse_spns']:
                    ax_wvl.scatter(flt_trk['wvl_spns'], flt_trk[var_plot['vname']][index_pnt, :], c=vars_plot[vname_plot]['color'], s=6, lw=0.0, zorder=4)
                    wvl_index = np.argmin(np.abs(flt_trk['wvl_spns']-flt_sim0.wvl))
                    ax_tms.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk[var_plot['vname']][:index_pnt+1, wvl_index], c=vars_plot[vname_plot]['color'], s=4, lw=0.0, zorder=4)
                    ax_wvl.axvline(flt_trk['wvl_spns'][wvl_index], color=vars_plot[vname_plot]['color'], ls='-', lw=1.0, alpha=0.5, zorder=0)

                elif var_plot['vname'].lower() in ['f-down_ssfr']:
                    ax_wvl.scatter(flt_trk['wvl_ssfr_zen'], flt_trk[var_plot['vname']][index_pnt, :], c=vars_plot[vname_plot]['color'], s=6, lw=0.0, zorder=4)
                    wvl_index = np.argmin(np.abs(flt_trk['wvl_ssfr_zen']-flt_sim0.wvl))
                    ax_tms.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk[var_plot['vname']][:index_pnt+1, wvl_index], c=vars_plot[vname_plot]['color'], s=4, lw=0.0, zorder=4)
                    ax_wvl.axvline(flt_trk['wvl_ssfr_zen'][wvl_index], color=vars_plot[vname_plot]['color'], ls='-', lw=1.0, alpha=0.5, zorder=0)

                elif var_plot['vname'].lower() in ['f-up_ssfr']:
                    ax_wvl.scatter(flt_trk['wvl_ssfr_nad'], flt_trk[var_plot['vname']][index_pnt, :], c=vars_plot[vname_plot]['color'], s=6, lw=0.0, zorder=4)
                    wvl_index = np.argmin(np.abs(flt_trk['wvl_ssfr_nad']-flt_sim0.wvl))
                    ax_tms.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk[var_plot['vname']][:index_pnt+1, wvl_index], c=vars_plot[vname_plot]['color'], s=4, lw=0.0, zorder=4)
                    ax_wvl.axvline(flt_trk['wvl_ssfr_nad'][wvl_index], color=vars_plot[vname_plot]['color'], ls='-', lw=1.0, alpha=0.5, zorder=0)
                elif var_plot['vname'].lower() in ['f-down_toa']:
                    ax_wvl.scatter(flt_trk['wvl_ssfr_zen'], flt_trk[var_plot['vname']]*np.cos(np.deg2rad(sza_current)), c=vars_plot[vname_plot]['color'], s=6, lw=0.0, zorder=1, alpha=0.6)
                    wvl_index = np.argmin(np.abs(flt_trk['wvl_ssfr_zen']-flt_sim0.wvl))
                    ax_tms.scatter(flt_trk['tmhr'][:index_pnt+1], np.cos(np.deg2rad(flt_trk['sza'][:index_pnt+1]))*flt_trk[var_plot['vname']][wvl_index], c=vars_plot[vname_plot]['color'], s=4, lw=0.0, zorder=1, alpha=0.6)
                else:
                    ax_tms.scatter(flt_trk['tmhr'][:index_pnt+1], flt_trk[var_plot['vname']][:index_pnt+1], c=vars_plot[vname_plot]['color'] , s=2, lw=0.0, zorder=4)

        else:

            logic_solid = (flt_trk['tmhr']>=tmhr_past) & (flt_trk['tmhr']<tmhr_current)
            logic_trans = np.logical_not(logic_solid)
            ax_map.scatter(flt_trk['lon'][logic_trans], flt_trk['lat'][logic_trans], c=flt_trk['alt'][logic_trans], s=0.5, lw=0.0, zorder=1, vmin=0.0, vmax=6.0, cmap='jet', alpha=0.1)
            ax_map.scatter(flt_trk['lon'][logic_solid], flt_trk['lat'][logic_solid], c=flt_trk['alt'][logic_solid], s=1  , lw=0.0, zorder=2, vmin=0.0, vmax=6.0, cmap='jet')
            ax_map0.scatter(flt_trk['lon'][logic_trans], flt_trk['lat'][logic_trans], c=flt_trk['alt'][logic_trans], s=2.5, lw=0.0, zorder=1, vmin=0.0, vmax=6.0, cmap='jet', alpha=0.1)
            ax_map0.scatter(flt_trk['lon'][logic_solid], flt_trk['lat'][logic_solid], c=flt_trk['alt'][logic_solid], s=4  , lw=0.0, zorder=2, vmin=0.0, vmax=6.0, cmap='jet')

            if logic_solid.sum() > 0:

                for vname_plot in vars_plot.keys():
                    var_plot = vars_plot[vname_plot]

                    if var_plot['vname'].lower() in ['alt']:
                        ax_alt.fill_between(flt_trk['tmhr'], flt_trk[var_plot['vname']], facecolor=vars_plot[vname_plot]['color'], alpha=0.25, lw=0.0, zorder=0)

                    elif var_plot['vname'].lower() in ['f-down-total_spns', 'f-down-diffuse_spns']:
                        wvl_index = np.argmin(np.abs(flt_trk['wvl_spns']-flt_sim0.wvl))
                        ax_tms.scatter(flt_trk['tmhr'], flt_trk[var_plot['vname']][:, wvl_index], c=vars_plot[vname_plot]['color'], s=4, lw=0.0, zorder=4)

                    elif var_plot['vname'].lower() in ['f-down_ssfr']:
                        wvl_index = np.argmin(np.abs(flt_trk['wvl_ssfr_zen']-flt_sim0.wvl))
                        ax_tms.scatter(flt_trk['tmhr'], flt_trk[var_plot['vname']][:, wvl_index], c=vars_plot[vname_plot]['color'], s=4, lw=0.0, zorder=4)
                    elif var_plot['vname'].lower() in ['f-up_ssfr']:
                        wvl_index = np.argmin(np.abs(flt_trk['wvl_ssfr_nad']-flt_sim0.wvl))
                        ax_tms.scatter(flt_trk['tmhr'], flt_trk[var_plot['vname']][:, wvl_index], c=vars_plot[vname_plot]['color'], s=4, lw=0.0, zorder=4)
                    elif var_plot['vname'].lower() in ['f-down_toa']:
                        wvl_index = np.argmin(np.abs(flt_trk['wvl_ssfr_zen']-flt_sim0.wvl))
                        ax_tms.scatter(flt_trk['tmhr'], np.cos(np.deg2rad(flt_trk['sza']))*flt_trk[var_plot['vname']][wvl_index], c=vars_plot[vname_plot]['color'], s=4, lw=0.0, zorder=1, alpha=0.6)
                    else:
                        ax_tms.scatter(flt_trk['tmhr'], flt_trk[var_plot['vname']], c=vars_plot[vname_plot]['color'] , s=2, lw=0.0, zorder=4)


    # figure settings
    #/----------------------------------------------------------------------------\#
    title_fig = '%s UTC' % (dtime_current.strftime('%Y-%m-%d %H:%M:%S'))
    fig.suptitle(title_fig, y=0.96, fontsize=20)
    #\----------------------------------------------------------------------------/#


    # map plot settings
    #/----------------------------------------------------------------------------\#
    ax_map.set_xlim(region[:2])
    ax_map.set_ylim(region[2:])
    ax_map.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 180.1, 2.0)))
    ax_map.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 90.1, 2.0)))
    ax_map.set_xlabel('Longitude [$^\circ$]')
    ax_map.set_ylabel('Latitude [$^\circ$]')

    title_map = '%s at %s UTC' % (flt_sim0.sat_imgs[index_trk]['imager'], er3t.util.jday_to_dtime(flt_sim0.sat_imgs[index_trk]['jday']).strftime('%H:%M'))
    time_diff = np.abs(flt_sim0.sat_imgs[index_trk]['tmhr']-tmhr_current)*3600.0
    if time_diff > 301.0:
        ax_map.set_title(title_map, color='red')
    else:
        ax_map.set_title(title_map)
    #\----------------------------------------------------------------------------/#


    # map0 plot settings
    #/----------------------------------------------------------------------------\#
    ax_map0.set_xlim((lon_current-0.25, lon_current+0.25))
    ax_map0.set_ylim((lat_current-0.25, lat_current+0.25))
    # ax_map0.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 180.1, 2.0)))
    # ax_map0.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 90.1, 2.0)))
    # ax_map0.set_xlabel('Longitude [$^\circ$]')
    # ax_map0.set_ylabel('Latitude [$^\circ$]')
    ax_map0.axis('off')

    title_map0 = 'False Color'
    time_diff = np.abs(flt_sim0.sat_imgs[index_trk]['tmhr']-tmhr_current)*3600.0
    if time_diff > 301.0:
        ax_map0.set_title(title_map0, color='red')
    else:
        ax_map0.set_title(title_map0)
    #\----------------------------------------------------------------------------/#


    # camera image plot settings
    #/----------------------------------------------------------------------------\#
    jday_cam  = get_jday_cam_img([fname_cam])[0]
    dtime_cam = er3t.util.jday_to_dtime(jday_cam)

    title_img = 'Camera at %s UTC' % (dtime_cam.strftime('%H:%M:%S'))
    time_diff = np.abs(jday_current-jday_cam)*86400.0
    if time_diff > 301.0:
        ax_img.set_title(title_img, color='red')
    else:
        ax_img.set_title(title_img)

    ax_img.axis('off')

    ax_img_hist.set_xlim((5, 255*4.0))
    ax_img_hist.set_ylim((0.0, 0.12))
    ax_img_hist.axis('off')

    ax_img.axis('off')
    #\----------------------------------------------------------------------------/#


    # navigation plot settings
    #/----------------------------------------------------------------------------\#
    ax_nav.set_xlim((-10.0, 10.0))
    ax_nav.set_ylim((-10.0, 10.0))
    ax_nav.axis('off')
    #\----------------------------------------------------------------------------/#


    # specta plot setting
    #/----------------------------------------------------------------------------\#
    ax_wvl.set_xlim((200, 2200))
    ax_wvl.set_ylim((0.0, 2.0))
    ax_wvl.xaxis.set_major_locator(FixedLocator(np.arange(0, 2401, 400)))
    ax_wvl.xaxis.set_minor_locator(FixedLocator(np.arange(0, 2401, 100)))
    ax_wvl.set_xlabel('Wavelength [nm]')
    ax_wvl.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 2.1, 0.5)))
    ax_wvl.set_ylabel('Flux [$\mathrm{W m^{-2} nm^{-1}}$]')

    # title_img = 'Spec' % (er3t.util.jday_to_dtime(flt_sim0.sat_imgs[index_trk]['jday']).strftime('%H:%M:%S'))
    # ax_wvl.set_title(title_img)
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
    ax_alt.set_ylim((0.0, 8.0))
    ax_alt.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 8.1, 2.0)))
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
    ax_tms.grid()
    ax_tms.set_xlim((tmhr_past-0.0000001, tmhr_current+0.0000001))
    ax_tms.xaxis.set_major_locator(FixedLocator([tmhr_past, tmhr_current-0.5*tmhr_length, tmhr_current]))
    ax_tms.xaxis.set_minor_locator(FixedLocator(np.arange(tmhr_past, tmhr_current+0.001, 5.0/60.0)))
    ax_tms.set_xticklabels(['%.4f' % (tmhr_past), '%.4f' % (tmhr_current-0.5*tmhr_length), '%.4f' % tmhr_current])
    ax_tms.set_xlabel('Time [hour]')

    ax_tms.set_ylim((0.0, 2.0))
    ax_tms.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 2.1, 0.5)))
    ax_tms.set_ylabel('Flux [$\mathrm{W m^{-2} nm^{-1}}$]')

    if alt_current < 1.0:
        # title_all = 'Longitude %9.4f$^\circ$, Latitude %8.4f$^\circ$, Altitude %6.1f  m (Wavelength %.1f nm)' % (lon_current, lat_current, alt_current*1000.0, flt_sim0.wvl)
        title_all = 'Longitude %9.4f$^\circ$, Latitude %8.4f$^\circ$, Altitude %6.1f  m' % (lon_current, lat_current, alt_current*1000.0)
    else:
        # title_all = 'Longitude %9.4f$^\circ$, Latitude %8.4f$^\circ$, Altitude %6.4f km (Wavelength %.1f nm)' % (lon_current, lat_current, alt_current, flt_sim0.wvl)
        title_all = 'Longitude %9.4f$^\circ$, Latitude %8.4f$^\circ$, Altitude %6.4f km' % (lon_current, lat_current, alt_current)
    ax_tms.set_title(title_all)

    ax_tms.spines['right'].set_visible(False)
    ax_tms.set_zorder(ax_alt.get_zorder()+1)
    ax_tms.patch.set_visible(False)
    #\----------------------------------------------------------------------------/#

    text1 = 'Acknowledgements:\n \
instruments engineered by Jeffery Drouet and Sebastian Schmidt\n \
instruments calibrated by Hong Chen, Yu-Wen Chen, and Ken Hirata\n \
instrument data collected by Arabella Chamberlain\n \
'
    ax.annotate(text1, xy=(0.0, 0.28), fontsize=8, color='gray', xycoords='axes fraction', ha='left', va='top')

    text2 = 'Acknowledgements:\n \
satellite imagery processed by Vikas Nataraja\n \
instrument data processed by Hong Chen, Yu-Wen Chen, and Ken Hirata\n \
video created by Hong Chen\n \
'
    ax.annotate(text2, xy=(1.0, 0.28), fontsize=8, color='gray', xycoords='axes fraction', ha='right', va='top')
    #\----------------------------------------------------------------------------/#


    # legend plot settings
    #/----------------------------------------------------------------------------\#
    patches_legend = []
    for key in vars_plot.keys():
        if key.lower() != 'altitude':
            patches_legend.append(mpatches.Patch(color=vars_plot[key]['color'], label=key))
    # ax_tms.legend(handles=patches_legend, bbox_to_anchor=(0.03, 1.23, 0.94, .102), loc=3, ncol=len(patches_legend), mode='expand', borderaxespad=0., frameon=True, handletextpad=0.2, fontsize=14)
    ax_wvl.legend(handles=patches_legend, loc='upper right', fontsize=10)
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
        time_step=1,
        wvl_step_spns=10,
        wvl_step_ssfr=5,
        ):


    # create data directory (for storing data) if the directory does not exist
    #/----------------------------------------------------------------------------\#
    date_s = date.strftime('%Y%m%d')

    fdir = os.path.abspath('%s/%s' % (_fdir_main_, date_s))
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    #\----------------------------------------------------------------------------/#


    # read data
    #/----------------------------------------------------------------------------\#

    # read in aircraft hsk data
    #/--------------------------------------------------------------\#
    fname_flt = '%s/%s-%s_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _hsk_.upper(), _platform_.upper(), date_s)

    f_flt = h5py.File(fname_flt, 'r')
    jday   = f_flt['jday'][...]
    sza    = f_flt['sza'][...]
    lon    = f_flt['lon'][...]
    lat    = f_flt['lat'][...]

    logic0 = (~np.isnan(jday) & ~np.isinf(jday))  & \
             (~np.isnan(sza)  & ~np.isinf(sza))   & \
             check_continuity(lon, threshold=1.0) & \
             check_continuity(lat, threshold=1.0)

    # print(jday[~logic0])
    # print(sza[~logic0])
    # print(lon[~logic0])
    # print(lat[~logic0])

    jday = jday[logic0][::time_step]
    sza  = sza[logic0][::time_step]
    lon  = lon[logic0][::time_step]
    lat  = lat[logic0][::time_step]

    tmhr   = f_flt['tmhr'][...][logic0][::time_step]
    alt    = f_flt['alt'][...][logic0][::time_step]

    f_flt.close()
    #\--------------------------------------------------------------/#
    # print(tmhr.shape)
    # print(alt.shape)
    # print(jday.shape)
    # print(sza.shape)
    # print(lon.shape)
    # print(lat.shape)
    # print(tmhr.shape)
    # print(alt.shape)


    # read in nav data
    #/--------------------------------------------------------------\#
    fname_nav = '%s/%s-%s_%s_%s_v1.h5' % (_fdir_data_, _mission_.upper(), _alp_.upper(), _platform_.upper(), date_s)
    f_nav = h5py.File(fname_nav, 'r')
    ang_hed = f_nav['ang_hed'][...][logic0][::time_step]
    ang_pit = f_nav['ang_pit_s'][...][logic0][::time_step]
    ang_rol = f_nav['ang_rol_s'][...][logic0][::time_step]
    ang_pit_m = f_nav['ang_pit_m'][...][logic0][::time_step]
    ang_rol_m = f_nav['ang_rol_m'][...][logic0][::time_step]
    f_nav.close()
    #\--------------------------------------------------------------/#


    # read in spns data
    #/--------------------------------------------------------------\#
    fname_spns = '%s/%s-%s_%s_%s_v2.h5' % (_fdir_data_, _mission_.upper(), _spns_.upper(), _platform_.upper(), date_s)
    f_spns = h5py.File(fname_spns, 'r')
    spns_tot_flux = f_spns['tot/flux'][...][logic0, :][::time_step, ::wvl_step_spns]
    spns_tot_wvl  = f_spns['tot/wvl'][...][::wvl_step_spns]
    spns_dif_flux = f_spns['dif/flux'][...][logic0, :][::time_step, ::wvl_step_spns]
    spns_dif_wvl  = f_spns['dif/wvl'][...][::wvl_step_spns]
    f_spns.close()
    #\--------------------------------------------------------------/#
    # print(spns_tot_flux.shape)
    # print(spns_tot_wvl.shape)
    # print(spns_dif_flux.shape)
    # print(spns_dif_wvl.shape)


    # read in ssfr data
    #/--------------------------------------------------------------\#
    fname_ssfr = '%s/%s-%s_%s_%s_v2.h5' % (_fdir_data_, _mission_.upper(), _ssfr_.upper(), _platform_.upper(), date_s)
    which_dset = 'dset1'
    f_ssfr = h5py.File(fname_ssfr, 'r')
    ssfr_zen_flux = f_ssfr['%s/flux_zen' % which_dset][...][logic0, :][::time_step, ::wvl_step_ssfr]
    ssfr_zen_wvl  = f_ssfr['%s/wvl_zen'  % which_dset][...][::wvl_step_ssfr]
    ssfr_nad_flux = f_ssfr['%s/flux_nad' % which_dset][...][logic0, :][::time_step, ::wvl_step_ssfr]
    ssfr_nad_wvl  = f_ssfr['%s/wvl_nad'  % which_dset][...][::wvl_step_ssfr]
    ssfr_zen_toa  = f_ssfr['%s/toa0'     % which_dset][...][::wvl_step_ssfr]*er3t.util.cal_sol_fac(date)
    f_ssfr.close()
    #\--------------------------------------------------------------/#
    # print(ssfr_zen_flux.shape)
    # print(ssfr_zen_wvl.shape)
    # print(ssfr_nad_flux.shape)
    # print(ssfr_nad_wvl.shape)


    # process camera imagery
    #/----------------------------------------------------------------------------\#
    fdir_cam = '%s/%s-%s_%s_%s_v1' % (_fdir_data_, _mission_.upper(), _cam_.upper(), _platform_.upper(), date_s)
    date_cam_s = date.strftime('%Y_%m_%d')
    fnames_cam = sorted(glob.glob('%s/%s__*.jpg' % (fdir_cam, date_cam_s)))
    jday_cam = get_jday_cam_img(fnames_cam)
    #\----------------------------------------------------------------------------/#


    # process satellite imagery
    #/----------------------------------------------------------------------------\#
    extent = get_extent(lon, lat, margin=0.2)

    dtime_s = er3t.util.jday_to_dtime(jday[0])
    dtime_e = er3t.util.jday_to_dtime(jday[-1])

    if False:
        # download_geo_sat_img(
        #     dtime_s,
        #     dtime_e=dtime_e,
        #     extent=extent,
        #     fdir_out=_fdir_sat_img_,
        #     )

        download_polar_sat_img(
            dtime_s,
            extent=extent,
            fdir_out=_fdir_sat_img_,
            )

    # get the avaiable satellite data and calculate the time in hour for each file
    date_sat_s  = date.strftime('%Y-%m-%d')
    fnames_sat_ = sorted(glob.glob('%s/*%sT*Z*.png' % (_fdir_sat_img_, date_sat_s)))
    jday_sat_ = get_jday_sat_img(fnames_sat_)

    jday_sat = np.sort(np.unique(jday_sat_))

    fnames_sat = []

    for jday_sat0 in jday_sat:

        indices = np.where(jday_sat_==jday_sat0)[0]
        fname0 = sorted([fnames_sat_[index] for index in indices])[-1] # pick polar imager over geostationary imager
        fnames_sat.append(fname0)
    #\----------------------------------------------------------------------------/#
    # print(jday_sat)
    # print(fnames_sat)


    # pre-process the aircraft and satellite data
    #/----------------------------------------------------------------------------\#
    # create a filter to remove invalid data, e.g., out of available satellite data time range,
    # invalid solar zenith angles etc.
    tmhr_interval = 10.0/60.0
    half_interval = tmhr_interval/48.0

    # jday_edges = np.append(jday_sat[0]-half_interval, jday_sat[1:]-(jday_sat[1:]-jday_sat[:-1])/2.0)
    # jday_edges = np.append(jday_edges, jday_sat[-1]+half_interval)

    # jday_s = ((jday_sat[0]  * 86400.0) // (half_interval*86400.0)    ) * (half_interval*86400.0) / 86400.0
    # jday_e = ((jday_sat[-1] * 86400.0) // (half_interval*86400.0) + 1) * (half_interval*86400.0) / 86400.0
    jday_s = ((jday[0]  * 86400.0) // (half_interval*86400.0) + 1) * (half_interval*86400.0) / 86400.0
    jday_e = ((jday[-1] * 86400.0) // (half_interval*86400.0)    ) * (half_interval*86400.0) / 86400.0

    jday_edges = np.arange(jday_s, jday_e+half_interval, half_interval*2.0)

    logic = (jday>=jday_s) & (jday<=jday_e)

    # create python dictionary to store valid flight data
    flt_trk = {}
    flt_trk['jday'] = jday[logic]
    flt_trk['lon']  = lon[logic]
    flt_trk['lat']  = lat[logic]
    flt_trk['sza']  = sza[logic]
    flt_trk['tmhr'] = tmhr[logic]
    flt_trk['alt']  = alt[logic]/1000.0
    flt_trk['ang_hed'] = ang_hed[logic]
    flt_trk['ang_pit'] = ang_pit[logic]
    flt_trk['ang_rol'] = ang_rol[logic]
    flt_trk['ang_pit_m'] = ang_pit_m[logic]
    flt_trk['ang_rol_m'] = ang_rol_m[logic]

    flt_trk['f-down-total_spns']   = spns_tot_flux[logic, :]
    flt_trk['f-down-diffuse_spns'] = spns_dif_flux[logic, :]
    flt_trk['f-down-direct_spns']  = flt_trk['f-down-total_spns'] - flt_trk['f-down-diffuse_spns']
    flt_trk['wvl_spns'] = spns_tot_wvl

    flt_trk['f-down_ssfr']  = ssfr_zen_flux[logic, :]
    flt_trk['f-up_ssfr']    = ssfr_nad_flux[logic, :]
    flt_trk['wvl_ssfr_zen'] = ssfr_zen_wvl
    flt_trk['wvl_ssfr_nad'] = ssfr_nad_wvl
    flt_trk['f-down_toa']   = ssfr_zen_toa

    # partition the flight track into multiple mini flight track segments
    flt_trks = partition_flight_track(flt_trk, jday_edges, margin_x=1.0, margin_y=1.0)
    #\----------------------------------------------------------------------------/#

    # process imagery
    #/----------------------------------------------------------------------------\#
    # create python dictionary to store corresponding satellite imagery data info
    #/--------------------------------------------------------------\#
    sat_imgs = []
    for i in range(len(flt_trks)):
        sat_img = {}

        index0  = np.argmin(np.abs(jday_sat-flt_trks[i]['jday0']))
        sat_img['imager'] = os.path.basename(fnames_sat[index0]).split('_')[0].replace('-', ' ')
        sat_img['fname_img']  = fnames_sat[index0]
        sat_img['extent_img'] = extent
        sat_img['jday'] = jday_sat[index0]
        sat_img['tmhr'] = 24.0*(jday_sat[index0]-int(jday_sat[index0]))

        sat_img['cam'] = []
        for j in range(flt_trks[i]['jday'].size):
            index_cam = np.argmin(np.abs(jday_cam-flt_trks[i]['jday'][j]))
            sat_img['cam'].append(fnames_cam[index_cam])

        sat_imgs.append(sat_img)
    #\--------------------------------------------------------------/#


    # generate flt-sat combined file
    #/----------------------------------------------------------------------------\#
    fname = '%s/%s-FLT-VID_%s_%s_v0.pk' % (_fdir_main_, _mission_.upper(), _platform_.upper(), date_s)
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

def main_vid(
        date,
        wvl0=_wavelength_
        ):

    date_s = date.strftime('%Y%m%d')

    fdir = 'tmp-graph'
    if os.path.exists(fdir):
        os.system('rm -rf %s' % fdir)
    os.makedirs(fdir)

    fname = '%s/%s-FLT-VID_%s_%s_v0.pk' % (_fdir_main_, _mission_.upper(), _platform_.upper(), date_s)
    flt_sim0 = flt_sim(fname=fname, overwrite=False)

    Ntrk        = len(flt_sim0.flt_trks)
    indices_trk = np.array([], dtype=np.int32)
    indices_pnt = np.array([], dtype=np.int32)
    for itrk in range(Ntrk):
        indices_trk = np.append(indices_trk, np.repeat(itrk, flt_sim0.flt_trks[itrk]['tmhr'].size))
        indices_pnt = np.append(indices_pnt, np.arange(flt_sim0.flt_trks[itrk]['tmhr'].size))

    Npnt        = indices_trk.size
    indices     = np.arange(Npnt)

    interval = 5
    indices_trk = indices_trk[::interval]
    indices_pnt = indices_pnt[::interval]
    indices     = indices[::interval]

    statements = zip([flt_sim0]*indices_trk.size, indices_trk, indices_pnt, indices)

    with mp.Pool(processes=15) as pool:
        r = list(tqdm(pool.imap(plot_video_frame, statements), total=indices_trk.size))

    # make video
    fname_mp4 = '%s-FLT-VID_%s_%s.mp4' % (_mission_.upper(), _platform_.upper(), date_s)
    os.system('ffmpeg -y -framerate 30 -pattern_type glob -i "tmp-graph/*.png" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p %s' % fname_mp4)



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
            # datetime.datetime(2024, 5, 17), # placeholder for ARCSIX test flight
            datetime.datetime(2018, 9, 30), # for test only
        ]

    for date in dates[::-1]:

        # prepare flight data
        #/----------------------------------------------------------------------------\#
        main_pre(date)
        #\----------------------------------------------------------------------------/#

        # generate video frames
        #/----------------------------------------------------------------------------\#
        main_vid(date, wvl0=_wavelength_)
        #\----------------------------------------------------------------------------/#

        pass

    sys.exit()

    # test
    #/----------------------------------------------------------------------------\#
    date = dates[-1]
    date_s = date.strftime('%Y%m%d')
    fname = '%s/%s-FLT-VID_%s_%s_v0.pk' % (_fdir_main_, _mission_.upper(), _platform_.upper(), date_s)
    flt_sim0 = flt_sim(fname=fname, overwrite=False)
    # statements = (flt_sim0, 0, 243, 1730)
    statements = (flt_sim0, 1, 443, 1730)
    plot_video_frame(statements, test=True)
    #\----------------------------------------------------------------------------/#

    pass
