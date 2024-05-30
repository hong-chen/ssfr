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
import copy
import time
from collections import OrderedDict
import datetime
import multiprocessing as mp
import pickle
from tqdm import tqdm
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
from PIL import ImageFile
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.image as mpl_img
import matplotlib.axes as maxes
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy
import cartopy.crs as ccrs
mpl.use('Agg')


import er3t


ImageFile.LOAD_TRUNCATED_IMAGES = True

_mission_      = 'arcsix'
_platform_     = 'p3b'

_hsk_          = 'hsk'
_alp_          = 'alp'
_spns_         = 'spns-a'
_ssfr1_        = 'ssfr-a'
_ssfr2_        = 'ssfr-b'
_cam_          = 'nac'

_fdir_main_       = 'data/%s/flt-vid' % _mission_
_fdir_sat_img_    = 'data/%s/sat-img' % _mission_
_fdir_cam_img_    = 'data/%s/2024-Spring/p3' % _mission_
_wavelength_      = 555.0

_fdir_sat_img_vn_ = 'data/%s/sat-img-vn' % _mission_

_preferred_region_ = 'lincoln_sea'

_fdir_data_ = 'data/%s/processed' % _mission_
_fdir_tmp_graph_ = 'tmp-graph_flt-vid'

_title_extra_ = 'ARCSIX Research Flight #1'

_tmhr_range_ = {
        '20240517': [19.20, 23.00],
        '20240521': [14.80, 17.50],
        '20240524': [ 9.90, 17.90],
        '20240528': [11.90, 18.60],
        '20240530': [10.90, 18.30],
        }


def download_geo_sat_img(
        dtime_s,
        dtime_e=None,
        extent=[-60.5, -58.5, 12, 14],
        satellite='GOES-East',
        instrument='ABI',
        # layer_name='Band2_Red_Visible_1km',
        layer_name=None,
        fdir_out='%s/sat_img' % _fdir_main_,
        dpi=200,
        ):

    if dtime_e is None:
        dtime_e = datetime.datetime(date_s.year, date_s.month, date_s.day, 23, 59, date_s.second)

    while dtime_s <= dtime_e:
        fname_img = er3t.util.download_worldview_image(dtime_s, extent, satellite=satellite, instrument=instrument, fdir_out=fdir_out, dpi=dpi, run=False, layer_name0=layer_name)
        if not os.path.exists(fname_img):
            fname_img = er3t.util.download_worldview_image(dtime_s, extent, satellite=satellite, instrument=instrument, fdir_out=fdir_out, dpi=dpi, run=True, layer_name0=layer_name)
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

def check_continuity(data, threshold=0.1):

    data = np.append(data[0], data)

    return (np.abs(data[1:]-data[:-1]) < threshold)

def partition_flight_track(flt_trk, jday_edges, margin_x=0.2, margin_y=0.2):

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

    deg_x = (lon_max-lon_min)*np.cos(np.deg2rad(lat_c))
    deg_y = (lat_max-lat_min)

    deg_half = (max([deg_x, deg_y])/2.0 * (1.0+margin/2.0))/np.cos(np.deg2rad(lat_c))

    lon0 = lon_c-deg_half
    lon1 = lon_c+deg_half
    lat0 = lat_c-deg_half
    lat1 = lat_c+deg_half

    extent = [lon0, lon1, max([lat0, 35.0]), min([lat1, 89.9])]

    return extent

def get_jday_cam_img(date, fnames):

    """
    Get UTC time in hour from the camera file name

    Input:
        fnames: Python list, file paths of all the camera jpg data

    Output:
        jday: numpy array, julian day
    """

    jday = []
    for fname in fnames:
        filename = os.path.basename(fname).split('.')[0]
        dtime_s_ = filename[:23].split(' ')[-1]
        dtime_s = '%s_%s' % (date.strftime('%Y_%m_%d'), dtime_s_)
        dtime0 = datetime.datetime.strptime(dtime_s, '%Y_%m_%d_%H_%M_%SZ')
        jday0 = er3t.util.dtime_to_jday(dtime0)
        jday.append(jday0)

    return np.array(jday)



def get_jday_sat_img_vn(fnames):

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

        # dtime0 = datetime.datetime.strptime(dtime_s, '%Y-%m-%dT%H:%M:%SZ')
        dtime0 = datetime.datetime.strptime(dtime_s, '%Y-%m-%d-%H%M%SZ')
        jday0 = er3t.util.dtime_to_jday(dtime0)
        jday.append(jday0)

    return np.array(jday)

def process_sat_img_vn(fnames_sat_, threshold=80.0):

    jday_sat_ = get_jday_sat_img_vn(fnames_sat_)
    jday_sat_unique = np.sort(np.unique(jday_sat_))

    fnames_sat = []
    jday_sat = []

    for jday_sat0 in jday_sat_unique:

        indices = np.where(jday_sat_==jday_sat0)[0]
        fname0 = sorted([fnames_sat_[index] for index in indices])[-1] # pick polar imager over geostationary imager

        try:
            img0 = mpl_img.imread(fname0)
            logic_black = ~(np.sum(img0[:, :, :-1], axis=-1)>0.0)
            p_coverage = (1.0-(logic_black.sum()/logic_black.size))*100.0
            if p_coverage > threshold:
                fnames_sat.append(fname0)
                jday_sat.append(jday_sat0)
        except Exception as error:
            print(fname0)
            print(error)

    return np.array(jday_sat), fnames_sat



def cal_proj_xy_extent(extent, closed=True):

    """
    Calculate globe map projection <ccrs.Orthographic> centered at the center of the granule defined by corner points

    Input:
        line_data: Python dictionary (details see <read_geo_meta>) that contains basic information of a satellite granule
        closed=True: if True, return five corner points with the last point repeating the first point;
                     if False, return four corner points

    Output:
        proj_xy: globe map projection <ccrs.Orthographic> centered at the center of the granule defined by corner points
        xy: dimension of (5, 2) if <closed=True> and (4, 2) if <closed=False>
    """

    import cartopy.crs as ccrs

    line_data = {
            'GRingLongitude1': extent[0],
            'GRingLongitude2': extent[1],
            'GRingLongitude3': extent[1],
            'GRingLongitude4': extent[0],
             'GRingLatitude1': extent[2],
             'GRingLatitude2': extent[2],
             'GRingLatitude3': extent[3],
             'GRingLatitude4': extent[3],
            }

    # get corner points
    #/----------------------------------------------------------------------------\#
    lon_  = np.array([
        float(line_data['GRingLongitude1']),
        float(line_data['GRingLongitude2']),
        float(line_data['GRingLongitude3']),
        float(line_data['GRingLongitude4']),
        float(line_data['GRingLongitude1'])
        ])

    lat_  = np.array([
        float(line_data['GRingLatitude1']),
        float(line_data['GRingLatitude2']),
        float(line_data['GRingLatitude3']),
        float(line_data['GRingLatitude4']),
        float(line_data['GRingLatitude1'])
        ])

    if (abs(lon_[0]-lon_[1])>180.0) | (abs(lon_[0]-lon_[2])>180.0) | \
       (abs(lon_[0]-lon_[3])>180.0) | (abs(lon_[1]-lon_[2])>180.0) | \
       (abs(lon_[1]-lon_[3])>180.0) | (abs(lon_[2]-lon_[3])>180.0):

        lon_[lon_<0.0] += 360.0
    #\----------------------------------------------------------------------------/#


    # roughly determine the center of granule
    #/----------------------------------------------------------------------------\#
    lon = lon_[:-1]
    lat = lat_[:-1]
    center_lon_ = lon.mean()
    center_lat_ = lat.mean()
    #\----------------------------------------------------------------------------/#


    # find the true center
    #/----------------------------------------------------------------------------\#
    proj_lonlat = ccrs.PlateCarree()

    proj_xy_ = ccrs.Orthographic(central_longitude=center_lon_, central_latitude=center_lat_)
    xy_ = proj_xy_.transform_points(proj_lonlat, lon, lat)[:, [0, 1]]

    center_x  = xy_[:, 0].mean()
    center_y  = xy_[:, 1].mean()
    center_lon, center_lat = proj_lonlat.transform_point(center_x, center_y, proj_xy_)
    #\----------------------------------------------------------------------------/#


    # convert lon/lat corner points into xy
    #/----------------------------------------------------------------------------\#
    proj_xy = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
    xy_  = proj_xy.transform_points(proj_lonlat, lon_, lat_)[:, [0, 1]]
    #\----------------------------------------------------------------------------/#


    if closed:
        return proj_xy, xy_
    else:
        return proj_xy, xy_[:-1, :]

def contain_lonlat_check(
             lon,
             lat,
             extent,
             ):

    # check cartopy and matplotlib
    #/----------------------------------------------------------------------------\#
    import cartopy.crs as ccrs
    import matplotlib.path as mpl_path
    #\----------------------------------------------------------------------------/#


    # convert longitude in [-180, 180] range
    # since the longitude in GeoMeta dataset is in the range of [-180, 180]
    # or check overlap within region of interest
    #/----------------------------------------------------------------------------\#
    lon[lon>180.0] -= 360.0
    logic = (lon>=-180.0)&(lon<=180.0) & (lat>=-90.0)&(lat<=90.0)
    lon   = lon[logic]
    lat   = lat[logic]
    #\----------------------------------------------------------------------------/#


    # loop through all the satellite "granules" constructed through four corner points
    # and find which granules contain the input data
    #/----------------------------------------------------------------------------\#
    proj_lonlat = ccrs.PlateCarree()

    # get bounds of the satellite overpass/granule
    proj_xy, xy_granule = cal_proj_xy_extent(extent, closed=True)
    sat_granule  = mpl_path.Path(xy_granule, closed=True)

    # check if the overpass/granule overlaps with region of interest
    xy_in      = proj_xy.transform_points(proj_lonlat, lon, lat)[:, [0, 1]]
    points_in  = sat_granule.contains_points(xy_in)

    Npoint_in  = points_in.sum()

    if (Npoint_in>0):
        contain = True
    else:
        contain = False
    #\----------------------------------------------------------------------------/#

    return contain



class flt_sim:

    def __init__(
            self,
            date=datetime.datetime.now(),
            fdir='./',
            extent=None,
            wavelength=None,
            flt_trks=None,
            flt_imgs=None,
            fname=None,
            overwrite=False,
            quiet=False,
            verbose=False
            ):

        self.date      = date
        self.wvl0      = wavelength
        self.fdir      = os.path.abspath(fdir)
        self.extent    = extent
        self.flt_trks  = flt_trks
        self.flt_imgs  = flt_imgs
        self.overwrite = overwrite
        self.quiet     = quiet
        self.verbose   = verbose

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif (((flt_trks is not None) and (flt_imgs is not None) and (wavelength is not None)) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             (((flt_trks is not None) and (flt_imgs is not None) and (wavelength is not None)) and (fname is not None) and (not os.path.exists(fname))):

            self.run_rtm()
            self.dump(fname)

        elif (((flt_trks is not None) and (flt_imgs is not None) and (wavelength is not None)) and (fname is None)):

            self.run()

        else:

            sys.exit('Error   [flt_sim]: Please check if \'%s\' exists or provide \'wavelength\', \'flt_trks\', and \'flt_imgs\' to proceed.' % fname)

    def load(self, fname):

        with open(fname, 'rb') as f:
            obj = pickle.load(f)
            if hasattr(obj, 'flt_trks') and hasattr(obj, 'flt_imgs'):
                if self.verbose:
                    print('Message [flt_sim]: Loading %s ...' % fname)
                self.date     = obj.date
                self.fdir     = obj.fdir
                self.extent   = obj.extent
                self.wvl0     = obj.wvl0
                self.fname    = obj.fname
                self.flt_trks = obj.flt_trks
                self.flt_imgs = obj.flt_imgs
            else:
                sys.exit('Error   [flt_sim]: File \'%s\' is not the correct pickle file to load.' % fname)

    def run_rtm(self, overwrite=True):

        N = len(self.flt_trks)

        for i in range(N):

            flt_trk = self.flt_trks[i]
            flt_img = self.flt_imgs[i]

    def dump(self, fname):

        self.fname = fname
        with open(fname, 'wb') as f:
            if self.verbose:
                print('Message [flt_sim]: Saving object into %s ...' % fname)
            pickle.dump(self, f)



# for test flights at NASA WFF
#/----------------------------------------------------------------------------\#
def plot_video_frame_wff(statements, test=False):

    # extract arguments
    #/----------------------------------------------------------------------------\#
    flt_sim0, index_trk, index_pnt, n = statements

    flt_trk0 = flt_sim0.flt_trks[index_trk]
    flt_img0 = flt_sim0.flt_imgs[index_trk]
    vnames_trk = flt_trk0.keys()
    vnames_img = flt_img0.keys()
    #\----------------------------------------------------------------------------/#


    # general plot settings
    #/----------------------------------------------------------------------------\#
    vars_plot = OrderedDict()

    vars_plot['SSFR-A↑']   = {
            'vname':'f-up_ssfr',
            'color':'red',
            'vname_wvl':'wvl_ssfr1_nad',
            'zorder': 7,
            }
    vars_plot['SSFR-A↓']   = {
            'vname':'f-down_ssfr',
            'color':'blue',
            'vname_wvl':'wvl_ssfr1_zen',
            'zorder': 6,
            }
    vars_plot['SSFR-B↑']   = {
            'vname':'r-up_ssfr',
            'color':'deeppink',
            'vname_wvl':'wvl_ssfr2_nad',
            'zorder': 2,
            }
    vars_plot['SSFR-B↓']   = {
            'vname':'r-down_ssfr',
            'color':'dodgerblue',
            'vname_wvl':'wvl_ssfr2_zen',
            'zorder': 3,
            }
    vars_plot['SPNS Total↓']   = {
            'vname':'f-down-total_spns',
            'color':'green',
            'vname_wvl':'wvl_spns',
            'zorder': 5,
            }
    vars_plot['SPNS Diffuse↓']   = {
            'vname':'f-down-diffuse_spns',
            'color':'springgreen',
            'vname_wvl':'wvl_spns',
            'zorder': 4,
            }
    vars_plot['TOA↓']   = {
            'vname':'f-down_toa',
            'color':'dimgray',
            'vname_wvl':'wvl_ssfr1_zen',
            'zorder': 1,
            }
    vars_plot['Altitude']   = {
            'vname':'alt',
            'color':'purple',
            'zorder': 0,
            }

    for vname in vars_plot.keys():

        vname_ori = vars_plot[vname]['vname']
        if vname_ori in vnames_trk:
            vars_plot[vname]['plot?'] = True
        else:
            vars_plot[vname]['plot?'] = False

        if 'vname_wvl' in vars_plot[vname].keys():
            vars_plot[vname]['spectra?'] = True
        else:
            vars_plot[vname]['spectra?'] = False
    #\----------------------------------------------------------------------------/#


    # check data
    #/----------------------------------------------------------------------------\#
    has_trk = True

    if ('ang_pit' in vnames_trk) and ('ang_rol' in vnames_trk):
        has_att = True
    else:
        has_att = False

    if ('ang_pit_m' in vnames_trk) and ('ang_rol_m' in vnames_trk):
        has_att_corr = True
    else:
        has_att_corr = False

    if ('fnames_sat0' in vnames_img) and ('extent_sat0' in vnames_img):
        has_sat0 = True
    else:
        has_sat0 = False

    if ('fnames_sat1' in vnames_img) and ('extent_sat1' in vnames_img):
        has_sat1 = True
    else:
        has_sat1 = False

    if ('fnames_cam0' in vnames_img):
        has_cam0 = True
    else:
        has_cam0 = False
    #\----------------------------------------------------------------------------/#


    # param settings
    #/----------------------------------------------------------------------------\#
    tmhr_current = flt_trk0['tmhr'][index_pnt]
    jday_current = flt_trk0['jday'][index_pnt]
    lon_current  = flt_trk0['lon'][index_pnt]
    lat_current  = flt_trk0['lat'][index_pnt]
    alt_current  = flt_trk0['alt'][index_pnt]
    sza_current  = flt_trk0['sza'][index_pnt]
    dtime_current = er3t.util.jday_to_dtime(jday_current)

    tmhr_length  = 0.5 # half an hour
    tmhr_past    = tmhr_current-tmhr_length
    #\----------------------------------------------------------------------------/#


    # flight direction
    #/----------------------------------------------------------------------------\#
    alt_cmap = mpl.colormaps[_alt_cmap_]
    alt_norm = mpl.colors.Normalize(vmin=0.0, vmax=9.0)

    dlon = flt_sim0.flt_imgs[index_trk]['extent_sat0'][1] - flt_sim0.flt_imgs[index_trk]['extent_sat0'][0]
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

    plot_arrow=False
    #\----------------------------------------------------------------------------/#


    # figure setup
    #/----------------------------------------------------------------------------\#
    fig = plt.figure(figsize=(16, 9))

    gs = gridspec.GridSpec(12, 17)

    # ax of all
    ax = fig.add_subplot(gs[:, :])

    # map of flight track overlay satellite imagery
    ax_map = fig.add_subplot(gs[:8, :7])

    # flight altitude next to the map
    divider = make_axes_locatable(ax_map)
    ax_alt = divider.append_axes('right', size='4%', pad=0.0)

    # aircraft and platform attitude status
    ax_nav  = fig.add_subplot(gs[:2, 7:9])

    # a secondary map
    ax_map0 = fig.add_subplot(gs[:5, 9:13])

    # camera imagery
    ax_img  = fig.add_subplot(gs[:5, 13:])
    ax_img_hist = ax_img.twinx()

    # spetral irradiance
    ax_wvl  = fig.add_subplot(gs[5:8, 9:])

    # time series
    ax_tms = fig.add_subplot(gs[9:, :])
    ax_tms_alt = ax_tms.twinx()

    fig.subplots_adjust(hspace=10.0, wspace=10.0)
    #\----------------------------------------------------------------------------/#


    # base plot
    #/----------------------------------------------------------------------------\#
    if has_sat0:
        extent_sat0 = [-75.92,-71.44,36.20,40.68] # WFF test flight #1
        # extent_sat0 = [-76.10,-74.58,37.46,38.98] # WFF test flight #2
        fname_sat = flt_img0['fnames_sat0'][index_pnt]
        img = mpl_img.imread(fname_sat)
        # ax_map.imshow(img, extent=flt_img0['extent_sat0'], origin='upper', aspect='auto', zorder=0)
        ax_map.imshow(img, extent=extent_sat0, origin='upper', aspect='auto', zorder=0)
        # rect = mpatches.Rectangle((lon_current-0.1, lat_current-0.1), 0.2, 0.2, lw=1.0, ec='k', fc='none')
        rect = mpatches.Rectangle((lon_current-0.25, lat_current-0.25), 0.5, 0.5, lw=1.0, ec='k', fc='none')
        ax_map.add_patch(rect)

    if has_sat1:
        fname_sat1 = flt_img0['fnames_sat1'][index_pnt]
        img = mpl_img.imread(fname_sat1)
        ax_map0.imshow(img, extent=flt_img0['extent_sat1'], origin='upper', aspect='auto', zorder=0)

    if has_cam0:
        # ang_cam_offset = -152.0 # for ORACLES
        ang_cam_offset = -53.0 # for ARCSIX
        cam_x_s = 5.0
        cam_x_e = 255.0*4.0
        cam_y_s = 0.0
        cam_y_e = 0.12
        cam_hist_x_s = 0.0
        cam_hist_x_e = 255.0
        cam_hist_bins = np.linspace(cam_hist_x_s, cam_hist_x_e, 31)

        fname_cam = flt_img0['fnames_cam0'][index_pnt]
        img = mpl_img.imread(fname_cam)[:-200, 540:-640, :]

        # if ('ang_hed' in vnames_trk):
        #     ang_hed0   = flt_trk0['ang_hed'][index_pnt]
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
        ax_img.imshow(img_plot, origin='upper', aspect='auto', zorder=0, extent=[cam_x_s, cam_x_e, cam_y_s, cam_y_e])

        ax_img_hist.hist(img[:, :, 0].ravel(), bins=cam_hist_bins, histtype='step', lw=0.5, alpha=0.9, density=True, color='r')
        ax_img_hist.hist(img[:, :, 1].ravel(), bins=cam_hist_bins, histtype='step', lw=0.5, alpha=0.9, density=True, color='g')
        ax_img_hist.hist(img[:, :, 2].ravel(), bins=cam_hist_bins, histtype='step', lw=0.5, alpha=0.9, density=True, color='b')
        ax_img.plot([255, 255], [0, 0.005], color='white', lw=1.0, ls='-')

    if has_att:
        ax_nav.axhspan(-10.0, 0.0, lw=0.0, color='orange', zorder=0, alpha=0.3)
        ax_nav.axhspan(0.0,  10.0, lw=0.0, color='deepskyblue', zorder=0, alpha=0.3)

        ax_nav.axvline(0.0, lw=0.5, color='gray', zorder=1)
        ax_nav.axhline(0.0, lw=0.5, color='gray', zorder=1)

        ang_pit0 = flt_trk0['ang_pit'][index_pnt]
        ang_rol0 = flt_trk0['ang_rol'][index_pnt]

        x  = np.linspace(-10.0, 10.0, 101)

        slope0  = -np.tan(np.deg2rad(ang_rol0))
        offset0 = ang_pit0
        y0 = slope0*x + offset0

        ax_nav.plot(x[15:-15], y0[15:-15], lw=1.0, color='red', zorder=1, alpha=0.6)
        ax_nav.scatter(x[50], y0[50], lw=0.0, s=40, c='red', zorder=1, alpha=0.6)

        if has_att_corr:
            ang_pit_m0 = flt_trk0['ang_pit_m'][index_pnt]
            ang_rol_m0 = flt_trk0['ang_rol_m'][index_pnt]

            ang_pit_offset = 0.0
            ang_rol_offset = 0.0

            slope1  = -np.tan(np.deg2rad(ang_rol0-ang_rol_m0+ang_rol_offset))
            offset1 = (ang_pit0-ang_pit_m0+ang_pit_offset)
            y1 = slope1*x + offset1

            ax_nav.plot(x[25:-25], y1[25:-25], lw=2.0, color='green', zorder=2, alpha=0.7)

    for vname in vars_plot.keys():
        var_plot = vars_plot[vname]
        if var_plot['plot?']:
            if 'vname_wvl' in var_plot.keys():
                wvl_x  = flt_trk0[var_plot['vname_wvl']]
                if 'toa' in var_plot['vname']:
                    spec_y = flt_trk0[var_plot['vname']] * np.cos(np.deg2rad(sza_current))
                else:
                    spec_y = flt_trk0[var_plot['vname']][index_pnt, :]

                ax_wvl.scatter(wvl_x, spec_y, c=var_plot['color'], s=4, lw=0.0, zorder=var_plot['zorder'])

                wvl_index = np.argmin(np.abs(wvl_x-flt_sim0.wvl0))
                ax_wvl.axvline(wvl_x[wvl_index], color=var_plot['color'], ls='-', lw=1.0, alpha=0.5, zorder=var_plot['zorder'])
    #\----------------------------------------------------------------------------/#


    # iterate through flight segments
    #/----------------------------------------------------------------------------\#
    for itrk in range(index_trk+1):

        flt_trk = flt_sim0.flt_trks[itrk]
        flt_img = flt_sim0.flt_imgs[itrk]

        logic_solid = (flt_trk['tmhr']>=tmhr_past) & (flt_trk['tmhr']<=tmhr_current)
        logic_trans = np.logical_not(logic_solid)

        if itrk == index_trk:
            alpha_trans = 0.0
        else:
            alpha_trans = 0.08

        ax_map.scatter(flt_trk['lon'][logic_trans], flt_trk['lat'][logic_trans], c=flt_trk['alt'][logic_trans], s=0.5, lw=0.0, zorder=1, vmin=0.0, vmax=9.0, cmap=_alt_cmap_, alpha=alpha_trans)
        ax_map.scatter(flt_trk['lon'][logic_solid], flt_trk['lat'][logic_solid], c=flt_trk['alt'][logic_solid], s=1  , lw=0.0, zorder=2, vmin=0.0, vmax=9.0, cmap=_alt_cmap_)

        ax_map0.scatter(flt_trk['lon'][logic_trans], flt_trk['lat'][logic_trans], c=flt_trk['alt'][logic_trans], s=2.5, lw=0.0, zorder=1, vmin=0.0, vmax=9.0, cmap=_alt_cmap_, alpha=alpha_trans)
        ax_map0.scatter(flt_trk['lon'][logic_solid], flt_trk['lat'][logic_solid], c=flt_trk['alt'][logic_solid], s=4  , lw=0.0, zorder=2, vmin=0.0, vmax=9.0, cmap=_alt_cmap_)


        if not plot_arrow:
            ax_map.scatter(lon_current, lat_current, facecolor='none', edgecolor='white', s=60, lw=1.0, zorder=3, alpha=0.6)
            ax_map.scatter(lon_current, lat_current, c=alt_current, s=60, lw=0.0, zorder=3, alpha=0.6, vmin=0.0, vmax=9.0, cmap=_alt_cmap_)
            # ax_map0.scatter(lon_current, lat_current, facecolor='none', edgecolor='white', s=60, lw=1.0, zorder=3, alpha=0.6)
            # ax_map0.scatter(lon_current, lat_current, c=alt_current, s=60, lw=0.0, zorder=3, alpha=0.6, vmin=0.0, vmax=9.0, cmap=_alt_cmap_)
        else:
            color0 = alt_cmap(alt_norm(alt_current))
            arrow_prop['facecolor'] = color0
            arrow_prop['relpos'] = (lon_current, lat_current)
            ax_map.annotate('', xy=(lon_point_to, lat_point_to), xytext=(lon_current, lat_current), arrowprops=arrow_prop, zorder=3)
            # ax_map0.annotate('', xy=(lon_point_to, lat_point_to), xytext=(lon_current, lat_current), arrowprops=arrow_prop, zorder=3)

        ax_map0.scatter(lon_current, lat_current, facecolor='none', edgecolor='white', s=60, lw=1.0, zorder=3, alpha=0.6)
        ax_map0.scatter(lon_current, lat_current, c=alt_current, s=60, lw=0.0, zorder=3, alpha=0.6, vmin=0.0, vmax=9.0, cmap=_cmap_)

        for vname in vars_plot.keys():

            var_plot = vars_plot[vname]

            if var_plot['plot?']:
                if 'vname_wvl' in var_plot.keys():
                    wvl_x  = flt_trk[var_plot['vname_wvl']]
                    index_wvl = np.argmin(np.abs(wvl_x-flt_sim0.wvl0))
                    if 'toa' in var_plot['vname']:
                        tms_y = flt_trk[var_plot['vname']][index_wvl] * np.cos(np.deg2rad(flt_trk['sza']))
                    else:
                        tms_y = flt_trk[var_plot['vname']][:, index_wvl]
                else:
                    tms_y = flt_trk[var_plot['vname']]

                if vname == 'Altitude':
                    ax_tms_alt.fill_between(flt_trk['tmhr'][logic_solid], tms_y[logic_solid], facecolor=vars_plot[vname]['color'], alpha=0.25, lw=0.0, zorder=var_plot['zorder'])
                else:
                    ax_tms.scatter(flt_trk['tmhr'][logic_solid], tms_y[logic_solid], c=vars_plot[vname]['color'], s=2, lw=0.0, zorder=var_plot['zorder'])
    #\----------------------------------------------------------------------------/#


    # figure settings
    #/----------------------------------------------------------------------------\#
    title_fig = '%s UTC' % (dtime_current.strftime('%Y-%m-%d %H:%M:%S'))
    fig.suptitle(title_fig, y=0.96, fontsize=20)
    #\----------------------------------------------------------------------------/#


    # map plot settings
    #/----------------------------------------------------------------------------\#
    if has_sat0:
        ax_map.set_xlim(extent_sat0[:2])
        ax_map.set_ylim(extent_sat0[2:])

        title_map = '%s at %s UTC' % (flt_img0['id_sat0'][index_pnt], er3t.util.jday_to_dtime(flt_img0['jday_sat0'][index_pnt]).strftime('%H:%M'))
        time_diff = np.abs(flt_img0['jday_sat0'][index_pnt]-jday_current)*86400.0
        if time_diff > 301.0:
            ax_map.set_title(title_map, color='gray')
        else:
            ax_map.set_title(title_map)

    ax_map.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 180.1, 1.0)))
    ax_map.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 90.1, 1.0)))
    # ax_map.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 180.1, 0.5)))
    # ax_map.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 90.1, 0.5)))
    ax_map.set_xlabel('Longitude [$^\\circ$]')
    ax_map.set_ylabel('Latitude [$^\\circ$]')
    #\----------------------------------------------------------------------------/#


    # navigation plot settings
    #/----------------------------------------------------------------------------\#
    ax_nav.set_xlim((-10.0, 10.0))
    ax_nav.set_ylim((-10.0, 10.0))
    ax_nav.axis('off')
    #\----------------------------------------------------------------------------/#


    # map0 plot settings
    #/----------------------------------------------------------------------------\#
    if has_sat1:
        title_map0 = 'Zoomed-in View'
        time_diff = np.abs(flt_img0['jday_sat1'][index_pnt]-jday_current)*86400.0
        if time_diff > 301.0:
            ax_map0.set_title(title_map0, color='gray')
        else:
            ax_map0.set_title(title_map0)

    ax_map0.set_xlim((lon_current-0.25, lon_current+0.25))
    ax_map0.set_ylim((lat_current-0.25, lat_current+0.25))
    # ax_map0.set_xlim((lon_current-0.1, lon_current+0.1))
    # ax_map0.set_ylim((lat_current-0.1, lat_current+0.1))
    ax_map0.axis('off')
    #\----------------------------------------------------------------------------/#


    # camera image plot settings
    #/----------------------------------------------------------------------------\#
    if has_cam0:
        jday_cam  = flt_img0['jday_cam0'][index_pnt]
        dtime_cam = er3t.util.jday_to_dtime(jday_cam)

        title_img = 'Camera at %s UTC' % (dtime_cam.strftime('%H:%M:%S'))
        time_diff = np.abs(jday_current-jday_cam)*86400.0
        if time_diff > 301.0:
            ax_img.set_title(title_img, color='gray')
        else:
            ax_img.set_title(title_img)

        ax_img_hist.set_xlim((cam_x_s, cam_x_e))
        ax_img_hist.set_ylim((cam_y_s, cam_y_e))

    ax_img.axis('off')
    ax_img_hist.axis('off')
    #\----------------------------------------------------------------------------/#


    # altitude/sza plot settings
    #/----------------------------------------------------------------------------\#
    ax_alt.axhspan(0, (90.0-sza_current)/10.0, color='gold', lw=0.0, zorder=0, alpha=0.5)

    color0 = alt_cmap(alt_norm(alt_current))
    ax_alt.axhline(alt_current, lw=2.0, color=color0, zorder=1)

    ax_alt.set_xlim((0.0, 1.0))
    ax_alt.set_ylim((0.0, 9.0))
    ax_alt.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 9.1, 3.0)))
    ax_alt.yaxis.set_minor_locator(FixedLocator(np.arange(0.0, 9.1, 1.0)))
    ax_alt.xaxis.set_ticks([])
    ax_alt.yaxis.tick_right()
    ax_alt.yaxis.set_label_position('right')
    ax_alt.set_ylabel('Sun Elevation [$\\times 10^\\circ$] / Altitude [km]', rotation=270.0, labelpad=18)
    #\----------------------------------------------------------------------------/#


    # altitude (time series) plot settings
    #/----------------------------------------------------------------------------\#
    ax_tms_alt.set_ylim((0.0, 8.0))
    ax_tms_alt.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 8.1, 2.0)))
    ax_tms_alt.yaxis.tick_right()
    ax_tms_alt.yaxis.set_label_position('right')
    ax_tms_alt.set_ylabel('Altitude [km]', rotation=270.0, labelpad=18, color=vars_plot['Altitude']['color'])

    ax_tms_alt.set_frame_on(True)
    for spine in ax_tms_alt.spines.values():
        spine.set_visible(False)
    ax_tms_alt.spines['right'].set_visible(True)
    ax_tms_alt.spines['right'].set_color(vars_plot['Altitude']['color'])
    ax_tms_alt.tick_params(axis='y', colors=vars_plot['Altitude']['color'])
    #\----------------------------------------------------------------------------/#


    # time series plot settings
    #/----------------------------------------------------------------------------\#
    ax_tms.grid()
    ax_tms.set_xlim((tmhr_past-0.0000001, tmhr_current+0.0000001))
    ax_tms.xaxis.set_major_locator(FixedLocator([tmhr_past, tmhr_current-0.5*tmhr_length, tmhr_current]))
    ax_tms.xaxis.set_minor_locator(FixedLocator(np.arange(tmhr_past, tmhr_current+0.001, 5.0/60.0)))
    ax_tms.set_xticklabels(['%.4f' % (tmhr_past), '%.4f' % (tmhr_current-0.5*tmhr_length), '%.4f' % tmhr_current])
    ax_tms.set_xlabel('Time [hour]')

    ax_tms.set_ylim(bottom=0.0)
    ax_tms.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 2.1, 0.5)))
    ax_tms.set_ylabel('Flux [$\\mathrm{W m^{-2} nm^{-1}}$]')

    if alt_current < 1.0:
        title_all = 'Longitude %9.4f$^\\circ$, Latitude %8.4f$^\\circ$, Altitude %6.1f  m, Solar Zenith %4.1f$^\\circ$' % (lon_current, lat_current, alt_current*1000.0, sza_current)
    else:
        title_all = 'Longitude %9.4f$^\\circ$, Latitude %8.4f$^\\circ$, Altitude %6.4f km, Solar Zenith %4.1f$^\\circ$' % (lon_current, lat_current, alt_current, sza_current)
    ax_tms.set_title(title_all)

    ax_tms.spines['right'].set_visible(False)
    ax_tms.set_zorder(ax_tms_alt.get_zorder()+1)
    ax_tms.patch.set_visible(False)
    #\----------------------------------------------------------------------------/#


    # spectra plot setting
    #/----------------------------------------------------------------------------\#
    ax_wvl.grid()
    ax_wvl.set_xlim((200, 2200))
    ax_wvl.set_ylim(ax_tms.get_ylim())
    ax_wvl.xaxis.set_major_locator(FixedLocator(np.arange(0, 2401, 400)))
    ax_wvl.xaxis.set_minor_locator(FixedLocator(np.arange(0, 2401, 100)))
    ax_wvl.set_xlabel('Wavelength [nm]')
    ax_wvl.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 2.1, 0.5)))
    ax_wvl.set_ylabel('Flux [$\\mathrm{W m^{-2} nm^{-1}}$]')
    #\----------------------------------------------------------------------------/#


    # acknowledgements
    #/----------------------------------------------------------------------------\#
    text1 = '\
presented by ARCSIX SSFR Team - Hong Chen, Vikas Nataraja, Yu-Wen Chen, Ken Hirata, Arabella Chamberlain, Katey Dong, Jeffery Drouet, and Sebastian Schmidt\n\
'
    ax.annotate(text1, xy=(0.5, 0.24), fontsize=8, color='gray', xycoords='axes fraction', ha='center', va='center')
    ax.axis('off')
    #\----------------------------------------------------------------------------/#


    # legend plot settings
    #/----------------------------------------------------------------------------\#
    patches_legend = []
    for vname in vars_plot.keys():
        var_plot = vars_plot[vname]
        if vname.lower() != 'altitude' and var_plot['plot?']:
            patches_legend.append(mpatches.Patch(color=var_plot['color'], label=vname))
    if len(patches_legend) > 0:
        ax_wvl.legend(handles=patches_legend, loc='upper right', fontsize=8)
        # ax_tms.legend(handles=patches_legend, bbox_to_anchor=(0.03, 1.23, 0.92, .102), loc=3, ncol=len(patches_legend), mode='expand', borderaxespad=0., frameon=True, handletextpad=0.2, fontsize=12)
    #\----------------------------------------------------------------------------/#


    if test:
        plt.show()
        sys.exit()
    else:
        plt.savefig('%s/%5.5d.png' % (_fdir_tmp_graph_, n), bbox_inches='tight')
        plt.close(fig)

def main_pre_wff(
        date,
        wvl0=_wavelength_,
        run_rtm=False,
        time_step=1,
        wvl_step_spns=10,
        wvl_step_ssfr=3,
        ):


    # create data directory (for storing data) if the directory does not exist
    #/----------------------------------------------------------------------------\#
    date_s = date.strftime('%Y%m%d')
    #\----------------------------------------------------------------------------/#


    # read data
    #/----------------------------------------------------------------------------\#
    # read in aircraft hsk data
    #/--------------------------------------------------------------\#
    fname_hsk = '%s/%s-%s_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _hsk_.upper(), _platform_.upper(), date_s)

    f_hsk = h5py.File(fname_hsk, 'r')
    jday   = f_hsk['jday'][...]
    tmhr   = f_hsk['tmhr'][...]
    sza    = f_hsk['sza'][...]
    lon    = f_hsk['lon'][...]
    lat    = f_hsk['lat'][...]

    logic0 = (~np.isnan(jday) & ~np.isinf(sza))   & \
             check_continuity(lon, threshold=1.0) & \
             check_continuity(lat, threshold=1.0) & \
             (tmhr>=_tmhr_range_[date_s][0]) & (tmhr<=_tmhr_range_[date_s][1])

    # print(jday[~logic0])
    # print(sza[~logic0])
    # print(lon[~logic0])
    # print(lat[~logic0])

    jday = jday[logic0][::time_step]
    tmhr = tmhr[logic0][::time_step]
    sza  = sza[logic0][::time_step]
    lon  = lon[logic0][::time_step]
    lat  = lat[logic0][::time_step]

    alt    = f_hsk['alt'][...][logic0][::time_step]

    f_hsk.close()
    #\--------------------------------------------------------------/#
    # print(tmhr.shape)
    # print(alt.shape)
    # print(jday.shape)
    # print(sza.shape)
    # print(lon.shape)
    # print(lat.shape)
    # print(tmhr.shape)
    # print(alt.shape)

    # process satellite imagery
    #/----------------------------------------------------------------------------\#
    extent = get_extent(lon, lat, margin=0.2)

    interval = 600.0 # seconds
    dtime_s = er3t.util.jday_to_dtime((jday[0] *86400.0//interval  )*interval/86400.0)
    dtime_e = er3t.util.jday_to_dtime((jday[-1]*86400.0//interval+1)*interval/86400.0)

    if False:
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
    date_sat_s  = date.strftime('%Y-%m-%d')
    fnames_sat_ = sorted(glob.glob('%s/*%sT*Z*.png' % (_fdir_sat_img_, date_sat_s)))
    jday_sat_ = get_jday_sat_img(fnames_sat_)

    jday_sat0 = np.sort(np.unique(jday_sat_))

    fnames_sat0 = []

    for jday_sat_i in jday_sat0:

        indices = np.where(jday_sat_==jday_sat_i)[0]
        fname0 = sorted([fnames_sat_[index] for index in indices])[-1] # pick polar imager over geostationary imager
        fnames_sat0.append(fname0)

    fnames_sat1 = fnames_sat0.copy()
    jday_sat1 = jday_sat0.copy()
    #\----------------------------------------------------------------------------/#


    # read in alp data
    #/--------------------------------------------------------------\#
    fname_alp = '%s/%s-%s_%s_%s_v1.h5' % (_fdir_data_, _mission_.upper(), _alp_.upper(), _platform_.upper(), date_s)
    f_alp = h5py.File(fname_alp, 'r')
    ang_pit_s = f_alp['ang_pit_s'][...][logic0][::time_step]
    ang_rol_s = f_alp['ang_rol_s'][...][logic0][::time_step]
    ang_pit_m = f_alp['ang_pit_m'][...][logic0][::time_step]
    ang_rol_m = f_alp['ang_rol_m'][...][logic0][::time_step]
    f_alp.close()
    #\--------------------------------------------------------------/#


    # read in spns data
    #/--------------------------------------------------------------\#
    # fname_spns = '%s/%s-%s_%s_%s_v2.h5' % (_fdir_data_, _mission_.upper(), _spns_.upper(), _platform_.upper(), date_s)
    fname_spns = '%s/%s-%s_%s_%s_v1.h5' % (_fdir_data_, _mission_.upper(), _spns_.upper(), _platform_.upper(), date_s)
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


    # read in ssfr-a data
    #/--------------------------------------------------------------\#
    fname_ssfr1 = '%s/%s-%s_%s_%s_v1.h5' % (_fdir_data_, _mission_.upper(), _ssfr1_.upper(), _platform_.upper(), date_s)
    f_ssfr1 = h5py.File(fname_ssfr1, 'r')
    ssfr1_zen_flux = f_ssfr1['zen/flux'][...][logic0, :][::time_step, ::wvl_step_ssfr]
    ssfr1_zen_wvl  = f_ssfr1['zen/wvl'][...][::wvl_step_ssfr]
    ssfr1_nad_flux = f_ssfr1['nad/flux'][...][logic0, :][::time_step, ::wvl_step_ssfr]
    ssfr1_nad_wvl  = f_ssfr1['nad/wvl'][...][::wvl_step_ssfr]
    ssfr1_zen_toa  = f_ssfr1['zen/toa0'][...][::wvl_step_ssfr]
    f_ssfr1.close()
    #\--------------------------------------------------------------/#
    # print(ssfr_zen_flux.shape)
    # print(ssfr_zen_wvl.shape)
    # print(ssfr_nad_flux.shape)
    # print(ssfr_nad_wvl.shape)

    # read in ssfr-b data
    #/--------------------------------------------------------------\#
    # fname_ssfr2 = '%s/%s-%s_%s_%s_v1.h5' % (_fdir_data_, _mission_.upper(), _ssfr2_.upper(), _platform_.upper(), date_s)
    # f_ssfr2 = h5py.File(fname_ssfr2, 'r')
    # ssfr2_zen_rad = f_ssfr2['zen/rad'][...][logic0, :][::time_step, ::wvl_step_ssfr]
    # ssfr2_zen_wvl = f_ssfr2['zen/wvl'][...][::wvl_step_ssfr]
    # ssfr2_nad_rad = f_ssfr2['nad/rad'][...][logic0, :][::time_step, ::wvl_step_ssfr]
    # ssfr2_nad_wvl = f_ssfr2['nad/wvl'][...][::wvl_step_ssfr]
    # f_ssfr2.close()
    #\--------------------------------------------------------------/#


    # process camera imagery
    #/----------------------------------------------------------------------------\#
    fdirs = er3t.util.get_all_folders(_fdir_cam_img_, pattern='*%4.4d*%2.2d*%2.2d*nac*jpg*' % (date.year, date.month, date.day))
    fdir_cam0 = sorted(fdirs, key=os.path.getmtime)[-1]
    fnames_cam0 = sorted(glob.glob('%s/*.jpg' % (fdir_cam0)))
    jday_cam0 = get_jday_cam_img(date, fnames_cam0)
    #\----------------------------------------------------------------------------/#




    # pre-process the aircraft and satellite data
    #/----------------------------------------------------------------------------\#
    # create a filter to remove invalid data, e.g., out of available satellite data time range,
    # invalid solar zenith angles etc.
    tmhr_interval = 10.0/60.0
    half_interval = tmhr_interval/48.0

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
    flt_trk['ang_pit'] = ang_pit[logic]
    flt_trk['ang_rol'] = ang_rol[logic]
    flt_trk['ang_pit_m'] = ang_pit_m[logic]
    flt_trk['ang_rol_m'] = ang_rol_m[logic]

    flt_trk['f-down-total_spns']   = spns_tot_flux[logic, :]
    flt_trk['f-down-diffuse_spns'] = spns_dif_flux[logic, :]
    flt_trk['f-down-direct_spns']  = flt_trk['f-down-total_spns'] - flt_trk['f-down-diffuse_spns']
    flt_trk['wvl_spns'] = spns_tot_wvl

    flt_trk['f-down_ssfr']   = ssfr1_zen_flux[logic, :]
    flt_trk['f-up_ssfr']     = ssfr1_nad_flux[logic, :]
    flt_trk['wvl_ssfr1_zen'] = ssfr1_zen_wvl
    flt_trk['wvl_ssfr1_nad'] = ssfr1_nad_wvl
    flt_trk['f-down_toa']    = ssfr1_zen_toa

    # flt_trk['r-down_ssfr']   = ssfr2_zen_rad[logic, :]
    # flt_trk['r-up_ssfr']     = ssfr2_nad_rad[logic, :]
    # flt_trk['wvl_ssfr2_zen'] = ssfr2_zen_wvl
    # flt_trk['wvl_ssfr2_nad'] = ssfr2_nad_wvl

    # partition the flight track into multiple mini flight track segments
    flt_trks = partition_flight_track(flt_trk, jday_edges, margin_x=0.2, margin_y=0.2)
    #\----------------------------------------------------------------------------/#

    # process imagery
    #/----------------------------------------------------------------------------\#
    # create python dictionary to store corresponding satellite imagery data info
    #/--------------------------------------------------------------\#
    flt_imgs = []
    for i in range(len(flt_trks)):

        flt_img = {}

        flt_img['id_sat0'] = []

        flt_img['fnames_sat0'] = []
        flt_img['extent_sat0'] = extent
        flt_img['jday_sat0'] = np.array([], dtype=np.float64)

        flt_img['fnames_sat1'] = []
        flt_img['extent_sat1'] = extent
        flt_img['jday_sat1'] = np.array([], dtype=np.float64)

        flt_img['fnames_cam0']  = []
        flt_img['jday_cam0'] = np.array([], dtype=np.float64)

        for j in range(flt_trks[i]['jday'].size):

            index_sat0 = np.argmin(np.abs(jday_sat0-flt_trks[i]['jday'][j]))
            flt_img['id_sat0'].append(os.path.basename(fnames_sat0[index_sat0]).split('_')[0].replace('-', ' '))
            flt_img['fnames_sat0'].append(fnames_sat0[index_sat0])
            flt_img['jday_sat0'] = np.append(flt_img['jday_sat0'], jday_sat0[index_sat0])

            # this will change
            #/--------------------------------------------------------------\#
            index_sat1 = np.argmin(np.abs(jday_sat1-flt_trks[i]['jday'][j]))
            flt_img['fnames_sat1'].append(fnames_sat1[index_sat1])
            flt_img['jday_sat1'] = np.append(flt_img['jday_sat1'], jday_sat1[index_sat1])
            #\--------------------------------------------------------------/#

            index_cam = np.argmin(np.abs(jday_cam0-flt_trks[i]['jday'][j]))
            flt_img['fnames_cam0'].append(fnames_cam0[index_cam])
            flt_img['jday_cam0'] = np.append(flt_img['jday_cam0'], jday_cam0[index_cam])

        flt_imgs.append(flt_img)
    #\--------------------------------------------------------------/#


    # generate flt-sat combined file
    #/----------------------------------------------------------------------------\#
    fname = '%s/%s-FLT-VID_%s_%s_v0.pk' % (_fdir_main_, _mission_.upper(), _platform_.upper(), date_s)
    sim0 = flt_sim(
            date=date,
            wavelength=wvl0,
            flt_trks=flt_trks,
            flt_imgs=flt_imgs,
            fname=fname,
            overwrite=True,
            )
    #\----------------------------------------------------------------------------/#

def main_vid_wff(
        date,
        wvl0=_wavelength_,
        interval=1,
        ):

    date_s = date.strftime('%Y%m%d')

    fdir = _fdir_tmp_graph_
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

    indices_trk = indices_trk[::interval]
    indices_pnt = indices_pnt[::interval]
    indices     = indices[::interval]

    statements = zip([flt_sim0]*indices_trk.size, indices_trk, indices_pnt, indices)

    with mp.Pool(processes=15) as pool:
        r = list(tqdm(pool.imap(plot_video_frame_wff, statements), total=indices_trk.size))

    # make video
    fname_mp4 = '%s-FLT-VID_%s_%s.mp4' % (_mission_.upper(), _platform_.upper(), date_s)
    os.system('ffmpeg -y -framerate 30 -pattern_type glob -i "%s/*.png" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p %s' % (fdir, fname_mp4))
#\----------------------------------------------------------------------------/#



# for research flights in the Arctic
#/----------------------------------------------------------------------------\#
def plot_video_frame(statements, test=False):

    if test:
        time_s = time.time()

    # extract arguments
    #/----------------------------------------------------------------------------\#
    flt_sim0, index_trk, index_pnt, n = statements

    flt_trk0 = flt_sim0.flt_trks[index_trk]
    flt_img0 = flt_sim0.flt_imgs[index_trk]
    vnames_trk = flt_trk0.keys()
    vnames_img = flt_img0.keys()
    #\----------------------------------------------------------------------------/#


    # general plot settings
    #/----------------------------------------------------------------------------\#
    vars_plot = OrderedDict()

    vars_plot['SSFR-A↑']   = {
            'vname':'f-up_ssfr',
            'color':'red',
            'vname_wvl':'wvl_ssfr1_nad',
            'zorder': 7,
            }
    vars_plot['SSFR-A↓']   = {
            'vname':'f-down_ssfr',
            'color':'blue',
            'vname_wvl':'wvl_ssfr1_zen',
            'zorder': 6,
            }
    vars_plot['SSFR-B↑']   = {
            'vname':'r-up_ssfr',
            'color':'deeppink',
            'vname_wvl':'wvl_ssfr2_nad',
            'zorder': 2,
            }
    vars_plot['SSFR-B↓']   = {
            'vname':'r-down_ssfr',
            'color':'dodgerblue',
            'vname_wvl':'wvl_ssfr2_zen',
            'zorder': 3,
            }
    vars_plot['SPNS Total↓']   = {
            'vname':'f-down-total_spns',
            'color':'green',
            'vname_wvl':'wvl_spns',
            'zorder': 5,
            }
    vars_plot['SPNS Diffuse↓']   = {
            'vname':'f-down-diffuse_spns',
            'color':'springgreen',
            'vname_wvl':'wvl_spns',
            'zorder': 4,
            }
    vars_plot['TOA↓']   = {
            'vname':'f-down_toa',
            'color':'dimgray',
            'vname_wvl':'wvl_ssfr1_zen',
            'zorder': 1,
            }
    vars_plot['Altitude']   = {
            'vname':'alt',
            'color':'orange',
            'zorder': 0,
            }

    for vname in vars_plot.keys():

        vname_ori = vars_plot[vname]['vname']
        if vname_ori in vnames_trk:
            vars_plot[vname]['plot?'] = True
        else:
            vars_plot[vname]['plot?'] = False

        if 'vname_wvl' in vars_plot[vname].keys():
            vars_plot[vname]['spectra?'] = True
        else:
            vars_plot[vname]['spectra?'] = False
    #\----------------------------------------------------------------------------/#


    # plot settings
    #/----------------------------------------------------------------------------\#
    _aspect_ = 'auto'
    _alt_cmap_ = 'gist_ncar'

    _alt_base_ = 0.0
    _alt_ceil_ = 8.0

    _flux_base_ = 0.0
    _flux_ceil_ = 1.5

    hist_bins = np.linspace(0.0, 2.0, 81)
    hist_x = (hist_bins[1:]+hist_bins[:-1])/2.0
    hist_bin_w = hist_bins[1]-hist_bins[0]
    hist_bottoms = {key:0.0 for key in vars_plot.keys()}
    #\----------------------------------------------------------------------------/#


    # check data
    #/----------------------------------------------------------------------------\#
    has_trk = True

    if ('ang_pit' in vnames_trk) and ('ang_rol' in vnames_trk):
        has_att = True
    else:
        has_att = False

    if ('ang_pit_m' in vnames_trk) and ('ang_rol_m' in vnames_trk):
        has_att_corr = True
    else:
        has_att_corr = False

    if ('fnames_sat0' in vnames_img) and ('extent_sat0' in vnames_img):
        has_sat0 = True
    else:
        has_sat0 = False

    if ('fnames_sat1' in vnames_img) and ('extent_sat1' in vnames_img):
        has_sat1 = True
    else:
        has_sat1 = False

    if ('fnames_cam0' in vnames_img):
        has_cam0 = True
    else:
        has_cam0 = False
    #\----------------------------------------------------------------------------/#


    # param settings
    #/----------------------------------------------------------------------------\#
    tmhr_current = flt_trk0['tmhr'][index_pnt]
    jday_current = flt_trk0['jday'][index_pnt]
    lon_current  = flt_trk0['lon'][index_pnt]
    lat_current  = flt_trk0['lat'][index_pnt]
    alt_current  = flt_trk0['alt'][index_pnt]
    sza_current  = flt_trk0['sza'][index_pnt]
    dtime_current = er3t.util.jday_to_dtime(jday_current)

    tmhr_length  = 0.5 # half an hour
    tmhr_past    = tmhr_current-tmhr_length
    #\----------------------------------------------------------------------------/#


    # figure setup
    #/----------------------------------------------------------------------------\#
    fig = plt.figure(figsize=(16, 9))

    gs = gridspec.GridSpec(12, 20)

    # ax of all
    ax = fig.add_subplot(gs[:, :])

    # map of flight track overlay satellite imagery
    # proj0 = ccrs.Orthographic(
    #         central_longitude=(flt_sim0.extent[0]+flt_sim0.extent[1])/2.0,
    #         central_latitude=(flt_sim0.extent[2]+flt_sim0.extent[3])/2.0,
    #         )
    proj0 = ccrs.Orthographic(
            central_longitude=lon_current,
            central_latitude=lat_current,
            )
    # proj0 = ccrs.PlateCarree()
    ax_map = fig.add_subplot(gs[:8, :9], projection=proj0, aspect=_aspect_)

    # altitude colorbar next to the map
    divider_map = make_axes_locatable(ax_map)
    ax_alt_cbar = divider_map.append_axes('right', size='4%', pad=0.0, axes_class=maxes.Axes)

    # profile (shared y axis) next to the map
    ax_alt_prof = divider_map.append_axes('right', size='32%', pad=0.0, axes_class=maxes.Axes)

    # data histogram (shared x axis) next to the map
    ax_alt_hist = ax_alt_prof.twinx()


    # a secondary map
    ax_map0 = fig.add_subplot(gs[:5, 10:15], projection=ccrs.PlateCarree(), aspect=_aspect_)

    # camera imagery
    ax_img  = fig.add_subplot(gs[:5, 15:])
    ax_img_hist = ax_img.twinx()

    # spetral irradiance
    ax_wvl  = fig.add_subplot(gs[5:8, 10:])

    # aircraft and platform attitude status
    ax_nav  = inset_axes(ax_wvl, width=1.0, height=0.7, loc='upper center')
    ax_tms = fig.add_subplot(gs[9:, :])
    ax_tms_alt  = ax_tms.twinx()

    fig.subplots_adjust(hspace=10.0, wspace=10.0)
    #\----------------------------------------------------------------------------/#





    # flight direction
    #/----------------------------------------------------------------------------\#
    alt_cmap = mpl.colormaps[_alt_cmap_]
    alt_norm = mpl.colors.Normalize(vmin=_alt_base_, vmax=_alt_ceil_)

    dlon = flt_sim0.flt_imgs[index_trk]['extent_sat0'][1] - flt_sim0.flt_imgs[index_trk]['extent_sat0'][0]
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

    plot_arrow=False
    #\----------------------------------------------------------------------------/#


    # base plot
    #/----------------------------------------------------------------------------\#
    if has_sat0:
        lon_half = 15.0
        lat_half = 2.5
        lat_low = max([lat_current-lat_half, 76.0])
        lat_high= min([lat_low+lat_half*2.0, 87.0])
        lat_low = max([lat_high-lat_half*2.0, 76.0])
        extent_img = [lon_current-lon_half, lon_current+lon_half, lat_low, lat_high]
        ax_map.set_extent(extent_img, crs=ccrs.PlateCarree())

        fname_sat = flt_img0['fnames_sat0'][index_pnt]

        img = mpl_img.imread(fname_sat)

        extent_ori = flt_img0['extent_sat0_ori']
        lon_1d = np.linspace(extent_ori[0], extent_ori[1], img.shape[1]+1)
        lat_1d = np.linspace(extent_ori[2], extent_ori[3], img.shape[0]+1)

        extend_x = 0.05
        extend_y = 0.05
        lon_1d_ = np.linspace(extent_ori[0], extent_ori[1], img.shape[1])
        lat_1d_ = np.linspace(extent_ori[2], extent_ori[3], img.shape[0])
        indices_x = np.where((lon_1d_>=(extent_img[0]-extend_x))&(lon_1d_<=(extent_img[1]+extend_x)))[0]
        indices_y = np.where((lat_1d_>=(extent_img[2]-extend_y))&(lat_1d_<=(extent_img[3]+extend_y)))[0]

        if indices_x.size>2 and indices_y.size>2:

            index_xs = indices_x[0]
            index_xe = indices_x[-1]
            index_ys = indices_y[0]
            index_ye = indices_y[-1]
            lon_1d = lon_1d[index_xs:index_xe+1]
            lat_1d = lat_1d[index_ys:index_ye+1][::-1]

            lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
            img = img[img.shape[0]-index_ye:img.shape[0]-index_ys, index_xs:index_xe, :]

            logic_black = ~(np.sum(img[:, :, :-1], axis=-1)>0.0)
            img[logic_black, -1] = 0.0
            ax_map.pcolormesh(lon_2d, lat_2d, img, transform=ccrs.PlateCarree())

    if has_sat1:

        lat_half0 = 0.25
        lon_half0 = lat_half0*(lon_half/lat_half)*2.0
        lon_s = lon_current-lon_half0
        lon_e = lon_current+lon_half0
        lat_s = lat_current-lat_half0
        lat_e = lat_current+lat_half0

        extent_img = [lon_s, lon_e, lat_s, lat_e]

        ax_map0.set_extent(extent_img, crs=ccrs.PlateCarree())

        fname_sat1 = flt_img0['fnames_sat1'][index_pnt]

        img = mpl_img.imread(fname_sat1)

        extent_ori = flt_img0['extent_sat1_ori']
        lon_1d = np.linspace(extent_ori[0], extent_ori[1], img.shape[1]+1)
        lat_1d = np.linspace(extent_ori[2], extent_ori[3], img.shape[0]+1)

        extend_x = 0.05
        extend_y = 0.05
        lon_1d_ = np.linspace(extent_ori[0], extent_ori[1], img.shape[1])
        lat_1d_ = np.linspace(extent_ori[2], extent_ori[3], img.shape[0])
        indices_x = np.where((lon_1d_>=(extent_img[0]-extend_x))&(lon_1d_<=(extent_img[1]+extend_x)))[0]
        indices_y = np.where((lat_1d_>=(extent_img[2]-extend_y))&(lat_1d_<=(extent_img[3]+extend_y)))[0]

        if indices_x.size>2 and indices_y.size>2:

            index_xs = indices_x[0]
            index_xe = indices_x[-1]
            index_ys = indices_y[0]
            index_ye = indices_y[-1]
            lon_1d = lon_1d[index_xs:index_xe+1]
            lat_1d = lat_1d[index_ys:index_ye+1][::-1]

            lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
            img = img[img.shape[0]-index_ye:img.shape[0]-index_ys, index_xs:index_xe, :]

            logic_black = ~(np.sum(img[:, :, :-1], axis=-1)>0.0)
            img[logic_black, -1] = 0.0
            ax_map.pcolormesh(lon_2d, lat_2d, img, transform=ccrs.PlateCarree(), zorder=1)
            ax_map0.pcolormesh(lon_2d, lat_2d, img, transform=ccrs.PlateCarree())

    if has_cam0:
        ang_cam_offset = -53.0 # for ARCSIX
        cam_x_s = 5.0
        cam_x_e = 255.0*4.0
        cam_y_s = 0.0
        cam_y_e = 0.12
        cam_hist_x_s = 0.0
        cam_hist_x_e = 255.0
        cam_hist_bins = np.linspace(cam_hist_x_s, cam_hist_x_e, 31)

        fname_cam = flt_img0['fnames_cam0'][index_pnt]
        img = mpl_img.imread(fname_cam)[:-200, 540:-640, :]

        # if ('ang_hed' in vnames_trk):
        #     ang_hed0   = flt_trk0['ang_hed'][index_pnt]
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
        ax_img.imshow(img_plot, origin='upper', aspect='auto', zorder=0, extent=[cam_x_s, cam_x_e, cam_y_s, cam_y_e])

        ax_img_hist.hist(img[:, :, 0].ravel(), bins=cam_hist_bins, histtype='step', lw=0.5, alpha=0.9, density=True, color='r')
        ax_img_hist.hist(img[:, :, 1].ravel(), bins=cam_hist_bins, histtype='step', lw=0.5, alpha=0.9, density=True, color='g')
        ax_img_hist.hist(img[:, :, 2].ravel(), bins=cam_hist_bins, histtype='step', lw=0.5, alpha=0.9, density=True, color='b')
        ax_img.plot([255, 255], [0, 0.005], color='white', lw=1.0, ls='-')

    if has_att:

        ax_nav.axhspan(-10.0, 0.0, lw=0.0, color='orange', zorder=0, alpha=0.3)
        ax_nav.axhspan(0.0,  10.0, lw=0.0, color='deepskyblue', zorder=0, alpha=0.3)

        ax_nav.axvline(0.0, lw=0.5, color='gray', zorder=1)
        ax_nav.axhline(0.0, lw=0.5, color='gray', zorder=1)

        ang_pit0 = flt_trk0['ang_pit'][index_pnt]
        ang_rol0 = flt_trk0['ang_rol'][index_pnt]

        x  = np.linspace(-10.0, 10.0, 101)

        slope0  = -np.tan(np.deg2rad(ang_rol0))
        offset0 = ang_pit0
        y0 = slope0*x + offset0

        ax_nav.plot(x[15:-15], y0[15:-15], lw=1.0, color='red', zorder=1, alpha=0.6)
        ax_nav.scatter(x[50], y0[50], lw=0.0, s=40, c='red', zorder=1, alpha=0.6)

        if has_att_corr:
            ang_pit_offset = 0.0
            ang_rol_offset = 0.0

            ang_pit0 = flt_trk0['ang_pit_s'][index_pnt]
            ang_rol0 = flt_trk0['ang_rol_s'][index_pnt]
            ang_pit_m0 = flt_trk0['ang_pit_m'][index_pnt]
            ang_rol_m0 = flt_trk0['ang_rol_m'][index_pnt]


            slope1  = -np.tan(np.deg2rad(ang_rol0-ang_rol_m0+ang_rol_offset))
            offset1 = (ang_pit0-ang_pit_m0+ang_pit_offset)
            y1 = slope1*x + offset1

            ax_nav.plot(x[25:-25], y1[25:-25], lw=2.0, color='green', zorder=2, alpha=0.7)

    for vname in vars_plot.keys():
        var_plot = vars_plot[vname]
        if var_plot['plot?']:
            if 'vname_wvl' in var_plot.keys():
                wvl_x  = flt_trk0[var_plot['vname_wvl']]
                if 'toa' in var_plot['vname']:
                    spec_y = flt_trk0[var_plot['vname']] * np.cos(np.deg2rad(sza_current))
                else:
                    spec_y = flt_trk0[var_plot['vname']][index_pnt, :]

                # ax_wvl.scatter(wvl_x, spec_y, c=var_plot['color'], s=4, lw=0.0, zorder=var_plot['zorder'])
                ax_wvl.plot(wvl_x, spec_y,
                        color=var_plot['color'], marker='o', markersize=2, lw=0.5, markeredgewidth=0.0, alpha=0.9, zorder=var_plot['zorder'])

                wvl_index = np.argmin(np.abs(wvl_x-flt_sim0.wvl0))
                ax_wvl.axvline(wvl_x[wvl_index], color=var_plot['color'], ls='-', lw=1.0, alpha=0.5, zorder=var_plot['zorder'])
    #\----------------------------------------------------------------------------/#


    # iterate through flight segments
    #/----------------------------------------------------------------------------\#
    for itrk in range(index_trk+1):

        flt_trk = flt_sim0.flt_trks[itrk]
        flt_img = flt_sim0.flt_imgs[itrk]

        logic_solid = (flt_trk['tmhr']>=tmhr_past) & (flt_trk['tmhr']<=tmhr_current)
        logic_trans = np.logical_not(logic_solid)

        if itrk == index_trk:
            alpha_trans = 0.0
        else:
            alpha_trans = 0.08

        ax_map.scatter(         flt_trk['lon'][logic_trans], flt_trk['lat'][logic_trans], c=flt_trk['alt'][logic_trans], s=0.5, lw=0.0, zorder=1, vmin=_alt_base_, vmax=_alt_ceil_, cmap=_alt_cmap_, alpha=alpha_trans, transform=ccrs.PlateCarree())
        cs_alt = ax_map.scatter(flt_trk['lon'][logic_solid], flt_trk['lat'][logic_solid], c=flt_trk['alt'][logic_solid], s=1  , lw=0.0, zorder=2, vmin=_alt_base_, vmax=_alt_ceil_, cmap=_alt_cmap_, transform=ccrs.PlateCarree())

        ax_map0.scatter(flt_trk['lon'][logic_trans], flt_trk['lat'][logic_trans], c=flt_trk['alt'][logic_trans], s=2.5, lw=0.0, zorder=1, vmin=_alt_base_, vmax=_alt_ceil_, cmap=_alt_cmap_, alpha=alpha_trans, transform=ccrs.PlateCarree())
        ax_map0.scatter(flt_trk['lon'][logic_solid], flt_trk['lat'][logic_solid], c=flt_trk['alt'][logic_solid], s=4  , lw=0.0, zorder=2, vmin=_alt_base_, vmax=_alt_ceil_, cmap=_alt_cmap_, transform=ccrs.PlateCarree())

        if not plot_arrow:
            ax_map.scatter(lon_current, lat_current, facecolor='none', edgecolor='white', s=60, lw=1.0, zorder=3, alpha=0.6, transform=ccrs.PlateCarree())
            ax_map.scatter(lon_current, lat_current, c=alt_current, s=60, lw=0.0, zorder=3, alpha=0.6, vmin=_alt_base_, vmax=_alt_ceil_, cmap=_alt_cmap_, transform=ccrs.PlateCarree())
            # ax_map0.scatter(lon_current, lat_current, facecolor='none', edgecolor='white', s=60, lw=1.0, zorder=3, alpha=0.6)
            # ax_map0.scatter(lon_current, lat_current, c=alt_current, s=60, lw=0.0, zorder=3, alpha=0.6, vmin=_alt_base_, vmax=_alt_ceil_, cmap=_alt_cmap_)
        else:
            color0 = alt_cmap(alt_norm(alt_current))
            arrow_prop['facecolor'] = color0
            arrow_prop['relpos'] = (lon_current, lat_current)
            ax_map.annotate('', xy=(lon_point_to, lat_point_to), xytext=(lon_current, lat_current), arrowprops=arrow_prop, zorder=3, transform=ccrs.PlateCarree())
            # ax_map0.annotate('', xy=(lon_point_to, lat_point_to), xytext=(lon_current, lat_current), arrowprops=arrow_prop, zorder=3)

        ax_map0.scatter(lon_current, lat_current, facecolor='none', edgecolor='white', s=60, lw=1.0, zorder=3, alpha=0.6, transform=ccrs.PlateCarree())
        ax_map0.scatter(lon_current, lat_current, c=alt_current, s=60, lw=0.0, zorder=3, alpha=0.6, vmin=_alt_base_, vmax=_alt_ceil_, cmap=_alt_cmap_, transform=ccrs.PlateCarree())

        for vname in vars_plot.keys():

            var_plot = vars_plot[vname]

            if var_plot['plot?']:
                if 'vname_wvl' in var_plot.keys():
                    wvl_x  = flt_trk[var_plot['vname_wvl']]
                    index_wvl = np.argmin(np.abs(wvl_x-flt_sim0.wvl0))
                    if 'toa' in var_plot['vname']:
                        tms_y = flt_trk[var_plot['vname']][index_wvl] * np.cos(np.deg2rad(flt_trk['sza']))
                    else:
                        tms_y = flt_trk[var_plot['vname']][:, index_wvl]
                else:
                    tms_y = flt_trk[var_plot['vname']]

                if vname == 'Altitude':
                    ax_tms_alt.fill_between(flt_trk['tmhr'][logic_solid], tms_y[logic_solid], facecolor=vars_plot[vname]['color'], alpha=0.25, lw=0.0, zorder=var_plot['zorder'])
                else:

                    if vname not in ['TOA↓']:
                        if has_att:
                            ang_pit_solid = flt_trk['ang_pit'][logic_solid]
                            ang_rol_solid = flt_trk['ang_rol'][logic_solid]
                            logic_stable = (np.abs(ang_pit_solid)<=5.0) & (np.abs(ang_rol_solid)<=2.5)
                            ax_alt_prof.scatter(tms_y[logic_solid][~logic_stable], flt_trk['alt'][logic_solid][~logic_stable], c=var_plot['color'], s=1, lw=0.0, zorder=var_plot['zorder'], alpha=0.15)
                            ax_alt_prof.scatter(tms_y[logic_solid][logic_stable] , flt_trk['alt'][logic_solid][logic_stable] , c=var_plot['color'], s=2, lw=0.0, zorder=var_plot['zorder']*2)

                            hist_y, _ = np.histogram(tms_y[logic_solid][logic_stable], bins=hist_bins)
                            ax_alt_hist.bar(hist_x, hist_y, width=hist_bin_w, bottom=hist_bottoms[vname], color=var_plot['color'], alpha=0.5, lw=0.0, zorder=var_plot['zorder']-1)
                            hist_bottoms[vname] += hist_y

                            ax_tms.scatter(flt_trk['tmhr'][logic_solid][~logic_stable], tms_y[logic_solid][~logic_stable], c=vars_plot[vname]['color'], s=1, lw=0.0, zorder=var_plot['zorder'], alpha=0.4)
                            ax_tms.scatter(flt_trk['tmhr'][logic_solid][logic_stable], tms_y[logic_solid][logic_stable], c=vars_plot[vname]['color'], s=2, lw=0.0, zorder=var_plot['zorder']*2)
                        else:
                            ax_alt_prof.scatter(tms_y[logic_solid], flt_trk['alt'][logic_solid], c=vars_plot[vname]['color'], s=1, lw=0.0, zorder=var_plot['zorder'])

                            hist_y = np.histogram(tms_y[logic_solid], bins=hist_bins)
                            ax_alt_hist.bar(hist_x, hist_y, width=hist_bin_w, bottom=hist_bottoms[vname], color=var_plot['color'], alpha=0.5, lw=0.0, zorder=var_plot['zorder']-1)
                            hist_bottoms[vname] += hist_y

                            ax_tms.scatter(flt_trk['tmhr'][logic_solid], tms_y[logic_solid], c=vars_plot[vname]['color'], s=2, lw=0.0, zorder=var_plot['zorder'])
                    else:
                        ax_tms.scatter(flt_trk['tmhr'][logic_solid], tms_y[logic_solid], c=vars_plot[vname]['color'], s=2, lw=0.0, zorder=var_plot['zorder'])
    #\----------------------------------------------------------------------------/#


    has_spectra = any([vars_plot[key]['plot?'] for key in vars_plot.keys() if vars_plot[key]['spectra?']])


    # figure settings
    #/----------------------------------------------------------------------------\#
    title_fig = '%s UTC' % (dtime_current.strftime('%Y-%m-%d %H:%M:%S'))
    if (_title_extra_ is not None) and (_title_extra_!=''):
        title_fig = '%s\n%s' % (_title_extra_, title_fig)
    fig.suptitle(title_fig, y=0.98, fontsize=20)
    #\----------------------------------------------------------------------------/#


    # map plot settings
    #/----------------------------------------------------------------------------\#
    if has_sat0:

        title_map = '%s at %s UTC' % (flt_img0['id_sat0'][index_pnt], er3t.util.jday_to_dtime(flt_img0['jday_sat0'][index_pnt]).strftime('%H:%M'))
        time_diff = np.abs(flt_img0['jday_sat0'][index_pnt]-jday_current)*86400.0
        if time_diff > 301.0:
            ax_map.set_title(title_map, color='gray')
        else:
            ax_map.set_title(title_map)

        ax_map.coastlines(resolution='10m', color='black', lw=0.5)
        g1 = ax_map.gridlines(lw=0.5, color='gray', draw_labels=True, ls='-')
        g1.xlocator = FixedLocator(np.arange(-180, 181, 10.0))
        g1.ylocator = FixedLocator(np.arange(-90.0, 89.9, 1.0))
        g1.top_labels = False
        g1.right_labels = False
    #\----------------------------------------------------------------------------/#


    # navigation plot settings
    #/----------------------------------------------------------------------------\#
    ax_nav.set_xlim((-10.0, 10.0))
    ax_nav.set_ylim((-10.0, 10.0))
    ax_nav.axis('off')
    #\----------------------------------------------------------------------------/#


    # map0 plot settings
    #/----------------------------------------------------------------------------\#
    if has_sat1:
        title_map0 = 'False Color 721'
        title_map0 = 'True Color'
        time_diff = np.abs(flt_img0['jday_sat1'][index_pnt]-jday_current)*86400.0
        if time_diff > 301.0:
            ax_map0.set_title(title_map0, color='gray')
        else:
            ax_map0.set_title(title_map0)

        ax_map0.coastlines(resolution='10m', color='black', lw=0.5)
        g2 = ax_map0.gridlines(lw=0.5, color='gray', ls='-')
        g2.xlocator = FixedLocator(np.arange(-180.0, 180.1, 1.0))
        g2.ylocator = FixedLocator(np.arange(-89.0, 89.1, 0.5))
    # ax_map0.axis('off')
    #\----------------------------------------------------------------------------/#


    # camera image plot settings
    #/----------------------------------------------------------------------------\#
    if has_cam0:
        jday_cam  = flt_img0['jday_cam0'][index_pnt]
        dtime_cam = er3t.util.jday_to_dtime(jday_cam)

        title_img = 'Camera at %s UTC' % (dtime_cam.strftime('%H:%M:%S'))
        time_diff = np.abs(jday_current-jday_cam)*86400.0
        if time_diff > 301.0:
            ax_img.set_title(title_img, color='gray')
        else:
            ax_img.set_title(title_img)

        ax_img_hist.set_xlim((cam_x_s, cam_x_e))
        ax_img_hist.set_ylim((cam_y_s, cam_y_e))

    ax_img.axis('off')
    ax_img_hist.axis('off')
    #\----------------------------------------------------------------------------/#


    # altitude (time series) plot settings
    #/----------------------------------------------------------------------------\#
    ax_tms_alt.set_ylim((0.0, 8.0))
    ax_tms_alt.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 8.1, 2.0)))
    ax_tms_alt.yaxis.tick_right()
    ax_tms_alt.yaxis.set_label_position('right')
    ax_tms_alt.set_ylabel('Altitude [km]', rotation=270.0, labelpad=18, color=vars_plot['Altitude']['color'])

    ax_tms_alt.set_frame_on(True)
    for spine in ax_tms_alt.spines.values():
        spine.set_visible(False)
    ax_tms_alt.spines['right'].set_visible(True)
    ax_tms_alt.spines['right'].set_color(vars_plot['Altitude']['color'])
    ax_tms_alt.tick_params(axis='y', which='both', colors=vars_plot['Altitude']['color'])
    #\----------------------------------------------------------------------------/#


    # time series plot settings
    #/----------------------------------------------------------------------------\#
    ax_tms.grid()
    ax_tms.set_xlim((tmhr_past-0.0000001, tmhr_current+0.0000001))
    ax_tms.xaxis.set_major_locator(FixedLocator([tmhr_past, tmhr_current-0.5*tmhr_length, tmhr_current]))
    ax_tms.xaxis.set_minor_locator(FixedLocator(np.arange(tmhr_past, tmhr_current+0.001, 1.0/60.0)))
    ax_tms.set_xticklabels(['%.4f' % (tmhr_past), '%.4f' % (tmhr_current-0.5*tmhr_length), '%.4f' % tmhr_current])
    ax_tms.set_xlabel('Time [hour]')

    if has_spectra:
        ax_tms.set_ylim(bottom=_flux_base_, top=min([_flux_ceil_, ax_tms.get_ylim()[-1]+0.15]))
        ax_tms.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 10.1, 0.5)))
        ax_tms.yaxis.set_minor_locator(FixedLocator(np.arange(0.0, 10.1, 0.1)))
        ax_tms.set_ylabel('Flux [$\\mathrm{W m^{-2} nm^{-1}}$]')
    else:
        ax_tms.yaxis.set_ticks([])

    if alt_current < 1.0:
        title_all = 'Longitude %9.4f$^\\circ$, Latitude %8.4f$^\\circ$, Altitude %6.1f  m, Solar Zenith %4.1f$^\\circ$' % (lon_current, lat_current, alt_current*1000.0, sza_current)
    else:
        title_all = 'Longitude %9.4f$^\\circ$, Latitude %8.4f$^\\circ$, Altitude %6.4f km, Solar Zenith %4.1f$^\\circ$' % (lon_current, lat_current, alt_current, sza_current)
    ax_tms.set_title(title_all)

    ax_tms.spines['right'].set_visible(False)
    ax_tms.set_zorder(ax_tms_alt.get_zorder()+1)
    ax_tms.patch.set_visible(False)
    #\----------------------------------------------------------------------------/#


    # spectra plot setting
    #/----------------------------------------------------------------------------\#
    if has_spectra:
        ax_wvl.set_xlim((200, 2200))
        ax_wvl.set_ylim(ax_tms.get_ylim())
        ax_wvl.xaxis.set_major_locator(FixedLocator(np.arange(0, 2401, 400)))
        ax_wvl.xaxis.set_minor_locator(FixedLocator(np.arange(0, 2401, 100)))
        ax_wvl.set_xlabel('Wavelength [nm]')
        ax_wvl.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 10.1, 0.5)))
        ax_wvl.yaxis.set_minor_locator(FixedLocator(np.arange(0.0, 10.1, 0.1)))
    else:
        ax_wvl.axis('off')
    #\----------------------------------------------------------------------------/#


    # profile plot
    #/----------------------------------------------------------------------------\#
    ax_alt_prof.axhline(alt_current, lw=1.0, color=vars_plot['Altitude']['color'], zorder=1, alpha=0.9)
    ax_alt_prof.grid()

    if has_spectra:
        ax_alt_prof.set_xlim(ax_tms.get_ylim())
        ax_alt_prof.xaxis.set_major_locator(FixedLocator(np.arange(0.5, 10.1, 0.5)))
        ax_alt_prof.xaxis.set_minor_locator(FixedLocator(np.arange(0.0, 10.1, 0.1)))
    else:
        ax_alt_prof.xaxis.set_ticks([])

    ax_alt_prof.set_ylim(
            bottom=max([_alt_base_, ax_alt_prof.get_ylim()[0]-0.5]),
            top=min([ax_alt_prof.get_ylim()[-1]+0.5, _alt_ceil_]),
            )
    ax_alt_prof.yaxis.set_label_position('right')
    ax_alt_prof.yaxis.tick_right()
    ax_alt_prof.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 8.1, 1.0)))
    ax_alt_prof.yaxis.set_minor_locator(FixedLocator(np.arange(0.0, 8.1, 0.1)))

    ax_alt_prof.set_ylabel('Altitude [km]', rotation=270.0, labelpad=18, color=vars_plot['Altitude']['color'])
    ax_alt_prof.spines['right'].set_visible(True)
    ax_alt_prof.spines['right'].set_color(vars_plot['Altitude']['color'])
    ax_alt_prof.tick_params(axis='y', which='both', colors=vars_plot['Altitude']['color'])
    #\----------------------------------------------------------------------------/#


    # histogram plot
    #/----------------------------------------------------------------------------\#
    ax_alt_hist.set_xlim(ax_tms.get_ylim())
    ax_alt_hist.set_ylim((0, 5000))
    ax_alt_hist.axis('off')
    #\----------------------------------------------------------------------------/#


    # altitude/sza plot settings
    #/----------------------------------------------------------------------------\#
    cbar = fig.colorbar(cs_alt, cax=ax_alt_cbar)
    ax_alt_cbar.set_ylim(ax_alt_prof.get_ylim())
    ax_alt_cbar.xaxis.set_ticks([])
    ax_alt_cbar.yaxis.set_ticks([])
    #\----------------------------------------------------------------------------/#


    # acknowledgements
    #/----------------------------------------------------------------------------\#
    text1 = '\
presented by ARCSIX SSFR Team - Hong Chen, Vikas Nataraja, Yu-Wen Chen, Ken Hirata, Arabella Chamberlain, Katey Dong, Jeffery Drouet, and Sebastian Schmidt\n\
'
    ax.annotate(text1, xy=(0.5, 0.25), fontsize=8, color='gray', xycoords='axes fraction', ha='center', va='center')
    ax.axis('off')
    #\----------------------------------------------------------------------------/#


    # legend plot settings
    #/----------------------------------------------------------------------------\#
    patches_legend = []
    for vname in vars_plot.keys():
        var_plot = vars_plot[vname]
        if vname.lower() != 'altitude' and var_plot['plot?']:
            patches_legend.append(mpatches.Patch(color=var_plot['color'], label=vname))
    if len(patches_legend) > 0:
        ax_wvl.legend(handles=patches_legend, loc='upper right', fontsize=10)
    #\----------------------------------------------------------------------------/#


    if test:
        time_e = time.time()
        print('Elapsed time: %.1f seconds.' % (time_e-time_s))
        plt.show()
        sys.exit()
    else:
        plt.savefig('%s/%5.5d.png' % (_fdir_tmp_graph_, n), bbox_inches='tight')
        plt.close(fig)

def main_pre(
        date,
        wvl0=_wavelength_,
        run_rtm=False,
        time_step=1,
        wvl_step_spns=10,
        wvl_step_ssfr=3,
        ):

    # date time stamp
    #/----------------------------------------------------------------------------\#
    date_s = date.strftime('%Y%m%d')
    #\----------------------------------------------------------------------------/#

    # read in aircraft hsk data
    #/----------------------------------------------------------------------------\#
    fname_hsk = '%s/%s-%s_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _hsk_.upper(), _platform_.upper(), date_s)

    f_hsk = h5py.File(fname_hsk, 'r')
    jday   = f_hsk['jday'][...]
    tmhr   = f_hsk['tmhr'][...]
    sza    = f_hsk['sza'][...]
    lon    = f_hsk['lon'][...]
    lat    = f_hsk['lat'][...]

    # !!!!!!!!!
    # force flight track to start at PSB
    #/--------------------------------------------------------------\#
    # location of Pituffik Space Base (PSB)
    # lon0 = -68.70379848070486
    # lat0 = 76.53111177550895

    # lon = (lon-lon[~np.isnan(lon)][0]) + lon0
    # lat = (lat-lat[~np.isnan(lat)][0])/5.0 + lat0
    #\--------------------------------------------------------------/#

    logic0 = (~np.isnan(jday) & ~np.isinf(sza))   & \
             check_continuity(lon, threshold=1.0) & \
             check_continuity(lat, threshold=1.0) & \
             (tmhr>=_tmhr_range_[date_s][0]) & (tmhr<=_tmhr_range_[date_s][1])

    jday = jday[logic0][::time_step]
    tmhr = tmhr[logic0][::time_step]
    sza  = sza[logic0][::time_step]
    lon  = lon[logic0][::time_step]
    lat  = lat[logic0][::time_step]

    alt    = f_hsk['alt'][...][logic0][::time_step]

    ang_pit_ = f_hsk['ang_pit'][...][logic0][::time_step]
    ang_rol_ = f_hsk['ang_rol'][...][logic0][::time_step]

    f_hsk.close()
    #\--------------------------------------------------------------/#


    # read in nav data
    #/--------------------------------------------------------------\#
    fname_alp = '%s/%s-%s_%s_%s_v1.h5' % (_fdir_data_, _mission_.upper(), _alp_.upper(), _platform_.upper(), date_s)
    if os.path.exists(fname_alp):
        has_alp = True
    else:
        has_alp = False

    if has_alp:
        f_alp = h5py.File(fname_alp, 'r')
        ang_pit_s = f_alp['ang_pit_s'][...][logic0][::time_step]
        ang_rol_s = f_alp['ang_rol_s'][...][logic0][::time_step]
        ang_pit_m = f_alp['ang_pit_m'][...][logic0][::time_step]
        ang_rol_m = f_alp['ang_rol_m'][...][logic0][::time_step]
        f_alp.close()
    #\--------------------------------------------------------------/#


    # read in spns data
    #/--------------------------------------------------------------\#
    fname_spns = '%s/%s-%s_%s_%s_v2.h5' % (_fdir_data_, _mission_.upper(), _spns_.upper(), _platform_.upper(), date_s)
    # fname_spns = '%s/%s-%s_%s_%s_v1.h5' % (_fdir_data_, _mission_.upper(), _spns_.upper(), _platform_.upper(), date_s)
    if os.path.exists(fname_spns):
        has_spns = True
    else:
        has_spns = False

    if has_spns:
        f_spns = h5py.File(fname_spns, 'r')
        spns_tot_flux = f_spns['tot/flux'][...][logic0, :][::time_step, ::wvl_step_spns]
        spns_tot_wvl  = f_spns['tot/wvl'][...][::wvl_step_spns]
        spns_dif_flux = f_spns['dif/flux'][...][logic0, :][::time_step, ::wvl_step_spns]
        spns_dif_wvl  = f_spns['dif/wvl'][...][::wvl_step_spns]
        f_spns.close()
    #\--------------------------------------------------------------/#


    # read in ssfr-a data
    #/--------------------------------------------------------------\#
    fname_ssfr1 = '%s/%s-%s_%s_%s_v2.h5' % (_fdir_data_, _mission_.upper(), _ssfr1_.upper(), _platform_.upper(), date_s)
    # fname_ssfr1 = '%s/%s-%s_%s_%s_v1.h5' % (_fdir_data_, _mission_.upper(), _ssfr1_.upper(), _platform_.upper(), date_s)
    if os.path.exists(fname_ssfr1):
        has_ssfr1 = True
    else:
        has_ssfr1 = False

    if has_ssfr1:
        f_ssfr1 = h5py.File(fname_ssfr1, 'r')
        ssfr1_zen_flux = f_ssfr1['zen/flux'][...][logic0, :][::time_step, ::wvl_step_ssfr]
        ssfr1_zen_wvl  = f_ssfr1['zen/wvl'][...][::wvl_step_ssfr]
        ssfr1_nad_flux = f_ssfr1['nad/flux'][...][logic0, :][::time_step, ::wvl_step_ssfr]
        ssfr1_nad_wvl  = f_ssfr1['nad/wvl'][...][::wvl_step_ssfr]
        ssfr1_zen_toa  = f_ssfr1['zen/toa0'][...][::wvl_step_ssfr]
        f_ssfr1.close()
    #\--------------------------------------------------------------/#


    # read in ssfr-b data
    #/--------------------------------------------------------------\#
    fname_ssfr2 = '%s/%s-%s_%s_%s_v2.h5' % (_fdir_data_, _mission_.upper(), _ssfr2_.upper(), _platform_.upper(), date_s)
    # fname_ssfr2 = '%s/%s-%s_%s_%s_v1.h5' % (_fdir_data_, _mission_.upper(), _ssfr2_.upper(), _platform_.upper(), date_s)
    if os.path.exists(fname_ssfr2):
        has_ssfr2 = True
    else:
        has_ssfr2 = False

    # !!!!!!!!!!!!!
    # turn off SSFR-B
    #/--------------------------------------------------------------\#
    has_ssfr2 = False
    #\--------------------------------------------------------------/#

    if has_ssfr2:
        f_ssfr2 = h5py.File(fname_ssfr2, 'r')
        ssfr2_zen_rad = f_ssfr2['zen/rad'][...][logic0, :][::time_step, ::wvl_step_ssfr]
        ssfr2_zen_wvl = f_ssfr2['zen/wvl'][...][::wvl_step_ssfr]
        ssfr2_nad_rad = f_ssfr2['nad/rad'][...][logic0, :][::time_step, ::wvl_step_ssfr]
        ssfr2_nad_wvl = f_ssfr2['nad/wvl'][...][::wvl_step_ssfr]
        f_ssfr2.close()
    #\--------------------------------------------------------------/#


    # pre-process the aircraft and satellite data
    #/----------------------------------------------------------------------------\#
    # create a filter to remove invalid data, e.g., out of available satellite data time range,
    # invalid solar zenith angles etc.
    tmhr_interval = 10.0/60.0
    half_interval = tmhr_interval/48.0

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

    flt_trk['ang_pit'] = ang_pit_[logic]
    flt_trk['ang_rol'] = ang_rol_[logic]

    if has_alp:
        flt_trk['ang_pit_s'] = ang_pit_s[logic]
        flt_trk['ang_rol_s'] = ang_rol_s[logic]
        flt_trk['ang_pit_m'] = ang_pit_m[logic]
        flt_trk['ang_rol_m'] = ang_rol_m[logic]

    if has_spns:
        flt_trk['f-down-total_spns']   = spns_tot_flux[logic, :]
        flt_trk['f-down-diffuse_spns'] = spns_dif_flux[logic, :]
        flt_trk['f-down-direct_spns']  = flt_trk['f-down-total_spns'] - flt_trk['f-down-diffuse_spns']
        flt_trk['wvl_spns'] = spns_tot_wvl

    if has_ssfr1:
        flt_trk['f-down_ssfr']   = ssfr1_zen_flux[logic, :]
        flt_trk['f-up_ssfr']     = ssfr1_nad_flux[logic, :]
        flt_trk['wvl_ssfr1_zen'] = ssfr1_zen_wvl
        flt_trk['wvl_ssfr1_nad'] = ssfr1_nad_wvl
        flt_trk['f-down_toa']    = ssfr1_zen_toa

    if has_ssfr2:
        flt_trk['r-down_ssfr']   = ssfr2_zen_rad[logic, :]
        flt_trk['r-up_ssfr']     = ssfr2_nad_rad[logic, :]
        flt_trk['wvl_ssfr2_zen'] = ssfr2_zen_wvl
        flt_trk['wvl_ssfr2_nad'] = ssfr2_nad_wvl

    # partition the flight track into multiple mini flight track segments
    flt_trks = partition_flight_track(flt_trk, jday_edges, margin_x=0.2, margin_y=0.2)
    #\----------------------------------------------------------------------------/#


    # process camera imagery
    #/----------------------------------------------------------------------------\#
    fdirs = er3t.util.get_all_folders(_fdir_cam_img_, pattern='*%4.4d*%2.2d*%2.2d*nac*jpg*' % (date.year, date.month, date.day))
    if len(fdirs) > 0:
        has_cam = True
        fdir_cam0 = sorted(fdirs, key=os.path.getmtime)[-1]
        fnames_cam0 = sorted(glob.glob('%s/*.jpg' % (fdir_cam0)))
        jday_cam0 = get_jday_cam_img(date, fnames_cam0)
    else:
        has_cam = False
    #\----------------------------------------------------------------------------/#


    # process satellite imagery
    #/----------------------------------------------------------------------------\#
    date_sat_s  = date.strftime('%Y-%m-%d')

    fnames_sat0  = {
            'ca_archipelago': {
                'extent': [-158.00, -21.03, 76.38, 88.06],
                },
            'lincoln_sea': {
                'extent': [-120.00,  36.69, 77.94, 88.88],
                },
            }

    fnames_sat1 = copy.deepcopy(fnames_sat0)

    for key in fnames_sat0.keys():

        fdir_in = '%s/%s' % (_fdir_sat_img_vn_, key)

        fnames_fc = er3t.util.get_all_files(fdir_in, pattern='*FalseColor721*%s*Z*.png' % date_sat_s)
        jday_sat0_ , fnames_sat0_  = process_sat_img_vn(fnames_fc)
        fnames_sat0[key]['jday']    = jday_sat0_
        fnames_sat0[key]['fnames']  = fnames_sat0_

        fnames_tc = er3t.util.get_all_files(fdir_in, pattern='*TrueColor*%s*Z*.png' % date_sat_s)
        jday_sat1_, fnames_sat1_ = process_sat_img_vn(fnames_tc)
        fnames_sat1[key]['jday']   = jday_sat1_
        fnames_sat1[key]['fnames'] = fnames_sat1_
    #\----------------------------------------------------------------------------/#



    # process imagery
    #/----------------------------------------------------------------------------\#
    # create python dictionary to store corresponding satellite imagery data info
    #/--------------------------------------------------------------\#
    extent = get_extent(flt_trk['lon'], flt_trk['lat'], margin=0.2)

    flt_imgs = []
    for i in range(len(flt_trks)):

        flt_img = {}

        flt_trk0 = flt_trks[i]

        region_contain = {}
        jday_diff = {}
        for key in fnames_sat0.keys():
            region_contain[key] = contain_lonlat_check(flt_trk0['lon'], flt_trk0['lat'], fnames_sat0[key]['extent'])
            jday_diff[key] = np.abs(flt_trk0['jday0']-fnames_sat0[key]['jday'][np.argmin(np.abs(flt_trk0['jday0']-fnames_sat0[key]['jday']))])

        key1, key2 = fnames_sat0.keys()
        if region_contain[key1] and (not region_contain[key2]):
            region_select = key1
        elif (not region_contain[key1]) and region_contain[key2]:
            region_select = key2
        elif (not region_contain[key1]) and (not region_contain[key2]):
            region_select = _preferred_region_
        elif region_contain[key1] and region_contain[key2]:
            if jday_diff[key1] < jday_diff[key2]:
                region_select = key1
            elif jday_diff[key1] > jday_diff[key2]:
                region_select = key2
            else:
                region_select = _preferred_region_


        flt_img['id_sat0'] = []
        flt_img['fnames_sat0'] = []
        flt_img['extent_sat0'] = extent
        flt_img['extent_sat0_ori'] = fnames_sat0[region_select]['extent']
        flt_img['jday_sat0']   = np.array([], dtype=np.float64)

        flt_img['fnames_sat1'] = []
        flt_img['extent_sat1'] = extent
        flt_img['extent_sat1_ori'] = fnames_sat1[region_select]['extent']
        flt_img['jday_sat1']   = np.array([], dtype=np.float64)

        if has_cam:
            flt_img['fnames_cam0']  = []
            flt_img['jday_cam0'] = np.array([], dtype=np.float64)

        for j in range(flt_trks[i]['jday'].size):

            jday_sat0_ = fnames_sat0[region_select]['jday']
            fnames_sat0_ = fnames_sat0[region_select]['fnames']
            index_sat0 = np.argmin(np.abs(jday_sat0_-flt_trks[i]['jday'][j]))
            flt_img['id_sat0'].append(os.path.basename(fnames_sat0_[index_sat0]).split('_')[0].replace('-', ' '))
            flt_img['fnames_sat0'].append(fnames_sat0_[index_sat0])
            flt_img['jday_sat0'] = np.append(flt_img['jday_sat0'], jday_sat0_[index_sat0])

            jday_sat1_ = fnames_sat1[region_select]['jday']
            fnames_sat1_ = fnames_sat1[region_select]['fnames']
            index_sat1 = np.argmin(np.abs(jday_sat1_-flt_trks[i]['jday'][j]))
            flt_img['fnames_sat1'].append(fnames_sat1_[index_sat1])
            flt_img['jday_sat1'] = np.append(flt_img['jday_sat1'], jday_sat1_[index_sat1])

            if has_cam:
                index_cam = np.argmin(np.abs(jday_cam0-flt_trks[i]['jday'][j]))
                flt_img['fnames_cam0'].append(fnames_cam0[index_cam])
                flt_img['jday_cam0'] = np.append(flt_img['jday_cam0'], jday_cam0[index_cam])

        flt_imgs.append(flt_img)
    #\--------------------------------------------------------------/#


    # generate flt-sat combined file
    #/----------------------------------------------------------------------------\#
    fname = '%s/%s-FLT-VID_%s_%s_v0.pk' % (_fdir_main_, _mission_.upper(), _platform_.upper(), date_s)
    sim0 = flt_sim(
            date=date,
            wavelength=wvl0,
            extent=extent,
            flt_trks=flt_trks,
            flt_imgs=flt_imgs,
            fname=fname,
            overwrite=True,
            )
    #\----------------------------------------------------------------------------/#

def main_vid(
        date,
        wvl0=_wavelength_,
        interval=10,
        ):

    date_s = date.strftime('%Y%m%d')

    fdir = _fdir_tmp_graph_
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

    indices_trk = indices_trk[::interval]
    indices_pnt = indices_pnt[::interval]
    indices     = indices[::interval]

    statements = zip([flt_sim0]*indices_trk.size, indices_trk, indices_pnt, indices)

    with mp.Pool(processes=15) as pool:
        r = list(tqdm(pool.imap(plot_video_frame, statements), total=indices_trk.size))

    # make video
    fname_mp4 = '%s-FLT-VID_%s_%s.mp4' % (_mission_.upper(), _platform_.upper(), date_s)
    os.system('ffmpeg -y -framerate 30 -pattern_type glob -i "%s/*.png" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p %s' % (fdir, fname_mp4))
#\----------------------------------------------------------------------------/#



if __name__ == '__main__':


    dates = [
            # datetime.datetime(2024, 5, 17), # ARCSIX test flight #1 near NASA WFF
            # datetime.datetime(2024, 5, 21), # ARCSIX test flight #2 near NASA WFF
            # datetime.datetime(2024, 5, 24), # ARCSIX transit flight #1 from NASA WFF to Pituffik Space Base
            datetime.datetime(2024, 5, 28), # ARCSIX research flight #1 over Lincoln Sea; clear-sky spiral
            # datetime.datetime(2024, 5, 30), # ARCSIX research flight #2 over Lincoln Sea; cloud wall
        ]

    for date in dates[::-1]:

        if date < datetime.datetime(2024, 5, 22):

            #/----------------------------------------------------------------------------\#
            main_pre_wff(date)
            main_vid_wff(date, wvl0=_wavelength_)
            #\----------------------------------------------------------------------------/#

        else:

            #/----------------------------------------------------------------------------\#
            # main_pre(date)
            main_vid(date, wvl0=_wavelength_, interval=60) # make quickview video
            # main_vid(date, wvl0=_wavelength_, interval=20) # make sharable video
            # main_vid(date, wvl0=_wavelength_, interval=5)  # make complete video
            #\----------------------------------------------------------------------------/#
            pass


    sys.exit()


    # test
    #/----------------------------------------------------------------------------\#
    date = dates[-1]
    date_s = date.strftime('%Y%m%d')
    fname = '%s/%s-FLT-VID_%s_%s_v0.pk' % (_fdir_main_, _mission_.upper(), _platform_.upper(), date_s)
    flt_sim0 = flt_sim(fname=fname, overwrite=False)
    statements = (flt_sim0, 0, 100, 1730)
    # statements = (flt_sim0, 3, 443, 1730)
    # plot_video_frame_wff(statements, test=True)
    plot_video_frame(statements, test=True)
    #\----------------------------------------------------------------------------/#

    pass
