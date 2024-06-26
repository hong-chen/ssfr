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

_fdir_main_    = 'data/arcsix/flt-vid'
_fdir_sat_img_ = 'data/arcsix/sat-img'
_wavelength_   = 745.0

_fdir_data_ = 'data/processed'



def download_geo_sat_img(
        dtime_s,
        dtime_e=None,
        extent=[-60.5, -58.5, 12, 14],
        satellite='GOES-East',
        instrument='ABI',
        layer_name='Band2_Red_Visible_1km',
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
            fdir='./',
            wavelength=None,
            flt_trks=None,
            flt_imgs=None,
            fname=None,
            overwrite=False,
            overwrite_rtm=False,
            quiet=False,
            verbose=False
            ):

        self.date      = date
        self.wvl0      = wavelength
        self.fdir      = os.path.abspath(fdir)
        self.flt_trks  = flt_trks
        self.flt_imgs  = flt_imgs
        self.overwrite = overwrite
        self.quiet     = quiet
        self.verbose   = verbose

        if ((fname is not None) and (os.path.exists(fname)) and (not overwrite)):

            self.load(fname)

        elif (((flt_trks is not None) and (flt_imgs is not None) and (wavelength is not None)) and (fname is not None) and (os.path.exists(fname)) and (overwrite)) or \
             (((flt_trks is not None) and (flt_imgs is not None) and (wavelength is not None)) and (fname is not None) and (not os.path.exists(fname))):

            self.run(overwrite=overwrite_rtm)
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
                self.wvl0     = obj.wvl0
                self.fname    = obj.fname
                self.flt_trks = obj.flt_trks
                self.flt_imgs = obj.flt_imgs
            else:
                sys.exit('Error   [flt_sim]: File \'%s\' is not the correct pickle file to load.' % fname)

    def run(self, overwrite=True):

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



def plot_video_frame(statements, test=False):

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

    vars_plot['SSFR↑']   = {
            'vname':'f-up_ssfr',
            'color':'red',
            'vname_wvl':'wvl_ssfr_nad',
            'zorder': 5,
            }
    vars_plot['SSFR↓']   = {
            'vname':'f-down_ssfr',
            'color':'blue',
            'vname_wvl':'wvl_ssfr_zen',
            'zorder': 4,
            }
    vars_plot['SPNS Total↓']   = {
            'vname':'f-down-total_spns',
            'color':'green',
            'vname_wvl':'wvl_spns',
            'zorder': 3,
            }
    vars_plot['SPNS Diffuse↓']   = {
            'vname':'f-down-diffuse_spns',
            'color':'springgreen',
            'vname_wvl':'wvl_spns',
            'zorder': 2,
            }
    vars_plot['TOA↓']   = {
            'vname':'f-down_toa',
            'color':'black',
            'vname_wvl':'wvl_ssfr_zen',
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

    if ('fnames_sat_img' in vnames_img) and ('extent_sat_img' in vnames_img):
        has_sat_img = True
    else:
        has_sat_img = False

    if ('fnames_sat_img0' in vnames_img) and ('extent_sat_img0' in vnames_img):
        has_sat_img0 = True
    else:
        has_sat_img0 = False

    if ('fnames_cam_img' in vnames_img):
        has_cam_img = True
    else:
        has_cam_img = False
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
    alt_cmap = mpl.cm.get_cmap('jet')
    alt_norm = mpl.colors.Normalize(vmin=0.0, vmax=6.0)

    dlon = flt_sim0.flt_imgs[index_trk]['extent_sat_img'][1] - flt_sim0.flt_imgs[index_trk]['extent_sat_img'][0]
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
    if has_sat_img:
        fname_sat = flt_img0['fnames_sat_img'][index_pnt]
        img = mpl_img.imread(fname_sat)
        ax_map.imshow(img, extent=flt_img0['extent_sat_img'], origin='upper', aspect='auto', zorder=0)
        rect = mpatches.Rectangle((lon_current-0.25, lat_current-0.25), 0.5, 0.5, lw=1.0, ec='k', fc='none')
        ax_map.add_patch(rect)

    if has_sat_img0:
        fname_sat0 = flt_img0['fnames_sat_img0'][index_pnt]
        img = mpl_img.imread(fname_sat0)
        ax_map0.imshow(img, extent=flt_img0['extent_sat_img0'], origin='upper', aspect='auto', zorder=0)

    if has_cam_img:
        ang_cam_offset = -152.0
        cam_x_s = 5.0
        cam_x_e = 255.0*4.0
        cam_y_s = 0.0
        cam_y_e = 0.12
        cam_hist_x_s = 0.0
        cam_hist_x_e = 255.0
        cam_hist_bins = np.linspace(cam_hist_x_s, cam_hist_x_e, 31)

        fname_cam = flt_img0['fnames_cam_img'][index_pnt]
        img = mpl_img.imread(fname_cam)[210:, 550:-650, :]

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

            ang_pit_offset = 4.444537204377897
            ang_rol_offset = -0.5463839481366073

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

                ax_wvl.scatter(wvl_x, spec_y, c=var_plot['color'], s=6, lw=0.0, zorder=var_plot['zorder'])

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

        ax_map.scatter(flt_trk['lon'][logic_trans], flt_trk['lat'][logic_trans], c=flt_trk['alt'][logic_trans], s=0.5, lw=0.0, zorder=1, vmin=0.0, vmax=6.0, cmap='jet', alpha=alpha_trans)
        ax_map.scatter(flt_trk['lon'][logic_solid], flt_trk['lat'][logic_solid], c=flt_trk['alt'][logic_solid], s=1  , lw=0.0, zorder=2, vmin=0.0, vmax=6.0, cmap='jet')

        ax_map0.scatter(flt_trk['lon'][logic_trans], flt_trk['lat'][logic_trans], c=flt_trk['alt'][logic_trans], s=2.5, lw=0.0, zorder=1, vmin=0.0, vmax=6.0, cmap='jet', alpha=alpha_trans)
        ax_map0.scatter(flt_trk['lon'][logic_solid], flt_trk['lat'][logic_solid], c=flt_trk['alt'][logic_solid], s=4  , lw=0.0, zorder=2, vmin=0.0, vmax=6.0, cmap='jet')


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
                    ax_tms.scatter(flt_trk['tmhr'][logic_solid], tms_y[logic_solid], c=vars_plot[vname]['color'], s=4, lw=0.0, zorder=var_plot['zorder'])
    #\----------------------------------------------------------------------------/#


    # figure settings
    #/----------------------------------------------------------------------------\#
    title_fig = '%s UTC' % (dtime_current.strftime('%Y-%m-%d %H:%M:%S'))
    fig.suptitle(title_fig, y=0.96, fontsize=20)
    #\----------------------------------------------------------------------------/#


    # map plot settings
    #/----------------------------------------------------------------------------\#
    if has_sat_img:
        ax_map.set_xlim(flt_img0['extent_sat_img'][:2])
        ax_map.set_ylim(flt_img0['extent_sat_img'][2:])

        title_map = '%s at %s UTC' % (flt_img0['satID'][index_pnt], er3t.util.jday_to_dtime(flt_img0['jday_sat_img'][index_pnt]).strftime('%H:%M'))
        time_diff = np.abs(flt_img0['jday_sat_img'][index_pnt]-jday_current)*86400.0
        if time_diff > 301.0:
            ax_map.set_title(title_map, color='gray')
        else:
            ax_map.set_title(title_map)

    ax_map.xaxis.set_major_locator(FixedLocator(np.arange(-180.0, 180.1, 2.0)))
    ax_map.yaxis.set_major_locator(FixedLocator(np.arange(-90.0, 90.1, 2.0)))
    ax_map.set_xlabel('Longitude [$^\circ$]')
    ax_map.set_ylabel('Latitude [$^\circ$]')
    #\----------------------------------------------------------------------------/#


    # navigation plot settings
    #/----------------------------------------------------------------------------\#
    ax_nav.set_xlim((-10.0, 10.0))
    ax_nav.set_ylim((-10.0, 10.0))
    ax_nav.axis('off')
    #\----------------------------------------------------------------------------/#


    # map0 plot settings
    #/----------------------------------------------------------------------------\#
    if has_sat_img0:
        title_map0 = 'Zoomed-in View'
        time_diff = np.abs(flt_img0['jday_sat_img0'][index_pnt]-jday_current)*86400.0
        if time_diff > 301.0:
            ax_map0.set_title(title_map0, color='gray')
        else:
            ax_map0.set_title(title_map0)

    ax_map0.set_xlim((lon_current-0.25, lon_current+0.25))
    ax_map0.set_ylim((lat_current-0.25, lat_current+0.25))
    ax_map0.axis('off')
    #\----------------------------------------------------------------------------/#


    # camera image plot settings
    #/----------------------------------------------------------------------------\#
    if has_cam_img:
        jday_cam  = flt_img0['jday_cam_img'][index_pnt]
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


    # spectra plot setting
    #/----------------------------------------------------------------------------\#
    ax_wvl.set_xlim((200, 2200))
    ax_wvl.set_ylim((0.0, 2.0))
    ax_wvl.xaxis.set_major_locator(FixedLocator(np.arange(0, 2401, 400)))
    ax_wvl.xaxis.set_minor_locator(FixedLocator(np.arange(0, 2401, 100)))
    ax_wvl.set_xlabel('Wavelength [nm]')
    ax_wvl.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 2.1, 0.5)))
    ax_wvl.set_ylabel('Flux [$\mathrm{W m^{-2} nm^{-1}}$]')
    #\----------------------------------------------------------------------------/#


    # altitude plot settings
    #/----------------------------------------------------------------------------\#
    ax_alt.axhspan(0, (90.0-sza_current)/10.0, color='gray', lw=0.0, zorder=0, alpha=0.3)

    color0 = alt_cmap(alt_norm(alt_current))
    ax_alt.axhline(alt_current, lw=2.0, color=color0, zorder=1)

    ax_alt.set_ylim((0.0, 9.0))
    ax_alt.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 9.1, 3.0)))
    ax_alt.yaxis.set_minor_locator(FixedLocator(np.arange(0.0, 9.1, 1.0)))
    ax_alt.xaxis.set_ticks([])
    ax_alt.yaxis.tick_right()
    ax_alt.yaxis.set_label_position('right')
    ax_alt.set_ylabel('Sun Elevation [$\\times 10^\circ$] / Altitude [km]', rotation=270.0, labelpad=18)
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

    ax_tms.set_ylim((0.0, 2.0))
    ax_tms.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 2.1, 0.5)))
    ax_tms.set_ylabel('Flux [$\mathrm{W m^{-2} nm^{-1}}$]')

    if alt_current < 1.0:
        title_all = 'Longitude %9.4f$^\circ$, Latitude %8.4f$^\circ$, Altitude %6.1f  m' % (lon_current, lat_current, alt_current*1000.0)
    else:
        title_all = 'Longitude %9.4f$^\circ$, Latitude %8.4f$^\circ$, Altitude %6.4f km' % (lon_current, lat_current, alt_current)
    ax_tms.set_title(title_all)

    ax_tms.spines['right'].set_visible(False)
    ax_tms.set_zorder(ax_tms_alt.get_zorder()+1)
    ax_tms.patch.set_visible(False)
    #\----------------------------------------------------------------------------/#


    # acknowledgements
    #/----------------------------------------------------------------------------\#
    ax.axis('off')
    text1 = '\
presented by ARCSIX SSFR Team - Hong Chen, Vikas Nataraja, Yu-Wen Chen, Ken Hirata, Arabella Chamberlain, Katey Dong, Jeffery Drouet, and Sebastian Schmidt\n\
'
    ax.annotate(text1, xy=(0.5, 0.24), fontsize=8, color='gray', xycoords='axes fraction', ha='center', va='center')
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

    logic0 = (~np.isnan(jday) & ~np.isinf(sza))  & \
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
    ssfr_zen_toa  = f_ssfr['%s/toa0'     % which_dset][...][::wvl_step_ssfr]
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

    interval = 600.0 # seconds
    dtime_s = er3t.util.jday_to_dtime((jday[0] *86400.0//interval  )*interval/86400.0)
    dtime_e = er3t.util.jday_to_dtime((jday[-1]*86400.0//interval+1)*interval/86400.0)

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

    fnames_sat0 = fnames_sat.copy()
    jday_sat0 = jday_sat.copy()
    #\----------------------------------------------------------------------------/#
    # print(jday_sat)
    # print(fnames_sat)


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
    flt_imgs = []
    for i in range(len(flt_trks)):
        flt_img = {}

        flt_img['satID'] = []

        flt_img['fnames_sat_img'] = []
        flt_img['extent_sat_img'] = extent
        flt_img['jday_sat_img'] = np.array([], dtype=np.float64)

        flt_img['fnames_sat_img0'] = []
        flt_img['extent_sat_img0'] = extent
        flt_img['jday_sat_img0'] = np.array([], dtype=np.float64)

        flt_img['fnames_cam_img']  = []
        flt_img['jday_cam_img'] = np.array([], dtype=np.float64)

        for j in range(flt_trks[i]['jday'].size):

            index_sat = np.argmin(np.abs(jday_sat-flt_trks[i]['jday'][j]))
            flt_img['satID'].append(os.path.basename(fnames_sat[index_sat]).split('_')[0].replace('-', ' '))
            flt_img['fnames_sat_img'].append(fnames_sat[index_sat])
            flt_img['jday_sat_img'] = np.append(flt_img['jday_sat_img'], jday_sat[index_sat])

            # this will change
            #/--------------------------------------------------------------\#
            index_sat0 = np.argmin(np.abs(jday_sat0-flt_trks[i]['jday'][j]))
            flt_img['fnames_sat_img0'].append(fnames_sat0[index_sat0])
            flt_img['jday_sat_img0'] = np.append(flt_img['jday_sat_img0'], jday_sat0[index_sat0])
            #\--------------------------------------------------------------/#

            index_cam = np.argmin(np.abs(jday_cam-flt_trks[i]['jday'][j]))
            flt_img['fnames_cam_img'].append(fnames_cam[index_cam])
            flt_img['jday_cam_img'] = np.append(flt_img['jday_cam_img'], jday_cam[index_cam])

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


if __name__ == '__main__':


    dates = [
            datetime.datetime(2017, 8, 13), # ORACLES research flight
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
