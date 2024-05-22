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
# mpl.use('Agg')


import er3t

_mission_      = 'arcsix'
_platform_     = 'p3b'

# _ssfr_         = 'ssfr-a'
_ssfr_         = 'ssfr-b'

_fdir_main_    = 'data/%s/ssfr-vid' % _mission_
_wavelength_   = 555.0

_fdir_data_      = 'data/%s/processed' % _mission_
_fdir_tmp_graph_ = 'tmp-graph_%s-vid' % _ssfr_



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
        filename = os.path.basename(fname)
        dtime_s_ = filename.split('.')[0].split(' ')[-1]
        dtime_s = '%s_%s' % (date.strftime('%Y_%m_%d'), dtime_s_)
        dtime0 = datetime.datetime.strptime(dtime_s, '%Y_%m_%d_%H_%M_%SZ')
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
    vnames_trk = flt_trk0.keys()
    #\----------------------------------------------------------------------------/#


    # general plot settings
    #/----------------------------------------------------------------------------\#
    vars_plot = OrderedDict()

    vars_plot['zen|si|raw']   = {
            'vname':'zen_si_cnt_raw',
            'color':'dodgerblue',
            'vname_wvl':'zen_si_wvl',
            'zorder': 5,
            }
    vars_plot['zen|in|raw']   = {
            'vname':'zen_in_cnt_raw',
            'color':'springgreen',
            'vname_wvl':'zen_in_wvl',
            'zorder': 6,
            }
    vars_plot['nad|si|raw']   = {
            'vname':'nad_si_cnt_raw',
            'color':'deeppink',
            'vname_wvl':'nad_si_wvl',
            'zorder': 4,
            }
    vars_plot['nad|in|raw']   = {
            'vname':'nad_in_cnt_raw',
            'color':'darkorange',
            'vname_wvl':'nad_in_wvl',
            'zorder': 3,
            }
    vars_plot['zen|si|dc']   = {
            'vname':'zen_si_cnt_dc',
            'color':'dodgerblue',
            'vname_wvl':'zen_si_wvl',
            'zorder': 5,
            }
    vars_plot['zen|in|dc']   = {
            'vname':'zen_in_cnt_dc',
            'color':'springgreen',
            'vname_wvl':'zen_in_wvl',
            'zorder': 6,
            }
    vars_plot['nad|si|dc']   = {
            'vname':'nad_si_cnt_dc',
            'color':'deeppink',
            'vname_wvl':'nad_si_wvl',
            'zorder': 4,
            }
    vars_plot['nad|in|dc']   = {
            'vname':'nad_in_cnt_dc',
            'color':'darkorange',
            'vname_wvl':'nad_in_wvl',
            'zorder': 3,
            }
    vars_plot['zen|spec']   = {
            'vname':'zen_spec',
            'color':'blue',
            'vname_wvl':'zen_wvl',
            'zorder': 5,
            }
    vars_plot['nad|spec']   = {
            'vname':'nad_spec',
            'color':'red',
            'vname_wvl':'nad_wvl',
            'zorder': 4,
            }
    vars_plot['shutter']   = {
            'vname':'shutter',
            'color':'dimgray',
            'zorder': 1,
            }
    vars_plot['shutter_dc']   = {
            'vname':'shutter_dc',
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

    # params
    #/----------------------------------------------------------------------------\#
    count_base = -2**15
    count_ceil = 2**15
    dynamic_range = count_ceil-count_base
    #\----------------------------------------------------------------------------/#

    # param settings
    #/----------------------------------------------------------------------------\#
    tmhr_current = flt_trk0['tmhr'][index_pnt]
    jday_current = flt_trk0['jday'][index_pnt]
    int_time_zen_si_current, int_time_zen_in_current, int_time_nad_si_current, int_time_nad_in_current = flt_trk0['int_time'][index_pnt, :]
    dtime_current = er3t.util.jday_to_dtime(jday_current)

    tmhr_length  = 0.1 # half an hour
    tmhr_past    = tmhr_current-tmhr_length
    #\----------------------------------------------------------------------------/#


    # figure setup
    #/----------------------------------------------------------------------------\#
    fig = plt.figure(figsize=(16, 16))

    gs = gridspec.GridSpec(17, 17)

    # ax of all
    ax = fig.add_subplot(gs[:, :])

    ax_zen_si = fig.add_subplot(gs[:5, :8])
    ax_zen_in = fig.add_subplot(gs[:5, 9:])
    ax_nad_si = fig.add_subplot(gs[5:10, :8])
    ax_nad_in = fig.add_subplot(gs[5:10, 9:])
    ax_zen_si0 = ax_zen_si.twinx()
    ax_zen_in0 = ax_zen_in.twinx()
    ax_nad_si0 = ax_nad_si.twinx()
    ax_nad_in0 = ax_nad_in.twinx()

    # spetral irradiance
    ax_wvl  = fig.add_subplot(gs[10:14, :])
    ax_wvl0 = ax_wvl.twinx()

    ax_temp0_= ax_wvl.twinx()
    ax_temp0 = ax_temp0_.twiny()

    # time series
    ax_tms  = fig.add_subplot(gs[14:, :])

    fig.subplots_adjust(hspace=100.0, wspace=5.0)
    #\----------------------------------------------------------------------------/#


    axes_spec = {
            'zen|si': ax_zen_si,
            'zen|in': ax_zen_in,
            'nad|si': ax_nad_si,
            'nad|in': ax_nad_in,
            }
    axes_spec0 = {
            'zen|si': ax_zen_si0,
            'zen|in': ax_zen_in0,
            'nad|si': ax_nad_si0,
            'nad|in': ax_nad_in0,
            }

    spec_info = {
            'zen|si': {'full_name': 'Zenith Silicon', 'int_time': int_time_zen_si_current},
            'zen|in': {'full_name': 'Zenith InGaAs' , 'int_time': int_time_zen_in_current},
            'nad|si': {'full_name': 'Nadir Silicon' , 'int_time': int_time_nad_si_current},
            'nad|in': {'full_name': 'Nadir InGaAs'  , 'int_time': int_time_nad_in_current},
            }

    Nchan = {
            'zen|si': 75,
            'zen|in': 202,
            }
    Nchan['nad|si'] = np.argmin(np.abs(flt_trk0['zen_si_wvl'][Nchan['zen|si']]-flt_trk0['nad_si_wvl']))
    Nchan['nad|in'] = np.argmin(np.abs(flt_trk0['zen_in_wvl'][Nchan['zen|in']]-flt_trk0['nad_in_wvl']))

    for key in axes_spec.keys():
        ax_spec  = axes_spec[key]
        ax_spec0 = axes_spec0[key]
        var_plot_raw = vars_plot['%s|raw' % key]
        var_plot_dc  = vars_plot['%s|dc' % key]

        ax_spec.plot(flt_trk0[var_plot_raw['vname']][index_pnt, :],
                color='k', marker='o', markersize=2, lw=0.5, markeredgewidth=0.0, alpha=1.0, zorder=10)
        ax_spec.plot(flt_trk0[var_plot_raw['vname']][index_pnt, :]-flt_trk0[var_plot_dc['vname']][index_pnt, :],
                color='r', marker='o', markersize=2, lw=0.5, markeredgewidth=0.0, alpha=0.6, zorder=8)
        ax_spec0.plot(flt_trk0[var_plot_dc['vname']][index_pnt, :],
                color='g', marker='o', markersize=2, lw=0.5, markeredgewidth=0.0, alpha=0.6, zorder=9)

        ax_spec.axhspan(count_base, count_ceil  , color='black', alpha=0.08, zorder=1)

        ax_spec.grid(lw=0.5)
        ax_spec.set_xlim((0, 255))
        ax_spec.set_ylim((count_base-2000, count_ceil+2000))

        ax_spec.xaxis.set_major_locator(FixedLocator(np.arange(0, 300, 50)))
        ax_spec.xaxis.set_minor_locator(FixedLocator(np.arange(0, 300, 10)))
        ax_spec.yaxis.set_major_locator(FixedLocator(np.arange(-40000, 80001, 10000)))
        ax_spec.yaxis.set_minor_locator(FixedLocator(np.arange(-40000, 80001, 5000)))
        ax_spec.ticklabel_format(axis='y', style='sci', scilimits=(0, 4), useMathText=True)

        if key in ['zen|si', 'nad|si']:
            ax_spec.set_ylabel('Digital Counts')
            ax_spec.axvline(Nchan[key], color=var_plot_dc['color'], ls='--')

        if key in ['nad|si', 'nad|in']:
            ax_spec.set_xlabel('Channel Number')

        ax_spec.set_title('%s (%3d ms)' % (spec_info[key]['full_name'], spec_info[key]['int_time']), color=var_plot_dc['color'])

        ax_spec0.set_ylim((-2000, dynamic_range+2000))
        ax_spec0.yaxis.set_major_locator(FixedLocator(np.arange(-40000, 80001, 20000)))
        ax_spec0.yaxis.set_minor_locator(FixedLocator(np.arange(-40000, 80001,  5000)))
        ax_spec0.ticklabel_format(axis='y', style='sci', scilimits=(0, 4), useMathText=True, useOffset=True)

        ax_spec0.set_frame_on(True)
        for spine in ax_spec0.spines.values():
            spine.set_visible(False)
        ax_spec0.spines['right'].set_visible(True)
        ax_spec0.spines['right'].set_color('green')
        ax_spec0.tick_params(axis='y', colors='green')

        if key in ['zen|in', 'nad|in']:
            ax_spec0.set_ylabel('Digital Counts', color='green', rotation=270, labelpad=18)
            ax_spec.axvline(Nchan[key], color=var_plot_dc['color'], ls='--')

    patches_legend = [
                      mpatches.Patch(color='black' , label='Raw Counts'), \
                      mpatches.Patch(color='red'   , label='Dark Counts'), \
                      mpatches.Patch(color='green' , label='Dark Corrected Counts'), \
                     ]
    ax.legend(handles=patches_legend, bbox_to_anchor=(0.2, 1.03, 0.6, .102), loc=3, ncol=len(patches_legend), mode="expand", borderaxespad=0., frameon=False, handletextpad=0.2, fontsize=16)


    # spectra plot setting
    #/----------------------------------------------------------------------------\#
    for key in ['zen', 'nad']:
        var_plot = vars_plot['%s|spec' % key]
        if var_plot['plot?']:
            ax_wvl.plot(flt_trk0[var_plot['vname_wvl']], flt_trk0[var_plot['vname']][index_pnt, :],
                    color=var_plot['color'], marker='o', markersize=2, lw=0.5, markeredgewidth=0.0, alpha=1.0, zorder=10)
        for key0 in ['si', 'in']:
            var_plot0 = vars_plot['%s|%s|dc' % (key, key0)]
            ax_wvl0.fill_between(flt_trk0[var_plot0['vname_wvl']], 0.0, flt_trk0[var_plot0['vname']][index_pnt, :],
                    color=var_plot0['color'], lw=0.0,  alpha=0.3, zorder=5)
            ax_wvl0.axvline(flt_trk0[var_plot0['vname_wvl']][Nchan['%s|%s' % (key, key0)]], color=var_plot0['color'], ls='--')

    ax_wvl.axvline(950.0, color='black', lw=1.0, alpha=1.0, zorder=1, ls=':')
    ax_wvl.axhline(0.0  , color='black', lw=1.0, alpha=1.0, zorder=1)
    # ax_wvl.grid(lw=0.5, zorder=0)
    ax_wvl.set_xlim((200, 2400))
    ax_wvl.xaxis.set_major_locator(FixedLocator(np.arange(200, 2401, 200)))
    ax_wvl.xaxis.set_minor_locator(FixedLocator(np.arange(0, 2401, 100)))
    ax_wvl.yaxis.set_major_locator(FixedLocator(np.arange(0.0, 2.1, 0.5)))
    ax_wvl.yaxis.set_minor_locator(FixedLocator(np.arange(0.0, 2.1, 0.1)))
    ax_wvl.set_xlabel('Wavelength [nm]')
    if _ssfr_ == 'ssfr-a':
        ax_wvl.set_ylabel('Flux [$\mathrm{W m^{-2} nm^{-1}}$]')
    elif _ssfr_ == 'ssfr-b':
        ax_wvl.set_ylabel('Rad. [$\mathrm{W m^{-2} nm^{-1} sr^{-1}}$]')

    ax_wvl.set_ylim((0, 2))

    patches_legend = [
                      mpatches.Patch(color='blue', label='Zenith'), \
                      mpatches.Patch(color='red' , label='Nadir'), \
                     ]
    ax_wvl.legend(handles=patches_legend, loc='lower right', fontsize=12)

    ax_wvl0.set_ylim((0, count_ceil*2.0))
    ax_wvl0.ticklabel_format(axis='y', style='sci', scilimits=(0, 4), useMathText=True, useOffset=True)
    ax_wvl0.set_ylabel('Digital Counts', labelpad=18, rotation=270)
    #\----------------------------------------------------------------------------/#


    # temperature plot
    #/----------------------------------------------------------------------------\#
    temperatures = {
            0: {'name': 'Ambient T' , 'units':'$^\circ C$'},
            1: {'name': 'Power Vol' , 'units':'V'},
            2: {'name': 'Zen In T'  , 'units':'$^\circ C$'},
            3: {'name': 'Nad In T'  , 'units':'$^\circ C$'},
            4: {'name': 'RH'        , 'units':'%'},
            5: {'name': 'Zen In TEC', 'units':'$^\circ C$'},
            6: {'name': 'Nad In TEC', 'units':'$^\circ C$'},
            7: {'name': 'Wvl Con T' , 'units':'$^\circ C$'},
            8: {'name': 'cRIO T'    , 'units':'$^\circ C$'},
            9: {'name': 'Plate T'   , 'units':'$^\circ C$'},
           10: {'name': 'N/A'       , 'units':''},
            }
    temp = flt_trk0['temperature'][index_pnt, :].copy()
    temp[(temp<-100.0)|(temp>50.0)] = np.nan
    temp_x = np.arange(temp.size)
    width = 0.6
    temp_color='gray'
    logic_temp= ~np.isnan(temp)
    ax_temp0.bar(temp_x, temp, width=width, color=temp_color, lw=1.0, alpha=0.4, zorder=0, ec='gray')
    for i, x0 in enumerate(temp_x):
        ax_temp0.text(x0, 0.0, temperatures[i]['name'], fontsize=10, color=temp_color, ha='center', va='bottom')

    for i, x0 in enumerate(temp_x[logic_temp]):
        y0 = temp[logic_temp][i]
        ax_temp0.text(x0, y0, '%.1f%s' % (y0, temperatures[x0]['units']), fontsize=10, color='black', ha='center', va='center')

    for i, x0 in enumerate(temp_x[~logic_temp]):
        ax_temp0.text(x0, -10.0, '%.1f%s' % (flt_trk0['temperature'][index_pnt, x0], temperatures[x0]['units']), fontsize=10, color=temp_color, ha='center', va='center')

    ax_temp0.axhline(0.0, color=temp_color, lw=1.0, ls='-')
    ax_temp0.set_xlim(temp_x[0]-width/2.0, temp_x[-1]+width/2.0)
    ax_temp0.set_ylim((-100, 50))

    ax_temp0.tick_params(top=False, labeltop=False, left=False, labelleft=False, right=False, labelright=False, bottom=False, labelbottom=False)
    ax_temp0.axis('off')
    ax_temp0_.tick_params(top=False, labeltop=False, left=False, labelleft=False, right=False, labelright=False, bottom=False, labelbottom=False)
    ax_temp0_.axis('off')
    #\----------------------------------------------------------------------------/#


    # time series
    #/----------------------------------------------------------------------------\#
    for itrk in range(index_trk+1):

        flt_trk = flt_sim0.flt_trks[itrk]

        logic_solid = (flt_trk['tmhr']>=tmhr_past) & (flt_trk['tmhr']<=tmhr_current)
        logic_trans = np.logical_not(logic_solid)

        # logic_solid0 = (flt_trk['tmhr']>=(tmhr_current-0.5*tmhr_length)) & (flt_trk['tmhr']<=tmhr_current)
        # logic_trans0 = np.logical_not(logic_solid0)

        if itrk == index_trk:
            alpha_trans = 0.0
        else:
            alpha_trans = 0.08

        for key in axes_spec.keys():

            vname = '%s|raw' % key
            vname0 = '%s|dc' % key

            var_plot = vars_plot[vname]
            var_plot0 = vars_plot[vname0]

            if var_plot['plot?']:

                tms_y = flt_trk[var_plot['vname']][:, Nchan[key]]
                tms_y0 = flt_trk[var_plot['vname']][:, Nchan[key]] - flt_trk[var_plot0['vname']][:, Nchan[key]]

                ax_tms.scatter(flt_trk['tmhr'][logic_solid], tms_y[logic_solid], c=vars_plot[vname]['color'], s=15, lw=0.0, zorder=var_plot['zorder'], alpha=0.8)
                ax_tms.plot(flt_trk['tmhr'][logic_solid], tms_y0[logic_solid], lw=0.5, color=var_plot0['color'], zorder=10, alpha=1.0)

        ax_tms.vlines(flt_trk['tmhr'][logic_solid][flt_trk['shutter'][logic_solid]==1], ymin=count_base, ymax=count_ceil, color='black', alpha=0.3, lw=2.0, zorder=0)
        ax_tms.vlines(flt_trk['tmhr'][logic_solid][flt_trk['shutter_dc'][logic_solid]==-1], ymin=count_base, ymax=count_ceil, color='red', alpha=0.3, lw=2.0, zorder=0)

    ax_tms.axhline(count_base, color='red', alpha=0.2, zorder=0)
    ax_tms.axhline(count_ceil, color='red', alpha=0.2, zorder=0)


    ax_tms.set_xlim((tmhr_past-0.0000001, tmhr_current+0.0000001))
    ax_tms.xaxis.set_major_locator(FixedLocator([tmhr_past, tmhr_current-0.5*tmhr_length, tmhr_current]))
    ax_tms.xaxis.set_minor_locator(FixedLocator(np.arange(tmhr_past, tmhr_current+0.001, 5.0/60.0)))
    ax_tms.set_xticklabels(['%.4f' % (tmhr_past), '%.4f' % (tmhr_current-0.5*tmhr_length), '%.4f' % tmhr_current])
    ax_tms.set_xlabel('Time [hour]')

    ax_tms.set_ylim((count_base-5000, count_ceil+5000))
    ax_tms.set_ylabel('Digital Counts')
    ax_tms.ticklabel_format(axis='y', style='sci', scilimits=(0, 4), useMathText=True)
    #\----------------------------------------------------------------------------/#


    # figure settings
    #/----------------------------------------------------------------------------\#
    title_fig = '%s UTC (%s)' % (dtime_current.strftime('%Y-%m-%d %H:%M:%S'), _ssfr_.upper())
    fig.suptitle(title_fig, y=0.96, fontsize=20)
    #\----------------------------------------------------------------------------/#


    ax.axis('off')

    if test:
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
        wvl_step_spns=1,
        wvl_step_ssfr=1,
        ):

    # create data directory (for storing data) if the directory does not exist
    #/----------------------------------------------------------------------------\#
    date_s = date.strftime('%Y%m%d')

    fdir = os.path.abspath('%s/%s' % (_fdir_main_, date_s))
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    #\----------------------------------------------------------------------------/#

    fname_ssfr1_v0 = '%s/%s-%s_%s_%s_v0.h5' % (_fdir_data_, _mission_.upper(), _ssfr_.upper(), _platform_.upper(), date_s)
    fname_ssfr1_v1 = '%s/%s-%s_%s_%s_v1.h5' % (_fdir_data_, _mission_.upper(), _ssfr_.upper(), _platform_.upper(), date_s)

    # read data
    #/----------------------------------------------------------------------------\#
    try:
        data_ssfr_v1 = er3t.util.load_h5(fname_ssfr1_v1)
        has_v1 = True
    except Exception as error:
        print(error)
        has_v1 = False

    data_ssfr_v0 = er3t.util.load_h5(fname_ssfr1_v0)

    if has_v1:
        jday = data_ssfr_v1['v0/jday'][::time_step]
        zen_spec = data_ssfr_v1['v0/spec_zen'][::time_step, ::wvl_step_ssfr]
        nad_spec = data_ssfr_v1['v0/spec_nad'][::time_step, ::wvl_step_ssfr]
        zen_wvl  = data_ssfr_v1['v0/wvl_zen'][::wvl_step_ssfr]
        nad_wvl  = data_ssfr_v1['v0/wvl_nad'][::wvl_step_ssfr]
    else:
        jday = data_ssfr_v0['raw/jday'][::time_step]

    tmhr = (jday-int(jday[0]))*24.0
    sza  = np.zeros_like(jday)
    lon  = np.zeros_like(jday)
    lat  = np.zeros_like(jday)
    alt  = np.zeros_like(jday)

    zen_si_cnt_raw = data_ssfr_v0['raw/count_raw'][::time_step, ::wvl_step_ssfr, 0]
    zen_in_cnt_raw = data_ssfr_v0['raw/count_raw'][::time_step, ::wvl_step_ssfr, 1]
    nad_si_cnt_raw = data_ssfr_v0['raw/count_raw'][::time_step, ::wvl_step_ssfr, 2]
    nad_in_cnt_raw = data_ssfr_v0['raw/count_raw'][::time_step, ::wvl_step_ssfr, 3]

    zen_si_cnt_dc = data_ssfr_v0['raw/count_dark-corr'][::time_step, ::wvl_step_ssfr, 0]
    zen_in_cnt_dc = data_ssfr_v0['raw/count_dark-corr'][::time_step, ::wvl_step_ssfr, 1]
    nad_si_cnt_dc = data_ssfr_v0['raw/count_dark-corr'][::time_step, ::wvl_step_ssfr, 2]
    nad_in_cnt_dc = data_ssfr_v0['raw/count_dark-corr'][::time_step, ::wvl_step_ssfr, 3]

    zen_si_wvl = data_ssfr_v0['raw/wvl_zen_si'][::wvl_step_ssfr]
    zen_in_wvl = data_ssfr_v0['raw/wvl_zen_in'][::wvl_step_ssfr]
    nad_si_wvl = data_ssfr_v0['raw/wvl_nad_si'][::wvl_step_ssfr]
    nad_in_wvl = data_ssfr_v0['raw/wvl_nad_in'][::wvl_step_ssfr]


    int_time = data_ssfr_v0['raw/int_time'][::time_step, :]

    shutter    = data_ssfr_v0['raw/shutter'][::time_step]
    shutter_dc = data_ssfr_v0['raw/shutter_dark-corr'][::time_step]
    temperature = data_ssfr_v0['raw/temp'][::time_step, :]

    tmhr_interval = 10.0/60.0
    half_interval = tmhr_interval/48.0

    jday_s = jday[0]
    jday_e = jday[-1]

    jday_edges = np.arange(jday_s, jday_e+half_interval, half_interval*2.0)

    # create python dictionary to store valid flight data
    flt_trk = {}
    flt_trk['jday'] = jday
    flt_trk['tmhr'] = tmhr
    flt_trk['lon'] = lon
    flt_trk['lat'] = lat
    flt_trk['alt'] = alt
    flt_trk['sza'] = sza

    flt_trk['zen_si_cnt_raw'] = zen_si_cnt_raw
    flt_trk['zen_in_cnt_raw'] = zen_in_cnt_raw
    flt_trk['nad_si_cnt_raw'] = nad_si_cnt_raw
    flt_trk['nad_in_cnt_raw'] = nad_in_cnt_raw

    flt_trk['zen_si_cnt_dc']  = zen_si_cnt_dc
    flt_trk['zen_in_cnt_dc']  = zen_in_cnt_dc
    flt_trk['nad_si_cnt_dc']  = nad_si_cnt_dc
    flt_trk['nad_in_cnt_dc']  = nad_in_cnt_dc

    flt_trk['zen_si_wvl'] = zen_si_wvl
    flt_trk['zen_in_wvl'] = zen_in_wvl
    flt_trk['nad_si_wvl'] = nad_si_wvl
    flt_trk['nad_in_wvl'] = nad_in_wvl

    if has_v1:
        flt_trk['zen_spec'] = zen_spec
        flt_trk['nad_spec'] = nad_spec
        flt_trk['zen_wvl']  = zen_wvl
        flt_trk['nad_wvl']  = nad_wvl

    flt_trk['int_time'] = int_time
    flt_trk['shutter']  = shutter
    flt_trk['shutter_dc']  = shutter_dc
    flt_trk['temperature'] = temperature

    # partition the flight track into multiple mini flight track segments
    flt_trks = partition_flight_track(flt_trk, jday_edges, margin_x=1.0, margin_y=1.0)
    #\----------------------------------------------------------------------------/#


    # generate flt-sat combined file
    #/----------------------------------------------------------------------------\#
    fname = '%s/%s-%s-VID_%s_%s_v0.pk' % (_fdir_main_, _mission_.upper(), _ssfr_.upper(), _platform_.upper(), date_s)
    sim0 = flt_sim(
            date=date,
            wavelength=wvl0,
            flt_trks=flt_trks,
            flt_imgs=[[] for i in range(len(flt_trks))],
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

    fdir = _fdir_tmp_graph_
    if os.path.exists(fdir):
        os.system('rm -rf %s' % fdir)
    os.makedirs(fdir)

    fname = '%s/%s-%s-VID_%s_%s_v0.pk' % (_fdir_main_, _mission_.upper(), _ssfr_.upper(), _platform_.upper(), date_s)
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

    fname_mp4 = '%s-%s-VID_%s_%s.mp4' % (_mission_.upper(), _ssfr_.upper(), _platform_.upper(), date_s)
    with mp.Pool(processes=15) as pool:
        r = list(tqdm(pool.imap(plot_video_frame, statements), total=indices_trk.size))

    # make video
    fname_mp4 = '%s-%s-VID_%s_%s.mp4' % (_mission_.upper(), _ssfr_.upper(), _platform_.upper(), date_s)
    os.system('ffmpeg -y -framerate 10 -pattern_type glob -i "%s/*.png" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p %s' % (fdir, fname_mp4))


if __name__ == '__main__':

    dates = [
            datetime.datetime(2024, 5, 17), # ARCSIX test flight #1
            # datetime.datetime(2024, 5, 21), # ARCSIX test flight #2
        ]

    for date in dates[::-1]:

        # prepare flight data
        #/----------------------------------------------------------------------------\#
        # main_pre(date)
        #\----------------------------------------------------------------------------/#

        # generate video frames
        #/----------------------------------------------------------------------------\#
        # main_vid(date, wvl0=_wavelength_)
        #\----------------------------------------------------------------------------/#

        pass

    # sys.exit()

    # test
    #/----------------------------------------------------------------------------\#
    date = dates[-1]
    date_s = date.strftime('%Y%m%d')
    fname = '%s/%s-%s-VID_%s_%s_v0.pk' % (_fdir_main_, _mission_.upper(), _ssfr_.upper(), _platform_.upper(), date_s)
    flt_sim0 = flt_sim(fname=fname, overwrite=False)
    # statements = (flt_sim0, 0, 243, 1730)
    statements = (flt_sim0, 1, 443, 1730)
    plot_video_frame(statements, test=True)
    #\----------------------------------------------------------------------------/#

    pass
