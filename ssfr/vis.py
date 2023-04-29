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



__all__ = ['quicklook_ssfr_raw', 'quicklook_bokeh_ssfr', 'quicklook_bokeh_ssfr_and_spns']



def quicklook_ssfr_raw(
        data0,
        ichan=100,
        extra_tag='',
        plot_corr=False,
        ):

    filename = os.path.basename(data0['general_info']['fnames'][0])
    file_ext = filename.split('.')[-1]
    ssfr_tag = data0['general_info']['ssfr_tag']

    if ssfr_tag == 'NASA Ames SSFR':
        wvls  = ssfr.nasa_ssfr.get_ssfr_wavelength()
        y_range = [0, 2**5]
    elif ssfr_tag == 'CU LASP SSFR':
        wvls  = ssfr.lasp_ssfr.get_ssfr_wavelength()
        y_range  = [-2**5, 2**5]
    else:
        msg = '\nError [plot_ssfr_raw]: Cannot recognize SSFR system from given file extension <.%s>.' % file_ext
        raise OSError(msg)

    data0['spectra'] /= 2**10
    Nchan = data0['spectra'].shape[1]
    xx = np.arange(Nchan)

    # figure
    #/----------------------------------------------------------------------------\#
    dtime_s = ssfr.util.jday_to_dtime(data0['jday'][0])
    dtime_e = ssfr.util.jday_to_dtime(data0['jday'][-1])
    dtime_s_ = dtime_s.strftime('%Y-%m-%d')
    info = 'Quicklook Plot for <%s...> from %s on %s' % (filename, ssfr_tag, dtime_s_)

    logic_light = (data0['shutter'] == 0)
    logic_dark  = (data0['shutter'] == 1)
    Nlight = logic_light.sum()
    Ndark  = logic_dark.sum()

    if True:
        plt.close('all')
        fig = plt.figure(figsize=(20, 8))
        fig.suptitle(info)
        # plot
        #/--------------------------------------------------------------\#

        # Zenith Silicon
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(241)

        index = 0
        if Nlight > 0:
            yy  = np.nanmean(data0['spectra'][logic_light, :, index], axis=0)
            yy_ = np.nanstd(data0['spectra'][logic_light, :, index], axis=0)
            ax1.fill_between(xx, yy-yy_, yy+yy_, color='r', lw=0.0, alpha=0.3)
            ax1.plot(xx, yy, color='r', lw=1.0)

            vname = 'spectra_dark_corr'
            if vname in data0.keys() and plot_corr:
                logic_light_ = data0['shutter_dark_corr'] == 0
                yy  = np.nanmean(data0[vname][logic_light_, :, index], axis=0)
                yy_ = np.nanstd(data0[vname][logic_light_, :, index], axis=0)
                ax1.fill_between(xx, yy-yy_, yy+yy_, color='g', lw=0.0, alpha=0.3)
                ax1.plot(xx, yy, color='g', lw=1.0)

        if Ndark > 0:
            yy  = np.nanmean(data0['spectra'][logic_dark, :, index], axis=0)
            yy_ = np.nanstd(data0['spectra'][logic_dark, :, index], axis=0)
            ax1.fill_between(xx, yy-yy_, yy+yy_, color='b', lw=0.0, alpha=0.3)
            ax1.plot(xx, yy, color='b', lw=1.0)

        ax1.axvline(ichan, color='k', ls=':')
        ax1.grid()
        ax1.set_title('Zenith Silicon (%s)' % ','.join(['%dms' % t for t in np.unique(data0['int_time'][:, index])]), color='red')
        ax1.set_xlabel('Channel Number')
        ax1.set_ylabel('Counts [$\\times 2^{10}$]')
        ax1.set_xlim((-10, Nchan+10))
        ax1.set_ylim((y_range[0]-2, y_range[-1]+2))
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, Nchan+1, 64)))
        ax1.yaxis.set_major_locator(FixedLocator(np.arange(y_range[0], y_range[-1]+1, 16)))
        #\--------------------------------------------------------------/#


        # Zenith InGaAs
        #/--------------------------------------------------------------\#
        ax2 = fig.add_subplot(242)

        index = 1
        if Nlight > 0:
            yy  = np.nanmean(data0['spectra'][logic_light, :, index], axis=0)
            yy_ = np.nanstd(data0['spectra'][logic_light, :, index], axis=0)
            ax2.fill_between(xx, yy-yy_, yy+yy_, color='r', lw=0.0, alpha=0.3)
            ax2.plot(xx, yy, color='r', lw=1.0)

            vname = 'spectra_dark_corr'
            if vname in data0.keys() and plot_corr:
                logic_light_ = data0['shutter_dark_corr'] == 0
                yy  = np.nanmean(data0[vname][logic_light_, :, index], axis=0)
                yy_ = np.nanstd(data0[vname][logic_light_, :, index], axis=0)
                ax2.fill_between(xx, yy-yy_, yy+yy_, color='g', lw=0.0, alpha=0.3)
                ax2.plot(xx, yy, color='g', lw=1.0)

        if Ndark > 0:
            yy  = np.nanmean(data0['spectra'][logic_dark, :, index], axis=0)
            yy_ = np.nanstd(data0['spectra'][logic_dark, :, index], axis=0)
            ax2.fill_between(xx, yy-yy_, yy+yy_, color='b', lw=0.0, alpha=0.3)
            ax2.plot(xx, yy, color='b', lw=1.0)

        ax2.axvline(ichan, color='k', ls=':')
        ax2.grid()
        ax2.set_title('Zenith InGaAs (%s)' % ','.join(['%dms' % t for t in np.unique(data0['int_time'][:, index])]), color='blue')
        ax2.set_xlabel('Channel Number')
        ax2.set_xlim((-10, Nchan+10))
        ax2.set_ylim((y_range[0]-2, y_range[-1]+2))
        ax2.xaxis.set_major_locator(FixedLocator(np.arange(0, Nchan+1, 64)))
        ax2.yaxis.set_major_locator(FixedLocator(np.arange(y_range[0], y_range[-1]+1, 16)))
        #\--------------------------------------------------------------/#

        # Nadir Silicon
        #/--------------------------------------------------------------\#
        ax3 = fig.add_subplot(243)

        index = 2
        if Nlight > 0:
            yy  = np.nanmean(data0['spectra'][logic_light, :, index], axis=0)
            yy_ = np.nanstd(data0['spectra'][logic_light, :, index], axis=0)
            ax3.fill_between(xx, yy-yy_, yy+yy_, color='r', lw=0.0, alpha=0.3)
            ax3.plot(xx, yy, color='r', lw=1.0)

            vname = 'spectra_dark_corr'
            if vname in data0.keys() and plot_corr:
                logic_light_ = data0['shutter_dark_corr'] == 0
                yy  = np.nanmean(data0[vname][logic_light_, :, index], axis=0)
                yy_ = np.nanstd(data0[vname][logic_light_, :, index], axis=0)
                ax3.fill_between(xx, yy-yy_, yy+yy_, color='g', lw=0.0, alpha=0.3)
                ax3.plot(xx, yy, color='g', lw=1.0)

        if Ndark > 0:
            yy  = np.nanmean(data0['spectra'][logic_dark, :, index], axis=0)
            yy_ = np.nanstd(data0['spectra'][logic_dark, :, index], axis=0)
            ax3.fill_between(xx, yy-yy_, yy+yy_, color='b', lw=0.0, alpha=0.3)
            ax3.plot(xx, yy, color='b', lw=1.0)

        ax3.axvline(ichan, color='k', ls=':')
        ax3.grid()
        ax3.set_title('Nadir Silicon (%s)' % ','.join(['%dms' % t for t in np.unique(data0['int_time'][:, index])]), color='magenta')
        ax3.set_xlabel('Channel Number')
        ax3.set_xlim((-10, Nchan+10))
        ax3.set_ylim((y_range[0]-2, y_range[-1]+2))
        ax3.xaxis.set_major_locator(FixedLocator(np.arange(0, Nchan+1, 64)))
        ax3.yaxis.set_major_locator(FixedLocator(np.arange(y_range[0], y_range[-1]+1, 16)))
        #\--------------------------------------------------------------/#


        # Nadir InGaAs
        #/--------------------------------------------------------------\#
        ax4 = fig.add_subplot(244)

        index = 3
        if Nlight > 0:
            yy  = np.nanmean(data0['spectra'][logic_light, :, index], axis=0)
            yy_ = np.nanstd(data0['spectra'][logic_light, :, index], axis=0)
            ax4.fill_between(xx, yy-yy_, yy+yy_, color='r', lw=0.0, alpha=0.3)
            ax4.plot(xx, yy, color='r', lw=1.0)

            vname = 'spectra_dark_corr'
            if vname in data0.keys() and plot_corr:
                logic_light_ = data0['shutter_dark_corr'] == 0
                yy  = np.nanmean(data0[vname][logic_light_, :, index], axis=0)
                yy_ = np.nanstd(data0[vname][logic_light_, :, index], axis=0)
                ax4.fill_between(xx, yy-yy_, yy+yy_, color='g', lw=0.0, alpha=0.3)
                ax4.plot(xx, yy, color='g', lw=1.0)

        if Ndark > 0:
            yy  = np.nanmean(data0['spectra'][logic_dark, :, index], axis=0)
            yy_ = np.nanstd(data0['spectra'][logic_dark, :, index], axis=0)
            ax4.fill_between(xx, yy-yy_, yy+yy_, color='b', lw=0.0, alpha=0.3)
            ax4.plot(xx, yy, color='b', lw=1.0)

        ax4.axvline(ichan, color='k', ls=':')
        ax4.grid()
        ax4.set_title('Nadir InGaAs (%s)' % ','.join(['%dms' % t for t in np.unique(data0['int_time'][:, index])]), color='cyan')
        ax4.set_xlabel('Channel Number')
        ax4.set_xlim((-10, Nchan+10))
        ax4.set_ylim((y_range[0]-2, y_range[-1]+2))
        ax4.xaxis.set_major_locator(FixedLocator(np.arange(0, Nchan+1, 64)))
        ax4.yaxis.set_major_locator(FixedLocator(np.arange(y_range[0], y_range[-1]+1, 16)))
        #\--------------------------------------------------------------/#


        # time series
        #/--------------------------------------------------------------\#
        ax5 = fig.add_subplot(212)
        ax5.plot(data0['jday'], data0['spectra'][:, ichan, 0], color='red')
        ax5.plot(data0['jday'], data0['spectra'][:, ichan, 1], color='blue')
        ax5.plot(data0['jday'], data0['spectra'][:, ichan, 2], color='magenta')
        ax5.plot(data0['jday'], data0['spectra'][:, ichan, 3], color='cyan')

        xticks = data0['jday'][::10]
        xticklabels = [ssfr.util.jday_to_dtime(jday0).strftime('%H:%M:%S') for jday0 in xticks]
        ax5.set_xticks(xticks)
        ax5.set_xticklabels(xticklabels, ha='right', rotation=30)
        ax5.set_title('Time Series at Channel #%d' % ichan)
        ax5.set_xlabel('UTC Time')
        ax5.set_ylabel('Counts [$\\times 2^{10}$]')
        ax5.set_ylim((y_range[0]-2, y_range[-1]+2))
        ax5.yaxis.set_major_locator(FixedLocator(np.arange(y_range[0], y_range[-1]+1, 16)))
        ax5.grid()

        if Ndark > 0:
            yy_min = np.repeat(y_range[0]-2.0, data0['jday'].size)
            yy_min[logic_light] = np.nan
            yy_max = np.repeat(y_range[-1]+2.0, data0['jday'].size)
            yy_max[logic_light] = np.nan
            ax5.fill_between(data0['jday'], yy_min, yy_max, color='k', lw=0.0, alpha=0.2, zorder=0)

        patches_legend = [
                         mpatches.Patch(color='red'    , label='Zenith Silicon'), \
                         mpatches.Patch(color='blue'   , label='Zenith InGaAs'), \
                         mpatches.Patch(color='magenta', label='Nadir Silicon'), \
                         mpatches.Patch(color='cyan'   , label='Nadir InGaAs'), \
                         ]
        ax5.legend(handles=patches_legend, loc='upper left', fontsize=12)
        #\--------------------------------------------------------------/#


        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s%s' % (extra_tag, filename.replace(file_ext, 'png')), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#





def lonlat_to_xy(lon, lat):

    import cartopy.crs as ccrs

    proj0 = ccrs.PlateCarree()
    proj1 = ccrs.Mercator.GOOGLE
    xy = proj1.transform_points(proj0, lon, lat)

    x = xy[:, 0]
    y = xy[:, 1]

    return x, y

def pre_bokeh(fname, fname_spns=None, tmhr0=None, wvl0=None, tmhr_range=None, tmhr_step=1, wvl_step=1, wvl_step_spns=4):

    data0 = ssfr.util.load_h5(fname)

    if tmhr_range is None:
        logic_tmhr = np.repeat(True, data0['tmhr'].size)
    else:
        logic_tmhr = (data0['tmhr']>=tmhr_range[0]) & (data0['tmhr']<=tmhr_range[1])

    data = {}
    data['tmhr']     = data0['tmhr'][logic_tmhr][::tmhr_step]
    data['jday']     = data0['jday'][logic_tmhr][::tmhr_step] - ssfr.util.dtime_to_jday(datetime.datetime(1969, 12, 31))
    data['lon']      = data0['lon'][logic_tmhr][::tmhr_step]
    data['lat']      = data0['lat'][logic_tmhr][::tmhr_step]
    data['alt']      = data0['alt'][logic_tmhr][::tmhr_step]/1000.0
    data['sza']      = data0['sza'][logic_tmhr][::tmhr_step]
    data['zen_wvl']  = data0['zen_wvl'][::wvl_step]
    data['nad_wvl']  = data0['nad_wvl'][::wvl_step]
    data['zen_flux'] = data0['zen_flux'][logic_tmhr, :][::tmhr_step, ::wvl_step]
    data['nad_flux'] = data0['nad_flux'][logic_tmhr, :][::tmhr_step, ::wvl_step]

    if 'solar' in data0.keys():
        has_solar = True
        data['solar']    = data0['solar'][::wvl_step]
        data['mu']       = np.cos(np.deg2rad(data['sza']))
    else:
        has_solar = False


    # prepare data for time series
    # ===================================================================================
    data_time = {}

    data_time['tmhr'] = data['tmhr']
    data_time['jday'] = data['jday']

    if wvl0 is None:
        wvl0 = 500.0

    index_zen = np.argmin(np.abs(data['zen_wvl']-wvl0))
    data_time['zen_plot'] = data['zen_flux'][:, index_zen]

    index_nad = np.argmin(np.abs(data['nad_wvl']-wvl0))
    data_time['nad_plot'] = data['nad_flux'][:, index_nad]

    for i in range(data['zen_wvl'].size):
        data_time['zen%d' % i] = data['zen_flux'][:, i]
    for i in range(data['nad_wvl'].size):
        data_time['nad%d' % i] = data['nad_flux'][:, i]

    if has_solar:
        data_time['mu'] = data['mu']
        data_time['sol_plot'] = data['solar'][index_zen] * data['mu']

    data_time['alt']  = data['alt'] / (np.nanmax(data['alt'])-np.nanmin(data['alt']))\
            * (np.nanmax(data['zen_flux'][:, index_zen])-np.nanmin(data['zen_flux'][:, index_zen]))
    # ===================================================================================


    # prepare data for spectra
    # ===================================================================================
    data_spec = {}

    data_spec['zen_wvl']  = data['zen_wvl']
    data_spec['nad_wvl']  = data['nad_wvl']

    if tmhr0 is None:
        index_tmhr = np.where(np.logical_not(np.isnan(data['zen_flux'][:, index_zen])))[0][0]
    else:
        index_tmhr = np.argmin(np.abs(data['tmhr']-tmhr0))

    data_spec['zen_plot'] = data['zen_flux'][index_tmhr, :]
    data_spec['nad_plot'] = data['zen_flux'][index_tmhr, :]

    for i in range(data['tmhr'].size):
        data_spec['zen%d' % i] = data['zen_flux'][i, :]
        data_spec['nad%d' % i] = data['nad_flux'][i, :]

    if has_solar:
        data_spec['solar']    = data['solar']
        data_spec['sol_plot'] = data['solar'] * data['mu'][index_tmhr]
    # ===================================================================================


    # prepare data for geo map
    # ===================================================================================
    data_geo = {}

    data_geo['lon']  = data['lon']
    data_geo['lat']  = data['lat']
    data_geo['alt']  = data['alt']
    data_geo['sza']  = data['sza']

    x, y = lonlat_to_xy(data['lon'], data['lat'])
    data_geo['x'] = x
    data_geo['y'] = y
    # ===================================================================================

    # prepare data for vertical profile
    # ===================================================================================
    data_prof = {}

    time_window_half = 900//tmhr_step # 15 minute window

    zen_prof = np.zeros(time_window_half*2, dtype=np.float64); zen_prof[...] = np.nan
    nad_prof = np.zeros(time_window_half*2, dtype=np.float64); nad_prof[...] = np.nan
    alt_prof = np.zeros(time_window_half*2, dtype=np.float64); alt_prof[...] = np.nan
    tmhr_prof = np.zeros(time_window_half*2, dtype=np.float64); tmhr_prof[...] = np.nan

    index_s = max([0, index_tmhr-time_window_half])
    index_e = min([data['tmhr'].size, index_tmhr+time_window_half])

    zen_prof[:index_e-index_s] = data_time['zen_plot'][index_s:index_e]
    nad_prof[:index_e-index_s] = data_time['nad_plot'][index_s:index_e]
    alt_prof[:index_e-index_s] = data_geo['alt'][index_s:index_e]
    tmhr_prof[:index_e-index_s] = data_time['tmhr'][index_s:index_e]

    data_prof['zen_plot'] = zen_prof
    data_prof['nad_plot'] = nad_prof
    data_prof['alt_plot'] = alt_prof
    data_prof['tmhr'] = tmhr_prof
    # ===================================================================================



    # prepare data for spns
    # ===================================================================================
    if fname_spns is not None:
        data1 = ssfr.util.load_h5(fname_spns)
        data['spns_wvl'] = data1['wvl'][::wvl_step_spns]
        data['dif_flux'] = data1['dif_flux'][logic_tmhr, :][::tmhr_step, ::wvl_step_spns]
        data['tot_flux'] = data1['tot_flux'][logic_tmhr, :][::tmhr_step, ::wvl_step_spns]

        index = np.argmin(np.abs(data['spns_wvl']-wvl0))
        data_time['dif_plot'] = data['dif_flux'][:, index]
        data_time['tot_plot'] = data['tot_flux'][:, index]
        for i in range(data['spns_wvl'].size):
            data_time['dif%d' % i] = data['dif_flux'][:, i]
            data_time['tot%d' % i] = data['tot_flux'][:, i]

        data_spns = {}
        data_spns['wvl']  = data['spns_wvl']
        data_spns['dif_plot'] = data['dif_flux'][index_tmhr, :]
        data_spns['tot_plot'] = data['tot_flux'][index_tmhr, :]
        for i in range(tmhr.size):
            data_spns['dif%d' % i] = data['dif_flux'][i, :]
            data_spns['tot%d' % i] = data['tot_flux'][i, :]
    # ===================================================================================


    if fname_spns is not None:
        return data_time, data_spec, data_spns, data_geo
    else:
        return data_time, data_spec, data_geo, data_prof

def quicklook_bokeh_ssfr(fname_ssfr, wvl0=None, tmhr0=None, tmhr_range=None, wvl_range=[300.0, 2200.0], tmhr_step=10, wvl_step=2, description=None, fname_html=None):

    from bokeh.layouts import layout, gridplot
    from bokeh.models import ColumnDataSource, ColorBar
    from bokeh.models.widgets import Select, Slider, CheckboxGroup
    from bokeh.models import Toggle, CustomJS, Legend, Span, HoverTool
    from bokeh.plotting import figure, output_file, save
    from bokeh.transform import linear_cmap
    from bokeh.palettes import RdYlBu6, Spectral6
    from bokeh.tile_providers import get_provider, Vendors

    # prepare data
    # ========================================================================================================
    data_time_dict, data_spec_dict, data_geo_dict, data_prof_dict = pre_bokeh(fname_ssfr, fname_spns=None, tmhr_range=tmhr_range, tmhr_step=tmhr_step, wvl_step=wvl_step, tmhr0=tmhr0, wvl0=wvl0)

    if wvl0 is None:
        wvl0 = 500.0
    index_zen_wvl = np.argmin(np.abs(data_spec_dict['zen_wvl']-wvl0))
    index_nad_wvl = np.argmin(np.abs(data_spec_dict['nad_wvl']-wvl0))

    if tmhr0 is None:
        index_tmhr = np.where(np.logical_not(np.isnan(data_time_dict['zen%d' % np.argmin(np.abs(data_spec_dict['zen_wvl']-wvl0))])))[0][0]
    else:
        index_tmhr = np.argmin(np.abs(data_time_dict['tmhr']-tmhr0))

    data_geo0  = ColumnDataSource(data={'x':[data_geo_dict['x'][index_tmhr]],
                                        'y':[data_geo_dict['y'][index_tmhr]],
                                        'lon':[data_geo_dict['lon'][index_tmhr]],
                                        'lat':[data_geo_dict['lat'][index_tmhr]],
                                        'alt':[data_geo_dict['alt'][index_tmhr]]})
    data_geo1  = ColumnDataSource(data=data_geo_dict)
    data_geo   = ColumnDataSource(data=data_geo_dict)

    data_prof  = ColumnDataSource(data=data_prof_dict)

    data_spec  = ColumnDataSource(data=data_spec_dict)
    data_time  = ColumnDataSource(data=data_time_dict)
    # ========================================================================================================

    # bokeh plot specifications
    # ========================================================================================================
    if description is not None:
        title = 'SSFR Quicklook Plot (%s)' % description
    else:
        title = 'SSFR Quicklook Plot'

    if fname_html is None:
        fname_html = 'ssfr-bokeh-plot_%s.html' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    output_file(fname_html, title=title, mode='inline')

    height_geo = 400
    width_geo  = 440
    height_spec = height_geo
    width_spec  = 650
    height_prof = height_geo
    width_prof  = 250
    height_time = 320
    width_time  = 10 + width_geo + width_spec + width_prof
    # ========================================================================================================

    # map plot
    # ========================================================================================================
    plt_geo  = figure(plot_height=height_geo, plot_width=width_geo,
                  title='Aircraft at %.4f km' % data_geo.data['alt'][index_tmhr],
                  tools='reset,save,box_zoom,wheel_zoom,pan',
                  active_scroll='wheel_zoom', active_drag='pan',
                  x_axis_label='Longitude', y_axis_label='Latitude',
                  x_axis_type="mercator", y_axis_type="mercator",
                  x_range=[data_geo.data['x'][index_tmhr]-50000.0, data_geo.data['x'][index_tmhr]+50000.0],
                  y_range=[data_geo.data['y'][index_tmhr]-50000.0, data_geo.data['y'][index_tmhr]+50000.0], output_backend='webgl')

    tile_provider = get_provider(Vendors.CARTODBPOSITRON)
    plt_geo.add_tile(tile_provider)

    htool = HoverTool(tooltips = [('Longitude', '@lon{0.0000}'), ('Latitude', '@lat{0.0000}'), ('Altitude', '@alt{0.0000}km'), ('Solar Zenith', '@sza{0.00}')], mode='mouse', line_policy='nearest')
    plt_geo.add_tools(htool)

    mapper = linear_cmap(field_name='alt', palette=Spectral6, low=0.0, high=6.0)

    plt_geo.circle('x', 'y', source=data_geo , color=mapper, size=6, fill_alpha=0.1, line_width=0.0)
    plt_geo.circle('x', 'y', source=data_geo0, color=mapper, line_color=mapper, size=15, fill_alpha=0.5, line_width=2.0)
    plt_geo.circle('x', 'y', source=data_geo1, color=mapper, size=4, fill_alpha=1.0, line_width=0.0)

    color_bar = ColorBar(color_mapper=mapper['transform'], width=10, location=(0,0))
    plt_geo.add_layout(color_bar, 'right')

    plt_geo.title.text_font_size = '1.3em'
    plt_geo.title.align = 'center'
    plt_geo.xaxis.axis_label_text_font_style = 'normal'
    plt_geo.yaxis.axis_label_text_font_style = 'normal'
    plt_geo.xaxis.axis_label_text_font_size  = '1.0em'
    plt_geo.xaxis.major_label_text_font_size = '1.0em'
    plt_geo.yaxis.axis_label_text_font_size  = '1.0em'
    plt_geo.yaxis.major_label_text_font_size = '1.0em'
    # ========================================================================================================

    # vertical profile plot
    # ========================================================================================================
    xrange_e = np.nanmax(data_prof.data['zen_plot'])*1.1
    yrange_s = np.nanmin(data_prof.data['alt_plot'])*0.9
    yrange_e = np.nanmax(data_prof.data['alt_plot'])*1.1

    plt_prof = figure(height=height_prof, width=width_prof,
                  title='Profile [%.4f, %.4f]' % (np.nanmin(data_prof.data['tmhr']), np.nanmax(data_prof.data['tmhr'])),
                  tools='reset,save,box_zoom,wheel_zoom,pan',
                  active_scroll='wheel_zoom', active_drag='pan', x_axis_label='Flux Density', y_axis_label='Altitude',
                  x_range=[0.0     , xrange_e],
                  y_range=[yrange_s, yrange_e], output_backend='webgl')

    htool = HoverTool(tooltips = [('Flux', '$x{0.0000}'), ('Altitude', '$y{0.0000}km')], mode='mouse', line_policy='nearest')
    plt_prof.add_tools(htool)

    plt_prof.circle('zen_plot', 'alt_plot', source=data_prof, color='dodgerblue', size=3, legend='Zenith')
    plt_prof.circle('nad_plot', 'alt_plot', source=data_prof, color='lightcoral', size=3, legend='Nadir')

    plt_prof.legend.location = 'top_right'
    plt_prof.legend.click_policy  = 'hide'
    plt_prof.title.text_font_size = '1.3em'
    plt_prof.title.align          = 'center'
    plt_prof.xaxis.axis_label_text_font_style = 'normal'
    plt_prof.yaxis.axis_label_text_font_style = 'normal'
    plt_prof.xaxis.axis_label_text_font_size  = '1.0em'
    plt_prof.xaxis.major_label_text_font_size = '1.0em'
    plt_prof.yaxis.axis_label_text_font_size  = '1.0em'
    plt_prof.yaxis.major_label_text_font_size = '1.0em'
    # ========================================================================================================

    # spectra plot
    # ========================================================================================================
    plt_spec = figure(plot_height=height_spec, plot_width=width_spec,
                  title='Spectra at %s UTC' % ((datetime.datetime(1970, 1, 1)+datetime.timedelta(days=data_time.data['jday'][index_tmhr]-1.0)).strftime('%Y-%m-%d %H:%M:%S')),
                  tools='reset,save,box_zoom,wheel_zoom,pan',
                  active_scroll='wheel_zoom', active_drag='pan', x_axis_label='Wavelength [nm]', y_axis_label='Flux Density',
                  x_range=wvl_range, y_range=[0.0, np.nanmax(data_spec.data['zen_plot'])*1.1], output_backend='webgl')

    htool = HoverTool(tooltips = [('Wavelength', '$x{0.00}nm'), ('Flux', '$y{0.0000}')], mode='mouse', line_policy='nearest')
    plt_spec.add_tools(htool)

    plt_spec.circle('zen_wvl', 'sol_plot', source=data_spec, color='black'     , size=3, legend='Kurudz', fill_alpha=0.4, line_alpha=0.4)
    plt_spec.circle('zen_wvl', 'zen_plot', source=data_spec, color='dodgerblue', size=3, legend='Zenith')
    plt_spec.circle('nad_wvl', 'nad_plot', source=data_spec, color='lightcoral', size=3, legend='Nadir')


    slider_spec = Slider(start=wvl_range[0], end=wvl_range[1], value=wvl0, step=0.01, width=width_spec, title='Wavelength [nm]', format='0[.]00')
    vline_spec  = Span(location=slider_spec.value, dimension='height', line_color='black', line_dash='dashed', line_width=1)
    plt_spec.add_layout(vline_spec)

    plt_spec.legend.click_policy  = 'hide'
    plt_spec.title.text_font_size = '1.3em'
    plt_spec.title.align     = 'center'
    plt_spec.legend.location = 'top_right'
    plt_spec.xaxis.axis_label_text_font_style = 'normal'
    plt_spec.yaxis.axis_label_text_font_style = 'normal'
    plt_spec.xaxis.axis_label_text_font_size  = '1.0em'
    plt_spec.xaxis.major_label_text_font_size = '1.0em'
    plt_spec.yaxis.axis_label_text_font_size  = '1.0em'
    plt_spec.yaxis.major_label_text_font_size = '1.0em'
    # ========================================================================================================


    # time series plot
    # ========================================================================================================
    xrange_s = np.nanmin(data_time.data['tmhr'])-0.1
    xrange_e = np.nanmax(data_time.data['tmhr'])+0.1
    yrange_e = np.nanmax(data_time.data['zen_plot'])*1.1

    plt_time = figure(height=height_time, width=width_time,
                  title='Time Series at %.2f nm (Zenith) and %.2f nm (Nadir)' % (data_spec.data['zen_wvl'][index_zen_wvl], data_spec.data['nad_wvl'][index_nad_wvl]),
                  tools='reset,save,box_zoom,wheel_zoom,pan',
                  active_scroll='wheel_zoom', active_drag='pan', x_axis_label='Time [Hour]', y_axis_label='Flux Density',
                  x_range=[xrange_s, xrange_e],
                  y_range=[0.0     , yrange_e], output_backend='webgl')

    htool = HoverTool(tooltips = [('Time', '$x{0.0000}'), ('Flux', '$y{0.0000}')], mode='mouse', line_policy='nearest')
    plt_time.add_tools(htool)

    plt_time.circle('tmhr', 'sol_plot', source=data_time, color='black'     , size=3, legend='Kurudz', fill_alpha=0.4, line_alpha=0.4)
    plt_time.circle('tmhr', 'zen_plot', source=data_time, color='dodgerblue', size=3, legend='Zenith')
    plt_time.circle('tmhr', 'nad_plot', source=data_time, color='lightcoral', size=3, legend='Nadir')
    plt_time.varea('tmhr', 0.0, 'alt' , source=data_time, fill_color='darkgreen' , alpha=0.15, level='underlay')
    plt_time.legend.location = 'top_right'

    slider_time = Slider(start=xrange_s, end=xrange_e, value=data_time.data['tmhr'][index_tmhr], step=0.0001, width=width_time, title='Time [Hour]', format='0[.]0000')
    vline_time  = Span(location=slider_time.value, dimension='height', line_color='black', line_dash='dashed', line_width=1)
    plt_time.add_layout(vline_time)

    plt_time.legend.click_policy  = 'hide'
    plt_time.title.text_font_size = '1.3em'
    plt_time.title.align          = 'center'
    plt_time.xaxis.axis_label_text_font_style = 'normal'
    plt_time.yaxis.axis_label_text_font_style = 'normal'
    plt_time.xaxis.axis_label_text_font_size  = '1.0em'
    plt_time.xaxis.major_label_text_font_size = '1.0em'
    plt_time.yaxis.axis_label_text_font_size  = '1.0em'
    plt_time.yaxis.major_label_text_font_size = '1.0em'
    # ========================================================================================================


    slider_spec.callback = CustomJS(args=dict(
                             plt_t    = plt_time,
                             plt_p    = plt_prof,
                             span_s   = vline_spec,
                             slider_s = slider_spec,
                             slider_t = slider_time,
                             src_t    = data_time,
                             src_s    = data_spec,
                             src_p    = data_prof,
                             ), code="""
function closest (num, arr) {
    var curr = 0;
    var diff = Math.abs (num - arr[curr]);
    for (var val = 0; val < arr.length; val++) {
        var newdiff = Math.abs (num - arr[val]);
        if (newdiff < diff) {
            diff = newdiff;
            curr = val;
        }
    }
    return curr;
}

function nanmax (arr) {
    var max0 = 0.0;
    for (i=0; i<arr.length; i++) {
        if (arr[i] != Number.NaN) {
            if (arr[i] > max0){
                max0 = arr[i];
            }
        }
    }
    return max0;
}

function nanmin (arr) {
    var min0 = 100.0;
    for (i=0; i<arr.length; i++) {
        if (arr[i] != Number.NaN) {
            if (arr[i] < min0){
                min0 = arr[i];
            }
        }
    }
    return min0;
}


var x  = src_t.data['tmhr'];

var zen_wvl     = src_s.data['zen_wvl'];
var zen_index   = closest(slider_s.value, zen_wvl);
var zen_index_s = zen_index.toString();
var v1 = 'zen' + zen_index_s;

var nad_wvl     = src_s.data['nad_wvl'];
var nad_index   = closest(slider_s.value, nad_wvl);
var nad_index_s = nad_index.toString();
var v2 = 'nad' + nad_index_s;

var zen_min0 = nanmin(src_t.data['zen_plot']);
var zen_max0 = nanmax(src_t.data['zen_plot']);
var zen_min1 = nanmin(src_t.data[v1]);
var zen_max1 = nanmax(src_t.data[v1]);
var alt_fac  = (zen_max1-zen_min1)/(zen_max0-zen_min0);

for (i = 0; i < x.length; i++) {
    src_t.data['zen_plot'][i] = src_t.data[v1][i];
    src_t.data['nad_plot'][i] = src_t.data[v2][i];
    src_t.data['sol_plot'][i] = src_s.data['solar'][zen_index] * src_t.data['mu'][i];
    src_t.data['alt'][i] = src_t.data['alt'][i] * alt_fac;
}
src_t.change.emit();

var title = 'Time Series at ' + zen_wvl[zen_index].toFixed(2).toString() + ' nm (Zenith) and ' + nad_wvl[nad_index].toFixed(2).toString() + ' nm (Nadir)' ;
plt_t.title.text = title;

if (zen_max0 > 0.0) {
plt_t.y_range.end = zen_max0*1.1;
}


var x  = src_t.data['tmhr'];
var index = closest(slider_t.value, x);

var N_time_window_half = src_p.data['alt_plot'].length/2;

for (i = 0; i < N_time_window_half*2; i++) {
    src_p.data['zen_plot'][i] = Number.NaN;
    src_p.data['nad_plot'][i] = Number.NaN;
}

if (index-N_time_window_half<0) {
var index_p_s = 0;
} else {
var index_p_s = index - N_time_window_half;
}

if (index+N_time_window_half>x.length) {
var index_p_e = x.length;
} else {
var index_p_e = index + N_time_window_half;
}

var icount = 0;
for (i = index_p_s; i < index_p_e; i++) {
    src_p.data['zen_plot'][icount] = src_t.data['zen_plot'][i];
    src_p.data['nad_plot'][icount] = src_t.data['nad_plot'][i];
    icount += 1;
}
src_p.change.emit();

plt_p.x_range.end   = nanmax(src_p.data['zen_plot'])*1.1;



span_s.location = slider_s.value;
    """)


    slider_time.callback = CustomJS(args=dict(
                             plt_s    = plt_spec,
                             plt_g    = plt_geo,
                             plt_p    = plt_prof,
                             span_t   = vline_time,
                             slider_t = slider_time,
                             src_t  = data_time,
                             src_s  = data_spec,
                             src_g  = data_geo,
                             src_g0 = data_geo0,
                             src_g1 = data_geo1,
                             src_p  = data_prof,
                             ), code="""
function closest (num, arr) {
    var curr = 0;
    var diff = Math.abs (num - arr[curr]);
    for (var val = 0; val < arr.length; val++) {
        var newdiff = Math.abs (num - arr[val]);
        if (newdiff < diff) {
            diff = newdiff;
            curr = val;
        }
    }
    return curr;
}

function nanmax (arr) {
    var max0 = 0.0;
    for (i=0; i<arr.length; i++) {
        if (arr[i] != Number.NaN) {
            if (arr[i] > max0){
                max0 = arr[i];
            }
        }
    }
    return max0;
}

function nanmin (arr) {
    var min0 = 100.0;
    for (i=0; i<arr.length; i++) {
        if (arr[i] != Number.NaN) {
            if (arr[i] < min0){
                min0 = arr[i];
            }
        }
    }
    return min0;
}

var x  = src_t.data['tmhr'];

var index = closest(slider_t.value, x);
var index_s = index.toString();
var v1 = 'zen' + index_s;
var v2 = 'nad' + index_s;

var max0 = nanmax(src_s.data[v1]);

for (i = 0; i < src_s.data['zen_wvl'].length; i++) {
    src_s.data['zen_plot'][i] = src_s.data[v1][i];
    src_s.data['nad_plot'][i] = src_s.data[v2][i];
    src_s.data['sol_plot'][i] = src_s.data['solar'][i] * src_t.data['mu'][index];
}
src_s.change.emit();

var msec0 = (src_t.data['jday'][index]-1.0)*86400000.0;
var date0 = new Date(msec0);

var month0 = date0.getUTCMonth() + 1
date_s = date0.getUTCFullYear() + '-'
        + ('0' + month0).slice(-2) + '-'
        + ('0' + date0.getUTCDate()).slice(-2) + ' '
        + ('0' + date0.getUTCHours()).slice(-2) + ':'
        + ('0' + date0.getUTCMinutes()).slice(-2) + ':'
        + ('0' + date0.getUTCSeconds()).slice(-2);

var title1 = 'Spectra at ' + date_s + ' UTC';
plt_s.title.text = title1;

if (max0 > 0.0) {
plt_s.y_range.end = max0*1.1;
}

var title2 = 'Aircraft at ' + src_g.data['alt'][index].toFixed(4).toString() + ' km';
plt_g.title.text = title2;

plt_g.x_range.start = src_g.data['x'][index]-50000.0;
plt_g.x_range.end   = src_g.data['x'][index]+50000.0;
plt_g.y_range.start = src_g.data['y'][index]-50000.0;
plt_g.y_range.end   = src_g.data['y'][index]+50000.0;

src_g0.data['x'][0]   = src_g.data['x'][index];
src_g0.data['y'][0]   = src_g.data['y'][index];
src_g0.data['alt'][0] = src_g.data['alt'][index];
src_g0.change.emit();

for (i = 0; i < src_g.data['x'].length; i++) {
    if (i<=index) {
    src_g1.data['x'][i] = src_g.data['x'][i];
    src_g1.data['y'][i] = src_g.data['y'][i];
    } else {
    src_g1.data['x'][i] = Number.NaN;
    src_g1.data['y'][i] = Number.NaN;
    }
}
src_g1.change.emit();



var N_time_window_half = src_p.data['alt_plot'].length/2;

for (i = 0; i < N_time_window_half*2; i++) {
    src_p.data['zen_plot'][i] = Number.NaN;
    src_p.data['nad_plot'][i] = Number.NaN;
    src_p.data['alt_plot'][i] = Number.NaN;
}

if (index-N_time_window_half<0) {
var index_p_s = 0;
} else {
var index_p_s = index - N_time_window_half;
}

if (index+N_time_window_half>src_g.data['alt'].length) {
var index_p_e = src_g.data['alt'].length;
} else {
var index_p_e = index + N_time_window_half;
}

var icount = 0;
for (i = index_p_s; i < index_p_e; i++) {
    src_p.data['zen_plot'][icount] = src_t.data['zen_plot'][i];
    src_p.data['nad_plot'][icount] = src_t.data['nad_plot'][i];
    src_p.data['alt_plot'][icount] = src_g.data['alt'][i];
    icount += 1;
}
src_p.change.emit();

plt_p.x_range.end   = nanmax(src_p.data['zen_plot'])*1.1;
plt_p.y_range.start = nanmin(src_p.data['alt_plot'])*0.9;
plt_p.y_range.end   = nanmax(src_p.data['alt_plot'])*1.1;

var title3 = 'Profile [' + src_t.data['tmhr'][index_p_s].toFixed(4).toString() + ', ' + src_t.data['tmhr'][index_p_e].toFixed(4).toString() + ']';
plt_p.title.text = title3;



span_t.location = slider_t.value;
    """)

    layout0 = layout(
              [[plt_spec, plt_geo, plt_prof],
               [slider_spec],
               [plt_time],
               [slider_time]], sizing_mode='fixed')

    save(layout0)

def quicklook_bokeh_ssfr_and_spns(fname_ssfr, fname_spns, wvl0=None, tmhr0=None, tmhr_range=None, wvl_range=[300.0, 2200.0], tmhr_step=10, wvl_step=2, description=None, fname_html=None):

    from bokeh.layouts import layout, gridplot
    from bokeh.models import ColumnDataSource, ColorBar
    from bokeh.models.widgets import Select, Slider, CheckboxGroup
    from bokeh.models import Toggle, CustomJS, Legend, Span, HoverTool
    from bokeh.plotting import figure, output_file, save
    from bokeh.transform import linear_cmap
    from bokeh.palettes import RdYlBu6, Spectral6
    from bokeh.tile_providers import get_provider, Vendors

    # prepare data
    # ========================================================================================================
    data_time_dict, data_spec_dict, data_spns_dict, data_geo_dict = pre_bokeh(fname_ssfr, fname_spns=fname_spns, tmhr_range=tmhr_range, tmhr_step=tmhr_step, wvl_step=wvl_step, tmhr0=tmhr0, wvl0=wvl0)

    if wvl0 is None:
        wvl0 = 500.0
    index_zen_wvl = np.argmin(np.abs(data_spec_dict['zen_wvl']-wvl0))
    index_nad_wvl = np.argmin(np.abs(data_spec_dict['nad_wvl']-wvl0))

    if tmhr0 is None:
        index_tmhr = np.where(np.logical_not(np.isnan(data_time_dict['zen%d' % np.argmin(np.abs(data_spec_dict['zen_wvl']-wvl0))])))[0][0]
    else:
        index_tmhr = np.argmin(np.abs(data_time_dict['tmhr']-tmhr0))

    data_geo0  = ColumnDataSource(data={'x':[data_geo_dict['x'][index_tmhr]],
                                        'y':[data_geo_dict['y'][index_tmhr]],
                                        'lon':[data_geo_dict['lon'][index_tmhr]],
                                        'lat':[data_geo_dict['lat'][index_tmhr]],
                                        'alt':[data_geo_dict['alt'][index_tmhr]]})
    data_geo1  = ColumnDataSource(data=data_geo_dict)
    data_geo   = ColumnDataSource(data=data_geo_dict)

    data_spec  = ColumnDataSource(data=data_spec_dict)
    data_time  = ColumnDataSource(data=data_time_dict)
    data_spns  = ColumnDataSource(data=data_spns_dict)
    # ========================================================================================================

    if description is not None:
        title = 'SSFR Quicklook Plot (%s)' % description
    else:
        title = 'SSFR Quicklook Plot'

    if fname_html is None:
        fname_html = 'ssfr-bokeh-plot_%s.html' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    output_file(fname_html, title=title, mode='inline')

    height_geo = 350
    width_geo  = 440
    height_spec = height_geo
    width_spec  = 650
    height_time = 320
    width_time  = 10 + width_geo + width_spec

    # map plot
    # ========================================================================================================
    plt_geo  = figure(plot_height=height_geo, plot_width=width_geo,
                  title='Aircraft at %.4f km' % data_geo.data['alt'][index_tmhr],
                  tools='reset,save,box_zoom,wheel_zoom,pan',
                  active_scroll='wheel_zoom', active_drag='pan',
                  x_axis_label='Longitude', y_axis_label='Latitude',
                  x_axis_type="mercator", y_axis_type="mercator",
                  x_range=[data_geo.data['x'][index_tmhr]-50000.0, data_geo.data['x'][index_tmhr]+50000.0],
                  y_range=[data_geo.data['y'][index_tmhr]-50000.0, data_geo.data['y'][index_tmhr]+50000.0], output_backend='webgl')

    tile_provider = get_provider(Vendors.CARTODBPOSITRON)
    plt_geo.add_tile(tile_provider)

    htool    = HoverTool(tooltips = [('Longitude', '@lon{0.0000}'), ('Latitude', '@lat{0.0000}'), ('Altitude', '@alt{0.0000}km'), ('Solar Zenith', '@sza{0.00}')], mode='mouse', line_policy='nearest')
    plt_geo.add_tools(htool)

    mapper = linear_cmap(field_name='alt', palette=Spectral6, low=0.0, high=6.0)

    plt_geo.circle('x', 'y', source=data_geo , color=mapper, size=6, fill_alpha=0.1, line_width=0.0)
    plt_geo.circle('x', 'y', source=data_geo0, color=mapper, line_color=mapper, size=15, fill_alpha=0.5, line_width=2.0)
    plt_geo.circle('x', 'y', source=data_geo1, color=mapper, size=4, fill_alpha=1.0, line_width=0.0)

    color_bar = ColorBar(color_mapper=mapper['transform'], width=10, location=(0,0))
    plt_geo.add_layout(color_bar, 'right')

    plt_geo.title.text_font_size = '1.3em'
    plt_geo.title.align = 'center'
    plt_geo.xaxis.axis_label_text_font_style = 'normal'
    plt_geo.yaxis.axis_label_text_font_style = 'normal'
    plt_geo.xaxis.axis_label_text_font_size  = '1.0em'
    plt_geo.xaxis.major_label_text_font_size = '1.0em'
    plt_geo.yaxis.axis_label_text_font_size  = '1.0em'
    plt_geo.yaxis.major_label_text_font_size = '1.0em'
    # ========================================================================================================


    # spectra plot
    # ========================================================================================================
    plt_spec = figure(plot_height=height_spec, plot_width=width_spec,
                  title='Spectra at %s UTC' % ((datetime.datetime(1970, 1, 1)+datetime.timedelta(days=data_time.data['jday'][index_tmhr]-1.0)).strftime('%Y-%m-%d %H:%M:%S')),
                  tools='reset,save,box_zoom,wheel_zoom,pan',
                  active_scroll='wheel_zoom', active_drag='pan', x_axis_label='Wavelength [nm]', y_axis_label='Flux Density',
                  x_range=[300, 2200], y_range=[0.0, np.nanmax(data_spec.data['zen_plot'])*1.1], output_backend='webgl')

    htool    = HoverTool(tooltips = [('Wavelength', '$x{0.00}nm'), ('Flux', '$y{0.0000}')], mode='mouse', line_policy='nearest')
    plt_spec.add_tools(htool)

    plt_spec.circle('zen_wvl', 'sol_plot', source=data_spec, color='black'     , size=3, legend='Kurudz', fill_alpha=0.4, line_alpha=0.4)
    plt_spec.circle('zen_wvl', 'zen_plot', source=data_spec, color='dodgerblue', size=3, legend='Zenith')
    plt_spec.circle('nad_wvl', 'nad_plot', source=data_spec, color='lightcoral', size=3, legend='Nadir')
    plt_spec.circle('wvl', 'dif_plot', source=data_spns, color='lightgreen', size=3, legend='Diffuse')
    plt_spec.circle('wvl', 'tot_plot', source=data_spns, color='green'     , size=3, legend='Total')


    slider_spec = Slider(start=300, end=2200, value=wvl0, step=0.01, width=width_spec, title='Wavelength [nm]', format='0[.]00')
    vline_spec  = Span(location=slider_spec.value, dimension='height', line_color='black', line_dash='dashed', line_width=1)
    plt_spec.add_layout(vline_spec)

    plt_spec.legend.click_policy  = 'hide'
    plt_spec.title.text_font_size = '1.3em'
    plt_spec.title.align     = 'center'
    plt_spec.legend.location = 'top_right'
    plt_spec.xaxis.axis_label_text_font_style = 'normal'
    plt_spec.yaxis.axis_label_text_font_style = 'normal'
    plt_spec.xaxis.axis_label_text_font_size  = '1.0em'
    plt_spec.xaxis.major_label_text_font_size = '1.0em'
    plt_spec.yaxis.axis_label_text_font_size  = '1.0em'
    plt_spec.yaxis.major_label_text_font_size = '1.0em'
    # ========================================================================================================


    # time series plot
    # ========================================================================================================
    xrange_s = np.nanmin(data_time.data['tmhr'])-0.1
    xrange_e = np.nanmax(data_time.data['tmhr'])+0.1
    yrange_e = np.nanmax(data_time.data['zen_plot'])*1.1

    plt_time = figure(height=height_time, width=width_time,
                  title='Time Series at %.2f nm (Zenith) and %.2f nm (Nadir)' % (data_spec.data['zen_wvl'][index_zen_wvl], data_spec.data['nad_wvl'][index_nad_wvl]),
                  tools='reset,save,box_zoom,wheel_zoom,pan',
                  active_scroll='wheel_zoom', active_drag='pan', x_axis_label='Time [Hour]', y_axis_label='Flux Density',
                  x_range=[xrange_s, xrange_e],
                  y_range=[0.0     , yrange_e], output_backend='webgl')

    htool = HoverTool(tooltips = [('Time', '$x{0.0000}'), ('Flux', '$y{0.0000}')], mode='mouse', line_policy='nearest')
    plt_time.add_tools(htool)

    plt_time.circle('tmhr', 'sol_plot', source=data_time, color='black'     , size=3, legend='Kurudz', fill_alpha=0.4, line_alpha=0.4)
    plt_time.circle('tmhr', 'zen_plot', source=data_time, color='dodgerblue', size=3, legend='Zenith')
    plt_time.circle('tmhr', 'nad_plot', source=data_time, color='lightcoral', size=3, legend='Nadir')
    plt_time.circle('tmhr', 'dif_plot', source=data_time, color='lightgreen', size=3, legend='Diffuse')
    plt_time.circle('tmhr', 'tot_plot', source=data_time, color='green'     , size=3, legend='Total')
    plt_time.legend.location = 'top_right'

    slider_time = Slider(start=xrange_s, end=xrange_e, value=data_time.data['tmhr'][index_tmhr], step=0.0001, width=width_time, title='Time [Hour]', format='0[.]0000')
    vline_time  = Span(location=slider_time.value, dimension='height', line_color='black', line_dash='dashed', line_width=1)
    plt_time.add_layout(vline_time)

    plt_time.legend.click_policy  = 'hide'
    plt_time.title.text_font_size = '1.3em'
    plt_time.title.align          = 'center'
    plt_time.xaxis.axis_label_text_font_style = 'normal'
    plt_time.yaxis.axis_label_text_font_style = 'normal'
    plt_time.xaxis.axis_label_text_font_size  = '1.0em'
    plt_time.xaxis.major_label_text_font_size = '1.0em'
    plt_time.yaxis.axis_label_text_font_size  = '1.0em'
    plt_time.yaxis.major_label_text_font_size = '1.0em'
    # ========================================================================================================


    slider_spec.callback = CustomJS(args=dict(
                             plt_t    = plt_time,
                             span_s   = vline_spec,
                             slider_s = slider_spec,
                             src_t    = data_time,
                             src_s    = data_spec,
                             src_spns = data_spns), code="""
function closest (num, arr) {
    var curr = 0;
    var diff = Math.abs (num - arr[curr]);
    for (var val = 0; val < arr.length; val++) {
        var newdiff = Math.abs (num - arr[val]);
        if (newdiff < diff) {
            diff = newdiff;
            curr = val;
        }
    }
    return curr;
}

var x  = src_t.data['tmhr'];

var zen_wvl     = src_s.data['zen_wvl'];
var zen_index   = closest(slider_s.value, zen_wvl);
var zen_index_s = zen_index.toString();
var v1 = 'zen' + zen_index_s;

var nad_wvl     = src_s.data['nad_wvl'];
var nad_index   = closest(slider_s.value, nad_wvl);
var nad_index_s = nad_index.toString();
var v2 = 'nad' + nad_index_s;

var spns_wvl     = src_spns.data['wvl'];
var spns_index   = closest(slider_s.value, spns_wvl);
var spns_index_s = spns_index.toString();
var v3 = 'dif' + spns_index_s;
var v4 = 'tot' + spns_index_s;

var max0 = 0.0;
for (i = 0; i < x.length; i++) {
    if (src_t.data[v1][i]>max0) {
    max0 = src_t.data[v1][i];
    }
    src_t.data['zen_plot'][i] = src_t.data[v1][i];
    src_t.data['nad_plot'][i] = src_t.data[v2][i];
    src_t.data['dif_plot'][i] = src_t.data[v3][i];
    src_t.data['tot_plot'][i] = src_t.data[v4][i];
    src_t.data['sol_plot'][i] = src_s.data['solar'][zen_index] * src_t.data['mu'][i];
}
src_t.change.emit();

var title = 'Time Series at ' + zen_wvl[zen_index].toFixed(2).toString() + ' nm (Zenith) and ' + nad_wvl[nad_index].toFixed(2).toString() + ' nm (Nadir)' ;
plt_t.title.text = title;

if (max0 > 0.0) {
plt_t.y_range.end = max0*1.1;
}

span_s.location = slider_s.value;
    """)


    slider_time.callback = CustomJS(args=dict(
                             plt_s    = plt_spec,
                             plt_g    = plt_geo,
                             span_t   = vline_time,
                             slider_t = slider_time,
                             src_t  = data_time,
                             src_s  = data_spec,
                             src_spns = data_spns,
                             src_g  = data_geo,
                             src_g0 = data_geo0,
                             src_g1 = data_geo1,
                             ), code="""
function closest (num, arr) {
    var curr = 0;
    var diff = Math.abs (num - arr[curr]);
    for (var val = 0; val < arr.length; val++) {
        var newdiff = Math.abs (num - arr[val]);
        if (newdiff < diff) {
            diff = newdiff;
            curr = val;
        }
    }
    return curr;
}

var x  = src_t.data['tmhr'];

var index = closest(slider_t.value, x);
var index_s = index.toString();
var v1 = 'zen' + index_s;
var v2 = 'nad' + index_s;
var v3 = 'dif' + index_s;
var v4 = 'tot' + index_s;

var max0 = 0.0;
for (i = 0; i < src_s.data['zen_wvl'].length; i++) {
    if (src_s.data[v1][i]>max0) {
    max0 = src_s.data[v1][i];
    }
    src_s.data['zen_plot'][i] = src_s.data[v1][i];
    src_s.data['nad_plot'][i] = src_s.data[v2][i];
    src_s.data['sol_plot'][i] = src_s.data['solar'][i] * src_t.data['mu'][index];
}
src_s.change.emit();

for (i = 0; i < src_spns.data['wvl'].length; i++) {
    src_spns.data['dif_plot'][i] = src_spns.data[v3][i];
    src_spns.data['tot_plot'][i] = src_spns.data[v4][i];
}
src_spns.change.emit();

var msec0 = (src_t.data['jday'][index]-1.0)*86400000.0;
var date0 = new Date(msec0);

var month0 = date0.getUTCMonth() + 1
date_s = date0.getUTCFullYear() + '-'
        + ('0' + month0).slice(-2) + '-'
        + ('0' + date0.getUTCDate()).slice(-2) + ' '
        + ('0' + date0.getUTCHours()).slice(-2) + ':'
        + ('0' + date0.getUTCMinutes()).slice(-2) + ':'
        + ('0' + date0.getUTCSeconds()).slice(-2);

var title1 = 'Spectra at ' + date_s + ' UTC';
plt_s.title.text = title1;

if (max0 > 0.0) {
plt_s.y_range.end = max0*1.1;
}

var title2 = 'Aircraft at ' + src_g.data['alt'][index].toFixed(4).toString() + ' km';
plt_g.title.text = title2;

plt_g.x_range.start = src_g.data['x'][index]-50000.0;
plt_g.x_range.end   = src_g.data['x'][index]+50000.0;
plt_g.y_range.start = src_g.data['y'][index]-50000.0;
plt_g.y_range.end   = src_g.data['y'][index]+50000.0;

src_g0.data['x'][0]   = src_g.data['x'][index];
src_g0.data['y'][0]   = src_g.data['y'][index];
src_g0.data['alt'][0] = src_g.data['alt'][index];
src_g0.change.emit();

for (i = 0; i < src_g.data['x'].length; i++) {
    if (i<=index) {
    src_g1.data['x'][i] = src_g.data['x'][i];
    src_g1.data['y'][i] = src_g.data['y'][i];
    } else {
    src_g1.data['x'][i] = Number.NaN;
    src_g1.data['y'][i] = Number.NaN;
    }
}
src_g1.change.emit();

span_t.location = slider_t.value;
    """)

    layout0 = layout(
              [[plt_spec, plt_geo],
               [slider_spec],
               [plt_time],
               [slider_time]], sizing_mode='fixed')

    save(layout0)


if __name__ == '__main__':

    fname = '/Users/hoch4240/Chen/mygit/arg-ssfr/examples/data/SSFR_20190803_HSK.h5'
    quicklook_bokeh(fname, tmhr_range=[12.0, 20.0], fname_html='fig.html')

    if False: # for hover animation
        invisible_circle = Circle(x='x', y='y', fill_color='gray', fill_alpha=0.05, line_color=None, size=20)
        visible_circle = Circle(x='x', y='y', fill_color='firebrick', fill_alpha=0.5, line_color=None, size=20)
        cr = p.add_glyph(source, invisible_circle, selection_glyph=visible_circle, nonselection_glyph=invisible_circle)

        # Add a hover tool, that selects the circle
        code = "source.set('selected', cb_data['index']);"
        callback = CustomJS(args={'source': source}, code=code)
        p.add_tools(HoverTool(tooltips=None, callback=callback, renderers=[cr], mode='hline'))
