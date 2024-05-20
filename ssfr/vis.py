import os
import sys
import copy
import glob
import datetime
import h5py
import numpy as np


import ssfr


__all__ = [
        'quicklook_bokeh_ssfr_and_spns',
        'quicklook_spns_bokeh',
        'quicklook_bokeh_ssfr_raw',
        'find_offset_bokeh',
        ]


def lonlat_to_xy(lon, lat):

    import cartopy.crs as ccrs

    proj0 = ccrs.PlateCarree()
    proj1 = ccrs.Mercator.GOOGLE
    xy = proj1.transform_points(proj0, lon, lat)

    x = xy[:, 0]
    y = xy[:, 1]

    return x, y

def pre_bokeh_spns(
        fname,
        tmhr0=None,
        wvl0=None,
        tmhr_range=None,
        tmhr_step=10,
        wvl_step=2,
        ):

    """
    In the original file (specified by <fname>),

        'jday': julian day (first day is 0001-01-01)
        'tmhr': time in hour
        'lon': longitude
        'lat': latitude
        'alt': altitude
        'sza': solar zenith angle

        'tot/toa0': TOA downwelling irradiance from Kurudz solar file

        'tot/wvl': wavelength for total spectral irradiance
        'tot/flux': total spectral irradiance

        'dif/wvl': wavelength for diffuse spectral irradiance
        'dif/flux': diffuse spectral irradiance
    """

    f0 = h5py.File(fname, 'r')

    # time range selection
    #/----------------------------------------------------------------------------\#
    tmhr = f0['tmhr'][...]
    if tmhr_range is None:
        logic_tmhr = np.repeat(True, tmhr.size)
    else:
        logic_tmhr = (tmhr>=tmhr_range[0]) & (tmhr<=tmhr_range[1])
    #\----------------------------------------------------------------------------/#


    # get general data
    #/----------------------------------------------------------------------------\#
    data = {}
    data['tmhr'] = tmhr[logic_tmhr][::tmhr_step]
    data['jday'] = f0['jday'][...][logic_tmhr][::tmhr_step] - ssfr.util.dtime_to_jday(datetime.datetime(1969, 12, 31))
    data['lon']  = f0['lon'][...][logic_tmhr][::tmhr_step]
    data['lat']  = f0['lat'][...][logic_tmhr][::tmhr_step]
    data['alt']  = f0['alt'][...][logic_tmhr][::tmhr_step]/1000.0
    data['sza']  = f0['sza'][...][logic_tmhr][::tmhr_step]

    data['mu']   = np.cos(np.deg2rad(data['sza']))
    data['toa0'] = f0['tot/toa0'][...][::wvl_step]

    # total
    data['wvl0']  = f0['tot/wvl'][...][::wvl_step]
    data['flux0'] = f0['tot/flux'][...][logic_tmhr, ...][::tmhr_step, ::wvl_step]

    # diffuse
    data['wvl1'] = f0['dif/wvl'][...][::wvl_step]
    data['flux1'] = f0['dif/flux'][...][logic_tmhr, ...][::tmhr_step, ::wvl_step]

    Nvar = 2
    #\----------------------------------------------------------------------------/#


    # prepare data for time series
    #/----------------------------------------------------------------------------\#
    if wvl0 is None:
        wvl0 = 500.0

    data_time = {}

    data_time['tmhr'] = data['tmhr']
    data_time['jday'] = data['jday']

    for ivar in range(Nvar):

        var_wvl  = 'wvl%d' % ivar
        var_flux = 'flux%d' % ivar

        index_wvl = np.argmin(np.abs(data[var_wvl]-wvl0))
        data_time['%s_plot' % var_flux] = data[var_flux][:, index_wvl]

        for iwvl in range(data[var_wvl].size):
            data_time['%s_%d' % (var_flux, iwvl)] = data[var_flux][:, iwvl]

    data_time['mu'] = data['mu']
    data_time['toa_plot'] = data['toa0'][index_wvl] * data['mu']

    data_time['alt'] = data['alt'] / (np.nanmax(data['alt'])-np.nanmin(data['alt'])) \
            * (np.nanmax(data['flux0'][:, index_wvl])-np.nanmin(data['flux0'][:, index_wvl]))
    #\----------------------------------------------------------------------------/#


    # prepare data for spectra
    #/----------------------------------------------------------------------------\#
    if tmhr0 is None:
        index_tmhr = np.where(np.logical_not(np.isnan(data[var_flux][:, index_wvl])))[0][0]
    else:
        index_tmhr = np.argmin(np.abs(data['tmhr']-tmhr0))

    data_spec = {}

    for ivar in range(Nvar):

        var_wvl  = 'wvl%d' % ivar
        var_flux = 'flux%d' % ivar

        data_spec[var_wvl]  = data[var_wvl]
        data_spec['%s_plot' % var_flux] = data[var_flux][index_tmhr, :]

        for itmhr in range(data['tmhr'].size):
            data_spec['%s_%d' % (var_flux, itmhr)] = data[var_flux][itmhr, :]

    data_spec['toa0']     = data['toa0']
    data_spec['toa_plot'] = data['toa0'] * data['mu'][index_tmhr]
    #\----------------------------------------------------------------------------/#


    # prepare data for geo map
    #/----------------------------------------------------------------------------\#
    data_geo = {}

    data_geo['lon']  = data['lon']
    data_geo['lat']  = data['lat']
    data_geo['alt']  = data['alt']
    data_geo['sza']  = data['sza']

    x, y = lonlat_to_xy(data['lon'], data['lat'])
    data_geo['x'] = x
    data_geo['y'] = y
    #\----------------------------------------------------------------------------/#

    f0.close()

    return data_time, data_spec, data_geo

def quicklook_bokeh_spns(
        fname,
        wvl0=None,
        tmhr0=None,
        tmhr_range=None,
        wvl_range=[350.0, 800.0],
        tmhr_step=10,
        wvl_step=2,
        description=None,
        fname_html=None
        ):

    from bokeh.layouts import layout, gridplot
    from bokeh.models import ColumnDataSource, ColorBar
    from bokeh.models.widgets import Select, Slider, CheckboxGroup
    from bokeh.models import Toggle, CustomJS, Legend, Span, HoverTool
    from bokeh.plotting import figure, output_file, save
    from bokeh.transform import linear_cmap
    from bokeh.palettes import RdYlBu6, Spectral6
    from bokeh.tile_providers import get_provider, Vendors

    # prepare data
    #/----------------------------------------------------------------------------\#
    data_time_dict, data_spec_dict, data_geo_dict = pre_bokeh_spns(fname, tmhr_range=tmhr_range, tmhr_step=tmhr_step, wvl_step=wvl_step, tmhr0=tmhr0, wvl0=wvl0)

    # get indices for selected wavelength and time
    #/--------------------------------------------------------------\#
    if wvl0 is None:
        wvl0 = 500.0
    index_wvl = np.argmin(np.abs(data_spec_dict['wvl0']-wvl0))

    if tmhr0 is None:
        index_tmhr = np.where(np.logical_not(np.isnan(data_time_dict['flux0_%d' % np.argmin(np.abs(data_spec_dict['wvl0']-wvl0))])))[0][0]
    else:
        index_tmhr = np.argmin(np.abs(data_time_dict['tmhr']-tmhr0))
    #\--------------------------------------------------------------/#

    # data_geo
    #/--------------------------------------------------------------\#
    # flight track
    data_geo   = ColumnDataSource(data=data_geo_dict)

    # aircraft location
    data_geo0  = ColumnDataSource(data={'x':[data_geo_dict['x'][index_tmhr]],
                                        'y':[data_geo_dict['y'][index_tmhr]],
                                        'lon':[data_geo_dict['lon'][index_tmhr]],
                                        'lat':[data_geo_dict['lat'][index_tmhr]],
                                        'alt':[data_geo_dict['alt'][index_tmhr]]})

    # past flight track
    data_geo_dict_ = copy.deepcopy(data_geo_dict)
    logic_good = (data_time_dict['tmhr']>(data_time_dict['tmhr'][index_tmhr]-0.25)) & (data_time_dict['tmhr']<=data_time_dict['tmhr'][index_tmhr])
    data_geo_dict_['x'][~logic_good] = np.nan
    data_geo_dict_['y'][~logic_good] = np.nan
    data_geo1  = ColumnDataSource(data=data_geo_dict_)
    #\--------------------------------------------------------------/#


    # data_time
    #/--------------------------------------------------------------\#
    data_time  = ColumnDataSource(data=data_time_dict)
    #\--------------------------------------------------------------/#

    # data_spec
    #/--------------------------------------------------------------\#
    data_spec  = ColumnDataSource(data=data_spec_dict)
    #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#


    # bokeh plot settings
    #/----------------------------------------------------------------------------\#
    if description is not None:
        title = 'SPN-S Quicklook Plot (%s)' % description
    else:
        title = 'SPN-S Quicklook Plot'

    if fname_html is None:
        fname_html = 'spns-bokeh-plot_created-at-%s.html' % datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    output_file(fname_html, title=title, mode='inline')

    height_geo = 350
    width_geo  = 440
    height_spec = height_geo
    width_spec  = 650
    height_time = 320
    width_time  = 10 + width_geo + width_spec
    #\----------------------------------------------------------------------------/#


    # map plot
    #/----------------------------------------------------------------------------\#
    if data_geo0.data['alt'][0] < 1.0:
        title_ = 'Aircraft at %.1f m' % data_geo0.data['alt'][0]
    else:
        title_ = 'Aircraft at %.4f km' % data_geo0.data['alt'][0]

    plt_geo  = figure(
                  plot_height=height_geo,
                  plot_width=width_geo,
                  title=title_,
                  tools='reset,save,box_zoom,wheel_zoom,pan',
                  active_scroll='wheel_zoom',
                  active_drag='pan',
                  x_axis_label='Longitude',
                  y_axis_label='Latitude',
                  x_axis_type="mercator",
                  y_axis_type="mercator",
                  x_range=[data_geo0.data['x'][0]-20000.0, data_geo0.data['x'][0]+20000.0],
                  y_range=[data_geo0.data['y'][0]-20000.0, data_geo0.data['y'][0]+20000.0],
                  output_backend='webgl',
                  )

    tile_provider = get_provider(Vendors.CARTODBPOSITRON)
    plt_geo.add_tile(tile_provider)

    htool = HoverTool(tooltips=[('Longitude', '@lon{0.0000}º'), ('Latitude', '@lat{0.0000}º'), ('Altitude', '@alt{0.0000}km'), ('Solar Zenith', '@sza{0.00}º')], mode='mouse', line_policy='nearest')
    plt_geo.add_tools(htool)

    mapper = linear_cmap(field_name='alt', palette=Spectral6, low=0.0, high=np.round(np.nanmax(data_geo_dict['alt']), decimals=0))

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
    #\----------------------------------------------------------------------------/#


    # spectra plot
    #/----------------------------------------------------------------------------\#
    plt_spec = figure(
                  plot_height=height_spec,
                  plot_width=width_spec,
                  title='Spectral Flux at %s UTC' % ((datetime.datetime(1970, 1, 1)+datetime.timedelta(days=data_time.data['jday'][index_tmhr]-1.0)).strftime('%Y-%m-%d %H:%M:%S')),
                  tools='reset,save,box_zoom,wheel_zoom,pan',
                  active_scroll='wheel_zoom',
                  active_drag='pan',
                  x_axis_label='Wavelength [nm]',
                  y_axis_label='Flux Density',
                  x_range=wvl_range,
                  y_range=[0.0, np.nanmax(data_spec.data['flux0_plot'])*1.1],
                  output_backend='webgl'
                  )

    htool = HoverTool(tooltips = [('Wavelength', '$x{0.00}nm'), ('Flux', '$y{0.0000}')], mode='mouse', line_policy='nearest')
    plt_spec.add_tools(htool)

    plt_spec.circle('wvl0', 'toa_plot'  , source=data_spec, color='gray'      , size=3, legend_label='TOA↓ (Kurudz)', fill_alpha=0.4, line_alpha=0.4)
    plt_spec.circle('wvl0', 'flux0_plot', source=data_spec, color='green'     , size=3, legend_label='Total↓ (SPN-S)')
    plt_spec.circle('wvl1', 'flux1_plot', source=data_spec, color='lightgreen', size=3, legend_label='Diffuse↓ (SPN-S)')


    slider_spec = Slider(start=wvl_range[0], end=wvl_range[-1], value=wvl0, step=0.01, width=width_spec, height=40, title='Wavelength [nm]', format='0[.]00')
    vline_spec  = Span(location=slider_spec.value, dimension='height', line_color='black', line_dash='dashed', line_width=1)
    plt_spec.add_layout(vline_spec)

    plt_spec.legend.click_policy  = 'hide'
    plt_spec.legend.location = 'top_right'
    plt_spec.legend.background_fill_alpha = 0.8
    plt_spec.title.text_font_size = '1.3em'
    plt_spec.title.align     = 'center'
    plt_spec.xaxis.axis_label_text_font_style = 'normal'
    plt_spec.yaxis.axis_label_text_font_style = 'normal'
    plt_spec.xaxis.axis_label_text_font_size  = '1.0em'
    plt_spec.xaxis.major_label_text_font_size = '1.0em'
    plt_spec.yaxis.axis_label_text_font_size  = '1.0em'
    plt_spec.yaxis.major_label_text_font_size = '1.0em'
    #\----------------------------------------------------------------------------/#


    # time series plot
    #/----------------------------------------------------------------------------\#
    xrange_s = np.nanmin(data_time.data['tmhr'])-0.1
    xrange_e = np.nanmax(data_time.data['tmhr'])+0.1
    yrange_e = np.nanmax(data_time.data['flux0_plot'])*1.1

    plt_time = figure(
                  height=height_time,
                  width=width_time,
                  title='Flux Time Series at %.2f nm' % (data_spec.data['wvl0'][index_wvl]),
                  tools='reset,save,box_zoom,wheel_zoom,pan',
                  active_scroll='wheel_zoom',
                  active_drag='pan',
                  x_axis_label='Time [Hour]',
                  y_axis_label='Flux Density',
                  x_range=[xrange_s, xrange_e],
                  y_range=[0.0     , yrange_e],
                  output_backend='webgl'
                  )

    htool = HoverTool(tooltips = [('Time', '$x{0.0000}'), ('Flux', '$y{0.0000}')], mode='mouse', line_policy='nearest')
    plt_time.add_tools(htool)

    plt_time.varea(x=data_time.data['tmhr'], y1=np.zeros_like(data_time.data['tmhr']), y2=data_time.data['alt'], fill_alpha=0.2, fill_color='purple')
    plt_time.circle('tmhr', 'toa_plot'  , source=data_time, color='gray'      , size=3, legend_label='TOA↓ (Kurudz)', fill_alpha=0.4, line_alpha=0.4)
    plt_time.circle('tmhr', 'flux0_plot', source=data_time, color='green'     , size=3, legend_label='Total↓ (SPN-S)')
    plt_time.circle('tmhr', 'flux1_plot', source=data_time, color='lightgreen', size=3, legend_label='Diffuse↓ (SPN-S)')

    slider_time = Slider(start=xrange_s, end=xrange_e, value=data_time.data['tmhr'][index_tmhr], step=0.0001, width=width_time, height=40, title='Time [Hour]', format='0[.]0000')
    vline_time  = Span(location=slider_time.value, dimension='height', line_color='black', line_dash='dashed', line_width=1)
    plt_time.add_layout(vline_time)

    plt_time.legend.click_policy  = 'hide'
    plt_time.legend.location = 'top_right'
    plt_time.legend.background_fill_alpha = 0.8
    plt_time.title.text_font_size = '1.3em'
    plt_time.title.align          = 'center'
    plt_time.xaxis.axis_label_text_font_style = 'normal'
    plt_time.yaxis.axis_label_text_font_style = 'normal'
    plt_time.xaxis.axis_label_text_font_size  = '1.0em'
    plt_time.xaxis.major_label_text_font_size = '1.0em'
    plt_time.yaxis.axis_label_text_font_size  = '1.0em'
    plt_time.yaxis.major_label_text_font_size = '1.0em'
    #\----------------------------------------------------------------------------/#

    # callback (spec slider)
    #/----------------------------------------------------------------------------\#
    slider_spec_callback = CustomJS(args=dict(
                             plt_t    = plt_time,
                             span_s   = vline_spec,
                             slider_s = slider_spec,
                             src_s    = data_spec,
                             src_t    = data_time,
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

var wvl0 = src_s.data['wvl0'];
var wvl0_index = closest(slider_s.value, wvl0);

var wvl1 = src_s.data['wvl1'];
var wvl1_index = closest(slider_s.value, wvl1);

var wvl0_index_s = wvl0_index.toString();
var wvl1_index_s = wvl1_index.toString();

var v0 = 'flux0_' + wvl0_index_s;
var v1 = 'flux1_' + wvl1_index_s;

var max0 = 0.0;
var i = 0;

for (i = 0; i < x.length; i++) {
    if (src_t.data[v0][i]>max0) {
    max0 = src_t.data[v0][i];
    }
    src_t.data['flux0_plot'][i] = src_t.data[v0][i];
    src_t.data['flux1_plot'][i] = src_t.data[v1][i];
    src_t.data['toa_plot'][i]   = src_s.data['toa0'][wvl0_index] * src_t.data['mu'][i];
}
src_t.change.emit();

var title = 'Flux Time Series at ' + wvl0[wvl0_index].toFixed(2).toString() + ' nm';
plt_t.title.text = title;

if (max0 > 0.0) {
plt_t.y_range.end = max0*1.1;
}

span_s.location = slider_s.value;
    """)
    #\----------------------------------------------------------------------------/#

    # callback (time slider)
    #/----------------------------------------------------------------------------\#
    slider_time_callback = CustomJS(args=dict(
                             plt_s    = plt_spec,
                             plt_g    = plt_geo,
                             span_t   = vline_time,
                             slider_t = slider_time,
                             src_t  = data_time,
                             src_s  = data_spec,
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

var v0 = 'flux0_' + index_s;
var v1 = 'flux1_' + index_s;

var max0 = 0.0;
var i = 0;

for (i = 0; i < src_s.data['wvl0'].length; i++) {
    if (src_s.data[v0][i]>max0) {
    max0 = src_s.data[v0][i];
    }
    src_s.data['flux0_plot'][i] = src_s.data[v0][i];
    src_s.data['flux1_plot'][i] = src_s.data[v1][i];
    src_s.data['toa_plot'][i]   = src_s.data['toa0'][i] * src_t.data['mu'][index];
}
src_s.change.emit();

var msec0 = (src_t.data['jday'][index]-1.0)*86400000.0;
var date0 = new Date(msec0);

var month0 = date0.getUTCMonth() + 1;

var date_s = '';

date_s = date0.getUTCFullYear() + '-'
        + ('0' + month0).slice(-2) + '-'
        + ('0' + date0.getUTCDate()).slice(-2) + ' '
        + ('0' + date0.getUTCHours()).slice(-2) + ':'
        + ('0' + date0.getUTCMinutes()).slice(-2) + ':'
        + ('0' + date0.getUTCSeconds()).slice(-2);

var title1 = 'Spectral Flux at ' + date_s + ' UTC';
plt_s.title.text = title1;

if (max0 > 0.0) {
plt_s.y_range.end = max0*1.1;
}

if (src_g.data['alt'][index] < 1.0){
var alt_m  = src_g.data['alt'][index]*1000.0;
var title2 = 'Aircraft at ' + alt_m.toFixed(1).toString() + ' m';
} else {
var title2 = 'Aircraft at ' + src_g.data['alt'][index].toFixed(4).toString() + ' km';
}
plt_g.title.text = title2;

src_g0.data['x'][0]   = src_g.data['x'][index];
src_g0.data['y'][0]   = src_g.data['y'][index];
src_g0.data['alt'][0] = src_g.data['alt'][index];
src_g0.change.emit();

plt_g.x_range.start = src_g0.data['x'][0]-20000.0;
plt_g.x_range.end   = src_g0.data['x'][0]+20000.0;
plt_g.y_range.start = src_g0.data['y'][0]-20000.0;
plt_g.y_range.end   = src_g0.data['y'][0]+20000.0;

var index_past = closest(slider_t.value-0.25, x);

for (i = 0; i < src_g.data['x'].length; i++) {
    if (i>index_past && i<=index) {
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
    #\----------------------------------------------------------------------------/#

    slider_spec.js_on_change('value', slider_spec_callback)
    slider_time.js_on_change('value', slider_time_callback)

    layout0 = layout(
                     children=[[plt_spec, plt_geo],
                               [slider_spec],
                               [plt_time],
                               [slider_time]
                               ],
                     sizing_mode='fixed'
                     )

    save(layout0)



def find_offset_bokeh(
        data_dict,
        offset_x_range=[-600, 600],
        offset_y_range=[-1.0, 1.0],
        y_reset=True,
        x_reset=True,
        description=None,
        fname_html=None,
        ):

    from bokeh.layouts import layout, gridplot
    from bokeh.models import ColumnDataSource, ColorBar
    from bokeh.models.widgets import Select, Slider, CheckboxGroup
    from bokeh.models import Toggle, CustomJS, Legend, Span, HoverTool
    from bokeh.plotting import figure, output_file, save
    from bokeh.transform import linear_cmap
    from bokeh.palettes import RdYlBu6, Spectral6
    from bokeh.tile_providers import get_provider, Vendors

    # prepare data
    #/----------------------------------------------------------------------------\#
    x0_min = np.nanmin(data_dict['x0'])
    x0_max = np.nanmax(data_dict['x0'])
    x1_min = np.nanmin(data_dict['x1'])
    x1_max = np.nanmax(data_dict['x1'])
    x_min = min((x0_min, x1_min))
    x_max = max((x0_max, x1_max))

    if x_reset:
        x0 = data_dict['x0']-x_min
        x1 = data_dict['x1']-x_min
    else:
        x0 = data_dict['x0']
        x1 = data_dict['x1']

    if y_reset:
        y0 = (data_dict['y0']-np.nanmin(data_dict['y0'])) / (np.nanmax(data_dict['y0'])-np.nanmin(data_dict['y0']))
        y1 = (data_dict['y1']-np.nanmin(data_dict['y1'])) / (np.nanmax(data_dict['y1'])-np.nanmin(data_dict['y1']))
    else:
        y0 = data_dict['y0']
        y1 = data_dict['y1']

    x1_new = x1 + 0.0
    y1_new = y1 * 1.0

    logic_notnan = (~np.isnan(data_dict['x0'])) & (~np.isnan(data_dict['y0']))
    data_dict0 = {
            'x0': x0[logic_notnan],
            'y0': y0[logic_notnan],
            }
    data0 = ColumnDataSource(data=data_dict0)

    logic_notnan = (~np.isnan(data_dict['x1'])) & (~np.isnan(data_dict['y1']))
    data_dict1 = {
            'x1': x1[logic_notnan],
            'y1': y1[logic_notnan],
            'x1_new': x1_new[logic_notnan],
            'y1_new': y1_new[logic_notnan],
            }
    data1 = ColumnDataSource(data=data_dict1)
    #\----------------------------------------------------------------------------/#


    # bokeh plot settings
    #/----------------------------------------------------------------------------\#
    if description is not None:
        title = 'Offset Check (%s)' % description
    else:
        title = 'Offset Check'

    if fname_html is None:
        fname_html = 'find-time-offset-bokeh-plot_created-at-%s.html' % datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    output_file(fname_html, title=title, mode='inline')

    height_time = 500
    width_time  = 1200
    #\----------------------------------------------------------------------------/#


    # time series plot
    #/----------------------------------------------------------------------------\#
    xrange_s = -0.1
    xrange_e = (x_max-x_min)+0.1
    yrange_s = -0.1
    yrange_e = 1.1

    plt_offset = figure(
                  height=height_time,
                  width=width_time,
                  title=title,
                  tools='reset,save,box_zoom,wheel_zoom,pan',
                  active_scroll='wheel_zoom',
                  active_drag='pan',
                  x_axis_label='x',
                  y_axis_label='y [scaled]',
                  x_range=[xrange_s, xrange_e],
                  y_range=[yrange_s, yrange_e],
                  output_backend='webgl'
                  )

    htool = HoverTool(tooltips = [('x', '$x{0.0000}'), ('y', '$y{0.0}')], mode='mouse', line_policy='nearest')
    plt_offset.add_tools(htool)

    plt_offset.circle('x0'    , 'y0'    , source=data0, color='black', size=3, legend_label='Reference')
    plt_offset.circle('x1'    , 'y1'    , source=data1, color='red', size=3, legend_label='Raw', visible=False)
    plt_offset.circle('x1_new', 'y1_new', source=data1, color='green', size=3, legend_label='With Offset')

    slider_x_offset = Slider(start=offset_x_range[0], end=offset_x_range[1], value=0.0, step=0.01, width=width_time, height=40, title='X Offset', format='0[.]00')
    slider_y_offset = Slider(start=offset_y_range[0], end=offset_y_range[0], value=0.0, step=0.01, width=width_time, height=40, title='Y Offset', format='0[.]00')
    slider_y_scale  = Slider(start=0.2, end=5.0, value=1, step=0.001, width=width_time, height=40, title='Y Scale Factor', format='0[.]000')

    plt_offset.legend.click_policy  = 'hide'
    plt_offset.legend.location = 'top_right'
    plt_offset.legend.background_fill_alpha = 0.8
    plt_offset.title.text_font_size = '1.3em'
    plt_offset.title.align          = 'center'
    plt_offset.xaxis.axis_label_text_font_style = 'normal'
    plt_offset.yaxis.axis_label_text_font_style = 'normal'
    plt_offset.xaxis.axis_label_text_font_size  = '1.0em'
    plt_offset.xaxis.major_label_text_font_size = '1.0em'
    plt_offset.yaxis.axis_label_text_font_size  = '1.0em'
    plt_offset.yaxis.major_label_text_font_size = '1.0em'
    #\----------------------------------------------------------------------------/#

    slider_callback = CustomJS(
            args=dict(data1=data1, x_offset=slider_x_offset, y_offset=slider_y_offset, y_scale=slider_y_scale),
            code=
"""
var x = data1.data['x1'];
var y = data1.data['y1'];
var i = 0;
for (i = 0; i < x.length; i++) {
    data1.data['x1_new'][i] = x[i]+x_offset.value;
    data1.data['y1_new'][i] = y[i]*y_scale.value + y_offset.value;
}
data1.change.emit();
"""
        )

    slider_x_offset.js_on_change('value', slider_callback)
    slider_y_offset.js_on_change('value', slider_callback)
    slider_y_scale.js_on_change('value', slider_callback)

    layout0 = layout(
                     children=[
                               [plt_offset],
                               [slider_x_offset],
                               [slider_y_scale],
                               [slider_y_offset],
                               ],
                     sizing_mode='fixed'
                     )

    save(layout0)




def quicklook_mpl_ssfr_raw(
        data0,
        ichan=100,
        extra_tag='',
        plot_corr=False,
        ):

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


    filename = os.path.basename(data0['info']['fnames'][0])
    file_ext = filename.split('.')[-1]
    ssfr_tag = data0['info']['ssfr_tag']

    if ssfr_tag == 'NASA Ames SSFR':
        wvls  = ssfr.nasa_ssfr.get_ssfr_wavelength()
        y_range = [0, 2**5]
    elif ssfr_tag == 'CU LASP SSFR':
        wvls  = ssfr.lasp_ssfr.get_ssfr_wavelength()
        y_range  = [-2**5, 2**5]
    else:
        msg = '\nError [plot_ssfr_raw]: Cannot recognize SSFR system from given file extension <.%s>.' % file_ext
        raise OSError(msg)

    # data0['spectra'] /= 2**10
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
        ax1.set_xlim((-10, Nchan+10))
        ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, Nchan+1, 64)))
        # ax1.set_ylabel('Counts [$\\times 2^{10}$]')
        # ax1.set_ylim((y_range[0]-2, y_range[-1]+2))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(y_range[0], y_range[-1]+1, 16)))
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
        ax2.xaxis.set_major_locator(FixedLocator(np.arange(0, Nchan+1, 64)))
        # ax2.set_ylim((y_range[0]-2, y_range[-1]+2))
        # ax2.yaxis.set_major_locator(FixedLocator(np.arange(y_range[0], y_range[-1]+1, 16)))
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
        ax3.xaxis.set_major_locator(FixedLocator(np.arange(0, Nchan+1, 64)))
        # ax3.set_ylim((y_range[0]-2, y_range[-1]+2))
        # ax3.yaxis.set_major_locator(FixedLocator(np.arange(y_range[0], y_range[-1]+1, 16)))
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
        ax4.xaxis.set_major_locator(FixedLocator(np.arange(0, Nchan+1, 64)))
        # ax4.set_ylim((y_range[0]-2, y_range[-1]+2))
        # ax4.yaxis.set_major_locator(FixedLocator(np.arange(y_range[0], y_range[-1]+1, 16)))
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
        # ax5.set_ylabel('Counts [$\\times 2^{10}$]')
        # ax5.set_ylim((y_range[0]-2, y_range[-1]+2))
        # ax5.yaxis.set_major_locator(FixedLocator(np.arange(y_range[0], y_range[-1]+1, 16)))
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



if __name__ == '__main__':

    fname = '/argus/field/magpie/2023/dhc6/processed/MAGPIE_SPN-S_2023-08-05_v1.h5'
    quicklook_bokeh_spns(fname, wvl0=None, tmhr0=None, tmhr_range=None, wvl_range=[350.0, 800.0], tmhr_step=10, wvl_step=4, description='MAGPIE', fname_html='test.html')
    pass
