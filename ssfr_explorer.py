# An interactive data analysis tool for SSFR
#
# by Hong Chen (me@hongchen.cz)

import os
import sys
import glob
import datetime
import multiprocessing as mp
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav

from bokeh.layouts import row, column, widgetbox, layout
from bokeh.resources import CDN
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Select, Slider, CheckboxGroup
from bokeh.models import Toggle, BoxAnnotation, CustomJS, Legend, Span
from bokeh.plotting import figure
from bokeh.embed import file_html




def PREP_SSFR_DATA(tmhr, wvl, data, tag='ssfr'):

    data_tmhr = {}
    data_tmhr['tmhr'] = tmhr

    wvl0 = 600.0
    ind_wvl0 = np.argmin(np.abs(wvl-wvl0))
    data_tmhr['plot_%s' % tag] = data[:, ind_wvl0]

    for i in range(wvl.size):
        data_tmhr['%s_%d' % (tag, i)] = data[:, i]

    data_wvl = {}
    data_wvl['wvl']  = wvl

    ind_tmhr0 = 0
    data_wvl['plot_%s' % tag] = data[ind_tmhr0, :]

    for i in range(tmhr.size):
        data_wvl['%s_%d' % (tag, i)] = data[i, :]

    return data_tmhr, data_wvl






def SSFR_QUICKLOOK(data_tmhr0, data_wvl0, tags, ylabel='Irradiance', fname_out='ssfr.html'):

    colors = ['black', 'red', 'blue', 'green', 'magenta', 'cyan', 'greenyellow', 'wheat']

    # SSFR Time Series
    # +
    data_tmhr  = ColumnDataSource(data=data_tmhr0)

    tmhr_x_start  = data_tmhr0['tmhr'].min()
    tmhr_x_end    = data_tmhr0['tmhr'].max()

    widget_tmhr = Slider(start=tmhr_x_start, end=tmhr_x_end, value=tmhr_x_start, step=0.0001, width=800, title="Time [Hour]", format="0[.]0000")

    plt_tmhr    = figure(plot_height=300, plot_width=800, title='SSFR Time Series',
                  tools="reset,save,box_zoom,ywheel_zoom", active_scroll="ywheel_zoom", x_axis_label='Time [Hour]', y_axis_label=ylabel,
                  x_range=[tmhr_x_start, tmhr_x_end], y_range=[0.0, 2.0], output_backend="webgl")

    plt_tmhr.title.text_font_size = "1.3em"
    plt_tmhr.title.align          = "center"
    plt_tmhr.xaxis.axis_label_text_font_style = "normal"
    plt_tmhr.yaxis.axis_label_text_font_style = "normal"
    plt_tmhr.xaxis.axis_label_text_font_size  = "1.0em"
    plt_tmhr.xaxis.major_label_text_font_size = "1.0em"
    plt_tmhr.yaxis.axis_label_text_font_size  = "1.0em"
    plt_tmhr.yaxis.major_label_text_font_size = "1.0em"

    tmhr_line = Span(location=widget_tmhr.value, dimension='height', line_color='gray', line_dash='dashed', line_width=2)
    plt_tmhr.add_layout(tmhr_line)

    for i, tag in enumerate(tags):
        index = i % len(colors)
        plt_tmhr.circle('tmhr', 'plot_%s' % tag, source=data_tmhr, color=colors[index] , size=3, legend=tag)

    plt_tmhr.legend.location     = 'top_right'
    plt_tmhr.legend.background_fill_color     = 'gray'
    plt_tmhr.legend.background_fill_alpha     = 0.2
    plt_tmhr.legend.click_policy = 'hide'
    # -


    # SSFR Spectra
    # +
    data_wvl   = ColumnDataSource(data=data_wvl0)

    wvl_x_start  = data_wvl0['wvl'].min()
    wvl_x_end    = data_wvl0['wvl'].max()

    widget_wvl  = Slider(start=wvl_x_start, end=wvl_x_end, value=600, step=1, width=800, title="Wavelength [nm]")

    plt_wvl =     figure(plot_height=300, plot_width=800, title='SSFR Spectra',
                  tools="reset,save,box_zoom,ywheel_zoom", active_scroll="ywheel_zoom", x_axis_label='Wavelength [nm]', y_axis_label=ylabel,
                  x_range=[wvl_x_start, wvl_x_end], y_range=[0.0, 2.0], output_backend="webgl")

    plt_wvl.title.text_font_size = "1.3em"
    plt_wvl.title.align          = "center"
    plt_wvl.xaxis.axis_label_text_font_style = "normal"
    plt_wvl.yaxis.axis_label_text_font_style = "normal"
    plt_wvl.xaxis.axis_label_text_font_size  = "1.0em"
    plt_wvl.xaxis.major_label_text_font_size = "1.0em"
    plt_wvl.yaxis.axis_label_text_font_size  = "1.0em"
    plt_wvl.yaxis.major_label_text_font_size = "1.0em"

    wvl_line = Span(location=widget_wvl.value, dimension='height', line_color='gray', line_dash='dashed', line_width=2)
    plt_wvl.add_layout(wvl_line)

    for i, tag in enumerate(tags):
        index = i % len(colors)
        plt_wvl.circle('wvl', 'plot_%s' % tag, source=data_wvl, color=colors[index] , size=3, legend=tag)

    plt_wvl.legend.location                  = 'top_right'
    plt_wvl.legend.background_fill_color     = 'gray'
    plt_wvl.legend.background_fill_alpha     = 0.2
    plt_wvl.legend.click_policy              = 'hide'
    # -


    # tags
    # +
    tags_dict = {}
    tags_dict['tags'] = tags
    data_tags  = ColumnDataSource(data=tags_dict)
    # -


    # Javascript
    # +
    widget_wvl.callback = CustomJS(args=dict(plt=plt_tmhr, span=wvl_line, slider=widget_wvl, source1=data_tmhr, source2=data_wvl, source3=data_tags), code="""
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

var data1 = source1.data;
var data2 = source2.data;
var data3 = source3.data;

var x = data2['wvl'];
var tags = data3['tags'];

var index = closest(slider.value, x);
var index_s = index.toString();

var title = 'SSFR Time Series at ' + x[index].toFixed(2).toString() + ' nm';

for (i = 0; i < tags.length; i++) {
    var tag = tags[i];

    var plt_name = 'plot_' + tag;
    var y = data1[plt_name];

    var tmp_name = tag + '_' + index_s;
    var tmp = data1[tmp_name];
    for (j = 0; j < y.length; j++) {
        y[j] = tmp[j];
    }
}

span.location = slider.value;
plt.title.text = title;
source1.change.emit();
    """)



    widget_tmhr.callback = CustomJS(args=dict(plt=plt_wvl, span=tmhr_line, slider=widget_tmhr, source1=data_tmhr, source2=data_wvl, source3=data_tags), code="""
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

var data1 = source1.data;
var data2 = source2.data;
var data3 = source3.data;

var x = data1['tmhr'];
var tags = data3['tags'];

var index = closest(slider.value, x);
var index_s = index.toString();

var title = 'SSFR Spectra at ' + x[index].toFixed(4).toString() + ' UTC';

for (i = 0; i < tags.length; i++) {
    var tag = tags[i];

    var plt_name = 'plot_' + tag;
    var y = data2[plt_name];

    var tmp_name = tag + '_' + index_s;
    var tmp = data2[tmp_name];
    for (j = 0; j < y.length; j++) {
        y[j] = tmp[j];
    }
}

plt.title.text = title;
span.location = slider.value;
source2.change.emit();
    """)
    # -

    # save html file
    # +
    layout = column(plt_tmhr, widgetbox(widget_wvl), widgetbox(widget_tmhr), plt_wvl)
    html = file_html(layout, CDN, 'SSFR Explorer')
    with open(fname_out, 'w') as f:
        f.write(html)
    # -






if __name__ == '__main__':

    f             = h5py.File('data/all.h5', 'r')
    tmhr          = f['tmhr'][...]
    wvl           = f['wvl_ssfr'][...]
    f_dn_lrt_01_k = f['f_dn_lrt_01_k'][...]
    f_dn_lrt_10_k = f['f_dn_lrt_10_k'][...]
    f_dn_lrt_10_a = f['f_dn_lrt_10_a'][...]
    f.close()

    f = h5py.File('data/att_corr.h5', 'r')
    f_dn_ssfr_20170505_759 = f['f_dn_ssfr_20170505_759'][...]
    f_dn_ssfr_20171013_506C= f['f_dn_ssfr_20171013_506C'][...]
    f_dn_ssfr_20171016_1324= f['f_dn_ssfr_20171016_1324'][...]
    f_dn_ssfr_20171102_1324= f['f_dn_ssfr_20171102_1324'][...]
    f_dn_ssfr_20171106_1324= f['f_dn_ssfr_20171106_1324'][...]
    f.close()

    tags = ['20170505 759', '20171013 506C', '20171016 1324', '20171102 1324', '20171106 1324', 'LRT Kurudz 0.1', 'LRT Kurudz 1.0', 'LRT ATLAS+MODTRAN']

    data_tmhr0, data_wvl0 = PREP_SSFR_DATA(tmhr, wvl, f_dn_ssfr_20170505_759, tag=tags[0])
    data_tmhr1, data_wvl1 = PREP_SSFR_DATA(tmhr, wvl, f_dn_ssfr_20171013_506C, tag=tags[1])
    data_tmhr2, data_wvl2 = PREP_SSFR_DATA(tmhr, wvl, f_dn_ssfr_20171016_1324, tag=tags[2])
    data_tmhr3, data_wvl3 = PREP_SSFR_DATA(tmhr, wvl, f_dn_ssfr_20171102_1324, tag=tags[3])
    data_tmhr4, data_wvl4 = PREP_SSFR_DATA(tmhr, wvl, f_dn_ssfr_20171106_1324, tag=tags[4])

    data_tmhr5, data_wvl5 = PREP_SSFR_DATA(tmhr, wvl, f_dn_lrt_01_k, tag=tags[5])
    data_tmhr6, data_wvl6 = PREP_SSFR_DATA(tmhr, wvl, f_dn_lrt_10_k, tag=tags[6])
    data_tmhr7, data_wvl7 = PREP_SSFR_DATA(tmhr, wvl, f_dn_lrt_10_a, tag=tags[7])

    data_tmhr = {**data_tmhr0, **data_tmhr1, **data_tmhr2, **data_tmhr3, **data_tmhr4, **data_tmhr5, **data_tmhr6, **data_tmhr7}
    data_wvl  = {**data_wvl0 , **data_wvl1 , **data_wvl2 , **data_wvl3 , **data_wvl4 , **data_wvl5 , **data_wvl6 , **data_wvl7}

    SSFR_QUICKLOOK(data_tmhr, data_wvl, tags=tags, ylabel='Irradiance', fname_out='ssfr_lrt_flux_comp.html')
