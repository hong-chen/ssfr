'''
Interactive Above Cloud Flux Study

run command on a terminal:
$ bokeh serve ssfr_explorer.py

open a browser and type the following link:
http://localhost:5006/ssfr_explorer
'''

import os
import sys
import glob
import datetime
import multiprocessing as mp
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav

from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox, column, layout
from bokeh.resources import CDN
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Select, Slider, CheckboxGroup
from bokeh.models import Toggle, BoxAnnotation, CustomJS, Legend, Span
from bokeh.plotting import figure
from bokeh.embed import file_html






def PREP_DATA(fname):

    f           = h5py.File(fname, 'r')
    tmhr        = f['tmhr'][...]
    wvl_zen     = f['wvl_zen'][...]
    wvl_nad     = f['wvl_nad'][...]
    temp        = f['temp'][...]
    spectra_zen =  f['spectra_zen'][...]
    spectra_nad =  f['spectra_nad'][...]
    f.close()

    data_time = {}
    data_spec = {}

    data_time['tmhr'] = tmhr
    data_time['plot_zen'] = np.repeat(np.nan, tmhr.size)
    data_time['plot_nad'] = np.repeat(np.nan, tmhr.size)
    for i in range(wvl_zen.size):
        data_time['zen%d' % i] = spectra_zen[:, i]
    for i in range(wvl_nad.size):
        data_time['nad%d' % i] = spectra_nad[:, i]
    for i in range(temp.shape[1]):
        data_time['temp%d' % i] = temp[:, i]

    data_spec['wvl_zen'] = wvl_zen
    data_spec['wvl_nad'] = wvl_nad
    # data_spec['plot_zen'] = np.repeat(np.nan, wvl_zen.size)
    # data_spec['plot_nad'] = np.repeat(np.nan, wvl_nad.size)
    data_spec['plot_zen'] = spectra_zen[0, :]
    data_spec['plot_nad'] = spectra_nad[0, :]
    for i in range(tmhr.size):
        data_spec['zen%d' % i] = spectra_zen[i, :]
        data_spec['nad%d' % i] = spectra_nad[i, :]

    return data_time, data_spec






def GEN_QUICKLOOK(data_time_dict, data_spec_dict, ssfr='Alvin', date='20180503'):

    data_time  = ColumnDataSource(data=data_time_dict)
    data_spec  = ColumnDataSource(data=data_spec_dict)

    t_x_s = 0.1 * (data_time_dict['tmhr'].min()//0.1 - 1)
    t_x_e = 0.1 * (data_time_dict['tmhr'].max()//0.1 + 1)

    w_time = Slider(start=t_x_s, end=t_x_e, value=t_x_s, step=0.0001, width=800, title="Time [Hour]", format="0[.]0000")
    w_spec = Slider(start=300, end=2200, value=600, step=1, width=800, title="Wavelength [nm]")

    plt_time    = figure(plot_height=300, plot_width=800, title='Time Series',
                  tools="reset,save,box_zoom,ywheel_zoom", active_scroll="ywheel_zoom", x_axis_label='Time [Hour]', y_axis_label='Rad./Irrad.',
                  x_range=[t_x_s, t_x_e], y_range=[0.0, 1.4], output_backend="webgl")

    plt_time.title.text_font_size = "1.3em"
    plt_time.title.align          = "center"
    plt_time.xaxis.axis_label_text_font_style = "normal"
    plt_time.yaxis.axis_label_text_font_style = "normal"
    plt_time.xaxis.axis_label_text_font_size  = "1.0em"
    plt_time.xaxis.major_label_text_font_size = "1.0em"
    plt_time.yaxis.axis_label_text_font_size  = "1.0em"
    plt_time.yaxis.major_label_text_font_size = "1.0em"

    c_tline = Span(location=w_time.value, dimension='height', line_color='gray', line_dash='dashed', line_width=2)
    plt_time.add_layout(c_tline)

    plt_time.circle('tmhr', 'plot_zen', source=data_time, color='red' , size=3, legend='Zenith')
    plt_time.circle('tmhr', 'plot_nad', source=data_time, color='blue', size=3, legend='Nadir')
    plt_time.legend.location = 'top_right'


    plt_spec= figure(plot_height=300, plot_width=800, title='SSFR Spectra',
                  tools="reset,save,box_zoom,ywheel_zoom", active_scroll="ywheel_zoom", x_axis_label='Wavelength [nm]', y_axis_label='Rad./Irrad.',
                  x_range=[300, 2200], y_range=[0.0, 1.4], output_backend="webgl")

    plt_spec.title.text_font_size = "1.3em"
    plt_spec.title.align     = "center"
    plt_spec.xaxis.axis_label_text_font_style = "normal"
    plt_spec.yaxis.axis_label_text_font_style = "normal"
    plt_spec.xaxis.axis_label_text_font_size  = "1.0em"
    plt_spec.xaxis.major_label_text_font_size = "1.0em"
    plt_spec.yaxis.axis_label_text_font_size  = "1.0em"
    plt_spec.yaxis.major_label_text_font_size = "1.0em"

    c_sline = Span(location=w_spec.value, dimension='height', line_color='gray', line_dash='dashed', line_width=2)
    plt_spec.add_layout(c_sline)

    plt_spec.circle('wvl_zen', 'plot_zen', source=data_spec, color='red' , size=3, legend='Zenith')
    plt_spec.circle('wvl_nad', 'plot_nad', source=data_spec, color='blue', size=3, legend='Nadir')
    plt_spec.legend.location = 'top_right'

    w_spec.callback = CustomJS(args=dict(plt=plt_time, span=c_sline, slider=w_spec, source1=data_time, source2=data_spec), code="""
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
var x = data1['tmhr'];
y1 = data1['plot_zen'];
y2 = data1['plot_nad'];

var data2 = source2.data;
var wvl = data2['wvl_zen'];
var index = closest(slider.value, wvl);
var index_s = index.toString();
var v1 = 'zen' + index_s;
var v2 = 'nad' + index_s;
var title = 'Time Series at ' + wvl[index].toFixed(2).toString() + ' nm';
var tmp1 = data1[v1];
var tmp2 = data1[v2];
for (i = 0; i < x.length; i++) {
    y1[i] = tmp1[i];
    y2[i] = tmp2[i];
}
source1.change.emit();
span.location = slider.value;
plt.title.text = title;
    """)

    w_time.callback = CustomJS(args=dict(plt=plt_spec, span=c_tline, slider=w_time, source1=data_time, source2=data_spec), code="""
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
var x = data1['tmhr'];

var data2 = source2.data;
y1 = data2['plot_zen'];
y2 = data2['plot_nad'];

var index = closest(slider.value, x);
var index_s = index.toString();
var v1 = 'zen' + index_s;
var v2 = 'nad' + index_s;
var tmp1 = data2[v1];
var tmp2 = data2[v2];
var title = 'SSFR Spectra at ' + x[index].toFixed(4).toString() + ' UTC';
for (i = 0; i < y1.length; i++) {
    y1[i] = tmp1[i];
    y2[i] = tmp2[i];
}
source2.change.emit();
plt.title.text = title;
span.location = slider.value;
    """)

    layout = column(plt_time, widgetbox(w_spec), widgetbox(w_time), plt_spec)
    # curdoc().add_root(layout)
    # curdoc().title = "SSFR Quick Look"
    html = file_html(layout, CDN, "%s on %s" % (ssfr, date))
    print(html)






if __name__ == '__main__':
    # for date_s in ['20180429', '20180430']:
    #     for wvl in [600, 1600]:
    #         PLOT_TIME_SERIES(date_s, wvl=wvl)
    # PLOT_TIME_SERIES('20180429', wvl=450.0)
    data_time, data_spec = PREP_DATA('20180430_Belana.h5')
    GEN_QUICKLOOK(data_time, data_spec, ssfr='Belana', date='20180430')
