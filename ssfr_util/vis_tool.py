'''
Interactive vew of SSFR
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



def PREP_DATA_2D(data_list):

    """
    data_dict: contains data information
    e.g.
    [
    {'name':'zenith', 'data':zen_spec, 'x':tmhr, 'y':wvl},
    {'name':'nadir' , 'data':nad_spec, 'x':tmhr, 'y':wvl}
    ]

    where data_list itself is a Python list that contains dictionaries of data
    For the above example,
    'name': string type
    'data': numpy.ndarray, e.g., zen_spec is an array contains spectral irradiance of SSFR
    'x'   : numpy.ndarray, first dimension of 'data', e.g., array of time in hour
    'y'   : numpy.ndarray, second dimension of 'data', e.g., array of wavelength

    One limitation is that the data should have the same dimenion
    e.g., zen_spec has dimension of [1000, 400], nad_spec will also need to have dimension of [1000, 400]
    """

    data_1 = {}
    data_2 = {}
    nml    = []

    for data_dict in data_list:

        nml.append(data_dict['name'])

        data_1['%s_x' % data_dict['name']] = data_dict['x']
        data_1['%s_y' % data_dict['name']] = data_dict['data'][:, 0]
        for i in range(data_dict['y'].size):
            data_1['%s_%d' % (data_dict['name'], i)] = data_dict['data'][:, i]

        data_2['%s_x' % data_dict['name']] = data_dict['y']
        data_2['%s_y' % data_dict['name']] = data_dict['data'][0, :]
        for i in range(data_dict['x'].size):
            data_2['%s_%d' % (data_dict['name'], i)] = data_dict['data'][i, :]

    return data_1, data_2, nml



def GEN_QUICKLOOK_2D(data_list):

    config = {
            'y': {'name':'Rad./Irrad.'},
            1  : {'name':'Time', 'units':'Hour', 'title':'Time Series'},
            2  : {'name':'Wavelength', 'units':'nm', 'title':'SSFR Spectra'},
            }
    colors = ['red', 'blue']

    data_1, data_2, nml = PREP_DATA_2D(data_list)

    data_1c  = ColumnDataSource(data=data_1)
    data_2c  = ColumnDataSource(data=data_2)

    xs_1s = []
    xe_1s = []
    for vname in nml:
        xs_1s.append(0.1*(data_1['%s_x' % vname].min()//0.1 - 1))
        xe_1s.append(0.1*(data_1['%s_x' % vname].max()//0.1 + 1))
    xs_1 = min(xs_1s)
    xe_1 = max(xe_1s)

    w_1 = Slider(start=xs_1, end=xe_1, value=xs_1, step=0.0001, width=800, title="%s [%s]" % (config[1]['name'], config[1]['units']), format="0[.]0000")

    p_1 = figure(plot_height=300, plot_width=800, title=config[1]['title'],
                 tools="reset,save,box_zoom,ywheel_zoom", active_scroll="ywheel_zoom", x_axis_label='%s [%s]' % (config[1]['name'], config[1]['units']), \
                 y_axis_label=config['y']['name'], x_range=[xs_1, xe_1], y_range=[0.0, 1.4], output_backend="webgl")
    p_1.title.text_font_size             = "1.3em"
    p_1.title.align                      = "center"
    p_1.xaxis.axis_label_text_font_style = "normal"
    p_1.yaxis.axis_label_text_font_style = "normal"
    p_1.xaxis.axis_label_text_font_size  = "1.0em"
    p_1.xaxis.major_label_text_font_size = "1.0em"
    p_1.yaxis.axis_label_text_font_size  = "1.0em"
    p_1.yaxis.major_label_text_font_size = "1.0em"

    c_line_1 = Span(location=w_1.value, dimension='height', line_color='gray', line_dash='dashed', line_width=2)
    p_1.add_layout(c_line_1)

    for i, vname in enumerate(nml):
        p_1.circle('%s_x' % vname, '%s_y' % vname, source=data_1c, color=colors[i], size=3, legend=vname)
    p_1.legend.location = 'top_right'



    w_2 = Slider(start=300, end=2200, value=600, step=1, width=800, title="%s [%s]" % (config[2]['name'], config[2]['units']))

    p_2 = figure(plot_height=300, plot_width=800, title=config[2]['title'],
                 tools="reset,save,box_zoom,ywheel_zoom", active_scroll="ywheel_zoom", x_axis_label='%s [%s]' % (config[2]['name'], config[2]['units']), \
                 y_axis_label=config['y']['name'], x_range=[300, 2200], y_range=[0.0, 1.4], output_backend="webgl")

    p_2.title.text_font_size             = "1.3em"
    p_2.title.align                      = "center"
    p_2.xaxis.axis_label_text_font_style = "normal"
    p_2.yaxis.axis_label_text_font_style = "normal"
    p_2.xaxis.axis_label_text_font_size  = "1.0em"
    p_2.xaxis.major_label_text_font_size = "1.0em"
    p_2.yaxis.axis_label_text_font_size  = "1.0em"
    p_2.yaxis.major_label_text_font_size = "1.0em"

    c_line_2 = Span(location=w_2.value, dimension='height', line_color='gray', line_dash='dashed', line_width=2)
    p_2.add_layout(c_line_2)

    for i, vname in enumerate(nml):
        p_2.circle('%s_x' % vname, '%s_y' % vname, source=data_2c, color=colors[i], size=3, legend=vname)
    p_2.legend.location = 'top_right'

    layout = column(p_1, widgetbox(w_2), widgetbox(w_1), p_2)
    # curdoc().add_root(layout)
    # curdoc().title = "SSFR Quick Look"
    html = file_html(layout, CDN, "quicklook")
    print(html)
    exit()


    w_2.callback = CustomJS(args=dict(plt=p_1, span=c_line_2, slider=w_2, source1=data_1c, source2=data_2c, vnames=nml), code="""

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
for (i=0; i < nml.length; i++) {
    var vname = vnames[i];
    var x = data1[vname+'_x'];
    var y = data1[vname+'_y'];
}

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

    w_1.callback = CustomJS(args=dict(plt=p_2, span=c_line_1, slider=w_1, source1=data_1c, source2=data_2c), code="""
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
var x = data1['x'];

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

    layout = column(plt_time, widgetbox(w_y), widgetbox(w_x), plt_y)
    # curdoc().add_root(layout)
    # curdoc().title = "SSFR Quick Look"
    html = file_html(layout, CDN, "%s on %s" % (ssfr, date))
    print(html)






if __name__ == '__main__':

    fname       = 'test/20180430_Belana.h5'
    f           = h5py.File(fname, 'r')
    tmhr        = f['tmhr'][...]
    wvl_zen     = f['wvl_zen'][...]
    wvl_nad     = f['wvl_nad'][...]
    temp        = f['temp'][...]
    spectra_zen =  f['spectra_zen'][...]
    spectra_nad =  f['spectra_nad'][...]
    f.close()

    data_list = [
            {'name':'zen' , 'data':spectra_zen, 'x':tmhr, 'y':wvl_zen},
            {'name':'nad' , 'data':spectra_nad, 'x':tmhr, 'y':wvl_nad}
            ]

    # data_x, data_y = PREP_DATA_2D(data_list)

    GEN_QUICKLOOK_2D(data_list)
