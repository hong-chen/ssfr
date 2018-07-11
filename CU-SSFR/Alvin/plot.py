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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import cartopy.crs as ccrs


from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox, column, layout
from bokeh.resources import CDN
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Select, Slider, CheckboxGroup
from bokeh.models import Toggle, BoxAnnotation, CustomJS, Legend
from bokeh.plotting import figure
from bokeh.embed import file_html


def READ_DATA(fname):

    data = {}
    f = h5py.File(fname, 'r')
    tmhr = f['tmhr'][...]
    tmhr[tmhr>24.0] -= 24.0
    data['tmhr']   = tmhr
    data['temp']   = f['temp'][...]
    data['spectra_zen']  = f['spectra_zen'][...]
    data['spectra_nad']  = f['spectra_nad'][...]
    data['wvl_zen'] = f['wvl_zen'][...]
    data['wvl_nad']  = f['wvl_nad'][...]
    f.close()
    return data


def GEN_HTML(data):

    data_source = ColumnDataSource(data=data)

    plt_time    = figure(plot_height=300, plot_width=1000,
                  tools="reset,save,box_zoom", y_axis_label='Radiance/Irradiance',
                  x_range=[data['tmhr'].min(), data['tmhr'].max()], y_range=[0.0, 1.5], output_backend="webgl")

    plt_time.xaxis.axis_label_text_font_style = "normal"
    plt_time.yaxis.axis_label_text_font_style = "normal"
    plt_time.xaxis.axis_label_text_font_size = "1.0em"
    plt_time.xaxis.major_label_text_font_size = "1.0em"
    plt_time.yaxis.axis_label_text_font_size = "1.0em"
    plt_time.yaxis.major_label_text_font_size = "1.0em"

    # c11 = .circle('tmhr', 'f_dn_bbr_int', source=data_source, color='blue', size=3, legend='BBR')
    # c13 = .circle('tmhr', 'f_dn_ssfr_int', source=data_source, color='red', size=3, legend='SSFR Corr.')
    # c14 = .circle('tmhr', 'f_dn_lrt_int', source=data_source, color='black', size=3, legend='MODIS Sim')
    # plt_time.legend.orientation = 'horizontal'
    # plt_time.legend.location = 'top_center'
    # plt_time.legend.spacing = 100
    # plt_time.legend.click_policy = "hide"
    # plt_time.legend.background_fill_alpha = 0.4
    # plt_time.legend.border_line_alpha = 0.4

    plt_spectra = figure(plot_height=600, plot_width=1000,
                  tools="reset,save,box_zoom", y_axis_label='Radiance/Irradiance',
                  x_range=[300, 2200], y_range=[0.0, 1.5], output_backend="webgl")

    plt_spectra.xaxis.axis_label_text_font_style = "normal"
    plt_spectra.yaxis.axis_label_text_font_style = "normal"
    plt_spectra.xaxis.axis_label_text_font_size = "1.0em"
    plt_spectra.xaxis.major_label_text_font_size = "1.0em"
    plt_spectra.yaxis.axis_label_text_font_size = "1.0em"
    plt_spectra.yaxis.major_label_text_font_size = "1.0em"
    # c21 = plt_spectra.circle('tmhr', 'f_up_bbr_int', source=data_source, color='blue', size=3, legend='BBR')
    # c22 = plt_spectra.circle('tmhr', 'f_up_ssfr_int', source=data_source, color='red', size=3, legend='SSFR*1.05')
    # c23 = plt_spectra.circle('tmhr', 'f_up_lrt_int', source=data_source, color='black', size=3, legend='MODIS Sim.')
    # plt_spectra.legend.orientation = 'horizontal'
    # plt_spectra.legend.location = 'top_center'
    # plt_spectra.legend.spacing = 100
    # plt_spectra.legend.click_policy = "hide"
    # plt_spectra.legend.background_fill_alpha = 0.4
    # plt_spectra.legend.border_line_alpha = 0.4

    # legend_items = [("SSFR", [c12, c22]), \
    #                 ("BBR", [c11, c21]), \
    #                 ("MODIS Sim.", [c13, c23]), \
    #                 ("MODIS COT", [c31])]
    # legend = Legend(items=legend_items, location=(0, 0))
    # legend.orientation = "horizontal"
    # legend.label_standoff = 5
    # legend.spacing = 10
    # legend.click_policy = "hide"
    # legend.glyph_width = 800
    # legend.label_text_font_size = '1.0em'
    # p_cot.add_layout(legend, 'below')

    layout = column(plt_time, plt_spectra)
    # curdoc().add_root(layout)
    # curdoc().title = "Validation - Above Clouds"
    html = file_html(layout, CDN, "SSFR")
    print(html)

    # Set up layouts and add to document
    # inputs = widgetbox(frac, scale)

    # whole = column(plot, inputs)


def PLOT_TIME_SERIES(date_s, wvl=600.0):

    a = READ_DATA('%s_Alvin.h5'  % date_s)
    b = READ_DATA('%s_Belana.h5' % date_s)

    index_zen_a = np.argmin(np.abs(wvl-a['wvl_zen']))
    index_nad_a = np.argmin(np.abs(wvl-a['wvl_nad']))
    index_zen_b = np.argmin(np.abs(wvl-b['wvl_zen']))
    index_nad_b = np.argmin(np.abs(wvl-b['wvl_nad']))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.scatter(a['tmhr'], a['spectra_zen'][:, index_zen_a], c='red'    , alpha=0.4, label='Alvin Zenith Irradiance')
    ax1.scatter(a['tmhr'], a['spectra_nad'][:, index_nad_a], c='blue'   , alpha=0.4, label='Alvin Nadir Irradiance')
    ax1.scatter(b['tmhr'], b['spectra_zen'][:, index_zen_b], c='black'  , alpha=0.4, label='Belana Zenith Radiance')
    ax1.scatter(b['tmhr'], b['spectra_nad'][:, index_nad_b], c='cyan'   , alpha=0.4, label='Belana Nadir Irradiance')
    ax1.set_xlabel('Time [hour]')
    ax1.set_ylabel('Radiance/Irradiance')
    ax1.set_title('%s Wavelength %.1f nm' % (date_s, wvl))
    if date_s == '20180429':
        ax1.set_xlim((0.0, 15.0))
    elif date_s == '20180430':
        ax1.set_xlim((3.5, 7.5))
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.4)
    plt.savefig('time_series_%s_%dnm.png' % (date_s, wvl))
    plt.show()
    # ---------------------------------------------------------------------





def PLOT_TIME_SERIES_TEMP(date_s, tag='Alvin'):

    a = READ_DATA('%s_%s.h5'  % (date_s, tag))

    indices = [0, 1, 2, 3, 5, 6, 7]
    temp_a = a['temp'][:, indices]
    labels = ['Ambient', 'InGaAs Body (Zen.)', 'InGaAs Body (Nad.)', 'Plate', 'InGaAs TEC (Zen.)', 'InGaAs TEC (Nad.)', 'PTC5K']
    colors = ['#e6194b', '#3cb44b', '#911eb4', '#0082c8', 'k', 'gray', '#f58231']

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(111)

    patches = []
    for i, label in enumerate(labels):
        ax1.scatter(a['tmhr'], temp_a[:, i], c=colors[i], alpha=0.4, marker='o', edgecolor='none')
        patches.append(mpatches.Patch(color=colors[i] , label=label))

    ax1.set_xlabel('Time [hour]')
    ax1.set_ylabel('Temperature [$\mathrm{^\circ C}$]')
    ax1.set_title('%s Temperature (%s)' % (date_s, tag), y=1.08)
    if date_s == '20180429':
        ax1.set_xlim((0.0, 15.0))
    elif date_s == '20180430':
        ax1.set_xlim((3.5, 7.5))
    ax1.set_ylim((-15, 30))
    ax1.legend(handles=patches, bbox_to_anchor=(0., 1.01, 1., .102), loc=3, ncol=len(patches), mode="expand", borderaxespad=0., frameon=False, handletextpad=0.2, fontsize=10)
    plt.savefig('time_series_%s_%s_temp.png' % (date_s, tag))
    # plt.show()
    # ---------------------------------------------------------------------



if __name__ == '__main__':
    # for date_s in ['20180429', '20180430']:
    for date_s in ['20180503']:
        for wvl in [500, 1000, 1300, 1600, 2000, 2100]:
            PLOT_TIME_SERIES(date_s, wvl=wvl)
    # PLOT_TIME_SERIES('20180429', wvl=450.0)
    # for date_s in ['20180429', '20180430']:
    # for date_s in ['20180503']:
    #     for tag in ['Alvin', 'Belana']:
    #         PLOT_TIME_SERIES_TEMP(date_s, tag=tag)
    # data = READ_DATA('20180430_Belana.h5')
    # GEN_HTML(data)
