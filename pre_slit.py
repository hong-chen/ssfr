import os
import sys
import glob
import datetime
import multiprocessing as mp
import h5py
import numpy as np
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from scipy.io import readsav
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import cartopy.crs as ccrs

def VIS_SLIT_FUNC():

    fname = '/Users/hoch4240/Chen/other/data/aux_ssfr/vis_1nm_s.dat'
    data_10 = np.loadtxt(fname)

    fname = '/Users/hoch4240/Chen/other/data/aux_ssfr/vis_0.1nm_s.dat'
    data_01 = np.loadtxt(fname)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)

    data_x = data_10[:, 0]
    data_y = data_10[:, 1]
    data_xx = np.linspace(-8.0, 8.0, 10000)
    data_yy = np.interp(data_xx, data_x, data_y)

    spline = UnivariateSpline(data_xx, data_yy-data_yy.max()/2.0, s=0)
    r1, r2 = spline.roots()

    # xx, yy = CAL_SLIT_FUNC(6.0, xx=np.arange(-8.0, 8.01, 1.0))
    # for i, xx0 in enumerate(xx):
    #     print('%5.1f  %.16f' % (xx0, yy[i]))
    # exit()

    ax1.plot(data_x, data_y, lw=2.0, color='k', marker='o', markersize=5, alpha=0.6, label='1.0 nm')
    ax1.plot([r1, r2], [data_yy.max()/2.0, data_y.max()/2.0], ls='--', color='k', alpha=0.6, lw=2.0)
    ax1.plot([r1, r1], [0.0, data_y.max()/2.0], ls='--', color='k', alpha=0.6, lw=2.0)
    ax1.plot([r2, r2], [0.0, data_y.max()/2.0], ls='--', color='k', alpha=0.6, lw=2.0)

    data_x = data_01[:, 0]
    data_y = data_01[:, 1]
    data_xx = np.linspace(-8.0, 8.0, 10000)
    data_yy = np.interp(data_xx, data_x, data_y)

    spline = UnivariateSpline(data_xx, data_yy-data_yy.max()/2.0, s=0)
    r1, r2 = spline.roots()
    ax1.plot(data_x, data_y, lw=2.0, color='r', marker='o', markersize=5, alpha=0.6, label='0.1 nm')
    ax1.plot([r1, r2], [data_yy.max()/2.0, data_y.max()/2.0], ls='--', color='r', alpha=0.6, lw=2.0)
    ax1.plot([r1, r1], [0.0, data_y.max()/2.0], ls='--', color='r', alpha=0.6, lw=2.0)
    ax1.plot([r2, r2], [0.0, data_y.max()/2.0], ls='--', color='r', alpha=0.6, lw=2.0)

    ax1.set_xlim((-8.0, 8.0))
    ax1.set_ylim((0.0, 1.2))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Slit Function')
    ax1.legend(loc='upper right', fontsize=14, framealpha=0.4)
    ax1.set_title('Slit Function (VIS)')
    plt.savefig('vis_slit_func.png')
    plt.show()
    exit()
    # ---------------------------------------------------------------------

def NIR_SLIT_FUNC():

    fname = '/Users/hoch4240/Chen/other/data/aux_ssfr/nir_1nm_s.dat'
    data_10 = np.loadtxt(fname)

    fname = '/Users/hoch4240/Chen/other/data/aux_ssfr/nir_0.1nm_s.dat'
    data_01 = np.loadtxt(fname)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)

    data_x = data_10[:, 0]
    data_y = data_10[:, 1]
    data_xx = np.linspace(-16.0, 16.0, 10000)

    data_yy = np.interp(data_xx, data_x, data_y)

    spline = UnivariateSpline(data_xx, data_yy-data_yy.max()/2.0, s=0)
    r1, r2 = spline.roots()

    # xx, yy = CAL_SLIT_FUNC(12.0, xx=np.arange(-16.0, 16.01, 1.0))
    # for i, xx0 in enumerate(xx):
    #     print('%5.1f  %.16f' % (xx0, yy[i]))
    # exit()

    ax1.plot(data_x, data_y, lw=2.0, color='k', marker='o', markersize=5, alpha=0.6, label='1.0 nm')
    ax1.plot([r1, r2], [data_yy.max()/2.0, data_y.max()/2.0], ls='--', color='k', alpha=0.6, lw=2.0)
    ax1.plot([r1, r1], [0.0, data_y.max()/2.0], ls='--', color='k', alpha=0.6, lw=2.0)
    ax1.plot([r2, r2], [0.0, data_y.max()/2.0], ls='--', color='k', alpha=0.6, lw=2.0)

    data_x = data_01[:, 0]
    data_y = data_01[:, 1]
    data_xx = np.linspace(-16.0, 16.0, 10000)
    data_yy = np.interp(data_xx, data_x, data_y)

    spline = UnivariateSpline(data_xx, data_yy-data_yy.max()/2.0, s=0)
    r1, r2 = spline.roots()
    print(r1, r2)
    ax1.plot(data_x, data_y, lw=2.0, color='r', marker='o', markersize=5, alpha=0.6, label='0.1 nm')
    ax1.plot([r1, r2], [data_yy.max()/2.0, data_y.max()/2.0], ls='--', color='r', alpha=0.6, lw=2.0)
    ax1.plot([r1, r1], [0.0, data_y.max()/2.0], ls='--', color='r', alpha=0.6, lw=2.0)
    ax1.plot([r2, r2], [0.0, data_y.max()/2.0], ls='--', color='r', alpha=0.6, lw=2.0)

    ax1.set_xlim((-16.0, 16.0))
    ax1.set_ylim((0.0, 1.2))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Slit Function')
    ax1.legend(loc='upper right', fontsize=14, framealpha=0.4)
    ax1.set_title('Slit Function (NIR)')
    plt.savefig('nir_slit_func.png')
    plt.show()
    exit()
    # ---------------------------------------------------------------------

def CAL_SLIT_FUNC(FWHM, xx=np.arange(-16.0, 16.1, 0.1), x0=0.0):

    sigma = FWHM / (2.0*np.sqrt(2.0*np.log(2.0)))

    yy = np.exp(-(xx-x0)**2/(2.0*sigma**2))

    return xx, yy


if __name__ == '__main__':

    VIS_SLIT_FUNC()
    # NIR_SLIT_FUNC()
