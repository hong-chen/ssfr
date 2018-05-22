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
import matplotlib.patches as patches
import cartopy.crs as ccrs


def CDATA_DIFFUSE_RATIO(date, altitude=np.arange(0.0, 10.1, 0.1), solar_zenith_angle=np.arange(0.0, 90.1, 0.1), wavelength=np.arange(300.0, 4001.0, 50.0)):

    from lrt_util import lrt_cfg, cld_cfg, aer_cfg
    from lrt_util import LRT_V2_INIT, LRT_RUN_MP, LRT_READ_UVSPEC

    for altitude0 in [7.0]:
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        for solar_zenith_angle0 in [0.0, 40.0, 80.0, 89.0]:
            inits = []
            for wavelength0 in wavelength:

                input_file  = 'data/LRT_input_%4.4d.txt' % wavelength0
                output_file = 'data/LRT_output_%4.4d.txt' % wavelength0
                init = LRT_V2_INIT(input_file=input_file, output_file=output_file, date=date, surface_albedo=0.03, solar_zenith_angle=solar_zenith_angle0, wavelength=wavelength0, output_altitude=7.0, lrt_cfg=lrt_cfg, cld_cfg=None, aer_cfg=None)

                inits.append(init)

            LRT_RUN_MP(inits)
            data = LRT_READ_UVSPEC(inits)

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            ax1.plot(wavelength, data.f_down_diffuse/data.f_down, label='SZA=%.1f' % (solar_zenith_angle0))
            # ax1.set_xlim(())
        ax1.set_ylim((0.0, 1.0))
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Diffuse/Total Ratio')
        ax1.legend(loc='upper right', fontsize=16, framealpha=0.4)
        plt.savefig('test.png')
        plt.show()
        exit()
            # ---------------------------------------------------------------------





if __name__ == '__main__':

    date = datetime.datetime(2017, 8, 13)
    CDATA_DIFFUSE_RATIO(date)
