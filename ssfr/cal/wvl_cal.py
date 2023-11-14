import os
import sys
import glob
import datetime
import numpy as np

import ssfr.common

__all__ = [
        'load_wvl_coef',
        'cal_wvl',
        'cal_wvl_coef',
        ]

# adapted from IDL code
# hg=[296.73,302.15,312.57,313.17,334.15,365.02,365.48,366.33,404.66,407.78,433.92,434.75,435.48,491.6,546.07,576.96,579.07,1014]
lamps_idl = {
        'hg': np.array([
            296.73, 302.15, 312.57, 313.17,
            334.15, 365.02, 365.48, 366.33,
            404.66, 407.78, 433.92, 434.75,
            435.48, 491.60, 546.07, 576.96,
            579.07, 1014.0
            ])
        }

# asterisk wavelengths (no neighbouring lines within 1.0 nm) from lamp manual
lamps = {
        'hg': np.array([
            296.7283, 334.1484, 404.6565, 407.7837,
            435.8335, 546.0750, 576.9610, 579.0670,
            1013.979, 1128.741, 1357.021, 1367.351,
            1395.055, 1529.597
            ]),
        'kr': np.array([
            450.235,  605.611,  758.741,  760.154,
            805.950,  828.105,  850.887,  892.869,
            985.624, 1022.146, 1145.748, 1181.938,
           1363.422, 1442.679, 1473.444, 1523.962,
           1678.513, 1693.581, 1816.732, 2190.251
            ])
        }


def load_wvl_coef(fname='%s/wvl_coef.dat' % ssfr.common.fdir_data):

    with open(fname, 'r') as f:
        lines = f.readlines()

    coefs = {}
    for line_ in lines:
        line = line_.strip().replace(' ', '').replace('\n', '')
        if line[0] != '#':
            data  = line.split(',')
            vname = data[0]
            coef  = np.array([float(data0) for data0 in data[1:]])
            if vname not in coefs.keys():
                coefs[vname] = coef

    return coefs


def cal_wvl(coef, Nchan=256):

    xchan = np.arange(Nchan, dtype=np.float64)

    wvl = np.zeros_like(xchan)
    for i, coef0 in enumerate(coef):
        wvl += coef0 * xchan**i

    return wvl


def cal_wvl_coef(spectra, which_spec='lasp|ssfr-a|zen|si', which_lamp='hg'):


    Nchan = spectra.size
    xchan = np.arange(Nchan, dtype=np.float64)

    coefs = load_wvl_coef()
    coef_base = coefs[which_spec]
    wvl_base = cal_wvl(coef_base, Nchan=Nchan)

    wvl_lamp = lamps[which_lamp]

    # figure
    #/----------------------------------------------------------------------------\#
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

    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.plot(wvl_base, spectra, color='red', lw=1.0)
        for wvl0 in wvl_lamp:
            ax1.axvline(wvl0, color='green', lw=1.0)
        # cs = ax1.imshow(.T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        # ax1.scatter(x, y, s=6, c='k', lw=0.0)
        # ax1.hist(.ravel(), bins=100, histtype='stepfilled', alpha=0.5, color='black')
        # ax1.plot([0, 1], [0, 1], color='k', ls='--')
        # ax1.set_xlim(())
        # ax1.set_ylim(())
        # ax1.set_xlabel('')
        # ax1.set_ylabel('')
        # ax1.set_title('')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        # fig.subplots_adjust(hspace=0.3, wspace=0.3)
        # _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        # fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
        sys.exit()
    #\----------------------------------------------------------------------------/#



if __name__ == '__main__':


    pass
