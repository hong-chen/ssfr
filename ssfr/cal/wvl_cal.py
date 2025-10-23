import os
import sys
import glob
import datetime
import copy
import numpy as np
from scipy.optimize import curve_fit
import ssfr.common

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.image as mpl_img
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable



__all__ = [
        'get_wvl_coef',
        'cal_wvl',
        'cal_wvl_coef',
        'load_wvl_coef',
        'load_ils_nir',
        'cal_wvl_coef_two_lamps_InGaAs',
        'cal_wvl_coef_two_lamps_Si'
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

lamps_ssfr_fitting = {
        'hg': np.array([
            # 296.7283, 
            # 312.567, 313.155, 313.184,
            # 334.1484, 
            365.015,
            404.6565, 407.7837,
            435.8335, 546.0750, 
            576.9610, 579.0670,
            1013.979, 
            1128.741, 
            1357.021, 1367.351,
            1395.055, 
            1529.597,
            # 1707.279,

            ]),
        'kr': np.array([
            # 450.235,  
            557.029,
            587.092,
            # 605.611,  
            758.741,  760.154,
            785.482,
            # 805.950,
            810.436, 811.290,  
            828.105,  850.887,  
            877.675,
            892.869,
            975.176,
            # 985.624, 
            # 1022.146, 
            # 1145.748, 
            1181.938,
            1286.189, 1317.741,
            1363.422, 1442.679, 1473.444, 
            1523.962, 
            1533.496,
            1678.513, 1693.581, 
            1816.732, 
            # 2116.547, # only strong enough for ssrr
        #    2190.251, # only strong enough for ssrr
            ]) 
        }

lamps_ssrr_fitting = {
        'hg': np.array([
            # 296.7283, 
            # 312.567, 313.155, 313.184,
            # 334.1484, 
            365.015,
            404.6565, 407.7837,
            435.8335, 546.0750, 
            576.9610, 579.0670,
            1013.979, 
            1128.741, 
            1357.021, 1367.351,
            1395.055, 
            1529.597,
            # 1707.279,
            1813.038,  # from NIST
            1970.017,  # from NIST  

            ]),
        'kr': np.array([
            # 450.235,  
            557.029,
            587.092,
            # 605.611,  
            758.741,  760.154,
            785.482,
            # 805.950,
            810.436, 811.290,  
            828.105,  850.887,  
            877.675,
            892.869,
            975.176,
            # 985.624, 
            # 1022.146, 
            # 1145.748, 
            1181.938,
            1220.453, # from NIST 
            1286.189, 1317.741,
            1363.422, 1442.679, 1473.444, 
            1523.962, 
            1533.496,
            1678.513, 1693.581, 
            1816.732, 
            2116.547, # only strong enough for ssrr
            2190.251, # only strong enough for ssrr
            ])
        }


def load_wvl_coef(fname='%s/wvl/wvl_coef.dat' % ssfr.common.fdir_data):

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


def load_ils_nir(fname='%s/nir_1nm_s.dat' % ssfr.common.fdir_data, fine_mode=False):

    if fine_mode:
        fname = fname.replace('1nm', '0.1nm')
    with open(fname, 'r') as f:
        lines = f.readlines()

    dx_list, rel_int_list = [], []
    for line in lines:
        # line = line_.strip().replace(' ', '').replace('\n', '')

        dx_list.append(float(line[:5]))
        rel_int_list.append(float(line[7:]))

    return dx_list, rel_int_list


def get_wvl_coef(
        which_spec,
        fname='%s/wvl/wvl_coef.dat' % ssfr.common.fdir_data
        ):

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

    return coefs[which_spec]


def cal_wvl(coef, Nchan=256):

    xchan = np.arange(Nchan, dtype=np.float64)

    wvl = np.zeros_like(xchan)
    for i, coef0 in enumerate(coef):
        wvl += coef0 * xchan**i

    return wvl


def select_wvl_lamp(wvl, window=20.0):

    wvl = np.sort(wvl)

    wvl_select = np.array([])
    for i in range(wvl.size):
        if i == 0:
            if abs(wvl[i+1]-wvl[i]) > window:
                wvl_select = np.append(wvl_select, wvl[i])
        elif i == (wvl.size-1):
            if abs(wvl[i-1]-wvl[i]) > window:
                wvl_select = np.append(wvl_select, wvl[i])
        else:
            if (abs(wvl[i-1]-wvl[i]) > window) and (abs(wvl[i+1]-wvl[i]) > window):
                wvl_select = np.append(wvl_select, wvl[i])

    return wvl_select


def select_chan_num(wvl, spectra, wvl_search, window=20.0):

    # spectra = spectra / np.nanmax(spectra)
    # spectra[spectra<0.08] = np.nan

    Nchan = wvl.size
    xchan = np.arange(Nchan, dtype=np.float64)

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
    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        # cs = ax1.imshow(.T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
        # ax1.scatter(x, y, s=6, c='k', lw=0.0)
        # ax1.hist(.ravel(), bins=100, histtype='stepfilled', alpha=0.5, color='black')
        ax1.plot(wvl, spectra, color='b', marker='o', markersize=3)
        for wvl0 in wvl_search:
            # ax1.axvspan(wvl0-window, wvl0+window, color='red', lw=1.0)
            ax1.axvline(wvl0, color='red', lw=1.0)
        ax1.set_xlim((900, 2300))
        # ax1.set_ylim(())
        ax1.set_xlabel('Wavelength [nm]')
        # ax1.set_ylabel('')
        ax1.set_title('SSFR-A|ZEN|IN')
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


    chan_select = np.array([])






    return chan_select


def cal_wvl_coef(spectra, which_spec='lasp|ssfr-a|zen|si'):

    """
    input:
        spectra: Python dictionary, e.g.,
                 spectra = {
                            'hg': np.array([...]),
                            'kr': np.array([...]),
                           }
    """

    which_grating = which_spec.split('|')[-1]
    if which_grating == 'in':
        window = 40.0
    elif which_grating == 'si':
        window = 20.0

    wvl_lamp = np.array([])
    chan_num = np.array([])
    for lamp_tag in spectra.keys():

        # initial guess of the wavelength from the old coefficients
        #/----------------------------------------------------------------------------\#
        spectra0 = spectra[lamp_tag]
        wvl0     = cal_wvl(get_wvl_coef(which_spec), Nchan=spectra0.size)
        #\----------------------------------------------------------------------------/#

        # select lamp wavelength
        #/----------------------------------------------------------------------------\#
        lamp0    = lamps[lamp_tag]
        # wvl_lamp = np.append(wvl_lamp, select_wvl_lamp(lamp0, window=window))
        wvl_lamp = np.append(wvl_lamp, lamp0)
        #\----------------------------------------------------------------------------/#

        # retrieve ssfr channel numbers for selected lamp wavelength
        #/----------------------------------------------------------------------------\#
        chan_num = np.append(chan_num, select_chan_num(wvl0, spectra0, wvl_lamp, window=window))
        #\----------------------------------------------------------------------------/#

    sys.exit()

    xchan = np.arange(Nchan, dtype=np.float64)

    coef = get_wvl_coef(which_spec)
    wvl_base = cal_wvl(coef, Nchan=Nchan)

    # wvl_lamp = lamps[which_lamp]

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
        ax1.plot(wvl_base, spectra0, color='red', lw=1.0)
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


def cal_wvl_coef_two_lamps_InGaAs(spectrum_Hg, spectrum_Kr, which_spec='lasp|ssfr-a|zen|si'):

    spectrum_Hg[spectrum_Hg<0] = 0
    spectrum_Kr[spectrum_Kr<0] = 0
    Nchan = spectrum_Hg.size
    xchan = np.arange(Nchan, dtype=np.float64)

    coefs = load_wvl_coef()
    radiance = True if 'ssrr' in which_spec else False
    which_spec_write = which_spec
    which_spec = which_spec.replace('ssrr', 'ssfr')  # need to be fixed
    coef_base = coefs[which_spec]
    wvl_base = cal_wvl(coef_base, Nchan=Nchan)
    
    if radiance:
        lamps_fitting = lamps_ssrr_fitting
    else:
        lamps_fitting = lamps_ssfr_fitting

    lamp_Hg = lamps_fitting['hg']
    lamp_Kr = lamps_fitting['kr']
    if not radiance:
        # remove 2116.547
        lamps_fitting['kr'] = lamps_fitting['kr'][lamps_fitting['kr'] != 2116.547]
    lamp_Kr = lamps_fitting['kr']

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.plot(wvl_base, spectrum_Hg, color='red', lw=1.0, label='Hg spectrum (original coeff)')
        ax1.plot(wvl_base, spectrum_Kr, color='blue', lw=1.0, label='Kr spectrum (original coeff)')
        ymin, ymax= ax1.get_ylim()
        ax1.vlines(lamp_Hg, ymin=0, ymax=ymax, color='green', lw=1.0, label='Hg lines')
        ax1.vlines(lamp_Kr, ymin=0, ymax=ymax, color='purple', lw=1.0, label='Kr lines')
        ax1.set_xlim(900, 2200)
        ax1.set_xlabel('Wavelength (nm)', fontsize=14)
        ax1.set_ylabel('Counts', fontsize=14)
        ax1.set_title(which_spec_write, fontsize=18)
        ax1.legend()
        fig.savefig(f'output/wvl-cal/wvl_cal_{which_spec_write}_original_spectrum.png', bbox_inches='tight')
        # plt.show()
        plt.close('all')
        
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)


        if which_spec == 'lasp|ssfr-a|zen|in':
            center_shift = 3.9
            low_shit, high_shit = 3.2, 4.6
        elif which_spec == 'lasp|ssfr-a|nad|in':
            center_shift = 0.65
            low_shit, high_shit = 0.2, 1.1
        elif which_spec == 'lasp|ssfr-b|zen|in':
            center_shift = -1.1
            low_shit, high_shit = -1.8, -0.4
        elif which_spec == 'lasp|ssfr-b|nad|in':
            center_shift = 1.3
            low_shit, high_shit = 0.6, 2.0
        Hg_p0 = []
        Hg_p0_bound_low = []
        Hg_p0_bound_high = []
        for wvl0 in lamp_Hg[lamp_Hg>900]:
            if radiance:
                Hg_p0.extend([5000, lambda_to_p(wvl0)+center_shift, 1.25])
                Hg_p0_bound_low.extend([100, lambda_to_p(wvl0)+low_shit, 0.75])
                Hg_p0_bound_high.extend([60000, lambda_to_p(wvl0)+high_shit, 1.75])
            else:
                Hg_p0.extend([500, lambda_to_p(wvl0)+center_shift, 1.25])
                Hg_p0_bound_low.extend([1, lambda_to_p(wvl0)+low_shit, 0.75])
                Hg_p0_bound_high.extend([6000, lambda_to_p(wvl0)+high_shit, 1.75])
        
        Hg_coeff, var_matrix = curve_fit(gauss_set, np.arange(0, 256), spectrum_Hg, p0=Hg_p0, maxfev=50000, bounds=(Hg_p0_bound_low, Hg_p0_bound_high))
        ax1.vlines(lambda_to_p(lamp_Hg[lamp_Hg>900]), ymin=0, ymax=ymax, color='green', lw=1.0, label='Hg lines')
        ax1.vlines(Hg_coeff[1::3], ymin=0, ymax=ymax, color='green', lw=1.0, linestyle='--', label='Hg lines fitting')
        xx = np.arange(0, 256)
        ax1.plot(xx, spectrum_Hg, color='red', lw=1.0, label='Hg spectrum')
        ax1.plot(xx, gauss_set(xx, *Hg_coeff), color='g', linewidth=1.5)

        Kr_p0 = []
        Kr_p0_bound_low = []
        Kr_p0_bound_high = []
        

        if which_spec == 'lasp|ssfr-a|zen|in':
            center_shift = 3.9
            low_shit, high_shit = 3.2, 4.6
        elif which_spec == 'lasp|ssfr-a|nad|in':
            center_shift = 0.65
            low_shit, high_shit = 0.2, 1.1
        elif which_spec == 'lasp|ssfr-b|zen|in':
            center_shift = -1.1
            low_shit, high_shit = -1.8, -0.4
        elif which_spec == 'lasp|ssfr-b|nad|in':
            center_shift = 1.3
            low_shit, high_shit = 0.6, 2.0

        for wvl0 in lamp_Kr[lamp_Kr>900]:
            if radiance:
                if wvl0 != 2116.547:
                    Kr_p0.extend([1000, lambda_to_p(wvl0)+center_shift, 1.25])
                    Kr_p0_bound_low.extend([10, lambda_to_p(wvl0)+low_shit, 0.75])
                    Kr_p0_bound_high.extend([60000, lambda_to_p(wvl0)+high_shit, 1.75])
                else:
                    Kr_p0.extend([500, lambda_to_p(wvl0)+center_shift, 1.25])
                    Kr_p0_bound_low.extend([1, lambda_to_p(wvl0)+low_shit, 0.75])
                    Kr_p0_bound_high.extend([5000, lambda_to_p(wvl0)+high_shit, 1.75])
            else:
                if wvl0 != 2116.547:
                    Kr_p0.extend([100, lambda_to_p(wvl0)+center_shift, 1.25])
                    Kr_p0_bound_low.extend([10, lambda_to_p(wvl0)+low_shit, 0.75])
                    Kr_p0_bound_high.extend([6000, lambda_to_p(wvl0)+high_shit, 1.75])
                else:
                    Kr_p0.extend([50, lambda_to_p(wvl0)+center_shift, 1.25])
                    Kr_p0_bound_low.extend([1, lambda_to_p(wvl0)+low_shit, 0.75])
                    Kr_p0_bound_high.extend([500, lambda_to_p(wvl0)+high_shit, 1.75])

        Kr_coeff, var_matrix = curve_fit(gauss_set, np.arange(0, 256), spectrum_Kr, p0=Kr_p0, maxfev=50000, bounds=(Kr_p0_bound_low, Kr_p0_bound_high))
        ax1.vlines(lambda_to_p(lamp_Kr[lamp_Kr>900]), ymin=0, ymax=ymax, color='purple', lw=1.0, label='Kr lines')
        ax1.vlines(Kr_coeff[1::3], ymin=0, ymax=ymax, color='purple', lw=1.0, linestyle='--', label='Kr lines fit')
        xx = np.arange(0, 256)
        ax1.plot(xx, spectrum_Kr, color='blue', lw=1.0, label='Kr spectrum')
        ax1.plot(xx, gauss_set(xx, *Kr_coeff), color='orange', linewidth=1.5)
        ax1.set_xlabel('Pixel', fontsize=14)
        ax1.set_ylabel('Counts', fontsize=14)
        ax1.set_title(which_spec_write, fontsize=18)
        ax1.legend()
        fig.tight_layout()
        fig.savefig(f'output/wvl-cal/wvl_cal_{which_spec_write}_line_fitting.png', bbox_inches='tight')
        # plt.show()


        Hg_p_to_lambda = {}
        for p in range(len(Hg_coeff)//3):
            _, mu, _ = Hg_coeff[3*p:3*p+3]
            Hg_p_to_lambda[mu] = lamp_Hg[lamp_Hg>900][p]

        Kr_p_to_lambda = {}
        for p in range(len(Kr_coeff)//3):
            _, mu, _ = Kr_coeff[3*p:3*p+3]
            Kr_p_to_lambda[mu] = lamp_Kr[lamp_Kr>900][p]

        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        x_p = np.arange(0, 256)

        coefs = load_wvl_coef()
        C_pre = coefs[which_spec]
        C_trial_low, C_trial_high = C_pre.copy(), C_pre.copy()
        C_trial_low[C_trial_low>0] *= 0.8
        C_trial_low[C_trial_low<0] *= 1.2
        C_trial_high[C_trial_high>0] *= 1.2
        C_trial_high[C_trial_high<0] *= 0.8
        xx = p_to_lambda(x_p, *C_pre)
        fianl_coeff, var_matrix = curve_fit(p_to_lambda, list(Kr_p_to_lambda.keys())+list(Hg_p_to_lambda.keys()),
                                                   list(Kr_p_to_lambda.values())+list(Hg_p_to_lambda.values()), 
                                                   p0=C_pre, maxfev=10000, bounds=(C_trial_low, C_trial_high))
        ax1.scatter(list(Kr_p_to_lambda.keys()), list(Kr_p_to_lambda.values()), color='r', marker='D', label='Kr lines')
        ax1.scatter(list(Hg_p_to_lambda.keys()), list(Hg_p_to_lambda.values()), color='green', marker='^', label='Hg lines')
        ax1.plot(x_p, p_to_lambda(x_p, *fianl_coeff), label='fitting')
        ax1.plot(x_p, p_to_lambda(x_p, *C_pre), 'k--', label='manual')
        from scipy.optimize import fsolve
        wvl_2190_before = fsolve(p_to_lambda_wvl, 25, args=(C_pre[0], C_pre[1], C_pre[2], C_pre[3], C_pre[4], 2190.251))
        ax1.scatter(wvl_2190_before, 2190.251, color='grey', marker='x', label='before fitting')
        wvl_2190_after = fsolve(p_to_lambda_wvl, 25, args=(fianl_coeff[0], fianl_coeff[1], fianl_coeff[2], fianl_coeff[3], fianl_coeff[4], 2190.251))
        ax1.scatter(wvl_2190_after, 2190.251, color='k', marker='s', label='after fitting')
        ax1.legend()
        ax1.set_xlabel('Pixel', fontsize=14)
        ax1.set_ylabel('Wavelength [nm]', fontsize=14)
        ax1.set_title(which_spec_write, fontsize=18)
        fig.tight_layout()
        fig.savefig(f'output/wvl-cal/wvl_cal_{which_spec_write}_coeff.png', bbox_inches='tight')
        # plt.show()
        # sys.exit()
        if os.path.isfile(f'output/wvl-cal/{which_spec_write}_wvl_cal_InGaAs.txt'):
            open_status = 'a'
        else:
            open_status = 'w'
        print(open_status)
        with open(f'output/wvl-cal/{which_spec_write}_wvl_cal_InGaAs.txt', open_status) as f:
            f.write(f'{which_spec_write}\n')
            f.write(f'# {" ".join([str(x) for x in fianl_coeff])}\n')
            
        wvl_base_new = cal_wvl(fianl_coeff, Nchan=Nchan)
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        ax1.plot(wvl_base_new, spectrum_Hg, color='red', lw=1.0, label='Hg spectrum (original coeff)')
        ax1.plot(wvl_base_new, spectrum_Kr, color='blue', lw=1.0, label='Kr spectrum (original coeff)')
        ymin, ymax= ax1.get_ylim()
        ax1.vlines(lamp_Hg, ymin=0, ymax=ymax, color='green', lw=1.0, label='Hg lines')
        ax1.vlines(lamp_Kr, ymin=0, ymax=ymax, color='purple', lw=1.0, label='Kr lines')
        ax1.set_xlim(900, 2250)
        ax1.set_xlabel('Wavelength (nm)', fontsize=14)
        ax1.set_ylabel('Counts', fontsize=14)
        ax1.set_title(which_spec_write, fontsize=18)
        ax1.legend()
        fig.savefig(f'output/wvl-cal/wvl_cal_{which_spec_write}_new_spectrum.png', bbox_inches='tight')
        # plt.show()
        
        # after coefficients fitting
        # fit the gaussian line shape in wavelength space
        
        

        if which_spec == 'lasp|ssfr-a|zen|in':
            center_shift = 0
            low_shit, high_shit = -0.25, 0.25
        elif which_spec == 'lasp|ssfr-a|nad|in':
            center_shift = 0
            low_shit, high_shit = -0.25, 0.25
        elif which_spec == 'lasp|ssfr-b|zen|in':
            center_shift = 0
            low_shit, high_shit = -0.25, 0.25
        elif which_spec == 'lasp|ssfr-b|nad|in':
            center_shift = 0
            low_shit, high_shit = -0.25, 0.25
        center_shift *= -1
        low_shit, high_shit = -high_shit, -low_shit
        
        pixel_wvl_factor = 5
        center_shift *= pixel_wvl_factor
        low_shit *= pixel_wvl_factor
        high_shit *= pixel_wvl_factor
        Hg_p0 = []
        Hg_p0_bound_low = []
        Hg_p0_bound_high = []
        for wvl0 in lamp_Hg[lamp_Hg>900]:
            if radiance:
                Hg_p0.extend([5000, wvl0+center_shift, 1.25*pixel_wvl_factor])
                Hg_p0_bound_low.extend([10, wvl0+low_shit, 0.75*pixel_wvl_factor])
                Hg_p0_bound_high.extend([60000, wvl0+high_shit, 1.75*pixel_wvl_factor])
            else:
                Hg_p0.extend([600, wvl0+center_shift, 1.25*pixel_wvl_factor])
                Hg_p0_bound_low.extend([1, wvl0+low_shit, 0.75*pixel_wvl_factor])
                Hg_p0_bound_high.extend([6000, wvl0+high_shit, 1.75*pixel_wvl_factor])
        
        Hg_coeff, Hg_var_matrix = curve_fit(gauss_set, wvl_base_new, spectrum_Hg, p0=Hg_p0, maxfev=50000, bounds=(Hg_p0_bound_low, Hg_p0_bound_high))
        Hg_perr = np.sqrt(np.diag(Hg_var_matrix))
        
        plt.close('all')
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.vlines((lamp_Hg[lamp_Hg>900]), ymin=0, ymax=ymax, color='green', lw=1.0, label='Hg lines')
        ax1.vlines(Hg_coeff[1::3], ymin=0, ymax=ymax, color='green', lw=1.0, linestyle='--', label='Hg lines fitting')
        xx = wvl_base_new
        ax1.plot(xx, spectrum_Hg, color='red', lw=1.0, label='Hg spectrum')
        ax1.plot(xx, gauss_set(xx, *Hg_coeff), color='g', linewidth=1.5)

        Kr_p0 = []
        Kr_p0_bound_low = []
        Kr_p0_bound_high = []
        
        if which_spec == 'lasp|ssfr-a|zen|in':
            center_shift = 0
            low_shit, high_shit = -0.25, 0.25
        elif which_spec == 'lasp|ssfr-a|nad|in':
            center_shift = 0
            low_shit, high_shit = -0.25, 0.25
        elif which_spec == 'lasp|ssfr-b|zen|in':
            center_shift = 0
            low_shit, high_shit = -0.25, 0.25
        elif which_spec == 'lasp|ssfr-b|nad|in':
            center_shift = 0
            low_shit, high_shit = -0.25, 0.25
        
        center_shift *= -1
        low_shit, high_shit = -high_shit, -low_shit
        center_shift *= pixel_wvl_factor
        low_shit *= pixel_wvl_factor
        high_shit *= pixel_wvl_factor

        for wvl0 in lamp_Kr[lamp_Kr>900]:
            if radiance:
                if wvl0 != 2116.547:
                    Kr_p0.extend([1000, wvl0+center_shift, 1.25*pixel_wvl_factor])
                    Kr_p0_bound_low.extend([10, wvl0+low_shit, 0.75*pixel_wvl_factor])
                    Kr_p0_bound_high.extend([60000, wvl0+high_shit, 1.8*pixel_wvl_factor])
                else:
                    Kr_p0.extend([500, wvl0+center_shift, 1.25*pixel_wvl_factor])
                    Kr_p0_bound_low.extend([1, wvl0+low_shit, 0.75*pixel_wvl_factor])
                    Kr_p0_bound_high.extend([5000, wvl0+high_shit, 1.75*pixel_wvl_factor])
            else:
                if wvl0 != 2116.547:
                    Kr_p0.extend([100, wvl0+center_shift, 1.25*pixel_wvl_factor])
                    Kr_p0_bound_low.extend([10, wvl0+low_shit, 0.75*pixel_wvl_factor])
                    Kr_p0_bound_high.extend([6000, wvl0+high_shit, 1.75*pixel_wvl_factor])
                else:
                    Kr_p0.extend([50, wvl0+center_shift, 1.25*pixel_wvl_factor])
                    Kr_p0_bound_low.extend([1, wvl0+low_shit, 0.75*pixel_wvl_factor])
                    Kr_p0_bound_high.extend([500, wvl0+high_shit, 1.75*pixel_wvl_factor])

        Kr_coeff, Kr_var_matrix = curve_fit(gauss_set, wvl_base_new, spectrum_Kr, p0=Kr_p0, maxfev=50000, bounds=(Kr_p0_bound_low, Kr_p0_bound_high))
        Kr_perr = np.sqrt(np.diag(Kr_var_matrix))
        ax1.vlines((lamp_Kr[lamp_Kr>900]), ymin=0, ymax=ymax, color='purple', lw=1.0, label='Kr lines')
        ax1.vlines(Kr_coeff[1::3], ymin=0, ymax=ymax, color='purple', lw=1.0, linestyle='--', label='Kr lines fit')
        xx = wvl_base_new
        ax1.plot(xx, spectrum_Kr, color='blue', lw=1.0, label='Kr spectrum')
        ax1.plot(xx, gauss_set(xx, *Kr_coeff), color='orange', linewidth=1.5)
        ax1.set_xlabel('Wavelength (nm)', fontsize=14)
        ax1.set_ylabel('Counts', fontsize=14)
        ax1.set_title(which_spec_write, fontsize=18)
        ax1.legend()

        ax2.errorbar(Hg_coeff[1::3], Hg_coeff[2::3], yerr=Hg_perr[2::3], fmt='ro', alpha=0.75)
        ax2.errorbar(Kr_coeff[1::3], Kr_coeff[2::3], yerr=Kr_perr[2::3], fmt='go', alpha=0.75)
        ax2.plot(Kr_coeff[1::3], Kr_coeff[2::3], 'ro', label='Kr lines')
        ax2.plot(Hg_coeff[1::3], Hg_coeff[2::3], 'go', label='Hg lines')
        ax2.set_xlabel('Wavelength (nm)', fontsize=14)
        ax2.set_ylabel('Sigma (nm)', fontsize=14)
        ax2.set_title(which_spec_write, fontsize=18)
        ax2.set_ylim(0, 10)
        
        # calculate average sigma with uncertainty as weights
        weights = np.concatenate((1./Hg_perr[2::3], 1./Kr_perr[2::3]))
        errs = np.concatenate((Hg_perr[2::3], Kr_perr[2::3]))
        sigma_avg = np.average(np.concatenate((Hg_coeff[2::3], Kr_coeff[2::3])), weights=weights)
        # weighted average error
        sigma_avg_err = np.sqrt(np.sum(errs**2 * weights)/ np.sum(weights)) 
        ax2.axhline(sigma_avg, color='black', lw=1.0, linestyle='--', label='average sigma')
        ax2.fill_between([xx[0], xx[-1]], sigma_avg-sigma_avg_err, sigma_avg+sigma_avg_err, color='grey', alpha=0.5)
        ax2.text(2100, 0.75, f'avg sigma = {sigma_avg:.2f} +/- {sigma_avg_err:.2f} nm', fontsize=12, color='black', ha='right')
        ax2.text(2100, 0.375, f'avg FWHM = {sigma_avg*2*np.sqrt(2*np.log(2)):.2f} +/- {sigma_avg_err*2*np.sqrt(2*np.log(2)):.2f} nm', fontsize=12, color='black', ha='right')
        ax2.legend()
        fig.tight_layout()
        fig.savefig(f'output/wvl-cal/wvl_cal_{which_spec_write}_line_fitting_lineshape.png', bbox_inches='tight')
        # plt.show()
        
        
        
        pixel_wvl_factor = 5
        center_shift *= pixel_wvl_factor
        low_shit *= pixel_wvl_factor
        high_shit *= pixel_wvl_factor
        
        combine_p0 = [1.25*pixel_wvl_factor]
        combine_p0_bound_low = [0.75*pixel_wvl_factor]
        combine_p0_bound_high = [1.75*pixel_wvl_factor]
        for wvl0 in lamp_Hg[lamp_Hg>900]:
            if radiance:
                combine_p0.extend([5000])
                combine_p0_bound_low.extend([10])
                combine_p0_bound_high.extend([60000])
            else:
                combine_p0.extend([500])
                combine_p0_bound_low.extend([1])
                combine_p0_bound_high.extend([6000])
        for wvl0 in lamp_Kr[lamp_Kr>900]:
            if radiance:
                combine_p0.extend([1000])
                combine_p0_bound_low.extend([10])
                combine_p0_bound_high.extend([60000])
            else:
                combine_p0.extend([100])
                combine_p0_bound_low.extend([10])
                combine_p0_bound_high.extend([6000])
        
        combine_spectrum_for_fitting = np.concatenate((spectrum_Hg, spectrum_Kr))
        combine_wvl_base_new = np.concatenate((wvl_base_new, wvl_base_new+2000))
        # plt.plot(combine_wvl_base_new, combine_spectrum_for_fitting, color='black', lw=1.0, label='combined spectrum for fitting')
        # plt.xlabel('Wavelength (nm)', fontsize=14)
        # plt.ylabel('Counts', fontsize=14)
        # # plt.show()
        # sys.exit()

        
        fit_function = lambda x, *arg: gauss_peaks_Hg_Kr_InGaAs(x, lamps_fitting, *arg)
        combine_coeff, combine_var_matrix = curve_fit(fit_function, combine_wvl_base_new, combine_spectrum_for_fitting, p0=combine_p0, maxfev=50000, bounds=(combine_p0_bound_low, combine_p0_bound_high))
        combine_perr = np.sqrt(np.diag(combine_var_matrix))
        

        # plt.plot(combine_wvl_base_new, gauss_peaks_Hg_Kr_Si(combine_wvl_base_new, *combine_coeff), color='black', lw=1.0, label='combined spectrum for fitting')
        # plt.xlabel('Wavelength (nm)', fontsize=14)
        # plt.ylabel('Counts', fontsize=14)
        # # plt.show()
        # sys.exit()
        plt.close('all')
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        Hg_lines_number = len(lamp_Hg[lamp_Hg>900])
        # ax1.vlines((lamp_Hg[lamp_Hg>900]), ymin=0, ymax=ymax, color='green', lw=1.0, label='Hg lines')
        ax1.vlines((lamp_Hg[lamp_Hg>900]), ymin=0, ymax=combine_coeff[1:Hg_lines_number+1], color='green', lw=1.0, label='Hg lines')
        # ax1.vlines(Hg_coeff[1::3], ymin=0, ymax=ymax, color='green', lw=1.0, linestyle='--', label='Hg lines fitting')
        xx = wvl_base_new
        ax1.plot(xx, spectrum_Hg, color='lime', lw=1.5, label='Hg spectrum')
        ax1.plot(xx, spectrum_Kr, color='orange', lw=1.5, label='Kr spectrum')
        fit_Hg_spectrum = gauss_peaks_Hg_Kr_InGaAs(combine_wvl_base_new, lamps_fitting, *combine_coeff)[:len(wvl_base_new)]
        ax1.plot(xx, fit_Hg_spectrum, color='green', linewidth=1., label='Hg fit spectrum')

        fit_Kr_spectrum = gauss_peaks_Hg_Kr_InGaAs(combine_wvl_base_new, lamps_fitting, *combine_coeff)[len(wvl_base_new):]
        ax1.plot(xx, fit_Kr_spectrum, color='red', linewidth=1., label='Kr fit spectrum')

        # ax1.vlines((lamp_Kr[lamp_Kr>900]), ymin=0, ymax=ymax, color='purple', lw=1.0, label='Kr lines')
        ax1.vlines((lamp_Kr[lamp_Kr>900]), ymin=0, ymax=combine_coeff[1+Hg_lines_number:], color='purple', lw=1.0, label='Kr lines')
        # ax1.vlines(Kr_coeff[1::3], ymin=0, ymax=ymax, color='purple', lw=1.0, linestyle='--', label='Kr lines fit')

        ax1.set_xlabel('Wavelength (nm)', fontsize=14)
        ax1.set_ylabel('Counts', fontsize=14)
        ax1.set_title(which_spec_write, fontsize=18)
        ax1.legend()

        gaussian_xx = np.linspace(-10, 10, 401)
        gaussian_yy_center = np.exp(-gaussian_xx**2/(2.*combine_coeff[0]**2))
        # calculate the uncertainty in the gaussian width
        gaussian_yy_narrow = np.exp(-gaussian_xx**2/(2.*(combine_coeff[0]-combine_perr[0])**2))
        gaussian_yy_wide = np.exp(-gaussian_xx**2/(2.*(combine_coeff[0]+combine_perr[0])**2))
        ax2.fill_between(gaussian_xx, gaussian_yy_narrow, gaussian_yy_wide, color='grey', alpha=0.5, label='sigma uncertainty')
        ax2.plot(gaussian_xx, gaussian_yy_center, color='black', label='slit function')
        # plot FWHM 
        x_left = gaussian_xx[gaussian_yy_center>0.5][0]
        x_right = gaussian_xx[gaussian_yy_center>0.5][-1]
        ax2.vlines(x_left, ymin=0, ymax=1, color='red', lw=1.0,)
        ax2.vlines(x_right, ymin=0, ymax=1, color='red', lw=1.0,)
        ax2.hlines(0.5, xmin=x_left, xmax=x_right, color='red', lw=1.0, linestyle='--', label='FWHM')
        # ax2.text((x_left+x_right)/2, 0.55, f'FWHM = {x_right-x_left:.2f}', color='red', fontsize=12, ha='center')
        ax2.text((x_left+x_right)/2, 0.55, f'FWHM = {combine_coeff[0]*2*np.sqrt(2*np.log(2)):.2f} +/- {combine_perr[0]*2*np.sqrt(2*np.log(2)):.2f}', color='red', fontsize=12, ha='center')
        ax2.set_xlabel('Wavelength (nm)', fontsize=14)
        ax2.set_ylabel('Relative Counts', fontsize=14)
        ax2.set_title(which_spec_write, fontsize=18)
        ax2.legend()
        fig.tight_layout()
        fig.savefig(f'output/wvl-cal/wvl_cal_{which_spec_write}_line_fitting_lineshape_2.png', bbox_inches='tight')
        plt.close('all')
    #\----------------------------------------------------------------------------/#



def cal_wvl_coef_two_lamps_Si(spectrum_Hg, spectrum_Kr, which_spec='lasp|ssfr-a|zen|si'):

    spectrum_Hg[spectrum_Hg<0] = 0
    spectrum_Kr[spectrum_Kr<0] = 0
    Nchan = spectrum_Hg.size
    xchan = np.arange(Nchan, dtype=np.float64)

    coefs = load_wvl_coef()
    radiance = True if 'ssrr' in which_spec else False
    which_spec_write = which_spec
    which_spec = which_spec.replace('ssrr', 'ssfr')  # need to be fixed
    coef_base = coefs[which_spec]
    wvl_base = cal_wvl(coef_base, Nchan=Nchan)
    
    if radiance:
        lamps_fitting = lamps_ssrr_fitting
    else:
        lamps_fitting = lamps_ssfr_fitting

    lamp_Hg = lamps_fitting['hg']
    lamp_Kr = lamps_fitting['kr']
    # only use the Hg lines below 1100 nm for Silicon detector
    lamp_Hg = lamp_Hg[lamp_Hg<1100]  
    lamp_Kr = lamp_Kr[lamp_Kr<1100]

    # figure
    #/----------------------------------------------------------------------------\#
    
    # import cartopy.crs as ccrs
    # mpl.use('Agg')

    if True:
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        # fig.suptitle('Figure')
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.plot(wvl_base, spectrum_Hg, color='red', lw=1.0, label='Hg spectrum (original coeff)')
        ax1.plot(wvl_base, spectrum_Kr, color='blue', lw=1.0, label='Kr spectrum (original coeff)')
        ymin, ymax= ax1.get_ylim()
        ax1.vlines(lamp_Hg[lamp_Hg<1100], ymin=0, ymax=ymax, color='green', lw=1.0, label='Hg lines')
        ax1.vlines(lamp_Kr[lamp_Kr<1100], ymin=0, ymax=ymax, color='purple', lw=1.0, label='Kr lines')
        ax1.set_xlim(300, 1150)
        ax1.set_xlabel('Wavelength (nm)', fontsize=14)
        ax1.set_ylabel('Counts', fontsize=14)
        ax1.set_title(which_spec_write, fontsize=18)
        ax1.legend()
        fig.savefig(f'output/wvl-cal/wvl_cal_{which_spec_write}_original_spectrum.png', bbox_inches='tight')

        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)

        if which_spec == 'lasp|ssfr-a|zen|si':
            center_shift = 0
            low_shit, high_shit = -0.3, 0.3
        elif which_spec == 'lasp|ssfr-a|nad|si':
            center_shift = 0
            low_shit, high_shit = -0.3, 0.3
        elif which_spec == 'lasp|ssfr-b|zen|si':
            center_shift = 0
            low_shit, high_shit = -0.3, 0.3
        elif which_spec == 'lasp|ssfr-b|nad|si':
            center_shift = 0
            low_shit, high_shit = -0.3, 0.3
        Hg_p0 = []
        Hg_p0_bound_low = []
        Hg_p0_bound_high = []
        for wvl0 in lamp_Hg[lamp_Hg<1100]:
            if radiance:
                Hg_p0.extend([6000, lambda_to_p_si(wvl0)+center_shift, 1.1])
                Hg_p0_bound_low.extend([5, lambda_to_p_si(wvl0)+low_shit, 0.65])
                Hg_p0_bound_high.extend([60000, lambda_to_p_si(wvl0)+high_shit, 1.6])
            else:
                Hg_p0.extend([600, lambda_to_p_si(wvl0)+center_shift, 1.1])
                Hg_p0_bound_low.extend([0.1, lambda_to_p_si(wvl0)+low_shit, 0.65])
                Hg_p0_bound_high.extend([6000, lambda_to_p_si(wvl0)+high_shit, 1.6])
        
        Hg_coeff, var_matrix = curve_fit(gauss_set, np.arange(0, 256), spectrum_Hg, p0=Hg_p0, maxfev=50000, bounds=(Hg_p0_bound_low, Hg_p0_bound_high))
        ax1.vlines(lambda_to_p_si(lamp_Hg[lamp_Hg<1100]), ymin=0, ymax=ymax, color='green', lw=1.0, label='Hg lines')
        ax1.vlines(Hg_coeff[1::3], ymin=0, ymax=ymax, color='green', lw=1.0, linestyle='--', label='Hg lines fitting')
        xx = np.arange(0, 256)
        ax1.plot(xx, spectrum_Hg, color='red', lw=1.0, label='Hg spectrum')
        ax1.plot(xx, gauss_set(xx, *Hg_coeff), color='g', linewidth=1.5)

        Kr_p0 = []
        Kr_p0_bound_low = []
        Kr_p0_bound_high = []
        
        if which_spec == 'lasp|ssfr-a|zen|si':
            center_shift = 0
            low_shit, high_shit = -0.3, 0.3
        elif which_spec == 'lasp|ssfr-a|nad|si':
            center_shift = 0
            low_shit, high_shit = -0.3, 0.3
        elif which_spec == 'lasp|ssfr-b|zen|si':
            center_shift = 0
            low_shit, high_shit = -0.3, 0.3
        elif which_spec == 'lasp|ssfr-b|nad|si':
            center_shift = 0
            low_shit, high_shit = -0.3, 0.3

        for wvl0 in lamp_Kr[lamp_Kr<1100]:
            if radiance:
                Kr_p0.extend([1000, lambda_to_p_si(wvl0)+center_shift, 1.1])
                Kr_p0_bound_low.extend([0, lambda_to_p_si(wvl0)+low_shit, 0.65])
                Kr_p0_bound_high.extend([30000, lambda_to_p_si(wvl0)+high_shit, 1.6])
            else:
                Kr_p0.extend([90, lambda_to_p_si(wvl0)+center_shift, 1.1])
                Kr_p0_bound_low.extend([1, lambda_to_p_si(wvl0)+low_shit, 0.65])
                Kr_p0_bound_high.extend([1500, lambda_to_p_si(wvl0)+high_shit, 1.6])

        Kr_coeff, var_matrix = curve_fit(gauss_set, np.arange(0, 256), spectrum_Kr, p0=Kr_p0, maxfev=50000, bounds=(Kr_p0_bound_low, Kr_p0_bound_high))
        ax1.vlines(lambda_to_p_si(lamp_Kr[lamp_Kr<1100]), ymin=0, ymax=ymax, color='purple', lw=1.0, label='Kr lines')
        ax1.vlines(Kr_coeff[1::3], ymin=0, ymax=ymax, color='purple', lw=1.0, linestyle='--', label='Kr lines fit')
        xx = np.arange(0, 256)
        ax1.plot(xx, spectrum_Kr, color='blue', lw=1.0, label='Kr spectrum')
        ax1.plot(xx, gauss_set(xx, *Kr_coeff), color='orange', linewidth=1.5)
        ax1.set_xlabel('Pixel', fontsize=14)
        ax1.set_ylabel('Counts', fontsize=14)
        ax1.set_title(which_spec_write, fontsize=18)
        ax1.legend()
        fig.tight_layout()
        fig.savefig(f'output/wvl-cal/wvl_cal_{which_spec_write}_line_fitting.png', bbox_inches='tight')

        Hg_p_to_lambda = {}
        for p in range(len(Hg_coeff)//3):
            _, mu, _ = Hg_coeff[3*p:3*p+3]
            Hg_p_to_lambda[mu] = lamp_Hg[lamp_Hg<1100][p]

        Kr_p_to_lambda = {}
        for p in range(len(Kr_coeff)//3):
            _, mu, _ = Kr_coeff[3*p:3*p+3]
            Kr_p_to_lambda[mu] = lamp_Kr[lamp_Kr<1100][p]

        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        x_p = np.arange(0, 256)

        coefs = load_wvl_coef()
        C_pre = coefs[which_spec]
        if C_pre[-1] == 0:
            C_pre = C_pre[:-1]
        C_trial_low, C_trial_high = C_pre.copy(), C_pre.copy()
        C_trial_low[C_trial_low>0] *= 0.7
        C_trial_low[C_trial_low<0] *= 1.3
        C_trial_high[C_trial_high>0] *= 1.3
        C_trial_high[C_trial_high<0] *= 0.7
        xx = p_to_lambda_3rd(x_p, *C_pre)
        fianl_coeff, var_matrix = curve_fit(p_to_lambda_3rd, list(Kr_p_to_lambda.keys())+list(Hg_p_to_lambda.keys()),
                                                   list(Kr_p_to_lambda.values())+list(Hg_p_to_lambda.values()), 
                                                   p0=C_pre, maxfev=50000, bounds=(C_trial_low, C_trial_high))
        ax1.scatter(list(Kr_p_to_lambda.keys()), list(Kr_p_to_lambda.values()), color='r', marker='D', label='Kr lines')
        ax1.scatter(list(Hg_p_to_lambda.keys()), list(Hg_p_to_lambda.values()), color='green', marker='^', label='Hg lines')
        ax1.plot(x_p, p_to_lambda_3rd(x_p, *fianl_coeff), label='fitting')
        ax1.plot(x_p, p_to_lambda_3rd(x_p, *C_pre), 'k--', label='manual')
        from scipy.optimize import fsolve
        # wvl_2190_before = fsolve(p_to_lambda_wvl_3rd, 25, args=(C_pre[0], C_pre[1], C_pre[2], C_pre[3], C_pre[4], 2190.251))
        # ax1.scatter(wvl_2190_before, 2190.251, color='grey', marker='x', label='before fitting')
        # wvl_2190_after = fsolve(p_to_lambda_wvl_3rd, 25, args=(fianl_coeff[0], fianl_coeff[1], fianl_coeff[2], fianl_coeff[3], 2190.251))
        # ax1.scatter(wvl_2190_after, 2190.251, color='k', marker='s', label='after fitting')
        ax1.legend()
        ax1.set_xlabel('Pixel', fontsize=14)
        ax1.set_ylabel('Wavelength [nm]', fontsize=14)
        ax1.set_title(which_spec_write, fontsize=18)
        fig.tight_layout()
        fig.savefig(f'output/wvl-cal/wvl_cal_{which_spec_write}_coeff_si.png', bbox_inches='tight')
        

        if os.path.isfile(f'output/wvl-cal/{which_spec_write}_wvl_cal_Si.txt'):
            open_status = 'a'
        else:
            open_status = 'w'
        print(open_status)
        with open(f'output/wvl-cal/{which_spec_write}_wvl_cal_Si.txt', open_status) as f:
            f.write(f'{which_spec_write}\n')
            f.write(f'# {" ".join([str(x) for x in fianl_coeff])}\n')
            
            
        wvl_base_new = cal_wvl(fianl_coeff, Nchan=Nchan)
        plt.close('all')
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        ax1.plot(wvl_base_new, spectrum_Hg, color='red', lw=1.0, label='Hg spectrum (original coeff)')
        ax1.plot(wvl_base_new, spectrum_Kr, color='blue', lw=1.0, label='Kr spectrum (original coeff)')
        ymin, ymax= ax1.get_ylim()
        ax1.vlines(lamp_Hg, ymin=0, ymax=ymax, color='green', lw=1.0, label='Hg lines')
        ax1.vlines(lamp_Kr, ymin=0, ymax=ymax, color='purple', lw=1.0, label='Kr lines')
        ax1.set_xlim(300, 1150)
        ax1.set_xlabel('Wavelength (nm)', fontsize=14)
        ax1.set_ylabel('Counts', fontsize=14)
        ax1.set_title(which_spec_write, fontsize=18)
        ax1.legend()
        fig.tight_layout()
        fig.savefig(f'output/wvl-cal/wvl_cal_{which_spec_write}_original_spectrum_new_coeff.png', bbox_inches='tight')        
        
        # after coefficients fitting
        # fit the gaussian line shape in wavelength space
 
        if which_spec == 'lasp|ssfr-a|zen|si':
            center_shift = 0
            low_shit, high_shit = -0.1, 0.1
        elif which_spec == 'lasp|ssfr-a|nad|si':
            center_shift = 0
            low_shit, high_shit = -0.1, 0.1
        elif which_spec == 'lasp|ssfr-b|zen|si':
            center_shift = 0
            low_shit, high_shit = -0.1, 0.1
        elif which_spec == 'lasp|ssfr-b|nad|si':
            center_shift = 0
            low_shit, high_shit = -0.1, 0.1
        
        
            
        pixel_wvl_factor = 3.3
        center_shift *= pixel_wvl_factor
        low_shit *= pixel_wvl_factor
        high_shit *= pixel_wvl_factor
        
        Hg_p0 = []
        Hg_p0_bound_low = []
        Hg_p0_bound_high = []
        for wvl0 in lamp_Hg[lamp_Hg<1100]:
            if radiance:
                Hg_p0.extend([6000, wvl0+center_shift, 1.1*pixel_wvl_factor])
                Hg_p0_bound_low.extend([50, wvl0+low_shit, 0.65*pixel_wvl_factor])
                Hg_p0_bound_high.extend([60000, wvl0+high_shit, 1.6*pixel_wvl_factor])
            else:
                Hg_p0.extend([600, wvl0+center_shift, 1.1*pixel_wvl_factor])
                Hg_p0_bound_low.extend([0.1, wvl0+low_shit, 0.65*pixel_wvl_factor])
                Hg_p0_bound_high.extend([6000, wvl0+high_shit, 1.6*pixel_wvl_factor])
        
        Hg_coeff, Hg_var_matrix = curve_fit(gauss_set, wvl_base_new, spectrum_Hg, p0=Hg_p0, maxfev=50000, bounds=(Hg_p0_bound_low, Hg_p0_bound_high))
        Hg_perr = np.sqrt(np.diag(Hg_var_matrix))
        plt.close('all')
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.vlines((lamp_Hg[lamp_Hg<1100]), ymin=0, ymax=ymax, color='green', lw=1.0, label='Hg lines')
        ax1.vlines(Hg_coeff[1::3], ymin=0, ymax=ymax, color='green', lw=1.0, linestyle='--', label='Hg lines fitting')
        xx = wvl_base_new
        ax1.plot(xx, spectrum_Hg, color='red', lw=1.0, label='Hg spectrum')
        ax1.plot(xx, gauss_set(xx, *Hg_coeff), color='g', linewidth=1.5)

        Kr_p0 = []
        Kr_p0_bound_low = []
        Kr_p0_bound_high = []
        
        if which_spec == 'lasp|ssfr-a|zen|si':
            center_shift = 0
            low_shit, high_shit = -0.1, 0.1
        elif which_spec == 'lasp|ssfr-a|nad|si':
            center_shift = 0
            low_shit, high_shit = -0.1, 0.1
        elif which_spec == 'lasp|ssfr-b|zen|si':
            center_shift = 0
            low_shit, high_shit = -0.1, 0.1
        elif which_spec == 'lasp|ssfr-b|nad|si':
            center_shift = 0
            low_shit, high_shit = -0.1, 0.1
        
        center_shift *= pixel_wvl_factor
        low_shit *= pixel_wvl_factor
        high_shit *= pixel_wvl_factor

        for wvl0 in lamp_Kr[lamp_Kr<1100]:
            if radiance:
                Kr_p0.extend([1000, wvl0+center_shift, 1.1*pixel_wvl_factor])
                Kr_p0_bound_low.extend([1, wvl0+low_shit, 0.65*pixel_wvl_factor])
                Kr_p0_bound_high.extend([30000, wvl0+high_shit, 1.6*pixel_wvl_factor])
            else:
                Kr_p0.extend([90, wvl0+center_shift, 1.1*pixel_wvl_factor])
                Kr_p0_bound_low.extend([1, wvl0+low_shit, 0.65*pixel_wvl_factor])
                Kr_p0_bound_high.extend([1500, wvl0+high_shit, 1.6*pixel_wvl_factor])

        Kr_coeff, Kr_var_matrix = curve_fit(gauss_set, wvl_base_new, spectrum_Kr, p0=Kr_p0, maxfev=50000, bounds=(Kr_p0_bound_low, Kr_p0_bound_high))
        Kr_perr = np.sqrt(np.diag(Kr_var_matrix))
        ax1.vlines((lamp_Kr[lamp_Kr<1100]), ymin=0, ymax=ymax, color='purple', lw=1.0, label='Kr lines')
        ax1.vlines(Kr_coeff[1::3], ymin=0, ymax=ymax, color='purple', lw=1.0, linestyle='--', label='Kr lines fit')
        xx = wvl_base_new
        ax1.plot(xx, spectrum_Kr, color='blue', lw=1.0, label='Kr spectrum')
        ax1.plot(xx, gauss_set(xx, *Kr_coeff), color='orange', linewidth=1.5)
        ax1.set_xlabel('Wavelength (nm)', fontsize=14)
        ax1.set_ylabel('Counts', fontsize=14)
        ax1.set_title(which_spec_write, fontsize=18)
        ax1.legend()

        ax2.errorbar(Kr_coeff[1::3], Kr_coeff[2::3], yerr=Kr_perr[2::3], fmt='ro', alpha=0.75)
        ax2.errorbar(Hg_coeff[1::3], Hg_coeff[2::3], yerr=Hg_perr[2::3], fmt='go', alpha=0.75)
        ax2.plot(Kr_coeff[1::3], Kr_coeff[2::3], 'ro', label='Kr lines')
        ax2.plot(Hg_coeff[1::3], Hg_coeff[2::3], 'go', label='Hg lines')
        ax2.set_xlabel('Wavelength (nm)', fontsize=14)
        ax2.set_ylabel('Sigma (nm)', fontsize=14)
        ax2.set_title(which_spec_write, fontsize=18)
        ax2.set_ylim(0, 8)
        
        # calculate average sigma with uncertainty as weights
        weights = np.concatenate((1./Hg_perr[2::3], 1./Kr_perr[2::3]))
        errs = np.concatenate((Hg_perr[2::3], Kr_perr[2::3]))
        sigma_avg = np.average(np.concatenate((Hg_coeff[2::3], Kr_coeff[2::3])), weights=weights)
        # weighted average error
        sigma_avg_err = np.sqrt(np.sum(errs**2 * weights)/ np.sum(weights)) 
        ax2.axhline(sigma_avg, color='black', lw=1.0, linestyle='--', label='average sigma')
        ax2.fill_between([xx[0], xx[-1]], sigma_avg-sigma_avg_err, sigma_avg+sigma_avg_err, color='grey', alpha=0.5)
        ax2.text(1000, 0.5, f'avg sigma = {sigma_avg:.2f} +/- {sigma_avg_err:.2f} nm', fontsize=12, color='black', ha='right')
        ax2.text(1000, 0.25, f'avg FWHM = {sigma_avg*2*np.sqrt(2*np.log(2)):.2f} +/- {sigma_avg_err*2*np.sqrt(2*np.log(2)):.2f} nm', fontsize=12, color='black', ha='right')
        ax2.legend()
        fig.tight_layout()
        fig.savefig(f'output/wvl-cal/wvl_cal_{which_spec_write}_line_fitting_lineshape.png', bbox_inches='tight')
        

        pixel_wvl_factor = 3.3
        center_shift *= pixel_wvl_factor
        low_shit *= pixel_wvl_factor
        high_shit *= pixel_wvl_factor
        
        combine_p0 = [1.2*pixel_wvl_factor]
        combine_p0_bound_low = [0.5*pixel_wvl_factor]
        combine_p0_bound_high = [2.0*pixel_wvl_factor]
        for wvl0 in lamp_Hg[lamp_Hg<1100]:
            if radiance:
                combine_p0.extend([6000])
                combine_p0_bound_low.extend([50])
                combine_p0_bound_high.extend([60000])
            else:
                combine_p0.extend([600])
                combine_p0_bound_low.extend([0.1])
                combine_p0_bound_high.extend([6000])
        for wvl0 in lamp_Kr[lamp_Kr<1100]:
            if radiance:
                combine_p0.extend([6000])
                combine_p0_bound_low.extend([50])
                combine_p0_bound_high.extend([60000])
            else:
                combine_p0.extend([90])
                combine_p0_bound_low.extend([0.1])
                combine_p0_bound_high.extend([1500])
        
        combine_spectrum_for_fitting = np.concatenate((spectrum_Hg, spectrum_Kr))
        combine_wvl_base_new = np.concatenate((wvl_base_new, wvl_base_new+2000))
        # plt.plot(combine_wvl_base_new, combine_spectrum_for_fitting, color='black', lw=1.0, label='combined spectrum for fitting')
        # plt.xlabel('Wavelength (nm)', fontsize=14)
        # plt.ylabel('Counts', fontsize=14)
        # # plt.show()
        # sys.exit()
        
        fit_function = lambda x, *arg: gauss_peaks_Hg_Kr_Si(x, lamps_fitting, *arg)
        combine_coeff, combine_var_matrix = curve_fit(fit_function, combine_wvl_base_new, combine_spectrum_for_fitting, p0=combine_p0, maxfev=50000, bounds=(combine_p0_bound_low, combine_p0_bound_high))
        combine_perr = np.sqrt(np.diag(combine_var_matrix))
        

        # plt.plot(combine_wvl_base_new, gauss_peaks_Hg_Kr_Si(combine_wvl_base_new, *combine_coeff), color='black', lw=1.0, label='combined spectrum for fitting')
        # plt.xlabel('Wavelength (nm)', fontsize=14)
        # plt.ylabel('Counts', fontsize=14)
        # # plt.show()
        # sys.exit()
        
        plt.close('all')
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        Hg_lines_number = len(lamp_Hg[lamp_Hg<1100])
        # ax1.vlines((lamp_Hg[lamp_Hg<1100]), ymin=0, ymax=ymax, color='green', lw=1.0, label='Hg lines')
        ax1.vlines((lamp_Hg[lamp_Hg<1100]), ymin=0, ymax=combine_coeff[1:Hg_lines_number+1], color='green', lw=1.0, label='Hg lines')
        # ax1.vlines(Hg_coeff[1::3], ymin=0, ymax=ymax, color='green', lw=1.0, linestyle='--', label='Hg lines fitting')
        xx = wvl_base_new
        ax1.plot(xx, spectrum_Hg, color='lime', lw=2.0, label='Hg spectrum')
        ax1.plot(xx, spectrum_Kr, color='orange', lw=2.0, label='Kr spectrum')

        fit_Hg_spectrum = gauss_peaks_Hg_Kr_Si(combine_wvl_base_new, lamps_fitting, *combine_coeff)[:len(wvl_base_new)]
        ax1.plot(xx, fit_Hg_spectrum, color='green', linewidth=1.0, label='Hg fit spectrum')

        fit_Kr_spectrum = gauss_peaks_Hg_Kr_Si(combine_wvl_base_new, lamps_fitting, *combine_coeff)[len(wvl_base_new):]
        ax1.plot(xx, fit_Kr_spectrum, color='red', linewidth=1.0, label='Kr fit spectrum')

        # ax1.vlines((lamp_Kr[lamp_Kr<1100]), ymin=0, ymax=ymax, color='purple', lw=1.0, label='Kr lines')
        ax1.vlines((lamp_Kr[lamp_Kr<1100]), ymin=0, ymax=combine_coeff[Hg_lines_number+1:], color='purple', lw=1.0, label='Kr lines')
        # ax1.vlines(Kr_coeff[1::3], ymin=0, ymax=ymax, color='purple', lw=1.0, linestyle='--', label='Kr lines fit')
        
        ax1.set_xlabel('Wavelength (nm)', fontsize=14)
        ax1.set_ylabel('Counts', fontsize=14)
        ax1.set_title(which_spec_write, fontsize=18)
        ax1.legend()

        gaussian_xx = np.linspace(-10, 10, 401)
        gaussian_yy_center = np.exp(-gaussian_xx**2/(2.*combine_coeff[0]**2))
        # calculate the uncertainty in the gaussian width
        gaussian_yy_narrow = np.exp(-gaussian_xx**2/(2.*(combine_coeff[0]-combine_perr[0])**2))
        gaussian_yy_wide = np.exp(-gaussian_xx**2/(2.*(combine_coeff[0]+combine_perr[0])**2))
        ax2.fill_between(gaussian_xx, gaussian_yy_narrow, gaussian_yy_wide, color='grey', alpha=0.5, label='sigma uncertainty')
        ax2.plot(gaussian_xx, gaussian_yy_center, color='black', label='slit function')
        # plot FWHM 
        x_left = gaussian_xx[gaussian_yy_center>0.5][0]
        x_right = gaussian_xx[gaussian_yy_center>0.5][-1]
        ax2.vlines(x_left, ymin=0, ymax=1, color='red', lw=1.0,)
        ax2.vlines(x_right, ymin=0, ymax=1, color='red', lw=1.0,)
        ax2.hlines(0.5, xmin=x_left, xmax=x_right, color='red', lw=1.0, linestyle='--', label='FWHM')
        # ax2.text((x_left+x_right)/2, 0.55, f'FWHM = {x_right-x_left:.2f}', color='red', fontsize=12, ha='center')
        ax2.text((x_left+x_right)/2, 0.55, f'FWHM = {combine_coeff[0]*2*np.sqrt(2*np.log(2)):.2f} +/- {combine_perr[0]*2*np.sqrt(2*np.log(2)):.2f}', color='red', fontsize=12, ha='center')
        ax2.set_xlabel('Wavelength (nm)', fontsize=14)
        ax2.set_ylabel('Relative Counts', fontsize=14)
        ax2.set_title(which_spec_write, fontsize=18)
        ax2.legend()
        fig.tight_layout()
        fig.savefig(f'output/wvl-cal/wvl_cal_{which_spec_write}_line_fitting_lineshape_2.png', bbox_inches='tight')
        plt.close('all')
    #\----------------------------------------------------------------------------/#

def lambda_to_p(lambda_):
    p0 = 403.854
    p1 = -0.20108
    p2 = 0.000079447
    p3 = -5.27978e-8
    p4 = 9.36204e-12
    return p0 + p1*lambda_ + p2*lambda_**2 + p3*lambda_**3 + p4*lambda_**4

def lambda_to_p_si(lambda_):
    p0 = -92.588
    p1 = 0.311531
    p2 = -2.37584e-5
    p3 = 1.53662e-8
    p4 = 0
    return p0 + p1*lambda_ + p2*lambda_**2 + p3*lambda_**3 + p4*lambda_**4

def p_to_lambda(p_, C0, C1, C2, C3, C4):
    return C0 + C1*p_ + C2*p_**2 + C3*p_**3 + C4*p_**4

def p_to_lambda_wvl(p_, C0, C1, C2, C3, C4, C_wvl):
    return C0 + C1*p_ + C2*p_**2 + C3*p_**3 + C4*p_**4 - C_wvl

def p_to_lambda_3rd(p_, C0, C1, C2, C3):
    return C0 + C1*p_ + C2*p_**2 + C3*p_**3 

def p_to_lambda_wvl_3rd(p_, C0, C1, C2, C3, C_wvl):
    return C0 + C1*p_ + C2*p_**2 + C3*p_**3 - C_wvl

def gauss_set(x, *args):
    output = 0
    for p in range(len(args)//3):
        A, mu, sigma = args[3*p:3*p+3]
        output += A*np.exp(-(x-mu)**2/(2.*sigma**2))
    return output

### define a new gauss_peaks_Hg_Kr_Si that can fit both Hg and Kr lines for two Energy (a.u.) outputs but only one sigma value
def gauss_peaks_Hg_Kr_Si(x, lamps_fitting, *args):
    output = 0
    sigma = args[0]
    Hg_lines = lamps_fitting['hg']
    Hg_lines_Si = Hg_lines[Hg_lines<1100]
    Kr_lines = lamps_fitting['kr']
    Kr_lines_Si = Kr_lines[Kr_lines<1100]
    combine_lines_Si = np.concatenate((Hg_lines_Si, Kr_lines_Si+2000))
    for p in range(len(args)-1):
        A = args[1:][p]
        mu = combine_lines_Si[p]
        output += A*np.exp(-(x-mu)**2/(2.*sigma**2))
    return output

def gauss_peaks_Hg_Kr_InGaAs(x, lamps_fitting, *args):
    output = 0
    sigma = args[0]
    Hg_lines = lamps_fitting['hg']
    Hg_lines_InGaAs = Hg_lines[Hg_lines>900]
    Kr_lines = lamps_fitting['kr']
    Kr_lines_InGaAs = Kr_lines[Kr_lines>900]
    combine_lines_InGaAs = np.concatenate((Hg_lines_InGaAs, Kr_lines_InGaAs+2000))
    for p in range(len(args)-1):
        A = args[1:][p]
        mu = combine_lines_InGaAs[p]
        output += A*np.exp(-(x-mu)**2/(2.*sigma**2))
    return output


if __name__ == '__main__':


    pass
