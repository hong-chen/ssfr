"""
Code for processing data collected by "radiation instruments" during NASA ARCSIX 2024.


Acknowledgements:
    Instrument engineering:
        Jeffery Drouet, Sebastian Schmidt
    Pre-mission calibration and analysis:
        Hong Chen, Yu-Wen Chen, Ken Hirata, Sebastian Schmidt, Bruce Kindel
    In-field calibration and on-flight operation:
        Arabella Chamberlain, Vikas Nataraja, Ken Hirata, Sebastian Schmidt
"""

import os
import sys
import glob
import datetime
import warnings
from tqdm import tqdm
import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav
from scipy.optimize import curve_fit
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
# mpl.use('TkAgg')


import ssfr



# parameters
#/----------------------------------------------------------------------------\#
_mission_     = 'arcsix'
_platform_    = 'p3b'

_hsk_         = 'hsk'
_alp_         = 'alp'
_spns_        = 'spns-a'
_ssfr1_       = 'ssfr-a'
_ssfr2_       = 'ssfr-b'
# _cam_         = 'cam'

_fdir_hsk_   = 'data/test/arcsix/2024-Spring/p3/aux/hsk'
_fdir_data_  = 'data/test/%s' % _mission_
_fdir_out_   = 'data/test/processed'
_fdir_cal_   = 'data/test/arcsix/cal'


_verbose_   = True
_test_mode_ = True

_fnames_ = {}
#\----------------------------------------------------------------------------/#







# functions for ssfr calibrations
#/----------------------------------------------------------------------------\#
def wvl_cal(ssfr_tag, lc_tag, lamp_tag, Nchan=256):

    fdir_data = '/argus/field/arcsix/cal/wvl-cal'

    indices_spec = {
            'zen': [0, 1],
            'nad': [2, 3]
            }

    fdir =  sorted(glob.glob('%s/*%s*%s*%s*' % (fdir_data, ssfr_tag, lc_tag, lamp_tag)))[0]
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))

    ssfr0 = ssfr.lasp_ssfr.read_ssfr(fnames, dark_corr_mode='interp')

    xchan = np.arange(Nchan)

    spectra0 = np.nanmean(ssfr0.dset0['spectra_dark-corr'][:, :, indices_spec[lc_tag]], axis=0)
    spectra1 = np.nanmean(ssfr0.dset1['spectra_dark-corr'][:, :, indices_spec[lc_tag]], axis=0)

    # spectra_inp = {lamp_tag.lower(): spectra0[:, 0]}
    # ssfr.cal.cal_wvl_coef(spectra_inp, which_spec='lasp|%s|%s|si' % (ssfr_tag.lower(), lc_tag.lower()))

    spectra_inp = {lamp_tag.lower(): spectra0[:, 1]}
    ssfr.cal.cal_wvl_coef(spectra_inp, which_spec='lasp|%s|%s|in' % (ssfr_tag.lower(), lc_tag.lower()))
    sys.exit()

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('%s %s (illuminated by %s Lamp)' % (ssfr_tag.upper(), lc_tag.title(), lamp_tag.upper()))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(121)
        ax1.plot(xchan, spectra0[:, 0], lw=1, c='r')
        ax1.plot(xchan, spectra1[:, 0], lw=1, c='b')
        ax1.set_xlabel('Channel #')
        ax1.set_ylabel('Counts')
        ax1.set_ylim(bottom=0)
        ax1.set_title('Silicon')

        ax2 = fig.add_subplot(122)
        ax2.plot(xchan, spectra0[:, 1], lw=1, c='r')
        ax2.plot(xchan, spectra1[:, 1], lw=1, c='b')
        ax2.set_xlabel('Channel #')
        ax2.set_ylabel('Counts')
        ax2.set_ylim(bottom=0)
        ax2.set_title('InGaAs')
        #\--------------------------------------------------------------/#

        patches_legend = [
                          mpatches.Patch(color='red' , label='IntTime set 1'), \
                          mpatches.Patch(color='blue', label='IntTime set 2'), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=16)

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_%s_%s_%s.png' % (_metadata['Function'], ssfr_tag.lower(), lc_tag.lower(), lamp_tag.lower()), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#

def rad_cal(ssfr_tag, lc_tag, lamp_tag, Nchan=256):

    fdir_data = 'data/arcsix/cal/rad-cal'

    indices_spec = {
            'zen': [0, 1],
            'nad': [2, 3]
            }

    fdir =  sorted(glob.glob('%s/*%s*%s*%s*' % (fdir_data, ssfr_tag, lc_tag, lamp_tag)))[0]
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))


    date_cal_s   = '2023-11-16'
    date_today_s = datetime.datetime.now().strftime('%Y-%m-%d')

    ssfr_ = ssfr.lasp_ssfr.read_ssfr(fnames)
    for i in range(ssfr_.Ndset):
        dset_tag = 'dset%d' % i
        dset_ = getattr(ssfr_, dset_tag)
        int_time = dset_['info']['int_time']

        fname = '%s/cal/%s|cal-rad-pri|lasp|%s|%s|%s-si%3.3d-in%3.3d|%s.h5' % (ssfr.common.fdir_data, date_cal_s, ssfr_tag.lower(), lc_tag.lower(), dset_tag.lower(), int_time['%s|si' % lc_tag], int_time['%s|in' % lc_tag], date_today_s)
        f = h5py.File(fname, 'w')

        resp_pri = ssfr.cal.cal_rad_resp(fnames, which_ssfr='lasp|%s' % ssfr_tag.lower(), which_lc=lc_tag.lower(), int_time=int_time, which_lamp=lamp_tag.lower())

        for key in resp_pri.keys():
            f[key] = resp_pri[key]

        f.close()

def ang_cal(fdir):

    """

    Notes:
        angular calibration is done for three different azimuth angles (reference to the vaccum port)
        60, 180, 300

        angles
    """

    tags = os.path.basename(fdir).split('_')
    ssfr_tag = tags[1]
    lc_tag   = tags[2]

    # get angles
    #/----------------------------------------------------------------------------\#
    angles_pos = np.concatenate((np.arange(0.0, 30.0, 3.0), np.arange(30.0, 50.0, 5.0), np.arange(50.0, 91.0, 10.0)))
    angles_neg = -angles_pos
    angles = np.concatenate((angles_pos, angles_neg, np.array([0.0])))
    #\----------------------------------------------------------------------------/#

    # make fnames, a dictionary <key:value> with file name as key, angle as value
    #/----------------------------------------------------------------------------\#
    fnames_ = sorted(glob.glob('%s/*.SKS' % fdir))
    fnames  = {
            fnames_[i]: angles[i] for i in range(angles.size)
            }
    #\----------------------------------------------------------------------------/#

    date_today_s = datetime.datetime.now().strftime('%Y-%m-%d')

    ssfr_ = ssfr.lasp_ssfr.read_ssfr([fnames_[0]])
    for i in range(ssfr_.Ndset):
        dset_tag = 'dset%d' % i
        dset_ = getattr(ssfr_, dset_tag)
        int_time = dset_['info']['int_time']

        filename_tag = '%s|%s|%s|%s' % (tags[0], tags[4], date_today_s, dset_tag)

        ssfr.cal.cdata_ang_resp(fnames, filename_tag=filename_tag, which_ssfr='lasp|%s' % ssfr_tag, which_lc=lc_tag, int_time=int_time)
#\----------------------------------------------------------------------------/#



# instrument calibrations
#/----------------------------------------------------------------------------\#
def main_calibration():

    """
    Notes:
        irradiance setup:
            SSFR-A (Alvin)
              - nadir : LC6 + stainless steel cased fiber
              - zenith: LC4 + black plastic cased fiber
    """

    # wavelength calibration
    #/----------------------------------------------------------------------------\#
    # for ssfr_tag in ['SSFR-A', 'SSFR-B']:
    #     for lc_tag in ['zen', 'nad']:
    #         for lamp_tag in ['kr', 'hg']:
    #             wvl_cal(ssfr_tag, lc_tag, lamp_tag)
    #\----------------------------------------------------------------------------/#

    # radiometric calibration
    #/----------------------------------------------------------------------------\#
    # for ssfr_tag in ['SSFR-A', 'SSFR-B']:
    #     for lc_tag in ['zen', 'nad']:
    #         for lamp_tag in ['1324']:
    #             rad_cal(ssfr_tag, lc_tag, lamp_tag)
    #\----------------------------------------------------------------------------/#

    # angular calibration
    #/----------------------------------------------------------------------------\#
    fdirs = [
            '/argus/field/arcsix/cal/ang-cal/2024-03-19_SSFR-A_zen-lc4_ang-cal_vaa-060_lamp-507_si-080-120_in-250-350',
            '/argus/field/arcsix/cal/ang-cal/2024-03-15_SSFR-A_zen-lc4_ang-cal_vaa-180_lamp-507_si-080-120_in-250-350',
            '/argus/field/arcsix/cal/ang-cal/2024-03-19_SSFR-A_zen-lc4_ang-cal_vaa-300_lamp-507_si-080-120_in-250-350',
            '/argus/field/arcsix/cal/ang-cal/2024-03-18_SSFR-A_nad-lc6_ang-cal_vaa-060_lamp-507_si-080-120_in-250-350',
            '/argus/field/arcsix/cal/ang-cal/2024-03-18_SSFR-A_nad-lc6_ang-cal_vaa-180_lamp-507_si-080-120_in-250-350',
            '/argus/field/arcsix/cal/ang-cal/2024-03-18_SSFR-A_nad-lc6_ang-cal_vaa-300_lamp-507_si-080-120_in-250-350',
            ]
    for fdir in fdirs:
        ang_cal(fdir)
    #\----------------------------------------------------------------------------/#

def test_data_a(ssfr_tag, lc_tag, lamp_tag, Nchan=256):

    fdir_data = '/argus/field/arcsix/cal/rad-cal'

    indices_spec = {
            'zen': [0, 1],
            'nad': [2, 3]
            }

    fdir =  sorted(glob.glob('%s/*%s*%s*%s*' % (fdir_data, ssfr_tag, lc_tag, lamp_tag)))[0]
    fnames = sorted(glob.glob('%s/*00001.SKS' % (fdir)))


    date_cal_s   = '2023-11-16'
    date_today_s = datetime.datetime.now().strftime('%Y-%m-%d')

    ssfr_ = ssfr.lasp_ssfr.read_ssfr(fnames)
    for i in range(ssfr_.Ndset):
        dset_tag = 'dset%d' % i
        dset_ = getattr(ssfr_, dset_tag)
        int_time = dset_['info']['int_time']

        fname = '%s/cal/%s|RAD-CAL-PRI|LASP|%s|%s|%s-SI%3.3d-IN%3.3d|%s.h5' % (ssfr.common.fdir_data, date_cal_s, ssfr_tag.upper(), lc_tag.upper(), dset_tag.upper(), int_time['%s|si' % lc_tag], int_time['%s|in' % lc_tag], date_today_s)
        f = h5py.File(fname, 'w')

        resp_pri = ssfr.cal.cal_rad_resp(fnames, which_ssfr='lasp|%s' % ssfr_tag.lower(), which_lc=lc_tag.lower(), int_time=int_time, which_lamp=lamp_tag.lower())

        for key in resp_pri.keys():
            f[key] = resp_pri[key]

        f.close()

def test_data_b(
        date,
        fdir_data=_fdir_out_,
        fdir_out=_fdir_out_,
        pitch_angle=0.0,
        roll_angle=0.0,
        ):

    date_s = date.strftime('%Y-%m-%d')

    fname_h5 = '%s/%s_%s_%s_v2.h5' % (fdir_out, _mission_.upper(), _ssfr_.upper(), date_s)
    f = h5py.File(fname_h5, 'w')

    fname_h5 = '%s/%s_%s_%s_v1.h5' % (fdir_data, _mission_.upper(), _ssfr_.upper(), date_s)
    f_ = h5py.File(fname_h5, 'r')
    tmhr = f_['tmhr'][...]
    for dset_s in f_.keys():

        if 'dset' in dset_s:

            # primary calibration (from pre-mission arcsix in lab on 2023-11-16)
            #/----------------------------------------------------------------------------\#
            wvls = ssfr.lasp_ssfr.get_ssfr_wavelength()
            wvl_start = 350.0
            wvl_end   = 2200.0
            wvl_join  = 950.0

            # zenith wavelength
            #/----------------------------------------------------------------------------\#
            logic_zen_si = (wvls['zen|si'] >= wvl_start) & (wvls['zen|si'] <= wvl_join)
            logic_zen_in = (wvls['zen|in'] >  wvl_join)  & (wvls['zen|in'] <= wvl_end)

            wvl_zen = np.concatenate((wvls['zen|si'][logic_zen_si], wvls['zen|in'][logic_zen_in]))

            indices_sort_zen = np.argsort(wvl_zen)
            wvl_zen = wvl_zen[indices_sort_zen]
            #\----------------------------------------------------------------------------/#

            # nadir wavelength
            #/----------------------------------------------------------------------------\#
            logic_nad_si = (wvls['nad|si'] >= wvl_start) & (wvls['nad|si'] <= wvl_join)
            logic_nad_in = (wvls['nad|in'] >  wvl_join)  & (wvls['nad|in'] <= wvl_end)

            wvl_nad = np.concatenate((wvls['nad|si'][logic_nad_si], wvls['nad|in'][logic_nad_in]))

            indices_sort_nad = np.argsort(wvl_nad)
            wvl_nad = wvl_nad[indices_sort_nad]
            #\----------------------------------------------------------------------------/#

            fnames_zen = sorted(glob.glob('%s/cal/*RAD-CAL-PRI|LASP|%s|ZEN|%s*.h5' % (ssfr.common.fdir_data, _ssfr_.upper(), dset_s.upper())))
            fnames_nad = sorted(glob.glob('%s/cal/*RAD-CAL-PRI|LASP|%s|NAD|%s*.h5' % (ssfr.common.fdir_data, _ssfr_.upper(), dset_s.upper())))
            if len(fnames_zen) == 1 and len(fnames_nad) == 1:
                fname_zen = fnames_zen[0]
                fname_nad = fnames_nad[0]

                f_zen = h5py.File(fname_zen, 'r')
                sec_resp_zen_si = f_zen['zen|si'][...]
                sec_resp_zen_in = f_zen['zen|in'][...]
                f_zen.close()

                f_nad = h5py.File(fname_nad, 'r')
                sec_resp_nad_si = f_nad['nad|si'][...]
                sec_resp_nad_in = f_nad['nad|in'][...]
                f_nad.close()

                sec_resp_zen = np.concatenate((sec_resp_zen_si[logic_zen_si], sec_resp_zen_in[logic_zen_in]))[indices_sort_zen]
                sec_resp_nad = np.concatenate((sec_resp_nad_si[logic_nad_si], sec_resp_nad_in[logic_nad_in]))[indices_sort_nad]
            #\----------------------------------------------------------------------------/#

            # zenith
            #/--------------------------------------------------------------\#
            cnt_zen = f_['%s/cnt_zen' % dset_s][...]
            wvl_zen = f_['%s/wvl_zen' % dset_s][...]

            # sec_resp_zen = np.interp(wvl_zen, wvl_resp_zen_, sec_resp_zen_)

            flux_zen = cnt_zen.copy()
            for i in range(tmhr.size):
                if np.isnan(cnt_zen[i, :]).sum() == 0:
                    flux_zen[i, :] = cnt_zen[i, :] / sec_resp_zen
            #\--------------------------------------------------------------/#

            # nadir
            #/--------------------------------------------------------------\#
            cnt_nad = f_['%s/cnt_nad' % dset_s][...]
            wvl_nad = f_['%s/wvl_nad' % dset_s][...]

            # sec_resp_nad = np.interp(wvl_nad, wvl_resp_nad_, sec_resp_nad_)

            flux_nad = cnt_nad.copy()
            for i in range(tmhr.size):
                if np.isnan(cnt_nad[i, :]).sum() == 0:
                    flux_nad[i, :] = cnt_nad[i, :] / sec_resp_nad
            #\--------------------------------------------------------------/#

            g = f.create_group(dset_s)
            g['flux_zen'] = flux_zen
            g['flux_nad'] = flux_nad
            g['wvl_zen']  = wvl_zen
            g['wvl_nad']  = wvl_nad

        else:

            f[dset_s] = f_[dset_s][...]

    f_.close()

    f.close()

    return
#\----------------------------------------------------------------------------/#




# functions for processing HSK and ALP
#/----------------------------------------------------------------------------\#
def cdata_arcsix_hsk_v0(
        date,
        fdir_data=_fdir_data_,
        fdir_out=_fdir_out_,
        run=True,
        ):

    """
    For processing aricraft housekeeping file

    Notes:
        The housekeeping data would require some corrections before its release by the
        data system team, we usually request the raw IWG file (similar data but with a
        slightly different data formatting) from the team right after each flight to
        facilitate our data processing in a timely manner.
    """

    date_s = date.strftime('%Y%m%d')

    # this would change if we are processing IWG file
    #/--------------------------------------------------------------\#
    fname = ssfr.util.get_all_files(fdir_data, pattern='*%4.4d*%2.2d*%2.2d*.ict' % (date.year, date.month, date.day))[0]
    data_hsk = ssfr.util.read_ict(fname)

    var_dict = {
            'lon': 'longitude',
            'lat': 'latitude',
            'alt': 'gps_altitude',
            'tmhr': 'tmhr',
            'ang_pit': 'pitch_angle',
            'ang_rol': 'roll_angle',
            'ang_hed': 'true_heading',
            }
    #\--------------------------------------------------------------/#

    # fake hsk for skywatch
    #/----------------------------------------------------------------------------\#
    # tmhr = np.arange(tmhr_range[0]*3600.0, tmhr_range[-1]*3600.0, 1.0)/3600.0
    # lon0 = -105.24227862207863 # skywatch longitude
    # lat0 =  40.01097849056196  # skywatch latitude
    # alt0 =  4.0                # skywatch altitude
    # pit0 = 0.0
    # rol0 = 0.0
    # hed0 = 0.0
    # data_hsk = {
    #         'tmhr': {'data': tmhr, 'units': 'hour'},
    #         'long': {'data': np.repeat(lon0, tmhr.size), 'units': 'degree'},
    #         'lat' : {'data': np.repeat(lat0, tmhr.size), 'units': 'degree'},
    #         'palt': {'data': np.repeat(alt0, tmhr.size), 'units': 'meter'},
    #         'pitch'   : {'data': np.repeat(pit0, tmhr.size), 'units': 'degree'},
    #         'roll'    : {'data': np.repeat(rol0, tmhr.size), 'units': 'degree'},
    #         'heading' : {'data': np.repeat(hed0, tmhr.size), 'units': 'degree'},
    #         }
    #\----------------------------------------------------------------------------/#

    fname_h5 = '%s/%s-%s_%s_v0.h5' % (fdir_out, _mission_.upper(), _hsk_.upper(), date_s)
    if run:

        # solar geometries
        #/----------------------------------------------------------------------------\#
        jday0 = ssfr.util.dtime_to_jday(date)
        jday  = jday0 + data_hsk[var_dict['tmhr']]['data']/24.0
        sza, saa = ssfr.util.cal_solar_angles(jday, data_hsk[var_dict['lon']]['data'], data_hsk[var_dict['lat']]['data'], data_hsk[var_dict['alt']]['data'])
        #\----------------------------------------------------------------------------/#

        # save processed data
        #/----------------------------------------------------------------------------\#
        f = h5py.File(fname_h5, 'w')
        for var in var_dict.keys():
            f[var] = data_hsk[var_dict[var]]['data']
        f['jday'] = jday
        f['sza']  = sza
        f['saa']  = saa
        f.close()
        #\----------------------------------------------------------------------------/#

    return fname_h5

def cdata_arcsix_alp_v0(
        date,
        fdir_data=_fdir_data_,
        fdir_out=_fdir_out_,
        run=True,
        ):

    """
    v0: directly read raw ALP (Active Leveling Platform) data

    Notes:
        ALP raw data has a finer temporal resolution than 1Hz and a higher measurement
        precision (or sensitivity) of the aircraft attitude.
    """

    date_s = date.strftime('%Y%m%d')

    # read ALP raw data
    #/----------------------------------------------------------------------------\#
    fname_h5 = '%s/%s-%s_%s_v0.h5' % (fdir_out, _mission_.upper(), _alp_.upper(), date_s)
    if run:
        fnames_alp = ssfr.util.get_all_files(fdir_data, pattern='*.plt3')
        if _verbose_:
            msg = '\nProcessing %s files:\n%s' % (_alp_.upper(), '\n'.join(fnames_alp))
            print(msg)

        alp0 = ssfr.lasp_alp.read_alp(fnames_alp, date=date)
        alp0.save_h5(fname_h5)
    #\----------------------------------------------------------------------------/#

    return os.path.abspath(fname_h5)

def cdata_arcsix_alp_v1(
        date,
        fname_v0,
        fname_hsk,
        fdir_out=_fdir_out_,
        run=True
        ):

    """
    v1:
    1) calculate time offset (seconds) between aircraft housekeeping data and ALP raw data
       (referencing to aircraft housekeeping)
    2) interpolate raw alp data to aircraft housekeeping time

    Notes:
        ALP raw data has a finer temporal resolution than 1Hz and a higher measurement
        precision (or sensitivity) of the aircraft attitude.
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_v1.h5' % (fdir_out, _mission_.upper(), _alp_.upper(), date_s)
    if run:

        # calculate time offset
        #/----------------------------------------------------------------------------\#
        data_hsk = ssfr.util.load_h5(fname_hsk)
        data_alp = ssfr.util.load_h5(fname_v0)

        time_step = 1.0 # 1Hz data
        data_ref = data_hsk['alt']
        data_tar = ssfr.util.interp(data_hsk['jday'], data_alp['jday'], data_alp['alt'])
        time_offset = time_step * ssfr.util.cal_step_offset(data_ref, data_tar)

        print('Find a time offset of %.2f seconds between %s and %s.' % (time_offset, _alp_.upper(), _hsk_.upper()))
        #\----------------------------------------------------------------------------/#

        f = h5py.File(fname_h5, 'w')
        f.attrs['description'] = 'v1:\n  1) raw data interpolated to HSK time frame;\n  2) time offset (seconds) was calculated and applied.'

        f['tmhr']        = data_hsk['tmhr']
        f['jday']        = data_hsk['jday']
        f['tmhr_ori']    = data_hsk['tmhr'] - time_offset/3600.0
        f['jday_ori']    = data_hsk['jday'] - time_offset/86400.0
        f['time_offset'] = time_offset
        f['sza']         = data_hsk['sza']
        f['saa']         = data_hsk['saa']

        jday_corr        = data_alp['jday'] + time_offset/86400.0
        for vname in data_alp.keys():
            if vname not in ['tmhr', 'jday']:
                f[vname] = ssfr.util.interp(data_hsk['jday'], jday_corr, data_alp[vname])
        f.close()

    return fname_h5

def process_alp_data(date, run=True):

    fdir_out = _fdir_out_
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    fdirs = ssfr.util.get_all_folders(_fdir_data_, pattern='*%4.4d*%2.2d*%2.2d*%s' % (date.year, date.month, date.day, _alp_))
    fdir_data = sorted(fdirs, key=os.path.getmtime)[-1]

    date_s = date.strftime('%Y%m%d')
    fname_hsk_v0 = cdata_arcsix_hsk_v0(date, fdir_data=_fdir_hsk_      , fdir_out=fdir_out, run=run)
    fname_alp_v0 = cdata_arcsix_alp_v0(date, fdir_data=fdir_data       , fdir_out=fdir_out, run=run)
    fname_alp_v1 = cdata_arcsix_alp_v1(date, fname_alp_v0, fname_hsk_v0, fdir_out=fdir_out, run=run)

    _fnames_['%s_hsk_v0' % date_s] = fname_hsk_v0
    _fnames_['%s_alp_v0' % date_s] = fname_alp_v0
    _fnames_['%s_alp_v1' % date_s] = fname_alp_v1
#\----------------------------------------------------------------------------/#




# functions for processing SPNS
#/----------------------------------------------------------------------------\#
def cdata_arcsix_spns_v0(
        date,
        fdir_data=_fdir_data_,
        fdir_out=_fdir_out_,
        run=True,
        ):

    """
    Process raw SPN-S data
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_v0.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), date_s)

    if run:

        # read spn-s raw data
        #/----------------------------------------------------------------------------\#
        fname_dif = ssfr.util.get_all_files(fdir_data, pattern='*Diffuse.txt')[-1]
        data0_dif = ssfr.lasp_spn.read_spns(fname=fname_dif)

        fname_tot = ssfr.util.get_all_files(fdir_data, pattern='*Total.txt')[-1]
        data0_tot = ssfr.lasp_spn.read_spns(fname=fname_tot)

        msg = 'Processing %s data:\n%s\n%s\n' % (_spns_.upper(), fname_dif, fname_tot)
        print(msg)
        #/----------------------------------------------------------------------------\#

        # read wavelengths and calculate toa downwelling solar flux
        #/----------------------------------------------------------------------------\#
        flux_toa = ssfr.util.get_solar_kurudz()

        wvl_tot = data0_tot.data['wvl']
        f_dn_sol_tot = np.zeros_like(wvl_tot)
        for i, wvl0 in enumerate(wvl_tot):
            f_dn_sol_tot[i] = ssfr.util.cal_weighted_flux(wvl0, flux_toa[:, 0], flux_toa[:, 1])
        #\----------------------------------------------------------------------------/#

        f = h5py.File(fname_h5, 'w')

        g1 = f.create_group('dif')
        for key in data0_dif.data.keys():
            if key in ['tmhr', 'jday', 'wvl', 'flux']:
                dset0 = g1.create_dataset(key, data=data0_dif.data[key], compression='gzip', compression_opts=9, chunks=True)

        g2 = f.create_group('tot')
        for key in data0_tot.data.keys():
            if key in ['tmhr', 'jday', 'wvl', 'flux']:
                dset0 = g2.create_dataset(key, data=data0_tot.data[key], compression='gzip', compression_opts=9, chunks=True)
        g2['toa0'] = f_dn_sol_tot

        f.close()

    return fname_h5

def cdata_arcsix_spns_v1(
        date,
        fname_spns_v0,
        fname_hsk,
        fdir_out=_fdir_out_,
        time_offset=0.0,
        run=True,
        ):

    """
    Check for time offset and merge SPN-S data with aircraft data
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_v1.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), date_s)

    if run:
        # read spn-s v0
        #/----------------------------------------------------------------------------\#
        data_spns_v0 = ssfr.util.load_h5(fname_spns_v0)
        #/----------------------------------------------------------------------------\#

        # read hsk v0
        #/----------------------------------------------------------------------------\#
        data_hsk= ssfr.util.load_h5(fname_hsk)
        #\----------------------------------------------------------------------------/#

        # calculate time offset
        #/----------------------------------------------------------------------------\#
        time_step = 1.0 # 1Hz data
        index_wvl = np.argmin(np.abs(555.0-data_spns_v0['tot/wvl']))
        data_ref = data_spns_v0['tot/toa0'][index_wvl] * np.cos(np.deg2rad(data_hsk['sza']))
        data_tar  = ssfr.util.interp(data_hsk['jday'], data_spns_v0['tot/jday'], data_spns_v0['tot/flux'][:, index_wvl])
        data_tar_ = ssfr.util.interp(data_hsk['jday'], data_spns_v0['dif/jday'], data_spns_v0['dif/flux'][:, index_wvl])
        diff_ratio = data_tar_/data_tar
        data_tar[(diff_ratio>0.1)|(diff_ratio<0.0)] = 0.0
        # time_offset = time_step * ssfr.util.cal_step_offset(data_ref, data_tar, offset_range=[-6000, -3000])
        if _test_mode_:
            time_offset = -3520.0
        # figure
        #/----------------------------------------------------------------------------\#
        if False:
            plt.close('all')
            fig = plt.figure(figsize=(8, 6))
            # fig.suptitle('Figure')
            # plot
            #/--------------------------------------------------------------\#
            ax1 = fig.add_subplot(111)
            # cs = ax1.imshow(.T, origin='lower', cmap='jet', zorder=0) #, extent=extent, vmin=0.0, vmax=0.5)
            # ax1.scatter(x, y, s=6, c='k', lw=0.0)
            # ax1.hist(.ravel(), bins=100, histtype='stepfilled', alpha=0.5, color='black')
            ax1.plot(data_ref, color='k', ls='-')
            ax1.plot(data_tar, color='r', ls='-')
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

        print('Find a time offset of %.2f seconds between %s and %s.' % (time_offset, _spns_.upper(), _hsk_.upper()))
        #\----------------------------------------------------------------------------/#

        # interpolate spn-s data to hsk time frame
        #/----------------------------------------------------------------------------\#
        flux_dif = np.zeros((data_hsk['jday'].size, data_spns_v0['dif/wvl'].size), dtype=np.float64)
        for i in range(flux_dif.shape[-1]):
            flux_dif[:, i] = ssfr.util.interp(data_hsk['jday'], data_spns_v0['dif/jday']+time_offset/86400.0, data_spns_v0['dif/flux'][:, i])

        flux_tot = np.zeros((data_hsk['jday'].size, data_spns_v0['tot/wvl'].size), dtype=np.float64)
        for i in range(flux_tot.shape[-1]):
            flux_tot[:, i] = ssfr.util.interp(data_hsk['jday'], data_spns_v0['tot/jday']+time_offset/86400.0, data_spns_v0['tot/flux'][:, i])
        #\----------------------------------------------------------------------------/#

        f = h5py.File(fname_h5, 'w')

        for key in data_hsk.keys():
            f[key] = data_hsk[key]

        f['time_offset'] = time_offset
        f['tmhr_ori'] = data_hsk['tmhr'] - time_offset/3600.0
        f['jday_ori'] = data_hsk['jday'] - time_offset/86400.0

        g1 = f.create_group('dif')
        g1['wvl']   = data_spns_v0['dif/wvl']
        dset0 = g1.create_dataset('flux', data=flux_dif, compression='gzip', compression_opts=9, chunks=True)

        g2 = f.create_group('tot')
        g2['wvl']   = data_spns_v0['tot/wvl']
        g2['toa0']  = data_spns_v0['tot/toa0']
        dset0 = g2.create_dataset('flux', data=flux_tot, compression='gzip', compression_opts=9, chunks=True)

        f.close()

    return fname_h5

def cdata_arcsix_spns_v2(
        date,
        fname_spns_v1,
        fname_hsk, # interchangable with fname_alp_v1
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        fdir_out=_fdir_out_,
        run=True,
        ):

    """
    Apply attitude correction to account for aircraft attitude (pitch, roll, heading)
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_v2.h5' % (fdir_out, _mission_.upper(), _spns_.upper(), date_s)

    if run:

        # read spn-s v1
        #/----------------------------------------------------------------------------\#
        data_spns_v1 = ssfr.util.load_h5(fname_spns_v1)
        #/----------------------------------------------------------------------------\#

        # read hsk v0
        #/----------------------------------------------------------------------------\#
        data_hsk = ssfr.util.load_h5(fname_hsk)
        #/----------------------------------------------------------------------------\#

        # correction factor
        #/----------------------------------------------------------------------------\#
        mu = np.cos(np.deg2rad(data_hsk['sza']))

        try:
            iza, iaa = ssfr.util.prh2za(data_hsk['ang_pit_a']+ang_pit_offset, data_hsk['ang_rol_a']+ang_rol_offset, data_hsk['ang_hed'])
        except Exception as error:
            print(error)
            iza, iaa = ssfr.util.prh2za(data_hsk['ang_pit']+ang_pit_offset, data_hsk['ang_rol']+ang_rol_offset, data_hsk['ang_hed'])
        dc = ssfr.util.muslope(data_hsk['sza'], data_hsk['saa'], iza, iaa)

        factors = mu / dc
        #\----------------------------------------------------------------------------/#

        # attitude correction
        #/----------------------------------------------------------------------------\#
        f_dn_dir = data_spns_v1['tot/flux'] - data_spns_v1['dif/flux']
        f_dn_dir_corr = np.zeros_like(f_dn_dir)
        f_dn_tot_corr = np.zeros_like(f_dn_dir)
        for iwvl in range(data_spns_v1['tot/wvl'].size):
            f_dn_dir_corr[..., iwvl] = f_dn_dir[..., iwvl]*factors
            f_dn_tot_corr[..., iwvl] = f_dn_dir_corr[..., iwvl] + data_spns_v1['dif/flux'][..., iwvl]
        #\----------------------------------------------------------------------------/#

        f = h5py.File(fname_h5, 'w')

        for key in data_hsk.keys():
            f[key] = data_hsk[key]

        g0 = f.create_group('att_corr')
        g0['mu'] = mu
        g0['dc'] = dc
        g0['factors'] = factors

        g1 = f.create_group('dif')
        g1['wvl']   = data_spns_v1['dif/wvl']
        dset0 = g1.create_dataset('flux', data=data_spns_v1['dif/flux'], compression='gzip', compression_opts=9, chunks=True)

        g2 = f.create_group('tot')
        g2['wvl']   = data_spns_v1['tot/wvl']
        g2['toa0']  = data_spns_v1['tot/toa0']
        dset0 = g2.create_dataset('flux', data=f_dn_tot_corr, compression='gzip', compression_opts=9, chunks=True)

        f.close()

    return fname_h5

def process_spns_data(date, run=True):

    """
    v0: raw data directly read out from the data files
    v1: data collocated/synced to aircraft nav
    v2: attitude corrected data
    """

    fdir_out = _fdir_out_
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    fdirs = ssfr.util.get_all_folders(_fdir_data_, pattern='*%4.4d*%2.2d*%2.2d*%s' % (date.year, date.month, date.day, _spns_))
    fdir_data = sorted(fdirs, key=os.path.getmtime)[-1]

    date_s = date.strftime('%Y%m%d')

    fname_spns_v0 = cdata_arcsix_spns_v0(date, fdir_data=fdir_data,
            fdir_out=fdir_out, run=run)
    fname_spns_v1 = cdata_arcsix_spns_v1(date, fname_spns_v0, _fnames_['%s_hsk_v0' % date_s],
            fdir_out=fdir_out, run=run)
    fname_spns_v2 = cdata_arcsix_spns_v2(date, fname_spns_v1, _fnames_['%s_alp_v1' % date_s],
            fdir_out=fdir_out, run=run)
    # fname_spns_v2 = cdata_arcsix_spns_v2(date, fname_spns_v1, _fnames_['%s_hsk_v0' % date_s],
    #         fdir_out=fdir_out, run=run)

    _fnames_['%s_spns_v0' % date_s] = fname_spns_v0
    _fnames_['%s_spns_v1' % date_s] = fname_spns_v1
    _fnames_['%s_spns_v2' % date_s] = fname_spns_v2
#\----------------------------------------------------------------------------/#




# functions for processing SSFR
#/----------------------------------------------------------------------------\#
def cdata_arcsix_ssfr_v0(
        date,
        fdir_data=_fdir_data_,
        fdir_out=_fdir_out_,
        which_ssfr='ssfr-a',
        run=True,
        ):

    """
    version 0: counts after dark correction
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_v0.h5' % (fdir_out, _mission_.upper(), which_ssfr.upper(), date_s)

    if run:
        fnames_ssfr = ssfr.util.get_all_files(fdir_data, pattern='*.SKS')
        if _verbose_:
            msg = '\nProcessing %s files:\n%s' % (which_ssfr.upper(), '\n'.join(fnames_ssfr))
            print(msg)

        ssfr0 = ssfr.lasp_ssfr.read_ssfr(fnames_ssfr, dark_corr_mode='interp', which_ssfr='lasp|%s' % which_ssfr.lower())

        # data that are useful
        #   wvl_zen [nm]
        #   cnt_zen [counts/ms]
        #   wvl_nad [nm]
        #   cnt_nad [counts/ms]
        #/----------------------------------------------------------------------------\#
        f = h5py.File(fname_h5, 'w')

        for i in range(ssfr0.Ndset):
            dset_s = 'dset%d' % i
            data = getattr(ssfr0, dset_s)
            g = f.create_group(dset_s)
            for key in data.keys():
                if key != 'info':
                    dset0 = g.create_dataset(key, data=data[key], compression='gzip', compression_opts=9, chunks=True)

        f.close()
        #\----------------------------------------------------------------------------/#

    return fname_h5

def cdata_arcsix_ssfr_v1(
        date,
        fname_ssfr_v0,
        fname_hsk,
        fdir_out=_fdir_out_,
        time_offset=0.0,
        which_ssfr='ssfr-a',
        run=True,
        ):

    """
    version 1: 1) time adjustment          : check for time offset and merge SSFR data with aircraft housekeeping data
               2) time synchronization     : interpolate raw SSFR data into the time frame of the housekeeping data
               3) counts-to-flux conversion: apply primary and secondary calibration to convert counts to fluxes
    """

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_v1.h5' % (fdir_out, _mission_.upper(), which_ssfr.upper(), date_s)

    if run:

        # load ssfr v0 data
        #/----------------------------------------------------------------------------\#
        data_ssfr_v0 = ssfr.util.load_h5(fname_ssfr_v0)
        #\----------------------------------------------------------------------------/#


        # load hsk
        #/----------------------------------------------------------------------------\#
        data_hsk = ssfr.util.load_h5(fname_hsk)
        #\----------------------------------------------------------------------------/#


        # time offset (currently we do it manually)
        #/----------------------------------------------------------------------------\#
        if _test_mode_:
            time_offset = (data_hsk['jday'][0] - data_ssfr_v0['dset0/jday'][0]) * 86400.0
        #\----------------------------------------------------------------------------/#


        # read wavelengths and calculate toa downwelling solar flux
        #/----------------------------------------------------------------------------\#
        flux_toa = ssfr.util.get_solar_kurudz()

        wvl_zen = data_ssfr_v0['dset0/wvl_zen']
        f_dn_sol_zen = np.zeros_like(wvl_zen)
        for i, wvl0 in enumerate(wvl_zen):
            f_dn_sol_zen[i] = ssfr.util.cal_weighted_flux(wvl0, flux_toa[:, 0], flux_toa[:, 1])
        #\----------------------------------------------------------------------------/#


        f = h5py.File(fname_h5, 'w')

        # processing data - since we have dual integration times, SSFR data with different
        # integration time will be processed seperately
        #/----------------------------------------------------------------------------\#
        for dset_s in ['dset0', 'dset1']:

            # select calibration file
            #/----------------------------------------------------------------------------\#
            fdir_cal = '%s/rad-cal' % _fdir_cal_
            fname_cal_zen = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324|*lamp-150c_after-pri|*%s*%s*zen*' % (dset_s, which_ssfr.lower())), key=os.path.getmtime)[-1]
            data_cal_zen = ssfr.util.load_h5(fname_cal_zen)

            fname_cal_nad = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*lamp-1324|*lamp-150c_after-pri|*%s*%s*nad*' % (dset_s, which_ssfr.lower())), key=os.path.getmtime)[-1]
            data_cal_nad = ssfr.util.load_h5(fname_cal_nad)
            #\----------------------------------------------------------------------------/#


            # interpolate ssfr data to hsk time frame
            # and convert counts to flux
            #/----------------------------------------------------------------------------\#
            jday_ssfr_v0 = data_ssfr_v0['%s/jday' % dset_s] + time_offset/86400.0

            wvl_zen  = data_ssfr_v0['%s/wvl_zen' % dset_s]
            cnt_zen  = np.zeros((data_hsk['jday'].size, wvl_zen.size), dtype=np.float64)
            flux_zen = np.zeros_like(cnt_zen)
            for i in range(wvl_zen.size):
                cnt_zen[:, i]  = ssfr.util.interp(data_hsk['jday'], jday_ssfr_v0, data_ssfr_v0['%s/cnt_zen' % dset_s][:, i])
                flux_zen[:, i] = cnt_zen[:, i] / data_cal_zen['sec_resp'][i]

            wvl_nad  = data_ssfr_v0['%s/wvl_nad' % dset_s]
            cnt_nad  = np.zeros((data_hsk['jday'].size, wvl_nad.size), dtype=np.float64)
            flux_nad = np.zeros_like(cnt_nad)
            for i in range(wvl_nad.size):
                cnt_nad[:, i]  = ssfr.util.interp(data_hsk['jday'], jday_ssfr_v0, data_ssfr_v0['%s/cnt_nad' % dset_s][:, i])
                flux_nad[:, i] = cnt_nad[:, i] / data_cal_nad['sec_resp'][i]
            #\----------------------------------------------------------------------------/#

            #\----------------------------------------------------------------------------/#


            # save processed data
            #/----------------------------------------------------------------------------\#
            g = f.create_group(dset_s)
            dset0 = g.create_dataset('wvl_zen' , data=wvl_zen     , compression='gzip', compression_opts=9, chunks=True)
            dset0 = g.create_dataset('cnt_zen' , data=cnt_zen     , compression='gzip', compression_opts=9, chunks=True)
            dset0 = g.create_dataset('flux_zen', data=flux_zen    , compression='gzip', compression_opts=9, chunks=True)
            dset0 = g.create_dataset('toa0'    , data=f_dn_sol_zen, compression='gzip', compression_opts=9, chunks=True)

            dset0 = g.create_dataset('wvl_nad' , data=wvl_nad     , compression='gzip', compression_opts=9, chunks=True)
            dset0 = g.create_dataset('cnt_nad' , data=cnt_nad     , compression='gzip', compression_opts=9, chunks=True)
            dset0 = g.create_dataset('flux_nad', data=flux_nad    , compression='gzip', compression_opts=9, chunks=True)
            #\----------------------------------------------------------------------------/#

        #\----------------------------------------------------------------------------/#



        # save processed data
        #/----------------------------------------------------------------------------\#
        for key in data_hsk.keys():
            f[key] = data_hsk[key]

        f['time_offset'] = time_offset
        f['tmhr_ori'] = data_hsk['tmhr'] - time_offset/3600.0
        f['jday_ori'] = data_hsk['jday'] - time_offset/86400.0

        f.close()
        #\----------------------------------------------------------------------------/#

    return fname_h5

def cdata_arcsix_ssfr_v2(
        date,
        fname_ssfr_v1,
        fname_alp_v1,
        fname_spns_v2,
        which_ssfr='ssfr-a',
        fdir_out=_fdir_out_,
        ang_pit_offset=0.0,
        ang_rol_offset=0.0,
        run=True,
        run_aux=False,
        ):

    """
    version 2: apply cosine correction to correct for non-linear angular resposne
               diffuse radiation: use cosine response integrated over the entire angular space
               direct radiation: use cosine response over the entire angular space measured in the lab

               diffuse and direct seperation is guided by the diffuse ratio measured by SPNS
    """

    def func_diff_ratio(x, a, b, c):

        return a * (x/500.0)**(b) + c

    def fit_diff_ratio(wavelength, ratio):

        popt, pcov = curve_fit(func_diff_ratio, wavelength, ratio, maxfev=1000000, bounds=(np.array([0.0, -np.inf, 0.0]), np.array([np.inf, 0.0, np.inf])))

        return popt, pcov

    date_s = date.strftime('%Y%m%d')

    fname_h5 = '%s/%s-%s_%s_v2.h5' % (fdir_out, _mission_.upper(), which_ssfr.upper(), date_s)

    if run:

        data_ssfr_v1 = ssfr.util.load_h5(fname_ssfr_v1)

        fname_aux = '%s/%s-%s_%s_v2-aux.h5' % (fdir_out, _mission_.upper(), which_ssfr.upper(), date_s)

        if run_aux:

            # calculate diffuse/global ratio from SPNS data
            #/----------------------------------------------------------------------------\#
            data_spns_v2 = ssfr.util.load_h5(fname_spns_v2)

            f_ = h5py.File(fname_aux, 'w')

            wvl_ssfr_zen = data_ssfr_v1['dset0/wvl_zen']
            wvl_spns     = data_spns_v2['tot/wvl']

            Nt, Nwvl = data_ssfr_v1['dset0/flux_zen'].shape

            diff_ratio = np.zeros((Nt, Nwvl), dtype=np.float64)
            diff_ratio[...] = np.nan

            poly_coefs = np.zeros((Nt, 3), dtype=np.float64)
            poly_coefs[...] = np.nan

            qual_flag = np.repeat(0, Nt)

            # do spectral fit based on 400 nm - 750 nm observations
            #/--------------------------------------------------------------\#
            for i in tqdm(range(Nt)):

                diff_ratio0_spns = data_spns_v2['dif/flux'][i, :] / data_spns_v2['tot/flux'][i, :]
                logic_valid = (~np.isnan(diff_ratio0_spns)) & (diff_ratio0_spns>=0.0) & (diff_ratio0_spns<=1.0) & (wvl_spns>=400.0) & (wvl_spns<=750.0)
                if logic_valid.sum() > 20:

                    x = data_spns_v2['tot/wvl'][logic_valid]
                    y = diff_ratio0_spns[logic_valid]
                    popt, pcov = fit_diff_ratio(x, y)

                    diff_ratio[i, :] = func_diff_ratio(wvl_ssfr_zen, *popt)
                    poly_coefs[i, :] = popt

                    qual_flag[i] = 1

            diff_ratio[diff_ratio<0.0] = 0.0
            diff_ratio[diff_ratio>1.0] = 1.0
            #\--------------------------------------------------------------/#

            # fill in nan values in time space
            #/--------------------------------------------------------------\#
            for i in range(Nwvl):

                logic_nan   = np.isnan(diff_ratio[:, i])
                logic_valid = ~logic_nan
                f_interp = interpolate.interp1d(data_ssfr_v1['tmhr'][logic_valid], diff_ratio[:, i][logic_valid], bounds_error=None, fill_value='extrapolate')
                diff_ratio[logic_nan, i] = f_interp(data_ssfr_v1['tmhr'][logic_nan])

            diff_ratio[diff_ratio<0.0] = 0.0
            diff_ratio[diff_ratio>1.0] = 1.0
            #\--------------------------------------------------------------/#

            # save data
            #/--------------------------------------------------------------\#
            f_.create_dataset('diff_ratio', data=diff_ratio  , compression='gzip', compression_opts=9, chunks=True)
            g_ = f_.create_group('diff_ratio_aux')
            g_.create_dataset('wvl'       , data=wvl_ssfr_zen, compression='gzip', compression_opts=9, chunks=True)
            g_.create_dataset('coef'      , data=poly_coefs  , compression='gzip', compression_opts=9, chunks=True)
            g_.create_dataset('qual_flag' , data=qual_flag   , compression='gzip', compression_opts=9, chunks=True)
            #\--------------------------------------------------------------/#
            #\----------------------------------------------------------------------------/#


            # alp
            #/----------------------------------------------------------------------------\#
            data_alp_v1  = ssfr.util.load_h5(fname_alp_v1)
            for key in data_alp_v1.keys():
                try:
                    f_.create_dataset(key, data=data_alp_v1[key], compression='gzip', compression_opts=9, chunks=True)
                except TypeError as error:
                    print(error)
                    f_[key] = data_alp_v1[key]
            f_.close()
            #\----------------------------------------------------------------------------/#

        data_aux = ssfr.util.load_h5(fname_aux)

        # diffuse ratio
        #/----------------------------------------------------------------------------\#
        diff_ratio = data_aux['diff_ratio']
        #\----------------------------------------------------------------------------/#


        # angles
        #/----------------------------------------------------------------------------\#
        angles = {}
        angles['sza'] = data_aux['sza']
        angles['saa'] = data_aux['saa']
        angles['ang_pit']        = data_aux['ang_pit_s'] # pitch angle from SPAN-CPT
        angles['ang_rol']        = data_aux['ang_rol_s'] # roll angle from SPAN-CPT
        angles['ang_hed']        = data_aux['ang_hed']
        angles['ang_pit_m']      = data_aux['ang_pit_m']
        angles['ang_rol_m']      = data_aux['ang_rol_m']
        angles['ang_pit_offset'] = ang_pit_offset
        angles['ang_rol_offset'] = ang_rol_offset
        #\----------------------------------------------------------------------------/#

        # save data
        #/----------------------------------------------------------------------------\#
        f = h5py.File(fname_h5, 'w')
        for key in ['tmhr', 'jday', 'lon', 'lat', 'alt', 'sza', 'saa', 'ang_pit_s', 'ang_rol_s', 'ang_hed', 'ang_pit_m', 'ang_rol_m']:
            f.create_dataset(key, data=data_aux[key], compression='gzip', compression_opts=9, chunks=True)

        for dset_s in ['dset0', 'dset1']:

            g = f.create_group(dset_s)

            # select calibration file for attitude correction
            #/----------------------------------------------------------------------------\#
            fdir_cal = '%s/ang-cal' % _fdir_cal_
            fname_cal_zen = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*vaa-180|*%s*%s*zen*' % (dset_s, which_ssfr.lower())), key=os.path.getmtime)[-1]
            fname_cal_nad = sorted(ssfr.util.get_all_files(fdir_cal, pattern='*vaa-180|*%s*%s*nad*' % (dset_s, which_ssfr.lower())), key=os.path.getmtime)[-1]
            #\----------------------------------------------------------------------------/#

            # calculate attitude correction factors
            #/----------------------------------------------------------------------------\#
            fnames_cal = {
                    'zen': fname_cal_zen,
                    'nad': fname_cal_nad,
                    }
            factors = ssfr.corr.att_corr(fnames_cal, angles, diff_ratio=diff_ratio)
            #\----------------------------------------------------------------------------/#

            g1 = g.create_group('att_corr')
            g1.create_dataset('factors_zen', data=factors['zen'], compression='gzip', compression_opts=9, chunks=True)
            g1.create_dataset('factors_nad', data=factors['nad'], compression='gzip', compression_opts=9, chunks=True)

            # apply attitude correction
            #/----------------------------------------------------------------------------\#
            g.create_dataset('flux_zen', data=data_ssfr_v1['%s/flux_zen' % dset_s]*factors['zen'], compression='gzip', compression_opts=9, chunks=True)
            g.create_dataset('flux_nad', data=data_ssfr_v1['%s/flux_nad' % dset_s]*factors['nad'], compression='gzip', compression_opts=9, chunks=True)
            #\----------------------------------------------------------------------------/#

        f.close()

    return fname_h5

def cdata_arcsix_ssfr_archive():

    # header
    #/----------------------------------------------------------------------------\#
    # comments_list = []
    # comments_list.append('Bandwidth of Silicon channels (wavelength < 950nm) as defined by the FWHM: 6 nm')
    # comments_list.append('Bandwidth of InGaAs channels (wavelength > 950nm) as defined by the FWHM: 12 nm')
    # comments_list.append('Pitch angle offset: %.1f degree' % pitch_angle)
    # comments_list.append('Roll angle offset: %.1f degree' % roll_angle)

    # for key in fnames_rad_cal.keys():
    #     comments_list.append('Radiometric calibration file (%s): %s' % (key, os.path.basename(fnames_rad_cal[key])))
    # for key in fnames_ang_cal.keys():
    #     comments_list.append('Angular calibration file (%s): %s' % (key, os.path.basename(fnames_ang_cal[key])))
    # comments = '\n'.join(comments_list)

    # print(date_s)
    # print(comments)
    # print()
    #\----------------------------------------------------------------------------/#

    # create hsk file for ssfr (nasa data archive)
    #/----------------------------------------------------------------------------\#
    fname_ssfr = '%s/ssfr_%s_hsk.h5' % (fdir_processed, date_s)
    # f = h5py.File(fname_ssfr, 'w')

    # dset = f.create_dataset('comments', data=comments)
    # dset.attrs['description'] = 'comments on the data'

    # dset = f.create_dataset('info', data=version_info)
    # dset.attrs['description'] = 'information on the version'

    # dset = f.create_dataset('utc', data=data_hsk['tmhr'])
    # dset.attrs['description'] = 'universal time (numbers above 24 are for the next day)'
    # dset.attrs['unit'] = 'decimal hour'

    # dset = f.create_dataset('altitude', data=data_hsk['gps_altitude'])
    # dset.attrs['description'] = 'altitude above sea level (GPS altitude)'
    # dset.attrs['unit'] = 'meter'

    # dset = f.create_dataset('longitude', data=data_hsk['longitude'])
    # dset.attrs['description'] = 'longitude'
    # dset.attrs['unit'] = 'degree'

    # dset = f.create_dataset('latitude', data=data_hsk['latitude'])
    # dset.attrs['description'] = 'latitude'
    # dset.attrs['unit'] = 'degree'

    # dset = f.create_dataset('zen_wvl', data=ssfr_v0.zen_wvl)
    # dset.attrs['description'] = 'center wavelengths of zenith channels (bandwidth see info)'
    # dset.attrs['unit'] = 'nm'

    # dset = f.create_dataset('nad_wvl', data=ssfr_v0.nad_wvl)
    # dset.attrs['description'] = 'center wavelengths of nadir channels (bandwidth see info)'
    # dset.attrs['unit'] = 'nm'

    # dset = f.create_dataset('zen_flux', data=zen_flux)
    # dset.attrs['description'] = 'downwelling shortwave spectral irradiance'
    # dset.attrs['unit'] = 'W / m2 / nm'

    # dset = f.create_dataset('nad_flux', data=nad_flux)
    # dset.attrs['description'] = 'upwelling shortwave spectral irradiance'
    # dset.attrs['unit'] = 'W / m2 / nm'

    # dset = f.create_dataset('pitch', data=pitch)
    # dset.attrs['description'] = 'aircraft pitch angle (positive values indicate nose up)'
    # dset.attrs['unit'] = 'degree'

    # dset = f.create_dataset('roll', data=roll)
    # dset.attrs['description'] = 'aircraft roll angle (positive values indicate right wing down)'
    # dset.attrs['unit'] = 'degree'

    # dset = f.create_dataset('heading', data=heading)
    # dset.attrs['description'] = 'aircraft heading angle (positive values clockwise, w.r.t north)'
    # dset.attrs['unit'] = 'degree'

    # dset = f.create_dataset('sza', data=sza)
    # dset.attrs['description'] = 'solar zenith angle'
    # dset.attrs['unit'] = 'degree'

    # f.close()
    #\----------------------------------------------------------------------------/#

    return fname_ssfr

def process_ssfr_data(date, which_ssfr='ssfr-a', run=True):

    fdir_out = _fdir_out_
    if not os.path.exists(fdir_out):
        os.makedirs(fdir_out)

    fdirs = ssfr.util.get_all_folders(_fdir_data_, pattern='*%4.4d*%2.2d*%2.2d*%s' % (date.year, date.month, date.day, _ssfr1_))
    fdir_data = sorted(fdirs, key=os.path.getmtime)[-1]

    date_s = date.strftime('%Y%m%d')

    fname_ssfr_v0 = cdata_arcsix_ssfr_v0(date, fdir_data=fdir_data,
            which_ssfr=which_ssfr, fdir_out=fdir_out, run=False)
    fname_ssfr_v1 = cdata_arcsix_ssfr_v1(date, fname_ssfr_v0, _fnames_['%s_hsk_v0' % date_s],
            which_ssfr=which_ssfr, fdir_out=fdir_out, run=False)
    fname_ssfr_v2 = cdata_arcsix_ssfr_v2(date, fname_ssfr_v1, _fnames_['%s_alp_v1' % date_s], _fnames_['%s_spns_v2' % date_s],
            which_ssfr=which_ssfr, fdir_out=fdir_out, run=run)

    pass
#\----------------------------------------------------------------------------/#




# functions for quicklook
#/----------------------------------------------------------------------------\#
def quicklook_alp(date):

    date_s = date.strftime('%Y%m%d')

    data_hsk_v0 = ssfr.util.load_h5(_fnames_['%s_hsk_v0' % date_s])
    data_alp_v0 = ssfr.util.load_h5(_fnames_['%s_alp_v0' % date_s])
    data_alp_v1 = ssfr.util.load_h5(_fnames_['%s_alp_v1' % date_s])

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('%s Quicklook (%s)' % (_alp_.upper(), date_s))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.scatter(data_hsk_v0['tmhr'], data_hsk_v0['alt'], s=2, c='k', lw=0.0)
        ax1.scatter(data_alp_v0['tmhr'], data_alp_v0['alt'], s=2, c='r', lw=0.0)
        ax1.scatter(data_alp_v1['tmhr'], data_alp_v1['alt'], s=2, c='g', lw=0.0)
        # ax1.scatter(data_alp_v0['tmhr'], data_alp_v0['ang_pit_m'], s=2, c='r', lw=0.0)
        # ax1.scatter(data_alp_v1['tmhr'], data_alp_v1['ang_pit_m'], s=2, c='g', lw=0.0)

        # ax1.set_xlabel('')
        # ax1.set_ylabel('')
        # ax1.set_title('')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_%s.png' % (_metadata['Function'], date_s), bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#

def quicklook_spns(date):

    date_s = date.strftime('%Y%m%d')

    data_spns_v1 = ssfr.util.load_h5(_fnames_['%s_spns_v1' % date_s])
    data_spns_v2 = ssfr.util.load_h5(_fnames_['%s_spns_v2' % date_s])

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('%s Quicklook (%s)' % (_spns_.upper(), date_s))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.scatter(data_spns_v1['tmhr'], data_spns_v1['tot/flux'][:, 100]-data_spns_v1['dif/flux'][:, 100], s=2, c='r', lw=0.0)
        ax1.scatter(data_spns_v2['tmhr'], data_spns_v2['tot/flux'][:, 100]-data_spns_v2['dif/flux'][:, 100], s=2, c='b', lw=0.0)

        # ax1.set_xlabel('')
        # ax1.set_ylabel('')
        # ax1.set_title('')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #\--------------------------------------------------------------/#
        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s_%s.png' % (_metadata['Function'], date_s), bbox_inches='tight', metadata=_metadata)
        plt.show()
        #\--------------------------------------------------------------/#
    #\----------------------------------------------------------------------------/#
#\----------------------------------------------------------------------------/#




# functions for visualization
#/----------------------------------------------------------------------------\#
def plot_time_series(date, wvl0=950.0):

    date_s = date.strftime('%Y-%m-%d')

    fname_h5 = '%s/%s_%s_%s_v2.h5' % (_fdir_out_, _mission_.upper(), _spns_.upper(), date_s)
    f = h5py.File(fname_h5, 'r')
    tmhr = f['tmhr'][...]
    wvl_ = f['tot/wvl'][...]
    flux_spns_tot = f['tot/flux'][...][:, np.argmin(np.abs(wvl_-wvl0))]
    f.close()

    fname_h5 = '%s/%s_%s_%s_v2.h5' % (_fdir_out_, _mission_.upper(), _ssfr_.upper(), date_s)
    f = h5py.File(fname_h5, 'r')
    wvl_ = f['dset0/wvl_zen'][...]
    # flux_ssfr_zen0 = f['dset0/flux_zen'][...][:, np.argmin(np.abs(wvl_-wvl0))] / 4.651062916040369
    flux_ssfr_zen0 = f['dset0/flux_zen'][...][:, np.argmin(np.abs(wvl_-wvl0))]
    wvl_ = f['dset0/wvl_nad'][...]
    # flux_ssfr_nad0 = f['dset0/flux_nad'][...][:, np.argmin(np.abs(wvl_-wvl0))] / 6.755421945458449
    flux_ssfr_nad0 = f['dset0/flux_nad'][...][:, np.argmin(np.abs(wvl_-wvl0))]

    wvl_ = f['dset1/wvl_zen'][...]
    # flux_ssfr_zen1 = f['dset1/flux_zen'][...][:, np.argmin(np.abs(wvl_-wvl0))] / 4.651062916040369
    flux_ssfr_zen1 = f['dset1/flux_zen'][...][:, np.argmin(np.abs(wvl_-wvl0))]
    wvl_ = f['dset1/wvl_nad'][...]
    # flux_ssfr_nad1 = f['dset1/flux_nad'][...][:, np.argmin(np.abs(wvl_-wvl0))] / 6.755421945458449
    flux_ssfr_nad1 = f['dset1/flux_nad'][...][:, np.argmin(np.abs(wvl_-wvl0))]
    f.close()

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.scatter(tmhr, flux_spns_tot, s=6, c='k', lw=0.0)
        ax1.scatter(tmhr, flux_ssfr_zen0, s=3, c='r', lw=0.0)
        ax1.scatter(tmhr, flux_ssfr_zen1, s=3, c='magenta', lw=0.0)
        ax1.scatter(tmhr, flux_ssfr_nad0, s=3, c='b', lw=0.0)
        ax1.scatter(tmhr, flux_ssfr_nad1, s=3, c='cyan', lw=0.0)
        ax1.set_xlabel('Time [Hour]')
        ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.set_title('Skywatch Test (%s, %s, %d nm)' % (_ssfr_.upper(), date_s, wvl0))
        #\--------------------------------------------------------------/#

        patches_legend = [
                          mpatches.Patch(color='black' , label='%s Total' % _spns_.upper()), \
                          mpatches.Patch(color='red'    , label='%s Zenith Si080In250' % _ssfr_.upper()), \
                          mpatches.Patch(color='magenta', label='%s Zenith Si120In350' % _ssfr_.upper()), \
                          mpatches.Patch(color='blue'   , label='%s Nadir Si080In250' % _ssfr_.upper()), \
                          mpatches.Patch(color='cyan'   , label='%s Nadir Si120In350' % _ssfr_.upper()), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
    #\----------------------------------------------------------------------------/#

def plot_spectra(date, tmhr0=20.830):

    date_s = date.strftime('%Y-%m-%d')

    fname_h5 = '%s/%s_%s_%s_v2.h5' % (_fdir_out_, _mission_.upper(), _spns_.upper(), date_s)
    f = h5py.File(fname_h5, 'r')
    tmhr = f['tmhr'][...]
    wvl_spns_tot  = f['tot/wvl'][...]
    flux_spns_tot = f['tot/flux'][...][np.argmin(np.abs(tmhr-tmhr0)), :]
    f.close()

    fname_h5 = '%s/%s_%s_%s_v2.h5' % (_fdir_out_, _mission_.upper(), _ssfr_.upper(), date_s)
    f = h5py.File(fname_h5, 'r')
    # flux_ssfr_zen0 = f['dset0/flux_zen'][...][np.argmin(np.abs(tmhr-tmhr0)), :] / 4.651062916040369
    # flux_ssfr_nad0 = f['dset0/flux_nad'][...][np.argmin(np.abs(tmhr-tmhr0)), :] / 6.755421945458449
    flux_ssfr_zen0 = f['dset0/flux_zen'][...][np.argmin(np.abs(tmhr-tmhr0)), :]
    flux_ssfr_nad0 = f['dset0/flux_nad'][...][np.argmin(np.abs(tmhr-tmhr0)), :]

    wvl_ssfr_zen = f['dset1/wvl_zen'][...]
    wvl_ssfr_nad = f['dset1/wvl_nad'][...]
    # flux_ssfr_zen1 = f['dset1/flux_zen'][...][np.argmin(np.abs(tmhr-tmhr0)), :] / 4.651062916040369
    # flux_ssfr_nad1 = f['dset1/flux_nad'][...][np.argmin(np.abs(tmhr-tmhr0)), :] / 6.755421945458449
    flux_ssfr_zen1 = f['dset1/flux_zen'][...][np.argmin(np.abs(tmhr-tmhr0)), :]
    flux_ssfr_nad1 = f['dset1/flux_nad'][...][np.argmin(np.abs(tmhr-tmhr0)), :]
    f.close()

    # figure
    #/----------------------------------------------------------------------------\#
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        # plot
        #/--------------------------------------------------------------\#
        ax1 = fig.add_subplot(111)
        ax1.scatter(wvl_spns_tot, flux_spns_tot, s=6, c='k', lw=0.0)
        ax1.scatter(wvl_ssfr_zen, flux_ssfr_zen0, s=3, c='r', lw=0.0)
        ax1.scatter(wvl_ssfr_zen, flux_ssfr_zen1, s=3, c='magenta', lw=0.0)
        ax1.scatter(wvl_ssfr_nad, flux_ssfr_nad0, s=3, c='b', lw=0.0)
        ax1.scatter(wvl_ssfr_nad, flux_ssfr_nad1, s=3, c='cyan', lw=0.0)
        ax1.set_xlabel('Wavelength [nm]')
        ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
        ax1.set_title('Skywatch Test (%s, %s, %.4f Hour)' % (_ssfr_.upper(), date_s, tmhr0))
        #\--------------------------------------------------------------/#

        patches_legend = [
                          mpatches.Patch(color='black' , label='%s Total' % _spns_.upper()), \
                          mpatches.Patch(color='red'    , label='%s Zenith Si080In250' % _ssfr_.upper()), \
                          mpatches.Patch(color='magenta', label='%s Zenith Si120In350' % _ssfr_.upper()), \
                          mpatches.Patch(color='blue'   , label='%s Nadir Si080In250' % _ssfr_.upper()), \
                          mpatches.Patch(color='cyan'   , label='%s Nadir Si120In350' % _ssfr_.upper()), \
                         ]
        ax1.legend(handles=patches_legend, loc='upper right', fontsize=12)

        # save figure
        #/--------------------------------------------------------------\#
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        _metadata = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        fig.savefig('%s.png' % _metadata['Function'], bbox_inches='tight', metadata=_metadata)
        #\--------------------------------------------------------------/#
        plt.show()
    #\----------------------------------------------------------------------------/#
#\----------------------------------------------------------------------------/#




# functions for generating quicklook video
#/----------------------------------------------------------------------------\#
def generate_quicklook_video(date):

    # quicklook_alp(date)
    # quicklook_spns(date)
    # ssfr.vis.quicklook_bokeh_spns(_fnames_['%s_spns_v2' % date_s], wvl0=None, tmhr0=None, tmhr_range=None, wvl_range=[350.0, 800.0], tmhr_step=10, wvl_step=5, description=_mission_.upper(), fname_html='%s_ql_%s_v2.html' % (_spns_, date_s))
    pass
#\----------------------------------------------------------------------------/#




# main program
#/----------------------------------------------------------------------------\#
def main_process_data(date, run=True):

    date_s = date.strftime('%Y%m%d')

    # 1&2. aircraft housekeeping file (need to request data from the P-3 data system)
    #      active leveling platform
    #    - longitude
    #    - latitude
    #    - altitude
    #    - UTC time
    #    - pitch angle
    #    - roll angle
    #    - heading angle
    #    - motor pitch angle
    #    - motor roll angle
    process_alp_data(date, run=False)

    # 3. SPNS - irradiance (400nm - 900nm)
    #    - spectral downwelling diffuse
    #    - spectral downwelling global/direct (direct=global-diffuse)
    process_spns_data(date, run=False)

    # 4. SSFR-A - irradiance (350nm - 2200nm)
    #    - spectral downwelling global
    #    - spectral upwelling global
    process_ssfr_data(date, which_ssfr='ssfr-a', run=True)
    sys.exit()

    # 5. SSFR-B - radiance (350nm - 2200nm)
    #    - spectral downwelling global
    #    - spectral upwelling global
    process_ssfr_data(date, which_ssfr='ssfr-b', run=True)

    # 5. SSFR-B - radiance (350nm - 2200nm)
    #    - spectral downwelling global
    #    - spectral upwelling global
    generate_quicklook_video(date)
#\----------------------------------------------------------------------------/#




if __name__ == '__main__':

    warnings.warn('!!!!!!!! Under development !!!!!!!!')

    # main_calibration()

    # data procesing
    #/----------------------------------------------------------------------------\#
    dates = [
             # datetime.datetime(2023, 10, 10),
             # datetime.datetime(2023, 10, 12),
             # datetime.datetime(2023, 10, 13),
             # datetime.datetime(2023, 10, 18), # SPNS-B and SSFR-B at Skywatch
             # datetime.datetime(2023, 10, 19), # SPNS-B and SSFR-B at Skywatch
             # datetime.datetime(2023, 10, 20), # SPNS-B and SSFR-B at Skywatch
             # datetime.datetime(2023, 10, 27), # SPNS-B and SSFR-A at Skywatch
             # datetime.datetime(2023, 10, 30), # SPNS-B and SSFR-A at Skywatch
             # datetime.datetime(2023, 10, 31), # SPNS-B and SSFR-A at Skywatch
             # datetime.datetime(2024, 5, 17), # placeholder for test flight at NASA WFF
             datetime.datetime(2018, 9, 30), # test using oracles data
            ]
    for date in dates:
        main_process_data(date)
    #\----------------------------------------------------------------------------/#

    pass
