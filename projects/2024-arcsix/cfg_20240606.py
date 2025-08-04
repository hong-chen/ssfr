import datetime
import ssfr

# parameters that need frequent change
#╭────────────────────────────────────────────────────────────────────────────╮#
date = datetime.datetime(2024, 6, 6); date_s = date.strftime('%Y%m%d'); date_s_ = date.strftime('%Y-%m-%d')
operator = 'Vikas Nataraja, Jeffery Drouet'
mission = 'arcsix'
year = '2024'
platform = 'p3b'
comments = '6th research flight, performed cloud wall'

hsk_tag  = 'hsk'
alp_tag  = 'alp'
hsr1_tag = 'hsr1-a'
ssfr_tag = 'ssfr-a'
ssrr_tag = 'ssfr-b'

hsk_aka  = 'hsk'
alp_aka  = 'alp'
hsr1_aka = 'hsr1'
ssfr_aka = 'ssfr'
ssrr_aka = 'ssrr'

alp_time_offset  = -18.08
hsr1_time_offset = 0.0
ssfr_time_offset = -180.41
ssrr_time_offset = -256.90

alp_ang_pit_offset = 0.0
alp_ang_rol_offset = 0.0
hsr1_ang_pit_offset = 0.0
hsr1_ang_rol_offset = 0.0

# fdir_data = '/Volumes/argus/field/%s/%s/%s' % (mission, year, platform)
# fdir_cal = '/Volumes/argus/field/%s/cal' % mission
# fdir_out = '/Users/kehi6101/Downloads/ssfr_test/%s/processed' % mission
fdir_data = f'data/{mission}/{year}/{platform}'
fdir_cal = f'data/{mission}/cal'
fdir_out = f'data/{mission}/processed'

# parameters that require extra processing
#╭──────────────────────────────────────────────────────────────╮#
# data directory
#╭────────────────────────────────────────────────╮#
# fdir_hsk = '%s/aux/hsk' % (fdir_data)
fdir_hsk = f'{fdir_data}/aux'
fdir_alp = ssfr.util.get_all_folders(fdir_data, pattern=f'*{date.year:04}*{date.month:02}*{date.day:02}*raw?{alp_tag}')[-1]
fdir_hsr1 = ssfr.util.get_all_folders(fdir_data, pattern=f'*{date.year:04}*{date.month:02}*{date.day:02}*raw?{hsr1_tag}')[-1]
fdir_ssfr = ssfr.util.get_all_folders(fdir_data, pattern=f'*{date.year:04}*{date.month:02}*{date.day:02}*raw?{ssfr_tag}')[-1]
fdir_ssrr = ssfr.util.get_all_folders(fdir_data, pattern=f'*{date.year:04}*{date.month:02}*{date.day:02}*raw?{ssrr_tag}')[-1]
#╰────────────────────────────────────────────────╯#

# data files
#╭────────────────────────────────────────────────╮#
fname_hsk = ssfr.util.get_all_files(fdir_hsk, pattern=f'*{date.year:04}*{date.month:02}*{date.day:02}*.???')[-1]
fnames_alp = ssfr.util.get_all_files(fdir_alp, pattern='*.plt3')
fnames_hsr1 = ssfr.util.get_all_files(fdir_hsr1, pattern='*.txt')
fnames_ssfr = ssfr.util.get_all_files(fdir_ssfr, pattern='*.SKS')
fnames_ssrr = ssfr.util.get_all_files(fdir_ssrr, pattern='*.SKS')

fname_hsk_v0 = f'{fdir_out}/{mission.upper()}-{hsk_aka.upper()}_{platform.upper()}_{date_s}_v0.h5'

fname_alp_v0 = f'{fdir_out}/{mission.upper()}-{alp_aka.upper()}_{platform.upper()}_{date_s}_v0.h5'
fname_alp_v1 = f'{fdir_out}/{mission.upper()}-{alp_aka.upper()}_{platform.upper()}_{date_s}_v1.h5'

fname_hsr1_v0 = f'{fdir_out}/{mission.upper()}-{hsr1_aka.upper()}_{platform.upper()}_{date_s}_v0.h5'
fname_hsr1_v1 = f'{fdir_out}/{mission.upper()}-{hsr1_aka.upper()}_{platform.upper()}_{date_s}_v1.h5'
fname_hsr1_v2 = f'{fdir_out}/{mission.upper()}-{hsr1_aka.upper()}_{platform.upper()}_{date_s}_v2.h5'

fname_ssfr_v0 = f'{fdir_out}/{mission.upper()}-{ssfr_aka.upper()}_{platform.upper()}_{date_s}_v0.h5'
fname_ssfr_v1 = f'{fdir_out}/{mission.upper()}-{ssfr_aka.upper()}_{platform.upper()}_{date_s}_v1.h5'
fname_ssfr_v2 = f'{fdir_out}/{mission.upper()}-{ssfr_aka.upper()}_{platform.upper()}_{date_s}_v2.h5'

fname_ssrr_v0 = f'{fdir_out}/{mission.upper()}-{ssrr_aka.upper()}_{platform.upper()}_{date_s}_v0.h5'
fname_ssrr_v1 = f'{fdir_out}/{mission.upper()}-{ssrr_aka.upper()}_{platform.upper()}_{date_s}_v1.h5'
fname_ssrr_v2 = f'{fdir_out}/{mission.upper()}-{ssrr_aka.upper()}_{platform.upper()}_{date_s}_v2.h5'
#╰────────────────────────────────────────────────╯#

# calibrations
#╭────────────────────────────────────────────────╮#
#╰────────────────────────────────────────────────╯#
#╰──────────────────────────────────────────────────────────────╯#
#╰────────────────────────────────────────────────────────────────────────────╯#


# common settings
#╭────────────────────────────────────────────────────────────────────────────╮#
common = {
        'date': date,
        'date_s': date_s,
        'date_s_': date_s_,
        'mission': mission.lower(),
        'platform': platform.lower(),
        'operator': operator,
        'comments': comments,
        'fdir_out': fdir_out,
        }
#╰────────────────────────────────────────────────────────────────────────────╯#



# House Keeping File
#╭────────────────────────────────────────────────────────────────────────────╮#
hsk = {
        'aka': hsk_aka.lower(),
        'tag': hsk_tag.lower(),
        'fname': fname_hsk,
        'fname_v0': fname_hsk_v0,
        }
#╰────────────────────────────────────────────────────────────────────────────╯#



# Hyper-Spectral Radiometer 1
#╭────────────────────────────────────────────────────────────────────────────╮#
hsr1 = {
        'aka': hsr1_aka.lower(),
        'tag': hsr1_tag.lower(),
        'fnames': fnames_hsr1,
        'fname_v0': fname_hsr1_v0,
        'fname_v1': fname_hsr1_v1,
        'fname_v2': fname_hsr1_v2,
        'time_offset': hsr1_time_offset,
        'ang_pit_offset': hsr1_ang_pit_offset,
        'ang_rol_offset': hsr1_ang_rol_offset,
        }
#╰────────────────────────────────────────────────────────────────────────────╯#



# Active Leveling Platform
#╭────────────────────────────────────────────────────────────────────────────╮#
alp = {
        'aka': alp_aka.lower(),
        'tag': alp_tag.lower(),
        'fnames': fnames_alp,
        'fname_v0': fname_alp_v0,
        'fname_v1': fname_alp_v1,
        'time_offset': alp_time_offset,
        'ang_pit_offset': alp_ang_pit_offset,
        'ang_rol_offset': alp_ang_rol_offset,
        }
#╰────────────────────────────────────────────────────────────────────────────╯#



# Solar Spectral Flux Radiometer
#╭────────────────────────────────────────────────────────────────────────────╮#
ssfr = {
        'aka': ssfr_aka.lower(),

        'tag': ssfr_tag.lower(),

        'fnames': fnames_ssfr,

        'fname_v0': fname_ssfr_v0,
        'fname_v1': fname_ssfr_v1,
        'fname_v2': fname_ssfr_v2,

        # 'fname_rad_cal': fname_rad_cal_ssfr,

        'which_ssfr': f'lasp|{ssfr_tag.lower()}'

        # wavelength setting
        'wvl_s': 350.0,  # beginning/first wavelength [nm] of the selected wavelength range
        'wvl_e': 2000.0, # ending/last wavelength [nm] of the selected wavelength range
        'wvl_j': 950.0,  # joinder wavelength within the overlapping wavelength coverage between Silicon and InGaAs spectrometers

        # time offset [seconds]
        'time_offset': 0.0,

        # number of data points to be excluded at the beginning and end of a dark cycle (due to slow shutter closing/opening glitch)
        'dark_extend': 1,

        # number of data points to be excluded at the beginning and end of a light cycle (due to slow shutter closing/opening glitch)
        'light_extend': 1,

        # dark correction mode: `interp`, linear interpolation using two adjacent dark cycles
        #   also available in `mean`, which uses the average to represent darks
        #   generally, `interp` is preferred
        'dark_corr_mode': 'interp',
        }
#╰────────────────────────────────────────────────────────────────────────────╯#



# Solar Spectral "Radiance" Radiometer
#╭────────────────────────────────────────────────────────────────────────────╮#
ssrr = {
        'aka': ssrr_aka.lower(),

        'tag': ssrr_tag.lower(),

        'fnames': fnames_ssrr,

        'fname_v0': fname_ssrr_v0,
        'fname_v1': fname_ssrr_v1,
        'fname_v2': fname_ssrr_v2,

        'fname_ssrr_rad_cal': None,

        'which_ssfr': f'lasp|{ssrr_tag.lower()}'

        # wavelength setting
        'wvl_s': 350.0,  # beginning/first wavelength [nm] of the selected wavelength range
        'wvl_e': 2000.0, # ending/last wavelength [nm] of the selected wavelength range
        'wvl_j': 950.0,  # joinder wavelength within the overlapping wavelength coverage between Silicon and InGaAs spectrometers

        # time offset [seconds]
        'time_offset': 0.0,

        # number of data points to be excluded at the beginning and end of a dark cycle (due to slow shutter closing/opening glitch)
        'dark_extend': 1,

        # number of data points to be excluded at the beginning and end of a light cycle (due to slow shutter closing/opening glitch)
        'light_extend': 1,

        # dark correction mode: `interp`, linear interpolation using two adjacent dark cycles
        #   also available in `mean`, which uses the average to represent darks
        #   generally, `interp` is preferred
        'dark_corr_mode': 'interp',
        }
#╰────────────────────────────────────────────────────────────────────────────╯#
