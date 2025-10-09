import os
import sys
import copy
import ssfr
import warnings
import h5py
import numpy as np
import pickle as pkl
from scipy import interpolate
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt



__all__ = ['cal_rad_resp', 'cdata_rad_resp', 'rad_resp_corr',]

@dataclass
class InstrumentConfig:
    ssfr_id: str
    lc_id: str
    spec_reverse: bool
    si_tag: str = ""
    in_tag: str = ""
    si_idx: int = 0
    in_idx: int = 1
    int_time: Dict[str, float] = field(default_factory=dict) 
    lamp_id: str = 'f-1324'  # default lamp

    def __post_init__(self):
        # Logic to set tags and indices moves here, out of the main function
        spec_map = {
            ('zen', False): ('zen', 0, 1), ('zen', True): ('nad', 2, 3),
            ('nad', False): ('nad', 2, 3), ('nad', True): ('zen', 0, 1)
        }  
        # --- Check SSFR spectrometer ---
        self.ssfr_id = self.ssfr_id.lower()
        self.lab_id  = self.ssfr_id.split('|')[0]

        # --- Check light collector ---
        self.lc_id = self.lc_id.lower()
        if (self.lc_id  in ['zenith', 'zen', 'z']) | ('zen' in self.lc_id ):
            self.lc_id  = 'zen'
        elif (self.lc_id  in ['nadir', 'nad', 'n']) | ('nad' in self.lc_id ):
            self.lc_id  = 'nad'
        else:
            msg = '\nError [cal_rad_resp]: <which_lc=> does not support <\'%s\'> (only supports <\'zenith, zen, z\'> or <\'nadir, nad, n\'>).' % which_lc
            raise ValueError(msg)
        
        spec, self.si_idx, self.in_idx = spec_map[(self.lc_id, self.spec_reverse)]
        self.si_tag = f"{spec}|si"
        self.in_tag = f"{spec}|in"
        
        if len(self.int_time) == 0:
            self.int_time = {self.si_tag: 80.0, self.in_tag: 250.0}
            
        if self.lab_id == 'nasa':
            import ssfr.nasa_ssfr as ssfr_toolbox
            self.wvls = ssfr_toolbox.get_ssfr_wvl()
        elif self.lab_id == 'lasp':
            import ssfr.lasp_ssfr as ssfr_toolbox
            self.wvls = ssfr_toolbox.get_ssfr_wvl(self.ssfr_id.lower())
        else:
            raise ValueError(f"Unsupported lab: '{self.lab_id}'. Use 'nasa' or 'lasp'.")

        
        self.wvl_si = self.wvls[self.si_tag]
        self.wvl_in = self.wvls[self.in_tag]
        

def planck(wvl, T):
    """
    Calculate spectral radiance using the Planck function.
    Args:
        wvl (float or np.ndarray): Wavelength in nm
        T (float): Temperature in K
    Returns:
        np.ndarray: Spectral radiance in W/m^2/sr/nm
    """
    h = 6.62607015e-34
    c = 2.99792458e8
    k = 1.380649e-23
    wvl_m = np.asarray(wvl) * 1e-9
    exponent = h * c / (wvl_m * k * T)
    spectral_radiance = (2 * h * c**2) / (wvl_m**5) / (np.exp(exponent) - 1)
    return spectral_radiance * 1e-9 * 4 * np.pi


def planck_scaled(wvl, T, scale):
    """Planck function with scaling factor."""
    return scale * planck(wvl, T)


def compute_response(spectra, spectra_std, int_time, resp, resp_std):
    """Compute radiometric response and its uncertainty."""
    spectra = np.where(spectra <= 0.0, np.nan, spectra)
    response = spectra / int_time / resp
    if resp_std is None:
        spectra_std = np.where(spectra_std <= 0.0, np.nan, spectra_std)
        response_std = spectra_std / int_time / resp
    else:
        spectra_std_frac = spectra_std / spectra
        resp_std_frac = resp_std / resp
        response_std = np.sqrt(
            (spectra_std_frac / int_time / resp) ** 2 +
            (spectra / int_time / resp * resp_std_frac) ** 2
        ) * response
    return response, response_std


def cal_rad_resp(
        fnames,
        resp=None,
        instrument_config=InstrumentConfig(ssfr_id='lasp|ssfr-a', lc_id='zen', lamp_id='f-1324', spec_reverse=False),
        int_time={'si':80.0, 'in':250.0},
        lamp_corr=False,
        dark_extend=5,
        light_extend=5,
        verbose=True,
        ):
    """
    Calculate radiometric response for SSFR calibration.
    Args:
        fnames: list of filenames
        resp: response dict or None
        which_ssfr: SSFR spectrometer string
        which_lc: light collector string
        spec_reverse: bool
        which_lamp: lamp string
        int_time: dict of integration times
        lamp_corr: bool
        dark_extend: int
        light_extend: int
        verbose: bool
    Returns:
        dict: radiometric response
    """
        
    # --- SSFR spectrometer selection ---
    config = instrument_config

    if instrument_config.lab_id == 'nasa':
        import ssfr.nasa_ssfr as ssfr_toolbox
    elif instrument_config.lab_id == 'lasp':
        import ssfr.lasp_ssfr as ssfr_toolbox
    else:
        msg = f'\nError [cal_rad_resp]: <which_ssfr=> does not support <\'{instrument_config.ssfr_id}\'> (only supports <\'nasa|ssfr-6\'> or <\'lasp|ssfr-a\'> or <\'lasp|ssfr-b\'>).' 
        raise ValueError(msg)

    index_si = instrument_config.si_idx
    index_in = instrument_config.in_idx
    si_tag = instrument_config.si_tag
    in_tag = instrument_config.in_tag
    int_time = instrument_config.int_time
    
    # --- print message ---
    if verbose:
        msg_type = 'primary response' if resp is None else 'transfer/secondary response'
        msg = f"\nMessage [cal_rad_resp]: processing {msg_type} for <{instrument_config.ssfr_id.upper()}|{instrument_config.lc_id.upper()}|SI-{int_time[si_tag]:.3f}|IN-{int_time[in_tag]:.3f}> ..."
        print(msg)
    
    # --- Get radiometric response ---
    # by default (resp=None), this function will perform primary radiometric calibration
    if resp is None:

        # Lamp selection
        which_lamp = instrument_config.lamp_id.lower()
        if (which_lamp[:4] == 'f-50') or (which_lamp[-3:-1] == '50') or (('50' in which_lamp) and ('150' not in which_lamp)):
            which_lamp = 'f-506c'
        elif (which_lamp[-4:] == '1324') or ('1324' in which_lamp):
            which_lamp = 'f-1324'

        # read in calibrated lamp data and interpolated/integrated at SSFR wavelengths/slits
        fname_lamp = f'{ssfr.common.fdir_data}/lamp/{which_lamp}.dat'
        if not os.path.exists(fname_lamp):
            raise OSError(f"[cal_rad_resp] cannot locate calibration file for lamp <{which_lamp}>.")

        if verbose:
            print(f"\nMessage [cal_rad_resp]: using calibrated lamp <{which_lamp}> with lamp file at \n  <{fname_lamp}>...")

        data = np.loadtxt(fname_lamp)
        data_wvl = data[:, 0]
        data_flux = data[:, 1] * (0.01 if which_lamp == 'f-506c' else 10000.0)      # W m^-2 nm^-1
        
        # apply planck function correction for lamp spectrum (tested with F-1324)
        if which_lamp == 'f-1324' and lamp_corr:
            lamp_fitT = 3139.5  # K
            lamp_scale = 1.4064602e-9
            test_T = lamp_fitT - 25.5
            
            lamp_corr_factor = planck_scaled(data_wvl, test_T, lamp_scale) / planck_scaled(data_wvl, lamp_fitT, lamp_scale)
            data_flux = data_flux * lamp_corr_factor
            
            # save to new lamp file
            fname_lamp_new = f'{ssfr.common.fdir_data}/lamp/{which_lamp}_{test_T:.1f}K.dat'
            with open(fname_lamp_new, 'w') as f:
                for i in range(data_wvl.size):
                    f.write('%.1f    %.6e\n' % (data_wvl[i], data_flux[i]/10000.0))
            
            if verbose:
                msg = f'\nMessage [cal_rad_resp]: applying Planck function correction for lamp <{which_lamp}> with fit temperature of {test_T:.1f} K ...'
                print(msg)


        # get ssfr wavelength for two spectrometers
        wvl_si = instrument_config.wvl_si
        wvl_in = instrument_config.wvl_in

        # use SSFR slit functions to get flux from lamp file
        # the other option is to interpolate the lamp file at SSFR wavelength, which is commented out
        # lamp_nist_si = np.zeros_like(wvl_si)
        # for i in range(lamp_nist_si.size):
        #     lamp_nist_si[i] = ssfr.util.cal_weighted_flux(wvl_si[i], data_wvl, data_flux, slit_func_file='%s/slit/vis_0.1nm_s.dat' % ssfr.common.fdir_data)

        # lamp_nist_in = np.zeros_like(wvl_in)
        # for i in range(lamp_nist_in.size):
        #     lamp_nist_in[i] = ssfr.util.cal_weighted_flux(wvl_in[i], data_wvl, data_flux, slit_func_file='%s/slit/nir_0.1nm_s.dat' % ssfr.common.fdir_data)

        lamp_nist_si = np.interp(wvl_si, data_wvl, data_flux)
        lamp_nist_in = np.interp(wvl_in, data_wvl, data_flux)

        resp = {si_tag: lamp_nist_si,
                in_tag: lamp_nist_in,
                si_tag+'_std': None,
                in_tag+'_std': None}

    # read raw data and perform dark correction and statistics
    try:
        ssfr0 = ssfr_toolbox.read_ssfr(fnames, dark_extend=dark_extend, light_extend=light_extend, verbose=False)

        # Integration time fallbac
        # in case the data does not contain measurement with given integration time
        int_time_si_diff = np.array([ssfr0.dset_info[dset_tag][si_tag] - int_time[si_tag] for dset_tag in ssfr0.dset_info])
        int_time_in_diff = np.array([ssfr0.dset_info[dset_tag][in_tag] - int_time[in_tag] for dset_tag in ssfr0.dset_info])

        idset_si = np.argmin(np.abs(int_time_si_diff))
        idset_in = np.argmin(np.abs(int_time_in_diff))

        int_time_new = copy.deepcopy(int_time)
        if int_time_si_diff[idset_si] != 0.0:
            fallback_si = int_time_si_diff[idset_si] + int_time[si_tag]
            warnings.warn(f"Cannot find given integration time for {si_tag}={int_time[si_tag]}ms, fallback to {si_tag}={fallback_si}ms")
            int_time_new[si_tag] = fallback_si

        if int_time_in_diff[idset_in] != 0.0:
            fallback_in = int_time_in_diff[idset_in] + int_time[in_tag]
            warnings.warn(f"Cannot find given integration time for {in_tag}={int_time[in_tag]}ms, fallback to {in_tag}={fallback_in}ms")
            int_time_new[in_tag] = fallback_in

        # Logical indexing for matching integration times
        si_time_match = np.isclose(ssfr0.data_raw['int_time'][:, index_si], int_time_new[si_tag], atol=1e-5)
        in_time_match = np.isclose(ssfr0.data_raw['int_time'][:, index_in], int_time_new[in_tag], atol=1e-5)

        # Dark correction and statistics
        shutter_si, counts_si = ssfr.corr.dark_corr(
            ssfr0.data_raw['tmhr'][si_time_match],
            ssfr0.data_raw['shutter'][si_time_match],
            ssfr0.data_raw['count_raw'][si_time_match, :, index_si],
            mode='interp', dark_extend=dark_extend, light_extend=light_extend
        )
        valid_si = (shutter_si == 0)
        spectra_si = np.nanmean(counts_si[valid_si, :], axis=0)
        spectra_si_std = np.nanstd(counts_si[valid_si, :], axis=0)

        shutter_in, counts_in = ssfr.corr.dark_corr(
            ssfr0.data_raw['tmhr'][in_time_match],
            ssfr0.data_raw['shutter'][in_time_match],
            ssfr0.data_raw['count_raw'][in_time_match, :, index_in],
            mode='interp', dark_extend=dark_extend, light_extend=light_extend
        )
        valid_in = (shutter_in == 0)
        spectra_in = np.nanmean(counts_in[valid_in, :], axis=0)
        spectra_in_std = np.nanstd(counts_in[valid_in, :], axis=0)

    except Exception as error:
        print(error)
        msg = '\nWarning [rad_cal_resp]: cannot process the data, set parameters to <None>.'
        warnings.warn(msg)
        spectra_si     = None
        spectra_si_std = None
        spectra_in     = None
        spectra_in_std = None

    # --- Silicon response ---
    # some placeholder ideas:
    # if nan is detected (e.g., spectra_si smaller than 0.0), one can use
    # interpolation to fill in the nan values
    if spectra_si is not None:
        rad_resp_si, rad_resp_si_std = compute_response(spectra=spectra_si,
                                                        spectra_std=spectra_si_std, 
                                                        int_time=int_time_new[si_tag], 
                                                        resp=resp[si_tag], 
                                                        resp_std=resp.get(si_tag+'_std'))
    else:
        rad_resp_si, rad_resp_si_std = None, None

    # --- InGaAs response ---
    if spectra_in is not None:
        rad_resp_in, rad_resp_in_std = compute_response(spectra=spectra_in,
                                                        spectra_std=spectra_in_std, 
                                                        int_time=int_time_new[in_tag], 
                                                        resp=resp[in_tag], 
                                                        resp_std=resp.get(in_tag+'_std'))
    else:
        rad_resp_in, rad_resp_in_std = None, None

    # response output
    rad_resp = {
               si_tag: rad_resp_si,
               in_tag: rad_resp_in,
               si_tag+'_std': rad_resp_si_std,
               in_tag+'_std': rad_resp_in_std,
               si_tag+'_count': spectra_si,
               in_tag+'_count': spectra_in,
               si_tag+'_count_std': spectra_si_std,
               in_tag+'_count_std': spectra_in_std,
               }

    return rad_resp


def cdata_rad_resp(
        fnames_pri=None,
        fnames_tra=None,
        fnames_sec=None,
        filename_tag=None,
        which_ssfr='lasp|ssfr-a',
        which_lc='zen',
        spec_reverse=False,
        which_lamp='f-1324',
        wvl_joint=950.0,
        wvl_range=[350.0, 2200.0],
        int_time={'si':80.0, 'in':250.0},
        lamp_corr=False,
        verbose=True,
        ):
    
    """
    Perform chained radiometric calibration and save results.
    Args:
        fnames_pri: list of primary calibration files
        fnames_tra: list of transfer calibration files
        fnames_sec: list of secondary calibration files
        filename_tag: output file tag
        which_ssfr: SSFR spectrometer string
        which_lc: light collector string
        spec_reverse: bool
        which_lamp: lamp string
        wvl_joint: joint wavelength value
        wvl_range: wavelength range [start, end]
        int_time: dict of integration times
        lamp_corr: bool
        verbose: bool
    Returns:
        str: output filename
    """

    # --- SSFR spectrometer selection ---
    config = InstrumentConfig(ssfr_id=which_ssfr, lc_id=which_lc, lamp_id=which_lamp,
                              spec_reverse=spec_reverse, int_time=int_time)

    if config.lab_id == 'nasa':
        import ssfr.nasa_ssfr as ssfr_toolbox
    elif config.lab_id == 'lasp':
        import ssfr.lasp_ssfr as ssfr_toolbox
    else:
        msg = f'\nError [cal_rad_resp]: <which_ssfr=> does not support <\'{config.ssfr_id}\'> (only supports <\'nasa|ssfr-6\'> or <\'lasp|ssfr-a\'> or <\'lasp|ssfr-b\'>).' 
        raise ValueError(msg)

    si_tag = config.si_tag
    in_tag = config.in_tag
    int_time = config.int_time

    
    # Calibration calls
    if fnames_pri is None:
        raise OSError("\nError [cdata_rad_resp]: cannot proceed without primary calibration files.")
    pri_resp = cal_rad_resp(
            fnames_pri,
            resp=None,
            instrument_config=config,
            int_time=int_time,
            lamp_corr=lamp_corr,
            verbose=verbose,
            )

    if fnames_tra is None:
        raise OSError("\nError [cdata_rad_resp]: cannot proceed without transfer calibration files.")
    transfer = cal_rad_resp(
            fnames_tra,
            resp=pri_resp,
            instrument_config=config,
            int_time=int_time,
            verbose=verbose,
            )

    if fnames_sec is not None:
        sec_resp = cal_rad_resp(
                fnames_sec,
                resp=transfer,
                instrument_config=config,
                int_time=int_time,
                verbose=verbose,
                )
    else:
        warnings.warn("'\nWarning [cdata_rad_resp]: secondary/field calibration files are not available, use transfer calibration files for secondary/field calibration ...")
        sec_resp = cal_rad_resp(
                fnames_tra,
                resp=transfer,
                instrument_config=config,
                int_time=int_time,
                verbose=verbose,
                )

    lc_data = {'pri_resp': pri_resp, 'transfer': transfer, 'sec_resp': sec_resp}
    
    # Wavelength slicing and sorting
    wvls = config.wvls

    # File saving
    if filename_tag is None:
        filename_tag = 'test'
    if not lamp_corr:
        fname_out = '%s|rad-resp|%s|%s|si-%3.3d|in-%3.3d.h5' % (filename_tag, config.ssfr_id, config.lc_id, int_time[si_tag], int_time[in_tag])
    else:
        fname_out = '%s|rad-resp|%s|%s|si-%3.3d|in-%3.3d|lamp-adjust.h5' % (filename_tag, config.ssfr_id, config.lc_id, int_time[si_tag], int_time[in_tag])
    
    # save resps to h5 files
    pri_resp_out = fname_out.replace('.h5', '|pri_resp.pkl')
    transfer_out = fname_out.replace('.h5', '|transfer.pkl')
    sec_resp_out = fname_out.replace('.h5', '|sec_resp.pkl')
        
    # # Save calibration results to pickle files
    for out_name, data in zip([pri_resp_out, transfer_out, sec_resp_out],
                              [pri_resp, transfer, sec_resp]):
        with open(out_name, 'wb') as f:
            pkl.dump(data, f)
            

    fname_resp_out = fname_out.replace('.h5', '|resp_intermediate.h5')

    with h5py.File(fname_resp_out, 'w') as f:
        # Create groups for each response type
        for group_name, data_dict in [('pri_resp', pri_resp),
                                      ('transfer', transfer),
                                      ('sec_resp', sec_resp)]:
            group = f.create_group(group_name)
            # Save each numpy array from the dictionary into the group
            for key, value in data_dict.items():
                if value is not None:
                    group.create_dataset(key, data=value)
    
    print(f"Saved intermediate calibration data to: {fname_resp_out}")
    
    # Save results to HDF5 file           
    _, _ = _save_combined_h5(
        fname_out=fname_out,
        wvls=wvls, data=lc_data, tags=(si_tag, in_tag),
        wvl_range=wvl_range, wvl_joint=wvl_joint
    )

    return fname_out

ResponseData = Dict[str, np.ndarray]

def _load_response_group(base_fname: str, delete_files: bool=True) -> Tuple[ResponseData, ResponseData, ResponseData]:
    """
    Loads primary, transfer, and secondary response data from a 
    structured intermediate HDF5 file.
    """
    pri_resp, transfer, sec_resp = {}, {}, {}

    fname_resp = base_fname.replace('.h5', '|resp_intermediate.h5')
    with h5py.File(fname_resp, 'r') as f:
        for group_name, data_dict in [('pri_resp', pri_resp),
                                      ('transfer', transfer),
                                      ('sec_resp', sec_resp)]:
            if group_name in f:
                for key in f[group_name]:
                    data_dict[key] = f[group_name][key][:] # [:] loads data into memory
            else:
                raise KeyError(f"Group '{group_name}' not found in {fname_resp}")
            
    if delete_files and os.path.exists(fname_resp):
        os.remove(fname_resp)
                
    return pri_resp, transfer, sec_resp


def load_responses_and_validate(fnames_resp_zen: str, fnames_resp_nad: str, 
                                delete_files: bool=True) -> Tuple:
    """Load and validate all response files."""
    if not fnames_resp_zen or not fnames_resp_nad:
        raise OSError("Cannot proceed without both zenith and nadir response files.")

    pri_resp_zen, transfer_zen, sec_resp_zen = _load_response_group(fnames_resp_zen, delete_files)
    pri_resp_nad, transfer_nad, sec_resp_nad = _load_response_group(fnames_resp_nad, delete_files)

    with h5py.File(fnames_resp_zen, 'r') as f:
        transfer_zen_ori = f['transfer'][:]
    with h5py.File(fnames_resp_nad, 'r') as f:
        transfer_nad_ori = f['transfer'][:]

    return (pri_resp_zen, transfer_zen, sec_resp_zen,
            pri_resp_nad, transfer_nad, sec_resp_nad,
            transfer_zen_ori, transfer_nad_ori)

def plot_comparison(
    datasets: List[Dict[str, Any]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_fname: str,
    joint_region=None, # Tuple[float, float] 
    ):
    """A generic plotting function to replace the three originals."""
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 5))

    for ds in datasets:
        ax.plot(ds['x'], ds['y'], ds.get('style', '-'), color=ds.get('color'), label=ds['label'])

    ymin, ymax = ax.get_ylim()
    if joint_region:
        ax.fill_betweenx([ymin, ymax], joint_region[0], joint_region[1],
                         color='gray', alpha=0.5, label='joint region')

    ax.set_ylim(0, ymax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    # ax.grid(True, linestyle='--')
    fig.tight_layout()
    fig.savefig(output_fname, dpi=300)
    print(f"Saved plot: {output_fname}")


def prepare_wavelengths(which_lab: str, which_ssfr: str) -> Tuple:
    """Prepare wavelength arrays for two spectrometers."""
    if which_lab == 'nasa':
        import ssfr.nasa_ssfr as ssfr_toolbox
    elif which_lab == 'lasp':
        import ssfr.lasp_ssfr as ssfr_toolbox
    else:
        raise ValueError(f"Unsupported lab: '{which_lab}'. Use 'nasa' or 'lasp'.")

    wvls = ssfr_toolbox.get_ssfr_wvl(which_ssfr.lower())
    return (wvls, wvls['zen|si'], wvls['zen|in'], wvls['nad|si'], wvls['nad|in'])


def update_secondary_response(
    count: np.ndarray, count_std: np.ndarray, int_time: float,
    transfer: np.ndarray, transfer_std: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update secondary response and propagate uncertainty.
    Args:
        count: np.ndarray, measured counts
        count_std: np.ndarray, std of measured counts
        int_time: float, integration time
        transfer: np.ndarray, transfer response
        transfer_std: np.ndarray, std of transfer response
    Returns:
        sec_resp: np.ndarray, updated secondary response
        sec_resp_std: np.ndarray, propagated uncertainty
    """
    sec_resp = count / (int_time * transfer)
    
    # Simplified and more robust uncertainty propagation using relative errors
    # (dy/y)^2 = (da/a)^2 + (db/b)^2 + ...
    # This avoids division by zero if count is zero
    with np.errstate(divide='ignore', invalid='ignore'):
        count_rel_err_sq = (count_std / count)**2
        transfer_rel_err_sq = (transfer_std / transfer)**2

    sec_resp_rel_err_sq = count_rel_err_sq + transfer_rel_err_sq
    sec_resp_std = sec_resp * np.sqrt(sec_resp_rel_err_sq)
    
    # Handle potential NaNs from division by zero
    sec_resp_std = np.nan_to_num(sec_resp_std)

    return sec_resp, sec_resp_std

def _apply_scaling_correction(
    data: Dict,
    wvl_base: np.ndarray, transfer_base: np.ndarray,
    wvl_target: np.ndarray, transfer_target: np.ndarray,
    interp_range: Tuple[float, float],
    int_time: float, data_key: str,
    scaling_method: str = 'mean',
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to perform interpolation, scaling, and uncertainty calculation."""
    wvl_interp = np.arange(interp_range[0], interp_range[1] + 0.1, 0.1)

    f_transfer_base = interpolate.interp1d(wvl_base, transfer_base, bounds_error=False, fill_value=np.nan)
    f_transfer_target = interpolate.interp1d(wvl_target, transfer_target, bounds_error=False, fill_value=np.nan)

    if scaling_method == 'mean':
        transfer_base_interp = np.nanmean(f_transfer_base(wvl_interp))
        transfer_target_interp = np.nanmean(f_transfer_target(wvl_interp))
        scaling_factor = transfer_base_interp / transfer_target_interp
    elif scaling_method == 'point':
        transfer_base_interp = f_transfer_base(wvl_interp)
        transfer_target_interp = f_transfer_target(wvl_interp)
        scaling_factor = transfer_base_interp / transfer_target_interp
        f_scaling = interpolate.interp1d(wvl_interp, scaling_factor, bounds_error=False, fill_value="extrapolate")
        scaling_factor = f_scaling(wvl_target)
    else:
        raise ValueError(f"Unsupported scaling method: '{scaling_method}'")


    # Apply scaling factor
    data['pri_resp'][data_key] /= scaling_factor

    # Recalculate transfer function and its uncertainty
    new_transfer = (data['transfer'][f"{data_key}_count"] /
                    int_time /
                    data['pri_resp'][data_key])

    count_std_frac = data['transfer'][f"{data_key}_count_std"] / data['transfer'][f"{data_key}_count"]
    pri_resp_std_frac = data['pri_resp'][f"{data_key}_std"] / data['pri_resp'][data_key]
    
    # Propagate uncertainty
    transfer_std_sq_term1 = (count_std_frac / (int_time * data['pri_resp'][data_key]))**2
    transfer_std_sq_term2 = (new_transfer * pri_resp_std_frac)**2
    new_transfer_std = np.sqrt(transfer_std_sq_term1 + transfer_std_sq_term2)

    return new_transfer, new_transfer_std

def _save_combined_h5(
    fname_out: str, wvls: Dict, data: Dict, tags: Tuple[str, str],
    wvl_range: List[float], wvl_joint: float
):
    """Helper to combine and save data to an HDF5 file."""
    si_tag, in_tag = tags
    
    logic_si = (wvls[si_tag] >= wvl_range[0]) & (wvls[si_tag] <= wvl_joint)
    logic_in = (wvls[in_tag] > wvl_joint) & (wvls[in_tag] <= wvl_range[1])

    combined = {}
    
    # Define base names and suffixes to build keys programmatically
    base_names = ['pri_resp', 'transfer', 'sec_resp']
    suffixes = ['', '_std', '_count', '_count_std']
    
    # wavelengths
    combined['wvl'] = np.concatenate((wvls[si_tag][logic_si], wvls[in_tag][logic_in]))
    
    # Loop to build all other arrays
    for base in base_names:
        for suffix in suffixes:
            key = f"{base}{suffix}"
            si_full_key = f"{si_tag}{suffix}"
            in_full_key = f"{in_tag}{suffix}"
            
            # data[base] would be pri_resp_dict, transfer_dict, etc.
            si_data = data[base][si_full_key][logic_si]
            in_data = data[base][in_full_key][logic_in]
            combined[key] = np.concatenate((si_data, in_data))
    
    # Sort all arrays by wavelength
    sort_indices = np.argsort(combined['wvl'])
    for key in combined:
        combined[key] = combined[key][sort_indices]

    # Save to HDF5
    with h5py.File(fname_out, 'w') as f:
        for key, value in combined.items():
            f.create_dataset(key, data=value)
        
        # Also save raw corrected data in groups
        raw_group = f.create_group('raw')
        for tag, group_name in [(si_tag, 'si'), (in_tag, 'in')]:
            group = raw_group.create_group(group_name)
            group['wvl'] = wvls[tag]
            for key, ds in data.items():
                 for suffix in ['', '_std', '_count', '_count_std']:
                    full_key = f"{tag}{suffix}"
                    if full_key in ds:
                        group.create_dataset(f"{key}{suffix}", data=ds[full_key])

    print(f"Saved corrected data to {fname_out}")
    
    return combined['wvl'], combined['transfer']


def rad_resp_corr(fnames_resp_zen: str,
                  fnames_resp_nad: str,
                  which_ssfr: str = 'lasp|ssfr-a',
                  int_time: Dict[str, float] = {'si': 80.0, 'in': 250.0},
                  wvl_joint: float = 950.0,
                  wvl_joint_range: float = 20.0,
                  wvl_range: List[float] = [350.0, 2200.0],
                  ):
    """
    Main workflow for applying radiometric response corrections.
    """
    # 1. Configure the instruments
    zen_config = InstrumentConfig(ssfr_id=which_ssfr, lc_id='zen', spec_reverse=False, int_time=int_time)
    nad_config = InstrumentConfig(ssfr_id=which_ssfr, lc_id='nad', spec_reverse=False, int_time=int_time)
    
    
    # 2. Load data
    (pri_resp_zen, transfer_zen, sec_resp_zen,
     pri_resp_nad, transfer_nad, sec_resp_nad,
     transfer_zen_ori, transfer_nad_ori) = load_responses_and_validate(fnames_resp_zen, fnames_resp_nad, delete_files=True)
        
    # Group data into a more manageable structure
    zen_data = {'pri_resp': pri_resp_zen, 'transfer': transfer_zen, 'sec_resp': sec_resp_zen}
    nad_data = {'pri_resp': pri_resp_nad, 'transfer': transfer_nad, 'sec_resp': sec_resp_nad}
    
    # Store original transfers for plotting
    transfer_in_nad_orig = nad_data['transfer']['nad|in'].copy()
    transfer_si_zen_orig = zen_data['transfer']['zen|si'].copy()
    transfer_in_zen_orig = zen_data['transfer']['zen|in'].copy()
    
    
    # 3. Prepare wavelengths
    wvls_zen, wvls_nad = zen_config.wvls, nad_config.wvls
    wvl_si_zen, wvl_in_zen, wvl_si_nad, wvl_in_nad = zen_config.wvl_si, zen_config.wvl_in, nad_config.wvl_si, nad_config.wvl_in
    
    
    # 4. Perform correction 
    wvl_start_joint = wvl_joint - wvl_joint_range / 2.0
    wvl_end_joint = wvl_joint + wvl_joint_range / 2.0
    
    # (4-1) Correct NAD-IN based on NAD-SI
    new_transfer_in_nad, new_transfer_in_std_nad = _apply_scaling_correction(
        data=nad_data,
        wvl_base=wvl_si_nad, transfer_base=nad_data['transfer']['nad|si'],
        wvl_target=wvl_in_nad, transfer_target=transfer_in_nad_orig,
        interp_range=(wvl_start_joint, wvl_end_joint),
        int_time=int_time['nad|in'], data_key='nad|in'
    )
    nad_data['transfer']['nad|in'] = new_transfer_in_nad
    nad_data['transfer']['nad|in_std'] = new_transfer_in_std_nad

    plot_comparison(
        datasets=[
            {'x': wvl_si_nad, 'y': nad_data['transfer']['nad|si'], 'label': 'NAD-SI (original)', 'color': 'blue'},
            {'x': wvl_in_nad, 'y': transfer_in_nad_orig, 'label': 'NAD-InGaAs (original)', 'color': 'red'},
            {'x': wvl_in_nad, 'y': nad_data['transfer']['nad|in'], 'label': 'NAD-InGaAs (scaled)', 'color': 'orange', 'style': '--'}
        ],
        title='Nadir Si-InGaAs Correction', xlabel='Wavelength (nm)', ylabel='Transfer flux ($W m^{-2} nm^{-1}$)',
        output_fname='rad_resp_corr_nad_si_in.png',
        joint_region=(wvl_start_joint, wvl_end_joint)
    )
    
    # (4-2) nad-si and zen-si transfer check
    wvl_si_start = np.min((wvl_si_nad.min(), wvl_si_zen.min()))
    wvl_si_end   = np.max((wvl_si_nad.max(), wvl_si_zen.max()))
    new_transfer_si_zen, new_transfer_si_std_zen = _apply_scaling_correction(
        data=zen_data,
        wvl_base=wvl_si_nad, transfer_base=nad_data['transfer']['nad|si'],
        wvl_target=wvl_si_zen, transfer_target=transfer_si_zen_orig,
        interp_range=(wvl_si_start, wvl_si_end), # Use full range for this correction
        int_time=int_time['zen|si'], data_key='zen|si',
        scaling_method='point'  # Use point-by-point scaling for this correction
    )
    zen_data['transfer']['zen|si'] = new_transfer_si_zen
    zen_data['transfer']['zen|si_std'] = new_transfer_si_std_zen
    
    plot_comparison(
        datasets=[
            {'x': wvl_si_nad, 'y': nad_data['transfer']['nad|si'], 'label': 'NAD-SI (original)', 'color': 'blue'},
            {'x': wvl_si_zen, 'y': transfer_si_zen_orig, 'label': 'ZEN-SI (original)', 'color': 'red'},
            {'x': wvl_si_zen, 'y': zen_data['transfer']['zen|si'], 'label': 'ZEN-SI (scaled)', 'color': 'orange', 'style': '--'}
        ],
        title='Zenith Si Correction', xlabel='Wavelength (nm)', ylabel='Transfer flux ($W m^{-2} nm^{-1}$)',
        output_fname='rad_resp_corr_zen_si_nad_si_2.png'
    )
    
    # (4-3) zen-si and zen-in transfer check
    new_transfer_in_zen, new_transfer_in_std_zen = _apply_scaling_correction(
        data=zen_data,
        wvl_base=wvl_si_zen, transfer_base=zen_data['transfer']['zen|si'],
        wvl_target=wvl_in_zen, transfer_target=transfer_in_zen_orig,
        interp_range=(wvl_start_joint, wvl_end_joint),
        int_time=int_time['zen|in'], data_key='zen|in'
    )
    zen_data['transfer']['zen|in'] = new_transfer_in_zen
    zen_data['transfer']['zen|in_std'] = new_transfer_in_std_zen
    
    plot_comparison(
        datasets=[
            {'x': wvl_si_zen, 'y': zen_data['transfer']['zen|si'], 'label': 'ZEN-Si (scaled)', 'color': 'blue', 'style': '--'},
            {'x': wvl_in_zen, 'y': transfer_in_zen_orig, 'label': 'ZEN-InGaAs (original)', 'color': 'red'},
            {'x': wvl_in_zen, 'y': zen_data['transfer']['zen|in'], 'label': 'ZEN-InGaAs (scaled)', 'color': 'orange', 'style': '--'},
            {'x': wvl_si_zen, 'y': transfer_si_zen_orig, 'label': 'ZEN-Si (original)', 'color': 'cyan'}
        ],
        title='Zenith Si-InGaAs Correction', xlabel='Wavelength (nm)', ylabel='Transfer flux ($W m^{-2} nm^{-1}$)',
        output_fname='rad_resp_corr_zen_si_zen_in_2.png',
        joint_region=(wvl_start_joint, wvl_end_joint)
    )
    
    
    # 5. Update secondary responses
    # assume nad-si is correct, update other three responses 
    sec_resp_in_nad_ori = nad_data['sec_resp']['nad|in'].copy()
    sec_resp_si_zen_ori = zen_data['sec_resp']['zen|si'].copy()
    sec_resp_in_zen_ori = zen_data['sec_resp']['zen|in'].copy()
    
    for data, prefix, itime_key in [(nad_data, 'nad', 'in'), (zen_data, 'zen', 'si'), (zen_data, 'zen', 'in')]:
        channel = f"{prefix}|{itime_key}"
        new_sec_resp, new_sec_resp_std = update_secondary_response(
            count=data['sec_resp'][f"{channel}_count"],
            count_std=data['sec_resp'][f"{channel}_count_std"],
            int_time=int_time[channel],
            transfer=data['transfer'][channel],
            transfer_std=data['transfer'][f"{channel}_std"]
        )
        data['sec_resp'][channel] = new_sec_resp
        data['sec_resp'][f"{channel}_std"] = new_sec_resp_std

    plot_sec_resp_before_after_corr(zen_data, nad_data,
                                    wvl_si_zen, wvl_in_zen,
                                    wvl_si_nad, wvl_in_nad,
                                    sec_resp_si_zen_ori, 
                                    sec_resp_in_zen_ori,
                                    sec_resp_in_nad_ori) 
    
    # 6. Save final corrected data using the save helper
    wvl_zen_, transfer_zen_ = _save_combined_h5(
        fname_out=fnames_resp_zen.replace('.h5', '|corr.h5'),
        wvls=wvls_zen, data=zen_data, tags=('zen|si', 'zen|in'),
        wvl_range=wvl_range, wvl_joint=wvl_joint
    )
    
    wvl_nad_, transfer_nad_ = _save_combined_h5(
        fname_out=fnames_resp_nad.replace('.h5', '|corr.h5'),
        wvls=wvls_nad, data=nad_data, tags=('nad|si', 'nad|in'),
        wvl_range=wvl_range, wvl_joint=wvl_joint
    )

    plot_transfer_before_after_corr(wvl_nad_, transfer_nad_ori,
                                    wvl_zen_, transfer_zen_ori,
                                    transfer_nad_, transfer_zen_)


def plot_sec_resp_before_after_corr(zen_data, nad_data,
                                    wvl_si_zen, wvl_in_zen,
                                    wvl_si_nad, wvl_in_nad,
                                    sec_resp_si_zen_ori, 
                                    sec_resp_in_zen_ori,
                                    sec_resp_in_nad_ori):
    """Plot secondary responses before and after correction."""
    plt.close('all')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))
    ax1.plot(wvl_si_nad, nad_data['sec_resp']['nad|si'], 'b-', label='NAD-si (original)')
    ax1.set_title('Nad-si')
    
    ax2.plot(wvl_in_nad, sec_resp_in_nad_ori, 'b-', label='NAD-in (original)')
    ax2.plot(wvl_in_nad, nad_data['sec_resp']['nad|in'], 'r--', label='NAD-in (updated)')
    ax2.set_title('Nad-InGaAs')
    
    ax3.plot(wvl_si_zen, sec_resp_si_zen_ori, 'b-', label='ZEN-si (original)')
    ax3.plot(wvl_si_zen, zen_data['sec_resp']['zen|si'], 'r--', label='ZEN-si (updated)')
    ax3.set_title('Zen-Si')
    
    ax4.plot(wvl_in_zen, sec_resp_in_zen_ori, 'b-', label='ZEN-in (original)')
    ax4.plot(wvl_in_zen, zen_data['sec_resp']['zen|in'], 'r--', label='ZEN-in (updated)')
    ax4.set_title('Zen-InGaAs')

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Secondary response (count/energy)')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)    
    fig.tight_layout()
    fig.savefig('rad_resp_corr_sec_resp_update.png', dpi=300)
    

def plot_transfer_before_after_corr(wvl_nad_, transfer_nad_ori,
                                    wvl_zen_, transfer_zen_ori,
                                    transfer_nad_, transfer_zen_):
    """Plot transfer functions before and after correction."""
    f_transfer_nad_ori = interpolate.interp1d(wvl_nad_, transfer_nad_ori, bounds_error=False, fill_value=np.nan)
    transfer_nad_ori_interp = f_transfer_nad_ori(wvl_zen_)
    f_transfer_nad_corr = interpolate.interp1d(wvl_nad_, transfer_nad_, bounds_error=False, fill_value=np.nan)
    transfer_nad_corr_interp = f_transfer_nad_corr(wvl_zen_)
    nad_zen_ratio_ori = transfer_nad_ori_interp / transfer_zen_ori
    nad_zen_ratio_corr = transfer_nad_corr_interp / transfer_zen_
    
    plt.close('all')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    ax1.plot(wvl_zen_, transfer_zen_ori, 'r-', label='original')
    ax1.plot(wvl_zen_, transfer_zen_, 'b--', label='scaled')
    ax1.set_title('ZEN transfer')
    
    ax2.plot(wvl_zen_, transfer_nad_ori_interp, 'r-', label='original')
    ax2.plot(wvl_zen_, transfer_nad_corr_interp, 'b--', label='scaled')
    ax2.set_title('NAD transfer (interpolated to ZEN wvl)')

    ax3.plot(wvl_zen_, nad_zen_ratio_ori, '-', color='cyan', label='original')
    ax3.plot(wvl_zen_, nad_zen_ratio_corr, '--', color='blue', label='scaled')
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Transfer flux (W m$^{-2}$ nm$^{-1}$)')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Transfer ratio (Nad/Zen)')
    ax4.axis('off') # set ax4 invisible
    fig.tight_layout()
    fig.savefig('rad_resp_corr_transfer_check.png', dpi=300)
    

if __name__ == '__main__':

    pass
