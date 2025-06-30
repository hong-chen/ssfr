# NASA ARCSIX 2024

This directory contains the source code and documentation for the 2024 ARCSIX data. (To be edited...)

## To test the SSRR calibration and data processing

### Calibration

1. Modify and run `_cal.py`
   - To run radiance calibration, ensure that within the `main_ssrr_rad_cal_all` function that the filepaths for `fdirs_pri` are correct for your setup.
   - Note that in the future, this might be changed.
2. Execute `main_ssrr_rad_cal_all` function, which reads in the SSRR raw (SKS) files and obtain the instrument response (sensitivity).

3. Data processing (apply calibration to the actual data)

   - Modify `cfg_YYYYMMDD.py`
Make sure that the file paths point to the correct field measurement and calibration data directories.
   - Modify and run `_arcsix.py`
   - Execute the `main_process_data_v0` function first to read in the raw data (note that data for other instruments may also be loaded).
       - Then execute the `main_process_data_v1` function which applies the calibration (conversion from raw counts to physical variables). The `cfg_YYYYMMDD.py` file is read in at the beginning of the process.
