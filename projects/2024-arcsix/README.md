# 2024-arcsix

This directory contains the source code and documentation for the 2024 ARCSIX data.

## Contents
- `_cal-ssfr.py`
Main calibration code that reads raw LabView output (SKS files) and produce radiometric/cosine response `.h5` files.
- `_arcsix-ssfr.py`
Main processing code for SSFR that applies the calibration (response) to the raw field data (SKS files).
- `_arcsix-ssrr.py`
Same but for SSRR.
- `_arcsix-hsr1.py`
Same but for HSR1.
- `_arcsix-alp.py`
Same but for ALP and HSK (housekeeping; aircraft location and attitude etc.) data.

`_cal.py` and `_arcsix.py` are legacy codes used previously.

## To test the SSRR calibration and data processing

1. Calibration
- Modify and run `_cal-ssfr.py`
Uncomment the respective functions at the bottom to choose the calibration to perform (radiometric vs cosine, SSFR-A or B, or nadir/zenith)

2. Data processing (apply calibration to the actual data)
- Modify `cfg_YYYYMMDD.py`
Make sure that the file paths point to the correct field measurement and calibration data directories.
- Modify and run `_arcsix-*.py`
The `cfg_YYYYMMDD.py` file is read in at the beginning of the process.