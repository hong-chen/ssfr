
==============
How to Install
==============
::

    git clone https://gitlab.com/cu-arg/ssfr.git
    python setup.py develop


|
|

Calibrations
~~~~~~~~~~~~

====================================
1. Primary and Transfer Calibrations
====================================

Standard lamp in the lab: F-1324 (or 506C)

- The standard lamps are NIST traceble lamps, which relate SSFR measured digital counts to
  radiative fluxes.

The primary calibration files are preferred to be stored under file directries using the following naming convention:

::

    /some/path/1324/zenith_LC1/s60i300/cal
    /some/path/1324/zenith_LC1/s60i300/dark
    /some/path/1324/nadir_LC2/s60i300/cal
    /some/path/1324/nadir_LC2/s60i300/dark

To process the primary calibration, use the following code block as a reference,


Lamp in the field calibrator: 150C (or 150D or 150E)

- These lamps are not calibrated.

The transfer calibration files are preferred to be stored under file directries using the following naming convention:

::

    /some/path/150C/zenith_LC1/s60i300/cal
    /some/path/150C/zenith_LC1/s60i300/dark
    /some/path/150C/nadir_LC2/s60i300/cal
    /some/path/150C/nadir_LC2/s60i300/dark




=======================
2. Angular Calibrations
=======================

In the lab, we use an uncalibrated lamp (507 or 508) to perform the angluar calibration. To perform this
calibration, we mount the SSFR light collector to a rotating stage, which can be controlled through
a computer via command lines.

For the reference, the angles [units: degree] we picked are:
::

    0.0,  5.0,  10.0,  15.0,  20.0,  25.0,  30.0,  40.0,  50.0,  60.0,  70.0,  80.0,  90.0,
    0.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0, -90.0, 0.0

We also have another set of angles with a higher resolution:
::

    0.0,  3.0,  6.0,  9.0,  12.0,  15.0,  18.0,  21.0,  24.0,  27.0,  30.0,  35.0,  40.0,  45.0,  50.0,  60.0,  70.0,  80.0,  90.0,
    0.0, -3.0, -6.0, -9.0, -12.0, -15.0, -18.0, -21.0, -24.0, -27.0, -30.0, -35.0, -40.0, -45.0, -50.0, -60.0, -70.0, -80.0, -90.0, 0.0



================
3. Distance Test
================

In the lab, we use an uncalibrated lamp (507) to perform the distance test. The lamp is mounted on
a rack that can be moved back and forth with a specified distance according to the readings of the
ruler attached to the rack.

For the reference, the distances [units: cm] we picked are:
::

    50.0 ,45.0 ,50.0 ,40.0 ,50.0 ,35.0 ,50.0 ,55.0 ,50.0 ,60.0 ,50.0 ,65.0, 50.0

======================
4*. Field Calibrations
======================

The field calibrator travels with SSFR during the research campaign. The field calibration will be performed regularly to
trace the stability of the SSFR measurements. The procedures are similar to the procedures of transfer calibration.

The calibration files are preferred to be stored under file directries using the following naming convention:

::

    /some/path/150C/zenith_LC1/s60i300/cal
    /some/path/150C/zenith_LC1/s60i300/dark
    /some/path/150C/nadir_LC2/s60i300/cal
    /some/path/150C/nadir_LC2/s60i300/dark



|
|

Corrections
~~~~~~~~~~~

==================
1. Dark Correction
==================

The effective SSFR counts are the dark counts (when shutter if on) deducted from the light counts (when shutter is off).


====================
2. Cosine Correction
====================

From the angular calibration, we obtained spectral angular responses. These angular responses can be used to correct
the direct light measurements (zenith when clear sky). However, when the light is diffused (nadir, zenith when cirrus above),
we will need to use integrated angular response to perform correction. The diffuse-to-global ratio is needed before cosine
correction. In ORACLES 2018 and CAMP2Ex, the direct measured diffuse-to-global ratio from SPN-S and SPN-F can be used
for the cosine correction of SSFR. RTM calculations can also be used to provide diffuse-to-global ratio if measurements are
not avaiable (e.g., ARISE, ORACLES 2016, ORACLES 2017).



===========================
3. Non-linearity Correction
===========================





======================
4*. Azimuth Correction
======================

This correction has only been applied to the ARISE dataset. The azimuth correction used the data collected during
a circling maneuver, where instrument can provide measurements at azimuth angle from 0 to 360.


|
|

Additional Notes
~~~~~~~~~~~~~~~~

===================================
IDL to Python Translation (Example)
===================================
::

     python         :          IDL
       l            :    long or lonarr
       B            :    byte or bytarr
       L            :    ulong
       h            :    intarr

     E.g., in IDL:

         spec  = {btime:lonarr(2)   , bcdtimstp:bytarr(12),  $ 2l12B
                  intime1:long(0)   , intime2:long(0)     ,  $ 6l
                  intime3:long(0)   , intime4:long(0)     ,  $
                  accum:long(0)     , shsw:long(0)        ,  $
                  zsit:ulong(0)     , nsit:ulong(0)       ,  $ 8L
                  zirt:ulong(0)     , nirt:ulong(0)       ,  $
                  zirx:ulong(0)     , nirx:ulong(0)       ,  $
                  xt:ulong(0)       , it:ulong(0)         ,  $
                  zspecsi:intarr(np), zspecir:intarr(np)  ,  $ 1024h
                  nspecsi:intarr(np), nspecir:intarr(np)}

     in Python:

         '<2l12B6l8L1024h'

