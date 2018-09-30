SSFR-util (Under Development...)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

==============
How to Install
==============
::

    pip3 install SSFR-util

==========
How to Use
==========

|
|

Calibrations
~~~~~~~~~~~~

=====================================
1. Primary and Transfer Calibrations
=====================================

Standard lamp in the lab: F-1324 (or 506C)

- The standard lamps are NIST traceble lamps, which relate SSFR measured digital counts to
  radiative fluxes.

Lamp in the field calibrator: 150C (or 150D or 150E)

- These lamps are not calibrated.



=======================
2. Angular Calibrations
=======================

In the lab, we use an uncalibrated lamp (507) to perform the angluar calibration. To perform this
calibration, we mount the SSFR light collector to a rotating stage, which can be controlled through
a computer via command lines.

For the reference, the angles [units: degree] we picked are:
::

    0.0,  5.0,  10.0,  15.0,  20.0,  25.0,  30.0,  40.0,  50.0,  60.0,  70.0,  80.0,  90.0,
    0.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0, -90.0, 0.0



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

The field calibrator is sent to the field with SSFR. The field calibration will be performed regularly to
trace the stability of the SSFR measurements.



|
|

Corrections
~~~~~~~~~~~

==================
1. Dark Correction
==================


====================
2. Cosine Correction
====================

Since the angular responses are different for direct and diffuse light, the wavelength dependent
diffuse-to-direct ratio needs to be calculated first.








===========================
3. Non-linearity Correction
===========================





======================
4*. Azimuth Correction
======================
This correction has only been applied to the dataset collected during Arctic Radiation - IceBridge
Sea&Ice Experiment (ARISE).




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

