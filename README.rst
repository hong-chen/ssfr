SSFR-util (Under Development...)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

==============
How to Install
==============
::

    pip install SSFR-util

==========
How to Use
==========


Calibrations
~~~~~~~~~~~~

=====================================
1. Primary and Transfer Calibrations
=====================================

Standard lamp: F-1324 or 506C
Field calibrator:

=======================
2. Angular Calibrations
=======================


================
3. Distance Test
================


======================
4*. Field Calibrations
======================





Corrections
~~~~~~~~~~~

=================
Cosine Correction
=================





========================
Non-linearity Correction
========================






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
       spec  = {btime:lonarr(2)   , bcdtimstp:bytarr(12),$     2l12B
                intime1:long(0)   , intime2:long(0)     ,$     6l
                intime3:long(0)   , intime4:long(0)     ,$
                accum:long(0)     , shsw:long(0)        ,$
                zsit:ulong(0)     , nsit:ulong(0)       ,$     8L
                zirt:ulong(0)     , nirt:ulong(0)       ,$
                zirx:ulong(0)     , nirx:ulong(0)       ,$
                xt:ulong(0)       , it:ulong(0)         ,$
                zspecsi:intarr(np), zspecir:intarr(np)  ,$     1024h
                nspecsi:intarr(np), nspecir:intarr(np)}
     in Python:

     '<2l12B6l8L1024h'

