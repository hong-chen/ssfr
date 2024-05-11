SSFR Software Package
~~~~~~~~~~~~~~~~~~~~~
SSFR (Solar Spectral Flux Radiometer) is an airborne instrument co-developed by teams
at NASA Ames Research Center (discontinued) and LASP of University of Colorado Boulder.
SSFR provides simultaneous downwelling and upwelling spectral irradiance measurements ranging
from 350 nm to 2200 nm, which can be used to study cloud/aerosol radiative effects and
retrieve cloud/aerosol optical properties.

This repository provides legacy code for processing SSFR data by applying necessary calibrations
and corrections. Due to different deployment conditions of the instrument, the data processing procedures can vary
significantly for different missions. Please contact `SSFR science team <https://lasp.colorado.edu/airs/group>`_
for more information if you have any questions regarding SSFR data processing for a specific mission.

So far, SSFR has participated in the following airborne missions:

* `NASA ARCSIX <https://espo.nasa.gov/ARCSIX/content/ARCSIX>`_ (will be on P-3 in 2024 May and July);

* `NASA CAMP²Ex <https://espo.nasa.gov/camp2ex/content/CAMP2Ex>`_ (on P-3 in 2019);

* `NASA ORACLES <https://espo.nasa.gov/ORACLES/content/ORACLES>`_ (on ER-2 and P-3 in 2016; on P-3 in 2017 and 2018) ;

* `NASA ARISE <https://espo.nasa.gov/arise/content/ARISE>`_ (on C-130 in 2014);

* `NASA SEAC⁴RS <https://espo.nasa.gov/seac4rs>`_ (on DC-8 in 2013);

* `NASA SAFARI 2000 <https://espo.nasa.gov/content/SAFARI_2000>`_ (on ER-2 and Convair-580 in 2000).

A more detailed SSFR manual can be found `here <https://docs.google.com/document/d/1ObczXucJQktyTgKZlBkL04fjhHFx1ydW0sPaiG7iZ9k/edit?usp=sharing>`_ (under development).

==============
How to Install
==============
::

    git clone https://github.com/hong-chen/ssfr.git

    cd ssfr

    conda env create -f ssfr-env.yml
    conda activate ssfr

    python setup.py develop

==========
References
==========

* `Chen et al., 2021 <https://doi.org/10.5194/amt-14-2673-2021>`_

  Chen, H., Schmidt, S., King, M. D., Wind, G., Bucholtz, A., Reid, E. A., Segal-Rozenhaimer, M.,
  Smith, W. L., Taylor, P. C., Kato, S., and Pilewskie, P.: The effect of low-level thin arctic
  clouds on shortwave irradiance: evaluation of estimates from spaceborne passive imagery with
  aircraft observations, Atmos. Meas. Tech., 14, 2673–2697, doi:10.5194/amt-14-2673-2021, 2021.

  * Instrument Highlight: azimuthal response and bias correction;

  * Science Highlight: 1) spectral surface albedo parameterization in the Arctic, and 2) satellite-aircraft spectral irradiance
    intercomparison for evaluating satellite-product-derived shortwave cloud radiative effects in the Arctic.


* `Schmidt and Pilewskie, 2012 <https://doi.org/10.1007/978-3-642-15531-4_6>`_

  Schmidt, S. and Pilewskie, P.: Airborne measurements of spectral shortwave radiation in cloud
  and aerosol remote sensing and energy budget studies, in: Light Scattering Reviews, Vol. 6:
  Light Scattering and Remote Sensing of Atmosphere and Surface, edited by: Kokhanovsky, A. A.,
  Springer, Berlin Heidelberg, 239–288, doi:10.1007/978-3-642-15531-4_6, 2012. 

  * Instrument Highlight: detailed instrument characterization;

  * Science Highlight: science development and outlook.

* `Pilewskie et al., 2003 <https://doi.org/10.1029/2002JD002411>`_

  Pilewskie, P., Pommier, J., Bergstrom, R., Gore, W., Howard, S., Rabbette, M., Schmid, B., Hobbs, P. V.,
  and Tsay, S. C.: Solar spectral radiative forcing during the Southern African Regional Science
  Initiative, J. Geophys. Res., 108, 8486, doi:10.1029/2002JD002411, 2003. 

  * Instrument Highlight: instrument characterization;

  * Science Highlight: measurement-derived flux divergence, fractional absorption, instantaneous heating rate, and absorption efficiency.
