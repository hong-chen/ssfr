SSFR Software Package
~~~~~~~~~~~~~~~~~~~~~
SSFR (Solar Spectral Flux Radiometer) is an airborne instrument co-developed by teams
at NASA Ames Research Center (discontinued) and at LASP of University of Colorado Boulder.

This repository contains code for processing SSFR data by applying required calibrations and corrections.

A more detailed description can be found `here <https://docs.google.com/document/d/1ObczXucJQktyTgKZlBkL04fjhHFx1ydW0sPaiG7iZ9k/edit?usp=sharing>`_ (under development).

So far, the SSFR has been deployed in the following airborne missions:

* `NASA CAMP²Ex <https://espo.nasa.gov/camp2ex/content/CAMP2Ex>`_ (on P-3 in 2019);

* `NASA ORACLES <https://espo.nasa.gov/ORACLES/content/ORACLES>`_ (on ER-2 and P-3 in 2016, 2017 and 2018) ;

* `NASA ARISE <https://espo.nasa.gov/arise/content/ARISE>`_ (on C-130 in 2014);

* `NASA SEAC⁴RS <https://espo.nasa.gov/seac4rs>`_ (on DC-8 in 2013);

* `NASA SAFARI 2000 <https://espo.nasa.gov/content/SAFARI_2000>`_ (on ER-2 and Convair-580 in 2000).

==============
How to Install
==============
::

    git clone https://github.com/hong-chen/ssfr.git
    python setup.py develop

==========
References
==========

* `Chen et al., 2021 <https://doi.org/10.5194/amt-14-2673-2021>`_

  Chen, H., Schmidt, S., King, M. D., Wind, G., Bucholtz, A., Reid, E. A., Segal-Rozenhaimer, M.,
  Smith, W. L., Taylor, P. C., Kato, S., and Pilewskie, P.: The effect of low-level thin arctic
  clouds on shortwave irradiance: evaluation of estimates from spaceborne passive imagery with
  aircraft observations, Atmos. Meas. Tech., 14, 2673–2697, doi:10.5194/amt-14-2673-2021, 2021.

* `Schmidt and Pilewskie, 2012 <https://doi.org/10.1007/978-3-642-15531-4_6>`_

  Schmidt, S. and Pilewskie, P.: Airborne measurements of spectral shortwave radiation in cloud
  and aerosol remote sensing and energy budget studies, in: Light Scattering Reviews, Vol. 6:
  Light Scattering and Remote Sensing of Atmosphere and Surface, edited by: Kokhanovsky, A. A.,
  Springer, Berlin Heidelberg, 239–288, doi:10.1007/978-3-642-15531-4_6, 2012. 

* `Pilewskie et al., 2003 <https://doi.org/10.1029/2002JD002411>`_

  Pilewskie, P., Pommier, J., Bergstrom, R., Gore, W., Howard, S., Rabbette, M., Schmid, B., Hobbs, P. V.,
  and Tsay, S. C.: Solar spectral radiative forcing during the Southern African Regional Science Initiative,
  J. Geophys. Res., 108, 8486, doi:10.1029/2002JD002411, 2003. 
