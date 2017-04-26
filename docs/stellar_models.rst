Stellar models
=====================================

In some circumstances, you might want to attempt to directly determine the absolute brightness temperature of the planet, instead of just the brightness relative to the star. Some physical models may require you to do this, for example the zhang and showman (2017) model.

To do this you need the absolute brightnes of the star - don't worry, spiderman has you covered!

Models that require thermal information take the stellar effective temperature and the bandpass of the observations must also be specified:


.. warning:: spiderman currently only interpolates the stellar flux in 1D (Temperature) and assumes that the star has a logg 


Using the spidermanrc file
--------------------------

spidermanrc is used to give global parameters to the code. At present, the only setting is the location of model stellar spectra.

.. note:: Currently, the only supported model spectra are the R=10000 PHOENIX models from this page: 
ftp://phoenix.astro.physik.uni-goettingen.de/MedResFITS/R10000FITS/PHOENIX-ACES-AGSS-COND-2011_R10000FITS_Z-0.0.zip

Download and unpack the model spectra, then make a ".spidermanrc" file in your home directory with a line that points to this model spectra with the keyword PHOENIX_DIR, like this:

PHOENIX_DIR : /path/to/PHOENIX/

A spidermanrc file is not required - SPIDERMAN will run happily without a spidermanrc file and just calculate blackbodies for the stellar atmosphere instead.

In future releases the spidermanrc file may contain additional global parameters.