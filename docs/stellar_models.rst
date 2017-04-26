Stellar models
=====================================

Using the spidermanrc file
--------------------

In some circumstances, you might want to attempt to directly determine the absolute brightness temperature of the planet, instead of just the brigtness relative to the star. Some physical models may require you to do this, for example the zhang and showman (2017) model.

spidermanrc at the moment is used to tell the code where the phoenix stellar spectral models are stored, if you want to use them instead of a simple blackbody model in the code. This works better if you are using models for the surface brightness of the planet that are calculated from brightness temperatures, like the Zhang and Showman model.
The ones I've been using are the R=10000 spectra from this page: ftp://phoenix.astro.physik.uni-goettingen.de/MedResFITS/R10000FITS/PHOENIX-ACES-AGSS-COND-2011_R10000FITS_Z-0.0.zip . Download and unpack them all into a directory, then make a ".spidermanrc" file in your home directory with a line that points to this directory, mine looks like this:

PHOENIX_DIR : /storage/astro2/phrmat/PHOENIX/

A spidermanrc file is not required - SPIDERMAN will run happily without a spidermanrc file and just calculate blackbodies.

In future releases the spidermanrc file may contain additional global parameters.