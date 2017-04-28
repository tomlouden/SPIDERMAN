Stellar models
=====================================

In some circumstances, you might want to attempt to directly determine the absolute brightness temperature of the planet, instead of just the brightness relative to the star. Some physical models may require you to do this, for example the zhang and showman (2017) model.

To do this you need the absolute brightnes of the star - don't worry, spiderman has you covered!

All models that require thermal information take the stellar effective temperature as a model parameter, and the bandpass of the observations must also be specified like so:

.. code-block:: python

	spider_params.l1 = 4520		# The stellar effective temperature in K
	spider_params.l1 = 1.1e-6	# The starting wavelength in meters
	spider_params.l2 = 1.7e-6	# The ending wavelength in meters

At the most basic level, you can assume that the spectrum of the star is simply blackbody with the effective temperature of the star. This is the default mode for spiderman. Over very wide bandpasses this will be good enough, but for narrower spectral ranges this could introduce significant errors, particularly for cooler stars with deep molecular absorption bands.

Of course, a better option would be to use realistic stellar spectra to determine the stellar flux - spiderman can do this as well, but it requires an external library of spectra, that are specified using the .spidermanrc file:

Using the spidermanrc file
--------------------------

a .spidermanrc file is used to give global parameters to the code. At present, the only setting needed is the location of model stellar spectra, but more customisation options may be added in future updates.

.. note:: Currently, the only supported model spectra are the R=10000 PHOENIX models from this page: ftp://phoenix.astro.physik.uni-goettingen.de/MedResFITS/R10000FITS/PHOENIX-ACES-AGSS-COND-2011_R10000FITS_Z-0.0.zip

Download and unpack the model spectra, then make a ".spidermanrc" file in your home directory with a line that points to this model spectra with the keyword PHOENIX_DIR, like this:

PHOENIX_DIR : /path/to/PHOENIX/

A spidermanrc file is not required - SPIDERMAN will run happily without a spidermanrc file and just calculate blackbodies for the stellar atmosphere instead.

In future releases the spidermanrc file may contain additional global parameters.

Precalculating grids of models
-------------------------------

In order to speed up computation, spiderman will automatically generate a grid of the *summed stellar flux* in the defined bandpass as a function of stellar effective temperature, when calculating the flux of the star given its effective temperature spiderman will then use this grid to interpolate on to find the appropriate temperature, to a high level of precision.

.. warning:: spiderman currently only interpolates the stellar flux in 1D (Temperature) and assumes by default that the star is a dwarf with logg 4.5 - 2d interpolation with logg will be included in a future update

If you a running an MCMC, or another numerical method that requires you to call spiderman many thousands of times, especially if you are including the stellar temperature as a model parameter, you should *precalculate* this grid before begining the model fit and pass it to spiderman to increase the efficiency. This can be done very easily with the stellar_grid module:

.. code-block:: python

	stellar_grid = spiderman.stellar_grid.gen_grid(l1,l2,logg=4.5)

Where l1 and l2 are the begining and end of the spectral window in meters, and logg is the cgs surface gravity of the star. The stellar_grid object is then passed to spiderman for every light curve generation instance, e.g.

.. code-block:: python

	lc = spider_params.lightcurve(t,stellar_grid=stellar_grid)

If a stellar grid is not provided, spiderman will calculate it internally every time lightcurve is called - this will be significantly less efficient for long runs.