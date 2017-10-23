Instrument response
=====================================

When calculating broadband phase curves, either for observations through a filter, or when summing all the data in a spectroscopic phase curve, the total instrument response may be required, as it weights how important each of the wavelengths are to the sum. Fortunately, spiderman has an easy way to account for this, simply provide a path to a "filter file" 

.. code-block:: python

	spider_params.filter = 'myfilter.txt'

This filter file must be a plain text file that consists of two columns, the wavelength in metres and the corresponding instument response value (typically a number between 0 and 1) and must be in units of counts per photon. The code will then convolve the given filter function with the fluxes when calculating physical models with grids of blackbodies or stellar model spectra. Spiderman will linearly interpolate between the provided wavelength points. 

If the filter function is too course, or if it contains a very sharply varying response then the results may not be accurate. In these cases it may be necessary to modify the "n_bb_seg" parameter in the lightcurve function, for which the default is 100.

.. warning:: SPIDERMAN normally calculates ratios in flux density assuming a uniform *energy flux* response - a uniform counts / photon response function will be different by a factor of hc / lambda, therefore it is possible to recover a different result with a uniform response function than you would get by setting the upper and lower bounds of the integration with l1 and l2.