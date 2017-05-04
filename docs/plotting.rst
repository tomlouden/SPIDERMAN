Plotting 
============

Generating a simple spectrum
-----------------------------

Sometimes you don't want to bother with running a full orbital model, and just want a quick estimate of the eclipse depth of system. Spiderman has a couple of techniques to allow you to do this.

.. code-block:: python

	spider_params.l1 = 1.1e-6	# The starting wavelength in meters
	spider_params.l2 = 1.7e-6	# The ending wavelength in meters

This method only returns the *blocked light* relative to the stellar brightness at the specified phase, so for an example, if you were to specify a grazing transit you would not recieve the total flux of the dayside.

If you do want the total flux of the planet from a specific phase, you can instead use the "phase_brightness" method. Using this method you could calulate an emission spectrum
