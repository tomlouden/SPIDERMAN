Plotting 
============

Generating a simple spectrum
-----------------------------

Sometimes you don't want to bother with running a full orbital model, and just want a quick estimate of the eclipse depth of system. Spiderman has a couple of techniques to allow you to do this.

If all you want is the eclipsed depth, you can use the "eclipse_depth" method, like so:

.. code-block:: python

	import spiderman as sp
	import numpy as np
	import matplotlib.pyplot as plt

	spider_params = sp.ModelParams(brightness_model='zhang')
	spider_params.n_layers = 5

for this example we'll use a Zhang and Showman type model with 5 layers, next the relevent model parameters are entered - 

.. code-block:: python

	spider_params.l1 = 1.1e-6	# The starting wavelength in meters
	spider_params.l2 = 1.7e-6	# The ending wavelength in meters

	spider_params.T_s = 4520
	spider_params.rp = 0.159692

	spider_params.xi = 0.1
	spider_params.T_n = 1000
	spider_params.delta_T = 1000

Note that if all you want is a simple eclipse depth is all you want, there's no need to enter the orbital parameters. Spiderman will assume a circular orbit and an inclination of 90 degrees unless you tell it otherwise. Now, you can call the eclipse_depth:

.. code-block:: python

	d = spider_params.eclipse_depth()
	print(d)
	>> 0.00045781826310942186

This method only returns the *blocked light* relative to the stellar brightness at the specified phase, so for an example, if you were to specify a grazing transit you would not recieve the total flux of the dayside.

If you do want the total flux of the planet from a specific phase, you can instead use the "phase_brightness" method. Using this method you could calulate an emission spectrum
