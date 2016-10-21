import spiderman as sp
import numpy as np
import matplotlib.pyplot as plt
import time as timing

def plot_test():
	spider_params = sp.ModelParams(brightness_model='zhang')

	spider_params.n_layers= 20

	spider_params.t0= 200               # Central time of PRIMARY transit [days]
	spider_params.per= 0.81347753       # Period [days]
	spider_params.a_abs= 0.01526        # The absolute value of the semi-major axis [AU]
	spider_params.inc= 82.33            # Inclination [degrees]
	spider_params.ecc= 0.0              # Eccentricity
	spider_params.w= 90                 # Argument of periastron
	spider_params.rp= 0.1594            # Planet to star radius ratio
	spider_params.a= 4.855              # Semi-major axis scaled by stellar radius
	spider_params.p_u1= 0               # Planetary limb darkening parameter
	spider_params.p_u2= 0               # Planetary limb darkening parameter

	spider_params.xi= 0.3       # Ratio of radiative to advective timescale             
	spider_params.T_n= 1128     # Temperature of nightside
	spider_params.delta_T= 942  # Day-night temperature contrast
	spider_params.T_s = 5000    # Temperature of the star

	spider_params.l1 = 1.3e-6    # start of integration channel in microns
	spider_params.l2 = 1.6e-6    # end of integration channel in microns

	t= spider_params.t0 + np.linspace(0, + spider_params.per,100)

	lc = sp.lightcurve(t,spider_params)

	plt.plot(t,lc)
	plt.show()


def time_test(nlayers=5,tpoints=100,nreps=1000):

	spider_params = sp.ModelParams(brightness_model='zhang')

#	spider_params = sp.ModelParams(brightness_model='uniform brightness')

	spider_params.n_layers= nlayers

	spider_params.t0= 200               # Central time of PRIMARY transit [days]
	spider_params.per= 0.81347753       # Period [days]
	spider_params.a_abs= 0.01526        # The absolute value of the semi-major axis [AU]
	spider_params.inc= 82.33            # Inclination [degrees]
	spider_params.ecc= 0.0              # Eccentricity
	spider_params.w= 90                 # Argument of periastron
	spider_params.rp= 0.1594            # Planet to star radius ratio
	spider_params.a= 4.855              # Semi-major axis scaled by stellar radius
	spider_params.p_u1= 0               # Planetary limb darkening parameter
	spider_params.p_u2= 0               # Planetary limb darkening parameter

	spider_params.xi= 0.3       # Ratio of radiative to advective timescale             
	spider_params.T_n= 1128     # Temperature of nightside
	spider_params.delta_T= 942  # Day-night temperature contrast
	spider_params.T_s = 4500    # Temperature of the star

	spider_params.l1 = 1.3e-6    # start of integration channel in microns
	spider_params.l2 = 1.6e-6    # end of integration channel in microns

	spider_params.pb = 0.01    # planet relative brightness

	t= spider_params.t0 + np.linspace(0, + spider_params.per,tpoints)

	print('')
	print('About to generate {} lightcurves with {} layers and {} timepoints'.format(nreps,spider_params.n_layers,tpoints))
	print('')

	start = timing.time()

	star_grid = sp.stellar_grid.gen_grid(spider_params.l1,spider_params.l2)

	ends = []
	for i in range(0,nreps):
		lc = sp.lightcurve(t,spider_params,stellar_grid=star_grid)
		ends += [timing.time()]
	ends = np.array(ends)

	exec_times = np.diff(ends)

	total = ends[-1] - start

	medtime = np.median(exec_times)
	stdtimes = np.std(exec_times)
	medtime = np.median(exec_times)


	print('In total it took {} seconds'.format(round(total,2)))
	print('Each function call was between {:.2E} and {:.2E}seconds'.format(np.min(exec_times),np.max(exec_times)))
	print('Median execution time was {:.2E} seconds'.format(medtime))
	print('Standard deviation was {:.2E} seconds'.format(stdtimes))
	print('{} lightcurves generated per second!'.format(round(1.0/medtime),1))
	print('')
