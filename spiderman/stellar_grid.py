# -*- coding: utf-8 -*-
import os.path
import fitsio
import matplotlib.pyplot as plt
import numpy as np
import spiderman
from scipy.interpolate import interp1d

def gen_grid(l1,l2,logg=4.5, response = False, stellar_model = "blackbody"):
        #options for stellar models are "blackbody", "PHOENIX", and "path_to_model"
	z = -0.0

	h =6.62607004e-34;          #m^2/kg/s
	c =299792458.0;             #m/s
	kb =1.38064852e-23;         #m^2 kg /s^2 K

	teffs = [2500,3000,3500,4000,4500,5000,5500,6000,6500,7000]

	warned = False

	filter = response
	if response != False:
		filter = spiderman.get_filter(response)

	totals = []
	for teff in teffs:
		if stellar_model == "PHOENIX":
                        if spiderman.rcParams.read == False: print('Add path to PHOENIX models to .spidermanrc file')
			wvl, flux = get_phoenix_spectra(teff,logg,z)
			PHOENIX_DIR = spiderman.rcParams['PHOENIX_DIR']

			if warned == False:
				print('using stellar spectra in '+PHOENIX_DIR)

			if ( ((l1 > np.min(wvl)) & (l1 < np.max(wvl))) & ((l2 > np.min(wvl)) & (l2 < np.max(wvl) )) ):
				totals += [sum_flux(wvl,flux,l1,l2,filter)]
			else:
				if warned == False:
					print('wavelengths out of bound for stellar model, using blackbody approximation')
				b_wvl = np.linspace(l1,l2,1000)
				b_flux = (2.0*h*(c**2)/(b_wvl**5))*(1.0/( np.exp( (h*c)/(b_wvl*kb*teff) )- 1.0));
				totals += [sum_flux(b_wvl,b_flux,l1,l2,filter)]
                elif stellar_model == "blackbody":
			if warned == False:
				print('no stellar models provided, using blackbody approximation')
			b_wvl = np.linspace(l1,l2,1000)
                        b_flux = (2.0*h*(c**2)/(b_wvl**5))*(1.0/( np.exp( (h*c)/(b_wvl*kb*teff) )- 1.0));       #SI units: W/sr/m^3
			totals += [sum_flux(b_wvl,b_flux,l1,l2,filter)]
                else:
                        if os.path.isfile(stellar_model):
                            spectrum = np.genfromtxt(stellar_model)
                            wvl, flux = spectrum[:,0], spectrum[:,1] 
                        else: print "Model stellar spectrum file", stellar_model, "not found"

			if ( ((l1 > np.min(wvl)) & (l1 < np.max(wvl))) & ((l2 > np.min(wvl)) & (l2 < np.max(wvl) )) ):
				totals += [sum_flux(wvl,flux,l1,l2,filter)]
			else:
				if warned == False:
					print('wavelengths out of bound for stellar model, using blackbody approximation')
				b_wvl = np.linspace(l1,l2,1000)
				b_flux = (2.0*h*(c**2)/(b_wvl**5))*(1.0/( np.exp( (h*c)/(b_wvl*kb*teff) )- 1.0));
				totals += [sum_flux(b_wvl,b_flux,l1,l2,filter)]
		warned = True


	teffs = np.array(teffs)
	totals = np.array(totals)

	return [teffs, totals]

def sum_flux(wvl,flux,l1,l2,filter=False):


	mask = [(wvl > l1) & (wvl < l2)]

	diff = np.diff(wvl)
	diff = (np.append(diff,diff[-1:]) + np.append(diff[1:],diff[-2:]))/2

	diff = diff[mask]

	wvl = wvl[mask]
	flux = flux[mask]

	if filter != False:
		f = interp1d(filter[0],filter[1],kind='linear',bounds_error=True,axis=0)
		r = f(wvl)
	else:
		r = np.array([1.0]*len(wvl))

	total = 0.0

	for i in range(0,len(wvl)):
		total += r[i]*flux[i]*diff[i]
	return total

def get_phoenix_spectra(teff,logg,z):
	ftemplate = 'lte{teff:05d}-{logg:4.2f}{z:+3.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'

	PHOENIX_DIR = spiderman.rcParams['PHOENIX_DIR']

	filename = os.path.join(PHOENIX_DIR,ftemplate.format(teff=teff,logg=logg,z=z))

	# changing to si, W / m^3 / str
	flux,h = fitsio.read(filename, ext=0, header=True)

	flux = flux*1e-7*1e6/(np.pi)

	crval = h['CRVAL1']
	cdelt = h['CDELT1']
	ctype = h['CTYPE1']

	if ctype == 'AWAV-LOG':
		wvl = (np.exp(crval + cdelt*np.arange(0,len(flux))))*1e-10
	else:
		print('ctype is not log! It  is {}'.format(ctype))

	return wvl, flux
