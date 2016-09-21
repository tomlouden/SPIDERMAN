import os.path
import fitsio
import matplotlib.pyplot as plt
import numpy as np

PHOENIX_DIR = '/storage/astro2/phrmat/PHOENIX'

def gen_grid(l1,l2):

	logg = 4.5
	z = -0.0

	print('Generating stellar flux grid')

	h =6.62607004e-34;
	c =299792458.0;
	kb =1.38064852e-23;

	teffs = [2500,3000,3500,4000,4500,5000,5500,6000,6500,7000]

	totals = []
	for teff in teffs:
		wvl, flux = get_phoenix_spectra(teff,logg,z)
		b_flux = (2.0*h*(c**2)/(wvl**5))*(1.0/( np.exp( (h*c)/(wvl*kb*teff) )- 1.0));
		totals += [sum_flux(wvl,flux,l1,l2)]
	teffs = np.array(teffs)
	totals = np.array(totals)

	return [teffs, totals]


def sum_flux(wvl,flux,l1,l2):

	mask = [(wvl > l1) & (wvl < l2)]

	diff = np.diff(wvl)
	diff = (np.append(diff,diff[-1:]) + np.append(diff[1:],diff[-2:]))/2

	diff = diff[mask]

	wvl = wvl[mask]
	flux = flux[mask]

	total = 0.0

	for i in range(0,len(wvl)):
		total += flux[i]*diff[i]
	return total

def get_phoenix_spectra(teff,logg,z):
	ftemplate = 'lte{teff:05d}-{logg:4.2f}{z:+3.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'

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