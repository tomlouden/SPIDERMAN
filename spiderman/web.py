import spiderman._web as _web
import spiderman as sp
import numpy as np

def one_in_one_out(c1,c2,e1,e2,r_inner,r_outer,r2,x2,y2):
	return _web.one_in_one_out(c1,c2,e1,e2,r_inner,r_outer,r2,x2,y2)

def heron(a,b,c):
	return _web.heron(a,b,c)

def find_segment_area(c1x,c2x,r,theta):
	return _web.find_segment_area(c1x,c2x,r,theta)

def find_quad_area(a,b,c,d):
	return _web.find_quad_area(a,b,c,d)

def segment(r,theta):
	return _web.segment(r,theta)

def circle_intersect(x1,y1,r1,x2,y2,r2):
	return _web.circle_intersect(x1,y1,r1,x2,y2,r2)

def line_intersect(x1,y1,x2,y2,r2):
	return _web.line_intersect(x1,y1,x2,y2,r2)

def generate_planet(spider_params,t,use_phase=False,stellar_grid=False):
	if use_phase == True:
		t = spider_params.t0 + spider_params.per*t
	brightness_params = spider_params.format_bright_params()

	if spider_params.thermal == True:
		if stellar_grid == False:
			star_grid = sp.stellar_grid.gen_grid(spider_params.l1,spider_params.l2,logg=4.5)
			teffs = star_grid[0]
			totals = star_grid[1]
		else:
			teffs = stellar_grid[0]
			totals = stellar_grid[1]
	else:
		teffs = []
		totals = []

	spider_params.calc_substellar(t)
	return np.array(_web.generate_planet(spider_params.n_layers,spider_params.lambda0,spider_params.phi0,spider_params.p_u1,spider_params.p_u2,spider_params.brightness_type,brightness_params,teffs,totals,len(totals),spider_params.rp))

def call_map_model(spider_params,la,lo):
	brightness_params = spider_params.format_bright_params()
	_web.call_map_model(la,lo,spider_params.brightness_type,brightness_params)
	return np.array(_web.call_map_model(la,lo,spider_params.brightness_type,brightness_params))

def blocked(n_layers,x2,y2,r2):
	return _web.blocked(n_layers,x2,y2,r2)

def zhang_2016(lat,lon,xi,T_n,delta_T):
	return _web.zhang_2016(lat,lon,xi,T_n,delta_T)

def separation_of_centers(t,spider_params):
	ratio = 1/spider_params.rp
	return _web.separation_of_centers(t,spider_params.t0,spider_params.per,spider_params.a_abs,spider_params.inc,spider_params.ecc,spider_params.w,spider_params.a,ratio)

def lightcurve(t,spider_params,stellar_grid=False):

	brightness_params = spider_params.format_bright_params()

	if spider_params.thermal == True:
		if stellar_grid == False:
			star_grid = sp.stellar_grid.gen_grid(spider_params.l1,spider_params.l2,logg=4.5)
			teffs = star_grid[0]
			totals = star_grid[1]
		else:
			teffs = stellar_grid[0]
			totals = stellar_grid[1]
	else:
		teffs = []
		totals = []

	return _web.lightcurve(spider_params.n_layers,t,spider_params.t0,spider_params.per,spider_params.a_abs,spider_params.inc,spider_params.ecc,spider_params.w,spider_params.a,spider_params.rp,spider_params.p_u1,spider_params.p_u2,spider_params.brightness_type,brightness_params,teffs,totals,len(totals), spider_params.eclipse)

def bb_grid(l1,l2,T_start,T_end,n_temps,n_segments):
	temps, fluxes, deriv = _web.bb_grid(l1,l2,T_start,T_end,n_temps,n_segments)
	return np.array(temps), np.array(fluxes), np.array(deriv)

