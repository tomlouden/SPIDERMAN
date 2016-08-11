import spiderman._web as _web

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

def generate_planet(n_layers,xi,T_n,delta_T,lambda0,phi0,u1,u2):
	return _web.generate_planet(n_layers,xi,T_n,delta_T,lambda0,phi0,u1,u2)

def blocked(n_layers,x2,y2,r2):
	return _web.blocked(n_layers,x2,y2,r2)

def zhang_2016(lat,lon,xi,T_n,delta_T):
	return _web.zhang_2016(lat,lon,xi,T_n,delta_T)

def separation_of_centers(t,tc,per,a,inc,ecc,omega,a_rs,r2):
	return _web.separation_of_centers(t,tc,per,a,inc,ecc,omega,a_rs,r2)

def lightcurve(t,sp):
	brightness_params = sp.format_bright_params()
	return _web.lightcurve(sp.n_layers,t,sp.t0,sp.per,sp.a_abs,sp.inc,sp.ecc,sp.w,sp.a,sp.rp,sp.p_u1,sp.p_u2,sp.brightness_type,brightness_params)