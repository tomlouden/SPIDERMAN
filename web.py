import SPIDERMAN._heron as _web
import SPIDERMAN._polar as _polar

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

def polar(r1,r2,theta1,theta2):
	return _polar.polar(r1,r2,theta1,theta2)