import spiderman.web as web
import numpy as np
import matplotlib.pyplot as plt

tc = 200.0
per = 0.81347753
a = 0.01526
inc = 82.33
ecc = 0.0
omega = 90
a_rs = 4.855
rp = 0.15938366414961666

xi = 0.38793124238743942
T_n = 326.46237915041326
delta_T = 240.91179648407561

#xi = 5.1
#xi = 0.03
#T_n = 294.7
#delta_T = 471.8

star_bright = 337073244342096.5

n_slice = 20

n_ts = 401

ts = np.linspace(tc-(per/2),tc+(per/2),n_ts)

phase_c = []
phases = []

ratio = 1/rp
star_r = ratio

dp = (T_n*1.2)**4
min_temp = T_n**4
max_temp = (T_n + delta_T)**4


for i in range(0,len(ts)-1):
	t = ts[i]
	coords = web.separation_of_centers(t,tc,per,a,inc,ecc,omega,a_rs,ratio)

	star_x = 0.0-coords[0]
	star_y = 0.0-coords[1]
	star_z = 0.0-coords[2]

	ni = str(i)

	if len(ni) < 2:
		ni = '0'+ni

	plotname = 'plots/test_'+ni+'.png'

	phase = ((t-tc)/per)

	if(phase > 1):
		phase = phase - np.floor(phase)
	if(phase < 0):
		phase = phase + np.ceil(phase) + 1

	lambda0 = (np.pi + phase*2*np.pi)
	phi0 = np.tan(star_y/star_z)
	if(lambda0 > 2*np.pi):
		lambda0 = lambda0 - 2*np.pi;
	if(lambda0 < -2*np.pi):
		lambda0 = lambda0 + 2*np.pi;


	phases += [phase]

	planet = np.array(web.generate_planet(n_slice,xi,T_n,delta_T,lambda0,phi0))

	print(i+1,len(ts))

	ax = plt.subplot(111, aspect='equal')

	phase_c += [np.sum(planet[:,16])]

	thetas = np.linspace(planet[0][10],planet[0][11],100)
	r = planet[0][14]
	radii = [0,r]
	xs = np.outer(radii, np.cos(thetas)) - coords[0]
	ys = np.outer(radii, np.sin(thetas)) - coords[1]
	xs[1,:] = xs[1,::-1]
	ys[1,:] = ys[1,::-1]

	val = (dp + planet[0][16]-min_temp)/(dp + max_temp-min_temp)
	print(0,val)

	c = plt.cm.inferno(val)
	ax.fill(np.ravel(xs), np.ravel(ys), edgecolor=c,color=c,zorder=2)

	for i in range (1,len(planet)):
		val = (dp + planet[i][16]-min_temp)/(dp + max_temp-min_temp)
		print(i,val)
		c = plt.cm.inferno(val)

		n = planet[i]

		r1 = n[13]
		r2 = n[14]
		radii = [r1,r2]

		thetas = np.linspace(n[10],n[11],100)

		xs = np.outer(radii, np.cos(thetas)) - coords[0]
		ys = np.outer(radii, np.sin(thetas)) - coords[1]

		xs[1,:] = xs[1,::-1]
		ys[1,:] = ys[1,::-1]

		ax.fill(np.ravel(xs), np.ravel(ys), edgecolor=c,color=c,zorder=2)

	plt.savefig("first_lc.pdf",set_bbox_inches='tight')

	thetas = np.linspace(planet[0][10],planet[0][11],100)
	radii = [0,star_r]
	xs = np.outer(radii, np.cos(thetas))
	ys = np.outer(radii, np.sin(thetas))
	xs[1,:] = xs[1,::-1]
	ys[1,:] = ys[1,::-1]
	c = plt.cm.inferno(1.0)

	if(abs(abs(phase)-0.5) > 0.25):
		#in front 
		s_zorder = 1
	else:
		s_zorder = 3

	ax.fill(np.ravel(xs), np.ravel(ys), edgecolor=c,color=c,zorder=s_zorder)

	ax.set_axis_bgcolor('black')

	#ax.set_xlim(-star_r-4,star_r+4)
	#ax.set_ylim(-star_r-4,star_r+4)

	ax.set_xlim(-40,40)
	ax.set_ylim(-10,10)

	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	plt.savefig(plotname, bbox_inches='tight')
	plt.close()

plt.plot(ts[:-1]-tc,phase_c)
plt.show()