import numpy as np
import spiderman as sp
import spiderman._web as _web
import spiderman.plot as splt
import matplotlib.pyplot as plt

class MultiModelParams(object):
	def __init__(self,brightness_models=['zhang','lambertian'],**kwargs):
			self.webs = []
			for i in range(0,len(brightness_models)):
				self.webs += [sp.ModelParams(brightness_models[i],**kwargs)]

	def lightcurve(self,*args,**kwargs):
		total = sp.lightcurve(*args,self.webs[0],**kwargs)
		for i in range(1,len(self.webs)):
			total += sp.lightcurve(*args,self.webs[i],**kwargs) - 1.0

		return total


class ModelParams(object):
	def __init__(self,brightness_model='zhang',thermal=False, nearest=None):

		self.n_layers = 5			# The default resolution for the grid

		self.t0= None				# The time of central **PRIMARY** transit [jd]
		self.per= None				# Orbital period of the planet [days]
		self.a_abs= None			# Absolute value of the semi major axis [AU]
		self.inc= None				# Inclination of the planetary orbit (90 is face on) [degrees]
		self.ecc= None				# Eccentricity
		self.w= None				# Longitude of periastron [degrees]
		self.a= None				# Semi major axis, scaled by stellar radius [-]
		self.rp= None				# Planet radius as a fraction of stellar radius [-]
		self.p_u1= None				# **PLANETARY** limb darkening coefficients [-]
		self.p_u2= None				# **PLANETARY** limb darkening coefficients [-]
		self.eclipse = True			# specifies whether to include the drop in flux due to the eclipse
		self.filter = False			# Can use an external response file.
		self.grid = [[],[],[[]]]			# needed in case the "direct read" method is wanted
		self.nearest = nearest         # used for choosing which interpolation model to use - default is spline


		if brightness_model == 'uniform brightness':
			self.n_layers = 1		# The default resolution for the grid

			self.brightness_type= 0	# Integer model identifier
			self.pb= None			# Relative planet brightness (Star is 1)
			self.thermal= False			# Is this a thermal distribution?

		elif brightness_model == 'uniform temperature':
			self.n_layers = 1		# The default resolution for the grid

			self.brightness_type= 1	# Integer model identifier
			self.T_p= None			# Relative planet brightness (Star is 1)
			self.T_s= None			# **STELLAR** effective temperature
			self.thermal= True			# Is this a thermal distribution?

		elif brightness_model == 'two brightness':
			self.brightness_type= 2 # Integer model identifer
			self.pb_d= None			# Relative planet brightness (Star is 1)
			self.pb_n= None			# Relative planet brightness (Star is 1)

			self.thermal= False			# Is this a thermal distribution?
			if not (hasattr(self,'grid_size')):
				self.grid_size = 10


		elif brightness_model == 'two temperature':
			self.brightness_type= 3	# Integer model identifier
			self.pb_d= None			# Relative planet brightness (Star is 1)
			self.pb_n= None			# Relative planet brightness (Star is 1)
			self.T_s= None			# **STELLAR** effective temperature
			self.thermal= True			# Is this a thermal distribution?
			if not (hasattr(self,'grid_size')):
				self.grid_size = 10

		elif brightness_model == 'zhang':
			self.brightness_type= 4	# Integer model identifier
			self.xi= None			# Ratio between radiative and advective timescales
			self.T_n= None			# Radiative solution temperature on night side
			self.delta_T= None		# Day/Night side difference between radiative-only temperature
			self.T_s= None			# **STELLAR** effective temperature
			self.thermal= True			# Is this a thermal distribution?

		elif brightness_model == 'spherical':
			self.a= None			# Ratio between radiative and advective timescales
			self.thermal= thermal			# Is this a thermal distribution?
			if thermal == True:
				self.brightness_type= 14	# Integer model identifier
			else:
				self.brightness_type= 5	# Integer model identifier

		elif brightness_model == 'kreidberg':
			self.brightness_type= 6 # Integer model identifer
			self.insol = None               # insolation in W/m^2
			self.albedo = None              # albedo
			self.redist = None              # fraction of incident energy redistributed to the night-side
			self.T_s = None

		elif brightness_model == 'hotspot_b':
			self.brightness_type= 7 # Integer model identifer
			self.thermal= False			# Is this a thermal distribution?
			if not (hasattr(self,'grid_size')):
				self.grid_size = 10


		elif brightness_model == 'hotspot_t':
			self.brightness_type= 8 # Integer model identifer
			self.T_s = None
			self.thermal= True			# Is this a thermal distribution?
			if not (hasattr(self,'grid_size')):
				self.grid_size = 10

		elif brightness_model == 'lambertian':
			self.brightness_type= 9 # Integer model identifer
			self.thermal= False			# Is this a thermal distribution?

		elif brightness_model == 'combine':
			self.brightness_type= 10 # Integer model identifer
			self.thermal= True			# Is this a thermal distribution?

		elif brightness_model == 'clouds':
			self.brightness_type= 11 # Integer model identifer
			self.thermal= True			# Is this a thermal distribution?

		elif brightness_model == 'direct_T':
			self.brightness_type= 12 # Integer model identifer
			self.thermal= True			# Is this a thermal distribution?

		elif brightness_model == 'direct_b':
			self.brightness_type= 13 # Integer model identifer
			self.thermal= False			# Is this a thermal distribution?

		else:
			print('Brightness model "'+str(brightness_model)+'" not recognised!')
			quit()

	def format_bright_params(self):
		if (self.brightness_type == 0):
			brightness_param_names = ['pb']
			try:
				brightness_params = [self.pb]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()
		elif (self.brightness_type == 1):
			brightness_param_names = ['T_s','l1','l2','T_p']
			try:
				brightness_params = [self.T_s,self.l1,self.l2,self.T_p]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()
		elif (self.brightness_type == 2):
			brightness_param_names = ['pb_d','pb_n']
#			try:
#				brightness_params = [self.pb_d,self.pb_n]

#			self.brightness_type = 7
			try:
				brightness_params = [0, 0, self.pb_n, self.grid_size,self.pb_d, 90]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()
		elif (self.brightness_type == 3):
			brightness_param_names = ['T_s','l1','l2','T_p_d','T_p_n']
#			try:
#				brightness_params = [self.T_s,self.l1,self.l2,self.T_p_d,self.T_p_n]
#			self.brightness_type = 8
			try:
				brightness_params = [self.T_s,self.l1, self.l2, self.grid_size, 0, 0, self.T_p_n, self.T_p_d, 90]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()
		elif (self.brightness_type == 4):
			brightness_param_names = ['T_s','l1','l2','xi','T_n','delta_T']
			try:
				brightness_params = [self.T_s,self.l1,self.l2,self.xi,self.T_n,self.delta_T]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()

		elif (self.brightness_type == 5 or self.brightness_type == 14):

			brightness_param_names = ['degree','sph','la0','lo0']
			if self.brightness_type == 14:
				brightness_param_names = ['T_s','l1','l2'] + brightness_param_names


			brightness_params = [self.degree,self.la0,self.lo0] + self.sph
			if self.brightness_type == 14:
				brightness_params = [self.T_s,self.l1,self.l2] + brightness_params

			try:
				brightness_params = [self.degree,self.la0,self.lo0] + self.sph
				if self.brightness_type == 14:
					brightness_params = [self.T_s,self.l1,self.l2] + brightness_params
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()
#			total_modes = (self.degree * (self.degree +1))/2.0
			total_modes = (self.degree)**2
			if len(self.sph) != total_modes:
				print('You have not specified the correct number of mode coefficients!')
				print('You gave '+str(int(len(self.sph)))+', there should be '+str(int(total_modes)))
				quit()

		elif (self.brightness_type == 6):
			brightness_param_names = ['T_s','l1','l2','insol','albedo','redist']
			try:
				brightness_params = [self.T_s, self.l1, self.l2, self.insol, self.albedo, self.redist]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()

		elif (self.brightness_type == 7):
			brightness_param_names = ['la0','lo0','p_b','spot_b','size']
			try:
				brightness_params = [self.la0, self.lo0, self.p_b, self.grid_size,self.spot_b, self.size]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()

		elif (self.brightness_type == 8):
			brightness_param_names = ['T_s','l1','l2','la0','lo0','p_T','spot_T','size']
			try:
				brightness_params = [self.T_s,self.l1, self.l2, self.grid_size, self.la0, self.lo0, self.p_T, self.spot_T, self.size]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()

		elif (self.brightness_type == 9):
			brightness_param_names = ['albedo']
			ars = 1.0/self.a
			r2 = 1.0/self.rp

			if not hasattr(self, 'T_s'):
			    self.T_s = 0
			    self.l1 = 0
			    self.l2 = 0

			try:
				brightness_params = [self.T_s,self.l1,self.l2,self.albedo,ars,r2]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()
		elif (self.brightness_type == 10):
			brightness_param_names = ['T_s','l1','l2','xi','T_n','delta_T','albedo']
			ars = 1.0/self.a
			r2 = 1.0/self.rp
			try:
				brightness_params = [self.T_s,self.l1,self.l2,self.xi,self.T_n,self.delta_T,self.albedo,ars,r2]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()

		elif (self.brightness_type == 11):
			brightness_param_names = ['T_s','l1','l2','xi','T_n','delta_T','cloud']
			try:
				brightness_params = [self.T_s,self.l1,self.l2,self.xi,self.T_n,self.delta_T,self.clouds]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()
		elif (self.brightness_type == 12):
			brightness_param_names = ['T_s','l1','l2','grid']
			n_lo = len(self.grid[0])
			n_la = len(self.grid[1])

#			chooses whether to use a 2d spline or a nearest neighbor method. default is spline.
			if self.nearest == None:
				nearest = 0
			elif self.nearest == True:
				nearest = 1 
			elif self.nearest == False:
				nearest = 0

			try:
				brightness_params = [self.T_s,self.l1,self.l2,n_lo,n_la,nearest]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()

		elif (self.brightness_type == 13):
			brightness_param_names = ['grid']
			n_lo = len(self.grid[0])
			n_la = len(self.grid[1])

#			chooses whether to use a 2d spline or a nearest neighbor method. default is spline.
			if self.nearest == None:
				nearest = 0
			elif self.nearest == True:
				nearest = 1 
			elif self.nearest == False:
				nearest = 0

			try:
				brightness_params = [n_lo,n_la,nearest]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()


		elif any(b == None for b in brightness_params):
			print('Brightness parameters incorrectly assigned')
			print('should be',brightness_param_names)
			quit()
		return brightness_params

	def plot_square(self,*args,**kwargs):
		return splt.plot_square(self,*args,**kwargs)

	def plot_system(self,*args,**kwargs):
		return splt.plot_system(self,*args,**kwargs)

	def plot_planet(self,*args,**kwargs):
		return splt.plot_planet(self,*args,**kwargs)

	def plot_quad(self,*args,**kwargs):
		return splt.plot_quad(self,*args,**kwargs)

	def plot_uncertainty(self,*args,**kwargs):
		return splt.plot_uncertainty(self,*args,**kwargs)

	def lightcurve(self,*args,**kwargs):
		return sp.lightcurve(*args,self,**kwargs)

	def calc_phase(self,t):
		self.phase = _web.calc_phase(t,self.t0,self.per)

	def calc_substellar(self,t):
		self.calc_phase(t)
		coords = sp.separation_of_centers(t,self)
		substellar = _web.calc_substellar(self.phase,np.array(coords))
		self.lambda0 = substellar[0]
		self.phi0 = substellar[1]

	def get_lims(self,t,temp_map=False,use_phase=False):
		if use_phase == True:
			if self.t0 == None:
				self.t0 = 0.0
			if self.per == None:
				self.per = 1.0
			t = self.t0 + self.per*t

		planet = sp.generate_planet(self,t)
		if temp_map == True:
			b_i = 17
		else:
			b_i = 16

		temps = planet[:,b_i]

		return [np.min(temps),np.max(temps)]

	def eclipse_depth(self,phase=0.5,stellar_grid=False):

		brightness_params = self.format_bright_params()

		if self.thermal == True:
			if stellar_grid == False:
				star_grid = sp.stellar_grid.gen_grid(self.l1,self.l2,logg=4.5,response=self.filter)
				teffs = star_grid[0]
				totals = star_grid[1]
			else:
				teffs = stellar_grid[0]
				totals = stellar_grid[1]
		else:
			teffs = []
			totals = []

		t = 0.0 + np.array([phase])

		if(self.inc == None):
			self.inc = 90.0

		if(self.p_u1 == None):
			self.p_u1 = 0.0

		if(self.p_u2 == None):
			self.p_u2 = 0.0

		if(self.a == None):
			self.a = 4.0

		if(self.w == None):
			self.w = 0.0

		if(self.ecc == None):
			self.ecc = 0.0


		if(self.a_abs == None):
			self.a_abs = 1.0

		if self.filter != False:
			use_filter = 1
			filter = get_filter(self.filter)
		else:
			use_filter = 0
			filter = [[],[]]

		n_wvls = len(filter[0])


		out = _web.lightcurve(self.n_layers,t,0.0,1.0,self.a_abs,self.inc,0.0,0.0,self.a,self.rp,self.p_u1,self.p_u2,self.brightness_type,brightness_params,teffs,totals,len(totals),0, filter[0], filter[1], n_wvls,use_filter,self.grid[0],self.grid[1],self.grid[2])[0] - _web.lightcurve(self.n_layers,t,0.0,1.0,self.a_abs,self.inc,0.0,0.0,self.a,self.rp,self.p_u1,self.p_u2,self.brightness_type,brightness_params,teffs,totals,len(totals),1, filter[0], filter[1], n_wvls,use_filter,self.grid[0],self.grid[1],self.grid[2])[0]

		return np.array(out)

	def total_luminosity(self,planet_radius,stellar_grid=False,reflection=False):
		p1,p2 = self.phase_brightness([0,0.5],stellar_grid=stellar_grid,reflection=reflection,planet_radius=planet_radius)
		return(p1+p2)


	def phase_brightness(self,phases,stellar_grid=False,reflection=False,planet_radius=False):

		if self.thermal == True:
			if stellar_grid == False:
				stellar_grid = sp.stellar_grid.gen_grid(self.l1,self.l2,logg=4.5,response=self.filter)
				teffs = stellar_grid[0]
				totals = stellar_grid[1]
			else:
				teffs = stellar_grid[0]
				totals = stellar_grid[1]
		else:
			teffs = []
			totals = []

		if type(phases) is not list: phases = [phases]

		brightness_params = self.format_bright_params()

		out_list = []
		for phase in phases:

			t = 0.0 + np.array([phase])

			if(self.t0 == None):
				self.t0 = 0.0

			if(self.per == None):
				self.per = 1.0

			if(self.inc == None):
				self.inc = 90.0

			if(self.p_u1 == None):
				self.p_u1 = 0.0

			if(self.p_u2 == None):
				self.p_u2 = 0.0

			if(self.a == None):
				self.a = 4.0

			if(self.w == None):
				self.w = 0.0

			if(self.ecc == None):
				self.ecc = 0.0


			if(self.a_abs == None):
				self.a_abs = 1.0


			planet = sp.generate_planet(self,t,stellar_grid=stellar_grid)

			brights = planet[:,16]
			areas = planet[:,15]
			inner = planet[:,13]
			outer = planet[:,14]

			avg_dist = ((inner + outer)/2)*np.pi/2
			avg_dist[0] = 0.0

			norm = np.sqrt(np.cos(avg_dist))

			min_bright = np.min(brights)
			max_bright = np.max(brights)

			if planet_radius == False:
				out = np.sum(brights*areas)/np.pi
			else:
				out = np.sum(brights*areas/norm)*planet_radius**2

			out_list += [out]

#			out = _web.lightcurve(self.n_layers,t,0.0,1.0,self.a_abs,self.inc,0.0,0.0,self.a,self.rp,self.p_u1,self.p_u2,self.brightness_type,brightness_params,teffs,totals,len(totals),0)[0] - 1.0
#			out_list += [out]


		if len(out_list) == 1:
			return out_list[0]

#		returns the total flux of the planet at each phase in the defined bandpass in Watts / m^2 / sr

		return out_list