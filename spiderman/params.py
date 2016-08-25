import numpy as np
import spiderman as sp
import spiderman._web as _web
import spiderman.plot as splt

class ModelParams(object):

	def __init__(self,brightness_model='xi'):

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

		if brightness_model == 'uniform brightness':
			self.n_layers = 1		# The default resolution for the grid

			self.brightness_type= 0	# Integer model identifier
			self.pb= None			# Relative planet brightness (Star is 1)

		elif brightness_model == 'uniform temperature':
			self.n_layers = 1		# The default resolution for the grid

			self.brightness_type= 1	# Integer model identifier
			self.pb= None			# Relative planet brightness (Star is 1)
			self.T_s= None			# **STELLAR** effective temperature

		elif brightness_model == 'two brightness':
			self.brightness_type= 2	# Integer model identifier
			self.pb_d= None			# Relative planet brightness (Star is 1)
			self.pb_n= None			# Relative planet brightness (Star is 1)

		elif brightness_model == 'two temperature':
			self.brightness_type= 3	# Integer model identifier
			self.pb_d= None			# Relative planet brightness (Star is 1)
			self.pb_n= None			# Relative planet brightness (Star is 1)
			self.T_s= None			# **STELLAR** effective temperature

		elif brightness_model == 'zhang':
			self.brightness_type= 4	# Integer model identifier
			self.xi= None			# Ratio between radiative and advective timescales
			self.T_n= None			# Radiative solution temperature on night side
			self.delta_T= None		# Day/Night side difference between radiative-only temperature
			self.T_s= None			# **STELLAR** effective temperature

		else:
			print('Brightness model "'+str(brightness_model)+'" not recognised!')
			quit()

	def format_bright_params(self):
		if (self.brightness_type == 0):
			brightness_params = [self.pb]
		if (self.brightness_type == 1):
			brightness_params = [self.T_s,self.l1,self.l2,self.pb]
		if (self.brightness_type == 2):
			brightness_params = [self.pb_d,self.pb_n]
		if (self.brightness_type == 3):
			brightness_params = [self.T_s,self.l1,self.l2,self.pb_d,self.pb_n]
		if (self.brightness_type == 4):
			brightness_params = [self.T_s,self.l1,self.l2,self.xi,self.T_n,self.delta_T]
		return brightness_params

	def calc_phase(self,t):
		self.phase = _web.calc_phase(t,self.t0,self.per)

	def calc_substellar(self,t):
		self.calc_phase(t)
		coords = sp.separation_of_centers(t,self)
		substellar = _web.calc_substellar(self.phase,np.array(coords))
		self.lambda0 = substellar[0]
		self.phi0 = substellar[1]

	def plot_system(self,t,ax=False,min_temp=False,max_temp=False,temp_map=False,min_bright=0,use_phase=False):
		if use_phase == True:
			t = self.t0 + self.per*t
		return splt.plot_system(self,t,ax=ax,min_temp=min_temp,max_temp=max_temp,temp_map=temp_map,min_bright=min_bright)

	def plot_planet(self,t,ax=False,min_temp=False,max_temp=False,temp_map=False,min_bright=0,scale_planet=1.0,planet_cen=[0.0,0.0],use_phase=False):
		if use_phase == True:
			t = self.t0 + self.per*t
		return splt.plot_planet(self,t,ax=ax,min_temp=min_temp,max_temp=max_temp,temp_map=temp_map,min_bright=min_bright,scale_planet=scale_planet,planet_cen=planet_cen)

	def lightcurve(self,t,use_phase=False):
		brightness_params = self.format_bright_params()
		if use_phase == True:
			t = self.t0 + self.per*t
		return _web.lightcurve(self.n_layers,t,self.t0,self.per,self.a_abs,self.inc,self.ecc,self.w,self.a,self.rp,self.p_u1,self.p_u2,self.brightness_type,brightness_params)