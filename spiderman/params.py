import numpy as np
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

		elif brightness_model == 'xi':
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
			brightness_params = [self.T_s,self.pb]
		if (self.brightness_type == 2):
			brightness_params = [self.pb_d,self.pb_n]
		if (self.brightness_type == 3):
			brightness_params = [self.T_s,self.pb_d,self.pb_n]
		if (self.brightness_type == 4):
			brightness_params = [self.T_s,self.xi,self.T_n,self.delta_T]
		return brightness_params

	def calc_phase(self,t):
		phase = ((t-self.t0)/self.per)
		if(phase > 1):
			phase = phase - np.floor(phase)
		if(phase < 0):
			phase = phase + np.ceil(phase) + 1
		self.phase = phase

	def calc_substellar(self,t,coords):
		star_x = 0.0-coords[0]
		star_y = 0.0-coords[1]
		star_z = 0.0-coords[2]
		self.calc_phase(t)
		lambda0 = (np.pi + self.phase*2*np.pi)
		phi0 = np.arctan2(star_y,star_z)
		if(lambda0 > 2*np.pi):
			lambda0 = lambda0 - 2*np.pi;
		if(lambda0 < -2*np.pi):
			lambda0 = lambda0 + 2*np.pi;
		self.lambda0 = lambda0
		self.phi0 = phi0