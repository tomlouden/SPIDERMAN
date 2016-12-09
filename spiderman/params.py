import numpy as np
import spiderman as sp
import spiderman._web as _web
import spiderman.plot as splt
import matplotlib.pyplot as plt

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
			self.thermal= False			# Is this a thermal distribution?

		elif brightness_model == 'uniform temperature':
			self.n_layers = 1		# The default resolution for the grid

			self.brightness_type= 1	# Integer model identifier
			self.T_p= None			# Relative planet brightness (Star is 1)
			self.T_s= None			# **STELLAR** effective temperature
			self.thermal= True			# Is this a thermal distribution?

		elif brightness_model == 'two brightness':
			self.brightness_type= 2	# Integer model identifier
			self.pb_d= None			# Relative planet brightness (Star is 1)
			self.pb_n= None			# Relative planet brightness (Star is 1)

		elif brightness_model == 'two temperature':
			self.brightness_type= 3	# Integer model identifier
			self.pb_d= None			# Relative planet brightness (Star is 1)
			self.pb_n= None			# Relative planet brightness (Star is 1)
			self.T_s= None			# **STELLAR** effective temperature
			self.thermal= True			# Is this a thermal distribution?

		elif brightness_model == 'zhang':
			self.brightness_type= 4	# Integer model identifier
			self.xi= None			# Ratio between radiative and advective timescales
			self.T_n= None			# Radiative solution temperature on night side
			self.delta_T= None		# Day/Night side difference between radiative-only temperature
			self.T_s= None			# **STELLAR** effective temperature
			self.thermal= True			# Is this a thermal distribution?

		elif brightness_model == 'spherical':
			self.brightness_type= 5	# Integer model identifier
			self.a= None			# Ratio between radiative and advective timescales
			self.thermal= False			# Is this a thermal distribution?

		elif brightness_model == 'kreidberg':
			self.brightness_type= 6 # Integer model identifer
			self.insol = None               # insolation in W/m^2
			self.albedo = None              # albedo
			self.redist = None              # fraction of incident energy redistributed to the night-side
			self.T_s = None

		elif brightness_model == 'hotspot_b':
			self.brightness_type= 7 # Integer model identifer
			self.thermal= False			# Is this a thermal distribution?

		elif brightness_model == 'hotspot_t':
			self.brightness_type= 8 # Integer model identifer
			self.T_s = None
			self.thermal= True			# Is this a thermal distribution?

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
		if (self.brightness_type == 1):
			brightness_param_names = ['T_s','l1','l2','T_p']
			try:
				brightness_params = [self.T_s,self.l1,self.l2,self.T_p]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()
		if (self.brightness_type == 2):
			brightness_param_names = ['pb_d','pb_n']
			try:
				brightness_params = [self.pb_d,self.pb_n]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()
		if (self.brightness_type == 3):
			brightness_param_names = ['T_s','l1','l2','T_p_d','T_p_n']
			try:
				brightness_params = [self.T_s,self.l1,self.l2,self.T_p_d,self.T_p_n]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()
		if (self.brightness_type == 4):
			brightness_param_names = ['T_s','l1','l2','xi','T_n','delta_T']
			try:
				brightness_params = [self.T_s,self.l1,self.l2,self.xi,self.T_n,self.delta_T]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()
		if (self.brightness_type == 5):
			brightness_param_names = ['orders','sph','la_o','lo_o']
			try:
				brightness_params = [self.orders,self.la_o,self.lo_o] + self.sph
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()
			total_modes = (self.orders * (self.orders +1))/2.0
			if len(self.sph) != total_modes:
				print('You have not specified the correct number of mode coefficients!')
				print('You gave '+str(int(len(self.sph)))+', there should be '+str(int(total_modes)))
				quit()
		if (self.brightness_type == 6):
			brightness_param_names = ['T_s','l1','l2','insol','albedo','redist']
			try:
				brightness_params = [self.T_s, self.l1, self.l2, self.insol, self.albedo, self.redist]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()

		if (self.brightness_type == 7):
			brightness_param_names = ['la0','lo0','p_b','spot_b','size']
			try:
				brightness_params = [self.la0, self.lo0, self.p_b, self.spot_b, self.size]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()

		if (self.brightness_type == 8):
			brightness_param_names = ['T_s','l1','l2','la0','lo0','p_T','spot_T','size']
			try:
				brightness_params = [self.T_s, self.l1, self.l2, self.la0, self.lo0, self.p_T, self.spot_T, self.size]
			except:
				print('Brightness parameters incorrectly assigned')
				print('should be',brightness_param_names)
				quit()

		if any(b == None for b in brightness_params):
			print('Brightness parameters incorrectly assigned')
			print('should be',brightness_param_names)
			quit()
		return brightness_params

	def calc_phase(self,t):
		self.phase = _web.calc_phase(t,self.t0,self.per)

	def calc_substellar(self,t):
		self.calc_phase(t)
		coords = sp.separation_of_centers(t,self)
		substellar = _web.calc_substellar(self.phase,np.array(coords))
		self.lambda0 = substellar[0]
		self.phi0 = substellar[1]

	def square_plot(self,ax=False,min_temp=False,max_temp=False,temp_map=False,min_bright=0.2,show_cax=True,mycmap=plt.cm.inferno,theme='black',show_axes=False,nla=100,nlo=100):
		return splt.square_plot(self,ax=False,min_temp=False,max_temp=False,temp_map=False,min_bright=0.2,scale_planet=1.0,planet_cen=[0.0,0.0],mycmap=plt.get_cmap('inferno'),show_cax=True,theme='black',show_axes=False,nla=nla,nlo=nlo)

	def plot_system(self,t,ax=False,min_temp=False,max_temp=False,temp_map=False,min_bright=0.2,use_phase=False,show_cax=True,mycmap=plt.cm.inferno,theme='black',show_axes=False):
		if use_phase == True:
			t = self.t0 + self.per*t
		return splt.plot_system(self,t,ax=ax,min_temp=min_temp,max_temp=max_temp,temp_map=temp_map,min_bright=min_bright,show_cax=show_cax,mycmap=mycmap,theme=theme,show_axes=show_axes)

	def plot_planet(self,t,ax=False,min_temp=False,max_temp=False,temp_map=False,min_bright=0.2,scale_planet=1.0,planet_cen=[0.0,0.0],use_phase=False,show_cax=True,mycmap=plt.cm.inferno,theme='black',show_axes=False):
		if use_phase == True:
			t = self.t0 + self.per*t
		return splt.plot_planet(self,t,ax=ax,min_temp=min_temp,max_temp=max_temp,temp_map=temp_map,min_bright=min_bright,scale_planet=scale_planet,planet_cen=planet_cen,show_cax=show_cax,mycmap=mycmap,theme=theme,show_axes=show_axes)

	def get_lims(self,t,temp_map=False,use_phase=False):
		if use_phase == True:
			t = self.t0 + self.per*t

		planet = sp.generate_planet(self,t)
		if temp_map == True:
			b_i = 17
		else:
			b_i = 16

		temps = planet[:,b_i]

		return [np.min(temps),np.max(temps)]


	def plot_quad(self,min_temp=False,max_temp=False,temp_map=False,min_bright=0.2,scale_planet=1.0,planet_cen=[0.0,0.0],use_phase=False,show_cax=True,mycmap=plt.cm.inferno,theme='black'):

		if theme == 'black':
			bg = 'black'
			tc = ("#04d9ff")
		else:
			bg = 'white'
			tc = 'black'

		fig, axs = plt.subplots(2,2,figsize=(8/0.865,8),facecolor=bg)

		# need a "get max" or "get min" function or something similar so the scale can be set properly

		if max_temp == False:
			blims1 = self.get_lims(0,temp_map=temp_map,use_phase=True)
			blims2 = self.get_lims(0.25,temp_map=temp_map,use_phase=True)
			blims3 = self.get_lims(0.5,temp_map=temp_map,use_phase=True)
			blims4 = self.get_lims(0.75,temp_map=temp_map,use_phase=True)

			min_temp = np.min(np.array([blims1,blims2,blims3,blims4]))
			max_temp = np.max(np.array([blims1,blims2,blims3,blims4]))

		dp = ((max_temp-min_temp)*min_bright)

		self.plot_planet(0,use_phase=True,ax=axs[0,0],show_cax=False,min_temp=min_temp,max_temp=max_temp,temp_map=temp_map,mycmap=mycmap,theme=theme,min_bright=min_bright)
		self.plot_planet(0.25,use_phase=True,ax=axs[0,1],show_cax=False,min_temp=min_temp,max_temp=max_temp,temp_map=temp_map,mycmap=mycmap,theme=theme,min_bright=min_bright)
		self.plot_planet(0.5,use_phase=True,ax=axs[1,0],show_cax=False,min_temp=min_temp,max_temp=max_temp,temp_map=temp_map,mycmap=mycmap,theme=theme,min_bright=min_bright)
		self.plot_planet(0.75,use_phase=True,ax=axs[1,1],show_cax=False,min_temp=min_temp,max_temp=max_temp,temp_map=temp_map,mycmap=mycmap,theme=theme,min_bright=min_bright)

#		divider = make_axes_locatable(fig)

#		zero_temp = min_temp - dp
		zero_temp = min_temp

		if temp_map == True:
			data = [np.linspace(zero_temp,max_temp,1000)]*2
		else:
			data = [np.linspace(zero_temp/max_temp,max_temp/max_temp,1000)]*2
		fake, fake_ax = plt.subplots()
		mycax = fake_ax.imshow(data, interpolation='none', cmap=mycmap)
		plt.close(fake)

		fig.subplots_adjust(right=0.80)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

		if show_cax == True:
#			cax = divider.append_axes("right", size="20%", pad=0.05)
		#	cbar = plt.colorbar(mycax, cax=cax,ticks=[1100,1300,1500,1700,1900])
			cbar = plt.colorbar(mycax, cax=cbar_ax)
			cbar.ax.tick_params(colors=tc)

			if temp_map == True:
				cbar.set_label('T (K)',color=tc)  # horizontal colorbar
			else:
				cbar.set_label('Relative brightness',color=tc)  # horizontal colorbar

		fig.subplots_adjust(wspace=0, hspace=0)

		return fig

	def plot_uncertainty(self,fs,min_temp=False,max_temp=False,temp_map=True,min_bright=0.2,scale_planet=1.0,planet_cen=[0.0,0.0],use_phase=False,show_cax=True,mycmap=plt.cm.viridis_r,theme='black'):

		if theme == 'black':
			bg = 'black'
			tc = ("#04d9ff")
		else:
			bg = 'white'
			tc = 'black'

		fig, axs = plt.subplots(2,2,figsize=(8/0.865,8),facecolor=bg)

		# need a "get max" or "get min" function or something similar so the scale can be set properly

		fs = np.array(fs)

		min_temp = np.min(fs)
		max_temp = np.max(fs)

		dp = ((max_temp-min_temp)*min_bright)

		sp.plot_dist(self,fs[0],ax=axs[0,0],show_cax=False,min_temp=min_temp,max_temp=max_temp,temp_map=temp_map,mycmap=mycmap,theme=theme,min_bright=min_bright)
		sp.plot_dist(self,fs[1],ax=axs[0,1],show_cax=False,min_temp=min_temp,max_temp=max_temp,temp_map=temp_map,mycmap=mycmap,theme=theme,min_bright=min_bright)
		sp.plot_dist(self,fs[2],ax=axs[1,0],show_cax=False,min_temp=min_temp,max_temp=max_temp,temp_map=temp_map,mycmap=mycmap,theme=theme,min_bright=min_bright)
		sp.plot_dist(self,fs[3],ax=axs[1,1],show_cax=False,min_temp=min_temp,max_temp=max_temp,temp_map=temp_map,mycmap=mycmap,theme=theme,min_bright=min_bright)

#		divider = make_axes_locatable(fig)

#		zero_temp = min_temp - dp
		zero_temp = min_temp

		data = [np.linspace(zero_temp,max_temp,1000)]*2
		fake, fake_ax = plt.subplots()
		mycax = fake_ax.imshow(data, interpolation='none', cmap=mycmap)
		plt.close(fake)

		fig.subplots_adjust(right=0.80)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

		if show_cax == True:
#			cax = divider.append_axes("right", size="20%", pad=0.05)
		#	cbar = plt.colorbar(mycax, cax=cax,ticks=[1100,1300,1500,1700,1900])
			cbar = plt.colorbar(mycax, cax=cbar_ax)
			cbar.ax.tick_params(colors=tc)

			cbar.ax.invert_yaxis()

			if temp_map == True:
				cbar.set_label(r'Temperature precision (\%)',color=tc)  # horizontal colorbar
			else:
				cbar.set_label(r'Flux precision (\%)',color=tc)  # horizontal colorbar

		fig.subplots_adjust(wspace=0, hspace=0)

		return fig

	def lightcurve(self,t,use_phase=False,stellar_grid=False):

		brightness_params = self.format_bright_params()

		if self.thermal == True:
			if stellar_grid == False:
				star_grid = sp.stellar_grid.gen_grid(self.l1,self.l2)
				teffs = star_grid[0]
				totals = star_grid[1]
			else:
				teffs = stellar_grid[0]
				totals = stellar_grid[1]
		else:
			teffs = []
			totals = []

		if use_phase == True:
			t = self.t0 + self.per*t

		out = _web.lightcurve(self.n_layers,t,self.t0,self.per,self.a_abs,self.inc,self.ecc,self.w,self.a,self.rp,self.p_u1,self.p_u2,self.brightness_type,brightness_params,teffs,totals,len(totals))
		return np.array(out)
