import spiderman as sp
import numpy as np

def open_PT_profile(fname,layer):
	layers = []
	for line in open(fname):
		try:
			layers += [int(line.strip('\n').split(',')[0])]
		except:
			''
	layers = np.array(layers)
	nlayers = np.max(layers)

#	print(nlayers)
	longitude = []
	latitude = []
	T = []


	for line in open(fname):
		try:
			this_layer = int(line.strip('\n').split(',')[0])
			if this_layer == layer:
				this_grid = line.strip('\n').split(',')
				longitude += [float(this_grid[1])]
				latitude += [float(this_grid[2])]
				T += [float(this_grid[4])]
		except:
			''
	T = np.array(T)
	longitude = np.array(longitude)
	latitude = np.array(latitude)
	return(longitude, latitude, T)