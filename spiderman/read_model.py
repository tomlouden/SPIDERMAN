import spiderman as sp
import numpy as np

def load_grid(fname,layer):
	longitude, latitude, T = open_PT_profile(fname,layer)
	grid = format_grid(longitude,latitude,T)
	return grid

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

def format_grid(longitude,latitude,temp):
	# puts the grid into a format that spiderman can use
	# including wrapping/reflecting the boundaries to allow smooth and continuous interpolation

	longitude = longitude.flatten()
	latitude = latitude.flatten()
	temp = temp.flatten()

	LO, LA = np.unique(longitude),np.unique(latitude)
	T = np.reshape(temp,(len(LO),len(LA)))
	T = np.vstack((T[-5],T[-4],T[-3],T[-2],T[-1],T,T[0],T[1],T[2],T[3],T[4]))
	LO = np.hstack((LO[-5:]-360,LO,LO[0:5]+360))
	T = np.vstack((T[:,4],T[:,3],T[:,2],T[:,1],T[:,0],T.T,T[:,-1],T[:,-2],T[:,-3],T[:,-4],T[:,-5])).T
	LA = np.hstack((-180 - LA[0:5][::-1],LA,180-LA[-5:][::-1]))

	grid = np.array([LO,LA,T])
	return grid