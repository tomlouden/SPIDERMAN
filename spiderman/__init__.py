from os.path import expanduser,join
home = expanduser("~")
param_file = join(home,'.spidermanrc')

class RcParams(object):
	def __init__(self):
		self.dict = {}
		try:
			print 'looking for spidermanrc file at '+param_file
			for line in open(param_file):
				key = line.split(':')[0].replace(' ','')
				cval = line.split(':')[1].replace(' ','')
				self.dict[key] = cval
			RcParams.read = True
		except:
			RcParams.read = False
			print 'no spidermanrc file detected'
	def __getitem__(self, key):
	    return self.dict[key]
rcParams = RcParams()

from spiderman.params import *
from spiderman.web import *
from spiderman.plot import *
from spiderman.test import *
from spiderman.stellar_grid import *

__all__ = ["web","params","_web","plot","test","stellar_grid"]