import numpy as np
from scipy.stats import moyal
from astropy.modeling.models import Gaussian1D



############################################################
# Base noise profile class.
############################################################

class ProfileAttribute(object):
    def __init__(self, value=None):
        self._attr = value
    def __get__(self, obj, objtype = None):
        if self._attr is None: print("The attribute you're trying to read is 'None'. Please initialise the value before using it.")
        return self._attr
    def __set__(self, obj, value):
        self._attr = value

class NoiseProfile:
        def __init__(self, name, **kwargs):
            self.name = name # string containing the name of the profile.

            if kwargs:
                for key, value in kwargs.items():
                    setattr(self, key, value)
                    
        # returns the probability density function.
        def PDF(self, X):
            raise NotImplementedError

        # Returns the limit of each value
        def limits(self, X):
            raise NotImplementedError

        # returns the default initial values.
        def init_values(self, X):
            raise NotImplementedError

############################################################
# Childrens of base noise profile class.
############################################################

class Landau(NoiseProfile):
    # Parameters of the profile.
    loc = ProfileAttribute()
    scale = ProfileAttribute()

    def __init__(self, **kwargs):
        super().__init__('Landau', **kwargs)

    def pdf(self, X, loc, scale):
        self.loc = loc
        self.scale = scale
        return moyal.pdf(X, loc, scale)
    
    def limits(self, X):
        return ((np.min(X), np.min(X)), (np.max(X), np.max(X)))
    
    def init_values(self, X):
        return (np.mean(X), np.std(X))

class Gauss(NoiseProfile):
    # Parameters of the profile.
    mean = ProfileAttribute()
    stdev = ProfileAttribute()

    def __init__(self, **kwargs):
        super().__init__('Gauss', **kwargs)

    def pdf(self, X, mean, stdev):
        self.mean = mean
        self.stdev = stdev
        gaussian = Gaussian1D(1/(stdev*np.sqrt(2*np.pi)), mean, stdev)
        return gaussian(X)
    
    def limits(self, X):
        return ((np.min(X), np.min(X)), (np.max(X), np.max(X)))
    
    def init_values(self, X):
        return (np.mean(X), np.std(X))