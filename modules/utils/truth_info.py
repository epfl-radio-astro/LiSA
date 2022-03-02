import numpy as np
from matplotlib.patches import Ellipse
import math



class TruthSource:
    var_titles = [ "ID", "RA", "Dec",
               "HI size", "line flux integral",
               "central freq", "pos_a", "inc_a", "w20"
              ]
    var_units = ["","[deg]","[deg]","[arcsec]","[Jy Hz]","[Hz]","[deg]","[deg]","[km/s]"]

    def __init__(self, data,  w = None):
        self.data = data
        self.w = w
    def ID(self):
        return int(self.data[0])
    def RA(self):
        return self.data[1]
    def Dec(self):
        return self.data[2]
    def hi_size(self):
        return self.data[3]
    def hi_size_deg(self):
        return self.hi_size()*0.000277778
    def hi_size_pix(self, d):
        return self.hi_size_deg()/d
    def line_flux_integral(self):
        return self.data[4]
    def central_freq(self):
        return self.data[5]
    def pos_a(self):
        return self.data[6]
    def inc_a(self):
        return self.data[7]
    def w20(self):
        return self.data[8]

    def w20_freq(self):
        #delta(freq) = (observed frequency)*delta(velocity)/c
        c = 299792.458 #km/s
        return self.data[5] * self.data[8] / c

    def w20_pix(self, d):
        return self.w20_freq()/d
    def setcoords(self,w):
        self.w = w
        self._setcoords()

    def _setcoords(self):
        #print(self.RA(),self.Dec(), self.central_freq())
        coords = self.w.wcs_world2pix([[ self.RA(),self.Dec(), self.central_freq()]], 0)[0]
        self._x, self._y, self._z = coords[0], coords[1], coords[2]
        #print (coords)
        #print (self.w.wcs_pix2world([coords], 0)[0])

    def x(self):
        try:
            return self._x
        except AttributeError:
            self._setcoords()
            return self._x

    def y(self):
        try:
            return self._y
        except AttributeError:
            self._setcoords()
            return self._y

    def z(self):
        try:
            return self._z
        except AttributeError:
            self._setcoords()
            return self._z

    def cube_indices(self, dy, dz):

        z_low  = int(self.z() -  self.w20_pix(dz)/2)
        z_high = int(self.z() +  self.w20_pix(dz)/2)
        d = self.hi_size_pix(dy)
        x_low  = int(self.x() - d/2)
        x_high = int(self.x() + d/2)
        y_low  = int(self.y() - d/2)
        y_high = int(self.y() + d/2)

        return (x_low, x_high, y_low, y_high, z_low, z_high)

    def shape(self):
        return Ellipse((self.Dec(),self.RA()),
                     width=self.hi_size_deg()*math.sin(self.inc_a()*math.pi/180),
                     height=self.hi_size_deg(),
                     angle = self.pos_a(),
                     edgecolor='red', facecolor='none',
                     linewidth=1)
    def shape_pix(self,d):
        return Ellipse((self.x(),self.y()),
                     width=self.hi_size_pix(d),
                     height=self.hi_size_pix(d)*math.sin(self.inc_a()*math.pi/180),
                     angle = self.pos_a(),
                     edgecolor='red', facecolor='none',
                     linewidth=1)

    def __str__(self):
        attributes = [ "{0}: {1:.2f} {2}".format(TruthSource.var_titles[i],self.data[i],TruthSource.var_units[i]) for i in range(9)]
        attribute_str = '\n'.join(attributes)
        attribute_str += "\nPosition: (x,y,z) = ({}, {}, {})".format(self.x(), self.y(), self.z())
        return attribute_str

    @staticmethod
    def catalog_to_dataframe(filename):
        import pandas as pd
        data = parse_truth_catalog(filename)
        df = pd.DataFrame(data=data, columns=TruthSource.var_titles)
        return df

    @staticmethod
    def catalog_to_array(filename, filter_func = None):
        data = None
        if '*' in filename:
            import glob
            files = glob.glob(filename)
            for i, f in enumerate(files):
                if i == 0:
                    data = TruthSource.catalog_to_array(f, filter_func)
                else:
                    data = np.row_stack( [data, TruthSource.catalog_to_array(f, filter_func)])
        else:
            with open(filename) as file:
                for i, line in enumerate(file):
                    # (ID, RA, dec, hi_size, line_flux_integral, central_freq, pos_a, inc_a, w20)
                    if i == 0: continue
                    entry = np.asarray([float(t) for t in line.split()])
                    if filter_func != None:
                        if not filter_func(entry): continue

                    try:
                        data = np.row_stack( [data, entry])
                    except:
                        data = entry
        return data

    @staticmethod
    def catalog_to_sources(filename, w = None):
        truth_data = TruthSource.catalog_to_array(filename)
        sources = [TruthSource(s, w) for s in truth_data]
        return sources

    @staticmethod
    def catalog_to_sources_in_domain(filename, domain):
        in_domain =  lambda x: domain.in_domain(ra = x[1], dec = x[2])
        truth_data = TruthSource.catalog_to_array(filename, in_domain)
        sources = [TruthSource(s, w = domain.w) for s in truth_data]
        return sources
