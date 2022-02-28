# read the relevant domain into memory
import numpy as np
import astropy.wcs as pywcs
from astropy.io import fits
import math
from enum import Enum

############################################################
# custom exceptions
############################################################

class TooFewDomainsError(Exception):
    pass

class DomainMissingError(Exception):
    pass

############################################################
# helper classes 
############################################################

class DomainData:
    """
    Class that keeps track of the specific domain data. Users shouold not be accessing this class directly
    """

    def __init__(self, axes = None, indices = None, res = None, data = None, infile = None):
        """Initialise the domain data.

        Parameters
        ----------
        axes : array_like
            axes is an array of strings ["freq", "Dec", "RA"] specifying the axis order in the data and indices
        indices : array_like
            indices is a list of arrays of the index values for the 3 axes the order of the axes is given by self.axes
        res : array_like
            contains the resolution of each axis
        data: array_like
            data contains the 3D datacube
        """

        self.axes = axes
        if indices != None:
            self.indices1 = indices[0]
            self.indices2 = indices[1]
            self.indices3 = indices[2]
        self.res = res
        self.data = data
        if infile != None:
            self._read_from_file(infile)

        assert self.data.shape == (len(self.indices1), len(self.indices2), len(self.indices3))
        self._set_order()

    def copy(self):
        other = DomainData(axes = self.axes, indices = [self.indices1, self.indices2, self.indices3], res= self.res, data = np.copy(self.data))
        return other


    def _set_order(self):
        """
        Set the class parameters depending on the axis order.
        Allows same functions to read freq, Dec, RA domains and RA, Dec, freq domains
        """
        for i, name in enumerate(self.axes):
            if name == "freq" or name == "f":
                self.freq = i
            if name == "RA" or name == "ra":
                self.RA = i
            if name == "Dec" or name == "dec":
                self.Dec = i

    def _read_from_file(self, infile):
        """
        Read domain data from file
        """
        with open(infile, 'rb') as f:
            self.axes = np.load(f)
            self.indices1 = np.load(f)
            self.indices2 = np.load(f)
            self.indices3 = np.load(f)
            self.res = np.load(f)
            self.data = np.load(f)

    def write(self, outfile):
        """
        Write domain data to file
        """
        with open(outfile, 'wb') as f:
            np.save(f, self.axes )
            np.save(f, self.indices1 )
            np.save(f, self.indices2 )
            np.save(f, self.indices3 )
            np.save(f, self.res )
            np.save(f, self.data )
    def contains(self, x, y, z, border):
        in_x = x >= self.ra_indices[border]  and x <= self.ra_indices[-border-1]
        in_x = in_x or (x <= self.ra_indices[border]  and x >= self.ra_indices[-border-1])
        if not in_x: return False
        in_y = y >= self.dec_indices[border] and y <= self.dec_indices[-border-1]
        in_y = in_y or (y <= self.dec_indices[border] and y >= self.dec_indices[-border-1])
        if not in_y: return False
        in_z = z >= self.f_indices[0] and z <= self.f_indices[-1]
        if not in_z: return False
        return True

    def transpose(self):
        """
        Reverse the axis order
        """
        start_shape = self.data.shape
        self.axes = self.axes[::-1]
        self.indices1, self.indices3 = self.indices3, self.indices1
        self.data = np.transpose(self.data)
        assert self.data.shape == (len(self.indices1), len(self.indices2), len(self.indices3))
        self._set_order()

    @property
    def f_indices(self):
        if self.freq == 0:
            return self.indices1
        else:
            return self.indices3

    @property
    def ra_indices(self):
        if self.RA == 0:
            return self.indices1
        else:
            return self.indices3

    @property
    def dec_indices(self):
        return self.indices2
'''
class MultiDomainData:
    def __init__(self,infiles):
        self.domains = []
        for f in infiles:
            self.domains.append(DomainData(infile=f))


    @property
    def f_indices(self):
        try:
            return self.freq
        except:
            l = [d.f_indices for d in domains]
            self.freq = list(dict.fromkeys(l))
            return self.freq
        

    @property
    def ra_indices(self):
        try:
            return self.RA
        except:
            l = [d.ra_indices for d in domains]
            self.RA = list(dict.fromkeys(l))
            return self.RA

    @property
    def dec_indices(self):
        try:
            return self.Dec
        except:
            l = [d.dec_indices for d in domains]
            self.Dec = list(dict.fromkeys(l))
            return self.Dec
'''

############################################################
# Domain reading superclass
# Controls domain boundary definitions and reading/writing domain data
############################################################
class DomainReader:
    def __init__(self, N, index, filepath_datacube = None, filepath_continuumimage = None, filepath_header = None, border = 5, forceSquare = False):
        """Initialise the domain data.

        Parameters
        ----------
        N : int
            total number of domains
        index : int
            rank of this domain among all domains, 0 thru N-1
        filepath_datacube : string
            filepath of the HI cube (.npy or .fits)
        filepath_continuumimage : string
            filepath of the continuum cube (.fits)
        filepath_header : string
            filepath of the FITS file that will be used to read header information
        border: int
            size of overlapping domain borders in voxels.
        forceSquare: bool
            force all domains to be square
        """
        self.N = N
        self.index = index
        assert index >= 0 and index < N

        self.filepath_datacube       = filepath_datacube
        self.filepath_continuumimage = filepath_continuumimage
        self.filepath_header         = filepath_header
        self.forceSquare = forceSquare
        self.border = border

        # maximum domain size to read
        self.xmin, self.xmax = 0, -1
        self.ymin, self.ymax = 0, -1
        self.zmin, self.zmax = 0, -1

        # domaindata object
        self.HI_cube = None
        self.cont_cube = None

    def is_data_loaded(self):
        return self.HI_cube != None

    def define_domains(self, full_cube_x = None, full_cube_y = None):

        # read in max x and y
        if full_cube_x == None or full_cube_y == None:
            with fits.open(self.filepath_header, memmap = True) as hdul:
                full_cube_x = hdul[0].header["NAXIS1"]
                full_cube_y = hdul[0].header["NAXIS2"]
        assert full_cube_x == full_cube_y
        
        # get square number of N = nxn domains
        square = int(math.sqrt(self.N))
        if self.N != square*square:
            print("Domain decomposition is only supported for a square number of domains ")
            raise NotImplementedError

        print("Subdividing cube of {0} x {1} into {2} domains".format(full_cube_x, full_cube_y, self.N ))

        #
        if self.forceSquare:
            domain_size = int( (full_cube_x-self.border*2)/square)
            last = domain_size*square + self.border*2
            xspan = np.arange(last)
            yspan = xspan
            xspan = np.array_split(xspan[self.border*2:last], square)
            yspan = np.array_split(yspan[self.border*2:last], square)
            print("Forcing Square Domains of {0}x{0}".format(domain_size+ self.border*2))
        else:
            xspan, yspan = np.arange(full_cube_x), np.arange(full_cube_y)
            xspan = np.array_split(xspan[self.border*2:], square)
            yspan = np.array_split(yspan[self.border*2:], square)

        x = self.index%square
        y = int(self.index/square)
        self.xmin = 0 if x == 0 else (xspan[x][0] - 2*self.border)
        self.ymin = 0 if y == 0 else (yspan[y][0] - 2*self.border)
        self.xmax  = xspan[x][-1]+1
        self.ymax  = yspan[y][-1]+1

        if self.forceSquare:
            assert self.xmax - self.xmin == self.ymax-self.ymin

    def __str__(self):
        s1 =  "Domain {0} of {1}".format(self.index, self.N )
        s2 = "x: {0} - {1}, y: {2} - {3}".format(self.xmin, self.xmax, self.ymin, self.ymax)
        s3 = "points within border x: {0} - {1}, y: {2} - {3}".format(self.xmin+self.border, self.xmax-self.border, self.ymin+self.border, self.ymax-self.border)
        return "{0}, {1}\n\t{2}".format(s1,s2,s3)

    #### placeholder readers ##############
    def _load_cube_domain(self):
        raise NotImplementedError

    def _load_continuum_domain(self):
        raise NotImplementedError

    ########################################

    def transpose(self):
        if self.HI_cube != None: self.HI_cube.transpose()
        if self.cont_cube != None: self.cont_cube.transpose()

    def in_domain(self, ra = None, dec = None, x= None, y = None, source = None):

        if source != None:
            return self.in_domain(x = source.x(), y = source.y())
        if y == None and x==None:
            x,y,__ = self.sky_to_pixels(ra, dec, 0 )
        if self.HI_cube == None:
            in_x = x - self.border >= self.xmin and x + self.border <= self.xmax
            in_y = y - self.border >= self.ymin and y + self.border <= self.ymax 
            return in_x and in_y
        return self.HI_cube.contains(x,y, self.border*3, self.border)

    # get cutout from domain
    def get_cube(self, istart, istop, jstart, jstop, kstart = 0, kstop = -1):
        jstart -= self.ymin
        jstop  -= self.ymin
        istart -= self.xmin
        istop  -= self.xmin
        istart = max(0, istart)
        jstart = max(0, jstart)
        kstart = max(0, kstart)
        kstop = -1 if kstop == -1 else min(kstop, self.HI_cube.f_indices[-1]+1)
        jstop = -1 if jstop == -1 else min(jstop, self.HI_cube.dec_indices[-1]+1)
        istop = -1 if istop == -1 else min(istop, self.HI_cube.ra_indices[-1]+1)



        if self.HI_cube.freq == 0:
            return self.HI_cube.data[kstart:kstop, jstart:jstop, istart:istop ]
        else:
            return self.HI_cube.data[istart:istop, jstart:jstop, kstart:kstop ]

    def get_cutout(self, center, cel_w = 15, freq_w = 100):
        i,j,k = center
        xstart, xstop, ystart, ystop, zstart, zstop = [int(x) for x in [i-cel_w, i+cel_w, j-cel_w, j+cel_w, k-freq_w, k+freq_w]]
        return self.safe_get_cube(xstart, xstop, ystart, ystop, zstart, zstop)

    def get_continuum_cutout(self, center, cel_w = 15):
        i,j,k = center
        xstart, xstop, ystart, ystop, = [int(x) for x in [i-cel_w - self.xmin, i+cel_w - self.xmin, j-cel_w - self.ymin, j+cel_w - self.ymin]]
        if xstart < 0 or xstart < 0: return np.array([])
        if xstop > self.HI_cube.dec_indices[-1] +1: return np.array([])
        if ystop > self.HI_cube.ra_indices[-1]  +1: return np.array([])
        return  self.cont_cube.data[:, ystart:ystop, xstart:xstop]

    # get cutout form domain. if indices go over domain borders, return None
    def safe_get_cube(self, istart, istop, jstart, jstop, kstart = 0, kstop = -1):
        jstart -= self.ymin
        jstop  -= self.ymin
        istart -= self.xmin
        istop  -= self.xmin
        if istart < 0 or jstart < 0 or kstart < 0: return np.array([])
        if kstop > self.HI_cube.f_indices[-1]   +1: return np.array([])
        if jstop > self.HI_cube.dec_indices[-1] +1: return np.array([])
        if istop > self.HI_cube.ra_indices[-1]  +1: return np.array([])

        if self.HI_cube.freq == 0:
            return self.HI_cube.data[kstart:kstop, jstart:jstop, istart:istop ]
        else:
            return self.HI_cube.data[istart:istop, jstart:jstop, kstart:kstop ]

    def get_continuum_cube(self, istart, istop, jstart, jstop, kstart = 0, kstop = -1):
        jstart -= self.ymin
        jstop  -= self.ymin
        istart -= self.xmin
        istop  -= self.xmin
        istart = max(0, istart)
        jstart = max(0, jstart)
        kstart = max(0, kstart)
        kstop = -1 if kstop == -1 else min(kstop, self.cont_cube.f_indices[-1]+1)
        jstop = -1 if jstop == -1 else min(jstop, self.cont_cube.dec_indices[-1]+1)
        istop = -1 if istop == -1 else min(istop, self.cont_cube.ra_indices[-1]+1)

        if self.cont_cube.freq == 0:
            return self.cont_cube.data[kstart:kstop, jstart:jstop, istart:istop ]
        else:
            return self.cont_cube.data[istart:istop, jstart:jstop, kstart:kstop ]

    def get_freq_line(self, i, j):
        line = self.get_cube(i, i+1, j, j+1, 0, -1)
        line.shape = (max(line.shape))
        return line

    def get_pencil(self, i, j, half_width):
        return self.get_cube(i - half_width, i+half_width, j- half_width, j+half_width, 0, -1)

    # get voxel cube corresponding to source
    def get_bounding_cube(self, indices = None, source = None):
        if indices == None:
            return self.get_cube(indices = source.cube_indices(self.dy, self.dz))
        (i_low, i_high, j_low, j_high, k_low, k_high) = indices
        i_high = i_high - self.xmin
        i_low = max(0,i_low- self.xmin)
        j_high = j_high - self.ymin
        j_low = max(0,j_low- self.ymin)
        if j_low == j_high: j_high+= 1
        if i_low == i_high: i_high+= 1
        if k_low == k_high: k_high+= 1
        return self.HI_cube.data[k_low:k_high,
                               j_low: j_high,
                               i_low: i_high]

    def pixels_to_sky(self, i, j, k):
        coords = self.w.wcs_pix2world([[ i, j, k]], 0)[0]
        return coords

    def sky_to_pixels(self, ra, dec, freq):
        coords = self.w.wcs_world2pix([[ ra, dec, freq]], 0)[0]
        return coords

    def read(self):
        if self.filepath_datacube != None: self._load_cube_domain()
        if self.filepath_continuumimage != None: self._load_continuum_domain()

    def _read_header(self):
        with fits.open(self.filepath_header, memmap = True) as hdul:
            self.w = pywcs.WCS(hdul[0].header)
            if self.xmax == -1:
                self.xmax = hdul[0].header["NAXIS1"]
            if self.ymax == -1:
                self.ymax = hdul[0].header["NAXIS2"]
            if self.zmax == -1:
                self.zmax = hdul[0].header["NAXIS3"]

    @property
    def ax1(self):
        try:
            return self._ax1
        except:
            axis = np.arange(self.xmin, self.xmax)
            for i, x in enumerate(axis):
                coords = self.w.wcs_pix2world([[ x,0, 0]], 0)[0]
                axis[i] = coords[0]
            self._ax1=axis
            return self._ax1

    @property
    def ax3(self):
        try:
            return self._ax3
        except:
            axis = np.arange(self.zmin, self.zmax)
            for i, z in enumerate(axis):
                coords = self.w.wcs_pix2world([[ 0,0, z]], 0)[0]
                axis[i] = coords[2]
            self._ax3=axis
            return self._ax3
    @property
    def dx(self):
        return self.HI_cube.res[0]

    @property
    def dy(self):
        return self.HI_cube.res[1]

    @property
    def dz(self):
        return self.HI_cube.res[2]

class AstropyDomainReader(DomainReader):
    def __init__(self,   N, index, filepath_datacube = None, filepath_continuumimage = None, filepath_header = None, border = 5, forceSquare = False):
        super().__init__(N, index, filepath_datacube, filepath_continuumimage, filepath_header, border, forceSquare)
        assert filepath_datacube[-5:] == ".fits"
        print("reading FITS file", self.filepath_datacube)
        self.filepath_header = self.filepath_datacube

    def _load_cube_domain(self):
        with fits.open(self.filepath_datacube, memmap = True) as hdul:
            self.w = pywcs.WCS(hdul[0].header)
            if self.xmax == -1:
                self.xmax = hdul[0].header["NAXIS1"]
            if self.ymax == -1:
                self.ymax = hdul[0].header["NAXIS2"]
            if self.zmax == -1:
                self.zmax = hdul[0].header["NAXIS3"]

            axes = np.array(["freq", "Dec", "RA"])
            i = [np.arange(self.zmin, self.zmax),
                 np.arange(self.ymin,self.ymax),
                 np.arange(self.xmin,self.xmax)]
            r = np.array([hdul[0].header["CDELT1"], hdul[0].header["CDELT2"],hdul[0].header["CDELT3"]])
            d = hdul[0].data[:,self.ymin:self.ymax,self.xmin:self.xmax]
            self.HI_cube = DomainData(axes, indices = i, res = r, data = d)

    def _load_continuum_domain(self):
        with fits.open(self.filepath_continuumimage, memmap = True) as hdul:
            axes = np.array(["freq", "Dec", "RA"])
            i = [np.arange(hdul[0].header["NAXIS3"]),
                 np.arange(self.ymin,self.ymax),
                 np.arange(self.xmin,self.xmax)]
            r = np.array([hdul[0].header["CDELT1"], hdul[0].header["CDELT2"],hdul[0].header["CDELT3"]])
            d = hdul[0].data[:,self.ymin:self.ymax,self.xmin:self.xmax]
            self.cont_cube = DomainData(axes, indices = i, res = r, data = d)

class BinaryDomainReader(DomainReader):
    def __init__(self,   N, index, filepath_datacube = None, filepath_continuumimage = None, filepath_header = None, nfiles = 1, border = 5, forceSquare = False):
        super().__init__(N, index, filepath_datacube, filepath_continuumimage, filepath_header, border, forceSquare)

        self.nfiles = nfiles
        assert filepath_datacube[-4:] == ".npy"
        import glob
        all_files = glob.glob(filepath_datacube)
        #if self.N > len(all_files):
        #    raise TooFewDomainsError("Some domains are missing! Expected {0} domains, found {1}".format(self.N,len(all_files) ))
        # find corresponding domain file
        fnames = [] if len(all_files) != 1 else [all_files[0]]

        self.xmin = 99999
        self.xmax = -1
        self.ymin = 99999
        self.ymax = -1

        for f in all_files:
            for t in f.split('_'):
                if 'task' in t:
                    i = int(t.split('-')[-1])
                    if nfiles == 1 and i == index: fnames = [f]
                    elif nfiles > 1 and i-index >=0 and i-index < nfiles:
                        fnames.append(f)
                    else: break
                elif 'x-' in t:
                    self.xmin, self.xmax = [int(i) for i in t.split('-')[1:]]
                elif 'y-' in t:
                    self.ymin, self.ymax = [int(i) for i in t.split('-')[1:]]
                elif 'border' in t:
                    self.border = int(t.split('.')[0][6:])

            if len(fnames) == nfiles: break

        assert len(fnames) == nfiles

        if self.nfiles == 1:
            self.filepath_datacube = fnames[0]
            print("reading binary NPY file", self.filepath_datacube)
        else:
            self.filepath_datacube = fnames
            print("reading binary NPY files:")
            for f in fnames: print(f)

    def _load_cube_domain(self):
        self._read_header()
        if self.nfiles == 1:
            self.HI_cube = DomainData(infile = self.filepath_datacube)
        else:
            self.HI_cube = MultiDomainData(infiles = self.filepath_datacube)

    def _load_continuum_domain(self):
        with fits.open(self.filepath_continuumimage, memmap = True) as hdul:
            axes = np.array(["freq", "Dec", "RA"])
            i = [np.arange(hdul[0].header["NAXIS3"]),
                 np.arange(self.ymin,self.ymax),
                 np.arange(self.xmin,self.xmax)]
            r = np.array([hdul[0].header["CDELT1"], hdul[0].header["CDELT2"],hdul[0].header["CDELT3"]])
            d = hdul[0].data[:,self.ymin:self.ymax,self.xmin:self.xmax]
            self.cont_cube = DomainData(axes, indices = i, res = r, data = d)



