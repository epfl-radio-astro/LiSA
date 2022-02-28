import numpy as np
import pysap
import pysparse


class StarletTransform(object):
    """
    Decomposition of an image using the Isotropic Undecimated Walevet Transform,
    also known as "starlet" or "B-spline", using the 'a trous' algorithm.

    Astronomical data (galaxies, stars, ...) are often very sparsely represented in the starlet basis.

    Based on Starck et al. : https://ui.adsabs.harvard.edu/abs/2007ITIP...16..297S/abstract
    """

    NOISE_TAB = np.array([ 0.8907963 ,  0.20066385,  0.08550751,  0.04121745,
                           0.02042497,  0.01018976,  0.00504662,  0.00368314])

    def __init__(self, num_procs=1, fast_inverse=True, show_pysap_plots=False):
        """
        Load pySAP package if found, and initialize the Starlet transform.

        :param num_procs: number of threads used for pySAP computations
        :param fast_inverse: if True, reconstruction is simply the sum of each scale (only for 1st generation starlet transform)
        """
        self._transf_class = pysap.load_transform('BsplineWaveletTransformATrousAlgorithm')
        self._fast_inverse = fast_inverse
        self._show_pysap_plots = show_pysap_plots
        self._num_procs = num_procs

    def reconstruct(self, coeffs, num_scales, num_pixels):
        """
        2D inverse starlet transform from starlet coefficients stored in coeffs

        :param coeffs: decomposition coefficients,
        ndarray with shape (num_scales, sqrt(num_pixels), sqrt(num_pixels))
        :param num_scales: number of decomposition scales
        :return: reconstructed signal as 2D array of shape (sqrt(num_pixels), sqrt(num_pixels))
        """
        return self._inverse_transform(coeffs, num_scales, num_pixels)

    def decompose(self, image, num_scales):
        """
        2D starlet transform from starlet coefficients stored in coeffs

        :param image: 2D image to be decomposed, ndarray with shape (sqrt(num_pixels), sqrt(num_pixels))
        :param num_scales: number of decomposition scales
        :return: reconstructed signal as 2D array of shape (num_scales, sqrt(num_pixels), sqrt(num_pixels))
        """
        return self._transform(image, num_scales)

    def energy_per_scale(self, num_scales):
        return self.NOISE_TAB[:num_scales]

    @property
    def num_precomputed(self):
        return len(self.NOISE_TAB)

    def _inverse_transform(self, coeffs, num_scales, num_pixels):
        """reconstructs image from starlet coefficients"""
        self._check_transform_pysap(num_scales, num_pixels)
        if self._fast_inverse:
            # for 1st gen starlet the reconstruction can be performed by summing all scales
            image = np.sum(coeffs, axis=0)
        else:
            coeffs = self._coeffs2pysap(coeffs)
            self._transf.analysis_data = coeffs
            result = self._transf.synthesis()
            if self._show_pysap_plots:
                result.show()
            image = result.data
        return image

    def _transform(self, image, num_scales):
        """decomposes an image into starlets coefficients"""
        self._check_transform_pysap(num_scales, image.size)
        self._transf.data = image
        self._transf.analysis()
        if self._show_pysap_plots:
            self._transf.show()
        coeffs = self._transf.analysis_data
        coeffs = self._pysap2coeffs(coeffs)
        return coeffs

    def _check_transform_pysap(self, num_scales, num_pixels):
        """if needed, update the loaded pySAP transform to correct number of scales"""
        if not hasattr(self, '_transf') or num_scales != self._num_scales or num_pixels != self._num_pixels:
            self._transf = self._transf_class(nb_scale=num_scales, verbose=False,
                                              nb_procs=self._num_procs)
            self._num_scales = num_scales
            self._num_pixels = num_pixels

    def _pysap2coeffs(self, coeffs):
        """convert pySAP decomposition coefficients to numpy array"""
        return np.asarray(coeffs)

    def _coeffs2pysap(self, coeffs):
        """convert coefficients stored in numpy array to list required by pySAP"""
        coeffs_list = []
        for i in range(coeffs.shape[0]):
            coeffs_list.append(coeffs[i, :, :])
        return coeffs_list


class Wavelet2D1DTransform(object):
    """Wavelet decomposition of a 3D data cube."""

    def __init__(self, transform_type=2):
        """Wrapper for pysparse's 2D-1D wavelet transform.

        Parameters
        ----------
        transform_type : int
            Type of wavelet transform to perform with `decompose`. See pysap
            documentation for all options. Default is 2, which takes a starlet
            transform in the 2D domain (undecimated) and a 7/9 transform
            in the 1D domain (decimated).

        """
        self.transform_type = transform_type

    def decompose(self, cube, num_scales_2d, num_scales_1d):
        """Forward 2D1D wavelet transform.

        Parameters
        ----------
        cube : 3D array
            Data to transform.
        num_scales_2d : int
            Number of wavelet scales in the 2D (i.e. the first two axes) domain.
        num_scales_1d : int
            Number of wavelet scales in the 1D (i.e. third axis) domain.

        Returns
        -------
        inds : nested list of tuples
            Pairs of index values arranged such that inds[i][j] gives the
            (start, end) indices for the band (2d_scale, 1d_scale) = (i, j).
            See `_extract_index_ranges` docstring.
        coeffs : 1D array
            A flattened array object containing all the wavelet coefficients in
            increasing order of scale. Also included are the sizes of the
            transformed cube bands along each axis as three ints just before
            the coefficients themselves. Unpacking this array to access
            specific coefficients for a given (i, j) band requires `inds`.

        """
        # Compute the transform
        self._mr2d1d = pysparse.MR2D1D(type_of_transform=self.transform_type,
                                       normalize=False,
                                       verbose=False,
                                       NbrScale2d=num_scales_2d,
                                       Nbr_Plan=num_scales_1d)
        coeffs = self._mr2d1d.transform(cube)

        # Determine the starting/ending index values of the 1d array that
        # correspond to the wavelet coefficients of band (scale_2d, scale_1d)
        inds = self._extract_index_ranges(coeffs)

        return inds, coeffs

    def reconstruct(self, coeffs):
        """Inverse 2D1D wavelet transform.

        Parameters
        ----------
        coeffs : 1D array
            Wavelet coefficients plus index markers packaged as a 1D array,
            i.e. the output of decompose().

        Returns
        -------
        cube : 3D array
            Reconstructed 3D data cube.

        """
        assert hasattr(self, '_mr2d1d'), "Need to run deconstruction first."

        return self._mr2d1d.reconstruct(coeffs)

    def _extract_index_ranges(self, coeffs):
        """Index ranges of transformed coefficients for all pairs of scales.

        Parameters
        ----------
        coeffs : 1D array
            Wavelet coefficients plus index markers packaged as a 1D array,
            i.e. the output of decompose().

        Returns
        -------
        inds : nested list of tuples
            Pairs of index values arranged such that inds[i][j] gives the
            (start, end) indices for the band (2d_scale, 1d_scale) = (i, j).
            The actual coefficients of the (i, j) band can therefore be accessed
            as coeffs[start:end].

        """
        n_scales_2d = int(coeffs[0])
        n_scales_1d = int(coeffs[1])

        inds = [[() for _ in range(n_scales_1d)] for _ in range(n_scales_2d)]

        # Starting index
        start = end = 2

        # Traverse the flattened array to pull out ranges for each index pair
        for ii in range(n_scales_2d):
            for jj in range(n_scales_1d):
                # Starting index for this band
                start = end + 3
                # Extract band sizes
                nx, ny, nz = map(int, coeffs[start-3 : start])
                # Total number of coefficients in this band
                ncoeff = nx * ny * nz
                # Ending index for this band
                end = start + ncoeff
                inds[ii][jj] = (start, end)

        return inds
