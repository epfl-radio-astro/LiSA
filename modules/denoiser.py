import numpy as np

from .util.wavelets import StarletTransform, Wavelet2D1DTransform

# quick copy-paste for debug
# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(model_k, cmap='gist_stern')
# plt.title(i)
# plt.colorbar()
# plt.show()


class SimpleStarletDenoiser(object):
    """
    Class that manages starlet denoising for an arbitrarily sized data cube.
    The model is simply Y = X + N, where X is the noiseless signal and N is the noise.
    The denoising implies minimising the mean-squared error, subject to the following constraints:
    - sparsity in starlet space, for each frequency slices INDEPENDENTLY
    - positivity constraint
    """

    def __init__(self, num_procs=1, threshold_type='soft', verbose=True):
        """Initialise the denoiser.
        Parameters
        ----------
        num_procs : int
            Number of threads ised by pySAP to perform the starlet transforms
        threshold_type : string
            Supported values are 'soft' or 'hard'. Default: 'soft'
        verbose : bool
            If True, prints some more info, useful for debugging. Default: True
        """
        self.starlet = StarletTransform(num_procs=num_procs, fast_inverse=True)
        self._threshold_type = threshold_type
        self._verbose = verbose

    def __call__(self, *args, **kwargs):
        """Alias for self.denoise()
        """
        return self.denoise(*args, **kwargs)

    def denoise(self, input_image, method='simple', threshold_level=3,
                threshold_increment_high_freq=2, num_scales=None, **kwargs_method):
        """Denoise the data according to the chosen method.
        Parameters
        ----------
        input_image : array_like
            Input data cube.
        method : string
            Denoising method, among 'naive', 'simple' and 'iterative'.
            'iterative' should give the better results, but is longer that the 'simple' method.
            Default: 'simple'.
        threshold_level : int
            Threshold level, as a detection signicance, in noise units (generally between 3 and 5 for '3-5 sigmas' detection).
            Default: 3
        threshold_increment_high_freq : int
            Increment of the above threshold_level for the highest frequencies (usually associated with pure noise).
            Default: 2
        num_scales : int
            Number of starlet decomposition scales. Maximal value is int(np.log2(input_image_.shape[-1])). Default: None (max. value).
        kwargs_method : dict
            [See docstring of each method]
        Returns
        -------
        array_like
            Denoised array
        """
        # standardize input shape
        input_image_ = np.copy(input_image)
        if len(input_image_.shape) == 2:
            input_image_ = input_image_[None, :, :]  # add extra fake dimension
            input_is_2d = True
        elif len(input_image_.shape) == 3:
            input_is_2d = False
        else:
            raise ValueError(f"Input image shape {input_image_.shape} is not supported")

        # set the number of decomposition scales
        smaller_edge = min(input_image_.shape[-2], input_image_.shape[-1])
        num_scales_max = int(np.log2(smaller_edge))
        if num_scales is None or num_scales < 2 or num_scales > num_scales_max:
            # choose the maximum allowed number of scales
            num_scales = num_scales_max
            if self._verbose is True:
                print(f"Number of wavelet scales set to {num_scales} (maximum value allowed by input image)")

        if num_scales > self.starlet.num_precomputed:
            raise NotImplementedError(f"Pre-computed noise in starlet space has been implemented"
                                      f" for up to {len(NOISE_TAB)} scales ({num_scales} required)")

        # initialise settings for the denoiser
        self._data = input_image_
        self._num_bands = self._data.shape[0]
        self._num_pixels = self._data.shape[1] * self._data.shape[2]
        self._num_scales = num_scales
        self._thresh_min = float(threshold_level)
        self._thresh_increm = float(threshold_increment_high_freq)

        # select and run the denoiser
        if method == 'simple':
            output_image = self._denoise_simple()
        elif method == 'iterative':
            output_image = self._denoise_iterative(**kwargs_method)
        elif method == 'naive':
            output_image = self._denoise_naive()
        else:
            raise ValueError(f"Denoising method '{method}' is not supported")

        # make sure output shape is identical to input shape
        if input_is_2d:
            output_image = np.squeeze(output_image)
        return output_image

    def _denoise_iterative(self, num_iter=20, num_reweight=2, progressive_threshold=True):
        """Denoise the data simply by doing an iterative thresholding in starlet space with positivity constraint,
        based on estimated noise per frequency slice propagated in starlet space.
        Parameters
        ----------
        num_iter : int
            Number of iterations. Default: 20
        num_reweight : int
            Number of times l1-reweighting is applied. Default: 2
        progressive_threshold : bool
            If True, the threshold is exponentially decreased from an initial values estimated from the data,
            to the chosen final value. Default: True
        Returns
        -------
        array_like
            Denoised array
        """
        if self._threshold_type == 'hard':
            num_reweight = 1

        num_iter_min_threshold = 5

        # apply denoising to each band INDEPENDENTLY
        data_denoised = np.zeros_like(self._data)
        for k in range(self._num_bands):
            if k%10 ==0: print("processing band {0}".format(k))
            # get the data for the current band
            data_k = self._data[k, :, :]
            # propagate noise in wavelet space
            noise_k = self._estimate_noise(data_k)
            noise_levels_k = self._propagate_noise(noise_k)
            # initialise the model
            model_k = np.zeros_like(data_k)

            # define the gradient of the loss function
            grad = lambda x: self.gradient(x, data_k)

            # gradient descent step size
            step_size = 1. # 0.001  # 1. / 0.98 # TODO: compute spectral norm

            # initialise with no weights for first pass
            weights = None

            for j in range(num_reweight):
                # get the initial threshold value (in noise units)
                if progressive_threshold is True and num_iter > num_iter_min_threshold:
                    thresh_init = self._estimate_threshold(data_k, noise_levels_k)
                    thresh = thresh_init
                else:
                    thresh = self._thresh_min

                for i in range(num_iter):
                    # performs proximal gradient descent update
                    prox = lambda x, y: self._proximal(x, thresh, noise_levels_k, weights=weights) # by convention, takes 2 args
                    model_k_next = self.step_forward_backward(model_k, grad, prox, step_size)
                    # update model for next iteration
                    model_k = model_k_next
                    # update threshold level if needed
                    if progressive_threshold is True and thresh > self._thresh_min:
                        thresh = self.exponential_decrease(thresh, thresh_init, self._thresh_min,
                                                           num_iter, num_iter_min_threshold)

                # update weights if necessary
                if num_reweight > 1:
                    weights = self._compute_weights(model_k, noise_levels_k, thresh)

            # now we are done with this band
            data_denoised[k, :, :] = model_k
        return data_denoised

    def _denoise_simple(self):
        """Denoise the data simply by doing a one-step thresholding in starlet space with positivity constraint,
        based on estimated noise per frequency slice propagated in starlet space.
        This is effectively equivalent to self._denoise_iterative() method, with num_iter=1 and no progressive_threshold=False.
        Returns
        -------
        array_like
            Denoised array
        """
        # apply denoising to each band INDEPENDENTLY
        data_denoised = np.zeros_like(self._data)
        for k in range(self._num_bands):
            # get the data for the current band
            data_k = self._data[k, :, :]
            # propagate noise in wavelet space
            noise_k = self._estimate_noise(data_k)
            noise_levels_k = self._propagate_noise(noise_k)
            # one-pass filter the data
            model_k = self._proximal(data_k, self._thresh_min, noise_levels_k)
            # now we are done with this band
            data_denoised[k, :, :] = model_k
        return data_denoised

    def _denoise_naive(self):
        """Denoise the data simply by thresholding the values in direct space,
        based on estimated noise per frequency slice.
        This does not apply any sparsity nor positivity constraint.
        Returns
        -------
        array_like
            Denoised array
        """
        # apply denoising to each band INDEPENDENTLY
        data_denoised = np.zeros_like(self._data)
        for k in range(self._num_bands):
            # get the data for the current band
            data_k = self._data[k, :, :]
            # estimate the noise level
            noise_k = self._estimate_noise(data_k)
            # naive thresholding of the data itself
            model_k = self.threshold(data_k, self._thresh_min*noise_k, threshold_type='hard')
            # now we are done with this band
            data_denoised[k, :, :] = model_k
        return data_denoised

    @staticmethod
    def step_forward_backward(x, grad, prox, step_size):
        """One step of the Forward Backward Algorithm.
        Parameters
        ----------
        x : array_like
            Variable being optimised
        grad : function
            Function that takes as a unique argument the variable x
        prox : function
            Function that takes two positional arguments: the variable being optimised, and the gradient step size
        step_size : float
            Step size for the gradient descent step
        Returns
        -------
        array_like
            Updated variable
        """
        x_next = prox(x - step_size * grad(x), step_size)
        return x_next

    @staticmethod
    def gradient(model, data):
        """Gradient of the data-fidelity term in the lost function, with respect to the main variable.
        The model is Y = X + N, where X in the denoised signal and N is the noise.
        The data-fidelity term is the mean-squared error ||Y' - Y||, where Y' represents the data.
        Parameters
        ----------
        model : array_like
            Any array
        data : array_like
            Any array
        Returns
        -------
        array_like
            Gradient of the data-fidelity term
        """
        res = data - model
        grad = - res
        return grad

    def _proximal(self, array, thresh, noise_levels, weights=None):
        """Proximal operator of the l0- or l1- sparsity + positivity constraints.
        Parameters
        ----------
        array : array_like
            1D or 2D array
        thresh : float
            Threshold value in noise units
        noise_levels : array_like
            Noise level per starlet scale
        weights:
            Weights per pixel per scale. If None, no weights are applied
        Returns
        -------
        array_like
            Array on which the constraints have been applied
        """
        # sparsity constraint in wavelet sapce
        array_proxed = self._prox_sparsity_constraint(array, thresh, noise_levels, weights=weights)
        # positivity constraint
        array_proxed = self._prox_positivity_constraint(array_proxed)
        return array_proxed

    def _prox_sparsity_constraint(self, array, thresh, noise_levels, weights=None):
        """Proximal operator of the l0- or l1- sparsity constraint (hard or soft threshold, respectively)
        Parameters
        ----------
        array : array_like
            1D or 2D array
        thresh : float
            Threshold value in noise units
        noise_levels : array_like
            Noise level per starlet scale
        weights:
            Weights per pixel per scale. If None, no weights are applied
        Returns
        -------
        array_like
            Array on which the sparsity constraint has been applied
        """
        # wavelet transform
        coeffs = self.starlet.decompose(array, self._num_scales)
        # make sure weights are a cube as well
        if weights is None:
            weights = np.ones_like(coeffs)
        # threshold coefficients except last (coarse) scale
        for c in range(self._num_scales-1):
            lambda_ = thresh
            if c == 0: lambda_ += self._thresh_increm
            thresh_ = lambda_ * noise_levels[c] * weights[c, :, :]
            coeffs[c, :, :] = self.threshold(coeffs[c, :, :], thresh_, self._threshold_type)
        # inverse wavelet transform
        array_proxed = self.starlet.reconstruct(coeffs, self._num_scales, self._num_pixels)
        return array_proxed

    @staticmethod
    def _prox_positivity_constraint(array):
        """Proximal operator of the positivity constraint
        Parameters
        ----------
        array : array_like
            Any array that supports index slicing
        Returns
        -------
        array_like
            Array with all negative entries set to zero
        """
        array_pos = np.copy(array)
        array_pos[array < 0] = 0.
        return array_pos

    @staticmethod
    def threshold(array, threshold_value, threshold_type='soft'):
        """Translate the noise estimation from direct space to starlet space
        Parameters
        ----------
        array : array_like
            1D or 2D array
        threshold_value : float
            Threshold in same units as array
        threshold_type : string
            Supported values are 'soft' or 'hard'
        Returns
        -------
        array_like
            Thresholded array
        Raises
        ------
        ValueError
            If the input array is not 1D or 2D
        ValueError
            If threshold_type is not supported
        """
        if len(array.shape) > 2:
            raise ValueError(f"Soft thresholding only supported for 1D or 2D arrays")
        if threshold_type == 'soft':
            array_th = np.sign(array) * np.maximum(np.abs(array) - threshold_value, 0.)
        elif threshold_type == 'hard':
            array_th = np.copy(array)
            array_th[np.abs(array) <= threshold_value] = 0.
        else:
            raise ValueError(f"Threshold type '{threshold_type}' is not supported")
        return array_th

    def _propagate_noise(self, noise):
        """Translate the noise estimation from direct space to starlet space
        Parameters
        ----------
        noise : float
            Noise value in direct space
        Returns
        -------
        float
            Noise per starlet scale
        """
        # scale the noise to each starlet scale
        return noise * self.starlet.energy_per_scale(self._num_scales)

    def _compute_weights(self, array, noise_levels, threshold):
        """Compute weight (per pixel per starlet scale) for l1-reweighting scheme
        (used for 'soft' threshold_type) based on the starlet decomposition of the input array
        Parameters
        ----------
        array : array_like
            Image on which the noise is estimated
        noise_levels : array_like
            1D array or list containing the noise level per starlet scale
        threshold : float
            Threshold in noise units
        Returns
        -------
        float
            Cube containing a weight value for each pixel and each scale
        """
        coeffs = self.starlet.decompose(array, self._num_scales)
        # construct cube with with constant threshold level per decomposition scale
        thresh_ = np.stack([nl * np.ones_like(array) for nl in noise_levels])
        thresh_[0, :, :]  *= (threshold + self._thresh_increm)
        thresh_[1:, :, :] *= threshold
        # compute weights
        weights = 1. / ( 1 + np.exp(10 * (coeffs - thresh_)) )
        return weights

    def _estimate_noise(self, array):
        """Estimate noise standard deviation from the median absolute deviation (MAD)
        on the first starlet decomposition scale
        Parameters
        ----------
        array : array_like
            Image on which the noise is estimated
        Returns
        -------
        float
            Noise standard deviation
        """
        array_eff = self.starlet.decompose(array, self._num_scales)[0, :, :]
        #array_eff = array
        mad = np.median(np.abs(array_eff - np.median(array_eff)))
        return 1.48 * mad

    def _estimate_threshold(self, array, noise_levels, fraction=0.9):
        """
        estimate maximum threshold, in units of noise, used for thresholding wavelets
        coefficients during optimization
        Parameters
        ----------
        data : array_like
            Imaging data.
        fraction : float, optional
            From 0 to 1, fraction of the maximum value of the image in transformed space, normalized by noise, that is returned as a threshold.
        Returns
        -------
        float
            Threshold level.
        """
        coeffs = self.starlet.decompose(array, self._num_scales)
        coeffs_norm = coeffs / noise_levels[:, None, None]
        coeffs_norm[noise_levels == 0] = 0
        # returns a fraction of max value, so only the highest coeffs is able to enter the solution
        return fraction * np.max(coeffs_norm[:-1]) # ignore last scale

    @staticmethod
    def exponential_decrease(curr_value, init_value, min_value, num_iter, num_iter_at_min_value):
        """Computes a exponentially decreasing value, for a given loop index, starting at a specified value.
        Parameters
        ----------
        curr_value : float
            Current value to be updated
        init_value : float
            Value at iteration 0.
        min_value : float
            Minimum value, reached at iteration num_iter - num_iter_at_min_value - 1.
        num_iter : int
            Total number of iterations.
        num_iter_at_min_value : int
            Number of iteration for which the returned value equals `min_value`.
        Returns
        -------
        float
            Exponentially decreased value.
        Raises
        ------
        ValueError
            If num_iter - num_iter_at_min_value < 1, cannot compute the value.
        """
        num_iter_eff = num_iter - num_iter_at_min_value
        if num_iter_eff < 1:
            raise ValueError(f"Too low number of iterations ({num_iter}) to decrease threshold")
        exp_factor = np.exp(np.log(min_value/init_value) / num_iter_eff)
        new_value = curr_value * exp_factor
        return max(new_value, min_value)


class Denoiser2D1D(object):
    """Denoise a data cube using a 2D1D wavelet decomposition.
    The model is simply Y = X + N, where X is the noiseless signal and N is
    the noise.
    """
    def __init__(self, threshold_type='soft', correlated_noise=False, verbose=True):
        """Initialise the denoiser.
        Parameters
        ----------
        threshold_type : string
            Supported values are 'soft' or 'hard'. Default: 'soft'
        correlated_noise : bool
            If True, an iterative noise estimation will be carried out to prevent under-estimating the noise.
        verbose : bool
            If True, prints some more info, useful for debugging. Default: True.
        """
        self.mr2d1d = Wavelet2D1DTransform()
        self._threshold_type = threshold_type
        self._correl_noise = correlated_noise
        self._verbose = verbose

    def __call__(self, *args, **kwargs):
        """Alias for self.denoise()"""
        return self.denoise(*args, **kwargs)

    def denoise(self, input_cube, threshold_level=3,
                threshold_increment_high_freq=2, num_scales_2d=None,
                num_scales_1d=None, noise_estimate=None,
                num_iter_noise=3, return_noise_levels=False):
        """Denoise a data cube according to the chosen method.
        Parameters
        ----------
        input_cube : array_like (3D)
            Input data cube. The frequency axis is assumed to be first.
        threshold_level : int
            Threshold level, as a detection signicance, in noise units
            (generally between 3 and 5 for a '3-5 sigma' detection).
            Default: 3
        threshold_increment_high_freq : int
            Increment of the above threshold_level for the highest frequencies
            (usually associated with pure noise).
            NOTE this option is not currently used.
            Default: 2
        num_scales_2d : int
            Number of starlet decomposition scales for the 2D images. Maximal
            value is int(np.log2(input_image_.shape[-1])).
            Default: None (max. value).
        num_scales_1d : int
            Number of wavelet scales for the 1D axis. Maximal value is
            int(np.log2(input_image_.shape[0])).
            Default: None (max. value).
        noise_estimate : array_like, same shape as input `input_cube`
            An estimate of the noise (e.g. by simulation). If not provided,
            the noise level is estimated automatically in each wavelet sub-band.
            Default: None
        num_iter_noise : number of iterations for estimating the noise in wavelet space.
            Only used if correlated_noise is True.
        Returns
        -------
        array_like
            Denoised array
        """
        # Set the number of 2D decomposition scales
        num_scales_2d_max = int(np.log2(input_cube.shape[1])) - 1
        if num_scales_2d is None or num_scales_2d < 2 or num_scales_2d > num_scales_2d_max:
            # choose the maximum allowed number of scales
            num_scales_2d = num_scales_2d_max
            if self._verbose is True:
                print(f"Number of 2D wavelet scales set to {num_scales_2d} "
                      "(maximum value allowed by input image)")

        # Set the number of 1D decomposition scales
        num_scales_1d_max = int(np.log2(input_cube.shape[0])) - 1
        if num_scales_1d is None or num_scales_1d < 2 or num_scales_1d > num_scales_1d_max:
            # choose the maximum allowed number of scales
            num_scales_1d = num_scales_1d_max
            if self._verbose is True:
                print(f"Number of 1D wavelet scales set to {num_scales_1d} "
                      "(maximum value allowed by input image)")

        # Check that the noise realisation has the same shape as the input
        if noise_estimate is not None:
            error_msg = "Noise estimate must have the same shape as the input."
            assert input_cube.shape == noise_estimate.shape, error_msg

        # Initialise settings for the denoiser
        self._data = input_cube
        self._num_bands = self._data.shape[0]
        self._num_pixels = self._data.shape[1] * self._data.shape[2]
        self._num_scales_2d = num_scales_2d
        self._num_scales_1d = num_scales_1d
        self._thresh_min = float(threshold_level)
        self._thresh_increm = float(threshold_increment_high_freq)
        self._noise = noise_estimate
        self._num_iter_noise = num_iter_noise

        # Run the denoiser (could implement iterative method if necessary)
        result = self._denoise_simple(return_noise_levels)

        return result

    def _denoise_simple(self, return_noise_levels):
        """Denoise the data using a one-step thresholding in 3D wavelet space.
        This is effectively equivalent to the iterative method with num_iter=1
        and progressive_threshold=False. A positivity constraint is enforced.
        Returns
        -------
        array_like
            Denoised data cube
        """
        # Denoise considering the full 3D wavelet deconstruction
        inds, w_data = self.mr2d1d.decompose(self._data,
                                             self._num_scales_2d,
                                             self._num_scales_1d)

        # Forward transform the noise cube if provided
        if self._noise is not None:
            _, w_noise = self.mr2d1d.decompose(self._noise,
                                               self._num_scales_2d,
                                               self._num_scales_1d)

        # Extract coeffs at smallest 2D scale for noise estimation
        # if self._correl_noise is False:
        noise_est = []
        for scale1d in range(self._num_scales_1d):
            start0, end0 = inds[0][scale1d]
            nz0 = (end0 - start0) // (self._num_pixels)
            c_data0 = w_data[start0:end0].reshape(nz0, self._num_pixels)
            noise_est.append(1.48 * self._mad2d(c_data0))

        # If noise is spatially correlated, iteratively estimate it in wavelet space from data residuals
        # else:
        residual_cube = np.copy(self._data)
        filtered_cube = 0.
        for i in range(self._num_iter_noise):
            r_inds, residual_coef = self.mr2d1d.decompose(residual_cube,
                                                        self._num_scales_2d,
                                                        self._num_scales_1d)
            for scale2d in range(self._num_scales_2d):
                for scale1d in range(self._num_scales_1d):
                    if scale1d == self._num_scales_1d-1 and scale2d == self._num_scales_2d-1:
                        continue  # coarse scale ?
                    start, end = r_inds[scale2d][scale1d]
                    nz = (end - start) // (self._num_pixels)
                    residual_coef_j = residual_coef[start:end].reshape(nz, self._num_pixels)
                    mad = self._mad2d(residual_coef_j)
                    std_dev = 1.48 * mad
                    residual_coef_j = self.threshold(residual_coef_j, 5.*std_dev, threshold_type='hard')
                    residual_coef[start:end] = residual_coef_j.flatten()
            filtered_cube = filtered_cube + self.mr2d1d.reconstruct(residual_coef)
            residual_cube = self._data - filtered_cube
        r_inds, residual_coef = self.mr2d1d.decompose(residual_cube,
                                                      self._num_scales_2d,
                                                      self._num_scales_1d)
        noise_est_new = []
        for scale2d in range(self._num_scales_2d):
            noise_est_new_in = []
            for scale1d in range(self._num_scales_1d):
                start, end = r_inds[scale2d][scale1d]
                nz = (end - start) // (self._num_pixels)
                residual_coef_j = residual_coef[start:end].reshape(nz, self._num_pixels)
                noise_est_new_in.append(1.48 * self._mad2d(residual_coef_j))
            noise_est_new.append(noise_est_new_in)

        # This will store the estimation of the noise level per 2D scale, averaged over 1D scales
        indices = []
        noise_levels_uncorrel = []
        noise_levels_correl = []

        # Filter (threshold) the data with one pass
        for scale2d in range(self._num_scales_2d):
            for scale1d in range(self._num_scales_1d):

                if scale1d == self._num_scales_1d-1 and scale2d == self._num_scales_2d-1:
                    continue  # coarse scale ?

                # Extract the coefficients for this sub-band
                # Note that the coeffs are stored as a 2D array, where the first
                # dimension is frequency, and the second represents the flattened
                # 2D spatial dimensions
                start, end = inds[scale2d][scale1d]
                nz = (end - start) // (self._num_pixels)
                c_data = w_data[start:end].reshape(nz, self._num_pixels)

                # z-dependent noise std. dev. of this sub-band
                if self._noise is not None:
                    c_noise = w_noise[start:end].reshape(nz, self._num_pixels)
                    noise_level = c_noise.std(axis=1)[..., None]
                else:
                    # Estimate the noise std. dev. by propagating the estimate
                    # from the finest 2D scale (j = 0)
                    noise_level_uncorrel = self._propagate_noise(noise_est[scale1d], scale2d)
                    
                    # or through estimation on pre-filtered cube
                    # noise_level = noise_est_new[scale2d, scale1d]
                    c_noise = residual_coef[start:end].reshape(nz, self._num_pixels)
                    noise_level_correl = 1.48 * self._mad2d(c_noise)

                    if self._correl_noise is False:
                        noise_level = noise_level_uncorrel
                    else:
                        noise_level = noise_level_correl

                    # print(f"mean noise_est at {scale2d}, {scale1d} : {np.mean(noise_level):.3e}")

                # Increase the threshold for the highest spatial freqs
                thresh = self._thresh_min
                if scale2d == 0: # or scale1d == 0:
                    thresh += self._thresh_increm

                # Update this band with thresholded coefficients
                c_thresh = self._prox_sparsity_constraint(c_data, thresh, noise_level)
                w_data[start:end] = c_thresh.flatten()

                # Save noise levels for output
                indices.append((scale2d, scale1d))
                noise_levels_uncorrel.append(np.mean(noise_level_uncorrel))
                noise_levels_correl.append(np.mean(noise_level_correl))

        # Bring back to direct space by inverse transform
        result = self.mr2d1d.reconstruct(w_data)

        # Apply the positivity constraint
        result = self._prox_positivity_constraint(result)

        if return_noise_levels is True:
            return result, (indices, noise_levels_uncorrel, noise_levels_correl)

        return result


    def _prox_sparsity_constraint(self, coeffs, thresh, noise_level, weights=None):
        """Proximal operator of the l0- or l1-sparsity constraint.
        Parameters
        ----------
        coeffs : array_like
            1D array of this sub-band
        thresh : float
            Threshold value in noise units
        noise_level : float
            Noise level at the scale represented by `array`
        weights : array_like
            Weights per pixel per scale. If None, no weights are applied
        Returns
        -------
        array_like
            Array on which the sparsity constraint has been applied
        """
        if weights is None:
            weights = np.ones_like(coeffs)

        # Threshold coefficients
        lmbda = thresh * noise_level * weights
        return self.threshold(coeffs, lmbda, self._threshold_type)

    @staticmethod
    def _prox_positivity_constraint(array):
        """Proximal operator of the positivity constraint.
        Parameters
        ----------
        array : array_like
            Any array
        Returns
        -------
        array_like
            Array with all negative entries set to zero
        """
        return np.maximum(0, array)

    @staticmethod
    def threshold(array, threshold_value, threshold_type='soft'):
        """Translate the noise estimation from direct space to wavelet space.
        Parameters
        ----------
        array : array_like
            1D or 2D array
        threshold_value : float
            Threshold in same units as array
        threshold_type : string
            Supported values are 'soft' or 'hard'
        Returns
        -------
        array_like
            Thresholded array
        Raises
        ------
        ValueError
            If the input array is not 1D or 2D
        ValueError
            If threshold_type is not supported
        """
        if len(array.shape) > 2:
            raise ValueError(f"Soft thresholding only supported for 1D or 2D arrays")
        if threshold_type == 'soft':
            array_th = np.sign(array) * np.maximum(np.abs(array) - threshold_value, 0.)
        elif threshold_type == 'hard':
            array_th = np.copy(array)
            array_th[np.abs(array) <= threshold_value] = 0.
        else:
            raise ValueError(f"Threshold type '{threshold_type}' is not supported")
        return array_th

    def _propagate_noise(self, noise_est0, scale2d):
        """Estimate the noise level in a wavelet sub-band.
        Parameters
        ----------
        noise_est0 : array_like, 2D
            Estimated noise std. dev. at the finest 2D scale (scale2d=0)
            for a particular 1D scale
        scale_2d : int
            2D wavelet scale to propagate the noise to
        Returns
        -------
        array :
            Frequency-dependent noise estimation at scale2d
        """
        if scale2d == 0:
            factor = 1
        elif scale2d == 1:
            factor = 4
        else:
            factor = 9 * 2**(scale2d - 2)

        return noise_est0 / factor

    @staticmethod
    def _mad2d(array):
        """Compute the median absolute deviation (MAD)
        Parameters
        ----------
        array : array_like, 2D
            Values on which to compute the MAD
        Returns
        -------
        array :
            MAD values (still 2D)
        """
        assert np.asarray(array).ndim == 2, "array must be 2D"

        # Take the MAD over the second axis of a 2D array
        mad = np.median(np.abs(array - np.median(array, axis=1)[..., None]), axis=1)
        # mad = np.median(np.abs(array - np.median(array)))

        # Return a 2D result
        return mad[:, None]