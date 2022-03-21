import numpy as np
from scipy import ndimage
from scipy import optimize as opt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col
from modules.util.truth_info import  TruthSource
from modules.util.source_candidate import  SourceCandidate
import time, random, os, math
from astropy import modeling
from modules.nht.tools import Tools as T
import lmfit as lmf
import itertools as itt
from scipy.integrate import quad
import inspect # for debug
from iminuit import Minuit # optimisation
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from numba import jit
from scipy.special import erf
from scipy.stats import moyal


############################################################
# Base source finding module
############################################################

class SourceFinder:
    # domain is an instance of DomainReader
    def __init__(self, domain):
        self.domain = domain

    def _get_cube(self, istart, istop, jstart, jstop, kstart = 0, kstop = -1):
        # If correction is needed, it's because we use the 0-size format instead of loc1-loc2 format. And both cases are useful, we I keep them both.
        correctX = not (self.domain.xmin <= istart and self.domain.xmax > istart)
        correctY = not (self.domain.ymin <= jstart and self.domain.ymax > jstart)
        if correctX:
            istart += self.domain.xmin
            istop += self.domain.xmin
        if correctY:
            jstart += self.domain.ymin
            jstop += self.domain.ymin
        return self.domain.safe_get_cube(istart, istop, jstart, jstop, kstart, kstop)

    def _get_line(self, i, j):
        # If correction is needed, it's because we use the 0-size format instead of loc1-loc2 format. And both cases are useful, we I keep them both.
        correctX = not (self.domain.xmin <= i and self.domain.xmax > i)
        correctY = not (self.domain.ymin <= j and self.domain.ymax > j)
        if correctX: i += self.domain.xmin
        if correctY: j += self.domain.ymin
        # print("x: {} <= {} < {} is {}".format(self.domain.xmin, i, self.domain.xmax, self.domain.xmin <= i and self.domain.xmax > i))
        # print("x: {} <= {} < {} is {}".format(self.domain.ymin, j, self.domain.ymax, self.domain.ymin <= j and self.domain.ymax > j))
        return self.domain.get_freq_line(i, j)

    def find_sources(self):
        sources_list = []
        for i in range(self.domain.xmin+self.domain.border, self.domain.xmax-self.domain.border):
            for j in range(self.domain.ymin+self.domain.border, self.domain.ymax-self.domain.border):
                #print("Searching in line at x = {0}, y = {1}".format(i,j))
                if i%10 == 0 and j%10 == 0: print("Searching in line at x = {0}, y = {1}".format(i,j))
                source_cands = self.find_source(i,j)
                if len(source_cands) > 0:
                    sources_list.extend(source_cands)
        return sources_list

    def compare_to_truth(self, filepath_truth, flux_threshold = 0, do_plot = False):
        source_list = TruthSource.catalog_to_sources_in_domain(filepath_truth, self.domain)
        source_list = [s for s in source_list if s.line_flux_integral() > flux_threshold]
        print("Comparing with {0} sources from catalog in domain {1}".format(len(source_list),self.domain.index))
        thresholds = [0, 10, 20, 30, 40, 50, 100, 200, 999]
        matches = np.zeros(len(thresholds)) 
        totals  = np.zeros(len(thresholds)) 
        for s in source_list:
            cand_sources = self.find_source(int(s.x()),int(s.y()), do_plot)
            # print("Canditates found by the likelihood test:", cand_sources)
            if cand_sources == None or len(cand_sources) == 0:
                match_found = False
            else:
                nbmatches = sum([c.matching(s) for c in cand_sources])
                match_found = nbmatches > 0
                print(T.tcol("Real sources: {} candidates, of which {} are matching catalogue.".format(len(cand_sources), nbmatches), "green"))
                for c in cand_sources: print(c)
            # add results to histogram
            for i, threshold in enumerate(thresholds):
                if s.line_flux_integral() < threshold:
                    totals[i] += 1
                    if match_found: matches[i] += 1
                    break

        return(thresholds, matches, totals)

    # It will find
    def compare_to_null(self, filepath_truth, n_trials = 10, margin = 5, seed = 1234):
        # first, build list of lines that are far from any truth sources
        source_list = TruthSource.catalog_to_sources_in_domain(filepath_truth, self.domain)
        n_total = 0
        n_matches = 0

        # random.seed(seed)

        while n_trials != n_total:
            i = random.randint(self.domain.xmin+self.domain.border, self.domain.xmax-self.domain.border-1)
            j = random.randint(self.domain.ymin+self.domain.border, self.domain.ymax-self.domain.border-1)

            # check if i,j is too close to a true source (unlikely!)
            # if so, restart
            cont = False
            for s in source_list:
                if abs(s.x() - i) < margin and abs(s.y() - j) < margin:
                    cont = True
            if cont: continue

            n_total += 1
            cand_sources = self.find_source(i,j)
            
            # print(n_matches)
            if len(cand_sources) != 0:
                n_matches += 1
                for c in cand_sources:
                    print(T.tcol("Null: Candidate at x = {0}, y = {1}, z = {2}".format( i, j, c.k), "red"))
            # print(n_matches)
            # print(len(cand_sources), cand_sources)
        return (n_matches, n_total)
            

    # i and j are the RA and Dec indices
    # should return a 
    def find_source(self, i, j, do_plot = False):
        raise NotImplementedError

############################################################
# Source finding subclasses - Basic Line Source Finder.
############################################################

class BasicLineSourceFinder(SourceFinder):
    def __init__(self, domain, noise_func, threshold = 1, sigma = 7):
        super().__init__(domain)
        self.noise_func = noise_func
        self.threshold = threshold
        self.sigma = sigma

    def _get_weighted_line(self, i , j):
        # If correction is needed, it's because we use the 0-size format instead of loc1-loc2 format. And both cases are useful, we I keep them both.
        correctX = not (self.domain.xmin <= i and self.domain.xmax > i)
        correctY = not (self.domain.ymin <= j and self.domain.ymax > j)
        if correctX: i += self.domain.xmin
        if correctY: j += self.domain.ymin
        line = self.domain.get_freq_line(i,j)
        # convert from flux to S/N
        f_indices = np.arange(len(line))
        line = line * self.noise_func(f_indices)
        return line

    def _get_weighted_pencil(self, i , j, width = 1):
        pencil = self.domain.get_pencil(i,j, width)
        # add up all pixels i each plane of RA, Dec
        pencil = np.sum(pencil, axis = (1,2))
        # convert from flux to S/N
        f_indices = np.arange(len(pencil))
        pencil = pencil * self.noise_func(f_indices)
        return pencil


    def find_source(self, i, j, do_plot = False):
        line = self._get_weighted_line(i,j)
        f_indices = np.arange(len(line))
        smooth_line = ndimage.gaussian_filter(line, sigma=self.sigma)
        smooth_line_indices= f_indices
        regions = ndimage.find_objects(ndimage.label(smooth_line > self.threshold)[0])

        if do_plot:
            fig, ax = plt.subplots(1)
            
            ax.hist(f_indices, weights = line, bins = len(line), color = "turquoise")
            ax.hist(smooth_line_indices, weights = smooth_line, bins = len(line), color = "blue", alpha = 0.5, histtype = "step")
            ax.text(0,4,"Frequency line at i = {0}, j = {1}".format(i,j), {'color': 'teal'}, horizontalalignment='left')
            ax.text(0,3.5,"Using Gaussian filter with sigma = {0}, looking for excesses above {1}".format(self.sigma, self.threshold), {'color': 'b'}, horizontalalignment='left')
            for i,r in enumerate(regions):
                box = plt.Rectangle((r[0].start, -1), r[0].stop-r[0].start, 4, fill = False, edgecolor="r")
                ax.add_patch(box)
                ax.text(0,3-0.5*i,"Found source candidate from k = {0}:{1}".format(r[0].start,r[0].stop), {'color': 'r'}, horizontalalignment='left')
            plt.show()

        return [ SourceCandidate(i, j, kslice = r[0], sig = np.max(smooth_line[r[0]])) for r in regions]

############################################################
# Source finding subclasses - Likelihood Source Finder.
############################################################


class LikelihoodFinder(SourceFinder):
    # Constructor of the LikelihoodFinder class.
    # REQUIRED PARAMETERS:
    #  - domain/DomainReader: 
    #  - binning/int: 
    # OPTIONAL PARAMETERS (kwargs):
    #  - ... all the stuff listed below
    def __init__(self, domain, binning, noise_profile, **kwargs):
        # run the parent class constructor.
        super().__init__(domain)

        # default parameter values
        self.outdir = "data/"
        self.noise_profile = noise_profile

        # Then we load all the non mandatory parameters.
        if kwargs:
            for key, value in kwargs.items():
                if key == "df": self.df = value
                elif key == "method": self.method = value
                elif key == "filepath_truth": self.filepath_truth = value
                elif key == "plotdf": self.plotdf = value
                elif key == "lkh_threshold_attempts": self.lkh_threshold_attempts = value
                elif key == "lkh_threshold_margin": self.lkh_threshold_margin = value
                elif key == "debug": self.debug = value
                elif key == "manualThreshold": self.log_lkh_threshold = value
                elif key == "out_dir":  self.outdir = value
                elif key == "comm": self.comm = value
                else: print(T.tcol("The parameter \"{}\" with value {} that you are trying to setup doesn't exist.".format(key, value), "red"))
        
        self.binning = binning

        # Then, we set and load all the mandatory parameters.
        # We load the noise data for the right binning

    # This method allows to check if the provided argument exist.
    # PARAMETER:
    #  - variable/str: is the variable that we want to check
    def _exists(self, *args):
        for variable in args:
            if not hasattr(self, variable):
                raise Exception(T.tcol("The variable \'{}\' that you tried to call doesn't exist.".format(variable), "red")) 


    # Here, we define all the setters and getters for the class parameters.
    # self._binning
    @property
    def binning(self):
        self._exists('_binning')
        return self._binning
    @binning.setter
    def binning(self, binning):
        self._binning = binning
        self.load_model_allf()

    # self._df
    @property
    def df(self):
        self._exists('_df')
        return self._df
    @df.setter
    def df(self, df):
        self._df = df
    
    # self._method
    @property
    def method(self):
        if hasattr(self, '_method'): return self._method
        return 'likelihood'
    @method.setter
    def method(self, method):
        method = method.lower()
        if method == "likelihood" or method == "stn": self._method = method
        else: raise Exception(T.tcol("The testing method ({}) you've chosen doesn't exist. Please choose a correct one.".format(method), "red"))

    # self._filepath_truth
    @property
    def filepath_truth(self):
        self._exists('_filepath_truth')
        return self._filepath_truth
    @filepath_truth.setter
    def filepath_truth(self, filepath):
        self._filepath_truth = filepath
    
    # self._plotdf
    @property
    def plotdf(self):
        if hasattr(self, '_plotdf'): return self._plotdf
        else: return self.df * 5
    @plotdf.setter
    def plotdf(self, plotdf):
        self._plotdf = plotdf

    # self._log_lkh_threshold
    @property
    def log_lkh_threshold(self):
        if hasattr(self, '_log_lkh_threshold'): return self._log_lkh_threshold
        else: self._log_lkh_threshold = self._compute_log_lkh_thresh()
        return self._log_lkh_threshold
    @log_lkh_threshold.setter
    def log_lkh_threshold(self, val):
        self._log_lkh_threshold = val

    # self._lkh_threshold_attempts
    @property
    def lkh_threshold_attempts(self):
        if hasattr(self, '_lkh_threshold_attempts'): return self._lkh_threshold_attempts
        else: return 50 # default setting.
    @lkh_threshold_attempts.setter
    def lkh_threshold_attempts(self, val):
        self._lkh_threshold_attempts = val
    
    # self._lkh_threshold_margin
    @property
    def lkh_threshold_margin(self):
        if hasattr(self, '_lkh_threshold_margin'): return self._lkh_threshold_margin
        else: return 5
    @lkh_threshold_margin.setter
    def lkh_threshold_margin(self, val):
        self._lkh_threshold_margin = val
    
    # self._debug
    @property
    def debug(self):
        if hasattr(self, '_debug'): return self._debug
        else: return False
    @debug.setter
    def debug(self, val):
        self._debug = val
    
    # self._comm
    @property
    def comm(self):
        if hasattr(self, '_comm'): return self._comm
        else: return None
    @comm.setter
    def comm(self, val):
        self._comm = val
        self.size = val.Get_size()
        self.rank = val.Get_rank()
    
    # self._size
    @property
    def size(self):
        if hasattr(self, '_size'): return self._size
        else: return 1 # default for non-mpi
    @size.setter
    def size(self, val):
        self._size = val
    
     # self._rank
    @property
    def rank(self):
        if hasattr(self, '_rank'): return self._rank
        else: return 0 # default for non-mpi
    @rank.setter
    def rank(self, val):
        self._rank = val


    # This function returns the type of domain we are dealing with (as we would have different noise data)
    def domainType(self):
        return type(self.domain).__name__
    
    # Assuming a gaussian-distributed noise, this method will find the distribution function for noise at
    # a given frequency, and return its parameters formated in a numpy array as [mean, stdev].
    # PARAMETERS:
    #  - f/int: frequency on which slice we compute the noise model parameters.
    #  - do_plot/bool: is the boolean that tells whether we plot the result.
    def _characterise_noise(self, f, do_plot = False):
        # For clarity
        binning = self.binning
        debug = self.debug
        profile = self.noise_profile

        # Depending on the binning, we'll choose the starting frequency.
        mod = f%binning
        if debug: print("Frequency modulo {}: {}".format(binning, mod))
        startf = f - mod
        if debug: print("The {}-bin for freq {} starts with frequency {}".format(binning, f, startf))

        # We have to choose a ending frequency (not included), but we have to be careful with end of array.
        # In this case, we take the end of the array. (so a smaller bin)
        freq_size = self._get_line(0, 0).shape[0]
        # print(T.tcol("THIS MESSAGE IS IN RED.", "red"))
        # print("{}={}".format(self.domain.HI_cube.data[:, 0, 0].shape[0], freq_size))
        stopf = startf + binning
        if stopf > freq_size: stopf = freq_size
        if debug: print("Freq. size: {}".format(freq_size))

        # From this, we can integrate the sky_domain over the bins.
        # sky = np.sum(self.domain.HI_cube.data[startf:stopf,:,:], axis=0)
        sky = np.sum(self._get_cube(0, -1, 0, -1, startf, stopf), axis=0)
        # print("{}={}".format(np.sum(self.domain.HI_cube.data[startf:stopf,:,:], axis=0), sky))
        if debug: print("Sky shape after sum: {}".format(sky.shape))
        # if debug: print("Sky shape before sum: {}".format(self.domain.HI_cube.data[f,:,:].shape))
        if debug: print("Sky shape before sum: {}".format(self._get_cube(0, -1, 0, -1, f, f+1).shape))
        # print("{}={}".format(self.domain.HI_cube.data[f,:,:].shape, self._get_cube(0, -1, 0, -1, f, f+1).shape))

        # We can now compute the values
        # if debug: print("{} = {} if bin=1".format(np.mean(self.domain.HI_cube.data[f,:,:]), np.mean(sky)))
        if debug: print("{} = {} if bin=1".format(np.mean(self._get_cube(0, -1, 0, -1, f, f+1)), np.mean(sky)))
        # print("{}={}".format(np.mean(self.domain.HI_cube.data[f,:,:]), np.mean(self._get_cube(0, -1, 0, -1, f, f+1))))
        # if debug: print("{} = {} if bin=1".format(np.std(self.domain.HI_cube.data[f,:,:]), np.std(sky)))
        if debug: print("{} = {} if bin=1".format(np.std(self._get_cube(0, -1, 0, -1, f, f+1)), np.std(sky)))
        # print("{}={}".format(np.std(self.domain.HI_cube.data[f,:,:]), np.std(self._get_cube(0, -1, 0, -1, f, f+1))))

        # We prepare the initial values of the profile.
        # init = profile.init_values(sky)


        tic = time.time()
        weights, bin_edges = np.histogram(sky.flatten(), bins=50, density=True)
        bin_centers = bin_edges[:-1] + abs(bin_edges[1] - bin_edges[0])/2
        # p0 = profile.init_values(sky)
        popt, pcov = curve_fit(profile.pdf, bin_centers, weights, p0=profile.init_values(sky), bounds=profile.limits(sky)) # gauss, lorentz, landau
        # popt, pcov = curve_fit(profile, bin_centers, weights, p0=(np.log(np.mean(sky)), np.log(np.std(sky)))) # lognorm
        # popt, pcov = curve_fit(profile, bin_centers, weights, p0=(np.mean(sky), np.std(sky), np.std(sky))) # voigt
        # popt, pcov = curve_fit(profile, bin_centers, weights, p0=(np.mean(sky), np.std(sky), 0, np.inf)) # trucated gaussian
        if debug: print("Fitted parameters: {}".format(popt))
        # print("Chi2: {}".format(chisquare(weights, profile(bin_centers, *popt), len(weights) - dof)[0]))
        toc = time.time()
        time_curve_fit = toc-tic
        if debug: print("Time for curve_fit: {}s".format(toc-tic))

        # We plot the result.
        if do_plot:
            try:
                print("Plotting.")
                fig = plt.figure()
                title = "{}(".format(profile.name)
                for p in popt:
                    title += "{:.2e}, ".format(p)
                title = title[:-2] + ")"
                plt.title(title)
                plt.hist(sky.flatten(), bins=bin_edges, density=True, label="Noise histogram")
                plt.plot(bin_centers, profile.pdf(bin_centers, *popt), '-', label="{} fit (curve_fit) in {:.1e}s".format(profile.name, time_curve_fit))
                plt.grid(True, linewidth=.1)
                plt.xlabel('Flux [Jy/beam]')
                plt.ylabel('Occurence')
                # plt.yscale('log')
                plt.legend()
                plt.show(block=False)
            except Exception as e:
                print("Plotting failed.")
                print(e)
        
        return popt


    # This method computes the model for all frequencies, and save it to a file.
    # Note that you'll still need to load it afterward.
    # In the pipeline, prefere the use of load_model_allf, which will load the data if available, and
    # create the dataset if needed.
    # PARAMETERS:
    #  - filename/str: is the location and name of the data to be saved. Note that None uses the default
    #    naming for the data files. Note: not very well supported.
    def write_model_allf(self, filename = None):
        # For clarity
        binning = self.binning
        profile = self.noise_profile

        # First, we determine the filename, if needed.
        if filename == None: filename = self.outdir + 'noise_parameters_{}_bin-{}_x-{}-{}_y-{}-{}_border-{}_profile-{}.npz'.format(self.domainType(), binning, self.domain.xmin, self.domain.xmax, self.domain.ymin, self.domain.ymax, self.domain.border, profile.name)
        
        print("Writing noise data to {0}".format(filename))
        # First, we compute the noise parameters.
        freq_size = self._get_line(0, 0).shape[0]
        noise_params = []
        for i in range(0, freq_size, binning):
            print("Calculating noise from {0} - {1}".format(i, i+binning-1))
            noise_params.append(self._characterise_noise(i, False))
        self.noise_params = np.array(noise_params)

        # Now we save this to disk.
        with open(filename, 'wb') as f:
            # TODO : For later implementation, add a check that we consider the correct domain in the loader.
            # x = [self.domain.xmin, self.domain.xmax]
            # y = [self.domain.ymin, self.domain.ymax]
            np.savez(f, params=self.noise_params, binning=binning)


    # This method loads the noise model into self._noise_params, and assignes the binning.
    # PARAMETERS:
    #  - filename/str: is the location and name of the data to be loaded. Note that None uses the default
    #    naming for the data files. Note: not very well supported.
    def load_model_allf(self, filename = None):
        # For clarity
        binning = self.binning
        profile = self.noise_profile

        # First, we determine the filename, if needed.
        if filename == None: filename = self.outdir + 'noise_parameters_{}_bin-{}_x-{}-{}_y-{}-{}_border-{}_profile-{}.npz'.format(self.domainType(), binning, self.domain.xmin, self.domain.xmax, self.domain.ymin, self.domain.ymax, self.domain.border, profile.name)
        
        print("Reading noise data from {0}".format(filename))

        # If the file exists, we can proceed and load the result. If not, we create the new data.
        if not os.path.isfile(filename):
            print("File {} doesn't exists, creating it now!".format(filename))
            self.write_model_allf()
        
        # Finally we load the data.
        with open(filename, 'rb') as f:
            data = np.load(f)
            self._noise_params = data['params']
            # self.binning = data['binning']       

    # This method returns the likelihood of X for the bin containing f (depending on binning).
    # If you have an error, you might want to check if you loaded the model before.
    # PARAMETERS:
    #  - X/float: is the flux for which we want to compute the likelihood.
    #  - f/int: is the frequency contained in the bin for which we want the likelihood.
    def likelihood_f(self, X, f):
        # For clarity
        binning = self.binning
        profile = self.noise_profile

        # Selects the right frequency index, depending on binning
        startf = (f - f % binning)//binning

        #FIX, if noise is less than mean, return mean value
        mean = self._noise_params[startf, :][0]
        if X < mean: X = mean#(mean + X)*0.5

        # Returns likelihood.
        return profile.pdf(X, *self._noise_params[startf, :])
    
    # This method returns the likelihood for a whole spectrum.
    # PARAMETERS:
    #  - line/np.array: is the flux line array for which we want to compute the spectrum.
    #  - (startf, stopf)/(int, int): are the raw start and stop frequencies of interest.
    #  - binning/int: is the binning choice. By default, it uses the current choice. 
    def log_likelihood_spectrum(self, line, startf, stopf, binning = None):
        # For clarity 
        debug = self.debug
        
        if binning == None:
            binning = self.binning
            old_binning = binning
        else:
            if debug: print("Temporarly changing the binning from {} to {}.".format(self.binning, binning))
            old_binning = self.binning
            self.binning = binning
        debug = self.debug

        startf, stopf = self._frequency_boundaries(startf, stopf, line.shape[0])

        # We start by selecting the line of interest.
        nline = line[startf:stopf]

        # Now, we rebin it if needed.
        if binning is not 1: line_binned = self._rebinning(nline, binning)
        else: line_binned = nline
        if debug: print("fline shape: {}\nfline_binned shape: {}".format(nline.shape, line_binned.shape))
        if debug: print(nline, line_binned)

        freqsteps = np.arange(startf, stopf, binning)
        lkh = [self.likelihood_f(line_binned[i], f) for i, f in enumerate(freqsteps)]
        log_lkh = np.log10(lkh)

        # We set back the binning if needed.
        if self.binning is not old_binning:
            self.binning = old_binning
            if debug: print("Set back binning to {}.".format(old_binning))

        # Finally, we return the array.
        return np.array(log_lkh)

    # This method makes a verification histogram to see if the noise model seems alright.
    # PARAMETERS:
    #  - (i, j, midf)/(int, int, int): are the space coordinates we are analysing.
    def verificationPlot(self, i, j, midf):
        # For clarity
        binning = self.binning
        df = self.df

        # We get the line and the data.
        line = self._get_line(i,j)

        # We choose a range of frequency on which to work
        startf = midf - df - (midf - df) % binning
        stopf = midf + df - (midf + df) % binning

        # We perform the binning and get the parameters
        line_binned = []
        means = []
        stdev = []
        for f in range(startf, stopf + 1, binning):
            line_binned.append(np.sum(line[f:f+binning]))
            means.append(self._noise_params[f//binning, 0])
            stdev.append(self._noise_params[f//binning, 1])
        means.pop()
        stdev.pop()
        line_binned = np.array(line_binned)
        means = np.array(means)
        stdev = np.array(stdev)
        print(line_binned, means, stdev)
        
        # Now we can plot
        X_hist = np.array(range(startf, stopf + 1, binning))
        X_plot = np.array(range(startf, stopf, binning)) + binning//2
        fig = plt.figure()
        plt.hist(X_hist, line_binned.shape[0]-1, weights=line_binned)
        plt.errorbar(X_plot, means, yerr=stdev, ls=None, marker='o')
        plt.grid(True, linewidth=.1)
        plt.xlabel("Frequency with binning = {}".format(binning))
        plt.ylabel("Flux")
        plt.tight_layout()
        plt.show()

    # This method returns the rebinned flux line.
    # PARAMETERS:
    #  - fline/np.array: is the flux line array that we want to rebin.
    #  - binning/int: is the binning choice. 
    def _rebinning(self, fline, binning = None):
        if binning == None: binning = self.binning
        size = fline.shape[0]
        new_line = []
        for f in range(0, size, binning):
            if f < size:
                new_line.append( np.sum(fline[f:f+binning]) )
            else:
                new_line.append( np.sum(fline[f:]) )
        return np.array(new_line)

    # This method allows to check the null hypothesis, based on the truth catalog.
    # PARAMETERS:
    #  - filepath_truth/str: is the path to the truth catalog.
    #  - flux_threshold/float: is a threshold for the line integral of the flux at each position.
    def null_hyp_check(self, flux_threshold = 100):
        # For clarity
        binning = self.binning
        df = self.plotdf
        debug = self.debug

        # First, we import the true sources, and keep the one with an integral flux above some threshold.
        source_list = TruthSource.catalog_to_sources_in_domain(self.filepath_truth, self.domain)
        source_list = [s for s in source_list if s.line_flux_integral() > flux_threshold]
        if debug: print("{} source candidates found.".format(len(source_list)))

        # Then, we randomly select one of them.
        s_idx = random.randint(0, len(source_list)-1)
        if debug: print("Index choice: {}".format(s_idx))
        source = source_list[s_idx]

        # From this source, we extract the position informations.
        if debug: print("\nSource choice:\n{}\n".format(source))
        i = round(source.x())
        j = round(source.y())
        f = round(source.z())

        # We extract the exact line, and rebin it (for later graph).
        line = self._get_line(i,j)
        line_binned = self._rebinning(line)

        # We are only interesed by the data around the galaxy.
        startf = f - df - (f - df)%binning
        stopf = f + df - (f + df)%binning
        if debug: print("Start freq : {} ({})\nStop freq : {} ({})\n() = int. div. by binning {}".format(startf, startf//binning, stopf, stopf//binning, binning))

        # We define frequencies
        freq = np.array(range(startf, stopf))
        freq_hist = np.array(range(startf, stopf+1))
        freq_binned = np.array(range(startf, stopf, binning))
        freq_binned_hist = np.array(range(startf, stopf+1, binning))
        
        # We obtain the likelihoods
        lkh = self.log_likelihood_spectrum(line, startf, stopf, 1)
        lkh_binned = self.log_likelihood_spectrum(line, startf, stopf)
        
        # and the parameters of the model for the binned frequency. (binned still loaded in memory)
        
        noise_param = self._noise_params[startf//binning:stopf//binning]
        if debug: print('noise param shape: {}\n'.format(noise_param.shape))
        if debug: print("line shape: {}\nbinned line shape: {}\nlikelihood shape: {}\nbinned likelihood shape: {}\n".format(line[startf:stopf].shape, line_binned[startf//binning:stopf//binning].shape, lkh.shape, lkh_binned.shape))
        if debug: print("freq shape: {}\nbinned freq shape: {}\nlikelihood shape: {}\nbinned likelihood shape: {}\n".format(freq.shape, freq_binned.shape, lkh.shape, lkh_binned.shape))
        if debug: print("freq hist: {}\nfreq binned hist: {}\n".format(freq_hist.shape, freq_binned_hist.shape))
        if debug:
            print("lkh=(bin freq., likelihood of bin, flux of bin), noise=(mean, stdev)")
            for i in range(lkh_binned.shape[0]):
                print("lkh=({}, {:.2e}, {:.2e}), noise=({:.4e}, {:.4e})".format(freq_binned[i], 10**lkh_binned[i], line_binned[i], *noise_param[i,:]))
        
        # Now we can plot all of these.
        fig = plt.figure()
        # plt.title("Source ID: {}".format(source.ID()))
        ax1 = fig.add_subplot(211)
        ax1.plot(freq, line[startf:stopf], color='blue')
        ax1.hist(freq_binned_hist[:-1], freq_binned_hist, weights=line_binned[startf//binning:stopf//binning], color='darkblue', alpha=0.5)
        ax1.errorbar(freq_binned + binning//2, noise_param[:,0], yerr=noise_param[:,1], color='black', ls='', marker='o')
        ax1.grid(True, linewidth=.1)
        ax1.set_xlabel("Frequency")
        ax1.set_ylabel("Flux [Jy/beam]")
        ax1.legend(["bin=1", "bin={}".format(binning), "bin={} noise param.".format(binning)])
        ax1.set_xlim(startf - 5, stopf + 5)


        ax2 = fig.add_subplot(212)
        # ax2.plot(freq, lkh, color='red', alpha = 0.3)
        ax2.hist(freq_binned_hist[:-1], freq_binned_hist, weights=lkh_binned, color='darkred', alpha = 0.5)
        ax2.grid(True, linewidth=.1)
        ax2.set_xlabel("Frequency")
        ax2.set_ylabel("log-likelihood") # plotting the log-likelihood is ugly
        # ax2.set_yscale("log")
        ax2.legend(["bin={}".format(binning)])
        ax2.set_xlim(startf - 5, stopf + 5)
        # ax2.legend(["bin=1", "bin={}".format(binning)])
        plt.tight_layout()
        plt.show(block=False)

    # This method returns the start and stop boundaries of the frequency with respect to binning
    # in the right format.
    # PARAMETER:
    #  - (startf, stopf)/(int, int): are the raw start and stop frequencies of interest.
    #  - frequency_length/int: is the length of the frequency space.
    def _frequency_boundaries(self, startf, stopf, frequency_length):
        # For clarity
        binning = self.binning

        startf = startf - startf%binning
        stopf = stopf - stopf%binning
        if stopf > frequency_length: stopf = frequency_length - frequency_length%binning
        if startf < 0: startf = 0
        # if debug: print("startf={}, stopf={}, binning={}".format(startf, stopf, binning))
        return [int(startf), int(stopf)]

    # This method returns the joint likelihood of multiple bins.
    # PARAMETERS:
    #  - line/np.array: full spectrum for one space position.
    #  - (startf, stopf)/(int, int): are the start and stop frequencies for the range in which we'll
    #    compute the joint likelihood.
    #  - binning/int: is the binning choice. By default, it uses the current choice. 
    def joint_log_likelihood(self, line, startf, stopf, binning = None):
        # We start by formating the startf and stopf.
        startf, stopf = self._frequency_boundaries(startf, stopf, line.shape[0])

        # For this line, we get the likelihoods.
        log_likelihoods = self.log_likelihood_spectrum(line, startf, stopf, binning)

        # Then we multiple all the likelihood together to have the joint likelihood.
        # if debug: print("joint likelihood: {}".format(np.prod(likelihoods, axis=0)))
        return np.sum(log_likelihoods, axis=0)

    # get array of joint likelihoods for frequency line at i,j
    def _get_joint_log_likelihoods(self, i, j):

        # get the line and likelihood for the whole spectrum.
        line = self._get_line(i, j)
        log_likelihoods = self.log_likelihood_spectrum(line, 0, line.shape[0])

        # We compute the joint likelihood for each possible window in the sample.
        windowSize = 2*self.df // self.binning
        lkhSize = log_likelihoods.shape[0]
        joint = []
        for startf, stopf in zip(range(0,lkhSize - windowSize), range(windowSize, lkhSize)):
            joint.append(np.sum(log_likelihoods[startf:stopf]))
        joint = np.array(joint)

        return joint


    # This method plots the line integral of the flux with respect to the log-likelihood for a large range of
    # frequencies to have good view on what we have.
    # PARAMETERS:
    #  - filepath_truth/str: is the path to the truth catalog.
    #  - flux_threshold/float: is a threshold for the line integral of the flux at each position.
    #  - binning/int: is the binning choice. By default, it uses the current choice. 
    def flux_likelihood_graph(self, flux_threshold, binning = None):
        # for clarity
        df = self.df
        if binning == None: binning = self.binning
        debug = self.debug

        # First, we import the true sources, and keep the one with an integral flux above some threshold.
        source_list = TruthSource.catalog_to_sources_in_domain(self.filepath_truth, self.domain)
        source_list = [s for s in source_list if s.line_flux_integral() > flux_threshold]
        if debug: print("{} source candidates found.".format(len(source_list)))

        likelihoods = []
        line_flux = []
        
        # print(skylim = self.domain.HI_cube.data)
        skylim = self._get_cube(0, -1, 0, -1, 0, -1).shape
        for source in source_list:
            # We obtain the position of the source.
            i = round(source.x())
            j = round(source.y())
            f = round(source.z())
            if i >= skylim[1]: i = skylim[1] - 1
            if j >= skylim[2]: j = skylim[2] - 1
            if f >= skylim[0]: j = skylim[0] - 1
            
            # We extract the exact line.
            line = self._get_line(i,j)

            # We obtain the line flux integral for this source.
            line_flux.append(source.line_flux_integral())

            # We choose an arbitrary search window.
            startf, stopf = self._frequency_boundaries(f-df, f+df, line.shape[0])

            # We compute the likelihood for this range of frequencies.
            likelihoods.append(self.joint_log_likelihood(line, startf, stopf, binning))
        likelihoods = np.array(likelihoods)
        line_flux = np.array(line_flux)
        
        # Now, we'll try some random position to get the joint likelihood of noise parts. (it's random, but will mostly be noise)
        position_rand = []
        for i in range(50):
            x = random.randint(0, skylim[1]-1)
            y = random.randint(0, skylim[2]-1)
            f = random.randint(0,skylim[0]-1)
            position_rand.append([x,y,f])
        position_rand = np.array(position_rand)

        # For these positions, we compute the likelihood.
        likelihoods_rand = []
        for p in position_rand:
            # We get the lines
            i,j,f = p
            line = self._get_line(i,j)

            # We obtain the likelihood at this position
            startf, stopf = self._frequency_boundaries(f-df, f+df, line.shape[0])

            # We compute the likelihood for this range of frequencies.
            likelihoods_rand.append(self.joint_log_likelihood(line, startf, stopf, binning))
        likelihoods_rand = np.array(likelihoods_rand)
        line_flux_rand = np.full(likelihoods_rand.shape, 0.)

        if debug: print("Likelihood threshold: {} ({})", 10**self.log_lkh_threshold, self.log_lkh_threshold)

        # We plot these.
        fig = plt.figure()
        plt.title("df={}, flux threshold={}, binning={}".format(df, flux_threshold, binning))
        ax = fig.add_subplot(111)
        ax.scatter(np.log10(likelihoods), line_flux, s=1, color='blue')
        ax.scatter(np.log10(likelihoods_rand), line_flux_rand, s=1, color='red')
        ax.set_xlabel("$\log_{10}$(likelihood)")
        ax.set_ylabel("Line integral of flux")
        plt.legend(["Source", "Noise"])
        plt.tight_layout()
        plt.grid(True, linewidth=.1)
        plt.show(block=False)


    # This method computes a threshold for the likelihood based of the plot obtained in flux_likelihood.
    # Note that this is a joint likelihood over the whole windows of size 2*df. 
    def _compute_log_lkh_thresh(self, do_plot = False):
        # We simplify the notations.
        attempts = self.lkh_threshold_attempts
        df = self.df
        binning = self.binning
        margin = self.lkh_threshold_margin

        # We first choose a bunch of random points for this binning, and check if they are close to
        # an actual source. If so, we retry.
        source_list = TruthSource.catalog_to_sources_in_domain(self.filepath_truth, self.domain)
        print("Source list size: ", len(source_list))
        skylim = self._get_cube(0, -1, 0, -1, 0, -1).shape 
        position_rand = []
        while len(position_rand) < attempts:
            x = random.randint(0, skylim[1]-1)
            y = random.randint(0, skylim[2]-1)
            f = random.randint(0, skylim[0]-1)
            cont = False
            for s in source_list:
                if abs(s.x() - x) < margin and abs(s.y() - y) < margin:
                    cont = True
            if cont: continue
            else: position_rand.append([x,y,f])
        print("Number of positions: ", len(position_rand))
        # For these positions, we compute the likelihood.
        log_likelihoods_noise = []
        df = self.df
        for i,j,f in position_rand:

            # Get the minimum of the joint likelihoods for different windows across the entire line
            joint = np.min(self._get_joint_log_likelihoods(i, j))
            log_likelihoods_noise.append(joint)
        log_likelihoods_noise = np.array(log_likelihoods_noise)

        log_likelihoods_weaksignal = []
        log_likelihoods_strongsignal = []
        for s in source_list:
            joint = self._get_joint_log_likelihoods( int(s.x()), int(s.y()))
            if abs (np.argmin(joint) - s.z()/binning) > 20: continue
            if s.line_flux_integral() > 100:
                log_likelihoods_strongsignal.append(joint[int(s.z()/binning)])
            else:
                log_likelihoods_weaksignal.append(joint[int(s.z()/binning)])


        # With all these likelihoods, we compute the mean and the standard deviations.
        mean = np.mean(log_likelihoods_noise)
        median = np.median(log_likelihoods_noise)
        stdev = np.std(log_likelihoods_noise)
        minval = np.min(log_likelihoods_noise)
        threshold_choice = mean - 10*stdev #minval
        print("Log (mean,median,stdev,min)=({0},{1},{2},{3})".format(mean, median, stdev, minval))
        mean_sig = np.mean(log_likelihoods_strongsignal + log_likelihoods_weaksignal)
        print("Avg log Lkh of signal = {0}".format(mean_sig))

        if do_plot:
            plt.figure()
            plt.hist(log_likelihoods_noise, bins = 100, range = (-50,50), histtype = 'step', color="salmon")
            plt.hist(log_likelihoods_strongsignal, bins = 100, range = (-50,50),  histtype = 'step', color="black")
            plt.hist(log_likelihoods_weaksignal, bins = 100, range = (-50,50),  histtype = 'step', color="dodgerblue")
            plt.title("{} noise samples".format(self.lkh_threshold_attempts))
            plt.axvline(mean, 0, 1, label="mean", color="red")
            plt.axvline(median, 0, 1, label="median", color="green")
            plt.axvline(threshold_choice, 0, 1, label="threshold choice", color="orange")
            plt.legend()
            plt.xlabel("Log-likelihoods of noise sample")
            plt.ylabel("Occurence")
            plt.grid(True, linewidth=.1)
            #plt.saveas("test")
            plt.show()

        # CHOICE: As a threshold, I choose the mean minus 3 times the stdev, as it is far from mean in normal
        # distribution. -> doesn't work, negative values most of the time.
        # self._likelihood_threshold = mean - 3*stdev
        # CHOICE: As a threshold, I choose the minimum value.
        # self._likelihood_threshold = minval
        print("Log Likelihood threshold: ", threshold_choice)
        return threshold_choice

    # Will return source candidates by checking if the likelihood is too small.
    # This method will search for source candidates for one flux line. It allows different tests to be
    # chosen (at the definition of this class).
    # PARAMETERS
    #  - (i, j)/(int, int): are the space coordinates we are analysing. 
    #  - do_plot/bool: is the boolean that tells whether we plot the result.
    def find_source(self, i, j, do_plot = False):
        # for clarity
        df = self.df
        method = self.method
        debug = self.debug

        # Here, we choose the choice of likelihood test.
        if method == "likelihood":
            return self._find_source_likelihood_test(i,j,do_plot)
        elif method == "stn":
            return self._find_source_STN_test(i,j)
        else:
            raise Exception(T.tcol("The current testing method ({}) isn't valid. Please choose a correct one, and select it using the choose_search_method method.".format(method), "red"))

    # This method returns candidats obtained via the likelihood test.
    # PARAMETERS:
    #  - (i, j)/(int, int): are the space coordinates we are analysing. 
    #  - do_plot/bool: is the boolean that tells whether we plot the result.
    def _find_source_likelihood_test(self, i, j, do_plot = False):
        # for clarity
        debug = self.debug
        #if j%10 == 0: print("Calculating joint likelihoods at x = {0}, y = {1}".format(i,j))

        # We search for a source candidate by sliding over the whole likelihood and computing.
        # Note that we manually compute the joint likelihood, because it would be much faster
        # (regarding computation time).
        # windowSize = 2*df // binning
        # lkhSize = likelihoods.shape[0]
        # joint = []
        # for startf, stopf in zip(range(0,lkhSize - windowSize), range(windowSize, lkhSize)):
        #     joint.append(np.prod(likelihoods[startf:stopf]))
        # joint = np.array(joint)

        # Get the joint likelihoods across the entire line
        log_lkhs = self._get_joint_log_likelihoods( i, j)

        # get regions that are below the likelihood threshold
        if debug: print("===== label:\n {}".format(ndimage.label(log_lkhs < self.log_lkh_threshold)))
        # print("label 0: {}".format(ndimage.label(log_lkhs < self.log_lkh_threshold)[0]))
        regions = ndimage.find_objects(ndimage.label(log_lkhs < self.log_lkh_threshold)[0])
        if debug: print("===== Region:\n {}".format(regions))

        # convert from bins to frequency
        freq_regions = [ slice(r[0].start*self.binning, r[0].stop*self.binning, None) for r in regions]
        if debug: print("===== freq_regions:\n {}".format(freq_regions))

        # the significance of each source will be the negative log likelihood
        significances = [ -1*np.min( log_lkhs[r]) for r in regions]
        if debug: print("===== significances:\n {}".format(significances))

        # We want to keep the region with the lowest significance (log-likelihood)
        if significances:
            max_sig_idx = np.argmax(significances)
        
        # We can plot the canditates.
        if do_plot:
            for r in freq_regions:
                f = (r.start + r.stop)/2
                self._plot_flux_likelihood(i, j, f)

        # That's if we want to keep ALL sources.
        # results =  [ SourceCandidate(i, j, kslice = fr, sig = sig) for fr, sig in zip(freq_regions, significances)]
        # We instead want to keep only the "best" value.
        if significances:
            results = [ SourceCandidate(i, j, kslice = freq_regions[max_sig_idx], sig = significances[max_sig_idx])]
            if debug: 
                # print("===== Results:\n [")
                res = ""
                for r in results:
                    res += "{}, ".format(r)
                print("===== Results:\n [{}]".format(res[:-2]))
        else:
            results = []
        
        return results


    # This method allows to easily plot the flux and likelihood at a given position.
    # PARAMETERS:
    #  - (i,j,f)/(int,int,int): are the coordinates of the source.
    # TODO : The likelihood plot seems shifted by 2*binning.
    def _plot_flux_likelihood(self, i, j, f):
        # for clarity
        df = self.df
        plotdf = self.plotdf
        binning = self.binning

        line = self._get_line(i, j)
        line_binned = self._rebinning(line, binning)

        startf, stopf = self._frequency_boundaries(f-plotdf, f+plotdf, line.shape[0])
        startf_df, stopf_df = self._frequency_boundaries(f-plotdf+df, f+plotdf-df, line.shape[0])

        
        freq = np.array(range(startf, stopf))
        freq_hist = np.array(range(startf, stopf+1))
        freq_binned = np.array(range(startf, stopf, binning))
        freq_binned_hist = np.array(range(startf, stopf+1, binning))
        freq_binned_hist_joint = np.array(range(startf_df, stopf_df+1, binning))

        # We obtain the likelihoods
        lkh = self.log_likelihood_spectrum(line, startf, stopf, 1)
        lkh_binned = self.log_likelihood_spectrum(line, startf, stopf)
        
        
        # We search for a source candidate by sliding over the whole likelihood and computing.
        # Note that we manually compute the joint likelihood, because it would be much faster
        # (regarding computation time).
        windowSize = 2*df // binning
        lkhSize = lkh_binned.shape[0]
        joint = []
        for startf_iter, stopf_iter in zip(range(0,lkhSize - windowSize), range(windowSize, lkhSize)):
            joint.append(np.sum(lkh_binned[startf_iter:stopf_iter]))
        joint = np.array(joint)
        
        # and the parameters of the model for the binned frequency. (binned still loaded in memory)
        noise_param = self._noise_params[startf//binning:stopf//binning]

        # Now we can plot all of these.
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(freq, line[startf:stopf], color='blue')
        ax1.hist(freq_binned_hist[:-1], freq_binned_hist, weights=line_binned[startf//binning:stopf//binning], color='darkblue', alpha=0.5)
        ax1.errorbar(freq_binned + binning//2, noise_param[:,0], yerr=noise_param[:,1], color='black', ls='', marker='o')
        ax1.grid(True, linewidth=.1)
        ax1.set_xlabel("Frequency")
        ax1.set_ylabel("Flux [Jy/beam]")
        ax1.legend(["bin=1", "bin={}".format(binning), "bin={} noise param.".format(binning)])
        ax1.set_xlim(startf - 5, stopf + 5)

        print(len(freq_binned_hist_joint[:-1]), len(freq_binned_hist_joint), len(joint))

        ax2 = fig.add_subplot(212)
        # ax2.plot(freq, lkh, color='red', alpha = 0.3)
        ax2.hist(freq_binned_hist_joint[:-1], freq_binned_hist_joint, weights=joint, color='darkred', alpha = 0.5)
        # ax2.axhline(self.lkh_threshold, 0, 1)
        ax2.grid(True, linewidth=.1)
        ax2.set_xlabel("Frequency")
        ax2.set_ylabel("Joint log-likelihood over 2*df") # plotting the log-likelihood is ugly
        # ax2.set_yscale("log")
        # ax2.legend(["threshold"])
        ax2.set_xlim(startf - 5, stopf + 5)
        # ax2.legend(["bin=1", "bin={}".format(binning)])
        plt.tight_layout()
        plt.show(block=False)
        # plt.savefig()


    # This method returns candidats obtained via the signal to noise test.
    # PARAMETERS:
    #  - (i, j)/(int, int): are the space coordinates we are analysing. 
    #  - do_plot/bool: is the boolean that tells whether we plot the result.
    def _find_source_STN_test(self, i, j, do_plot = False):
        # for clarity
        df = self.df
        binning = self.binning

        # First, we obtain the line, and we bin it.
        flux = self._get_line(i, j)
        flux_binned = self._rebinning(flux)

        # Then, we iterate over the whole spectrum to get the signal to noise ratio over every window.
        df = self.df
        windowSize = 2*df // binning
        fluxSize = flux_binned.shape[0]
        binSizeHz = self.domain.ax3[df] - self.domain.ax3[0] # constant, so not useful to compute.
        STN = []
        signals = []
        noises = []
        for startf, stopf in zip(range(0,fluxSize - windowSize), range(windowSize, fluxSize)):
            signal = np.sum(flux_binned[startf:stopf]) * binSizeHz
            signals.append(signal)
            noise = np.linalg.norm(self._noise_params[startf:stopf,1])# * binSizeHz
            noises.append(noise)
            if debug:
                if np.abs(signal/noise) > 1. and np.abs(signal/noise) < 2.: print(T.tcol("Signal ({:.2e}) to Noise ({:.2e}) Ratio ({:.2})".format(signal, noise, signal/noise), "green" ))
                elif np.abs(signal/noise) > 2. and np.abs(signal/noise) < 3.: print(T.tcol("Signal ({:.2e}) to Noise ({:.2e}) Ratio ({:.2})".format(signal, noise, signal/noise), "yellow" ))
                elif np.abs(signal/noise) > 3. and np.abs(signal/noise) < 4.: print(T.tcol("Signal ({:.2e}) to Noise ({:.2e}) Ratio ({:.2})".format(signal, noise, signal/noise), "red" ))
                elif np.abs(signal/noise) > 4.: print(T.tcol("Signal ({:.2e}) to Noise ({:.2e}) Ratio ({:.2})".format(signal, noise, signal/noise), "purple" ))
                else: print("Signal ({:.2e}) to Noise ({:.2e}) Ratio ({:.2})".format(signal, noise, signal/noise))
            STN.append(signal/noise)
        STN = np.array(STN)

        if do_plot:
            fig = plt.figure()
            plt.title("Position: ({}, {})".format(i, j))
            ax = fig.add_subplot(111)
            ax.hist(STN, np.linspace(-5, 5, 50))
            ax.grid(True, linewidth=.1)
            ax.set_xlabel("S/N")
            plt.show(block=False)

            fig = plt.figure()
            ax1 = fig.add_subplot(311)
            ax1.plot(signals)
            ax1.set_ylabel('Signal')
            ax1.grid(True, linewidth=.1)

            ax2 = fig.add_subplot(312)
            ax2.plot(noises)
            ax2.set_ylabel('Noise')
            ax2.grid(True, linewidth=.1)

            ax3 = fig.add_subplot(313)
            ax3.plot(STN)
            ax3.set_ylabel('S/N')
            ax3.set_xlabel('Frequency')
            ax3.grid(True, linewidth=.1)
            ax3.set_ylim([-5, 5])
            
            plt.tight_layout()
            plt.show(block=False)
        
        # We can now search for candidates.
        STN_threshold = 4
        joint_bool = np.abs(STN) > STN_threshold
        candidates = []
        if joint_bool.any():
            isSame = False
            start = 0
            # stop = 0
            for i, b in enumerate(joint_bool):
                if b:
                    if not isSame:
                        start = i
                        isSame = True
                else:
                    if isSame:
                        candidates.append(slice(start*self._binning, i*self._binning))
                        start = 0
                        isSame = False  

        # We return the source candidates.  
        return candidates

    
    # With this method, one can fit the source with different flux profiles. 
    # The goal of this method is to be as generic as possible, and allows to
    # give a flux profile as parameter.
    # Note: This source fitter is the most generic one. It will make discrete variables to be continuous.
    #       If this doesn't please you, please use self.fit_source.
    # PARAMETERS
    #  - source/SourceCandidate: is the source canditates we try to fit.
    #  - do_plot/bool: is the usual boolean that tells if we want to plot the result of not (mainly for debugging)
    #  - profile/func: is a flux profile function
    #  - params/ordered dictionary: is a Parameters class that contains all the parameters.
    #    It's an ordered dictionary where each element is a list in the following format:
    #    {'name': [init_value, min, max]}
    def fit_source_continuous(self, source, do_plot, profile, params):
        tic = time.time()

        # for clarity
        plotdf = self.plotdf
        df = self.df
        binning = self.binning
        debug = self.debug

        # First, we need to retrieve all the needed information from the flux.
        i,j,f = int(source.x()), int(source.y()), int(source.z())
        flux = self._get_line(i, j)
        flux_binned = self._rebinning(flux)

        # Next, we need to keep only the windows of interest.
        # windowSize = 2*df
        startf, stopf = self._frequency_boundaries(f-plotdf, f+plotdf, flux.shape[0])
        startf, stopf = startf//binning, stopf//binning
        freq = np.array(range(startf, stopf, 1))

        int_profile = lambda x, *args : [quad(profile, val, val+1, args=args)[0] for val in x]
        
        LSQ = lambda *args : np.sum((flux_binned[startf:stopf] - profile(freq, *args)) ** 2 / flux_binned[startf:stopf] ** 2)
        # LSQ = lambda *args : chisquare(flux_binned[startf:stopf], profile(freq, *args))[0]
        
        # cost = sum( (y[i] - f(x[i]) )^2/y[i]^2)
        # diff = lambda x, A, w, f0: np.abs(int_profile(x, A, w, f0) - flux_binned[x-startf])
        p0 = []
        limits = []
        for item in params.items():
            p0.append(item[1][0])
            limits.append((item[1][1], item[1][2]))
        print(p0, limits)

        # We obtain the name of the parameters in the function.
        params_name = inspect.getfullargspec(profile)[0]

        # We remove the parameters matching 'self'
        for idx, el in enumerate(params_name):
            if el is 'self':
                params_name.pop(idx)
        params_name.pop(0) # to remove the data parameter (x) 
        if debug: print("profile parameters' name: {}".format(params_name))

        # popt, _ = curve_fit(int_profile, freq, flux_binned[startf:stopf], p0=tuple(p0), bounds=(tuple(mins), tuple(maxs)), method='trf')
        # print(popt)

        # We setup the basic minuit class.
        m = Minuit(LSQ, *p0)
        print(limits)
        m.limits = limits
        # m.values = [12, 12, 12]
        fit = m.migrad()
        
        print(m.values)
        print(m.limits)

        toc = time.time()
        # print(fit.values)
        print("Fitting time: {}".format(toc-tic))

        if do_plot:
            var = []
            titleStr = ""
            counter = 0
            for key, value in params.items():
                var.append(m.values[counter])
                titleStr += "{}={:.2e} ".format(key, m.values[counter])
                counter += 1
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(freq*binning, np.append(freq*binning, freq[-1]*binning + binning), weights=flux_binned[startf:stopf], alpha=0.5, label="data")
            freq_plot = np.linspace(startf, stopf, 500)
            ax.plot(freq_plot*binning, profile(freq_plot, *m.values), label="fit, {:.2e}s".format(toc-tic))
            ax.plot(freq_plot*binning, self.square_hat_profile(freq_plot, m.values[0], source.w20_pix(self.domain.dz), source.z()), label="truth catalogue")
            ax.legend()
            ax.grid(True, linewidth=.1)
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Flux [Jy/beam]")
            plt.title(titleStr)
            plt.tight_layout()
            plt.show(block=False)



    # With this method, one can fit the source with different flux profiles. 
    # The goal of this method is to be as generic as possible, and allows to
    # give a flux profile as parameter.
    # Note: This source fitter is in the case self.fit_source_continuous doesn't work.
    # PARAMETERS
    #  - source/SourceCandidate: is the source canditates we try to fit.
    #  - do_plot/bool: is the usual boolean that tells if we want to plot the result of not (mainly for debugging)
    #  - profile/func: is a flux profile function
    #  - params/lmf.Parameters: is a Parameters class that contains all the parameters. The documentation
    #    is available here: https://lmfit.github.io/lmfit-py/parameters.html 
    def fit_source(self, source, do_plot, profile, params):
        whole_tic = time.time()

        # for clarity
        plotdf = self.plotdf
        df = self.df
        binning = self.binning
        debug = self.debug

        # First, we need to retrieve all the needed information from the flux.
        i,j,f = int(source.x()), int(source.y()), int(source.z())
        flux = self._get_line(i, j)
        flux_binned = self._rebinning(flux)

        # Next, we need to keep only the windows of interest.
        # windowSize = 2*df
        print(f-plotdf, f+plotdf, flux_binned.shape[0], flux.shape[0])
        startf, stopf = self._frequency_boundaries((f-plotdf), (f+plotdf), flux.shape[0])
        startf, stopf = startf//binning, stopf//binning
        print(startf, stopf)
    
        #windowSize = 2*df // binning
        #startf, stopf = f//binning-plotdf, f//binning+plotdf
        freq = np.array(range(startf, stopf, 1))

        # We do the fit in three passes.
        model = lmf.Model(profile)
        if debug: print('parameter names: {}'.format(model.param_names))
        if debug: print('independent variables: {}'.format(model.independent_vars))

        tic1 = time.time()
        # 1. Least-square fit on the non-brute force parameters, to have an initial guess on their value.
        # First we chose which parameters will be varied
        if debug: print("1st pass")
        for key, value in params.valuesdict().items():
            if params[key].brute_step == None:
                params[key].set(vary=True)
            else:
                params[key].set(vary=False)
        if debug: print(params.pretty_print())
        
        # We fit with least square.
        print("LS shape: {}, {}".format(flux_binned[startf:stopf].shape, freq.shape))
        # print(startf, stopf, flux_binned)
        fit_params_pass1 = model.fit(flux_binned[startf:stopf], x=freq, params=params, method='leastsq')
        if debug: print(fit_params_pass1.fit_report())

        # We update the parameters that must be updated.
        new_params = fit_params_pass1.params
        for key, value in new_params.valuesdict().items():
            if params[key].vary:
                params[key].set(value=new_params[key].value)
        toc1 = time.time()
        
        tic2 = time.time()
        # 2. Grid-search on the parameters that have brute_step defined.
        # First we chose which parameters will be varied
        if debug: print("2nd pass")
        for key, value in params.valuesdict().items():
            if params[key].brute_step == None:
                params[key].set(vary=False)
            else:
                params[key].set(vary=True)
        if debug: print(params.pretty_print())

        # We fit with grid search.
        fit_params_pass2 = model.fit(flux_binned[startf:stopf], x=freq, params=params, method='brute')
        if debug: print(fit_params_pass2.fit_report())

        # We update the parameters that must be updated.
        new_params = fit_params_pass2.params
        for key, value in new_params.valuesdict().items():
            if params[key].vary:
                params[key].set(value=new_params[key].value)
        toc2 = time.time()
        
        tic3 = time.time()
        # 3. Least-square fit again on all parameters.    
        # First we vary all the parameters.  
        if debug: print("3rd pass")
        for key, value in params.valuesdict().items():
            if params[key].brute_step == None:
                params[key].set(vary=True)
            else:
                params[key].set(vary=False)
            # params[key].set(vary=True)
        if debug: print(params.pretty_print())

        # We fit with least square.
        fit_params_pass3 = model.fit(flux_binned[startf:stopf], x=freq, params=params, method='leastsq')
        if debug: print(fit_params_pass3.fit_report())
        toc3 = time.time()
        
        whole_toc = time.time()

        print("Elapsed time pass 1: {}s".format(toc1-tic1))
        print("Elapsed time pass 2: {}s".format(toc2-tic2))
        print("Elapsed time pass 3: {}s".format(toc3-tic3))
        print("Elapsed time total: {}s".format(whole_toc-whole_tic))
        # We can plot it, if needed
        if do_plot:
            var = []
            titleStr = ""
            for key, value in fit_params_pass3.params.valuesdict().items():
                if fit_params_pass3.params[key].brute_step == None:
                    var.append(value)
                    titleStr += "{}={:.2e} ".format(key, value)
                else:
                    var.append(int(value))
                    titleStr += "{}={} ".format(key, int(value))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(freq*binning, np.append(freq*binning, (freq[-1]*binning + binning)), weights=flux_binned[startf:stopf], alpha=0.5, label="data")
            # ax.plot(freq, flux_binned[startf:stopf], label="data")
            freq_plot = np.linspace(startf, stopf, 500)
            ax.plot(freq_plot*binning, self.square_hat_profile(freq_plot, *var), label="fit, {:.2e}s".format(whole_toc-whole_tic))
            ax.plot(freq_plot*binning, self.square_hat_profile(freq_plot, var[0], source.w20_pix(self.domain.dz), source.z()), label="truth catalogue")
            ax.legend()
            ax.grid(True, linewidth=.1)
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Flux [Jy/beam]")
            plt.title(titleStr)
            plt.tight_layout()
            plt.show(block=False)

        # Finally we return the result.
        print("Final parameters: ", fit_params_pass3.params)
        print("chi2: {}".format(fit_params_pass3.chisqr))
        return fit_params_pass3.params



    # This method computes the ratio between the 
    def STN_likelihood_ratio_hat(self, source, do_plot, params):
        binning = self.binning
        # First, we start by fitting the source.
        fitted_params = self.fit_source(source, do_plot, self.square_hat_profile, params)
        print(fitted_params)

        # from these three parameters, we'll compute the source likelihood, and we return the likelihoods.
        Lsb = self.source_lkh_hat(source, self.square_hat_profile, fitted_params)
        Lb = self.bg_lkh_hat(source, self.square_hat_profile, fitted_params)
        print("Lsb: {}".format(Lsb))
        print("Lb: {}".format(Lb))
        print("Lsb/Lb: {}".format(Lsb/Lb))
        
        return Lsb/Lb

    def source_lkh_hat(self, source, profile, fitted_params):
        binning = self.binning

        # First, we get the flux informations.
        i,j,f = int(source.x()), int(source.y()), int(source.z())
        flux = self._get_line(i, j)
        flux_binned = self._rebinning(flux)
        model = lmf.Model(profile)
        # print(int(f - fitted_params['w'].value/2), int(f + fitted_params['w'].value/2))
        startf, stopf = 1 + int(f - fitted_params['w'].value/2)//binning, int(f + fitted_params['w'].value/2)//binning
        # startf, stopf = self._frequency_boundaries(int(f - fitted_params['w'].value/2)//binning, int(f + fitted_params['w'].value/2)//binning, flux.shape[0])

        # We define the modified noise parameters.
        # print(self._noise_params.shape, self._noise_params)
        freq = np.arange(startf, stopf)
        print(freq)
        self._noise_params[startf:stopf, 0] += self.square_hat_profile(freq, fitted_params['A'].value, fitted_params['w'].value, fitted_params['f0'].value)

        # We compute all the likelihoods.
        joint_lkh = self.joint_log_likelihood(flux, startf*binning, stopf*binning, binning = None)

        # We come back to the unmodified noise.
        self._noise_params[startf:stopf, 0] -= self.square_hat_profile(freq, fitted_params['A'].value, fitted_params['w'].value, fitted_params['f0'].value)

        # We return the product of all the likelihoods.
        # return np.prod(lkh)
        return joint_lkh


    def bg_lkh_hat(self, source, profile, fitted_params):
        binning = self.binning

        # First, we get the flux informations.
        i,j,f = int(source.x()), int(source.y()), int(source.z())
        flux = self._get_line(i, j)
        flux_binned = self._rebinning(flux)
        model = lmf.Model(profile)
        startf, stopf = 1 + int(f - fitted_params['w'].value/2)//binning, int(f + fitted_params['w'].value/2)//binning
        
        # We compute all the likelihoods.
        joint_lkh = self.joint_log_likelihood(flux, startf*binning, stopf*binning, binning = None)

        # We return the product of all the likelihoods.
        return joint_lkh

        

    # This function returns a rectangular profile with width w, amplitude A and centre f0.
    # PARAMETERS:
    #  - x/int: is the input flux for the SQP. Note that we use variable 'x' in every profile for generality.
    #  - A/float: is the amplitude of the SQP.
    #  - w/int: is the width of the SQP.
    #  - f0/int: is the centre of the SQP
    # @staticmethod
    def square_hat_profile(self, x, A, w, f0):
        # print(x, A, w, f0)
        binning = self.binning
        mask = np.abs(x - f0/binning) < w/binning/2
        # mask = np.abs(x - f0//binning) < w//binning/2 # for discrete.
        result = np.zeros(mask.shape)
        result[mask] = A
        return result

    # just returns a random source with line integral of flux higher than threshold.
    # PARAMETERS:
    #  - threshold/float: is the limit of flux line integral we want to consider.
    def rand_truth_source(self, threshold):
        source_list = TruthSource.catalog_to_sources_in_domain(self.filepath_truth, self.domain)
        source_list = [s for s in source_list if s.line_flux_integral() > threshold]
        print("Source size: {}".format(len(source_list)))
        if len(source_list) == 1:
            return source_list[0]
        elif len(source_list) == 0:
            return None
        return source_list[random.randint(0, len(source_list)-1)]
    
    # returns the source associated to an ID.
    #  - ID/int: is the ID of the source...
    def source_from_id(self, ID):
        source_list = TruthSource.catalog_to_sources_in_domain(self.filepath_truth, self.domain)
        for s in source_list:
            if s.ID() == ID:
                return s    

    # just returns a random noise line position, far enough from any source.
    def rand_noise(self):
        margin = self.lkh_threshold_margin
        source_list = TruthSource.catalog_to_sources_in_domain(self.filepath_truth, self.domain)
        while True:
            cont = False
            i = random.randint(self.domain.xmin+self.domain.border, self.domain.xmax-self.domain.border-1)
            j = random.randint(self.domain.ymin+self.domain.border, self.domain.ymax-self.domain.border-1)
            for s in source_list:
                if abs(s.x() - i) < margin and abs(s.y() - j) < margin:
                    cont = True
                    print("Line ({},{}) is too close to ({}, {})".format(i,j,s.x(),s.y()))
            if cont: continue
            print(T.tcol("Line ({},{}) is far from everyone.".format(i,j), "green"))
            return [i, j]

    # This function makes a roc curve. Note: it only works with the likelihood test.
    # PARAMETERS:
    #  - flux_threshold/float: is the limit of flux line integral we want to consider.
    #  - nbSources/int: is the number of sources (and noise samples) we want to consider.
    #  - nbThresholds/int: is the number of threshold values we want to try.
    #  - logThreshLim/[float, float]: is the log-likelihood boundaries we want to consider for the search.
    #    In the end, the likelihood will be distrubited uniformely, in log scale, between these values.
    #  - do_plot/bool: is the boolean that controls if a plot if wanted or not.
    #  - manualData/np.array: is the variable to set if data was already computed, which skips the computation part.
    #    When "None" (default), the data will be computed.
    def roc_curve(self, flux_threshold, nbSources = 10, nbThresholds = 10, logThreshLim = [5, 20],  do_plot = True, manualData = None):
        # for clarity
        df = self.df
        binning = self.binning
        size = self.size
        rank = self.rank
        comm = self.comm
        debug = self.debug

        flux_bins = [0, 20, 60, 100, 999]
        thresholds = np.linspace(logThreshLim[0], logThreshLim[1], nbThresholds)
        if manualData is None:
            # First, we get some sources
            sources = []
            noises = []
            for i in range(len(flux_bins)-1):
                # We first choose the sources.
                source_list = TruthSource.catalog_to_sources_in_domain(self.filepath_truth, self.domain)
                source_list = [s for s in source_list if  flux_bins[i+1] > s.line_flux_integral() > flux_bins[i]]
                if len(source_list) <= nbSources:
                    sources.append(source_list)
                else:
                    random_choice = np.random.choice(source_list, nbSources).tolist()
                    sources.append(random_choice)
                # then the noises
                n = []
                for i in range(len(sources[-1])):
                    n.append(self.rand_noise())
                noises.append(n) 
            # Now, we try to construct the roc curve.
            TPs = []
            FPs = []
            Ps = []
            Fs = []
            for src_idx, src in enumerate(sources):
                TP_all = []
                FP_all = []
                P_all = []
                F_all = []
                for t in thresholds:
                    # We initialise
                    TP = 0
                    FP = 0
                    P = 0
                    F = 0

                    # We choose the current threshold
                    self.log_lkh_threshold = t

                    # We iterate over all sources and noises to see how it goes.
                    for idx in range(len(src)):
                        # Sources
                        s = src[idx]
                        i, j = int(s.x()), int(s.y())
                        
                        # we try to find the source 
                        cand = self.find_source(i, j, False)

                        # We see if there is a match or not.
                        if cand:
                            if cand[0].matching(src[idx]):
                                # print("Match!")
                                TP += 1
                        P += 1

                        # Noises
                        i, j = noises[src_idx][idx]
                        cand = self.find_source(i, j, False)
                        # print(cand)
                        if cand:
                            FP += 1
                        F += 1
                    
                    TP_all.append(TP)
                    FP_all.append(FP)
                    P_all.append(P)
                    F_all.append(F)

                TPs.append(np.array(TP_all))
                FPs.append(np.array(FP_all))
                Ps.append(np.array(P_all))
                Fs.append(np.array(F_all))
            
            TPs = np.array(TPs)
            FPs = np.array(FPs)
            Ps = np.array(Ps)
            Fs = np.array(Fs)

            if size > 1:
                TPs_gather = comm.gather(TPs, root=0)
                FPs_gather = comm.gather(FPs, root=0)
                Ps_gather = comm.gather(Ps, root=0)
                Fs_gather = comm.gather(Fs, root=0)
                if rank == 0:
                    TPs = np.sum(TPs_gather, axis=0)
                    FPs = np.sum(FPs_gather, axis=0)
                    Ps = np.sum(Ps_gather, axis=0)
                    Fs = np.sum(Fs_gather, axis=0)   
            
            if rank == 0:
                # We save everything.
                idx = 0
                filename = self.outdir + "roc_{}_idx-{}.npz".format(self.domainType(), idx)
                while os.path.isfile(filename):
                    idx+=1
                    filename = self.outdir + "roc_{}_idx-{}.npz".format(self.domainType(), idx)
                np.savez(filename, TPs=TPs, FPs=FPs, Ps=Ps, Fs=Fs)
                print(T.tcol("DATA SAVED IN: {}".format(filename), "red"))         

        if rank == 0:
            if manualData is not None:
                data = np.load(manualData)
                TPs = np.array(data['TPs'])
                FPs = np.array(data['FPs'])
                Ps  = np.array(data['Ps'])
                Fs  = np.array(data['Fs'])

                if debug: print("TP: {}".format(TP))
                if debug: print("P: {}".format(P))
                if debug: print("FP: {}".format(FP))
                if debug: print("F: {}".format(F))

            if do_plot:
                # We can now compute the TPR and FPR.
                TPR = []
                FPR = []
                for i in range(len(flux_bins)-1):
                    TPR.append(TPs[i]/Ps[i])
                    FPR.append(FPs[i]/Fs[i])
                FPR = np.array(FPR)
                TPR = np.array(TPR)
                print(FPR.shape, TPR.shape)

                # we prepare the colours for the scatter plot.
                cmap = plt.get_cmap('brg')
                colour_idx = np.linspace(0, 1, TPR[i].shape[0])
                colours = cmap(colour_idx)

                # Now the markers
                markers = ["o", "v", "s", "*"]

                # Now we have all the information we want, we can plot the roc curve.
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.title("Roc curve, bin={}".format(binning))
                for i in range(len(TPR)):
                    ax.scatter(FPR[i], TPR[i], label="{} < val < {}".format(flux_bins[i], flux_bins[i+1]), c=colours, cmap='jet', marker=markers[i])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.colorbar(cm.ScalarMappable(norm=col.Normalize(vmin=thresholds[0], vmax=thresholds[-1]), cmap=cmap), ax=ax, label="Log-likelihood threshold")
                plt.grid(True, linewidth=.1)
                plt.legend()
                plt.tight_layout()
                plt.show(block=False)

    # This function makes a flux line integral vs S/N ratio plot for the truth catalogue.
    # PARAMETERS:
    #  - threshold/float: is the flux threshold that we want to consider.
    #  - nbSources/int: is the number of source we want to consider in the catalogue. If this number is
    #    higher than the actual available number of sources, we just choose all of them.
    #  - data_filename/String: is the filename of the data that we want to load.
    #    If it is something else than "None", nothing will be computed.
    #  - do_plot/bool: is the boolean that tells if a plot must be done or not. Note that by default,
    #    a plot will be done.
    def flux_vs_STN(self, threshold, nbSources, data_filename = None, do_plot = True):
        # for clarity
        size = self.size
        rank = self.rank
        comm = self.comm

        if data_filename is None:
            # We gather all the informations.
            # First, we load all the sources we want.
            source_list = TruthSource.catalog_to_sources_in_domain(self.filepath_truth, self.domain)
            source_list = [s for s in source_list if s.line_flux_integral() > threshold]

            # We select only the number we are interessed in.
            if len(source_list) <= nbSources or nbSources == -1:
                sources = source_list
            else:
                sources = random.sample(source_list, nbSources)

            # for each source, we obtain the information we need.
            X = [] # STN
            Y = [] # flux line integral
            Z = [] # w20 (for colour)
            ID = [] # Contains the ID of each source.
            old_binning = self.binning
            self.binning = 1
            print("Number of sources: {}".format(len(sources)))
            for s in sources:
                # print(s)
                Y.append(s.line_flux_integral())
                Z.append(s.w20()) # km/s
                ID.append(s.ID())
                x, y, z = int(s.x()), int(s.y()), int(s.z())
                x_low, x_high, y_low, y_high, z_low, z_high = s.cube_indices(self.domain.dy, self.domain.dz)
                # print(x_low, x_high, y_low, y_high, z_low, z_high)
                # print(x, y, z)
                
                # signal = s.line_flux_integral() # To complicated to
                # flux = self._get_line(x, y)
                flux = self.domain.safe_get_cube(x, x+1, y, y+1, z_low, z_high)
                # print(flux.shape)
                binSizeHz = self.domain.ax3[z_high] - self.domain.ax3[z_low]
                # print(self.domain.ax3[z_high] - self.domain.ax3[z_low], self.domain.ax3[z_high-z_low] - self.domain.ax3[0])
                # print(flux)
                signal = np.sum(flux)# * binSizeHz
                # signal = np.sum(flux[z_low:z_high]) * binSizeHz
                noise = np.linalg.norm(self._noise_params[z_low:z_high,1])
                # print("signal / noise = {} / {}".format(signal, noise))
                # We need to convert the noise from Jy/beam into Jy. Jy = (Jy * pixel / beam) / (pixel / beam)
                STN = signal/noise

                X.append(STN)

            X = np.array(X)
            Y = np.array(Y)
            Z = np.array(Z)
            ID = np.array(ID)

            if size > 1:
                X_gather = np.array(comm.gather(X, root=0))
                Y_gather = np.array(comm.gather(Y, root=0))
                Z_gather = np.array(comm.gather(Z, root=0))
                ID_gather = np.array(comm.gather(ID, root=0))
                if rank == 0:
                    X = np.stack(np.concatenate(X_gather))
                    Y = np.stack(np.concatenate(Y_gather))
                    Z = np.stack(np.concatenate(Z_gather))
                    ID = np.stack(np.concatenate(ID_gather))
            
            if rank == 0:
                # saving data.
                filename = None
                f_idx = 0
                while filename is None:
                    name = 'data/F_to_STN_{}.npz'.format(f_idx)
                    if os.path.exists(name):
                        f_idx += 1
                    else:
                        filename = name
                        np.savez(name, X=X, Y=Y, Z=Z, ID=ID)   
                        print("DATA WRITTEN IN: {}".format(name)) 
            self.binning = old_binning
        else:
            if rank == 0:
                data = np.load(data_filename, allow_pickle=True)
                X = data['X']
                Y = data['Y']
                Z = data['Z']
                ID = data['ID']
            
        #print(T.tcol("[{}, {}, {}]".format(np.array2string(X, separator=', '), np.array2string(Y, separator=', '), np.array2string(Z, separator=', ')), "red"))

        # We come back to old binning. 
        

        if size == 1:
            if do_plot:
                # plots flux line integral vs S/N
                # we prepare the colours for the scatter plot.
                cmap = plt.get_cmap('brg')
                colour_idx = np.linspace(0, 1, Z.shape[0])
                colours = cmap(colour_idx)

                # Now the markers
                # markers = ["o", "v", "s", "*"]

                # Now we have all the information we want, we can plot the roc curve.
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(X[:], Y[:], c=colours, cmap='jet', marker='.')
                plt.xlabel("Signal to noise ratio")
                plt.ylabel("Flux line integral [Jy Hz]")
                # plt.xscale('log')
                # plt.colorbar(cm.ScalarMappable(norm=col.LogNorm(vmin=np.min(Z), vmax=np.max(Z)), cmap=cmap), ax=ax, label="w20 [km/s]")
                plt.colorbar(cm.ScalarMappable(norm=col.LogNorm(vmin=np.min(Z), vmax=np.max(Z)), cmap=cmap), ax=ax, label="w20 [km/s]")
                plt.grid(True, linewidth=.1)
                # plt.legend()
                plt.tight_layout()
                plt.show(block=False)
        
        if rank == 0:
            # We return the filename where the computed data has being saved.
            return filename


    # This function makes a roc curve, but with S/N instead of likelihood for the separation.
    # Note: it only works with the likelihood test.
    # PARAMETERS:
    #  - STN_threshold/float: is the lower limit of flux line integral we want to consider.
    #  - nbSources/int: is the number of sources (and noise samples) we want to consider.
    #  - nbThresholds/int: is the number of threshold values we want to try.
    #  - logThreshLim/[float, float]: is the log-likelihood boundaries we want to consider for the search.
    #    In the end, the likelihood will be distrubited uniformely, in log scale, between these values.
    #  - source_info_filename/String: is the location of the STN data. Please first run 
    #  - do_plot/bool: is the boolean that controls if a plot if wanted or not.
    #  - manualData/String: is the variable to set if data was already computed, which skips the computation part.
    #    When "None" (default), the data will be computed.
    def roc_curve_STN(self, STN_threshold, source_info_filename = None, nbSources = 10, nbThresholds = 10, logThreshLim = [5, 20], do_plot = True, STN_bins = None, manualData = None):
        # for clarity
        df = self.df
        binning = self.binning
        size = self.size
        rank = self.rank
        comm = self.comm
        debug = self.debug

        if source_info_filename is None:
            source_info_filename = self.flux_vs_STN(0, -1)

        if STN_bins is None:
            STN_bins = [0, 2, 5, 10, 999]
        thresholds = np.linspace(logThreshLim[0], logThreshLim[1], nbThresholds)

        if manualData is None:
            # First, we get some sources
            sources = []
            noises = []
            source_list = TruthSource.catalog_to_sources_in_domain(self.filepath_truth, self.domain)
            source_list = [s for s in source_list if s.line_flux_integral()]
            if nbSources is -1 or len(source_list) <= nbSources:
                sources = source_list
            else:
                sources = np.random.choice(source_list, nbSources).tolist()
            
            # Then the noise samples.
            for i in range(len(sources)):
                noises.append(self.rand_noise())
            noises = np.array(noises)

            print(T.tcol("In task {} out of {}, we have: {} sources".format(rank, size, len(sources)), "blue"))
            
            
            # Now, we try to construct the roc curve.
            TPs = []
            FPs = []
            for src_idx, s in enumerate(sources):
                # For each source, we test each threshold, and if we find a matching source, set to True, and if not, False.
                TP_thresh = []
                FP_thresh = []
                for t in thresholds:
                    self.log_lkh_threshold = t

                    # Source
                    i, j = int(s.x()), int(s.y())
                        
                    # we try to find the source 
                    cand = self.find_source(i, j, False)
                    if debug: print("Is it matching ?")
                    if cand:
                        TP_thresh.append(cand[0].matching(s))
                        if debug: print(TP_thresh[-1])
                    else:
                        TP_thresh.append(False)
                        if debug: print(False)

                    # noise
                    i, j = noises[src_idx]
                    cand = self.find_source(i, j, False)

                    if cand:
                        FP_thresh.append(True)
                    else:
                        FP_thresh.append(False)
                TPs.append(TP_thresh)
                FPs.append(FP_thresh)
            TPs = np.array(TPs)
            FPs = np.array(FPs)
            if debug: print(TPs.shape, FPs.shape)
            if debug: print("TPs: {}".format(TPs))
            if debug: print("FPs: {}".format(FPs))

            # Now that we search for sources in all zone of interessed, we can organise the data by STN.
            # For this, we'll use masks.
            # First, load the relevant data, and put them in a dictionary to easily access them.
            IDs = []
            loaded_data = np.load(source_info_filename)
            STN_IDS = {}
            for ids, stn in zip(loaded_data['ID'], loaded_data['X']):
                STN_IDS[ids] = stn
            
            # Now, we classify all the sources in the relevant category.
            sources_categories = np.zeros(len(sources))
            for idx, s in enumerate(sources):
                for i in range(len(STN_bins) - 1):
                    if STN_bins[i] <= STN_IDS[s.ID()] < STN_bins[i+1]:
                        sources_categories[idx] = i
            sources_categories = np.array(sources_categories)

            # And now we can simply create the masks for each category.
            masks = []
            for i in range(len(STN_bins) - 1):
                masks.append(sources_categories == i)
            
            # From this, we can now compute the ROC data that we want, for each category.
            # sources
            TP = []
            P = []
            for mask in masks:
                # each sub array contains the result of one source for different thresholds.
                # [[T, F, ...], [T, T, F, ...]]
                tp = []
                p = []
                for i in range(len(thresholds)):
                    if TPs[:,i][mask].any():
                        tp.append( np.sum(TPs[:,i][mask]) )
                        print(TPs[:,i][mask].shape[0])
                        p.append( TPs[:,i][mask].shape[0] )
                    else:
                        tp.append(0)
                        p.append(TPs[:,i][mask].shape[0])
                TP.append(tp)
                P.append(p)

            # noise
            FP = []
            F = []
            for i in range(len(thresholds)):
                FP.append( np.sum(FPs[:,i]) )
                F.append( FPs[:,i].shape[0] )
            
            TP = np.array(TP)
            FP = np.array(FP)
            P = np.array(P)
            F = np.array(F)

            if size > 1:
                TPs_gather = comm.gather(TP, root=0)
                FPs_gather = comm.gather(FP, root=0)
                Ps_gather = comm.gather(P, root=0)
                Fs_gather = comm.gather(F, root=0)
                if rank == 0:
                    TP = np.sum(TPs_gather, axis=0)
                    FP = np.sum(FPs_gather, axis=0)
                    P = np.sum(Ps_gather, axis=0)
                    F = np.sum(Fs_gather, axis=0)
            if rank == 0:
                # We save everything.
                idx = 0
                filename = self.outdir + "roc_STN_{}_idx-{}.npz".format(self.domainType(), idx)
                while os.path.isfile(filename):
                    idx+=1
                    filename = self.outdir + "roc_STN_{}_idx-{}.npz".format(self.domainType(), idx)
                np.savez(filename, TP=TP, FP=FP, P=P, F=F)
                print(T.tcol("DATA SAVED IN: {}".format(filename), "red"))
                

        if rank == 0:
            if manualData is not None:
                data = np.load(manualData)
                TP = np.array(data['TP'])
                FP = np.array(data['FP'])
                P  = np.array(data['P'])
                F  = np.array(data['F'])

            if debug: print("TP: {}".format(TP))
            if debug: print("P: {}".format(P))
            if debug: print("FP: {}".format(FP))
            if debug: print("F: {}".format(F))
            # print(T.tcol("[{},\n {},\n {},\n {}]".format(np.array2string(TP, separator=', '), np.array2string(FP, separator=', '), np.array2string(P, separator=', '), np.array2string(F, separator=', ')), "red"))

            if do_plot:
                # We can now compute the TPR and FPR.
                TPR = []
                for i in range(len(TP)):
                    TPR.append( TP[i]/P[i] )
                TPR = np.array(TPR)
                FPR = np.array( FP/F )
                # print(FPR, TPR)
                print(FPR.shape, TPR.shape)

                # we prepare the colours for the scatter plot.
                cmap = plt.get_cmap('brg')
                colour_idx = np.linspace(0, 1, TPR[i].shape[0])
                colours = cmap(colour_idx)

                # Now the markers
                markers = ["o", "v", "s", "p", "+", "x", "8"]

                # Now we have all the information we want, we can plot the roc curve.
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.title("Roc curve, bin={}".format(binning))
                for i in range(len(TPR)):
                    ax.scatter(FPR, TPR[i], label="{} < STN < {}".format(STN_bins[i], STN_bins[i+1]), c=colours, cmap='jet', marker=markers[i])
                    # ax.plot(FPR, TPR[i], linewidth=.5, c=(0.0, 0.0, 0.0, 1.0)) # to link data, if needed
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.colorbar(cm.ScalarMappable(norm=col.Normalize(vmin=thresholds[0], vmax=thresholds[-1]), cmap=cmap), ax=ax, label="Log-likelihood threshold")
                plt.grid(True, linewidth=.1)
                plt.legend()
                plt.tight_layout()
                plt.show(block=False)

    # Will plot an histogram for the STN of the sources.
    # PARAMETERS:
    #  - stn_path/String: is the path to the catalogue containing all the revelant STN.
    def sources_distribution_STN(self, stn_path):
        # for clarity
        binning = self.binning

        # we first load the data.
        loaded_data = np.load(stn_path)
        STN = []
        counter = 0
        for ids, stn in zip(loaded_data['ID'], loaded_data['X']):
            if stn < 20:
                STN.append(stn)
            else:
                counter += 1
        STN = np.array(STN)

        print("{} sources with STN > 20 were omitted.".format(counter))

        # We plot the distribution.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title("STN distribution, bin={}".format(binning))
        plt.hist(STN, 20)
        plt.xlabel("STN")
        plt.ylabel("Number of sources")
        plt.grid(True, linewidth=.1)
        plt.tight_layout()
        plt.show(block=False)
    
    # Will plot an histogram for the FLI of the sources.
    def sources_distribution_FLI(self):
        # for clarity
        binning = self.binning

        # we load the sources
        source_list = TruthSource.catalog_to_sources_in_domain(self.filepath_truth, self.domain)
        source_list = [s for s in source_list if s.line_flux_integral()]

        FLI = []
        counter = 0
        for s in source_list:
            if s.line_flux_integral() < 100:
                FLI.append(s.line_flux_integral())
            else: counter+=1

        FLI = np.array(FLI)

        print("{} sources with FLI >= 100 were omitted.".format(counter))

        # We plot the distribution.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title("distribution of flux line integral, bin={}".format(binning))
        plt.hist(FLI, 20)
        plt.xlabel("Flux Line Integral [Jy Hz]")
        plt.ylabel("Number of sources")
        plt.grid(True, linewidth=.1)
        plt.tight_layout()
        plt.show(block=False)
