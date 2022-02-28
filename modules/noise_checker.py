
"""
A module for studying noise
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from astropy import modeling

class NoiseChecker:

    """
    check how Gaussian the noise is in each frequency slice
    """
    def check_noise(self,reader):
        start_time =  time.process_time()
        domain = reader.sky_domain[:100,:,:]
        maxdiff = -9999
        maxf = -1

        for f in range(domain.shape[0]):
            diff = self.check_noise_in_slice(f, domain)
            if maxdiff < diff:
                maxdiff = diff
                maxf = f
        print(maxf, diff )

        print("Runtime: {0}".format( time.process_time() - start_time))
        ftag = "Freq. slice f = {0} MHz".format(reader.ax3[maxf]/1e6)
        print("Show noise {}".format(ftag))
        self.check_noise_in_slice(maxf, domain,  ftag = ftag,show =True)

    def check_noise_in_slice(self, f, domain, ftag = None, show = False):
  
        #hist_y, hist_x, _ = plt.hist(domain[-100,:,:].flatten(), bins = 100)
        hist, bin_edges = np.histogram(domain[f,:,:],100)
        bin_centers = bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2. 

        # now fit
        fitter = modeling.fitting.LevMarLSQFitter()
        model = modeling.models.Gaussian1D(amplitude=100000, mean=0, stddev=1e-4) 
        fitted_model = fitter(model, bin_centers, hist)

        diff = np.mean(hist - fitted_model(bin_centers))/np.sum(hist )

        if show:
            try:
                plt.figure()
                plt.plot(bin_centers, hist,'.', label = "development data")
                plt.plot(bin_centers, fitted_model(bin_centers), label = "Gaussian fit")
                plt.xlabel("flux")
                plt.ylabel("counts")
                if ftag:
                    plt.text(0.05,0.8,ftag,transform=plt.gca().transAxes)
                plt.legend(loc='upper left')
                plt.show()
            except:
                print("Plotting not working")

        return diff
 
    def write_noise(self, reader, out_file):
        
        #self.fjitter    = np.std(reader., axis = (1,2))
        with open(out_file,'w') as f:
            f.write("index frequencyHz mean_flux stdev_flux")
            for i, freq in enumerate(reader.ax3):
                print (i, freq)
                stdev = np.std(reader.sky_domain[i,:,:])
                mean  = np.mean(reader.sky_domain[i,:,:])
                f.write("{0} {1} {2} {3}\n".format(i, freq, mean, stdev))


    def calculate_noise_cube(self, domain, out_file = None):
        (nfreq, ndec, nra) = domain.shape
        self.fjitter    = np.std(domain, axis = (1,2))

        if out_file != None:
            with open(out_file, 'wb') as f:
                np.save(f, self.fjitter )

    def load_noise_cube(self, in_file):
        with open(in_file, 'rb') as f:
            self.fjitter = np.load(f)

    def get_coefficient(self,i):
        return 1./self.fjitter[i]

