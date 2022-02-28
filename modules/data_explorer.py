import numpy as np
from modules.truth_info import TruthSource
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
import sys 
class DataExplorer:

    def __init__(self, weight_func = None, truth_filepath = None, reader = None):
        self.weight_func = weight_func
        self.truth_filepath = truth_filepath
        self.reader = reader

    def inspect_truth_sources(self, nsrc = 1):
        if self.truth_filepath == None:
            raise RuntimeError("must set truth filepath")

        for s in TruthSource.catalog_to_sources_in_domain(self.truth_filepath, self.reader):

            #if s.line_flux_integral() < 100: continue
            if s.ID() != 796: continue
            self.inspect_source(s)
            user_input = input("Press 0  to exit, any other key to continue")
            if user_input == '0': break


    def inspect_line(self,  dec_index,ra_index, do_weighting = False):

        f_indices = self.reader.HI_cube.f_indices

        f_line = self.reader.get_freq_line(ra_index, dec_index)[f_indices[0]:f_indices[-1]+1]

        w = self.weight_func(f_indices)
        if do_weighting:
            f_line = f_line*w

        plt.figure()
        plt.plot(self.reader.ax3[f_indices[0]:f_indices[-1]+1], f_line)
        plt.xlabel("frequency [Hz]")
        if do_weighting:
            plt.ylabel("noise-corrected flux")
            plt.text(0.1,0.85,"1/Sf noise correction\nRA index = {0}, Dec index = {1}".format(ra_index,dec_index),transform=plt.gca().transAxes)
        else:
            plt.ylabel("flux")
            plt.text(0.1,0.85,"No noise correction\nRA index = {0}, Dec index = {1}".format(ra_index,dec_index),transform=plt.gca().transAxes)
        plt.show()

    def inspect_source(self,source):

        i, j, k = int(source.x()), int(source.y()), int(source.z())

        print("Source ID: {0}, index: {1}, {2}, {3}".format(source.ID(),i,j,k))

        # define range in frequency
        f_indices = self.reader.HI_cube.f_indices

        fwidth = source.w20_pix(self.reader.dz)
        fw_down    = int(k - fwidth/2)
        fw_up      = int(k + fwidth/2)

        fstart = int(k - fwidth*1.5)
        fend   = int(k + fwidth*1.5)
        print(fstart, k, fend)

        # trim full range
        print("f_indices", f_indices.shape)
        f_indices = f_indices[fstart:fend]
        print("f_indices", f_indices.shape)

        maj_diameter_pix= source.hi_size_pix(self.reader.dy)


        # denoising weight array
        w = self.weight_func(f_indices)

        print("w", w.shape)

        # get relevant frequency lines
        print("f_line", (self.reader.get_freq_line(i,j)[fstart:fend]).shape)
        f_line = self.reader.get_freq_line(i,j)[fstart:fend]*w
        print("f_line", f_line.shape)
        f_neighbors =  np.array([self.reader.get_freq_line(i-1,j)[fstart:fend]*w,
                                 self.reader.get_freq_line(i+1,j)[fstart:fend]*w,
                                 self.reader.get_freq_line(i,j+1)[fstart:fend]*w,
                                 self.reader.get_freq_line(i,j-1)[fstart:fend]*w,
                                ])
        f_mean_neighbors = np.mean(f_neighbors,axis=0)
        f_stdev_neighbors = np.std(f_neighbors,axis=0)

        border = 10

        f_slice   = self.reader.get_cube(i-border, i+border, j-border, j+border, fw_down, fw_up)
        print(f_slice.shape, i-border, i+border)
        for f in np.arange(fw_down,fw_up):
            f_slice[f-fw_down,:,:] = self.weight_func(f)*f_slice[f-fw_down,:,:]
        f_sliver  = f_slice[k-fw_down,:,:]

        continuum = self.reader.get_continuum_cube(i-border, i+border, j-border, j+border)

        # plotting
        fig = plt.figure(figsize=(10,6))
        ax = [plt.subplot2grid((2, 3), (0, 1), colspan=2),
              plt.subplot2grid((2, 3), (1, 0)),
              plt.subplot2grid((2, 3), (1, 1)),
              plt.subplot2grid((2, 3), (1, 2))]

        plt.subplots_adjust(hspace=0.5, wspace = 0.9)

        nbins = int(fwidth)
        print(f_mean_neighbors.shape)
        print("Making a histogram with {0} bins from {0} voxels".format(nbins, len(f_indices)))
        fmean_up = f_mean_neighbors + f_stdev_neighbors
        fmean_down = f_mean_neighbors - f_stdev_neighbors
        #for i in range(len(f_mean_neighbors)): print(fmean_down[i], f_mean_neighbors[i], fmean_up[i])
        ax[0].hist(f_indices, bins=nbins, weights = fmean_up, color = 'paleturquoise')
        ax[0].hist(f_indices, bins=nbins, weights = fmean_down, color = 'paleturquoise')
        hist_y, hist_x, _ = ax[0].hist(f_indices, bins=nbins, weights = f_line, histtype = "step", color = 'k')

        ax[0].hist(f_indices, bins=nbins, weights = f_mean_neighbors, histtype = "step", color = 'turquoise')
        ax[0].plot([k,k], [-2*hist_y.max(),2*hist_y.max()], 'w')
        ax[0].plot([k,k], [-hist_y.max(),hist_y.max()], 'r')
        ax[0].plot([fw_down,fw_up], [0,0], 'r')
        ax[0].text(k,hist_y.max()*1.5,"central frequency", {'color': 'r'}, horizontalalignment='center')
        size_info = "HI major axis diameter = {0:.2f} pixels".format(maj_diameter_pix)
        pos_info = "Source at x = {0}, y = {1}, z = {2}".format(i,j,k)
        ax[0].text(-0.9,0.0, "{0}\n{1}\n{2}".format(pos_info,source.__str__(), size_info),transform=ax[0].transAxes)

        ax[0].set(xlabel="Z (frequency axis)", ylabel="S/N")
        ax[0].set_title('Frequency line', fontsize="9")

        img_extent = [i-border, i + border, j - border, j + border]

        im1 = ax[1].imshow( np.sum(f_slice,axis=0), cmap = 'GnBu_r', extent=img_extent, origin = 'lower')
        plt.colorbar(im1, ax = ax[1],fraction=0.046, pad=0.04)
        ax[1].set_title('S/N summed across w20\n', fontsize="9")

        im2 = ax[2].imshow( f_sliver, cmap = 'GnBu_r', extent=img_extent, origin = 'lower')
        plt.colorbar(im2, ax = ax[2],fraction=0.046, pad=0.04)
        
        ax[2].set_title('S/N in central frequency slice\n', fontsize="9")

        im3 = ax[3].imshow( np.sum(continuum,axis=0), cmap = 'GnBu_r', extent=img_extent, origin = 'lower')
        plt.colorbar(im3, ax = ax[3],fraction=0.046, pad=0.04)
        ax[3].set_title('Flux of continuum image\n', fontsize="9")

        truth_data = TruthSource.catalog_to_array(self.truth_filepath)
        truth_sources = [TruthSource(s, w = self.reader.w) for s in truth_data]
        truth_sources = [s for s in truth_sources if s.x() > img_extent[0] and s.x() < img_extent[1] and s.y() > img_extent[2] and s.y() < img_extent[3] ]


        for i in [1,2,3]:
            ax[i].set(xlabel="X (RA axis)", ylabel="Y (Dec axis)")
            ax[i].add_patch(source.shape_pix(self.reader.dy))
            ax[i].plot([s.x() for s in truth_sources], [s.y() for s in truth_sources], 'rx')
            #try:
            #    ax[i].plot(truth_data[:,1], truth_data[:,2], 'rx')
            #except:
            #    ax[i].plot(truth_data[1], truth_data[2], 'rx')

        fig.show()
        plt.show()
