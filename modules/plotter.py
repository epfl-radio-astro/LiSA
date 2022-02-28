""" Some useful plotting scripts"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

def plot_slice(data, freq_slice = 0):
	print(data.shape)
	plt.figure()
	plt.imshow( np.abs(data[freq_slice,:,:]), cmap = 'bone', norm=LogNorm(vmin=0.0001, vmax=1))
	plt.colorbar()
	plt.xlabel("RA")
	plt.ylabel("Dec")
	#plt.savefig("mygraph.png")
	plt.show()

def plot_freq(data, f):
	freq = np.sum(data, axis = (1,2))
	fig, ax = plt.subplots(2)
	fig.tight_layout(pad = 3.0)
	ax[0].plot(f, freq)
	ax[0].set(xlabel="frequency [Hz]", ylabel='summed flux in RA and Dec')
	ax[1].plot(f[:500], freq[:500])
	ax[1].set(xlabel="frequency [Hz]", ylabel='summed flux in RA and Dec')
	plt.show()

def plot_cube(data):
	fig = plt.figure()
	ax = Axes3D(fig)

	coords = np.argwhere(data > 5e-4)
	print(coords.shape)
	print(coords)

	ax.scatter(coords[:,1], coords[:,2], coords[:,0])
	ax.set_zlabel('frequency index')
	ax.set_xlabel('RA index')
	ax.set_xlabel('Dec index')
	plt.show()


