import numpy as np
from modules.domain_reader import AstropyDomainReader   as Reader
from modules.domain_reader import DomainData
from modules.source_finder import LikelihoodFinder as Finder
from modules.source_finder import merge_source_candidates, remove_suspicious_candidates, prune_candidates
import sys
from pathlib import Path

####### BEGIN USER CONFIG #######

#===== input filepaths  =====
filepath_datacube       = "/scratch/etolley/SDC2/dev/sky_ldev_v2.fits"
filepath_continuumimage = "/scratch/etolley/SDC2/dev/cont_ldev.fits" 

#===== output directory  =====
output_directory = "/scratch/etolley/SDC2/ldev_output_3Ddenoise/"

####### END USER CONFIG #######

#===== domain indexing  =====
n_domains = 1
domain_index = 0

if len(sys.argv) > 3:
    print("Program does not understand extra arguments. Expected input:\npython pipeline_full.py {n_domains} {domain_index} ")
    sys.exit()
elif len(sys.argv) > 1:
    n_domains = int(sys.argv[1])
    domain_index = int(sys.argv[2]) + rank
    assert n_domains > domain_index
else:
    print("No command line arguments received to override domain numbering.")

print("This is task index {0} of {1}, running on domain index {2} of {3}".format(rank, size, domain_index, n_domains))

#===== output file name templates  =====
lkh_threshold = 12
denoised_file_name          = "/scratch/etolley/SDC2/dev_3Ddenoise/{0}domains/".format(n_domains, output_directory)
source_candidates_file_name = "{1}/likelihood_sources_th{2}_{0}.txt".format(domain_index, output_directory, lkh_threshold)
merged_source_candidates_file_name = "{1}/merged_sources_th{2}_{0}.txt".format(domain_index, output_directory, lkh_threshold)
output_catalog              = "{1}/catalog_new4_th{2}_{0}.txt".format(domain_index, output_directory, lkh_threshold)

#===== create necessary directories  =====
Path(output_directory).mkdir(parents=True, exist_ok=True)
Path(denoised_file_name).mkdir(parents=True, exist_ok=True)

######################################################
# Step 1: Denoise
######################################################

print("====== Domain {0}: starting ======".format(domain_index ))

#===== define readers  =====

reader_original = Reader(n_domains, domain_index, filepath_datacube, filepath_continuumimage, border = 15,  forceSquare=True)
reader_original.define_domains()
reader_original._read_header()

reader_denoised = Reader(n_domains, domain_index, filepath_datacube, filepath_continuumimage, border = 15,  forceSquare=True)
reader_denoised.define_domains()
reader_denoised._read_header()

# reduce domain size if not running with MPI
if n_domains == 1:
    reader_original.xmax, reader_original.ymax = 134, 134
    reader_denoised.xmax, reader_denoised.ymax = 134, 134
print("I am task {0} of {1}".format(domain_index, n_domains), flush=True)
print(reader_original)
print(reader_denoised)

#===== denoise  =====

# define output file name
denoised_file_name+= "sky_ldev_3Ddenoised_thr3"
if n_domains > 1: denoised_file_name +=  "_task-{0}".format(domain_index)
denoised_file_name += "_x-{0}-{1}_y-{2}-{3}_border{4}.npy".format(reader_original.xmin, reader_original.xmax, reader_original.ymin, reader_original.ymax, reader_original.border)

if Path(denoised_file_name).is_file():
    print ("Domain {0}: Denoised file {1} already exists, reading file".format(domain_index, denoised_file_name))
    reader_denoised.HI_cube = DomainData(infile = denoised_file_name)

else:
    print("Domain {0}: generating denoised domain {1}".format(domain_index,denoised_file_name ))
    # read the domain into memory
    reader_denoised.read()

    # denoise the domain
    from modules.denoiser import Denoiser2D1D as Denoiser
    denoiser = Denoiser() 
    output_cube = np.zeros(reader_denoised.HI_cube.data.shape)
    steps = np.linspace(0,reader_denoised.HI_cube.data.shape[0],50, dtype = np.int)
    for k0, k1, k2, k3 in zip(np.append(steps[0],steps[:-2]), steps[:-1], steps[1:], np.append(steps[2:],steps[-1])):
        print(k0,k1,k2,k3)
        input_image = reader_denoised.HI_cube.data[k0:k3,:,:]
        denoised_cube = denoiser.denoise( np.copy(input_image), threshold_level=3)#, num_scales_2d = 5, num_scales_1d = 5)
        output_cube[k1:k2,:,:] = denoised_cube[k1-k0:k2-k0,:,:]
    reader_denoised.HI_cube.data = output_cube
    reader_denoised.HI_cube.write(denoised_file_name)

######################################################
# Step 2: Find sources
######################################################

if Path(source_candidates_file_name).is_file():
    print ("Domain {0}: Source candidate file {1} already exists, reading file".format(domain_index, source_candidates_file_name))
else:
    print("Domain {0}: finding candidate sources...".format(domain_index,source_candidates_file_name ))
    #===== find sources  =====
    from modules.noise_profile import Landau, Gauss
    finder = Finder(reader_denoised, 10, Landau(), out_dir = output_directory, method="likelihood", df=20, lkh_threshold_attempts=10)
    finder._log_lkh_threshold = lkh_threshold
    source_candidates = finder.find_sources()
    print("Domain {0}: found {1} candidate sources.".format(domain_index,len(source_candidates) ), flush=True)
    with open(source_candidates_file_name, 'w') as f:
        for s in source_candidates: f.write(s.to_line())

print("Domain {0}: now merging source candidates".format(domain_index), flush=True)

merge_source_candidates(source_candidates_file_name, merged_source_candidates_file_name)

print("Domain {0}: now setting up ML".format(domain_index), flush=True)

######################################################
# Step 3: Characterize
######################################################
import tensorflow as tf
from modules.ai.utils import loss_fn, Score
from modules.ai import  enums
import modules.ai.classifier as classifier
import modules.ai.regressor as regressor
from modules.ai.utils import get_cutouts
from modules.ai.generator import DataGenerator as Gen

#===== set up CNN and input/output scale factors  =====

cutout_dim = (200, 30, 30, 1)

CNN_classifier = classifier.CNN(cutout_dim, use_denoised = True, use_continuum = False)
CNN_regressor = regressor.CNN(cutout_dim)
CNN_classifier.model=tf.keras.models.load_model("data/SKA_SDC2_classifier_CNN")
CNN_regressor.model=tf.keras.models.load_model("data/SKA_SDC2_regressor_CNN",
                                                custom_objects={'loss_fn': loss_fn})

CNN_classifier.model.summary()

from pickle import load
from modules.truth_info import transforms
input_sf = load(open('data/regressor_input_transform.pkl', 'rb'))

#===== classify sources  =====

if not reader_original.is_data_loaded():
    reader_original.read()

results = {}
classifier_thresholds = [0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
for c_thr in classifier_thresholds:
    results[c_thr] = []
with open(merged_source_candidates_file_name, 'r') as f:
    for line in f:
        i, j, k, sig = line.split()
        i, j, k, sig = int(i), int(j), int(k), float(sig)
        ra, dec, freq = reader_original.pixels_to_sky(i,j,k)
 
        original_cutout, denoised_cutout = get_cutouts(i, j, k, reader_original, reader_denoised, zwidth = 100)
        #continuum_cutout = reader_original.get_continuum_cutout([i,j,k],15)


        if original_cutout.shape[0] != cutout_dim[0]:
            print("Unable to classify source with cutout dim", original_cutout.shape)
            continue

        cutout = original_cutout
        cutout.shape = (1,*cutout.shape)
        denoised_cutout.shape = (1,*denoised_cutout.shape)
        denoised_cutout /= np.max(denoised_cutout)

        print(cutout.shape)

        z = np.array([k/6000])
        z.shape = (1,1)

        classifier_score = CNN_classifier.model.predict( cutout*input_sf )

        print("Classifier score: ", classifier_score[0][0])

        if classifier_score[0][0] < classifier_thresholds[0]: continue

        # predict
        predictions  = CNN_regressor.model.predict( cutout*input_sf)

        # transform outputs
        predictions_tr = transforms.inv_transform(predictions)
        flux, hisize, sinposa, cosposa, inca, w20 = predictions_tr[0]
        posa = np.arctan2(sinposa,cosposa)
        result = "{0} {1} {2} {3} {4} {5} {6} {7}\n".format(ra, dec, hisize, flux, freq, posa, inca, w20)
        print(result[:-2], flush=True)

        for c_thr in classifier_thresholds:
            if classifier_score[0][0] > c_thr:
                results[c_thr].append(result)

for c_thr in classifier_thresholds:
    print("Domain {0}: classified {1} sources with c_score > {2}.".format(domain_index,len(results[c_thr]),c_thr ), flush=True)
    c_thr_str = "{0:.2f}".format(c_thr).replace(".","p")
    out_catalog_name = output_catalog.replace(".txt","_c{0}.txt".format(c_thr_str))
    with open(out_catalog_name, 'w') as f:
        for r in results[c_thr]:
            f.write(r)


        



