import numpy as np
import sys, time, os
import matplotlib.pyplot as plt
import tensorflow as tf
    from pathlib import Path

from modules.truth_info import TruthSource
from modules.domain_reader import BinaryDomainReader as DenoisedReader
from modules.domain_reader import AstropyDomainReader as Reader
from modules.ai.enums import AugmentMode
from modules.ai.regressor import CNN
from modules.ai.utils import get_cutouts, loss_fn, asymmetry
from modules.truth_info import transforms


def make_training_data(size, rank, training_filepath,  filepath_truth, skip_existing = True):

    original_data = "/scratch/etolley/SDC2/dev/sky_ldev_v2.fits"
    denoised_data = "/scratch/etolley/SDC2/dev_denoised3d/sky_ldev_3Ddenoised_thr3_*.npy"
    reader1 = None
    reader2 = None
    prev_index = -1

    for domain_index in range(144):
        print ("Processing sources in domain {0} of {1}" .format(domain_index, 144))
        
        reader1 = Reader(1, 0, original_data, border = 15)
        reader2 = DenoisedReader(144, domain_index, denoised_data,  filepath_header = original_data, border = 15)
        print(" x: {0} - {1}, y: {2} - {3}".format(reader2.xmin, reader2.ymin, reader2.xmax, reader2.ymax))
        reader1.xmin, reader1.ymin, reader1.xmax, reader1.ymax =  reader2.xmin, reader2.ymin, reader2.xmax, reader2.ymax
        reader1._read_header()
        sources = TruthSource.catalog_to_sources_in_domain(filepath_truth, reader1)
        print("Domain has {0} sources".format(len(sources)))
        for s in sources:
            if s.line_flux_integral() < 60: continue
            ID = int(s.ID())
            orig_name = "{0}/cutout_original_{1}".format(training_filepath, ID)
            denoised_name = "{0}/cutout_denoised_{1}".format(training_filepath, ID)
            if skip_existing and Path(orig_name).is_file():
                continue
            if not reader1.is_data_loaded(): reader1.read()
            if not reader2.is_data_loaded(): reader2.read()
            orig_cutout      = reader1.get_cutout( (s.x(),s.y(),s.z()), 15, 100) 
            denoised_cutout  = reader2.get_cutout( (s.x(),s.y(),s.z()), 15, 100)   


            if orig_cutout.shape != denoised_cutout.shape or orig_cutout.shape[0] == 0:
                print("************* Skipping problematic cutout ************* ")
                print("* ",orig_cutout.shape, denoised_cutout.shape)
                print("* ",s.x(),s.y(),s.z())
                print("******************************************************* ")
                continue 

            plane = np.sum( denoised_cutout,  axis = 0)
            if asymmetry(plane) < -0.05:
                print("========== weird source ========== ")
                print("=",reader2)
                print("= ID:",s.ID(),"pos:",s.x(),s.y(),s.z())
                print("================================== ")
            np.save(orig_name, orig_cutout)
            np.save(denoised_name, denoised_cutout)

#==========================================
def permute_training_data(training_filepath, file_truth, permuted_training_filepath):
    from scipy import ndimage

    file_truth_permuted = "{0}/{1}".format(permuted_training_filepath, file_truth.split('/')[-1].replace('.txt','_permute.txt'))
    sources = TruthSource.catalog_to_sources(file_truth)

    files_original = glob.glob(training_filepath + "*original*.npy")
    files_original.sort()
    files_denoised = glob.glob(training_filepath + "*denoised*.npy")
    files_denoised.sort()
    assert len(files_original) == len(files_denoised)

    print("Permuting {0} files".format(len(files_original) + len(files_denoised)))

    source_count = 0
    with open(file_truth_permuted, 'w') as outfile:
        for s in sources:
            ID = int(s.ID())
            index = -999
            d1 = None
            for i, f in enumerate(files_original):
                file_ID = int(f.split('.')[0].split('_')[-1])
                if file_ID != ID: continue
                d1 = np.load(f)
                index = i
                break
            if index == -999: continue
            f_denoised = files_denoised[index]
            file_ID = int(f.split('.')[0].split('_')[-1])
            assert file_ID == ID
            d2 = np.load(f_denoised)

            assert d1.shape == d2.shape

            print("Now permuting source {0}".format(ID))
            
            rotations = np.linspace(45,315,13)
            for r in rotations:
                s_array = np.copy(s.data)
                d1_p = ndimage.rotate(d1, r, axes=(1, 2), mode = 'reflect', reshape = False)
                d2_p = ndimage.rotate(d2, r, axes=(1, 2), mode = 'reflect', reshape = False)
                s_array[6] += r
                s_array[6] %= 360
                np.save("{0}/cutout_original_{1}.npy".format(permuted_training_filepath, source_count), d1_p)
                np.save("{0}/cutout_denoised_{1}.npy".format(permuted_training_filepath, source_count), d2_p)

                out_source = ' '.join([ str(s_array[i]) for i in range(1,9)])
                out_source = "{0} {1}\n".format(source_count, out_source)
                print(" " ,out_source[:-2])
                outfile.write(out_source)
                source_count += 1

            #if source_count > 100: break

if __name__ == "__main__":

    ############################################
    # Setup
    ############################################

    #===== MPI  =====
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        print("This is task %d out of %d" % (comm.rank, comm.size))
    except:
        size = 1
        rank = 0

    #===== filepaths  =====
    filepath_truth = "/scratch/etolley/SDC2/dev/sky_ldev_truthcat_v2.txt"

    remake_training_data = False
    transform_type = 'log' # 'quantile' or 'log' or 'power'
    doubled_data = False

    training_filepath = "/scratch/etolley/SDC2/dev_regressor_training_large/"
    if remake_training_data:
        make_training_data(size, rank, training_filepath, filepath_truth,  skip_existing = False)


    training_filepath_permuted = "/scratch/etolley/SDC2/dev_regressor_training_permute13/"
    filepath_truth_permuted = "/scratch/etolley/SDC2/dev_regressor_training_permute13/sky_ldev_truthcat_v2_permute.txt"
    if not Path(filepath_truth_permuted).is_file():
        permute_training_data(training_filepath, filepath_truth, training_filepath_permuted)

    ############################################
    # Run pipeline
    ############################################

    #=====  initialize modules =====

    # CNN wrapper
    cutout_dim = (200, 60 if doubled_data else 30, 30, 1)
    cnn = CNN(cutout_dim) #InputMode.SPREAD THREEDEE
    cnn.out_dict = {
                "Line Flux Integral": lambda x: x.line_flux_integral(),
                "HI size": lambda x: x.hi_size(),
                "Cos(Pos A)": lambda x: np.cos(x.pos_a()*np.pi/180),
                "Sin(Pos A)": lambda x: np.sin(x.pos_a()*np.pi/180),
                "Inc A": lambda x: x.inc_a(),
                "w20":lambda x: x.w20(),
            }
    n_epochs = 1000

    flux_threshold = 20
    augmentstr = "test-permute"

    outname = "data/CNN_regressor-inception-{0}_{1}".format(transform_type, "2x" if doubled_data else "1x")
    outname += "_th-{0}_vars-{1}_aug-{2}_epochs-{3}".format(flux_threshold, cnn.n_out_params, augmentstr, n_epochs)
    print("####################\nWriting all outputs to {0}*\n####################".format(outname))
     
    #===== run modules =====

    from modules.ai.generator import DataGenerator as Gen
    generator = Gen(path_original = training_filepath_permuted + "*original*.npy",
                   path_denoised = training_filepath_permuted + "*denoised*.npy" if doubled_data else None,
                   path_truth = filepath_truth_permuted, properties = cnn.out_dict, dim = cutout_dim,
                   batch_size = 32, transform_type = transform_type)

    generator.augment = AugmentMode.TRAIN # FAST TRAIN FULL 
    generator.load_data(flux_threshold = flux_threshold)
    generator.gen_by_flux = True

    from pickle import dump
    dump(generator.input_sf, open('data/regressor_input_transform.pkl', 'wb'))

    validation_generator = Gen(path_original = training_filepath + "*original*.npy",
                   path_denoised = training_filepath + "*denoised*.npy" if doubled_data else None,
                   path_truth = filepath_truth, properties = cnn.out_dict, dim = cutout_dim, batch_size = 32,
                   transform_type = transform_type)
    validation_generator.load_data(flux_threshold = flux_threshold)
    validation_generator.gen_by_flux = True
    

    print("########################")
    fig, axs = plt.subplots(2,10, figsize=(12, 5))
    plt.tight_layout()
    for i in range(10):
        d = generator[i][0][0]
        print(d.shape, np.min(d), np.max(d))
        axs[0,i].imshow(np.sum(d, axis = 0))
        axs[1,i].imshow(np.sum(d, axis = 2))
    plt.savefig("check_regression_training_data.png")
    print("########################")

    print("Now defining and compiling model...")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        cnn.build_architecture()
    #tf.keras.utils.plot_model(cnn.model, to_file=outname + 'structure.png', show_shapes=True, show_layer_names=True)

    print("Now training...")

    # simple early stopping
    checkpoint_filepath = '/tmp/cnn_reg_checkpoint'
    checkpoint_filepath2 = '/tmp/cnn_reg_checkpoint2'
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mcp_save = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath,
                                                  save_weights_only=True,
                                                  save_best_only=True,
                                                  monitor='val_loss', mode='min',  verbose = 1)

    #with tf.device("device:GPU:0"): #/device:XLA_GPU:0
    history1= cnn.model.fit(generator, epochs=n_epochs, validation_data = validation_generator,
                               callbacks = [ mcp_save, es]) #validation_data = generator_val

    cnn.model.load_weights(checkpoint_filepath)
    min_loss = np.min(history1.history['val_loss'])
    outname1 = outname +  "_1st-training_"+ ("loss{0:.3f}".format(min_loss)).replace('.','p')
    cnn.model.save(outname1 + "_network")
    #try:
    #    tf.keras.utils.plot_model(cnn.model, to_file=outname + 'structure.png', show_shapes=True, show_layer_names=True)
    #except:
    #    "Unable to make network plot"


    #=================================================

    fig, axs = plt.subplots(2, cnn.n_out_params + 1, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.5, hspace=0.4)

    try:
        axs[0,0].plot(history1.history['loss'])
        axs[0,0].plot(history1.history['val_loss'])
        axs[0,0].set_yscale('log')
        axs[0,0].set(ylabel = "Loss (Mean Square Error)", xlabel = "Training Epoch")
        axs[1,0].plot(history1.history['mae'])
        axs[1,0].plot(history1.history['val_mae'])
        axs[1,0].set_yscale('log')
        axs[1,0].set(ylabel = "Mean Absolute Error", xlabel = "Training Epoch")

    except: pass

    val_data, val_truth = validation_generator.X, validation_generator.Y
    val_predict  = cnn.model.predict( val_data)

    

    if transform_type == "log":
        val_predict_orig = transforms.inv_transform(np.copy(val_predict))
    elif transform_type == "quantile":
        val_predict_orig = generator.qt.inverse_transform(np.copy(val_predict))
    elif transform_type == "power":
        val_predict_orig = generator.pt.inverse_transform(np.copy(val_predict))
    val_truth_orig = validation_generator.Y_orig

    train_data, train_truth = generator[0]
    train_predict  = cnn.model.predict( train_data)

    print("N params predicted:", cnn.n_out_params)
    for i in range(cnn.n_out_params):

        p_min, p_max = 0, 1

        axs[0,1+i].plot([p_min,p_max], [p_min,p_max], 'k-')
        axs[0,1+i].plot(train_truth[:,i], train_predict[:,i], 'c.')
        axs[0,1+i].plot(val_truth[:,i],   val_predict[:,i], 'bx')

        #axs[1,1+i].plot(train_truth_orig[:,i], train_predict_orig[:,i], 'c.')
        p_min, p_max = np.min(val_truth_orig[:,i]), np.max(val_truth_orig[:,i])
        axs[1,1+i].plot([p_min,p_max], [p_min,p_max], 'k-')
        axs[1,1+i].plot(val_truth_orig[:,i],   val_predict_orig[:,i], 'bx')
        axs[1,1+i].set_xlim([p_min, p_max])
        axs[1,1+i].set_ylim([p_min, p_max])
        
        axs[0,1+i].set(ylabel = "Prediction {0}".format(cnn.out_names[i]),
                       xlabel = "Truth {0}".format(cnn.out_names[i]))
        axs[1,1+i].set(ylabel = "Prediction {0}".format(cnn.out_names[i]),
                       xlabel = "Truth {0}".format(cnn.out_names[i]))

    plt.savefig(outname1 + "_plot")

    #plt.show()
    #plt.clf()
    ##plt.plot(history.history['mean_absolute_error'])
    #plt.plot(history.history['val_mean_absolute_error'])
    #plt.title('2 conv 2 max pooling 2 dense - 16,8 and 8,4 filters -lr =0.1 - 5sigma ')
    #plt.savefig('/home/aliqoliz/outputs/acc_flux.png')
    #plt.show()




