import glob, sys
import numpy as np
from pathlib import Path
from modules.domain_reader import BinaryDomainReader as DenoisedReader
from modules.domain_reader import AstropyDomainReader as Reader
import matplotlib.pyplot as plt
import matplotlib
from modules.ai.classifier import CNN
from modules.ai.enums import AugmentMode
from modules.ai.utils import get_cutouts, get_domain_index



def read_source_list(sources_found_filename):
    source_positions = []
    with open(sources_found_filename, 'r') as f:
        for line in f:
            if line[0] == '#': continue
            x, y, z, __ = [i for i in line.split()]
            x, y, z = [int(i) for i in [x,y,z]]
            source_positions.append([x,y,z])
    return source_positions

def _make_training_data_helper(source_positions, label, training_filepath, skip_existing = True):
    original_data = "/scratch/etolley/SDC2/dev/sky_ldev_v2.fits"
    denoised_data = "/scratch/etolley/SDC2/dev_denoised3d/sky_ldev_3Ddenoised_thr3_*.npy"
    reader1 = None
    reader2 = None
    prev_index = -1
    for pos in source_positions:
        orig_outname = "{0}/{4}_orig_{1}-{2}-{3}.npy".format(training_filepath,*pos, label)
        denoised_outname = "{0}/{4}_denoised_{1}-{2}-{3}.npy".format(training_filepath, *pos, label)
        if skip_existing and Path(orig_outname).is_file():
            continue

        index, xmin, ymin, xmax, ymax = get_domain_index(*pos, denoised_data)
        if index!= prev_index:
            reader1 = Reader(1, 0, original_data, border = 15)
            reader1.xmin, reader1.ymin, reader1.xmax, reader1.ymax = xmin,ymin,xmax,ymax
            reader1.read()
            reader2 = DenoisedReader(144, index, denoised_data,  filepath_header = original_data, border = 15)
            reader2.read()
            prev_index = index

        print("Reading cube at", *pos)
        orig_cutout, denoised_cutout = get_cutouts(*pos, reader1, reader2, zwidth = 100)
        if orig_cutout.shape != denoised_cutout.shape or orig_cutout.shape[0] == 0:
            print("Skipping problematic cutout")
            print(orig_cutout.shape, denoised_cutout.shape)
            print(index,  "x:{0}-{1}".format(xmin, xmax), "y:{0}-{1}".format(ymin, ymax))
            continue
        np.save(orig_outname, orig_cutout)
        np.save(denoised_outname, denoised_cutout)

def make_training_data(size, rank, training_filepath, skip_existing = True, Lkh_th = 13):
       # read true and false source positions
    true_source_positions =  read_source_list("data/true_candidate_sources_th{0}.txt".format(Lkh_th))
    false_source_positions = read_source_list("data/false_candidate_sources_th{0}.txt".format(Lkh_th))

    true_source_positions = np.array_split(true_source_positions, size)[rank]
    false_source_positions = np.array_split(false_source_positions, size)[rank]

    print("Task {0} is reading {1} true sources, {2} false sources".format(rank, len(true_source_positions), len(false_source_positions)), flush=True)
    

    Path(training_filepath).mkdir(parents=True, exist_ok=True)
    _make_training_data_helper(true_source_positions, "true", training_filepath, skip_existing)
    _make_training_data_helper(false_source_positions, "false", training_filepath, skip_existing)

def check_training_data(training_filepath):
    # plotting the cutouts for a few different sources
    fig, axs = plt.subplots(4, 5)
    plt.tight_layout()
    
    for i, f in enumerate(glob.glob(training_filepath + "true_denoised*.npy")[:5]):
        cutout = np.load(f)
        axs[0,i].imshow(np.sum(cutout, axis = 0), cmap = "GnBu_r")
        axs[1,i].imshow(np.sum(cutout, axis = 1), cmap = "GnBu_r")

    for i, f in enumerate(glob.glob(training_filepath + "false_denoised*.npy")[:5]):
        cutout = np.load(f)
        axs[2,i].imshow(np.sum(cutout, axis = 0), cmap = "GnBu_r")
        axs[3,i].imshow(np.sum(cutout, axis = 1), cmap = "GnBu_r")

    plt.show()


if __name__ == "__main__":

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

    remake_training_data = False
    doubled_data = False

    Lkh_th =  12 #12 13 14
    training_filepath = "/scratch/etolley/SDC2/dev_classifier_training_th{0}_large/".format(Lkh_th)
    if remake_training_data:
        make_training_data(size, rank, training_filepath,  skip_existing = True, Lkh_th = Lkh_th)
    #check_training_data(training_filepath)

    if doubled_data:
        from modules.ai.generator import CombinedLabeledDataGenerator as Gen
        gen_args = [[training_filepath + "true_orig*.npy" , training_filepath + "false_orig*.npy"],
                    [training_filepath + "true_denoised*.npy" , training_filepath + "false_denoised*.npy"],
                    ["true", "false"], (200,60,30,1) ]
    else:
        from modules.ai.classifier_generator import LabeledDataGenerator as Gen
        gen_args = [[training_filepath + "true_orig*.npy" , training_filepath + "false_orig*.npy"],
                    ["true", "false"], (200,30,30,1) ]

    generator = Gen(*gen_args)
    generator.batch_size = 32
    generator.create_training_dict()
    generator.load_data()
    n_true_sources = len(generator.data['true'])
    n_false_sources = len(generator.data['false'])

    print("Have {0} true sources and {1} false sources".format(n_true_sources, n_false_sources ))
    validation_generator = generator.spawn(1000, exclusive = False)
    test_generator = generator.spawn(100, exclusive = True)
    #generator.truncate(128)
    generator.augment = AugmentMode.TRAIN # FAST ALL
    validation_generator.augment = AugmentMode.OFF
    validation_generator.batch_size = 64
    test_generator.batch_size = 64




    #validation_data = generator_val
    true_data = generator.get_class("true")[:2000]
    false_data = generator.get_class("false")[:2000]
    true_data_val = validation_generator.get_class("true")[:2000]
    false_data_val = validation_generator.get_class("false")[:2000]


    import tensorflow as tf
    cnn = CNN(generator.dim) 
    cnn.data_info(ntrue = n_true_sources, nfalse = n_false_sources)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        cnn.build_architecture()
    #cnn.model=tf.keras.models.load_model("data/CNN_classifier_orig_network_test")

    # simple early stopping
    checkpoint_filepath = '/tmp/cnn_checkpoint'
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mcp_save = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath,
                                                  save_weights_only=True,
                                                  save_best_only=True,
                                                  monitor='val_score', mode='max', verbose = 1)
    
    #with tf.device("device:GPU:0"): #/device:XLA_GPU:0
    history= cnn.model.fit(generator, epochs=10,  validation_data = validation_generator, callbacks = [es, mcp_save])
    cnn.model.load_weights(checkpoint_filepath)


    results = cnn.model.evaluate(validation_generator)
    test_results = cnn.model.evaluate(test_generator)
    print(results)
    print("Valudation results:")
    print(test_results)


    #remove custom metrics
    with strategy.scope():
        cnn.model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=[])


    predictions_true = cnn.model.predict( true_data)
    predictions_false = cnn.model.predict( false_data)
    predictions_true_val = cnn.model.predict( true_data_val)
    predictions_false_val = cnn.model.predict( false_data_val)

    print("## Predictions:")
    print( np.mean(predictions_true))
    print( np.mean(predictions_false))

    fig, axs = plt.subplots(1,3, figsize=(10,4))
    axs = axs.flatten()
    plt.subplots_adjust(wspace=0.5)

    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].legend()
    axs[0].set(ylabel = "Loss", xlabel = "Epoch")

    max_range = max(np.max(predictions_true), np.max(predictions_false))
    min_range = min(np.min(predictions_true), np.min(predictions_false))
    axs[1].hist(predictions_true, bins = 20, range = (min_range, max_range), color = 'dodgerblue', histtype = 'step', label = "train true")
    axs[1].hist(predictions_false,bins = 20, range = (min_range, max_range), color = 'red',        histtype = 'step', label = "train false")
    axs[1].hist(predictions_true_val, bins = 20, range = (min_range, max_range), color = 'blue', histtype = 'step', label = "val true")
    axs[1].hist(predictions_false_val,bins = 20, range = (min_range, max_range), color = 'darkred',        histtype = 'step', label = "val false")
    axs[1].legend()
    axs[1].set(ylabel = "Count", xlabel = "Classifier Score")

    print("Have {0} true sources and {1} false sources".format(n_true_sources, n_false_sources ))
    thresholds = np.linspace(0.5,1,6)
    n_true_array = []
    n_false_array = []
    score_array = []
    max_score = -9999
    for thr in thresholds:
        true_positive_rate  = np.sum(predictions_true_val  > thr) / len(predictions_true_val)
        false_positive_rate = np.sum(predictions_false_val > thr) / len(predictions_false_val)
        n_true  = true_positive_rate*n_true_sources
        n_false = false_positive_rate*n_false_sources
        estimated_score = n_true*0.52 - n_false
        if max_score < estimated_score: max_score = estimated_score
        n_true_array.append(n_true)
        n_false_array.append(n_false)
        score_array.append(estimated_score)
        print("Score_NN > {3:.1f} -- Estimated true sources: {0:.2f}, false sources: {1:.2f}, score: {2:.2f}".format(n_true, n_false, estimated_score, thr))
    axs[2].hist(thresholds, weights = n_true_array, bins = 5, color = 'dodgerblue', histtype = 'step', label = "estimated # match")
    axs[2].hist(thresholds, weights = n_false_array,bins = 5, color = 'red',        histtype = 'step', label = "estimated # false")
    axs[2].hist(thresholds, weights = score_array,  bins = 5, color = 'black',      histtype = 'step', label = "estimated score")
    axs[2].set(xlabel = "Classifier Score Threshold")
    axs[2].legend()
    
    max_score = int(max_score)
    tag = "both" if doubled_data else "original"
    cnn.model.save("data/CNN_classifier{1}_{2}_maxScore{0}".format(max_score, Lkh_th,tag))
    plt.savefig("classifier{1}_{2}_training_maxScore{0}.png".format(max_score, Lkh_th,tag))

    

    
    
