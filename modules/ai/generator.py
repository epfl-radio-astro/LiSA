import tensorflow.keras as keras
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from modules.ai.enums import InputMode, AugmentMode
import numpy as np
import random
import scipy
from modules.ai.utils import transforms

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path_original, path_denoised, path_truth, properties, dim,  transform_type = 'log', batch_size=128, shuffle=True):
        'Initialization'
        self.path_original = path_original
        self.path_denoised = path_denoised
        self.path_truth = path_truth
        self.augment = AugmentMode.OFF
        self.dim = dim
        self.properties = properties
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.qt = QuantileTransformer()
        self.pt = PowerTransformer()
        self.gen_by_flux = False
        self.out_index = -1
        self.transform_type = transform_type

    def spawn(self, pct, exclusive = True, shuffle = True):
        n = int(pct*self.X.shape[0])
        gen = DataGenerator(self.path_original, self.path_denoised, self.path_truth, self.properties, self.dim, self.batch_size)
        gen.qt = self.qt

        if self.gen_by_flux:
            indexes = []
            nsamples = int(n/sum(len(v) > 0 for v in self.flux_dict.values()))
            print("Take {0} samples from each flux bin to spawn new generator".format(nsamples))
            for k, i_list in self.flux_dict.items():
                if len(i_list) == 0: continue
                [indexes.append(random.choice(i_list)) for i in range(nsamples)]
            gen.X = self.X[indexes]
            gen.Y = self.Y[indexes]
            gen.Y_orig = self.Y_orig[indexes]
        else:
            np.random.shuffle(self.indexes)
            gen.X = self.X[:n]
            gen.Y = self.Y[:n]
            gen.Y_orig = self.Y_orig[:n]
            
            if exclusive:
                self.X = self.X[n:]
                self.Y = self.Y[n:]  
                self.Y_orig = self.Y_orig[n:]
        gen.on_epoch_end()
        self.on_epoch_end()
        return gen

        
    def load_data(self, flux_threshold = 0):
        print("Loading data...")
        import glob
        from modules.utils.truth_info import TruthSource

        files_original = glob.glob(self.path_original)
        files_original.sort()
        if self.path_denoised != None:
            files_denoised = glob.glob(self.path_denoised)
            files_denoised.sort()

        self.flux_dict = {0: [],
                          20: [],
                          40: [],
                          60: [],
                          80: [],
                          100: [],
                          120: [],
                          140: [],
                          160: [],
                          180: [],
                          200: []}

        sources = TruthSource.catalog_to_sources(self.path_truth)
        self.X = []
        self.Y = []
        source_index = 0
        for s in sources:
            if s.line_flux_integral() < flux_threshold: continue
            ID = int(s.ID())

            d1 = None
            index = -999
            for i, f in enumerate(files_original):
                file_ID = int(f.split('.')[0].split('_')[-1])
                if file_ID != ID: continue
                d1 = np.load(f)
                index = i
                break

            # training data doesn't exist for file
            if index == -999: continue


            d = np.zeros(self.dim)
            try:
                d[:,0:30,:,0] = d1
            except:
                print("Problem with source {0}, data shape is:".format(ID),d1.shape)
                continue

            if self.path_denoised != None:
                f_denoised = files_denoised[index]
                file_ID = int(f.split('.')[0].split('_')[-1])
                assert file_ID == ID
                d2 = np.load(f_denoised)
                d[:,30:60,:,0] = d2

            self.X.append(d)
            #print("Now transforming to properties",[k for k in self.properties.keys()])
            p = np.array([f(s) for f in self.properties.values()])
            self.Y.append(p)
            for k in [ki for ki in self.flux_dict.keys()][::-1]:
                if s.line_flux_integral() > k:
                   self.flux_dict[k].append(source_index) 
                   break
            source_index += 1

        for k in self.flux_dict.keys():
            print(k, len(self.flux_dict[k]))
        assert len(self.X) == len(self.Y)
        self.X    = np.array(self.X)
        self.Y    = np.array(self.Y)
        self.Y_orig = np.copy(self.Y)

        if self.transform_type == 'log':
            self.Y = transforms.transform(self.Y)
        elif self.transform_type == 'quantile':
            self.Y = self.qt.fit_transform(self.Y)
        elif self.transform_type == 'power':
            self.Y = self.pt.fit_transform(self.Y)
        else:
            raise RuntimeError("Transform type {0} not recognized".format(self.transform_type))

        self.input_sf = 1./np.max(self.X)
        self.X *=self.input_sf
        print(self.X.shape)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        l = int (self.X.shape[0] / self.batch_size)
        if self.augment == AugmentMode.ALL:
            l*= 10
        elif self.augment == AugmentMode.FAST:
            l*= 4
        return max(1, l)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if self.gen_by_flux:
            indexes = []
            while len(indexes) < self.batch_size:
                for k, i_list in self.flux_dict.items():
                    if len(i_list) == 0: continue
                    indexes.append(random.choice(i_list))
                    if len(indexes) >= self.batch_size: break
        else:
            if self.batch_size < len(self.indexes):
                indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            else:
                indexes = [random.choice(self.indexes) for i in range(self.batch_size)]
        # Generate data
        X, y = self.__data_generation(indexes)
        if self.out_index >= 0:
            y = y[:,self.out_index,]
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.X.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_data(self, idx):
        l = np.copy(self.Y[idx])
        d = np.copy(self.X[idx,:,:,:,:])
        d, l = self.permute(d,l)
        return d, l

    def permute(self, d, l):

        ax_ra  = 2
        ax_dec = 1

        if self.path_denoised != None:
            d1 = d[:,0:30,:,:]
            d2 = d[:,30:60,:,:]
        else:
            d1 = d
            d2 = None
            

        # the positon angle will be between [0,1] instead of [0,360]
        pos_a = 0 if len(l) < 3 else l[2]

        # FLIP
        if self.augment == AugmentMode.ALL or self.augment == AugmentMode.FAST or self.augment == AugmentMode.TRAIN:
            if random.choice([True, False]):
                d1 = np.flip(d1, axis = ax_ra)
                if self.path_denoised != None:
                    d2 = np.flip(d2, axis = ax_ra)
                pos_a = 1 -pos_a 

        #SHIFT
        if self.augment == AugmentMode.ALL or self.augment == AugmentMode.FAST or self.augment == AugmentMode.TRAIN:
            dec_shift  =  random.choice(range(-3,3)) + random.choice(range(-3,3))
            ra_shift   =  random.choice(range(-3,3)) + random.choice(range(-3,3))
            freq_shift =  random.choice(range(-20,20)) + random.choice(range(-20,20))
            d1 = np.roll(d1, shift = (freq_shift, dec_shift, ra_shift), axis = (0, ax_dec, ax_ra))
            if self.path_denoised != None:
                d2 = np.roll(d2, shift = (freq_shift, dec_shift, ra_shift), axis = (0, ax_dec, ax_ra))

        #ANY ROTATION
        if self.augment == AugmentMode.ALL:
            rot = random.uniform(45,360-45)
            d1 = scipy.ndimage.rotate(d1, rot, axes=(ax_dec, ax_ra), mode = 'reflect', reshape = False)
            if self.path_denoised != None:
                d2 = scipy.ndimage.rotate(d2, rot, axes=(ax_dec, ax_ra), mode = 'reflect', reshape = False)
            pos_a += rot/360
            pos_a %= 1


        if self.augment == AugmentMode.TRAIN:
            k = random.choice([1,2,3])
            d1 = np.rot90(d1, k, axes=(ax_dec, ax_ra))
            if self.path_denoised != None:
                d2 = np.rot90(d2, k, axes=(ax_dec, ax_ra))
            pos_a += k*90./360


        # scaling
        if self.augment == AugmentMode.ALL or self.augment == AugmentMode.TRAIN or self.augment == AugmentMode.TEST or self.augment == AugmentMode.FAST:
            if l[0] > 0.5:
                s = random.uniform(0.8,1.5)
            elif l[0] > 0.2:
                s = random.uniform(0.9,1.2)
            elif self.augment == AugmentMode.TRAIN or self.augment == AugmentMode.ALL:
                s = random.uniform(0.95,1.05)
            else:
                s = 1
            d1 *= s
            if self.path_denoised != None:
                d2 *= s
            l[0] *= s
            #print("Generator", s, l[0])
        
        l[2] = pos_a%1
        if self.path_denoised != None:
            d[:,0:30,:,:] = d1
            d[:,30:60,:,:] = d2
        else:
            d = d1

        return d,l

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, len(self.properties)))

        # Generate data
        for i, idx in enumerate(indexes):
            # Store sample
            #print(i, idx, len(indexes))

            d,l = self.get_data(idx)

            X[i,] = d
            y[i,] = l

        return X, y

#================================================================================================


class LabeledDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,  labels, dim, original_files, 
                       denoised_files = None, 
                       continuum_files = None, batch_size=128, gen_z = True):
        'Initialization'
        assert len(original_files) == len(labels) 
        if denoised_files != None:  assert len(denoised_files) == len(labels) 
        if continuum_files != None: assert len(continuum_files) == len(labels) 

        self.data = {}
        self.dim = dim
        self.augment = AugmentMode.OFF
        self.original_files  = original_files
        self.denoised_files  = denoised_files
        self.continuum_files = continuum_files
        self.gen_z = gen_z
        self.labels = labels
        self.batch_size = batch_size
        self.data_loaded = False
        self.do_transform = False
        self.do_filter = False
        self.qt = QuantileTransformer()

        from pickle import load
        self.input_sf =  load(open('data/regressor_input_transform.pkl', 'rb'))

    def create_training_dict(self):
        import glob
        for i, l in enumerate(self.labels):
            # load the data and check that files match
            f_orig = sorted(glob.glob(self.original_files[i]))
            if self.continuum_files != None:
                f_cont = sorted(glob.glob(self.continuum_files[i]))
                assert len(f_orig) == len(f_cont)
                for j in range(len(f_cont)):
                    assert f_cont[j].split('_')[-1] == f_orig[j].split('_')[-1]
            if self.denoised_files != None:
                f_denoised = sorted(glob.glob(self.denoised_files[i]))
                assert len(f_orig) == len(f_denoised)
                for j in range(len(f_denoised)):
                    assert f_denoised[j].split('_')[-1] == f_orig[j].split('_')[-1]


            # zip up each file into a tuple
            self.denoised_index = None
            self.continuum_index = None
            if self.continuum_files != None and self.denoised_files != None:
                self.data[l] = [ (f1,f2, f3) for f1, f2, f3 in zip(f_orig, f_denoised, f_cont)]
                self.continuum_index = 2
                self.denoised_index = 1
            elif self.continuum_files != None:
                self.data[l] = [ (f1,f2) for f1, f2 in zip(f_orig, f_cont)]
                self.continuum_index = 1
            elif self.denoised_files != None:
                self.data[l] = [ (f1,f2) for f1, f2 in zip(f_orig, f_denoised)]
                self.denoised_index = 1
            else:
                self.data[l] = [[f] for f in f_orig]

    def pad(self,n):
        for k, v in self.data.items():
            for i in range(n):
                v.append(v[0])


    def spawn(self, n, exclusive = True, shuffle = True):
        out_gen = LabeledDataGenerator(self.labels, self.dim, self.original_files, self.denoised_files, self.continuum_files)
        for label, files in self.data.items():
            if shuffle:
                random.shuffle(files)
            if exclusive:
                self.data[label] = files[n:]
            out_gen.data[label] = list(np.copy(files[:n]))
        out_gen.qt = self.qt
        out_gen.data_loaded = self.data_loaded
        out_gen.do_transform = self.do_transform
        out_gen.continuum_index = self.continuum_index
        out_gen.denoised_index = self.denoised_index
        out_gen.gen_z = self.gen_z
        return out_gen

    # f: filename (string)
    # index: 0 original, 1/2 denoised or continuum
    def _load(self, f):
        d = [np.load(fi) for fi in f]
        #print(f)
        #original
        '''d[0]*=self.input_sf 
        if self.continuum_index != None:
            d[self.continuum_index] /= np.max(d[self.continuum_index])
        if self.denoised_index != None:
            d[self.denoised_index] /= np.max(d[self.denoised_index])'''
        for i in range(len(d)):
            if d[i].shape[-1] != 1: d[i].shape = ( *(d[i].shape), 1)

        if self.gen_z:
            pos = f[0].split('.')[0].split('_')[-1]
            z = int(pos.split('-')[-1])
            z /= 6000
            d.append(z)
        return d

    def load_data(self):
        print("Now loading ALL training data...")
        print("Disable this if you encounter memory errors.")
        
        for k, v in self.data.items():
            print("Loading {0} {1} items".format(len(v),k)) 
            steps = int(len(v)/10)
            for i, f in enumerate(v):
                if i%steps == 0: print("{0:.2f}%".format(i/len(v)*100))
                v[i] = self._load(f)
        self.data_loaded = True

    def truncate(self, n):
        print("Now truncating training data to {0} of each class".format(n))
        for k, v in self.data.items():
            self.data[k] = v[:n]

    def get_data(self, d, do_permute = False):
        if not self.data_loaded:
            d = self._load(d)
        d = np.copy(d)
        if do_permute:
            if self.gen_z:
                d[:-1] = self.permute(d[:-1])
            else:
                d = self.permute(d)
        return d

    def get_class(self, label, n):
        X = {'original': list()}
        if self.denoised_files != None:
            X['denoised'] = list() 
        if self.continuum_files != None:
            X['continuum'] = list() 
        if self.gen_z: X['z'] = list()

        for i, f in enumerate(self.data[label]):
            d = self.get_data(f)
            X['original'].append(d[0])
            if self.gen_z: X['z'].append( np.asarray( [d[-1]]) )
            if self.denoised_files != None:  X['denoised'].append(d[self.denoised_index])
            if self.continuum_files != None: X['continuum'].append(d[self.continuum_index])
            if i > n: break
        X['original'] = np.array(X['original'])
        if self.gen_z: X['z'] = np.array(X['z'])
        if self.denoised_files != None:  X['denoised'] = np.array(X['denoised'])
        if self.continuum_files != None: X['continuum'] = np.array(X['continuum'])
        keys = [k for k in X.keys()]
        if len(keys) > 1:
            return [X[k] for k in X.keys()]
        return X[keys[0]]

    def shuffle(self):
        for k in self.data.keys():
            print( k, len(self.data[k]), self.data[k][0])
            np.random.shuffle(self.data[k])
            print( k, len(self.data[k]), self.data[k][0])

    def __len__(self):
        'Denotes the number of batches per epoch'
        nfiles_for_each_label = [len(v) for v in self.data.values()]
        total_files = min(nfiles_for_each_label)*len(self.data.values())
        l = int( total_files / self.batch_size /2)
        return l
        #return max(1, int(np.floor(len(self.data) / self.batch_size))*2)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        selection = []
        for k, v in self.data.items():
            files = [random.choice(v) for i in range( int(self.batch_size/2))]
            for f in files:
                selection.append( (k, f))

        np.random.shuffle(selection)
        # Generate data
        #print(selection)
        X, y = self.__data_generation(selection)
        return X, y

    def permute(self, d):

        if self.augment == AugmentMode.OFF:
            #print("...PERMUTING with no augment")
            return d

        if self.augment == AugmentMode.TEST:
            #print("...PERMUTING with test augment")
            for i in range(len(d)): d[i][0,0,0,0] = 0
            return d

        # FLIP
        if self.augment == AugmentMode.ALL or self.augment == AugmentMode.FAST or self.augment == AugmentMode.TRAIN:
            if random.choice([True, False]):
                for i in range(len(d)):
                    d[i] = np.flip(d[i], axis = 2)

        #SHIFT
        if self.augment == AugmentMode.ALL or self.augment == AugmentMode.FAST or self.augment == AugmentMode.TRAIN or self.augment == AugmentMode.TEST:
            dec_shift  =  random.choice(range(-6,6))
            ra_shift   =  random.choice(range(-6,6))
            freq_shift =  random.choice(range(-50,50))
            
            d[0] = np.roll(d[0], shift = (freq_shift, dec_shift, ra_shift), axis = (0, 1, 2))
            if self.denoised_index != None:
                d[self.denoised_index] = np.roll(d[self.denoised_index], shift = (freq_shift, dec_shift, ra_shift), axis = (0, 1, 2))
            if self.continuum_index != None:
                d[self.continuum_index] = np.roll(d[self.continuum_index], shift = (dec_shift, ra_shift), axis = (1, 2))


        #ANY ROTATION
        if self.augment == AugmentMode.ALL:
            rot = random.uniform(45,360-45)
            for i in range(len(d)):
                d[i] =  scipy.ndimage.rotate(d[i], rot, axes=(1, 2), mode = 'reflect', reshape = False)
 
        # 90 DEG ROTATION
        if self.augment == AugmentMode.FAST:
            k = random.choice([0,1,2,3])
            for i in range(len(d)):
                d[i] =  np.rot90(d[i], k, axes=(1, 2))

        if self.augment == AugmentMode.TRAIN:
            k = random.choice([1,2,3])
            for i in range(len(d)):
                d[i] =  np.rot90(d[i], k, axes=(1, 2))

        # scaling
        if self.augment == AugmentMode.ALL or self.augment == AugmentMode.TRAIN or self.augment == AugmentMode.TEST or self.augment == AugmentMode.FAST:
            s = random.uniform(0.9,1.5)
            for i in range(len(d)): d[i] *= s
            #print("Generator", s, l[0])

        return d

    def on_epoch_end(self):
        print("EPOCH END")
        for k, v in self.data.items():
            print (k, len(v))
            print(k, v[0][0][0,0,0,0])

    def __data_generation(self, selection):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = {'original': list()}
        if self.denoised_files != None:
            X['denoised'] = list() 
        if self.continuum_files != None:
            X['continuum'] = list() 
        if self.gen_z:
            X['z'] = list()
        y = np.empty((self.batch_size, 1))

        # Generate data
        for i, (label, item) in enumerate(selection):
            # Store sample
            d = self.get_data(item, do_permute = True)
            X['original'].append(d[0])
            if self.gen_z:
                X['z'].append( np.asarray( [d[-1]]) )
            if self.denoised_files != None:  X['denoised'].append(d[self.denoised_index])
            if self.continuum_files != None: X['continuum'].append(d[self.continuum_index])
            y[i,] = 1 if label == 'true' else 0

        X['original'] = np.array(X['original'])
        if self.gen_z: X['z'] = np.array(X['z'])
        if self.denoised_files != None:  X['denoised'] = np.array(X['denoised'])
        if self.continuum_files != None: X['continuum'] = np.array(X['continuum'])
        #for k, a in X.items(): print("debug", k, a.shape)
        keys = [k for k in X.keys()]
        #print("DEBUG",keys)
        if len(keys) > 1:
            return [X[k] for k in X.keys()], y
        return X[keys[0]], y
