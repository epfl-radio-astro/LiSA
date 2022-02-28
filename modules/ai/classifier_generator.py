
import tensorflow.keras as keras
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from modules.ai.enums import InputMode, AugmentMode
import numpy as np
import random
import scipy

class LabeledDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, paths, labels, dim, batch_size=128):
        'Initialization'
        assert len(paths) == len(labels) 
        self.data = {}
        self.dim = dim
        self.augment = AugmentMode.OFF
        self.paths = paths
        self.labels = labels
        self.batch_size = batch_size
        self.data_loaded = False
        self.do_transform = False
        self.do_filter = False
        self.qt = QuantileTransformer()

        from pickle import load
        self.input_sf =  load(open('data/regressor_input_transform.pkl', 'rb'))

    def set_filter(self, *args):
        self.keep_sources = {}
        for file in args:
            with open(file, 'r') as f:
                for line in f:
                    if line[0] == '#': continue
                    x, y, z, sig = line.split()
                    x, y, z = int(x), int(y), int(z)
                    if x not in self.keep_sources: self.keep_sources[x] = {}
                    if y not in self.keep_sources[x]: self.keep_sources[x][y] = {}
                    if z not in self.keep_sources[x][y]: self.keep_sources[x][y][z] = True
        self.do_filter = True

    def _filter(self, files):
        out_files = []
        print(self.keep_sources)
        for f in files:
            indices = f.split('.')[0].split('/')[-1].split('_')[-1].split('-')
            indices = [int(i) for i in indices]
            try:
                assert self.keep_sources[x][y][z] == True
                out_files.append(f)
            except:
                continue
        print(len(out_files))
        return out_files

    def create_training_dict(self):
        import glob
        for i, l in enumerate(self.labels):
            files = glob.glob(self.paths[i])
            if self.do_filter:
                files = self._filter(files)
            
            self.data[l] = files

    def spawn(self, n, exclusive = True):
        out_gen = LabeledDataGenerator(self.paths,self.labels, self.dim, n)
        for label, v in self.data.items():
            random.shuffle(v)
            if exclusive:
                self.data[label] = v[n:]
            out_gen.data[label] = v[:n]
        out_gen.qt = self.qt
        out_gen.data_loaded = self.data_loaded
        out_gen.do_transform = self.do_transform
        return out_gen

    # scale factors to normalize data
    def calc_scale_factors(self):
        print("Calculating transform...")
        

        self.do_transform = True
        
        all_data = []
        for k, v in self.data.items():
            for f in v:
                d = np.load(f).flatten()
                all_data.append(d)
        all_data = np.array(all_data)
        all_data = all_data.flatten()
        all_data = all_data.reshape(-1, 1)
        self.qt.fit(all_data)



    def transform_and_save_data(self, outpath):
        if self.data_loaded:
            raise RuntimeError("Data already loaded!")
        self.calc_scale_factors()
        print("Loading training data...")
        for k, v in self.data.items():
            print("Loading {0} items".format(k)) 
            steps = int(len(v)/10)
            for i, f in enumerate(v):
                if i%steps == 0: print("{0:.2f}%".format(i/len(v)*100))
                d = self._load(f)
                outname = outpath + f.split('/')[-1].replace('.npy','_transformed.npy')
                np.save(d,f)
        

    def _load(self, f):

        d = np.load(f)

        #normalize data
        #d = (d - self.data_min)/(self.data_max - self.data_min)
        #d = d / np.max(d)
        data_shape = d.shape
        d.shape = (d.shape[0]*d.shape[1]*d.shape[2],1)
        if self.do_transform:
            d = self.qt.transform(d)
        else:
            #d -= np.min(d)
            #d /= abs(np.max(d))
            d *= self.input_sf

        d.shape = data_shape
        return d

    def load_data(self):
        print("Loading training data...")
        for k, v in self.data.items():
            print("Loading {0} items".format(k)) 
            steps = int(len(v)/10)
            for i, f in enumerate(v):
                if i%steps == 0: print("{0:.2f}%".format(i/len(v)*100))
                v[i] = self._load(f)
        self.data_loaded = True

    def truncate(self, n):
        for k, v in self.data.items():
            self.data[k] = v[:n]

    def get_data(self, d, do_permute = False):
        if not self.data_loaded:
            d = self._load(d)
        if d.shape[-1] != 1:
            d.shape = ( *(d.shape), 1)
        if do_permute:
            d = self.permute(d)
        return d

    def get_class(self, label):
        X = np.empty((len(self.data[label]), *self.dim))
        for i, f in enumerate(self.data[label]):
            X[i,] = self.get_data(f)
        return X

    def __len__(self):
        'Denotes the number of batches per epoch'
        nfiles_for_each_label = [len(v) for v in self.data.values()]
        total_files = min(nfiles_for_each_label)*len(self.data.values())
        l = int( total_files / self.batch_size)
        if self.augment == AugmentMode.FAST:
            l*= 4
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
        X, y = self.__data_generation(selection)
        return X, y

    def permute(self, d):

        # FLIP
        if self.augment == AugmentMode.ALL or self.augment == AugmentMode.FAST or self.augment == AugmentMode.TRAIN:
            if random.choice([True, False]):
                d = np.flip(d, axis = 2)

        #SHIFT
        if self.augment == AugmentMode.ALL or self.augment == AugmentMode.FAST or self.augment == AugmentMode.TRAIN:
            dec_shift  =  random.choice(range(-6,6))
            ra_shift   =  random.choice(range(-6,6))
            freq_shift =  random.choice(range(-40,40))
            d = np.roll(d, shift = (freq_shift, dec_shift, ra_shift), axis = (0, 1, 2))

        #ANY ROTATION
        if self.augment == AugmentMode.ALL:
            rot = random.uniform(0,360)
            d = scipy.ndimage.rotate(d, rot, axes=(1, 2), mode = 'reflect', reshape = False)

        # 90 DEG ROTATION
        if self.augment == AugmentMode.FAST:
            k = random.choice([0,1,2,3])
            d = np.rot90(d, k, axes=(1, 2))

        if self.augment == AugmentMode.TRAIN:
            k = random.choice([1,2,3])
            d = np.rot90(d, k, axes=(1, 2))

        # scaling
        if self.augment == AugmentMode.ALL or self.augment == AugmentMode.TRAIN or self.augment == AugmentMode.TEST or self.augment == AugmentMode.FAST:
            s = random.uniform(0.95,1.3)
            d *= s
            #print("Generator", s, l[0])

        return d


    def __data_generation(self, selection):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, 1))

        # Generate data
        for i, (label, item) in enumerate(selection):
            # Store sample

            d = self.get_data(item, do_permute = True)

            X[i,] = d
            y[i,] = 1 if label == 'true' else 0

        return X, y

class CombinedLabeledDataGenerator(LabeledDataGenerator):
    def __init__(self, paths1, paths2, labels, dim, batch_size=128):
        super().__init__(paths1,labels, dim, batch_size)
        self.paths1 = paths1
        self.paths2 = paths2
        self.qt1 = QuantileTransformer()
        self.qt2 = QuantileTransformer()
    def create_training_dict(self):
        import glob
        for i, l in enumerate(self.labels):
            files1 = sorted(glob.glob(self.paths1[i]))
            files2 = sorted(glob.glob(self.paths2[i]))
            if self.do_filter:
                files1 = self._filter(files1)
                files2 = self._filter(files2)
            if len(files1) != len(files2):
                print(len(files1), len(files2))
                for i in range(len(files1) ):
                    if files1[i].split('_')[-1] != files2[i].split('_')[-1]:
                        print(files1[i], files2[i])
                        break
            assert len(files1) == len(files2)
            self.data[l] = [ (f1,f2) for f1, f2 in zip(files1, files2)]

    # scale factors to normalize data
    def calc_scale_factors(self):
        print("Calculating transform...")
        self.do_transform = True    
        all_data1 = []
        all_data2 = []
        for k, v in self.data.items():
            for f in v:
                d1 = np.load(f[0]).flatten()
                all_data1.append(d1)
                d2 = np.load(f[1]).flatten()
                all_data2.append(d2)
        all_data1 = np.array(all_data1).flatten().reshape(-1, 1)
        all_data2 = np.array(all_data2).flatten().reshape(-1, 1)
        self.qt1.fit(all_data1)
        self.qt2.fit(all_data2)

    def spawn(self, n, exclusive = True):
        out_gen = CombinedLabeledDataGenerator(self.paths1, self.paths2 ,self.labels, self.dim, n)
        for label, v in self.data.items():
            random.shuffle(v)
            if exclusive:
                self.data[label] = v[n:]
            out_gen.data[label] = np.copy(v[:n])
        out_gen.qt1 = self.qt1
        out_gen.qt2 = self.qt2
        out_gen.data_loaded = self.data_loaded
        out_gen.do_transform = self.do_transform
        return out_gen

    def _load(self, f, qt = None):
        d = np.load(f)
        data_shape = d.shape
        d.shape = (d.shape[0]*d.shape[1]*d.shape[2],1)
        if self.do_transform and qt != None:
            d = self.qt.transform(d)
        else:
            d *= self.input_sf
        #    #d /= abs(np.max(d))
        d.shape = data_shape
        return d

    def load_data(self):
        print("Loading training data...")
        for k, v in self.data.items():
            print("Loading {0} items".format(k)) 
            steps = int(len(v)/10)
            for i, f in enumerate(v):
                if i%steps == 0:
                    print("{0:.2f}%".format(i/len(v)*100))
                    #print(f[0],f[1] )
                v[i] = (self._load(f[0], self.qt1), self._load(f[1], self.qt2))
                #assert 
                #print(np.min(v[i][0]), np.min(v[i][1]), np.max(v[i][0]), np.max(v[i][1]))
        self.data_loaded = True

    def permute(self, d1, d2):

        # FLIP
        if self.augment == AugmentMode.ALL or self.augment == AugmentMode.FAST or self.augment == AugmentMode.TRAIN:
            if random.choice([True, False]):
                d1 = np.flip(d1, axis = 2)
                d2 = np.flip(d2, axis = 2)

        #SHIFT
        #if self.augment == AugmentMode.ALL or self.augment == AugmentMode.FAST:
        #    d = np.roll(d, shift = random.choice([-1,0,1]), axis = 1)
        #    d = np.roll(d, shift = random.choice([-1,0,1]), axis = 2)

                #SHIFT
        if self.augment == AugmentMode.ALL or self.augment == AugmentMode.FAST or self.augment == AugmentMode.TRAIN:
            dec_shift  =  random.choice(range(-5,5))
            ra_shift   =  random.choice(range(-5,5))
            freq_shift =  random.choice(range(-15,15))
            d1 = np.roll(d1, shift = (freq_shift, dec_shift, ra_shift), axis = (0, 1, 2))
            d2 = np.roll(d2, shift = (freq_shift, dec_shift, ra_shift), axis = (0, 1, 2))

        #ANY ROTATION
        if self.augment == AugmentMode.ALL:
            rot = random.uniform(45,315)
            d1 = scipy.ndimage.rotate(d1, rot, axes=(1, 2), mode = 'reflect', reshape = False)
            d2 = scipy.ndimage.rotate(d2, rot, axes=(1, 2), mode = 'reflect', reshape = False)

        # 90 DEG ROTATION
        if self.augment == AugmentMode.FAST:
            k = random.choice([0,1,2,3])
            d1 = np.rot90(d1, k, axes=(1, 2))
            d2 = np.rot90(d2, k, axes=(1, 2))

        if self.augment == AugmentMode.TRAIN:
            k = random.choice([1,2,3])
            d1 = np.rot90(d1, k, axes=(1, 2))
            d2 = np.rot90(d2, k, axes=(1, 2))

        # scaling
        if self.augment == AugmentMode.ALL or self.augment == AugmentMode.TRAIN or self.augment == AugmentMode.TEST or self.augment == AugmentMode.FAST:
            s = random.uniform(0.95,1.3)
            d *= s
            #print("Generator", s, l[0])

        return d1, d2

    def get_data(self, d, do_permute = False):
        if not self.data_loaded:
            d1 = self._load(d[0], self.qt1)
            d2 = self._load(d[1], self.qt2)
        else:
            d1 = d[0]
            d2 = d[1]
        #print(np.min(d1), np.min(d2), np.max(d1), np.max(d2))
        if d1.shape[-1] != 1:
                d1.shape = ( *(d1.shape), 1)
        if d2.shape[-1] != 1:
                d2.shape = ( *(d2.shape), 1) 
        if do_permute:
            d1, d2 = self.permute(d1,d2)
        d = np.zeros(self.dim)
        d[:,0:30,:,:] = d1
        d[:,30:60,:,:] = d2
        return d

    def get_class(self, label):
        X = np.empty((len(self.data[label]), *self.dim))
        for i, f in enumerate(self.data[label]):
            X[i,] = self.get_data(f)
        return X