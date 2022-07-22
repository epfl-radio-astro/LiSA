
import numpy as np
import scipy
#import tensorflow.keras as keras
import random
#from sklearn.preprocessing import QuantileTransformer

from modules.util.truth_info import TruthSource
from modules.ai.utils import source_properties, loss_fn


class CNN:
    def __init__(self, dim):
        self.dim = dim
        self.out_dict = source_properties
        self.lr = 1e-4
        self.kernel_size = (3,3,3)

    @property
    def n_out_params(self):
        return len(self.out_dict)

    @property
    def out_names(self):
        return list(self.out_dict.keys())

    def build_architecture(self):
        #self.build_architecture3D()
        #self.build_architecture_simple()
        self.build_inception()

    def build_inception(self):
        from tensorflow.keras.layers import Input, Dense, Dropout, Conv3D, MaxPooling3D, AveragePooling3D, Flatten, BatchNormalization, SpatialDropout3D, GlobalAveragePooling3D
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import SGD, Adam
        import tensorflow as tf

        input_cube = Input(shape= self.dim)

        dropout_level = 0.05

        layer_1 = Conv3D(32, (3,3,3), activation='relu', name = "A_conv3_1")(input_cube)
        layer_1 = MaxPooling3D((2,2,2), name ="A_pool2_1")(layer_1)
        layer_1 = Conv3D(32, (3,3,3), activation='relu', name = "A_conv3_2")(layer_1)
        layer_1 = MaxPooling3D((2,2,2), name = "A_pool2_2")(layer_1)
        layer_1 = Conv3D(32, (3,3,3), activation='relu', name = "A_conv3_3")(layer_1)
        layer_1 = SpatialDropout3D(dropout_level)(layer_1)
        layer_1 = MaxPooling3D((4,2,2), name = "A_pool2_3")(layer_1)
        
        #layer_1 = MaxPooling3D((4,2,2), name = "A_pool2_3")(layer_1)
        layer_1 = Flatten(name = "A_out")(layer_1)

        layer_2 = AveragePooling3D((5,1,1), name = "B_celpool")(input_cube)
        layer_2 = Conv3D(16, (5,5,5), activation='relu', name = "B_conv5x3x3_1")(layer_2)
        layer_2 = MaxPooling3D((3,3,3), name = "B_pool4x2x2_1")(layer_2)
        layer_2 = Conv3D(32, (5,5,5), activation='relu', name = "B_conv5x3x3_2")(layer_2)
        layer_2 = SpatialDropout3D(dropout_level)(layer_2)
        #layer_2 = MaxPooling3D((2,2,2), name = "B_pool4x2x2_2")(layer_2)
        layer_2 = Flatten(name = "B_out")(layer_2)

        mid_1 = tf.keras.layers.concatenate([layer_1, layer_2], axis = 1)

        dense_1 = Dropout(dropout_level*2)(mid_1)
        dense_1 = Dense(1024, activation='relu')(dense_1)
        dense_1 = Dropout(dropout_level*2)(dense_1)
        dense_1 = Dense(512, activation='relu')(dense_1)
        dense_1 = Dropout(dropout_level*2)(dense_1)
        dense_1 = Dense(256, activation='relu')(dense_1)
        output = Dense(self.n_out_params, activation='linear')(dense_1)

        self.model = Model([input_cube], output)
        opt = Adam(lr=self.lr) # SGD(lr=0.1)
        #opt = SGD(lr=0.001)
        #self.model.compile(optimizer=opt,loss='mean_squared_error', metrics=['mae'])
        self.model.compile(optimizer=opt,loss=loss_fn, metrics=['mae'])
        self.model.summary()

        self.model = Model([input_cube], output)
        opt = Adam(lr=self.lr) # SGD(lr=0.1)
        #opt = SGD(lr=0.005, clipvalue=2.0)
        #self.model.compile(optimizer=opt,loss='mean_squared_error', metrics=['mae'])
        self.model.compile(optimizer=opt,loss=loss_fn, metrics=['mae'])
        self.model.summary()


    def build_architecture_simple(self):
        from tensorflow.keras.optimizers import SGD, Adam
        from tensorflow.keras.layers import Input,Dense, Conv3D,  SpatialDropout3D, Cropping3D, MaxPooling3D, AveragePooling3D, Flatten, BatchNormalization, Dropout, GlobalAveragePooling3D
        from tensorflow.keras.models import Sequential

        self.model = Sequential()
        self.model.add(Input(shape = self.dim))
        self.model.add(Conv3D(16, kernel_size=(5,5,5), activation='relu'))
        self.model.add(BatchNormalization(center=True, scale=True))
        self.model.add(MaxPooling3D(pool_size=(5,3,3)))
        

        #self.model.add(Conv3D(64, kernel_size=(5,5,5), activation='relu'))
        #self.model.add(BatchNormalization(center=True, scale=True))
        #self.model.add(MaxPooling3D(pool_size=(3,1,1)))

        self.model.add(Conv3D(32, kernel_size=(5,5,5), activation='relu'))
        self.model.add(BatchNormalization(center=True, scale=True))
        self.model.add(MaxPooling3D(pool_size=(3,3,3)))

        self.model.add(SpatialDropout3D(0.2))

        #self.model.add(GlobalAveragePooling3D())
        self.model.add(Flatten())
        #self.model.add(Dropout(0.2))
        
        
        self.model.add(Dense(self.n_out_params*16, activation='relu')) # relu or sigmoid
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.n_out_params*8, activation='relu')) # relu or sigmoid
        #self.model.add(Dropout(0.3))
        #self.model.add(Dense(self.n_out_params*8, activation='relu')) # relu or sigmoid
        self.model.add(Dense(self.n_out_params, activation='linear'))
        opt = Adam(lr=self.lr) # SGD(lr=0.1)
        #opt = SGD(lr=0.005, clipvalue=2.0)
        #self.model.compile(optimizer=opt,loss='mean_squared_error', metrics=['mae'])
        self.model.compile(optimizer=opt,loss=loss_fn, metrics=['mae'])
        self.model.summary()

    def build_architecture3D(self):
        from tensorflow.keras.optimizers import SGD, Adam
        from tensorflow.keras.layers import Input,Dense, Conv3D,  SpatialDropout3D, Cropping3D, MaxPooling3D, AveragePooling3D, Flatten, BatchNormalization, Dropout, GlobalAveragePooling3D
        from tensorflow.keras.models import Sequential
        self.model = Sequential()

        #self.model.add(BatchNormalization(center=True, scale=True, input_shape=self.dim))
        self.norm_layers = [BatchNormalization(center=True, scale=True),
                            BatchNormalization(center=True, scale=True),
                            BatchNormalization(center=True, scale=True)]

        self.model.add(Input(shape = self.dim))
        self.model.add(Conv3D(64, kernel_size=self.kernel_size, activation='relu'))
        self.model.add(self.norm_layers[0])
        self.model.add(SpatialDropout3D(0.1))
        self.model.add(MaxPooling3D(pool_size=(4,2,2)))
        

        self.model.add(Conv3D(64, kernel_size=self.kernel_size, activation='relu'))
        self.model.add(self.norm_layers[1])
        self.model.add(MaxPooling3D(pool_size=(2,2,2)))

        self.model.add(Conv3D(64, kernel_size=self.kernel_size, activation='relu'))
        self.model.add(self.norm_layers[2])
        self.model.add(MaxPooling3D(pool_size=(2,2,2)))

        #self.model.add(SpatialDropout3D(0.2))

        #self.model.add(Conv3D(128, kernel_size=(3,3,3), activation='relu'))
        #self.model.add(self.norm_layers[2])
        #self.model.add(MaxPooling3D(pool_size=(2,2,2)))

        #self.model.add(Conv3D(256, kernel_size=(3,3,3), activation='relu'))
        #self.model.add(BatchNormalization(center=True, scale=True))
        #self.model.add(MaxPooling3D(pool_size=(2,2,2)))
        
        #self.model.add(Flatten())
        self.model.add(GlobalAveragePooling3D())
        
        self.model.add(Dense(self.n_out_params*16, activation='relu')) # relu or sigmoid
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.n_out_params*8, activation='relu')) # relu or sigmoid
        self.model.add(Dropout(0.3))
        #self.model.add(Dense(self.n_out_params*8, activation='relu')) # relu or sigmoid
        self.model.add(Dense(self.n_out_params, activation='linear'))
        opt = Adam(lr=self.lr) # SGD(lr=0.1)
        #opt = SGD(lr=0.005, clipvalue=2.0)
        #self.model.compile(optimizer=opt,loss='mean_squared_error', metrics=['mae'])
        self.model.compile(optimizer=opt,loss=loss_fn, metrics=['mae'])
        self.model.summary()

    def layer_normalization(self, active):
        from tensorflow.keras.optimizers import SGD, Adam
        for l in self.norm_layers:
            l.training = active
        opt = Adam(lr=self.lr)
        self.model.compile(optimizer=opt,loss=loss_fn, metrics=['mae'])

#================================================================
class MultiCNN:
    def __init__(self, dim):
        self.dim = dim
        self.out_dict = source_properties
    @property
    def n_out_params(self):
        #return len(self.out_dict)
        return 1

    @property
    def out_names(self):
        return list(self.out_dict.keys())

    def build_architecture(self):
        self.build_flux_architecture()

    def build_flux_architecture(self):
        from tensorflow.keras.optimizers import SGD, Adam
        from tensorflow.keras.layers import Activation,Dense, Conv3D,  Cropping3D, MaxPooling3D, AveragePooling3D, Flatten, BatchNormalization, Dropout, GlobalAveragePooling3D
        from tensorflow.keras.models import Sequential
        self.fluxmodel = Sequential()

        self.fluxmodel.add(Conv3D(32, kernel_size=(3,3,3), input_shape=self.dim, activation='relu'))
        self.fluxmodel.add(BatchNormalization(center=True, scale=True))
        self.fluxmodel.add(MaxPooling3D(pool_size=(4,2,2)))

        #self.fluxmodel.add(Conv3D(32, kernel_size=(3,3,3), activation='relu'))
        #self.fluxmodel.add(BatchNormalization(center=True, scale=True))
        #self.fluxmodel.add(MaxPooling3D(pool_size=(2,2,2)))

        self.fluxmodel.add(Conv3D(64, kernel_size=(3,3,3), activation='relu'))
        self.fluxmodel.add(BatchNormalization(center=True, scale=True))
        self.fluxmodel.add(MaxPooling3D(pool_size=(2,2,2)))
        
        #self.model.add(Flatten())
        self.fluxmodel.add(GlobalAveragePooling3D())
        
        self.fluxmodel.add(Dense(32, activation='relu')) # relu or sigmoid
        self.fluxmodel.add(Dropout(0.3))
        self.fluxmodel.add(Dense(1, activation='linear'))
        opt = Adam(lr=1e-5) # SGD(lr=0.1)
        #opt = SGD(lr=0.005, clipvalue=2.0)
        self.fluxmodel.compile(optimizer=opt,loss='mean_squared_error', metrics=['mae'])
        self.fluxmodel.summary()

    def fit_flux(self, generator, epochs, validation_generator, callbacks):
        generator.out_index = 0
        validation_generator.out_index = 0
        history = self.fluxmodel.fit(generator, epochs=epochs, validation_data = validation_generator, callbacks = callbacks)
        generator.out_index = -1
        validation_generator.out_index = -1
        return history

    def predict(self, X):
        Y = np.zeros( (X.shape[0],5))
        Y[:,0] = self.fluxmodel.predict(X)[0]
        return Y

    def save(self, outname):
        self.fluxmodel.save(outname + "_fluxmodel")


