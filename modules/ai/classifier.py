
import numpy as np
import scipy
#import tensorflow.keras as keras
import tensorflow as tf
import random


from modules.util.truth_info import TruthSource
from modules.ai.utils import source_properties, loss_classifier
from modules.ai.enums import InputMode

advanced_metrics_enabled = False
try:
    from modules.ai.utils import Score
    advanced_metrics_enabled = True
except:
    advanced_metrics_enabled = False

class CNN:

    # wcl is the half-width of the cube in ra and dec
    # wf is the half-width of the cube in frequency
    def __init__(self,dim, use_denoised = True, use_continuum = False):
        self.dim = dim
        self.use_denoised = use_denoised
        self.use_continuum = use_continuum

    def data_info(self, ntrue, nfalse):
        self.ntrue = ntrue
        self.nfalse = nfalse
        self.metrics = [tf.keras.metrics.Precision([0.8], name='precision'),
           Score(thresholds = [0.6,0.7,0.8,0.9], true_sources=self.ntrue,  false_sources=self.nfalse, name='score'),
           #Score(threshold = 0.9, true_sources=self.ntrue,  false_sources=self.nfalse, name='score9'),
           tf.keras.metrics.TruePositives( thresholds = [0.8], name = 'true_pos'),
           tf.keras.metrics.FalsePositives( thresholds = [0.8], name = 'false_pos'),
           ]

    def build_architecture(self):
        #self.build_architecture3D()
        self.build_inception1()

    def build_inception1(self):
        from tensorflow.keras.layers import Input, Dense, Dropout, Conv3D, MaxPooling3D, AveragePooling3D, Flatten, BatchNormalization, SpatialDropout3D, GlobalAveragePooling3D
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import SGD, Adam, RMSprop 
        import tensorflow as tf


        input_cube = Input(shape= self.dim)


        layer_2 = BatchNormalization(center=True, scale=True)(input_cube)
        layer_2 = Conv3D(32, (8,8,8), activation='relu', name = "B_conv3_1")(input_cube)
        layer_2 = MaxPooling3D((5,3,3), name ="B_pool2_1")(layer_2)
        layer_2 = Conv3D(32, (5,5,5), activation='relu',  name = "B_conv3_2")(layer_2)
        layer_2 = MaxPooling3D((3,2,2), name ="B_pool2_2")(layer_2)
        layer_2 = Flatten(name = "B_out")(layer_2)

        dense_1 = Dense(64, activation='relu')(layer_2)
        dense_1 = Dropout(0.2)(dense_1)
        dense_1 = Dense(8, activation='relu')(dense_1)
        output = Dense(1, activation='sigmoid')(dense_1)

        self.model = Model(input_cube, output)

        try:
            self.metrics = [tf.keras.metrics.Precision([0.8], name='precision'),
                           Score(thresholds = [0.6,0.7,0.8,0.9, 0.95, 0.99], true_sources=self.ntrue,  false_sources=self.nfalse, name='score'),
                           #Score(threshold = 0.9, true_sources=self.ntrue,  false_sources=self.nfalse, name='score9'),
                           tf.keras.metrics.TruePositives( thresholds = [0.8], name = 'true_pos'),
                           tf.keras.metrics.FalsePositives( thresholds = [0.8], name = 'false_pos'),
                           
                            ]
        except:
            self.metrics = [tf.keras.metrics.Precision([0.8], name='precision'),
                           tf.keras.metrics.TruePositives( thresholds = [0.8], name = 'true_pos'),
                           tf.keras.metrics.FalsePositives( thresholds = [0.8], name = 'false_pos'),
                           
                           ]
        self.opt = RMSprop(learning_rate=0.001)
        self.loss = "binary_crossentropy"
        self.model.compile(optimizer=self.opt,loss=self.loss, metrics=self.metrics)
        
        self.model.summary()


    def build_inception2(self):
        from tensorflow.keras.layers import Input, Dense, Dropout, Conv3D, MaxPooling3D, AveragePooling3D, Flatten, BatchNormalization, SpatialDropout3D, GlobalAveragePooling3D
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import SGD, Adam, RMSprop
        import tensorflow as tf

        input_cube = Input(shape= self.dim)
        input_norm = BatchNormalization(center=True, scale=True)(input_cube)

        layer_1 = Conv3D(32, (3,3,3), activation='relu', name = "A_conv3_1")(input_norm)
        layer_1 = MaxPooling3D((3,2,2), name ="A_pool2_1")(layer_1)
        layer_1 = Conv3D(32, (3,3,3), activation='relu',  name = "A_conv3_2")(layer_1)
        layer_1 = MaxPooling3D((5,3,3), name ="A_pool2_2")(layer_1)
        layer_1 = Flatten(name = "A_out")(layer_1)

        layer_2 = Conv3D(32, (8,8,8), activation='relu', name = "B_conv3_1")(input_norm)
        layer_2 = MaxPooling3D((5,3,3), name ="B_pool2_1")(layer_2)
        layer_2 = Conv3D(32, (5,5,5), activation='relu',  name = "B_conv3_2")(layer_2)
        layer_2 = MaxPooling3D((3,2,2), name ="B_pool2_2")(layer_2)
        layer_2 = Flatten(name = "B_out")(layer_2)

        mid_1 = tf.keras.layers.concatenate([layer_1, layer_2], axis = -1)

        dense_1 = Dense(64, activation='relu')(mid_1)
        dense_1 = Dropout(0.2)(dense_1)
        dense_1 = Dense(8, activation='relu')(dense_1)
        output = Dense(1, activation='sigmoid')(dense_1)

        self.model = Model(input_cube, output)

        try:
            self.metrics = [tf.keras.metrics.Precision([0.8], name='precision'),
                           Score(thresholds = [0.6,0.7,0.8,0.9, 0.95, 0.99], true_sources=self.ntrue,  false_sources=self.nfalse, name='score'),
                           #Score(threshold = 0.9, true_sources=self.ntrue,  false_sources=self.nfalse, name='score9'),
                           tf.keras.metrics.TruePositives( thresholds = [0.8], name = 'true_pos'),
                           tf.keras.metrics.FalsePositives( thresholds = [0.8], name = 'false_pos'),
                           
                            ]
        except:
            self.metrics = [tf.keras.metrics.Precision([0.8], name='precision'),
                           tf.keras.metrics.TruePositives( thresholds = [0.8], name = 'true_pos'),
                           tf.keras.metrics.FalsePositives( thresholds = [0.8], name = 'false_pos'),
                           
                           ]
        self.opt = RMSprop(learning_rate=0.001)
        self.loss = "binary_crossentropy"
        self.model.compile(optimizer=self.opt,loss=self.loss, metrics=self.metrics)
        
        self.model.summary()

    def build_inception(self):
        from tensorflow.keras.layers import Input, Dense, Dropout, Conv3D, MaxPooling3D, AveragePooling3D, Flatten, BatchNormalization, SpatialDropout3D, GlobalAveragePooling3D
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import SGD, Adam, RMSprop
        import tensorflow as tf


        input_cube      = Input(shape= self.dim)
        input_cube2     = Input(shape= self.dim)
        input_continuum = Input(shape= (21, self.dim[1], self.dim[2], self.dim[3]))
        input_z      = Input(shape= (1))

        dropout_level = 0.1

        layer_1 = BatchNormalization(center=True, scale=True)(input_cube)
        layer_1 = Conv3D(32, (3,3,3), activation='relu', name = "A_conv3_1")(layer_1)
        layer_1 = MaxPooling3D((2,2,2), name ="A_pool2_1")(layer_1)
        layer_1 = Conv3D(32, (3,3,3), activation='relu', name = "A_conv3_2")(layer_1)
        layer_1 = SpatialDropout3D(dropout_level)(layer_1)
        layer_1 = MaxPooling3D((2,2,2), name = "A_pool2_2")(layer_1)
        layer_1 = Flatten(name = "A_out")(layer_1)

        layer_2 = BatchNormalization(center=True, scale=True)(input_cube)
        layer_2 = AveragePooling3D((5,2,2), name = "B_pool-freq")(layer_2)
        layer_2 = Conv3D(16, (3,3,3), activation='relu', name = "B_conv5x3x3_1")(layer_2)
        layer_2 = MaxPooling3D((2,2,2), name = "B_pool4x2x2_1")(layer_2)
        layer_2 = Conv3D(16, (3,3,3), activation='relu', name = "B_conv5x3x3_2")(layer_2)
        layer_2 = SpatialDropout3D(dropout_level)(layer_2)
        layer_2 = MaxPooling3D((3,2,2), name = "B_pool4x2x2_2")(layer_2)
        layer_2 = Flatten(name = "B_out")(layer_2)

        concat_list = [layer_1, layer_2,input_z]
        input_list = [input_cube]

        if self.use_denoised:
            layer_4 = AveragePooling3D((3,2,2), name = "D_pool-freq")(input_cube2)
            layer_4 = Conv3D(32, (3,3,3), activation='relu', name = "D_conv3_1")(layer_4)
            layer_4 = MaxPooling3D((5,2,2), name ="D_pool2_1")(layer_4)
            layer_4 = Conv3D(32, (3,3,3), activation='relu', name = "D_conv3_2")(layer_4)
            layer_4 = SpatialDropout3D(dropout_level)(layer_4)
            layer_4 = MaxPooling3D((3,3,3), name = "D_pool2_2")(layer_4)
            layer_4 = Flatten(name = "D_out")(layer_4)
            concat_list.append(layer_4)
            input_list.append(input_cube2)

        if self.use_continuum:
            layer_3 = AveragePooling3D((1,2,2), name = "C_pool-cel")(input_continuum)
            layer_3 = Conv3D(16, (3,3,3), activation='relu', name = "C_conv3_1")(layer_3)
            layer_3 = MaxPooling3D((2,2,2), name = "C_pool2_1")(layer_3)
            layer_3 = Conv3D(16, (3,3,3), activation='relu', name = "C_conv3_2")(layer_3)
            layer_3 = SpatialDropout3D(dropout_level)(layer_3)
            layer_3 = MaxPooling3D((2,2,2), name = "C_pool2_2")(layer_3)
            layer_3 = Flatten(name = "C_out")(layer_3)
            concat_list.append(layer_3)
            input_list.append(input_continuum)

        input_list.append(input_z)

        mid_1 = tf.keras.layers.concatenate(concat_list, axis = -1)

        dense_1 = Dropout(dropout_level*2)(mid_1)
        #dense_1 = Dense(128, activation='relu')(dense_1)
        #dense_1 = Dropout(dropout_level*2)(dense_1)
        dense_1 = Dense(64, activation='relu')(dense_1)
        dense_1 = Dropout(dropout_level*2)(dense_1)
        dense_1 = Dense(32, activation='relu')(dense_1)
        output = Dense(1, activation='sigmoid')(dense_1)

        #self.model = Model([input_cube, input_cube2, input_continuum, input_z], output)
        self.model = Model(input_list, output)

        try:
            self.metrics = [tf.keras.metrics.Precision([0.8], name='precision'),
                           Score(thresholds = [0.6,0.7,0.8,0.9, 0.95, 0.99], true_sources=self.ntrue,  false_sources=self.nfalse, name='score'),
                           #Score(threshold = 0.9, true_sources=self.ntrue,  false_sources=self.nfalse, name='score9'),
                           tf.keras.metrics.TruePositives( thresholds = [0.8], name = 'true_pos'),
                           tf.keras.metrics.FalsePositives( thresholds = [0.8], name = 'false_pos'),
                           
                            ]
        except:
            self.metrics = [tf.keras.metrics.Precision([0.8], name='precision'),
                           tf.keras.metrics.TruePositives( thresholds = [0.8], name = 'true_pos'),
                           tf.keras.metrics.FalsePositives( thresholds = [0.8], name = 'false_pos'),
                           
                           ]
        self.opt = RMSprop(learning_rate=0.001)
        self.loss = loss_classifier
        #opt = Adam(learning_rate=0.001)
        #self.model.compile(optimizer=opt,loss=loss_classifier, metrics=metrics)
        self.model.compile(optimizer=self.opt,loss=self.loss, metrics=self.metrics)
        #self.model.compile(optimizer=opt,loss="binary_crossentropy", metrics=metrics)
        
        self.model.summary()





    def build_architecture3D(self):
        from tensorflow.keras.optimizers import SGD, Adam, RMSprop
        from tensorflow.keras.layers import Activation,Dense, SpatialDropout3D, Conv3D,  Cropping3D, MaxPooling3D, AveragePooling3D, Flatten, BatchNormalization, Dropout, GlobalAveragePooling3D
        from tensorflow.keras.models import Sequential
        self.model = Sequential()
        
        self.model.add(Conv3D(32, kernel_size=(8,8,8), input_shape=self.dim, activation='relu'))
        self.model.add(BatchNormalization(center=True, scale=True))
        self.model.add(MaxPooling3D(pool_size=(5,3,3)))

        self.model.add(Conv3D(32, kernel_size=(5,5,5), activation='relu'))
        self.model.add(BatchNormalization(center=True, scale=True))
        self.model.add(MaxPooling3D(pool_size=(3,2,2)))
        self.model.add(SpatialDropout3D(0.2))

        self.model.add(Flatten())
        
        self.model.add(Dense(64, activation='relu')) # relu or sigmoid
        self.model.add(Dropout(0.3))
        self.model.add(Dense(8, activation='relu'))

        self.model.add(Dense(1, activation='sigmoid'))
        #opt = Adam(lr=1e-4) # SGD(lr=0.1)
        self.opt = SGD(lr=0.001)
        #opt = RMSprop(learning_rate=0.001)
        self.loss = 'binary_crossentropy'


        try:
            self.metrics = [tf.keras.metrics.Precision([0.8], name='precision'),
                       Score(thresholds = [0.6,0.7,0.8,0.9], true_sources=self.ntrue,  false_sources=self.nfalse, name='score'),
                       #Score(threshold = 0.9, true_sources=self.ntrue,  false_sources=self.nfalse, name='score9'),
                       tf.keras.metrics.TruePositives( thresholds = [0.8], name = 'true_pos'),
                       tf.keras.metrics.FalsePositives( thresholds = [0.8], name = 'false_pos'),
                       ]
        except:
            self.metrics = [tf.keras.metrics.Precision([0.8], name='precision'),
                       tf.keras.metrics.TruePositives( thresholds = [0.8], name = 'true_pos'),
                       tf.keras.metrics.FalsePositives( thresholds = [0.8], name = 'false_pos'),
                       ]

        self.model.compile(optimizer=self.opt,loss=self.loss, metrics=self.metrics)

        #self.model.compile(optimizer=opt,loss=loss_score, metrics=['accuracy'])

        self.model.summary()
