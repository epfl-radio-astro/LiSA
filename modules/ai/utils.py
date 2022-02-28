import tensorflow as tf
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend
import numpy as np
from modules.truth_info import TruthSource

source_properties = {
            "Line Flux Integral": lambda x: x.line_flux_integral(),
            "HI size": lambda x: x.hi_size(),
            "Pos A": lambda x: x.pos_a(),
            "Inc A": lambda x: x.inc_a(),
            "w20":lambda x: x.w20(),
            }

def asymmetry(plane):
    center = np.sum(plane[10:20,10:20])
    sides = [np.sum(plane[0:10,0:10]),  np.sum(plane[0:10,10:20]),  np.sum(plane[0:10,20:30]),
             np.sum(plane[10:20,0:10]),                             np.sum(plane[10:20,20:30]),
             np.sum(plane[20:30,0:10]), np.sum(plane[20:30,10:20]), np.sum(plane[20:30,20:30]),]
    return center - max(sides)

def loss_fn(y_true, y_pred, sample_weight=None):
    # first y should always be the line flux integral
    squared_difference = tf.square(y_true - y_pred)
    return (y_true[:,0]+1.) * tf.reduce_mean(squared_difference, axis=-1)

def loss_classifier(y_true, y_pred):
    TP = tf.cast((y_true * y_pred > 0.7), tf.float32)
    FP = tf.cast(((1-y_true) * y_pred > 0.7), tf.float32)
    TP = tf.reduce_sum(TP)
    FP = tf.reduce_sum(FP)
    score = TP*0.5-FP
    #score = TP*0.5*0.15-FP*0.85
    #return tf.keras.losses.binary_crossentropy(y_true, y_pred) - score*0.5
    #return tf.keras.losses.binary_crossentropy(y_true, y_pred) - TP*0.1
    positives = tf.reduce_sum( y_pred*(y_true))*0.1
    negatives = tf.reduce_sum( y_pred*(1-y_true))*0.1
    #return tf.keras.losses.binary_crossentropy(y_true, y_pred)  + tf.reduce_sum( y_pred*(1-2*y_true))*0.01
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) - positives


class Score(tf.keras.metrics.Metric):
    def __init__(self, thresholds=[0.8], true_sources=10, false_sources=10,**kwargs):
        super(Score, self).__init__(**kwargs)
        num_thresholds = len(thresholds)
        self.true_sources = tf.convert_to_tensor(true_sources, dtype=tf.float32)
        self.false_sources = tf.convert_to_tensor(false_sources, dtype=tf.float32)

        self.thresholds = thresholds
        self.true_positives = self.add_weight(
            'tp',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight(
            'fp',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight(
            'fn',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.true_negatives = self.add_weight(
            'tn',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives
            },
            y_true, y_pred,
            thresholds=self.thresholds,
            sample_weight=sample_weight)

    def result(self):
        true_scale  = math_ops.div_no_nan(self.true_sources,  self.false_negatives  + self.true_positives)
        false_scale = math_ops.div_no_nan(self.false_sources, self.false_positives  + self.true_negatives)
        result = self.true_positives*true_scale*0.54 - self.false_positives*false_scale
        return result[0] if len(self.thresholds) == 1 else math_ops.reduce_max(result)

    def reset_state(self):
        #pass
        n_thr = len(self.thresholds)
        backend.batch_set_value(
            [(v, np.zeros((n_thr,))) for v in self.variables])
    def reset_states(self):
        # Backwards compatibility alias of `reset_state`. New classes should
        # only implement `reset_state`.
        return self.reset_state()

    def get_config(self):

        config = {
            'thresholds': self.thresholds,
            'true_sources': self.true_sources,
            'false_sources': self.false_sources,
            'summation_method': self.summation_method.value,
        }
        base_config = super(Score, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


'''
class Score(tf.keras.metrics.Metric):
    def __init__(self, threshold, true_sources, false_sources,**kwargs):
        #from tensorflow.python.ops import init_ops
        super(Score, self).__init__(**kwargs)
        num_thresholds = 1
        self.true_sources = tf.convert_to_tensor(true_sources, dtype=tf.float32)
        self.false_sources = tf.convert_to_tensor(false_sources, dtype=tf.float32)
        self.true_positives  = tf.keras.metrics.TruePositives( thresholds = [threshold], name = 'tp')
        self.false_positives = tf.keras.metrics.FalsePositives(thresholds = [threshold], name = 'fp')
        self.true_negatives  = tf.keras.metrics.TrueNegatives( thresholds = [threshold], name = 'tn')
        self.false_negatives = tf.keras.metrics.FalseNegatives(thresholds = [threshold], name = 'fn')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.true_positives.update_state(  y_true, y_pred, sample_weight)
        self.false_positives.update_state( y_true, y_pred, sample_weight)
        self.true_negatives.update_state(  y_true, y_pred, sample_weight)
        self.false_negatives.update_state( y_true, y_pred, sample_weight)

    def reset_state(self):
        self.true_positives.reset_state()
        self.false_positives.reset_state()
        self.true_negatives.reset_state()
        self.false_negatives.reset_state()

    def result(self):
        from tensorflow.python.ops import math_ops

        tp = self.true_positives.result()
        fp = self.false_positives.result()
        tn = self.true_negatives.result()
        fn = self.false_negatives.result()
        true_scale  = math_ops.div_no_nan(self.true_sources,  fn  + tp)
        false_scale = math_ops.div_no_nan(self.false_sources, fp  + tn)

        result = tp*true_scale - fp*false_scale
        return result
'''

# for a position at index i,j,k in the original data cube,
# return the bounding cube from the original and denoised data cubes
# the cube dimensions are (zwidth*2, border*2, border*2)
def get_domain_index(i,j,k, denoised_data, border = 15):
    import glob
    index = -1
    xmin, xmax, ymin, ymax = 0,0,0,0
    for f in glob.glob(denoised_data):
        for t in f.split('_'):
            if 'task' in t: index = int(t.split('-')[-1])
            if 'x-' in t: xmin, xmax = [int(i) for i in t.split('-')[1:]]
            if 'y-' in t: ymin, ymax = [int(i) for i in t.split('-')[1:]]
        if i >= xmin+border and i <= xmax-border and j >= ymin+border and j <= ymax-border:
            break
    return index,  xmin, ymin, xmax, ymax

def get_cutouts(i,j,k, reader1, reader2, border = 15, zwidth = 50):

    # obtain cutouts around i,j,k
    xstart, xstop, ystart, ystop, zstart, zstop = [int(x) for x in [i-border, i+border, j-border, j+border, k-zwidth, k+zwidth]]
    cutout1 = reader1.safe_get_cube(xstart, xstop, ystart, ystop, zstart, zstop)
    cutout2 = reader2.safe_get_cube(xstart, xstop, ystart, ystop, zstart, zstop)

    return cutout1, cutout2

def get_cutout(i,j,k, reader1,  border = 15, zwidth = 50):

    # obtain cutouts around i,j,k
    xstart, xstop, ystart, ystop, zstart, zstop = [int(x) for x in [i-border, i+border, j-border, j+border, k-zwidth, k+zwidth]]
    cutout1 = reader1.safe_get_cube(xstart, xstop, ystart, ystop, zstart, zstop)

    return cutout1 

def make_training_set(domain, truthfilepath, outdir, wcl, wf, threshold = 10):
    sourcesTruth = TruthSource.catalog_to_sources_in_domain(truthfilepath, domain)
    for s in sourcesTruth:
        if s.line_flux_integral() < 10: continue
        xstart = int( s.x()-wcl )
        xstop  = int( s.x()+wcl )
        ystart = int( s.y()-wcl )
        ystop  = int( s.y()+wcl )
        zstart = int( s.z()-wf  )
        zstop  = int( s.z()+wf  )
        cutout = domain.safe_get_cube(xstart, xstop, ystart, ystop, zstart, zstop)
        if len(cutout) == 0:
            print("Source {0} too close to domain edge".format(s.ID()))
            continue
        outname = outdir+'/cube_{0}_{1}x{2}x{3}'.format(int(s.ID()), *(cutout.shape))
        print("Writing out cube for Source {0} ({2}, {3}, {4}) to {1}".format(s.ID(), outname, s.x(), s.y(), s.z()))
        np.save(outname, cutout)