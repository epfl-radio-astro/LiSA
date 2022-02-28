from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend

class Score(tf.keras.metrics.Metric):
    def __init__(self, thresholds, true_sources, false_sources,**kwargs):
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
        result = self.true_positives*true_scale*0.4 - self.false_positives*false_scale
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