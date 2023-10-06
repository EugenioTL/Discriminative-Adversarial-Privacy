import math
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import backend as K

class BetaScheduler(tf.keras.callbacks.Callback):
    def __init__(self, ratio=0.5, min_beta=0.5, max_beta=2.):
        super(BetaScheduler, self).__init__()
        self.ratio = ratio
        self.min_beta = min_beta
        self.max_beta = max_beta
    
    def on_epoch_end(self, epoch, logs=None):
        cce_loss = logs.get("val_cce")
        bce_loss = logs.get("val_disc_bce")
        if(bce_loss==0 or cce_loss==0):
            return
        if(bce_loss==None or cce_loss==None):
            return
        new_beta = tf.math.multiply(cce_loss,self.ratio)
        new_beta = tf.math.divide(new_beta,bce_loss)
        new_beta = tf.math.maximum(new_beta,self.min_beta)
        new_beta = tf.math.minimum(new_beta,self.max_beta)
        tfk.backend.set_value(self.model.beta, new_beta)
        
class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, T_max, eta_max=0.01, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)