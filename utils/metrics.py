import tensorflow as tf

class AOP(tf.keras.metrics.Metric):
    def __init__(self, l=2, name='aop', **kwargs):
        super(AOP, self).__init__(name=name, **kwargs)
        self.aop_values = self.add_weight(name='aop', initializer='zeros')
        self.l = l

    def update_state(self, ACC, AUC, sample_weight=None):
        aop = ACC / (2 * tf.maximum(AUC, 0.5) ) ** self.l
        self.aop_values.assign(aop)

    def result(self):
        return self.aop_values

    def reset_states(self):
        self.aop_values.assign(0.0)