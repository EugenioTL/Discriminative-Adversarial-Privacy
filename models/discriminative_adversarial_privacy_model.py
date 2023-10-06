import tensorflow as tf
import tensorflow.keras as tfk
from ..utils.metrics import AOP

class ProtectedClassifier(tfk.Model):
    def __init__(self, classifier, discriminator):
        super(ProtectedClassifier, self).__init__()
        self.classifier = classifier
        self.softmax = tfk.Sequential(classifier.layers[-1:])
        self.softmax.build(classifier.layers[-2].output_shape)
        self.discriminator = discriminator
        self.beta = tf.Variable(1., trainable=False, name="beta", validate_shape=False) 

        self.loss_tracker = tfk.metrics.Mean(name='loss')
        self.classification_loss_tracker = tfk.metrics.Mean(name='cce')
        self.discriminator_loss_tracker = tfk.metrics.Mean(name='disc_bce')
        self.accuracy_tracker = tfk.metrics.CategoricalAccuracy(name="accuracy")

        self.val_loss_tracker = tfk.metrics.Mean(name='val_loss')
        self.val_classification_loss_tracker = tfk.metrics.Mean(name='val_cce')
        self.val_discriminator_loss_tracker = tfk.metrics.Mean(name='val_disc_bce')
        self.val_accuracy_tracker = tfk.metrics.CategoricalAccuracy(name="val_accuracy")
        self.val_auc_mia_tracker = tfk.metrics.AUC(name="val_auc_mia")
        self.aop_tracker = AOP(name="val_aop")
        
    @property
    def metrics(self):
        return[
            self.loss_tracker,
            self.classification_loss_tracker,
            self.discriminator_loss_tracker,
            self.accuracy_tracker,

            self.val_loss_tracker,
            self.val_classification_loss_tracker,
            self.val_discriminator_loss_tracker,
            self.val_accuracy_tracker,
            self.val_auc_mia_tracker,
            self.aop_tracker
        ]

    def compile(self, optimizer):
        super(ProtectedClassifier, self).compile()
        self.optimizer = optimizer

    @tf.function
    def train_step(self,data):
        
        train, test = data
        images, labels = train
        test_images, test_labels = test

        # step to optimize the classifier
        with tf.GradientTape() as tape:
            predictions = self.classifier(images)
            
            cce_extended_loss = tfk.losses.categorical_crossentropy(labels,predictions)
            cce_loss = tf.reduce_mean(cce_extended_loss)
            step_loss = cce_loss
            
        grads = tape.gradient(step_loss, self.classifier.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.classifier.trainable_weights))
        del tape
        
        # step to protect the classifier
        with tf.GradientTape() as tape:
            predictions = self.classifier(images)
            
            cce_extended_loss = tfk.losses.categorical_crossentropy(labels,predictions)
            cfce_extended_loss = tfk.losses.categorical_focal_crossentropy(labels,predictions)
            ch_extended_loss = tfk.losses.categorical_hinge(labels,predictions)

            mi_data = tf.concat([
                predictions,
                tf.expand_dims(cce_extended_loss,axis=-1),
                tf.expand_dims(cfce_extended_loss,axis=-1),
                tf.expand_dims(ch_extended_loss,axis=-1),
                tf.expand_dims(tf.cast(tf.argmax(labels,axis=1),tf.float32),axis=1)],axis=-1)
            # Inverted label
            mi_data = tf.where(tf.math.is_nan(mi_data), 0., mi_data)
            mi_labels = tf.ones((tf.shape(mi_data)[0],1))

            
            
            test_predictions = self.classifier(test_images)
            
            test_cce_extended_loss = tfk.losses.categorical_crossentropy(test_labels,test_predictions)
            test_cfce_extended_loss = tfk.losses.categorical_focal_crossentropy(test_labels,test_predictions)
            test_ch_extended_loss = tfk.losses.categorical_hinge(test_labels,test_predictions)
            
            test_mi_data = tf.concat([
                test_predictions,
                tf.expand_dims(test_cce_extended_loss,axis=-1),
                tf.expand_dims(test_cfce_extended_loss,axis=-1),
                tf.expand_dims(test_ch_extended_loss,axis=-1),
                tf.expand_dims(tf.cast(tf.argmax(test_labels,axis=1),tf.float32),axis=1)],axis=-1)
            # Inverted label
            test_mi_data = tf.where(tf.math.is_nan(test_mi_data), 0., test_mi_data)
            test_mi_labels = tf.zeros((tf.shape(test_mi_data)[0],1))

            
            mi_data = tf.concat((mi_data,test_mi_data),axis=0)
            mi_labels = tf.concat((mi_labels,test_mi_labels),axis=0)

            
            disc_predictions = self.discriminator(mi_data)
            disc_bce = tf.reduce_mean(tfk.losses.binary_crossentropy(mi_labels,disc_predictions))
            disc_bce = tf.where(tf.math.is_nan(disc_bce), 0., disc_bce)
            step_loss = disc_bce * self.beta
            
        grads = tape.gradient(step_loss, self.softmax.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.softmax.trainable_weights))

        loss = cce_loss + step_loss

        self.loss_tracker.update_state(loss)
        self.classification_loss_tracker.update_state(cce_loss)
        self.discriminator_loss_tracker.update_state(disc_bce)
        self.accuracy_tracker.update_state(labels,predictions)
        
        
        return{
            "loss": self.loss_tracker.result(),
            "cce": self.classification_loss_tracker.result(),
            "disc_bce": self.discriminator_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
            "beta": self.beta
        }
    
    
    @tf.function
    def test_step(self,data):
        images, labels = data
        
        val, train = data
        val_images, val_labels = val
        train_images, train_labels = train
        
        
        val_predictions = self.classifier(val_images)
        
        val_cce_extended_loss = tfk.losses.categorical_crossentropy(val_labels,val_predictions)
        val_cfce_extended_loss = tfk.losses.categorical_focal_crossentropy(val_labels,val_predictions)
        val_ch_extended_loss = tfk.losses.categorical_hinge(val_labels,val_predictions)
        val_cce_loss = tf.reduce_mean(val_cce_extended_loss)
        
        val_mi_data = tf.concat([
            val_predictions,
            tf.expand_dims(val_cce_extended_loss,axis=-1),
            tf.expand_dims(val_cfce_extended_loss,axis=-1),
            tf.expand_dims(val_ch_extended_loss,axis=-1),
            tf.expand_dims(tf.cast(tf.argmax(val_labels,axis=1),tf.float32),axis=1)],axis=-1)
        # Correct label
        val_mi_data = tf.where(tf.math.is_nan(val_mi_data), 0., val_mi_data)
        val_mi_labels = tf.ones((tf.shape(val_mi_data)[0],1))
        
        
        train_predictions = self.classifier(train_images)

        train_cce_extended_loss = tfk.losses.categorical_crossentropy(train_labels,train_predictions)
        train_cfce_extended_loss = tfk.losses.categorical_focal_crossentropy(train_labels,train_predictions)
        train_ch_extended_loss = tfk.losses.categorical_hinge(train_labels,train_predictions)
        train_mi_data = tf.concat([
            train_predictions,
            tf.expand_dims(train_cce_extended_loss,axis=-1),
            tf.expand_dims(train_cfce_extended_loss,axis=-1),
            tf.expand_dims(train_ch_extended_loss,axis=-1),
            tf.expand_dims(tf.cast(tf.argmax(train_labels,axis=1),tf.float32),axis=1)],axis=-1)
        # Correct label
        train_mi_data = tf.where(tf.math.is_nan(train_mi_data), 0., train_mi_data)
        train_mi_labels = tf.zeros((tf.shape(train_mi_data)[0],1))



        val_mi_data = tf.concat((val_mi_data,train_mi_data),axis=0)
        val_mi_labels = tf.concat((val_mi_labels,train_mi_labels),axis=0)

        
        val_disc_predictions = self.discriminator(val_mi_data)
        val_disc_bce = tf.reduce_mean(tfk.losses.binary_crossentropy(val_mi_labels,val_disc_predictions))
        val_disc_bce = tf.where(tf.math.is_nan(val_disc_bce), 0., val_disc_bce)

        val_loss = val_cce_loss + val_disc_bce * self.beta

        self.val_loss_tracker.update_state(val_loss)
        self.val_classification_loss_tracker.update_state(val_cce_loss)
        self.val_discriminator_loss_tracker.update_state(val_disc_bce)
        self.val_accuracy_tracker.update_state(val_labels,val_predictions)    
        self.val_auc_mia_tracker.update_state(val_mi_labels,val_disc_predictions)      
        self.aop_tracker.update_state(self.val_accuracy_tracker.result(),self.val_auc_mia_tracker.result(),10)
        
        return{
            "loss": self.val_loss_tracker.result(),
            "cce": self.val_classification_loss_tracker.result(),
            "disc_bce": self.val_discriminator_loss_tracker.result(),
            "accuracy": self.val_accuracy_tracker.result(),
            "auc_mia": self.val_auc_mia_tracker.result(),
            "aop": self.aop_tracker.result()
        }
    
    
    def call(self, data):
        return self.classifier(data)