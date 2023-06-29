seed = 42

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
np.random.seed(seed)

import logging
import tensorflow as tf
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

import random
random.seed(seed)

tfk = tf.keras
tfkl = tf.keras.layers

from tensorflow.keras import backend as K
import tf2onnx
import onnxruntime as rt
rt.set_default_logger_severity(3)

import sklearn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.utils.class_weight import compute_class_weight
import cv2


from scipy import special
import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData, AttackType, SlicingSpec



def test_model_privacy(model, X_train, y_train, X_test, y_test):
    
    def extract_auc(text,first,last):
        pattern = r'\b%s\b(.*?)\b%s\b' % (first, last)
        match = re.search(pattern, text)
        if match:
            auc = match.group(1).strip()
            return auc
        else:
            return None
    
    logits_train = model.predict(X_train, verbose=0)
    logits_test = model.predict(X_test, verbose=0)

    prob_train = special.softmax(logits_train, axis=-1)
    prob_test = special.softmax(logits_test, axis=-1)

    cce = tf.keras.backend.categorical_crossentropy
    constant = tf.keras.backend.constant

    loss_train = cce(constant(y_train.astype(int)), constant(prob_train), from_logits=False).numpy()
    loss_test = cce(constant(y_test.astype(int)), constant(prob_test), from_logits=False).numpy()
    
    
    attack_input = AttackInputData(
      logits_train = logits_train,
      logits_test = logits_test,
      loss_train = loss_train,
      loss_test = loss_test,
      labels_train = np.argmax(y_train,axis=1).astype(int),
      labels_test = np.argmax(y_test,axis=1).astype(int)
    )

    slicing_spec = SlicingSpec(
        entire_dataset = True,
        by_classification_correctness = False,
        by_class = False
    )

    attack_types = [
        AttackType.THRESHOLD_ATTACK,
        AttackType.LOGISTIC_REGRESSION,
        AttackType.RANDOM_FOREST
    ] 

    attacks_result = mia.run_attacks(attack_input=attack_input,
                                     slicing_spec=slicing_spec,
                                     attack_types=attack_types,)

    print(attacks_result.get_result_with_max_auc())
    print(plotting.plot_roc_curve(attacks_result.get_result_with_max_auc().roc_curve))
    print(attacks_result.summary(by_slices = True))
    
    auc = float(extract_auc(attacks_result.summary(by_slices = True),'of', ' on'))
    
    return auc


class AOP(tf.keras.metrics.Metric):
    def __init__(self, name='aop', **kwargs):
        super(AOP, self).__init__(name=name, **kwargs)
        self.aop_values = self.add_weight(name='aop', initializer='zeros')

    def update_state(self, ACC, AUC, l, sample_weight=None):
        aop = ACC / (2 * tf.maximum(AUC, 0.5) ) ** l
        self.aop_values.assign(aop)

    def result(self):
        return self.aop_values

    def reset_states(self):
        self.aop_values.assign(0.0)




def build_model(
    input_shape,
    output_shape,
    task,
    seed=seed,
    dropout=0.5,
    **kwargs
):
    if len(input_shape) == 3: # Image
        if task == 'classification':
            return build_image_classifier(input_shape, output_shape, seed, dropout)            
        else:
            return build_image_regressor(input_shape, output_shape, seed, dropout) 
    elif len(input_shape) == 1: # Tabular
        if task == 'classification':
            return build_tabular_classifier(input_shape, output_shape, seed, dropout)            
        else:
            return build_tabular_regressor(input_shape, output_shape, seed, dropout) 
    return None

def build_image_regressor(
    input_shape,
    output_shape,
    seed,
    dropout,
    **kwargs
):
    input_layer = tfkl.Input(input_shape, name='input_layer')
    
    x = tf.keras.layers.RandomFlip(mode='horizontal', name='random_flip')(input_layer)
    x = tfkl.Conv2D(64, 3, padding='same', activation='relu', name='conv1')(x)
    x = tfkl.MaxPooling2D(name='mp1')(x)
    x = tfkl.Conv2D(128, 3, padding='same', activation='relu', name='conv2')(x)
    x = tfkl.MaxPooling2D(name='mp2')(x)
    x = tfkl.Conv2D(256, 3, padding='same', activation='relu', name='conv3')(x)
    
    x = tfkl.GlobalAveragePooling2D(name='gap')(x)
    x = tfkl.Dropout(dropout, seed=seed, name='dropout')(x)
    output_layer = tfkl.Dense(output_shape[-1], activation='tanh', name='output_layer')(x)
    
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='CNN')
    
    loss = tfk.losses.MeanSquaredError()
    optimizer = tfk.optimizers.Adam()
    
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
    
    return model
    
def build_tabular_regressor(
    input_shape,
    output_shape,
    seed,
    dropout,
    **kwargs
):
    input_layer = tfkl.Input(input_shape, name='input_layer')
    
    x = tfkl.Dense(64, activation='relu', name='fc1')(input_layer)
    x = tfkl.Dense(128, activation='relu', name='fc2')(x)
    x = tfkl.Dense(256, activation='relu', name='fc3')(x)
    
    x = tfkl.Dropout(dropout, seed=seed, name='dropout')(x)
    output_layer = tfkl.Dense(output_shape[-1], activation='tanh', name='output_layer')(x)
    
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='FFNN')
    
    loss = tfk.losses.MeanSquaredError()
    optimizer = tfk.optimizers.Adam()
    
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
    
    return model
    
def build_image_classifier(
    input_shape,
    output_shape,
    seed,
    dropout,
    **kwargs
):
    input_layer = tfkl.Input(input_shape, name='input_layer')
    
    x = tf.keras.layers.RandomFlip(mode='horizontal', name='random_flip')(input_layer)
    x = tfkl.Conv2D(64, 3, padding='same', activation='relu', name='conv1')(x)
    x = tfkl.MaxPooling2D(name='mp1')(x)
    x = tfkl.Conv2D(128, 3, padding='same', activation='relu', name='conv2')(x)
    x = tfkl.MaxPooling2D(name='mp2')(x)
    x = tfkl.Conv2D(256, 3, padding='same', activation='relu', name='conv3')(x)
    
    x = tfkl.GlobalAveragePooling2D(name='gap')(x)
    x = tfkl.Dropout(dropout, seed=seed, name='dropout')(x)
    output_layer = tfkl.Dense(output_shape[-1], activation='softmax', name='output_layer')(x)
    
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='CNN')
    
    loss = tfk.losses.CategoricalCrossentropy()
    optimizer = tfk.optimizers.Adam()
    
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    return model
    
def build_tabular_classifier(
    input_shape,
    output_shape,
    seed,
    dropout,
    **kwargs
):
    input_layer = tfkl.Input(input_shape, name='input_layer')
    
    x = tfkl.Dense(64, activation='relu', name='fc1')(input_layer)
    x = tfkl.Dense(128, activation='relu', name='fc2')(x)
    x = tfkl.Dense(256, activation='relu', name='fc3')(x)
    
    x = tfkl.Dropout(dropout, seed=seed, name='dropout')(x)
    output_layer = tfkl.Dense(output_shape[-1], activation='softmax', name='output_layer')(x)
    
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='FFNN')
    
    loss = tfk.losses.CategoricalCrossentropy()
    optimizer = tfk.optimizers.Adam()
    
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    return model




def build_shadow_datasets(X, y, callbacks=callbacks, seed=seed, models=10, epochs=epochs, batch_size=batch_size):
    
    logits_train_data = []
    logits_test_data = []
    loss_train_data = []
    loss_test_data = []
    labels_train_data = []
    labels_test_data = []
    
    input_shape = X.shape[1:]
    output_shape = y.shape[1:]
    
    for i in range(models):
        print('Model {}/{}'.format(i+1,models))
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=.1, random_state=seed+i, stratify=np.argmax(y,axis=1))
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=len(X_test), random_state=seed+i, stratify=np.argmax(y_train_val,axis=1))   
        
        shadow_model = build_model(input_shape,output_shape,'classification')
        history = shadow_model.fit(
            X_train,
            y_train,
            epochs=epochs,
            validation_data=(X_val,y_val),
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        ).history
        
        
        logits_train = shadow_model.predict(X_train, verbose=0)
        logits_test = shadow_model.predict(X_test, verbose=0)

        prob_train = special.softmax(logits_train, axis=-1)
        prob_test = special.softmax(logits_test, axis=-1)

        cce = tfk.backend.categorical_crossentropy
        constant = tfk.backend.constant

        loss_train = cce(constant(y_train.astype(int)), constant(np.squeeze(prob_train)), from_logits=False).numpy()
        loss_test = cce(constant(y_test.astype(int)), constant(np.squeeze(prob_test)), from_logits=False).numpy()
        
        

        logits_train_data.append(logits_train)
        logits_test_data.append(logits_test)
        loss_train_data.append(loss_train)
        loss_test_data.append(loss_test)
        labels_train_data.append(np.argmax(y_train,axis=1).astype(int))
        labels_test_data.append(np.argmax(y_test,axis=1).astype(int))
        
    logits_train_data = np.reshape(logits_train_data,(-1, output_shape[0]))
    logits_test_data = np.reshape(logits_test_data,(-1, output_shape[0]))
    loss_train_data = np.reshape(loss_train_data,(-1,1))
    loss_test_data = np.reshape(loss_test_data,(-1,1))
    labels_train_data = np.reshape(labels_train_data,(-1,1))
    labels_test_data = np.reshape(labels_test_data,(-1,1))
    
    train_data = np.concatenate((logits_train_data,loss_train_data,labels_train_data),axis=1)
    test_data = np.concatenate((logits_test_data,loss_test_data,labels_test_data),axis=1)
    
    data = np.concatenate((train_data,test_data),axis=0)
    labels = np.concatenate(((np.ones((len(train_data),1))),(np.zeros((len(test_data),1)))),axis=0)
    
    return data, labels



def build_discriminator(
    input_shape, 
    seed=seed,
    **kwargs
):
    
    input_layer = tfkl.Input(input_shape, name='input_layer')
    
    x = tfkl.Dense(64, activation='relu', name='fc1')(input_layer)
    x = tfkl.Dense(128, activation='relu', name='fc2')(x)
    x = tfkl.Dense(256, activation='relu', name='fc3')(x)
    
    x = tfkl.Dropout(0.5, seed=seed)(x)
    x = tfkl.Dense(1, name='output_layer')(x)
    output_layer = tfkl.Activation('sigmoid', name='sigmoid')(x)
    discriminator = tfk.Model(inputs=input_layer, outputs=output_layer, name='Discriminator')
    discriminator.compile(loss=tfk.losses.BinaryCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics=['accuracy','AUC'])

    return discriminator



class BetaScheduler(tfk.callbacks.Callback):
    def __init__(self, ratio=0.5, min_beta=1e-6, max_beta=1e+6):
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



class ProtectedClassifier(tfk.Model):
    def __init__(self, classifier, discriminator):
        super(ProtectedClassifier, self).__init__()
        self.classifier = classifier
        self.softmax = tfk.Sequential(classifier.layers[-3:])
        self.softmax.build(classifier.layers[-4].output_shape)
        self.discriminator = discriminator
        self.beta = tf.Variable(1., trainable=False, name="beta", validate_shape=False) 
        
        self.loss_tracker = tfk.metrics.Mean(name='loss')
        self.classification_loss_tracker = tfk.metrics.Mean(name='cce')
        self.discriminator_loss_tracker = tfk.metrics.Mean(name='disc_bce')
        self.accuracy_tracker = tfk.metrics.CategoricalAccuracy(name="accuracy")
        self.auc_mia_tracker = tfk.metrics.AUC(name="auc_mia")
        self.aop_tracker = AOP()
        
        self.val_loss_tracker = tfk.metrics.Mean(name='val_loss')
        self.val_classification_loss_tracker = tfk.metrics.Mean(name='val_cce')
        self.val_discriminator_loss_tracker = tfk.metrics.Mean(name='val_disc_bce')
        self.val_accuracy_tracker = tfk.metrics.CategoricalAccuracy(name="val_accuracy")
        
    @property
    def metrics(self):
        return[
            self.loss_tracker,
            self.classification_loss_tracker,
            self.discriminator_loss_tracker,
            self.accuracy_tracker,
            self.auc_mia_tracker,
            self.aop_tracker,

            self.val_loss_tracker,
            self.val_classification_loss_tracker,
            self.val_discriminator_loss_tracker,
            self.val_accuracy_tracker
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
            cce_loss = tf.reduce_mean(cce_extended_loss)
            
            # same input of the discriminator
            mi_data = tf.concat([
                predictions,
                tf.expand_dims(cce_extended_loss,axis=-1),
                tf.expand_dims(tf.cast(tf.argmax(labels,axis=1),tf.float32),axis=1)],axis=-1)
            
            # take only misclassifications (can be changed)
#             mi_data = mi_data[tf.math.not_equal(tf.cast(tf.argmax(mi_data[:,:predictions.shape[-1]],axis=-1), tf.float32),mi_data[:,-1])]
            # prevent nan (not mandatory)
            mi_data = tf.where(tf.math.is_nan(mi_data), 0., mi_data)
            
            
            mi_labels = tf.zeros((tf.shape(mi_data)[0],1))

            
            disc_predictions = self.discriminator(mi_data)
            disc_bce = tf.reduce_mean(tfk.losses.binary_crossentropy(mi_labels,disc_predictions))
            
            disc_bce = tf.where(tf.math.is_nan(disc_bce), 0., disc_bce)
            step_loss = disc_bce * self.beta
            
            
            
            mi_labels = tf.ones((tf.shape(mi_data)[0],1))
            
            test_predictions = self.classifier(test_images)
            test_cce_extended_loss = tfk.losses.categorical_crossentropy(test_labels,test_predictions)
            test_mi_data = tf.concat([
                test_predictions,
                tf.expand_dims(test_cce_extended_loss,axis=-1),
                tf.expand_dims(tf.cast(tf.argmax(test_labels,axis=1),tf.float32),axis=1)],axis=-1)
            test_mi_data = test_mi_data[tf.math.not_equal(tf.cast(tf.argmax(test_mi_data[:,:test_predictions.shape[-1]],axis=-1), tf.float32),test_mi_data[:,-1])]
            test_mi_data = tf.where(tf.math.is_nan(test_mi_data), 0., test_mi_data)
            test_mi_labels = tf.zeros((tf.shape(test_mi_data)[0],1))
            
            mi_data = tf.concat((mi_data,test_mi_data),axis=0)
            mi_labels = tf.concat((mi_labels,test_mi_labels),axis=0)
            
            disc_predictions = self.discriminator(mi_data)
            
            
            
        grads = tape.gradient(step_loss, self.softmax.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.softmax.trainable_weights))

        loss = cce_loss + disc_bce * self.beta
        
        self.loss_tracker.update_state(loss)
        self.classification_loss_tracker.update_state(cce_loss)
        self.discriminator_loss_tracker.update_state(disc_bce)
        self.accuracy_tracker.update_state(labels,predictions)
        self.auc_mia_tracker.update_state(mi_labels,disc_predictions)      
        self.aop_tracker.update_state(self.accuracy_tracker.result(),self.auc_mia_tracker.result(),2)
        
        
        return{
            "loss": self.loss_tracker.result(),
            "cce": self.classification_loss_tracker.result(),
            "disc_bce": self.discriminator_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
            "auc_mia": self.auc_mia_tracker.result(),
            "aop": self.aop_tracker.result(),
            "beta": self.beta
        }
    
    
    @tf.function
    def test_step(self,data):
        images, labels = data


        predictions = self.classifier(images)

        cce_extended_loss = tfk.losses.categorical_crossentropy(labels,predictions)
        cce_loss = tf.reduce_mean(cce_extended_loss)

        mi_data = tf.concat([
            predictions,
            tf.expand_dims(cce_extended_loss,axis=-1),
            tf.expand_dims(tf.cast(tf.argmax(labels,axis=1),tf.float32),axis=1)],axis=-1)


#         mi_data = mi_data[tf.math.not_equal(tf.cast(tf.argmax(mi_data[:,:predictions.shape[-1]],axis=-1), tf.float32),mi_data[:,-1])]
        mi_data = tf.where(tf.math.is_nan(mi_data), 0., mi_data)
        mi_labels = tf.zeros((tf.shape(mi_data)[0],1))

        disc_predictions = self.discriminator(mi_data)
        disc_bce = tf.reduce_mean(tfk.losses.binary_crossentropy(mi_labels,disc_predictions))
        disc_bce = tf.where(tf.math.is_nan(disc_bce), 0., disc_bce)
        disc_bce = disc_bce 
        loss = cce_loss + disc_bce * self.beta
        
        self.val_loss_tracker.update_state(loss)
        self.val_classification_loss_tracker.update_state(cce_loss)
        self.val_discriminator_loss_tracker.update_state(disc_bce)
        self.val_accuracy_tracker.update_state(labels,predictions)    
        
        
        return{
            "loss": self.val_loss_tracker.result(),
            "cce": self.val_classification_loss_tracker.result(),
            "disc_bce": self.val_discriminator_loss_tracker.result(),
            "accuracy": self.val_accuracy_tracker.result(),
        }
    
    
    def call(self, data):
        return self.classifier(data)



def dap_fit(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    task='classification',
    epochs=100,
    batch_size=128,
    num_shadow_models=10,
    misclassified_data=False,
    optimizer=None,
    r = 0.2,
    min_beta = 0.,
    max_beta = 2.,
    privacy_test=True,
    verbose=1
):
    assert len(X_train) == len(y_train), 'X_train and y_train must have the same length.'
    assert len(X_val) == len(y_val), 'X_val and y_val must have the same length.'
    assert len(X_test) == len(y_test), 'X_test and y_test must have the same length.'
    assert task in ['classification'], 'The current allowed tasks are \'classification\' and \'regression\'.'
    assert r >= 0., 'The parameter \'r\' must be positive.'
    assert min_beta >= 0., 'The parameter \'min_beta\' must be positive.'
    assert max_beta >= min_beta, 'The parameter \'max_beta\' must be greater than the parameter \'m_beta\'.'
    
    global_history = {}
    
    if task == 'regression':
        callbacks=[
            tfk.callbacks.EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True, mode='min'),
            tfk.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.1, patience=10, min_lr=1e-4, mode='min')
        ]
    elif task == 'classification':
        callbacks = [
        tfk.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, mode='max'),
        tfk.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, patience=10, min_lr=1e-4, mode='max')
        ]
    
    
    if verbose > 0: print("Shadow models training started...")
    data, labels = build_shadow_datasets(X_test, y_test, models=num_shadow_models)
    mia_input_shape = data.shape[1:]
    if verbose > 0: print("Shadow models training completed!")

    
    if misclassified_data:
        wrong_data = data[np.argmax(data[:,:output_shape[0]],axis=1) != data[:,-1]]
        wrong_labels = labels[np.argmax(data[:,:output_shape[0]],axis=1) != data[:,-1]]
        print('Membership Inference Attack\nFeatures space {} \tLabels space {}'.format(wrong_data.shape, wrong_labels.shape))
        X_train_mi, X_val_mi, y_train_mi, y_val_mi = train_test_split(wrong_data, wrong_labels, test_size=.2, random_state=seed, stratify=wrong_labels)
    else:
        print('Membership Inference Attack\nFeatures space {} \tLabels space {}'.format(data.shape, labels.shape))
        X_train_mi, X_val_mi, y_train_mi, y_val_mi = train_test_split(data, labels, test_size=.2, random_state=seed, stratify=labels)
    
    
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_mi), y=np.squeeze(y_train_mi))
    class_weights = dict(zip(np.unique(y_train_mi), class_weights))
    
    if verbose > 0: print("DAP discriminator training started...")
    disc = build_discriminator(mia_input_shape)
    discriminator_history = disc.fit(
        X_train_mi,
        y_train_mi,
        epochs=epochs,
        validation_data=(X_val_mi,y_val_mi),
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=0
    ).history
    global_history['discriminator_history'] = discriminator_history
    global_history['discriminator'] = disc
    if verbose > 0: print("DAP discriminator training completed!")
        
    repeated_X_val = X_val
    repeated_y_val = y_val    
    while len(X_train)>len(repeated_X_val):
        repeated_X_val = np.concatenate((repeated_X_val,X_val),axis=0)
        repeated_y_val = np.concatenate((repeated_y_val,y_val),axis=0)
    repeated_X_val = repeated_X_val[:len(X_train)]
    repeated_y_val = repeated_y_val[:len(y_train)]
    
    
    model = ProtectedClassifier(
        classifier = build_model(X_train.shape[1:],y_train.shape[1:],task,dropout=0.),
        discriminator = disc
    )
    
    if optimizer == None:
        optimizer = tfk.optimizers.Adam()  
    model.compile(optimizer=optimizer)
    
    if verbose > 0: print("DAP protected model training started...")
    protected_model_history = model.fit(
        [X_train,y_train],
        [repeated_X_val,repeated_y_val],
        epochs=epochs,
        validation_data=(X_val,y_val),
        batch_size=batch_size,
        callbacks=[
            tfk.callbacks.EarlyStopping(monitor='aop', patience=15, restore_best_weights=True, mode='max'),
            tfk.callbacks.ReduceLROnPlateau(monitor="aop", factor=0.1, patience=10, min_lr=1e-4, mode='max'),
            BetaScheduler(ratio=r, min_beta=min_beta, max_beta=max_beta)
        ]
    ).history
    global_history['protected_model_history'] = protected_model_history
    global_history['protected_model'] = model.classifier
    if verbose > 0: print("DAP protected model training completed!")
    
    if verbose > 0:
        best_epoch = np.argmax(protected_model_history['aop'])

        plt.rc('font', size=20) 
        plt.figure(figsize=(24,12))
        plt.subplot(4,1,1)
        plt.plot(protected_model_history['loss'], label='Training Loss', alpha=.8, linewidth=3, color='#FFA522')
        plt.plot(protected_model_history['val_loss'], label='Validation Loss', alpha=.8, linewidth=3, color='#4081EA')
        plt.legend()
        plt.grid(alpha=.3)

        plt.subplot(4,1,2)
        plt.plot(protected_model_history['accuracy'], label='Training Accuracy', alpha=.8, linewidth=3, color='#FFA522')
        plt.plot(protected_model_history['val_accuracy'], label='Validation Accuracy', alpha=.8, linewidth=3, color='#4081EA')
        plt.plot(best_epoch, protected_model_history['val_accuracy'][best_epoch], marker="*", markersize=10, markerfacecolor="#4081EA")
        plt.legend()
        plt.grid(alpha=.3)

        plt.subplot(4,1,3)
        plt.plot(protected_model_history['aop'], label='AOP', alpha=.8, linewidth=3, color='#4081EA')
        plt.legend()
        plt.grid(alpha=.3)

        plt.subplot(4,1,4)
        plt.plot(protected_model_history['beta'], label='Beta', alpha=.8, linewidth=3, color='#4B9328')
        plt.legend()
        plt.grid(alpha=.3)
        plt.show()
        
    if privacy_test:
        if verbose > 0: print("Membership Inference Attack test started...")
        test_model_privacy(model, X_train, y_train, X_test, y_test)
        if verbose > 0: print("Membership Inference Attack test completed!")
            
    return global_history
