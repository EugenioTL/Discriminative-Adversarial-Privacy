import numpy as np
from sklearn.model_selection import train_test_split
import math
import tensorflow as tf
import tensorflow.keras as tfk
from scipy import special
import random

from .factory import build_model

def build_shadow_datasets(X, y, epochs, batch_size, callbacks, learning_rate, models=10, seed=42, shadow_model_structure='default',
    verbose=0):
    
    logits_train_data = []
    logits_test_data = []
    cce_loss_train_data = []
    cce_loss_test_data = []
    cfce_loss_train_data = []
    cfce_loss_test_data = []
    ch_loss_train_data = []
    ch_loss_test_data = []
    labels_train_data = []
    labels_test_data = []
    
    input_shape = X.shape[1:]
    output_shape = y.shape[1:]
    
    for i in range(models):
        print('Model {}/{}'.format(i+1,models))
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=.1, random_state=seed+i, stratify=np.argmax(y,axis=1))
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=len(X_test), random_state=seed+i, stratify=np.argmax(y_train_val,axis=1))
        
        
        if len(input_shape) > 2:
            x_max = X_train.max(axis=(0,1,2))
            x_min = X_train.min(axis=(0,1,2))
        
            X_train = (X_train - x_min)/(x_max - x_min) * 2 - 1
            X_val = (X_val - x_min)/(x_max - x_min) * 2 - 1
            X_test = (X_test - x_min)/(x_max - x_min) * 2 - 1
        else:
            x_max = X_train.max(axis=(0))
            x_min = X_train.min(axis=(0))
        
            X_train = (X_train - x_min)/(x_max - x_min) * 2 - 1
            X_val = (X_val - x_min)/(x_max - x_min) * 2 - 1
            X_test = (X_test - x_min)/(x_max - x_min) * 2 - 1
        
        

        shadow_model_ready = False
        while not shadow_model_ready:
        
            shadow_model = build_model(
                input_shape,
                output_shape,
                kernel=random.choice([3,5,7]),
                units=2**np.random.randint(4,9),
                model_structure=shadow_model_structure,
            )
            
            history = shadow_model.fit(
                X_train,
                y_train,
                epochs=epochs,
                validation_data=(X_val,y_val),
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            ).history     

            pred = shadow_model.predict(X_train[:2],verbose=0)[0][0]
            if not math.isnan(pred):
                shadow_model_ready = True
            else:
                print('Shadow model training restarted...')

        if verbose>1:
            print('Top-1 Accuracy:',round(max(history['val_accuracy']),4))
        
        logits_train = shadow_model.predict(X_train, verbose=0)
        logits_test = shadow_model.predict(X_test, verbose=0)

        prob_train = special.softmax(logits_train, axis=-1)
        prob_test = special.softmax(logits_test, axis=-1)

        constant = tfk.backend.constant
        
        cce = tfk.losses.categorical_crossentropy
        cce_loss_train = cce(constant(y_train.astype(int)), constant(np.squeeze(prob_train)), from_logits=False).numpy()
        cce_loss_test = cce(constant(y_test.astype(int)), constant(np.squeeze(prob_test)), from_logits=False).numpy()

        cfce = tfk.losses.categorical_focal_crossentropy
        cfce_loss_train = cfce(constant(y_train.astype(int)), constant(np.squeeze(prob_train)), from_logits=False).numpy()
        cfce_loss_test = cfce(constant(y_test.astype(int)), constant(np.squeeze(prob_test)), from_logits=False).numpy()

        ch = tfk.losses.categorical_hinge
        ch_loss_train = ch(constant(y_train.astype(int)), constant(np.squeeze(prob_train))).numpy()
        ch_loss_test = ch(constant(y_test.astype(int)), constant(np.squeeze(prob_test))).numpy()
        

        logits_train_data.append(logits_train)
        logits_test_data.append(logits_test)
        cce_loss_train_data.append(cce_loss_train)
        cce_loss_test_data.append(cce_loss_test)
        cfce_loss_train_data.append(cfce_loss_train)
        cfce_loss_test_data.append(cfce_loss_test)
        ch_loss_train_data.append(ch_loss_train)
        ch_loss_test_data.append(ch_loss_test)
        labels_train_data.append(np.argmax(y_train,axis=1).astype(int))
        labels_test_data.append(np.argmax(y_test,axis=1).astype(int))
        
    logits_train_data = np.reshape(logits_train_data,(-1, output_shape[0]))
    logits_test_data = np.reshape(logits_test_data,(-1, output_shape[0]))
    cce_loss_train_data = np.reshape(cce_loss_train_data,(-1,1))
    cce_loss_test_data = np.reshape(cce_loss_test_data,(-1,1))
    cfce_loss_train_data = np.reshape(cfce_loss_train_data,(-1,1))
    cfce_loss_test_data = np.reshape(cfce_loss_test_data,(-1,1))
    ch_loss_train_data = np.reshape(ch_loss_train_data,(-1,1))
    ch_loss_test_data = np.reshape(ch_loss_test_data,(-1,1))
    labels_train_data = np.reshape(labels_train_data,(-1,1))
    labels_test_data = np.reshape(labels_test_data,(-1,1))
    
    train_data = np.concatenate((logits_train_data,cce_loss_train_data,cfce_loss_train_data,ch_loss_train_data,labels_train_data),axis=1)
    test_data = np.concatenate((logits_test_data,cce_loss_test_data,cfce_loss_test_data,ch_loss_test_data,labels_test_data),axis=1)
    
    data = np.concatenate((train_data,test_data),axis=0)
    labels = np.concatenate(((np.zeros((len(train_data),1))),(np.ones((len(test_data),1)))),axis=0)
    
    return data, labels