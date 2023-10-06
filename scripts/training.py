import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from .privacy import test_model_privacy
from ..models.shadow_models import build_shadow_datasets
from ..models.factory import build_discriminator, build_model
from ..models.discriminative_adversarial_privacy_model import ProtectedClassifier
from ..utils.schedulers import BetaScheduler, CosineAnnealingScheduler

def dap_fit(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    epochs=200,
    batch_size=256,
    num_shadow_models=10,
    learning_rate=0.003,
    discriminator_learning_rate=0.01,
    patience=50,
    r = 0.2,
    min_beta = 0.25,
    max_beta = 4.,
    privacy_test=True,
    model_structure='default',
    shadow_model_structure='default',
    verbose=1,
    seed=42
):
    assert len(X_train) == len(y_train), 'X_train and y_train must have the same length.'
    assert len(X_val) == len(y_val), 'X_val and y_val must have the same length.'
    assert len(X_test) == len(y_test), 'X_test and y_test must have the same length.'
    assert r >= 0., 'The parameter \'r\' must be positive.'
    assert min_beta >= 0., 'The parameter \'min_beta\' must be positive.'
    assert max_beta >= min_beta, 'The parameter \'max_beta\' must be greater than the parameter \'m_beta\'.'
    
    global_history = {}
    
    ############################
    # Build the shadow dataset #
    ############################
    if verbose > 0: print("Shadow models training started...")
    
    if len(X_train.shape) > 2:
        shadow_callbacks = [
            tfk.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True),
            tfk.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, patience=max(2,patience), min_lr=1e-4, mode='max')
        ]
    else:
        shadow_callbacks = [
            tfk.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True),
            CosineAnnealingScheduler(T_max=epochs, eta_max=learning_rate),
        ]
        
    data, labels = build_shadow_datasets(X_val, y_val, models=num_shadow_models, batch_size=batch_size, epochs=epochs, callbacks=shadow_callbacks, learning_rate=learning_rate, shadow_model_structure=shadow_model_structure, verbose=verbose, seed=seed)
    mia_input_shape = data.shape[1:]
    if verbose > 0: print("Shadow models training completed!")

    X_train_mi, X_val_mi, y_train_mi, y_val_mi = train_test_split(data, labels, test_size=.1, random_state=seed, stratify=labels)

    X_train_mi_mean = X_train_mi.mean(axis=0)
    X_train_mi_var = X_train_mi.var(axis=0)

    disc_patience = np.minimum(np.maximum(patience*2,50),epochs//2)
    disc_callbacks = [
        tfk.callbacks.EarlyStopping(monitor='val_auc', patience=disc_patience, mode='max', restore_best_weights=True),
        CosineAnnealingScheduler(T_max=epochs, eta_max=discriminator_learning_rate),
        tfk.callbacks.TerminateOnNaN()
    ]

    
    ###########################
    # Train the discriminator #
    ###########################
    if verbose > 0: print("DAP discriminator training started...")
    discriminator_ready = False
    while not discriminator_ready:
    
        disc = build_discriminator(mia_input_shape, learning_rate=discriminator_learning_rate, mean=X_train_mi_mean, var=X_train_mi_var)
        discriminator_history = disc.fit(
            X_train_mi,
            y_train_mi,
            epochs=epochs,
            validation_data=(X_val_mi,y_val_mi),
            batch_size=batch_size,
            callbacks=disc_callbacks,
            verbose=0
        ).history

        pred = disc.predict(X_train_mi[:2],verbose=0)[0][0]
        if not math.isnan(pred):
            discriminator_ready = True
        else:
            print('Discriminator training restarted...')
    
    global_history['discriminator_history'] = discriminator_history
    global_history['discriminator'] = disc
    best_disc_epoch = np.argmax(global_history['discriminator_history']['val_auc'])
    print('Discriminator Accuracy',round(global_history['discriminator_history']['val_accuracy'][best_disc_epoch],4))
    print('Discriminator AUC',round(global_history['discriminator_history']['val_auc'][best_disc_epoch],4))
    if verbose > 0: print("DAP discriminator training completed!")
        
    repeated_X_val = X_val
    repeated_y_val = y_val    
    while len(X_train)>len(repeated_X_val):
        repeated_X_val = np.concatenate((repeated_X_val,X_val),axis=0)
        repeated_y_val = np.concatenate((repeated_y_val,y_val),axis=0)
    repeated_X_val = repeated_X_val[:len(X_train)]
    repeated_y_val = repeated_y_val[:len(y_train)]

    x_train_mean = (X_train/255).mean(axis=tuple(np.arange(len(X_train.shape)-1)))
    x_train_var = (X_train/255).var(axis=tuple(np.arange(len(X_train.shape)-1)))

    prot_patience = np.minimum(np.maximum(patience*2,50),epochs//2)
    prot_callbacks = [
        tfk.callbacks.EarlyStopping(monitor='val_aop', patience=prot_patience, restore_best_weights=True, mode='max'),
        tfk.callbacks.ReduceLROnPlateau(monitor="val_aop", factor=0.1, patience=max(2,patience), min_lr=1e-4, mode='max'),
        BetaScheduler(ratio=r, min_beta=min_beta, max_beta=max_beta),
        tfk.callbacks.TerminateOnNaN()
    ]
    
    if len(X_train.shape) > 2:
        prot_callbacks = [
            tfk.callbacks.EarlyStopping(monitor='val_aop', patience=prot_patience, restore_best_weights=True, mode='max'),
            tfk.callbacks.ReduceLROnPlateau(monitor="val_aop", factor=0.1, patience=max(2,patience), min_lr=1e-4, mode='max'),
            BetaScheduler(ratio=r, min_beta=min_beta, max_beta=max_beta),
            tfk.callbacks.TerminateOnNaN()
        ]
    else:
        prot_callbacks = [
            tfk.callbacks.EarlyStopping(monitor='val_aop', patience=prot_patience, restore_best_weights=True, mode='max'),
            CosineAnnealingScheduler(T_max=epochs, eta_max=learning_rate),
            BetaScheduler(ratio=r, min_beta=min_beta, max_beta=max_beta),
            tfk.callbacks.TerminateOnNaN()
        ]
    
    
    #####################################
    # Train the protected model via DAP #
    #####################################
    if verbose > 0: print("DAP protected model training started...")
    protected_model_ready = False
    while not protected_model_ready:
    
        model = ProtectedClassifier(
        classifier = build_model(X_train.shape[1:],y_train.shape[1:],model_structure=model_structure),
        discriminator = disc
        )
        
        if len(X_train.shape) > 2:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, weight_decay=5e-4)
            
        model.compile(optimizer=optimizer)
    
        protected_model_history = model.fit(
            [X_train,y_train],
            [repeated_X_val,repeated_y_val],
            epochs=epochs,
            validation_data=([X_val,y_val],[X_train[:len(X_val)],y_train[:len(y_val)]]),
            batch_size=batch_size,
            callbacks=prot_callbacks,
            verbose=verbose
        ).history
    
        pred = model.predict(X_train[:2],verbose=0)[0][0]
        if not math.isnan(pred):
            protected_model_ready = True
        else:
            print('Protected model training restarted...')
    global_history['protected_model_history'] = protected_model_history
    global_history['protected_model'] = model.classifier
    if verbose > 0: print("DAP protected model training completed!")
    
    if verbose > 0:
        ignore = 1
        
        best_epoch = np.argmax(protected_model_history['val_aop']) - ignore

        plt.rc('font', size=20) 
        plt.figure(figsize=(24,12))
        plt.subplot(4,1,1)
        plt.plot(protected_model_history['loss'][ignore:], label='Training Categorical Crossentropy', alpha=.8, linewidth=3, color='#FFA522')
        plt.plot(protected_model_history['val_loss'][ignore:], label='Validation Categorical Crossentropy', alpha=.8, linewidth=3, color='#4081EA')
        plt.legend()
        plt.grid(alpha=.3)

        plt.subplot(4,1,2)
        plt.plot(protected_model_history['accuracy'][ignore:], label='Training Accuracy', alpha=.8, linewidth=3, color='#FFA522')
        plt.plot(protected_model_history['val_accuracy'][ignore:], label='Validation Accuracy', alpha=.8, linewidth=3, color='#4081EA')
        plt.legend()
        plt.grid(alpha=.3)

        plt.subplot(4,1,3)
        plt.plot(protected_model_history['val_aop'][ignore:], label='AOP', alpha=.8, linewidth=3, color='#4081EA')
        plt.plot(best_epoch, protected_model_history['val_aop'][best_epoch + ignore], marker="*", markersize=10, markerfacecolor="#4081EA")
        plt.legend()
        plt.grid(alpha=.3)

        plt.subplot(4,1,4)
        plt.plot(protected_model_history['beta'][ignore:], label='Beta', alpha=.8, linewidth=3, color='#4B9328')
        plt.legend()
        plt.grid(alpha=.3)
        plt.show()
        
        
    ##################################
    # Test protected model's privacy #
    ##################################
    if privacy_test:
        if verbose > 0: print("Membership Inference Attack test started...")
        test_model_privacy(model, X_train, y_train, X_test, y_test)
        if verbose > 0: print("Membership Inference Attack test completed!")
            
    return global_history