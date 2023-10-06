import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

def build_model(
    input_shape,
    output_shape,
    model_structure,
    learning_rate=0.003,
    kernel=3,
    units=128,
    **kwargs
):
    if len(input_shape) == 3: # Image
        return build_image_classifier(input_shape=input_shape, output_shape=output_shape, learning_rate=learning_rate, kernel=kernel, model_structure=model_structure)            
    elif len(input_shape) == 1: # Tabular
        return build_tabular_classifier(input_shape=input_shape, output_shape=output_shape, learning_rate=learning_rate, units=units)            
    return None

    
def build_image_classifier(
    input_shape,
    output_shape,
    learning_rate,
    kernel,
    model_structure='default',
    **kwargs
):
    input_layer = tfkl.Input(input_shape, name='input_layer')

    x = tfkl.RandomFlip(mode='horizontal')(input_layer)

    if model_structure == 'deep':

        x1 = tfkl.Conv2D(64, kernel, padding='same', name='conv11', kernel_initializer=tfk.initializers.HeNormal())(x)
        x1 = tfkl.BatchNormalization(name='bn11')(x1)
        x1 = tfkl.ReLU(name='relu11')(x1)
        x1 = tfkl.Conv2D(64, kernel, padding='same', name='conv12', kernel_initializer=tfk.initializers.HeNormal())(x1)
        x1 = tfkl.BatchNormalization(name='bn12')(x1)
        x1 = tfkl.ReLU(name='relu12')(x1)
        x = tfkl.Concatenate(axis=-1, name='concat1')([x,x1])
        x = tfkl.MaxPooling2D(name='mp1')(x)
        x1 = tfkl.Conv2D(128, kernel, padding='same', name='conv21', kernel_initializer=tfk.initializers.HeNormal())(x)
        x1 = tfkl.BatchNormalization(name='bn21')(x1)
        x1 = tfkl.ReLU(name='relu21')(x1)
        x1 = tfkl.Conv2D(128, kernel, padding='same', name='conv22', kernel_initializer=tfk.initializers.HeNormal())(x1)
        x1 = tfkl.BatchNormalization(name='bn22')(x1)
        x1 = tfkl.ReLU(name='relu22')(x1)
        x = tfkl.Concatenate(axis=-1, name='concat2')([x,x1])
        x = tfkl.MaxPooling2D(name='mp2')(x)
        x1 = tfkl.Conv2D(256, kernel, padding='same', name='conv31', kernel_initializer=tfk.initializers.HeNormal())(x)
        x1 = tfkl.BatchNormalization(name='bn31')(x1)
        x1 = tfkl.ReLU(name='relu31')(x1)
        x1 = tfkl.Conv2D(256, kernel, padding='same', name='conv32', kernel_initializer=tfk.initializers.HeNormal())(x1)
        x1 = tfkl.BatchNormalization(name='bn32')(x1)
        x1 = tfkl.ReLU(name='relu32')(x1)
        x1 = tfkl.Conv2D(256, kernel, padding='same', name='conv33', kernel_initializer=tfk.initializers.HeNormal())(x1)
        x1 = tfkl.BatchNormalization(name='bn33')(x1)
        x1 = tfkl.ReLU(name='relu33')(x1)
        x = tfkl.Concatenate(axis=-1, name='concat3')([x,x1])
        x = tfkl.MaxPooling2D(name='mp3')(x)
        x1 = tfkl.Conv2D(512, kernel, padding='same', name='conv41', kernel_initializer=tfk.initializers.HeNormal())(x)
        x1 = tfkl.BatchNormalization(name='bn41')(x1)
        x1 = tfkl.ReLU(name='relu41')(x1)
        x1 = tfkl.Conv2D(512, kernel, padding='same', name='conv42', kernel_initializer=tfk.initializers.HeNormal())(x1)
        x1 = tfkl.BatchNormalization(name='bn42')(x1)
        x1 = tfkl.ReLU(name='relu42')(x1)
        x1 = tfkl.Conv2D(512, kernel, padding='same', name='conv43', kernel_initializer=tfk.initializers.HeNormal())(x1)
        x1 = tfkl.BatchNormalization(name='bn43')(x1)
        x1 = tfkl.ReLU(name='relu43')(x1)
        x = tfkl.Concatenate(axis=-1, name='concat4')([x,x1])
    
        x = tfkl.GlobalAveragePooling2D(name='gap')(x)
        x = tfkl.Dropout(0.5, name='dropout')(x)
        output_layer = tfkl.Dense(output_shape[-1], activation='softmax', kernel_initializer=tfk.initializers.GlorotNormal(), name='output_layer')(x)
    else:
        x = tfkl.Conv2D(64,kernel,padding='same', kernel_initializer=tfk.initializers.HeNormal())(x)
        x = tfkl.ReLU()(x)
        x = tfkl.MaxPooling2D()(x)
        x = tfkl.Conv2D(128,kernel,padding='same', kernel_initializer=tfk.initializers.HeNormal())(x)
        x = tfkl.ReLU()(x)
        x = tfkl.MaxPooling2D()(x)
        x = tfkl.Conv2D(256,kernel,padding='same', kernel_initializer=tfk.initializers.HeNormal())(x)
        x = tfkl.ReLU()(x)
        
        x = tfkl.GlobalAveragePooling2D(name='gap')(x)
        output_layer = tfkl.Dense(output_shape[-1], activation='softmax', kernel_initializer=tfk.initializers.GlorotNormal(), name='output_layer')(x)
        
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='CNN')
    
    model.compile(
        loss=tfk.losses.CategoricalCrossentropy(), 
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
        metrics=['accuracy']
    )
    
    return model

def dense_residual_block(
    x, 
    units,
    **kwargs
):
    x1 = tfkl.BatchNormalization()(x)
    x1 = tfkl.ReLU()(x1)
    x1 = tfkl.Dense(units, kernel_initializer=tfk.initializers.HeNormal())(x1)
    x1 = tfkl.BatchNormalization()(x1)
    x1 = tfkl.ReLU()(x1)
    x1 = tfkl.Dense(units, kernel_initializer=tfk.initializers.HeNormal())(x1)
    x1 = tfkl.BatchNormalization()(x1)
    x1 = tfkl.ReLU()(x1)
    x1 = tfkl.Dense(units, kernel_initializer=tfk.initializers.HeNormal())(x1)

    x2 = tfkl.Dense(units, kernel_initializer=tfk.initializers.HeNormal())(x)

    return tfkl.Add()([x1,x2])


def build_tabular_classifier(
    input_shape,
    output_shape,
    learning_rate,
    units=128,
    **kwargs
):
    input_layer = tfkl.Input(input_shape, name='input_layer')
    
    x = dense_residual_block(input_layer, units=units)
    x = tfkl.Dropout(0.2)(x)
    x = dense_residual_block(x, units=units)
    x = tfkl.Dropout(0.2)(x)
    x = dense_residual_block(x, units=units)
    x = tfkl.Dropout(0.2)(x)
    output_layer = tfkl.Dense(output_shape[-1], activation='softmax', kernel_initializer=tfk.initializers.GlorotNormal(), name='output_layer')(x)
    
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='FFNN')
    
    model.compile(
        loss=tfk.losses.CategoricalCrossentropy(), 
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, weight_decay=5e-4), 
        metrics=['accuracy','AUC']
    )
    
    return model

def build_discriminator(
    input_shape, 
    learning_rate,
    mean,
    var,
    units=128,
    **kwargs
):
    
    input_layer = tfkl.Input(input_shape, name='input_layer')
    
    x = tfkl.Normalization(mean=mean, variance=var)(input_layer)
    x = dense_residual_block(x, units=units)
    x = tfkl.Dropout(0.2)(x)
    x = dense_residual_block(x, units=units)
    x = tfkl.Dropout(0.2)(x)
    x = dense_residual_block(x, units=units)
    x = tfkl.Dropout(0.2)(x)
    output_layer = tfkl.Dense(1, activation='sigmoid', kernel_initializer=tfk.initializers.GlorotNormal(), name='output_layer')(x)
    
    discriminator = tfk.Model(inputs=input_layer, outputs=output_layer, name='Discriminator')
    
    discriminator.compile(
        loss=tfk.losses.BinaryCrossentropy(), 
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, weight_decay=5e-4),
        metrics=['accuracy','AUC']
    )
    
    return discriminator