import keras
from keras import layers

def get_model(img_size):

    inputs = keras.Input(shape=img_size)

    ### First half of model, downsampling inputs

    # entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x      # set aside residual

    # blocks 1, 2 and 3 are the same apart from the feature depth
    for filters in [64, 128, 256]:

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )

        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual


    ### Second half of model. Upsampling

    for filters in [256, 128, 64, 32]:

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        # Project residual
        residual = layers.Conv2DTranspose(filters, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.Conv2DTranspose(filters, 1, strides=2, padding="same")(x)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual


    # unsure how much this layer actually does
    outputs = layers.Conv2D(1, 3, activation="linear", padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model


def new_model(img_size, model_name, summary=False, lr=1e-4):

    img_size = (img_size[0], img_size[1], 1)

    model = get_model(img_size)

    if summary:
        model.summary()

    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mean_squared_error")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="models/"+f"{model_name}"+"_{epoch:04d}.keras",
            save_best_only=False,
            save_freq='epoch'
        )
    ]

    return model, callbacks


def existing_model(prev_model, model_name, summary=False, lr=1e-4):

    model = keras.saving.load_model(prev_model)

    if summary:
        model.summary()

    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mean_squared_error")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="models/"+f"{model_name}"+"_{epoch:04d}.keras",
            save_best_only=False,
            save_freq='epoch'
        )
    ]

    return model, callbacks


def train(model, tf_train, tf_test, epochs, callbacks, verbose=1, export_history=True):

    history = model.fit(
                tf_train,
                epochs=epochs,
                validation_data=tf_test,
                callbacks=callbacks,
                verbose=verbose,
            )
    
    if export_history:

        import pickle

        with open('training_history.pkl', 'wb') as file:
            pickle.dump(history.history, file)