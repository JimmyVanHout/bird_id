import os
import pathlib
import tensorflow as tf

IMAGE_LENGTH = 240
NUM_CLASSES = 200

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    train_directory_name = os.getcwd() + "/train"
    validation_directory_name = os.getcwd() + "/validation"
    test_directory_name = os.getcwd() + "/test"

    train = tf.keras.utils.image_dataset_from_directory(pathlib.Path(train_directory_name), image_size=(IMAGE_LENGTH, IMAGE_LENGTH), batch_size=32).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation = tf.keras.utils.image_dataset_from_directory(pathlib.Path(validation_directory_name), image_size=(IMAGE_LENGTH, IMAGE_LENGTH), batch_size=32).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test = tf.keras.utils.image_dataset_from_directory(pathlib.Path(test_directory_name), image_size=(IMAGE_LENGTH, IMAGE_LENGTH), batch_size=32)

    inputs = tf.keras.Input(shape=(IMAGE_LENGTH, IMAGE_LENGTH, 3))
    x = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])(inputs)
    x = tf.keras.layers.Rescaling(1 / 255)(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath="cnn_from_scratch", save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6)
    ]
    history = model.fit(train, epochs=100, validation_data=validation, callbacks=callbacks)
    model = tf.keras.models.load_model(os.getcwd() + "/cnn_from_scratch")
    test_loss, test_accuracy = model.evaluate(test)
    print("Test accuracy: " + str(round(test_accuracy * 100, 2)) + "%")
