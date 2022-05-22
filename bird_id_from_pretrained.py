import os
import pathlib
import sys
import tensorflow as tf

FINE_TUNING_LAYERS = {
    "xception": [0, 2, 3, 5],
    "resnet50": [0, 1, 3],
    "vgg16": [0, 1, 2],
} # layers from last layer
IMAGE_LENGTH = 240
NUM_CLASSES = 200
CONV_BASES = {
    "xception": tf.keras.applications.xception.Xception,
    "resnet50": tf.keras.applications.resnet50.ResNet50,
    "vgg16": tf.keras.applications.vgg16.VGG16,
}
INPUT_PREPROCESSORS = {
    "xception": tf.keras.applications.xception.preprocess_input,
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
    "vgg16": tf.keras.applications.vgg16.preprocess_input,
}

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    train_directory_name = os.getcwd() + "/train"
    validation_directory_name = os.getcwd() + "/validation"
    test_directory_name = os.getcwd() + "/test"

    train = tf.keras.utils.image_dataset_from_directory(pathlib.Path(train_directory_name), image_size=(IMAGE_LENGTH, IMAGE_LENGTH), batch_size=32).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation = tf.keras.utils.image_dataset_from_directory(pathlib.Path(validation_directory_name), image_size=(IMAGE_LENGTH, IMAGE_LENGTH), batch_size=32).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test = tf.keras.utils.image_dataset_from_directory(pathlib.Path(test_directory_name), image_size=(IMAGE_LENGTH, IMAGE_LENGTH), batch_size=32)

    pretrained_model_name = sys.argv[1]
    fine_tuning = True if len(sys.argv) > 2 and sys.argv[2] == "--fine-tuning" else False
    without_ft_path = os.getcwd() + "/cnn_from_pretrained_" + pretrained_model_name
    with_ft_path = os.getcwd() + "/cnn_from_pretrained_" + pretrained_model_name + "_ft"

    conv_base = CONV_BASES[pretrained_model_name](weights="imagenet", include_top=False)
    conv_base.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_LENGTH, IMAGE_LENGTH, 3))
    x = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.2)
    ])(inputs)
    x = INPUT_PREPROCESSORS[pretrained_model_name](x)
    x = conv_base(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # first run
    model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=without_ft_path, save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6)
    ]
    history = model.fit(train, epochs=100, validation_data=validation, callbacks=callbacks)
    model = tf.keras.models.load_model(without_ft_path)
    test_loss, test_accuracy = model.evaluate(test)
    print("Test accuracy: " + str(round(test_accuracy * 100, 2)) + "%")

    # second run using fine-tuning
    if fine_tuning:
        for i in FINE_TUNING_LAYERS[pretrained_model_name]:
            conv_base.layers[len(conv_base.layers) - 1 - i].trainable = True
        model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=with_ft_path, save_best_only=True, monitor="val_accuracy"),
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6)
        ]
        history = model.fit(train, epochs=100, validation_data=validation, callbacks=callbacks)
        model = tf.keras.models.load_model(with_ft_path)
        test_loss, test_accuracy = model.evaluate(test)
        print("Test accuracy: " + str(round(test_accuracy * 100, 2)) + "%")
