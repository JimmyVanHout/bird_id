import numpy
import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing import image as tf_image

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
    images_directory_path = os.getcwd() + "/images"
    image_path = sys.argv[1]
    pretrained_model_name = sys.argv[2]
    use_fine_tuning = True if len(sys.argv) > 3 and sys.argv[3] == "--use-fine-tuning" else False
    model_file_name = "cnn_from_pretrained_" + pretrained_model_name + "_ft" if use_fine_tuning else "cnn_from_pretrained_" + pretrained_model_name
    labels = [dir for dir in sorted(os.listdir(images_directory_path))]
    image = tf_image.load_img(image_path, target_size=(IMAGE_LENGTH, IMAGE_LENGTH))
    image_arr = tf_image.img_to_array(image)
    image_batch = numpy.expand_dims(image_arr, axis=0)
    model = tf.keras.models.load_model(model_file_name)
    predictions = model.predict(image_batch)
    predictions = predictions.tolist()[0]
    predictions_and_labels = []
    for i in range(len(predictions)):
        prediction = round(predictions[i], 2)
        if prediction != 0:
            predictions_and_labels.append((labels[i], prediction))
    predictions_and_labels.sort(key=lambda x: x[1], reverse=True)
    for label, prediction in predictions_and_labels:
        print("{label}: {prediction:.2f}%".format(label=label, prediction=prediction * 100))
