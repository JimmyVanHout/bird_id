# Bird ID

A Python-based library to identify bird species from the [Caltech-UCSD Birds 200](https://vision.cornell.edu/se3/caltech-ucsd-birds-200/) dataset using either a simple, custom convolutional neural network (CNN) or one of several CNN models ([ResNet50](https://arxiv.org/abs/1512.03385), [Xception](https://arxiv.org/abs/1610.02357), or [VGG16](https://arxiv.org/abs/1409.1556)) that has been pretrained on [ImageNet](https://www.image-net.org/) and can optionally be fine-tuned by the training program. Utilizes the [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) libraries.

## Installation and Setup

> Note that a Linux or Windows operating system and a CUDA-and-cuDNN-capable-GPU are necessary to train a CNN model.

1. Install the library, available from [GitHub](https://github.com/JimmyVanHout/bird_id):

    ```
    git clone https://github.com/JimmyVanHout/bird_id.git
    ```

1. Install TensorFlow and Keras (Keras is installed with TensorFlow):

    ```
    pip3 install tensorflow
    ```

1. Install [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

1. Install [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

1. Download the images from [Google Drive](https://drive.google.com/uc?export=download&id=1GDr1OkoXdhaXWGA8S3MAq3a522Tak-nx) (they were not accessible via the [TensorFlow Datasets API](https://www.tensorflow.org/datasets) during development). Rename the downloaded images directory to `images` and save it to the program directory.

1. Clean up the image names in the images directory:

    ```
    python3 rename_images.py
    ```

1. Run the following to randomly assign the images in the images dataset to either the training, validation, or test datasets, consisting of 60%, 20%, and 20% of the images in the original dataset, respectively (this will create the directories `train`, `validation`, and `test`):

    ```
    python3 create_sets.py
    ```

## Usage

### Training a Custom-Built CNN from Scratch

To train a simple, custom-built CNN from scratch, run:

```
python3 bird_id_from_scratch.py
```

The model will be saved to a directory named `cnn_from_scratch`. After training is completed and the model is evaluated on the test dataset, the test accuracy will be displayed. Expect it to be low (< 30%), since it is a small and simple model (e.g. it does not make use of residual connections or batch normalization) and the Caltech-UCSD Birds 200 dataset is a challenging dataset with an average of only 59 images per class in the entire dataset (an average of only 35 images per class in the training set that is 60% the size of the original dataset) with many similar-looking bird species. Note, however, that even a 20% accuracy is still 40x that which would be achieved in randomly classifying images.

### Training a Pretrained, Optionally Fine-Tuned Model

To achieve far greater accuracy (~60%, depending on the chosen model), use one of the following CNN models, each of which has been pretrained on ImageNet and can also optionally be fine-tuned by the program to achieve slightly greater accuracy ("FT" indicates fine-tuning):

Model | Accuracy without FT | Accuracy with FT
--- | --- | ---
ResNet50 | 58.25% | 60.66%
Xception | 54.26% | 60.83%
VGG16 | 56.25% | 56.51%

To train a model, run:

```
python3 bird_id_from_pretrained.py <model> [--fine-tuning]
```

where:

* `<model>` is one of the following:

    * `resnet50`
    * `xception`
    * `vgg16`

* `--fine-tuning` specifies that fine-tuning should be used

For example:

```
python3 bird_id_from_pretrained.py resnet50 --fine-tuning
```

A model without fine-tuning will be saved as a directory named `cnn_from_pretrained_<model>`, where `<model>` is the name of the model. If fine-tuning is specified, then a model using it will also be saved as a directory named `cnn_from_pretrained_<model>_ft`.

The test accuracy will be displayed after the training is completed and the model is evaluated. Note that even a 60% accuracy is still 120x the accuracy of random classification.

### Predicting an Image's Class Using a Fully-Trained Model

> Note that the simple, from-scratch CNN model cannot be used for the following, because the other models offer far better predictions.

To attempt to classify an image using a model that has been fully trained following the instructions above, run:

```
python3 predict_image.py <image_path> <model> [--use-fine-tuning]
```

where:

* `<image_path>` is the full path to the image to be classified

* `<model>` is one of the following:

    * `resnet50`
    * `xception`
    * `vgg16`

* `--use-fine-tuning` specifies that the version of the fully trained model that has used fine-tuning should be used for classification.

For example:

```
python3 predict_image.py /home/username/image xception --use-fine-tuning
```

After running the program, a probability distribution over the predicted labels will be displayed.
