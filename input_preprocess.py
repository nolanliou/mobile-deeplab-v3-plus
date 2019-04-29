# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Prepares the data used for DeepLab training/evaluation."""
import numpy as np
import tensorflow as tf

from PIL import Image
import utils


# colour map
label_colours = [(0, 0, 0),  # 0=background
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
                 (128, 0, 128),
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
                 (64, 128, 0),
                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
                 (192, 128, 128),
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                 (0, 64, 128)]


# The probability of flipping the images and labels
# left-right during training
_PROB_OF_FLIP = 0.5


MEAN_UINT_PIXEL = [127.5, 127.5, 127.5]


def _preprocess_zero_mean_unit_range(inputs):
    """Map image values from [0, 255] to [-1, 1]."""
    return (2.0 / 255.0) * tf.to_float(inputs) - 1.0


def decode_org_image(inputs):
    return (tf.to_float(inputs) + 1.0) * (255.0 / 2.0)


def preprocess_image_and_label(image,
                               label,
                               model_input_height,
                               model_input_width,
                               min_scale_factor=1.,
                               max_scale_factor=1.,
                               scale_factor_step_size=0,
                               ignore_label=255,
                               is_training=True):
    """Preprocesses the image and label.

    Args:
      image: Input image.
      label: Ground truth annotation label.
      model_input_height: The height of the input feed to model extractor.
      model_input_width: The width of the input feed to model extractor.
      min_scale_factor: Minimum scale factor value.
      max_scale_factor: Maximum scale factor value.
      scale_factor_step_size: The step size from min scale factor to max scale
        factor. The input is randomly scaled based on the value of
        (min_scale_factor, max_scale_factor, scale_factor_step_size).
      ignore_label: The label value which will be ignored for training and
        evaluation.
      is_training: If the preprocessing is used for training or not.

    Returns:
      original_image: Original image (could be resized).
      processed_image: Preprocessed image.
      label: Preprocessed ground truth segmentation label.

    Raises:
      ValueError: Ground truth label not provided during training.
    """
    if is_training and label is None:
        raise ValueError('During training, label must be provided.')

    processed_image = tf.cast(image, tf.float32)

    if label is not None:
        label = tf.cast(label, tf.int32)

    # Data augmentation by randomly scaling the inputs.
    if is_training:
        scale = utils.get_random_scale(
          min_scale_factor, max_scale_factor, scale_factor_step_size)
        processed_image, label = utils.randomly_scale_image_and_label(
          processed_image, label, scale)
        processed_image.set_shape([None, None, 3])

    if not is_training:
        processed_image, label = utils.resize_to_target(
            processed_image,
            label,
            model_input_height,
            model_input_width)

    # Pad image and label to have
    # dimensions >= [model_input_height, model_input_width]
    image_shape = tf.shape(processed_image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    target_height = image_height +\
        tf.maximum(model_input_height - image_height, 0)
    target_width = image_width +\
        tf.maximum(model_input_width - image_width, 0)

    # Pad image with mean pixel value.
    mean_pixel = tf.reshape(MEAN_UINT_PIXEL, [1, 1, 3])
    processed_image = utils.pad_to_bounding_box(
      processed_image, 0, 0, target_height, target_width, mean_pixel)

    if label is not None:
        label = utils.pad_to_bounding_box(
            label, 0, 0, target_height, target_width, ignore_label)

    # Randomly crop the image and label.
    if is_training and label is not None:
        processed_image, label = utils.random_crop(
            [processed_image, label],
            model_input_height,
            model_input_width)
        
    processed_image.set_shape([model_input_height, model_input_width, 3])

    if label is not None:
        label.set_shape([model_input_height, model_input_width, 1])

    if is_training:
        # Randomly left-right flip the image and label.
        processed_image, label, _ = utils.flip_dim(
            [processed_image, label], _PROB_OF_FLIP, dim=1)

    processed_image = _preprocess_zero_mean_unit_range(processed_image)

    return processed_image, label


def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w, c = mask.shape
    assert(n >= num_images), \
        'Batch size %d should be greater or equal than number' \
        ' of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs
