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

"""Utility functions related to preprocessing inputs."""
import collections

import tensorflow as tf


_Op = collections.namedtuple('Op', ['op', 'params'])


def op(opfunc, **params):
    return _Op(opfunc, params=params)


def flip_dim(tensor_list, prob=0.5, dim=1):
    """Randomly flips a dimension of the given tensor.

    The decision to randomly flip the `Tensors` is made together.
    In other words, all or none of the images pass in are flipped.

    Note that tf.random_flip_left_right and tf.random_flip_up_down isn't used
     so that we can control for the probability as well as ensure the
     same decision is applied across the images.

    Args:
      tensor_list: A list of `Tensors` with the same number of dimensions.
      prob: The probability of a left-right flip.
      dim: The dimension to flip, 0, 1, ..

    Returns:
      outputs: A list of the possibly flipped `Tensors` as well as an indicator
      `Tensor` at the end whose value is `True` if the inputs were flipped and
      `False` otherwise.

    Raises:
      ValueError: If dim is negative or greater than dimension of a `Tensor`.
    """
    random_value = tf.random_uniform([])

    def flip():
        flipped = []
        for tensor in tensor_list:
            if dim < 0 or dim >= len(tensor.get_shape().as_list()):
                raise ValueError('dim must represent a valid dimension.')
            flipped.append(tf.reverse_v2(tensor, [dim]))
        return flipped

    is_flipped = tf.less_equal(random_value, prob)
    outputs = tf.cond(is_flipped, flip, lambda: tensor_list)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    outputs.append(is_flipped)

    return outputs


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, pad_value):
    """Pads the given image with the given pad_value.

    Works like tf.image.pad_to_bounding_box, except it can pad the image
    with any given arbitrary pad value and also handle images whose sizes
    are not known during graph construction.

    Args:
      image: 3-D tensor with shape [height, width, channels]
      offset_height: Number of rows of zeros to add on top.
      offset_width: Number of columns of zeros to add on the left.
      target_height: Height of output image.
      target_width: Width of output image.
      pad_value: Value to pad the image tensor with.

    Returns:
      3-D tensor of shape [target_height, target_width, channels].

    Raises:
      ValueError: If the shape of image is incompatible with the offset_* or
      target_* arguments.
    """
    image_rank = tf.rank(image)
    image_rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong image tensor rank [Expected] [Actual]',
         3, image_rank])
    with tf.control_dependencies([image_rank_assert]):
        image -= pad_value
    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]
    target_width_assert = tf.Assert(
        tf.greater_equal(
            target_width, width),
        ['target_width must be >= width'])
    target_height_assert = tf.Assert(
        tf.greater_equal(target_height, height),
        ['target_height must be >= height'])
    with tf.control_dependencies([target_width_assert]):
        after_padding_width = target_width - offset_width - width
    with tf.control_dependencies([target_height_assert]):
        after_padding_height = target_height - offset_height - height
    offset_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(after_padding_width, 0),
            tf.greater_equal(after_padding_height, 0)),
        ['target size not possible with the given target offsets'])

    height_params = tf.stack([offset_height, after_padding_height])
    width_params = tf.stack([offset_width, after_padding_width])
    channel_params = tf.stack([0, 0])
    with tf.control_dependencies([offset_assert]):
        paddings = tf.stack([height_params, width_params, channel_params])
    padded = tf.pad(image, paddings)
    return padded + pad_value


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.

    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.

    Args:
      image: an image of shape [height, width, channels].
      offset_height: a scalar tensor indicating the height offset.
      offset_width: a scalar tensor indicating the width offset.
      crop_height: the height of the cropped image.
      crop_width: the width of the cropped image.

    Returns:
      The cropped (and resized) image.

    Raises:
      ValueError: if `image` doesn't have rank of 3.
      InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = tf.shape(image)

    if len(image.get_shape().as_list()) != 3:
        raise ValueError('input must have rank of 3')
    original_channels = image.get_shape().as_list()[2]

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    image = tf.reshape(image, cropped_shape)
    image.set_shape([crop_height, crop_width, original_channels])
    return image


def random_crop(image_list, crop_height, crop_width):
    """Crops the given list of images.

    The function applies the same crop to each image in the list. This can be
    effectively applied when there are multiple image inputs of the same
    dimension such as:

      image, depths, normals = random_crop([image, depths, normals], 120, 150)

    Args:
      image_list: a list of image tensors of the same dimension but possibly
        varying channel.
      crop_height: the new height.
      crop_width: the new width.

    Returns:
      the image_list with cropped images.

    Raises:
      ValueError: if there are multiple image inputs provided with different
       size or the images are smaller than the crop dimensions.
    """
    if not image_list:
        raise ValueError('Empty image_list.')

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    #
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width,
                  crop_height, crop_width) for image in image_list]


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
    """Gets a random scale value.

    Args:
      min_scale_factor: Minimum scale value.
      max_scale_factor: Maximum scale value.
      step_size: The step size from minimum to maximum value.

    Returns:
      A random scale value selected between minimum and maximum value.

    Raises:
      ValueError: min_scale_factor has unexpected value.
    """
    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')

    if min_scale_factor == max_scale_factor:
        return tf.to_float(min_scale_factor)

    # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return tf.random_uniform([1],
                                 minval=min_scale_factor,
                                 maxval=max_scale_factor)

    # When step_size != 0, we randomly select one discrete value from
    # [min, max].
    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = tf.lin_space(min_scale_factor, max_scale_factor, num_steps)
    shuffled_scale_factors = tf.random_shuffle(scale_factors)
    return shuffled_scale_factors[0]


def randomly_scale_image_and_label(image, label=None, scale=1.0):
    """Randomly scales image and label.

    Args:
      image: Image with shape [height, width, 3].
      label: Label with shape [height, width, 1].
      scale: The value to scale image and label.

    Returns:
      Scaled image and label.
    """
    # No random scaling if scale == 1.
    if scale == 1.0:
        return image, label
    image_shape = tf.shape(image)
    new_dim = tf.to_int32(tf.to_float([image_shape[0], image_shape[1]]) * scale)

    # Need squeeze and expand_dims because image interpolation takes
    # 4D tensors as input.
    image = tf.squeeze(tf.image.resize_bilinear(
        tf.expand_dims(image, 0),
        new_dim,
        align_corners=True), [0])
    if label is not None:
        label = tf.squeeze(tf.image.resize_nearest_neighbor(
            tf.expand_dims(label, 0),
            new_dim,
            align_corners=True), [0])

    return image, label


def resolve_shape(tensor, rank=None, scope=None):
    """Fully resolves the shape of a Tensor.

    Use as much as possible the shape components already known during graph
    creation and resolve the remaining ones during runtime.

    Args:
      tensor: Input tensor whose shape we query.
      rank: The rank of the tensor, provided that we know it.
      scope: Optional name scope.

    Returns:
      shape: The full shape of the tensor.
    """
    with tf.name_scope(scope, 'resolve_shape', [tensor]):
        if rank is not None:
            shape = tensor.get_shape().with_rank(rank).as_list()
        else:
            shape = tensor.get_shape().as_list()

        if None in shape:
            shape_dynamic = tf.shape(tensor)
            for i in range(len(shape)):
                if shape[i] is None:
                    shape[i] = shape_dynamic[i]

        return shape


def resize_to_target(image,
                     label=None,
                     target_height=None,
                     target_width=None,
                     align_corners=True,
                     scope=None,
                     method=tf.image.ResizeMethod.BILINEAR):
    """Resizes image or label so their sides are within the provided range.

    The output size can be described by two cases:
    1. If the image can be rescaled so its minimum size is equal to min_size
       without the other side exceeding max_size, then do so.
    2. Otherwise, resize so the largest side is equal to max_size.

    An integer in `range(factor)` is added to the computed sides so that the
    final dimensions are multiples of `factor` plus one.

    Args:
      image: A 3D tensor of shape [height, width, channels].
      label: (optional) A 3D tensor of shape [height, width, channels] (default)
        or [channels, height, width] when label_layout_is_chw = True.
      target_height: (scalar) desired max height.
      target_width: (scalar) desired max width.
      align_corners: If True, exactly align all 4 corners of input and output.
      scope: Optional name scope.
      method: Image resize method. Defaults to tf.image.ResizeMethod.BILINEAR.

    Returns:
      A 3-D tensor of shape [new_height, new_width, channels]

    Raises:
      ValueError: If the image is not a 3D tensor.
    """
    with tf.name_scope(scope, 'resize_to_target_size', [image]):
        new_tensor_list = []
        target_height = tf.to_float(target_height)
        target_width = tf.to_float(target_width)

        [orig_height, orig_width, _] = resolve_shape(image, rank=3)
        orig_height = tf.to_float(orig_height)
        orig_width = tf.to_float(orig_width)

        # Calculate the larger of the possible sizes
        scale_factor = tf.minimum(target_height / orig_height,
                                  target_width / orig_width)
        scale_factor = tf.minimum(scale_factor, 1.0)
        dst_height = tf.to_int32(tf.ceil(orig_height * scale_factor))
        dst_width = tf.to_int32(tf.ceil(orig_width * scale_factor))
        new_size = tf.stack([dst_height, dst_width])

        new_tensor_list.append(tf.image.resize_images(
            image, new_size, method=method, align_corners=align_corners))
        if label is not None:
            # Input label has shape [height, width, channel].
            resized_label = tf.image.resize_images(
                label, new_size,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                align_corners=align_corners)
            new_tensor_list.append(resized_label)
        else:
            new_tensor_list.append(None)
        return new_tensor_list


def scale_dimension(dim, scale):
    """Scales the input dimension.

    Args:
      dim: Input dimension (a scalar or a scalar Tensor).
      scale: The amount of scaling applied to the input.

    Returns:
      Scaled dimension.
    """
    if isinstance(dim, tf.Tensor):
        return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0,
                       dtype=tf.int32)
    else:
        return int((float(dim) - 1.0) * scale + 1.0)


def get_model_learning_rate(
        global_step,
        learning_policy, base_learning_rate, learning_rate_decay_step,
        learning_rate_decay_factor, training_number_of_steps, learning_power,
        slow_start_step, slow_start_learning_rate):
    """Gets model's learning rate.

    Computes the model's learning rate for different learning policy.
    Right now, only "step" and "poly" are supported.
    (1) The learning policy for "step" is computed as follows:
      current_learning_rate = base_learning_rate *
        learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
    See tf.train.exponential_decay for details.
    (2) The learning policy for "poly" is computed as follows:
      current_learning_rate = base_learning_rate *
        (1 - global_step / training_number_of_steps) ^ learning_power

    Args:
      global_step: global_step.
      learning_policy: Learning rate policy for training.
      base_learning_rate: The base learning rate for model training.
      learning_rate_decay_step: Decay the base learning rate at a fixed step.
      learning_rate_decay_factor: The rate to decay the base learning rate.
      training_number_of_steps: Number of steps for training.
      learning_power: Power used for 'poly' learning policy.
      slow_start_step: Training model with small learning rate for the first
        few steps.
      slow_start_learning_rate: The learning rate employed during slow start.

    Returns:
      Learning rate for the specified learning policy.

    Raises:
      ValueError: If learning policy is not recognized.
    """
    if learning_policy == 'step':
        learning_rate = tf.train.exponential_decay(
            base_learning_rate,
            global_step,
            learning_rate_decay_step,
            learning_rate_decay_factor,
            staircase=True)
    elif learning_policy == 'poly':
        learning_rate = tf.train.polynomial_decay(
            base_learning_rate,
            global_step,
            training_number_of_steps,
            end_learning_rate=0,
            power=learning_power)
    else:
        raise ValueError('Unknown learning policy.')

    # Employ small learning rate at the first few steps for warm start.
    return tf.where(global_step < slow_start_step, slow_start_learning_rate,
                    learning_rate)

