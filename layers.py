""" common layers for convenient
"""

import tensorflow as tf


def depthwise_conv(input_tensor,
                   kernel_size=3,
                   depth_multiplier=1,
                   stride=1,
                   padding='SAME',
                   dilation_rate=1,
                   stddev=0.09,
                   use_bias=False,
                   use_bn=True,
                   bn_momentum=0.997,
                   bn_epsilon=1e-3,
                   activation_fn=tf.nn.relu6,
                   is_training=True,
                   scope=None):
    in_channel = input_tensor.get_shape().as_list()[-1]
    net = input_tensor
    kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)
    with tf.variable_scope(scope, default_name="conv"):
        weight = tf.get_variable(
            'weight',
            [kernel_size, kernel_size, in_channel, depth_multiplier],
            regularizer=None,
            initializer=kernel_initializer)
        net = tf.nn.depthwise_conv2d(
            net,
            weight,
            [1, stride, stride, 1],
            padding,
            rate=[dilation_rate, dilation_rate])
        if not use_bias and use_bn:
            # there is no trainable of tf.keras.layers.BatchNormalization
            # in TF-1.8
            net = tf.layers.BatchNormalization(
                momentum=bn_momentum,
                epsilon=bn_epsilon,
                trainable=is_training)(net)
        if activation_fn:
            net = activation_fn(net)
        return net


def global_pool(input_tensor):
    """Applies avg pool to produce 1x1 output.

    NOTE: This function is funcitonally equivalenet to reduce_mean,
    but it has baked in average pool
    which has better support across hardware.

    Args:
      input_tensor: input tensor
    Returns:
      a tensor batch_size x 1 x 1 x depth.
    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size = tf.convert_to_tensor(
            [tf.shape(input_tensor)[1],
             tf.shape(input_tensor)[2]])
    else:
        kernel_size = [shape[1], shape[2]]
    output = tf.keras.layers.AvgPool2D(pool_size=kernel_size,
                                       strides=[1, 1],
                                       padding='VALID')(input_tensor)
    # Recover output shape, for unknown shape.
    output.set_shape([None, 1, 1, None])
    return output


def resize_bilinear(images, size, output_dtype=tf.float32):
    """Returns resized images as output_type.

    Args:
      images: A tensor of size [batch, height_in, width_in, channels].
      size: A 1-D int32 Tensor of 2 elements: new_height, new_width.
            The new size for the images.
      output_dtype: The destination type.
    Returns:
      A tensor of size [batch, height_out, width_out, channels] as a dtype of
        output_dtype.
    """
    images = tf.image.resize_bilinear(images, size, align_corners=True)
    return tf.cast(images, dtype=output_dtype)
