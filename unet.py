import tensorflow as tf


class UNet(object):
    def __init__(self,
                 num_classes):
        self.num_classes = num_classes
        self.losses_list = []

    def losses(self):
        return self.losses_list

    def _conv2d(self,
                input_tensor,
                num_outputs,
                kernel_size=3,
                stride=1,
                padding='SAME',
                dilation_rate=1,
                stddev=0.09,
                weight_decay=0.0004,
                use_bias=False,
                use_bn=True,
                bn_momentum=0.997,
                activation_fn=tf.nn.relu,
                is_training=True,
                scope=None):
        net = input_tensor
        kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)
        with tf.variable_scope(scope, default_name="conv"):
            conv2d = tf.keras.layers.Conv2D(
                filters=num_outputs,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                dilation_rate=dilation_rate,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
            net = conv2d(net)
            self.losses_list.extend(conv2d.losses)
            if not use_bias and use_bn:
                net = tf.layers.batch_normalization(
                    net,
                    momentum=bn_momentum,
                    training=is_training)
            if activation_fn:
                net = activation_fn(net)
            return net

    def _upsample(self,
                  input_tensors,
                  weight_decay=0.0004,
                  stddev=0.09,
                  scope=None):
        with tf.variable_scope(scope, default_name="upsample"):
            kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)
            input_depth = input_tensors[0].get_shape().as_list()[3]
            conv2d_transpose = tf.keras.layers.Conv2DTranspose(
                filters=(input_depth // 2),
                kernel_size=[2, 2],
                strides=[2, 2],
                kernel_initializer=kernel_initializer,
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                use_bias=False)
            net = conv2d_transpose(input_tensors[0])
            self.losses_list.extend(conv2d_transpose.losses)
            net = tf.keras.layers.Concatenate()([net, input_tensors[1]])
        return net

    def forward_pass(self, input_tensor, is_training=True):
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE) as s, \
                tf.name_scope(s.original_name_scope):
            conv1 = self._conv2d(input_tensor, 16, is_training=is_training)
            conv1 = self._conv2d(conv1, 16, is_training=is_training)
            net = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(conv1)
            conv2 = self._conv2d(net, 32, is_training=is_training)
            conv2 = self._conv2d(conv2, 32, is_training=is_training)
            net = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(conv2)
            conv3 = self._conv2d(net, 64, is_training=is_training)
            conv3 = self._conv2d(conv3, 64, is_training=is_training)
            net = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(conv3)
            conv4 = self._conv2d(net, 128, is_training=is_training)
            conv4 = self._conv2d(conv4, 128, is_training=is_training)
            net = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(conv4)

            conv5 = self._conv2d(net, 256, is_training=is_training)
            conv5 = self._conv2d(conv5, 256, is_training=is_training)

        with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE) as s, \
                tf.name_scope(s.original_name_scope):
            net = self._upsample([conv5, conv4])
            net = self._conv2d(net, 128, is_training=is_training)
            net = self._conv2d(net, 128, is_training=is_training)
            net = self._upsample([net, conv3])
            net = self._conv2d(net, 64, is_training=is_training)
            net = self._conv2d(net, 64, is_training=is_training)
            net = self._upsample([net, conv2])
            net = self._conv2d(net, 32, is_training=is_training)
            net = self._conv2d(net, 32, is_training=is_training)
            net = self._upsample([net, conv1])
            net = self._conv2d(net, 16, is_training=is_training)
            net = self._conv2d(net, 16, is_training=is_training)

            outputs = self._conv2d(net,
                                   self.num_classes,
                                   kernel_size=1,
                                   use_bias=True,
                                   use_bn=False,
                                   activation_fn=None,
                                   is_training=is_training)
            return outputs
