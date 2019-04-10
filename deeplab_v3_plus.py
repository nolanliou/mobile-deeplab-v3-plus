import tensorflow as tf

import layers
from mobilenet_v2 import MobilenetV2
from utils import scale_dimension

# Default end point for MobileNetv2.
_MOBILENET_V2_FINAL_ENDPOINT = 'layer_18'

LOGITS_SCOPE_NAME = 'logits'
ASPP_SCOPE = "aspp"
CONCAT_PROJECTION_SCOPE = 'concat_projection'


class DeeplabV3Plus(object):
    def __init__(self,
                 num_classes,
                 model_input_size=None,
                 atrous_rates=None,
                 output_stride=None,
                 depth_multiplier=1.0,
                 weight_decay=0.0001,
                 add_image_level_feature=True,
                 image_pooling_crop_size=None,
                 aspp_with_batch_norm=True,
                 aspp_with_separable_conv=True,
                 decoder_output_stride=None):
        self.num_classes = num_classes
        self.model_input_size = model_input_size
        self.atrous_rates = atrous_rates
        self.output_stride = output_stride
        self.depth_multiplier = depth_multiplier
        self.weight_decay = weight_decay
        self.add_image_level_feature = add_image_level_feature
        self.image_pooling_crop_size = image_pooling_crop_size
        self.aspp_with_batch_norm = aspp_with_batch_norm
        self.aspp_with_separable_conv = aspp_with_separable_conv
        self.decoder_output_stride = decoder_output_stride

    @staticmethod
    def _conv2d(input_tensor,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                dilation_rate=1,
                stddev=-1.0,
                weight_decay=0.0001,
                use_bias=False,
                use_bn=True,
                bn_momentum=0.9997,
                bn_epsilon=1e-5,
                activation_fn=tf.nn.relu,
                is_training=True,
                scope=None):
        net = input_tensor
        with tf.variable_scope(scope, default_name="Conv"):
            if stddev > 0:
                kernel_initializer =\
                    tf.truncated_normal_initializer(stddev=stddev)
            else:
                kernel_initializer = \
                    tf.contrib.layers.xavier_initializer()
            conv2d = tf.keras.layers.Conv2D(
                filters=num_outputs,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                dilation_rate=dilation_rate,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                name='conv2d')
            net = conv2d(net)
            tf.summary.histogram('Weights', conv2d.weights[0])
            if not use_bias and use_bn:
                net = tf.layers.batch_normalization(
                    net,
                    momentum=bn_momentum,
                    epsilon=bn_epsilon,
                    training=is_training,
                    name='BatchNorm')
            if activation_fn:
                net = activation_fn(net)
                tf.summary.histogram('Activation', net)
            return net

    @staticmethod
    def _separable_conv(input_tensor,
                        num_outputs,
                        kernel_size,
                        depth_multiplier=1,
                        stride=1,
                        padding='SAME',
                        dilation_rate=1,
                        dw_stddev=0.33,
                        weight_decay=0.0001,
                        pw_stddev=0.06,
                        bn_momentum=0.9997,
                        bn_epsilon=1e-5,
                        activation_fn=tf.nn.relu,
                        is_training=True,
                        scope=None):
        with tf.variable_scope(scope, default_name="separable_conv"):
            # depthwise convolution
            net = layers.depthwise_conv(input_tensor,
                                        kernel_size=kernel_size,
                                        depth_multiplier=depth_multiplier,
                                        stride=stride,
                                        padding=padding,
                                        dilation_rate=dilation_rate,
                                        stddev=dw_stddev,
                                        bn_momentum=bn_momentum,
                                        bn_epsilon=bn_epsilon,
                                        activation_fn=activation_fn,
                                        is_training=is_training,
                                        scope=scope + "_depthwise")
            # pointwise convolution
            net = DeeplabV3Plus._conv2d(net,
                                        num_outputs=num_outputs,
                                        kernel_size=[1, 1],
                                        stride=stride,
                                        padding=padding,
                                        stddev=pw_stddev,
                                        weight_decay=weight_decay,
                                        bn_momentum=bn_momentum,
                                        bn_epsilon=bn_epsilon,
                                        activation_fn=activation_fn,
                                        is_training=is_training,
                                        scope=scope + "_pointwise")
            return net

    def _atrous_spatial_pyramid_pooling(self,
                                        features,
                                        weight_decay=0.0001,
                                        is_training=True):
        depth = 256
        branch_logits = []

        if self.add_image_level_feature:
            if self.model_input_size is not None:
                image_pooling_crop_size = self.image_pooling_crop_size
                # If image_pooling_crop_size is not specified, use crop_size.
                if image_pooling_crop_size is None:
                    image_pooling_crop_size = self.model_input_size
                pool_height = scale_dimension(
                    image_pooling_crop_size[0],
                    1. / self.output_stride)
                pool_width = scale_dimension(
                    image_pooling_crop_size[1],
                    1. / self.output_stride)
                image_feature = tf.keras.layers.AvgPool2D(
                    pool_size=[pool_height, pool_width],
                    strides=[1, 1],
                    padding='VALID')(features)
                resize_height = scale_dimension(
                    self.model_input_size[0],
                    1. / self.output_stride)
                resize_width = scale_dimension(
                    self.model_input_size[1],
                    1. / self.output_stride)
            else:
                # If crop_size is None, we simply do global pooling.
                pool_height = tf.shape(features)[1]
                pool_width = tf.shape(features)[2]
                image_feature = tf.reduce_mean(
                    features, axis=[1, 2], keepdims=True)
                resize_height = pool_height
                resize_width = pool_width
            image_feature = self._conv2d(
                input_tensor=image_feature,
                num_outputs=depth,
                kernel_size=1,
                weight_decay=weight_decay,
                is_training=is_training,
                scope='image_pooling')
            image_feature = layers.resize_bilinear(
                image_feature,
                [resize_height, resize_width],
                image_feature.dtype)
            branch_logits.append(image_feature)

        # Employ a 1x1 convolution.
        branch_logits.append(self._conv2d(
            input_tensor=features,
            num_outputs=depth,
            kernel_size=1,
            weight_decay=weight_decay,
            is_training=is_training,
            scope=ASPP_SCOPE + str(0)))

        if self.atrous_rates:
            # Employ 3x3 convolutions with different atrous rates.
            for i, rate in enumerate(self.atrous_rates, 1):
                scope = ASPP_SCOPE + str(i)
                if self.aspp_with_separable_conv:
                    aspp_features = self._separable_conv(
                        features,
                        num_outputs=depth,
                        kernel_size=[3, 3],
                        padding='SAME',
                        dilation_rate=rate,
                        weight_decay=weight_decay,
                        is_training=is_training,
                        scope=scope)
                else:
                    aspp_features = self._conv2d(
                        features,
                        num_outputs=depth,
                        kernel_size=3,
                        dilation_rate=rate,
                        weight_decay=weight_decay,
                        is_training=is_training,
                        scope=scope)
                branch_logits.append(aspp_features)
        # Merge branch logits.
        concat_logits = tf.concat(branch_logits, 3)
        concat_logits = self._conv2d(
            concat_logits,
            num_outputs=depth,
            kernel_size=1,
            weight_decay=weight_decay,
            is_training=is_training,
            scope=CONCAT_PROJECTION_SCOPE)
        if is_training:
            with tf.variable_scope(CONCAT_PROJECTION_SCOPE + '_dropout'):
                concat_logits = tf.keras.layers.Dropout(rate=0.1)(concat_logits)

        return concat_logits

    def encode(self,
               input_tensor,
               pretrained_backbone_model_dir,
               is_training=True):
        # extract features
        mobilenet_model = MobilenetV2(
            self.output_stride,
            self.depth_multiplier,
            min_depth=8 if self.depth_multiplier == 1.0 else 1,
            divisible_by=8 if self.depth_multiplier == 1.0 else 1)
        features, endpoints = mobilenet_model.forward_base(
            input_tensor,
            _MOBILENET_V2_FINAL_ENDPOINT,
            is_training=is_training)
        
        if pretrained_backbone_model_dir and is_training:
            base_architecture = 'MobilenetV2'
            exclude = [base_architecture + '/Logits', 'global_step']
            variables_to_restore = tf.contrib.slim.get_variables_to_restore(
                exclude=exclude)
            tf.logging.info('init from %s model' % base_architecture)
            tf.train.init_from_checkpoint(pretrained_backbone_model_dir,
                                          {v.name.split(':')[0]: v for v in
                                           variables_to_restore})

        features = self._atrous_spatial_pyramid_pooling(
            features, weight_decay=self.weight_decay, is_training=is_training)

        return features, endpoints

    def decode(self,
               input_tensor,
               endpoints,
               is_training=True):
        pass

    def forward(self,
                input_tensor,
                pretrained_backbone_model_dir=None,
                is_training=True):
        input_height = (
            self.model_input_size[0]
            if self.model_input_size else tf.shape(input_tensor)[1])
        input_width = (
            self.model_input_size[1]
            if self.model_input_size else tf.shape(input_tensor)[2])

        features, endpoints = self.encode(input_tensor,
                                          pretrained_backbone_model_dir,
                                          is_training)

        # if self.decoder_output_stride is not None:
        #     features = self.decode(features, endpoints)
        with tf.variable_scope(LOGITS_SCOPE_NAME):
            logits = self._conv2d(features,
                                  num_outputs=self.num_classes,
                                  kernel_size=1,
                                  stddev=0.01,
                                  weight_decay=self.weight_decay,
                                  use_bias=True,
                                  use_bn=False,
                                  activation_fn=None,
                                  is_training=is_training,
                                  scope='semantic')

        # Resize the logits to have the same dimension before merging.
        output_logit = tf.image.resize_bilinear(
            logits, [input_height, input_width],
            align_corners=True)

        return output_logit

