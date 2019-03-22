import tensorflow as tf

import layers
from input_preprocess import decode_labels, decode_org_image
from mobilenet_v2 import MobilenetV2
from utils import scale_dimension, get_model_learning_rate

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
        with tf.variable_scope(scope, default_name="conv"):
            if stddev > 0:
                kernel_initializer =\
                    tf.truncated_normal_initializer(stddev=stddev)
            else:
                kernel_initializer = \
                    tf.contrib.layers.xavier_initializer()
            net = tf.keras.layers.Conv2D(
                filters=num_outputs,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                dilation_rate=dilation_rate,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(net)
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
                        scope=scope)
                else:
                    aspp_features = self._conv2d(
                        features,
                        num_outputs=depth,
                        kernel_size=3,
                        dilation_rate=rate,
                        weight_decay=weight_decay,
                        scope=scope)
                branch_logits.append(aspp_features)
        # Merge branch logits.
        concat_logits = tf.concat(branch_logits, 3)
        concat_logits = self._conv2d(
            concat_logits,
            num_outputs=depth,
            kernel_size=1,
            weight_decay=weight_decay,
            scope=CONCAT_PROJECTION_SCOPE)
        if is_training:
            with tf.variable_scope(CONCAT_PROJECTION_SCOPE + '_dropout'):
                concat_logits = tf.keras.layers.Dropout(rate=0.1)(concat_logits)

        return concat_logits

    def encode(self,
               input_tensor,
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

        features = self._atrous_spatial_pyramid_pooling(
            features, weight_decay=self.weight_decay, is_training=is_training)

        return features, endpoints

    def decode(self,
               input_tensor,
               endpoints,
               is_training=True):
        pass

    def forward_pass(self,
                     input_tensor,
                     is_training=True):
        input_height = (
            self.model_input_size[0]
            if self.model_input_size else tf.shape(input_tensor)[1])
        input_width = (
            self.model_input_size[1]
            if self.model_input_size else tf.shape(input_tensor)[2])

        features, endpoints = self.encode(input_tensor, is_training)

        # if self.decoder_output_stride is not None:
        #     features = self.decode(features, endpoints)

        logits = self._conv2d(features,
                              num_outputs=self.num_classes,
                              kernel_size=1,
                              stddev=0.01,
                              weight_decay=self.weight_decay,
                              use_bias=True,
                              use_bn=False,
                              activation_fn=None,
                              is_training=is_training,
                              scope=LOGITS_SCOPE_NAME)

        # Resize the logits to have the same dimension before merging.
        output_logit = tf.image.resize_bilinear(
            logits, [input_height, input_width],
            align_corners=True)

        return output_logit


def deeplab_v3_plus_model_fn(features,
                             labels,
                             mode,
                             params):
    num_classes = params['num_classes']
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    deep_model = DeeplabV3Plus(num_classes=num_classes,
                               model_input_size=params['model_input_size'],
                               output_stride=params['output_stride'],
                               weight_decay=params['weight_decay'])
    logits = deep_model.forward_pass(features,
                                     is_training=is_training)
    pred_labels = tf.expand_dims(
        tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)
    one_hot_labels = tf.one_hot(labels,
                                depth=num_classes,
                                on_value=1.0,
                                off_value=0.0,
                                axis=-1)

    logits_by_num_classes = tf.reshape(logits, [-1, num_classes])
    labels_by_num_classes = tf.reshape(one_hot_labels, [-1, num_classes])

    labels_flat = tf.reshape(labels, [-1, ])
    valid_indices = tf.to_int32(labels_flat <= (num_classes - 1))
    valid_labels = tf.dynamic_partition(labels_flat, valid_indices,
                                        num_partitions=2)[1]
    valid_preds = tf.dynamic_partition(tf.reshape(pred_labels, [-1, ]),
                                       valid_indices,
                                       num_partitions=2)[1]
    labels_flat = tf.reshape(valid_labels, [-1, ])
    pred_labels_flat = tf.reshape(valid_preds, [-1, ])

    confusion_matrix = tf.confusion_matrix(
        labels_flat,
        pred_labels_flat,
        num_classes=params['num_classes'])

    predictions = {
        'gt_classes': labels,
        'pred_classes': pred_labels,
        'confusion_matrix': confusion_matrix,
    }

    # Predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

    with tf.name_scope('loss'):
        cross_entropy = tf.losses.softmax_cross_entropy(
            logits=logits_by_num_classes, onehot_labels=labels_by_num_classes)

        # Create a tensor named cross_entropy for logging purposes.
        tf.identity(cross_entropy, name='loss')
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('accuracy'):
        accuracy = tf.metrics.accuracy(
            labels_flat, pred_labels_flat)
        mean_iou = tf.metrics.mean_iou(labels_flat, pred_labels_flat,
                                       params['num_classes'])
    metrics = {'pixel_accuracy': accuracy, 'mean_iou': mean_iou}

    # evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=cross_entropy,
            eval_metric_ops=metrics,
            evaluation_hooks=None,
        )

    images = tf.cast(
        tf.map_fn(decode_org_image, features),
        tf.uint8)
    # Scale up summary image pixel values for better visualization.
    pixel_scaling = max(1, 255 // params['num_classes'])
    summary_label = tf.cast(labels * pixel_scaling, tf.uint8)

    summary_pred_labels = tf.cast(pred_labels * pixel_scaling, tf.uint8)
    tf.summary.image('samples/image', images)
    tf.summary.image('samples/label', summary_label)
    tf.summary.image('samples/prediction', summary_pred_labels)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = get_model_learning_rate(
        global_step,
        params['learning_policy'],
        params['base_learning_rate'],
        params['learning_rate_decay_step'],
        params['learning_rate_decay_factor'],
        params['training_number_of_steps'],
        params['learning_power'],
        params['slow_start_step'],
        params['slow_start_learning_rate'])
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=params['momentum'])
    # Batch norm requires update ops to be added as a dependency to
    # the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(cross_entropy, global_step)

    def compute_mean_iou(total_cm, name='mean_iou'):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
        sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
        cm_diag = tf.to_float(tf.diag_part(total_cm))
        denominator = sum_over_row + sum_over_col - cm_diag

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(tf.cast(
            tf.not_equal(denominator, 0), dtype=tf.float32))

        # If the value of the denominator is 0, set it to 1 to avoid
        # zero division.
        denominator = tf.where(
            tf.greater(denominator, 0),
            denominator,
            tf.ones_like(denominator))
        iou = tf.div(cm_diag, denominator)

        for i in range(params['num_classes']):
            tf.identity(iou[i], name='train_iou_class{}'.format(i))
            tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

        # If the number of valid entries is 0 (no classes) we return 0.
        result = tf.where(
            tf.greater(num_valid_entries, 0),
            tf.reduce_sum(iou, name=name) / num_valid_entries,
            0)
        return result

    train_mean_iou = compute_mean_iou(mean_iou[1])

    tf.identity(train_mean_iou, name='train_mean_iou')
    tf.summary.scalar('train_mean_iou', train_mean_iou)

    tensors_to_log = {
        'learning_rate': learning_rate,
        'cross_entropy': cross_entropy,
        'train_pixel_accuracy': accuracy[1],
        'train_mean_iou': train_mean_iou,
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
    train_hooks = [logging_hook]

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=cross_entropy,
        train_op=train_op,
        training_hooks=train_hooks
    )

