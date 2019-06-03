"""MobileNet v3 model
# Reference
- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
"""
import os

import tensorflow as tf

import layers
from utils import op


class MobilenetV3(object):
    def __init__(self,
                 model_type='large',
                 output_stride=None,
                 quant_friendly=False):
        if output_stride is not None:
            if output_stride == 0 or \
                    (output_stride > 1 and output_stride % 2):
                raise ValueError(
                    'Output stride must be None, 1 or a multiple of 2.')
        self.model_type = model_type
        self.output_stride = output_stride
        # remove bn and activation behind depthwise-convolution
        # replace relu6 with relu
        self.quant_friendly = quant_friendly
        self.losses_list = []

    def losses(self):
        return self.losses_list

    def _conv2d(self,
                input_tensor,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                dilation_rate=1,
                stddev=0.09,
                weight_decay=0.00004,
                use_bias=False,
                use_bn=True,
                bn_momentum=0.997,
                activation_fn=tf.nn.relu,
                quant_friendly=False,
                is_training=True,
                scope=None):
        net = input_tensor
        kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)
        with tf.variable_scope(scope, default_name="Conv"):
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
            self.losses_list.extend(conv2d.losses)
            tf.summary.histogram('Weights', conv2d.weights[0])
            if not use_bias and use_bn:
                # keras layers' update op is not in global update_op collections
                net = tf.layers.batch_normalization(
                    net,
                    momentum=bn_momentum,
                    training=is_training,
                    name='BatchNorm')
            if activation_fn:
                net = activation_fn(net)
                tf.summary.histogram('Activation', net)
            return tf.identity(net, name="output")

    def _expanded_conv(self,
                       input_tensor,
                       expansion_size,
                       num_outputs,
                       kernel_size=3,
                       stride=1,
                       padding='SAME',
                       dilation_rate=1,
                       use_se=False,
                       weight_decay=0.00004,
                       quant_friendly=False,
                       activation_fn=tf.nn.relu,
                       is_training=True,
                       scope=None):
        input_depth = input_tensor.get_shape().as_list()[3]
        net = input_tensor
        with tf.variable_scope(scope, default_name="expanded_conv") as s, \
                tf.name_scope(s.original_name_scope):
            # expansion
            net = self._conv2d(net,
                               num_outputs=expansion_size,
                               kernel_size=[1, 1],
                               weight_decay=weight_decay,
                               is_training=is_training,
                               activation_fn=activation_fn,
                               scope="expand")
            net = tf.identity(net, name="expand_output")
            # depthwise convolution
            net = layers.depthwise_conv(
                net,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation_rate=dilation_rate,
                use_se=use_se,
                activation_fn=activation_fn,
                quant_friendly=quant_friendly,
                is_training=is_training,
                scope="depthwise")
            net = tf.identity(net, name="depthwise_output")
            # projection
            net = self._conv2d(net,
                               num_outputs=num_outputs,
                               kernel_size=[1, 1],
                               weight_decay=weight_decay,
                               activation_fn=None,
                               is_training=is_training,
                               scope="project")
            net = tf.identity(net, name="project_output")
            output_depth = net.get_shape().as_list()[3]
            if stride == 1 and input_depth == output_depth:
                net += input_tensor
            return tf.identity(net, name="output")

    def large_model_def(self):
        model_def = dict(
            spec=[
                op(self._conv2d, kernel_size=3, num_outputs=16,
                   activation_fn=layers.hard_swish, stride=2),
                op(self._expanded_conv, kernel_size=3,
                   expansion_size=16, num_outputs=16, use_se=False, stride=1),
                op(self._expanded_conv, kernel_size=3,
                   expansion_size=64, num_outputs=24, use_se=False, stride=2),
                op(self._expanded_conv, kernel_size=3,
                   expansion_size=72, num_outputs=24, use_se=False, stride=1),
                op(self._expanded_conv, kernel_size=5,
                   expansion_size=72, num_outputs=40, use_se=True, stride=2),
                op(self._expanded_conv, kernel_size=5,
                   expansion_size=120, num_outputs=40, use_se=True, stride=1),
                op(self._expanded_conv, kernel_size=5,
                   expansion_size=120, num_outputs=40, use_se=True, stride=1),
                op(self._expanded_conv, kernel_size=3,
                   expansion_size=240, num_outputs=80, use_se=False, stride=2,
                   activation_fn=layers.hard_sigmoid),
                op(self._expanded_conv, kernel_size=3,
                   expansion_size=200, num_outputs=80, use_se=False, stride=1,
                   activation_fn=layers.hard_sigmoid),
                op(self._expanded_conv, kernel_size=3,
                   expansion_size=184, num_outputs=80, use_se=False, stride=1,
                   activation_fn=layers.hard_sigmoid),
                op(self._expanded_conv, kernel_size=3,
                   expansion_size=184, num_outputs=80, use_se=False, stride=1,
                   activation_fn=layers.hard_sigmoid),
                op(self._expanded_conv, kernel_size=3,
                   expansion_size=480, num_outputs=112, use_se=True, stride=1,
                   activation_fn=layers.hard_sigmoid),
                op(self._expanded_conv, kernel_size=3,
                   expansion_size=672, num_outputs=112, use_se=True, stride=1,
                   activation_fn=layers.hard_sigmoid),
                op(self._expanded_conv, kernel_size=5,
                   expansion_size=672, num_outputs=112, use_se=True, stride=1,
                   activation_fn=layers.hard_sigmoid),
                op(self._expanded_conv, kernel_size=5,
                   expansion_size=672, num_outputs=160, use_se=True, stride=2,
                   activation_fn=layers.hard_sigmoid),
                op(self._expanded_conv, kernel_size=5,
                   expansion_size=960, num_outputs=160, use_se=True, stride=1,
                   activation_fn=layers.hard_sigmoid),
                op(self._conv2d, kernel_size=1, num_outputs=960,
                   activation_fn=layers.hard_swish, stride=1),
            ]
        )
        return model_def

    def small_model_def(self):
        model_def = dict(
            spec=[
                op(self._conv2d, kernel_size=3, num_outputs=16,
                   activation_fn=layers.hard_swish, stride=2),
                op(self._expanded_conv, kernel_size=3,
                   expansion_size=16, num_outputs=16, use_se=True, stride=2),
                op(self._expanded_conv, kernel_size=3,
                   expansion_size=72, num_outputs=24, use_se=False, stride=2),
                op(self._expanded_conv, kernel_size=3,
                   expansion_size=88, num_outputs=24, use_se=False, stride=1),
                op(self._expanded_conv, kernel_size=5,
                   expansion_size=96, num_outputs=40, use_se=True, stride=2,
                   activation_fn=layers.hard_sigmoid),
                op(self._expanded_conv, kernel_size=5,
                   expansion_size=240, num_outputs=40, use_se=True, stride=1,
                   activation_fn=layers.hard_sigmoid),
                op(self._expanded_conv, kernel_size=5,
                   expansion_size=240, num_outputs=40, use_se=True, stride=1,
                   activation_fn=layers.hard_sigmoid),
                op(self._expanded_conv, kernel_size=5,
                   expansion_size=120, num_outputs=48, use_se=True, stride=1,
                   activation_fn=layers.hard_sigmoid),
                op(self._expanded_conv, kernel_size=5,
                   expansion_size=114, num_outputs=48, use_se=True, stride=1,
                   activation_fn=layers.hard_sigmoid),
                op(self._expanded_conv, kernel_size=5,
                   expansion_size=288, num_outputs=96, use_se=True, stride=2,
                   activation_fn=layers.hard_sigmoid),
                op(self._expanded_conv, kernel_size=5,
                   expansion_size=576, num_outputs=96, use_se=True, stride=1,
                   activation_fn=layers.hard_sigmoid),
                op(self._expanded_conv, kernel_size=5,
                   expansion_size=576, num_outputs=96, use_se=True, stride=1,
                   activation_fn=layers.hard_sigmoid),
                op(self._conv2d, kernel_size=1, num_outputs=576,
                   activation_fn=layers.hard_swish, stride=1),
            ]
        )
        return model_def

    def forward_base(self,
                     input_tensor,
                     final_endpoint=None,
                     is_training=True,
                     scope="MobilenetV3"):
        if self.model_type == 'small':
            model_def = self.small_model_def()
        else:
            model_def = self.large_model_def()
        endpoints = {}
        scopes = {}
        with tf.variable_scope(scope) as s, \
                tf.name_scope(s.original_name_scope):
            # The current_stride variable keeps track of the output stride of
            # the activations, i.e., the running product of convolution strides
            # up to the current network layer.
            # This allows us to invoke atrous convolution whenever applying the
            # next convolution would result in the activations
            # having output stride larger than the target output_stride.
            current_stride = 1

            # The atrous convolution rate parameter.
            rate = 1

            net = input_tensor

            for i, opdef in enumerate(model_def['spec']):
                params = dict(opdef.params)

                params['is_training'] = is_training
                stride = params.get('stride', 1)
                if self.output_stride is not None and \
                        current_stride == self.output_stride:
                    # If we have reached the target output_stride,
                    # then we need to employ atrous convolution with stride=1
                    # and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    layer_stride = 1
                    layer_rate = rate
                    rate *= stride
                else:
                    layer_stride = stride
                    layer_rate = 1
                    current_stride *= stride
                # Update params.
                params['quant_friendly'] = self.quant_friendly
                params['stride'] = layer_stride
                # Only insert rate to params if rate > 1.
                if layer_rate > 1:
                    params['dilation_rate'] = layer_rate
                endpoint = 'layer_%d' % (i + 1)
                try:
                    net = opdef.op(net, **params)
                except Exception:
                    print('Failed to create op %i: %r params: %r' %
                          (i, opdef, params))
                    raise
                endpoints[endpoint] = net
                scope_name = os.path.dirname(net.name)
                scopes[scope_name] = endpoint
                if final_endpoint is not None and endpoint == final_endpoint:
                    break
            # Add all tensors that end with 'output' to endpoints
            for t in net.graph.get_operations():
                scope_name = os.path.dirname(t.name)
                bn = os.path.basename(t.name)
                if scope_name in scopes and t.name.endswith('output'):
                    endpoints[scopes[scope_name] + '/' + bn] = t.outputs[0]
            return net, endpoints

    def forward(self,
                input_tensor,
                num_classes=1001,
                final_endpoint=None,
                prediction_fn=tf.nn.softmax,
                is_training=True,
                base_only=False):
        input_shape = input_tensor.get_shape().as_list()
        if len(input_shape) != 4:
            raise ValueError(
                'Expected rank 4 input, was: %d' % len(input_shape))

        with tf.variable_scope('MobilenetV3', reuse=tf.AUTO_REUSE) as scope:
            inputs = tf.identity(input_tensor, 'input')
            net, end_points = self.forward_base(
                inputs,
                final_endpoint=final_endpoint,
                is_training=is_training,
                scope=scope)
            if base_only:
                return net, end_points

            net = tf.identity(net, name='embedding')

            with tf.variable_scope('Logits'):
                net = layers.global_pool(net)
                end_points['global_pool'] = net
                if not num_classes:
                    return net, end_points
                if is_training:
                    net = tf.keras.layers.Dropout(rate=0.2)(net)
                net = self._conv2d(
                    net,
                    1280,
                    [1, 1],
                    use_bias=True,
                    use_bn=False,
                    activation_fn=layers.hard_swish,
                    is_training=is_training,
                    scope='Conv2d_1x1_0')
                # 1 x 1 x num_classes
                # Note: legacy scope name.
                logits = self._conv2d(
                    net,
                    num_classes,
                    [1, 1],
                    use_bias=True,
                    use_bn=False,
                    activation_fn=None,
                    is_training=is_training,
                    scope='Conv2d_1x1_1')

                logits = tf.squeeze(logits, [1, 2])

                logits = tf.identity(logits, name='output')
            end_points['Logits'] = logits
            if prediction_fn:
                end_points['Predictions'] = prediction_fn(logits,
                                                          name='Predictions')
        return logits, end_points

