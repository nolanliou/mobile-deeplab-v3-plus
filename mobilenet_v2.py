"""MobileNet v2 model
# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""
import os

import tensorflow as tf

import layers
from utils import op


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def expand_input_by_factor(n, divisible_by=8):
    return lambda num_inputs, **_: _make_divisible(num_inputs * n, divisible_by)


def depth_multiply(output_params,
                   multiplier,
                   divisible_by=8,
                   min_depth=8):
    if 'num_outputs' not in output_params:
        return
    d = output_params['num_outputs']
    output_params['num_outputs'] = _make_divisible(d * multiplier,
                                                   divisible_by,
                                                   min_depth)


class Conv2DBN(tf.keras.layers.Layer):
    def __init__(self,
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
                 activation_fn=tf.nn.relu6):
        super(Conv2DBN, self).__init__()
        kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)
        self.conv2d = tf.keras.layers.Conv2D(
            filters=num_outputs,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        if not use_bias and use_bn:
            self.bn = tf.keras.layers.BatchNormalization(
                momentum=bn_momentum,
                center=True,
                scale=True)
        self.activation = activation_fn

    def call(self, inputs, training=True):
        x = self.conv2d(inputs)
        if self.bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x


class MobilenetV2(object):
    def __init__(self,
                 output_stride=None,
                 depth_multiplier=1.0,
                 min_depth=8,
                 divisible_by=8,
                 quant_friendly=False):
        if output_stride is not None:
            if output_stride == 0 or \
                    (output_stride > 1 and output_stride % 2):
                raise ValueError(
                    'Output stride must be None, 1 or a multiple of 2.')
        self.output_stride = output_stride
        self.depth_multiplier = depth_multiplier
        self.min_depth = min_depth
        self.divisible_by = divisible_by
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
                activation_fn=tf.nn.relu6,
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
                if quant_friendly:
                    activation_fn = tf.nn.relu
                net = activation_fn(net)
                tf.summary.histogram('Activation', net)
            return tf.identity(net, name="output")

    def _expanded_conv(self,
                       input_tensor,
                       num_outputs,
                       kernel_size=3,
                       stride=1,
                       padding='SAME',
                       dilation_rate=1,
                       expansion_size=expand_input_by_factor(6),
                       depthwise_location='expand',
                       depthwise_multiplier=1,
                       weight_decay=0.00004,
                       quant_friendly=False,
                       residual=True,
                       is_training=True,
                       scope=None):
        input_depth = input_tensor.get_shape().as_list()[3]
        net = input_tensor
        with tf.variable_scope(scope, default_name="expanded_conv") as s, \
                tf.name_scope(s.original_name_scope):
            if depthwise_location not in [None, 'input', 'output', 'expand']:
                raise TypeError('%r is unknown value for depthwise_location' %
                                depthwise_location)
            if callable(expansion_size):
                expansion_chan = expansion_size(num_inputs=input_depth)
            else:
                expansion_chan = expansion_size
            # expansion
            if depthwise_location == 'expand':
                net = self._conv2d(net,
                                   num_outputs=expansion_chan,
                                   kernel_size=[1, 1],
                                   weight_decay=weight_decay,
                                   is_training=is_training,
                                   quant_friendly=quant_friendly,
                                   scope="expand")
                net = tf.identity(net, name="expand_output")
            # depthwise convolution
            net = layers.depthwise_conv(
                net,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation_rate=dilation_rate,
                depth_multiplier=depthwise_multiplier,
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
            if residual and stride == 1 and input_depth == output_depth:
                net += input_tensor
            return tf.identity(net, name="output")

    def model_def(self):
        model_def = dict(
            spec=[
                op(self._conv2d, num_outputs=32, stride=2,
                   kernel_size=[3, 3]),
                op(self._expanded_conv, num_outputs=16,
                   expansion_size=expand_input_by_factor(1, 1),
                   depthwise_location='input'),
                op(self._expanded_conv, num_outputs=24, stride=2),
                op(self._expanded_conv, num_outputs=24, stride=1),
                op(self._expanded_conv, num_outputs=32, stride=2),
                op(self._expanded_conv, num_outputs=32, stride=1),
                op(self._expanded_conv, num_outputs=32, stride=1),
                op(self._expanded_conv, num_outputs=64, stride=2),
                op(self._expanded_conv, num_outputs=64, stride=1),
                op(self._expanded_conv, num_outputs=64, stride=1),
                op(self._expanded_conv, num_outputs=64, stride=1),
                op(self._expanded_conv, num_outputs=96, stride=1),
                op(self._expanded_conv, num_outputs=96, stride=1),
                op(self._expanded_conv, num_outputs=96, stride=1),
                op(self._expanded_conv, num_outputs=160, stride=2),
                op(self._expanded_conv, num_outputs=160, stride=1),
                op(self._expanded_conv, num_outputs=160, stride=1),
                op(self._expanded_conv, num_outputs=320, stride=1),
                op(self._conv2d, num_outputs=1280, stride=1,
                   kernel_size=[1, 1]),
            ]
        )
        return model_def

    def forward_base(self,
                     input_tensor,
                     final_endpoint=None,
                     is_training=True,
                     scope="MobilenetV2"):
        model_def = self.model_def()
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
                depth_multiply(params,
                               self.depth_multiplier,
                               self.divisible_by,
                               self.min_depth)

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

        with tf.variable_scope('MobilenetV2', reuse=tf.AUTO_REUSE) as scope:
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
                    scope='Conv2d_1c_1x1')

                logits = tf.squeeze(logits, [1, 2])

                logits = tf.identity(logits, name='output')
            end_points['Logits'] = logits
            if prediction_fn:
                end_points['Predictions'] = prediction_fn(logits,
                                                          name='Predictions')
        return logits, end_points

