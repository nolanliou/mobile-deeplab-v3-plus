# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for mobilenet_v3."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np

from mobilenet_v3 import MobilenetV3


def find_ops(optype):
    """Find ops of a given type in graphdef or a graph.

    Args:
      optype: operation type (e.g. Conv2D)
    Returns:
       List of operations.
    """
    gd = tf.get_default_graph()
    return [var for var in gd.get_operations() if var.type == optype]


class MobilenetV3Test(tf.test.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def testCreation(self):
        model = MobilenetV3()
        net, endpoints = model.forward(
            tf.placeholder(tf.float32, (10, 224, 224, 16)))

        # for t in net.graph.get_operations():
        #     print(t.name)
        spec = model.large_model_def()
        num_convs = len(find_ops('Conv2D'))

        # This is mostly a sanity test. No deep reason for these particular
        # constants.
        #
        # All but first 2 and last one have  two convolutions, and there is one
        # extra conv that is not in the spec. (logits)
        self.assertEqual(num_convs, len(spec['spec']) * 2 - 2 + 2)
        # Check that depthwise are exposed.
        for i in range(2, 17):
            self.assertIn('layer_%d/depthwise_output' % i, endpoints)

    def testCreationNoClasses(self):
        model = MobilenetV3()
        net, endpoints = model.forward(
            tf.placeholder(tf.float32, (10, 224, 224, 16)),
            num_classes=None)
        self.assertIs(net, endpoints['global_pool'])

    def testImageSizes(self):
        model = MobilenetV3()
        for input_size, output_size in [(224, 7), (192, 6), (160, 5),
                                        (128, 4), (96, 3)]:
            tf.reset_default_graph()
            net, endpoints = model.forward(
                tf.placeholder(tf.float32, (10, input_size, input_size, 16)))

            self.assertEqual(
                endpoints['layer_17/output'].get_shape().as_list()[1:3],
                [output_size] * 2)

    def testWithOutputStride8(self):
        model = MobilenetV3(output_stride=8)
        net, _ = model.forward_base(
            tf.placeholder(tf.float32, (10, 224, 224, 16)))
        self.assertEqual(net.get_shape().as_list()[1:3], [28, 28])

    def testMobilenetBase(self):
        tf.reset_default_graph()
        # Verifies that mobilenet_base returns pre-pooling layer.
        model = MobilenetV3()
        net, _ = model.forward_base(
            tf.placeholder(tf.float32, (10, 224, 224, 16)))
        self.assertEqual(net.get_shape().as_list(), [10, 7, 7, 960])

    def testWithOutputStride16(self):
        tf.reset_default_graph()
        model = MobilenetV3(output_stride=16)
        net, _ = model.forward_base(
            tf.placeholder(tf.float32, (10, 224, 224, 16)))
        self.assertEqual(net.get_shape().as_list()[1:3], [14, 14])


if __name__ == '__main__':
    tf.test.main()
