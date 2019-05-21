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

"""Tests for DeepLab model and some helper functions."""

import tensorflow as tf

from deeplab_v3_plus import DeeplabV3Plus


class DeeplabModelTest(tf.test.TestCase):

    def testForwardpassDeepLabv3plus(self):
        input_size = [33, 33]

        model = DeeplabV3Plus(num_classes=3,
                              model_input_size=input_size,
                              output_stride=16,
                              add_image_level_feature=True,
                              aspp_with_batch_norm=True)

        g = tf.Graph()
        with g.as_default():
            with self.test_session(graph=g) as sess:
                inputs = tf.random_uniform(
                    (1, input_size[0], input_size[1], 3))
                logits = model.forward(inputs)
                # for t in logits.graph.get_operations():
                #     print(t.name)

                sess.run(tf.global_variables_initializer())
                outputs = sess.run(logits)

                self.assertTrue(outputs.any())

    def testForwardpassDeepLabv3plusMobilenetV3(self):
        input_size = [512, 512]

        model = DeeplabV3Plus(num_classes=3,
                              backbone='MobilenetV3',
                              model_input_size=input_size,
                              output_stride=16,
                              add_image_level_feature=True,
                              aspp_with_batch_norm=True)

        g = tf.Graph()
        with g.as_default():
            with self.test_session(graph=g) as sess:
                inputs = tf.random_uniform(
                    (1, input_size[0], input_size[1], 3))
                logits = model.forward(inputs)
                # for t in logits.graph.get_operations():
                #     print(t.name)

                sess.run(tf.global_variables_initializer())
                outputs = sess.run(logits)

                self.assertTrue(outputs.any())


if __name__ == '__main__':
  tf.test.main()
