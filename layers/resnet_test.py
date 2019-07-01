# coding=utf-8
# Copyright 2019 The Tensor2Robot Authors.
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

"""Tests for tensor2robot.layers.resnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from tensor2robot.layers import resnet
import tensorflow as tf


class ResnetTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(('',), ('fubar',), ('dummy/scope'))
  def test_intermediate_values(self, scope):
    with tf.variable_scope(scope):
      image = tf.zeros((2, 224, 224, 3), dtype=tf.float32)
      end_points = resnet.resnet_model(image,
                                       is_training=True,
                                       num_classes=1001,
                                       return_intermediate_values=True)
    tensors = ['initial_conv', 'initial_max_pool', 'pre_final_pool',
               'final_reduce_mean', 'final_dense']
    tensors += [
        'block_layer{}'.format(i + 1) for i in range(4)]
    self.assertEqual(set(tensors), set(end_points.keys()))


if __name__ == '__main__':
  tf.test.main()
