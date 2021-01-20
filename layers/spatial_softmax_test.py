# coding=utf-8
# Copyright 2021 The Tensor2Robot Authors.
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

"""Tests for Spatial Softmax Layer."""

import numpy as np
from tensor2robot.layers import spatial_softmax
import tensorflow.compat.v1 as tf


class SpatialSoftmaxTest(tf.test.TestCase):

  def test_SpatialGumbelSoftmax(self):

    features = tf.convert_to_tensor(
        np.random.normal(size=(32, 16, 16, 64)).astype(np.float32))
    with tf.variable_scope('mean_pool'):
      expected_feature_points, softmax = spatial_softmax.BuildSpatialSoftmax(
          features, spatial_gumbel_softmax=False)
    with tf.variable_scope('gumbel_pool'):
      gumbel_feature_points, gumbel_softmax = (
          spatial_softmax.BuildSpatialSoftmax(
              features, spatial_gumbel_softmax=True))
    self.assertEqual(expected_feature_points.shape, gumbel_feature_points.shape)
    self.assertEqual(softmax.shape, gumbel_softmax.shape)

if __name__ == '__main__':
  tf.test.main()
