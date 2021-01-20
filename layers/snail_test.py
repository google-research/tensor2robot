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

# Lint as python3
"""Tests for SNAIL."""

import numpy as np
from six.moves import range
from tensor2robot.layers import snail
import tensorflow.compat.v1 as tf


class SNAILTest(tf.test.TestCase):

  def test_CausalConv(self):
    x = tf.random.normal((4, 10, 8))
    y = snail.CausalConv(x, 1, 5)
    self.assertEqual(y.shape, (4, 10, 5))

  def test_DenseBlock(self):
    x = tf.random.normal((4, 10, 8))
    y = snail.DenseBlock(x, 1, 5)
    self.assertEqual(y.shape, (4, 10, 13))

  def test_TCBlock(self):
    sequence_length = 10
    x = tf.random.normal((4, sequence_length, 8))
    y = snail.TCBlock(x, sequence_length, 5)
    self.assertEqual(y.shape, (4, 10, 8 + 4*5))

  def test_CausallyMaskedSoftmax(self):
    num_rows = 5
    x = tf.random.normal((num_rows, 3))
    logits = tf.matmul(x, tf.linalg.transpose(x))
    y = snail.CausallyMaskedSoftmax(logits)
    with self.test_session() as sess:
      y_ = sess.run(y)
      idx = np.triu_indices(num_rows, 1)
      np.testing.assert_array_equal(y_[idx], 0.)
      # Testing that each row sums to 1.
      for i in range(num_rows):
        np.testing.assert_almost_equal(np.sum(y_[i, :]), 1.0)

  def test_AttentionBlock(self):
    x = tf.random.normal((4, 10, 8))
    y, end_points = snail.AttentionBlock(x, 3, 5)
    self.assertEqual(y.shape, (4, 10, 5+8))
    self.assertEqual(end_points['attn_prob'].shape, (4, 10, 10))

if __name__ == '__main__':
  tf.test.main()
