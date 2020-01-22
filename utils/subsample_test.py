# coding=utf-8
# Copyright 2020 The Tensor2Robot Authors.
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

# Lint as: python3
"""Tests for tensor2robot.utils.subsample."""

from absl.testing import parameterized
import numpy as np
from tensor2robot.utils import subsample
import tensorflow.compat.v1 as tf


class SubsampleTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      'testcase_name': 'length1',
      'min_length': 1,
  }, {
      'testcase_name': 'length_shorter',
      'min_length': 4,
  }, {
      'testcase_name': 'length_longer',
      'min_length': 10,
  })
  def test_subsampling(self, min_length):
    testing_tensor1 = tf.placeholder(tf.float32, shape=(4, None))
    testing_tensor2 = tf.placeholder(tf.float32, shape=(4, None))
    sequence_lengths = tf.placeholder(tf.int64, shape=(4,))
    indices = subsample.get_subsample_indices(sequence_lengths,
                                              min_length)
    sampled1 = tf.gather(testing_tensor1, indices, batch_dims=1)
    sampled2 = tf.gather(testing_tensor2, indices, batch_dims=1)
    with tf.Session() as sess:
      input1 = np.array([[1, 2, 3, 4, 5, 6, 7],
                         [1, 2, 3, 4, 5, 6, 0],
                         [1, 2, 3, 4, 5, 0, 0],
                         [1, 2, 3, 4, 0, 0, 0]])
      input2 = -input1
      seq_len = [7, 6, 5, 4]
      samp1, samp2 = sess.run([sampled1, sampled2],
                              feed_dict={testing_tensor1: input1,
                                         testing_tensor2: input2,
                                         sequence_lengths: seq_len})
      # Verify we never sample the padding.
      self.assertGreater(np.min(np.abs(samp1)), 0)
      self.assertGreater(np.min(np.abs(samp2)), 0)
      # Verify indices are the same for both tensors, should sum to array of all
      # zeros.
      total = samp1 + samp2
      self.assertEqual(total.min(), 0)
      self.assertEqual(total.max(), 0)
      if min_length > 1:
        # Verify first and last always included. In test tensor final entry
        # matches sequence length.
        self.assertAllEqual(samp1.min(axis=1), np.ones(4))
        self.assertAllEqual(samp1.max(axis=1), seq_len)


if __name__ == '__main__':
  tf.test.main()
