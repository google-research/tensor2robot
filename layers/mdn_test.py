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

"""Test for tensor2robot.layers.mdn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from tensor2robot.layers import mdn
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


class MDNTest(tf.test.TestCase, parameterized.TestCase):

  def test_get_mixture_distribution(self):
    sample_size = 10
    num_alphas = 5
    batch_shape = (4, 2)
    alphas = tf.random.normal(batch_shape + (num_alphas,))
    mus = tf.random.normal(batch_shape + (sample_size * num_alphas,))
    sigmas = tf.random.normal(batch_shape + (sample_size * num_alphas,))
    params = tf.concat([alphas, mus, sigmas], -1)
    output_mean_np = np.random.normal(size=(sample_size,))
    gm = mdn.get_mixture_distribution(
        params, num_alphas, sample_size, output_mean=output_mean_np)
    self.assertEqual(gm.batch_shape, batch_shape)
    self.assertEqual(gm.event_shape, sample_size)

    # Check that the component means were translated by output_mean_np.
    component_means = gm.components_distribution.mean()
    with self.test_session() as sess:
      # Note: must get values from the same session run, since params will be
      # randomized across separate session runs.
      component_means_np, mus_np = sess.run([component_means, mus])
      mus_np = np.reshape(mus_np, component_means_np.shape)
      self.assertAllClose(component_means_np, mus_np + output_mean_np)

  @parameterized.parameters((True,), (False,))
  def test_predict_mdn_params(self, condition_sigmas):
    sample_size = 10
    num_alphas = 5
    inputs = tf.random.normal((2, 16))
    with tf.variable_scope('test_scope'):
      dist_params = mdn.predict_mdn_params(
          inputs, num_alphas, sample_size, condition_sigmas=condition_sigmas)
    expected_num_params = num_alphas * (1 + 2 * sample_size)
    self.assertEqual(dist_params.shape.as_list(), [2, expected_num_params])

    gm = mdn.get_mixture_distribution(dist_params, num_alphas, sample_size)
    stddev = gm.components_distribution.stddev()
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      stddev_np = sess.run(stddev)
      if condition_sigmas:
        # Standard deviations should vary with input.
        self.assertNotAllClose(stddev_np[0], stddev_np[1])
      else:
        # Standard deviations should *not* vary with input.
        self.assertAllClose(stddev_np[0], stddev_np[1])

  def test_gaussian_mixture_approximate_mode(self):
    sample_size = 10
    num_alphas = 5
    # Manually set alphas to 1 in zero-th column and 0 elsewhere, making the
    # first component the most likely.
    alphas = tf.one_hot(2 * [0], num_alphas)
    mus = tf.random.normal((2, num_alphas, sample_size))
    sigmas = tf.ones_like(mus)
    mix_dist = tfp.distributions.Categorical(logits=alphas)
    comp_dist = tfp.distributions.MultivariateNormalDiag(
        loc=mus, scale_diag=sigmas)
    gm = tfp.distributions.MixtureSameFamily(
        mixture_distribution=mix_dist, components_distribution=comp_dist)
    approximate_mode = mdn.gaussian_mixture_approximate_mode(gm)
    with self.test_session() as sess:
      approximate_mode_np, mus_np = sess.run([approximate_mode, mus])
      # The approximate mode should be the mean of the zero-th (most likely)
      # component.
      self.assertAllClose(approximate_mode_np, mus_np[:, 0, :])


if __name__ == '__main__':
  tf.test.main()
