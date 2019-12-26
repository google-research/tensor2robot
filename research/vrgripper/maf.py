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

# Lint as: python2, python3
"""Conditional density estimation with masked autoregressive flow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from tensorflow.contrib import slim
tfd = tfp.distributions
tfb = tfp.bijectors


def init_once(x, name):
  """Return a variable initialized with a constant value.

  This is used to initialize Permutation bijectors. See [1] for more information
  on why returning Permute(np.random.permutation(event_size)) is unsafe.

  Args:
    x: A TF Variables initializer or constant-valued tensor.
    name: String name for the returned variable.

  Returns:
    Variable copy of the tensor.

  References:

  [1] https://www.tensorflow.org/probability/api_docs/python/
  tfp/bijectors/Permute
  """
  return tf.get_variable(name, initializer=x, trainable=False)


def maf_bijector(event_size, num_flows, hidden_layers):
  """Construct a chain of MAF flows into a single bijector."""
  bijectors = []
  for i in range(num_flows):
    bijectors.append(tfb.MaskedAutoregressiveFlow(
        shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
            hidden_layers=hidden_layers)))
    bijectors.append(
        tfb.Permute(
            permutation=init_once(
                np.random.permutation(event_size).astype('int32'),
                name='permute_%d' % i)))
  # Chain the bijectors, leaving out the last permutation bijector.
  return tfb.Chain(list(reversed(bijectors[:-1])))


@gin.configurable
class MAFDecoder(object):
  """Decoder using a Masked Autoregressive Flow.

  Conditioning is specified by warping the centers of the base isotropic normal
  distributions, e.g. MAF(N(mu, 1)), where mu is the incoming conditioning
  parameters. This allows us to avoid having to incorporate conditioning into
  the actual bijector.
  """

  def __init__(self, num_flows=1, hidden_layers=None):
    self._num_flows = num_flows
    self._hidden_layers = hidden_layers or [512, 512]

  def __call__(self, params, output_size):
    mus = slim.fully_connected(
        params, output_size, activation_fn=None, scope='maf_mus')
    base_dist = tfd.MultivariateNormalDiag(
        loc=mus, scale_diag=tf.ones_like(mus))
    event_shape = base_dist.event_shape.as_list()
    if np.any([event_shape[0] > l for l in self._hidden_layers]):
      raise ValueError(
          'MAF hidden layers have to be at least as wide as event size.')
    self._maf = tfd.TransformedDistribution(
        distribution=base_dist,
        bijector=maf_bijector(
            event_shape[0], self._num_flows, self._hidden_layers))
    return self._maf.sample()

  def loss(self, labels):
    nll_local = -self._maf.log_prob(labels.action)
    # Average across batch, sequence.
    return tf.reduce_mean(nll_local)
