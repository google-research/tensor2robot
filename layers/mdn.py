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

"""Functions for building Mixture Density Networks (MDN)."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
import numpy as np
from tensor2robot.meta_learning import meta_tfdata
import tensorflow as tf  # tf
import tensorflow_probability as tfp
from typing import Optional, Tuple, Union

slim = tf.contrib.slim


def get_mixture_distribution(
    params,
    num_alphas,
    sample_size,
    output_mean = None
):
  """Construct a mixture of isotropic gaussians from tensor params.

  Args:
    params: A tensor of shape [..., num_alphas + 2 * num_alphas * sample_size].
    num_alphas: The number of mixture components.
    sample_size: Scalar, the size of a single distribution sample.
    output_mean: Optional translation for each component mean.
  Returns:
    A gaussian mixture distribution.
  """
  # Total size of means across all mixture components.
  num_mus = num_alphas * sample_size
  if params.shape[-1] != num_alphas + 2 * num_mus:
    raise ValueError(
        'Params has unexpected shape {:d}.'.format(params.shape.as_list()[-1]))
  alphas = params[Ellipsis, :num_alphas]
  offset = num_alphas
  batch_dims = []
  # Strip None dimensions from list (for exporter placeholders).
  for d in params.shape.as_list()[:-1]:
    batch_dims.append(-1 if d is None else d)
  mus_shape = batch_dims + [num_alphas, sample_size]
  mus = tf.reshape(params[Ellipsis, offset:offset+num_mus], mus_shape)
  offset += num_mus
  # Assume a diagonal covariance, so sigmas.shape == mus.shape.
  sigmas = tf.reshape(params[Ellipsis, offset:offset+num_mus], mus_shape)
  if output_mean is not None:
    mus = mus + output_mean
  mix_dist = tfp.distributions.Categorical(logits=alphas)
  comp_dist = tfp.distributions.MultivariateNormalDiag(
      loc=mus, scale_diag=tf.nn.softplus(sigmas))
  gm = tfp.distributions.MixtureSameFamily(
      mixture_distribution=mix_dist, components_distribution=comp_dist)
  return gm


@gin.configurable
def predict_mdn_params(
    inputs,
    num_alphas,
    sample_size,
    condition_sigmas = False):
  """Outputs parameters of a mixture density network given inputs.

  Args:
    inputs: A tensor input to compute the MDN parameters from.
    num_alphas: The number of mixture components.
    sample_size: Scalar, the size of a single distribution sample.
    condition_sigmas: If True, the sigma params are conditioned on `inputs`.
      Otherwise they are simply learned variables.
  Returns:
    dist_params: A tensor of shape
      [..., num_alphas + 2 * num_alphas * sample_size]
    aux_output: auxiliary output of shape [..., aux_output_dim] if
      aux_output_dim is > 0.
  """
  num_mus = num_alphas * sample_size
  # Assume isotropic gaussian components.
  num_sigmas = num_alphas * sample_size
  num_fc_outputs = num_alphas + num_mus
  if condition_sigmas:
    num_fc_outputs = num_fc_outputs + num_sigmas
  dist_params = slim.fully_connected(
      inputs, num_fc_outputs, activation_fn=None,
      scope='mdn_params')
  if not condition_sigmas:
    # Sigmas initialized so that softplus(sigmas) = 1.
    sigmas = tf.get_variable(
        'mdn_stddev_inputs',
        shape=[num_sigmas],
        dtype=tf.float32,
        initializer=tf.constant_initializer(np.log(np.e - 1)))
    tiled_sigmas = tf.tile(
        sigmas[None], tf.stack([tf.shape(dist_params)[0], 1]))
    dist_params = tf.concat([dist_params, tiled_sigmas], axis=-1)
  return dist_params


def gaussian_mixture_approximate_mode(
    gm):
  """Returns the mean of the most probable mixture component."""
  # Find the most likely mixture component.
  mode_alpha = gm.mixture_distribution.mode()[Ellipsis, None]
  mus = gm.components_distribution.mean()
  # Gather the mean of the most likely component.
  return tf.squeeze(tf.batch_gather(mus, mode_alpha), axis=-2)


@gin.configurable
class MDNDecoder(object):
  """Object-oriented API for creating mixture density networks."""

  def __init__(self, num_mixture_components = 1):
    """Constructor.

    Args:
      num_mixture_components: The number of gaussian mixture components. Use 1
        for standard mean squared error regression.
    """
    self._num_mixture_components = num_mixture_components

  def __call__(self, params, output_size):
    """Applies the model.

    Args:
      params: Features conditioning the output distribution.
      output_size: Dimensionality of output distribution.
    Returns:
      tfp.Distribution object.
    """
    # predict_mdn_params cannot handle meta-batches.
    dist_params = meta_tfdata.multi_batch_apply(
        predict_mdn_params, 3,
        params, self._num_mixture_components,
        output_size, condition_sigmas=False)
    # TODO(ejang): One limitation of a stateful module builder is that a wrapper
    # model (i.e. MAMLModel) may call this module multiple times and overwrite
    # self._gm before the appropriate loss() is called. A potential solution is
    # decoder objects that implement stateless functions (i.e. passing the state
    # to a handler). We should fix this later.
    self._gm = get_mixture_distribution(
        dist_params, self._num_mixture_components, output_size)
    action = gaussian_mixture_approximate_mode(self._gm)
    return action

  def loss(self, labels):
    nll_local = -self._gm.log_prob(labels.action)
    # Average across batch, sequence.
    return tf.reduce_mean(nll_local)

