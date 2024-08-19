# coding=utf-8
# Copyright 2024 The Tensor2Robot Authors.
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

"""Build optimizer with the given hyperparamaters.
"""

from absl import logging
import tensorflow.compat.v1 as tf
from tensorflow.contrib import opt as contrib_opt
from tensorflow.contrib.tpu.python.tpu import tpu_function


def BuildOpt(hparams):
  """Constructs the optimizer.

  Args:
    hparams: An instance of tf.HParams, with these parameters:
    - batch_size
    - examples_per_epoch
    - learning_rate
    - learning_rate_decay_factor
    - model_weights_averaging
    - momentum
    - num_epochs_per_decay
    - optimizer
    - rmsprop_decay
    - use_avg_model_params

  Returns:
    opt: The optimizer.
  """
  logging.info('Hyperparameters: %s', hparams)
  batch_size = hparams.batch_size
  examples_per_epoch = hparams.examples_per_epoch
  learning_rate_decay_factor = hparams.learning_rate_decay_factor
  learning_rate = hparams.learning_rate
  model_weights_averaging = hparams.model_weights_averaging
  momentum = hparams.momentum
  num_epochs_per_decay = hparams.num_epochs_per_decay
  optimizer = hparams.optimizer
  rmsprop_decay = hparams.rmsprop_decay
  rmsprop_epsilon = hparams.rmsprop_epsilon
  adam_beta2 = hparams.get('adam_beta2', 0.999)
  adam_epsilon = hparams.get('adam_epsilon', 1e-8)
  use_avg_model_params = hparams.use_avg_model_params

  global_step = tf.train.get_or_create_global_step()

  # Configure the learning rate using an exponetial decay.
  decay_steps = int(examples_per_epoch / batch_size *
                    num_epochs_per_decay)

  learning_rate = tf.train.exponential_decay(
      learning_rate,
      global_step,
      decay_steps,
      learning_rate_decay_factor,
      staircase=True)
  if not tpu_function.get_tpu_context():
    tf.summary.scalar('Learning Rate', learning_rate)

  if optimizer == 'momentum':
    opt = tf.train.MomentumOptimizer(learning_rate, momentum)
  elif optimizer == 'rmsprop':
    opt = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=rmsprop_decay,
        momentum=momentum,
        epsilon=rmsprop_epsilon)
  else:
    opt = tf.train.AdamOptimizer(
        learning_rate,
        beta1=momentum,
        beta2=adam_beta2,
        epsilon=adam_epsilon)

  if use_avg_model_params:
    # Callers of BuildOpt() with use_avg_model_params=True expect the
    # MovingAverageOptimizer to be the last optimizer returned by this function
    # so that the swapping_saver can be constructed from it.
    return contrib_opt.MovingAverageOptimizer(
        opt, average_decay=model_weights_averaging)

  return opt
