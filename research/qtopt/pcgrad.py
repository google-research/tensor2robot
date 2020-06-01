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
"""Tensorflow implementation of PCGrad.
"""
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS
GATE_OP = 1
PCGRAD_LOSSES_COLLECTION = "pcgrad_losses"


class PCGrad(tf.train.Optimizer):
  """Tensorflow implementation of PCGrad.

  Gradient Surgery for Multi-Task Learning:
  https://arxiv.org/pdf/2001.06782.pdf. Copied from the github repo:
  https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py.
  """

  def __init__(self,
               optimizer_to_wrap,
               use_collection_losses=False,
               use_locking=False):
    """Initializes the PCGrad class.

    Args:
      optimizer_to_wrap: The optimizer that is being wrapped with PCGrad.
      use_collection_losses: Whether to query collections pcgrad_losses for
        losses list.
      use_locking: A boolean flag.
    """
    super(PCGrad, self).__init__(use_locking, self.__class__.__name__)
    self._optimizer = optimizer_to_wrap
    self._use_collection_losses = use_collection_losses

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    if self._use_collection_losses:
      loss = tf.get_collection(PCGRAD_LOSSES_COLLECTION)
    assert isinstance(loss, list), "The loss is not a list: %s" % type(loss)
    num_tasks = len(loss)
    loss = tf.stack(loss)
    tf.random.shuffle(loss)

    # Compute per-task gradients.
    def compute_per_task_grads(task):
      grad_list = []
      for grad in tf.gradients(task, var_list):
        if grad is None:
          continue
        grad_list.append(tf.reshape(grad, [
            -1,
        ]))
      return tf.concat(grad_list, axis=0)

    grads_task = tf.vectorized_map(compute_per_task_grads, loss)

    # Compute gradient projections.
    def proj_grad(grad_task):
      for k in range(num_tasks):
        inner_product = tf.reduce_sum(grad_task * grads_task[k])
        proj_direction = inner_product / tf.reduce_sum(
            grads_task[k] * grads_task[k])
        grad_task = grad_task - tf.minimum(proj_direction, 0.) * grads_task[k]
      return grad_task

    proj_grads_flatten = tf.vectorized_map(proj_grad, grads_task)

    # Unpack flattened projected gradients back to their original shapes.
    proj_grads = []
    for j in range(num_tasks):
      start_idx = 0
      for idx, var in enumerate(var_list):
        grad_shape = var.get_shape()
        flatten_dim = np.prod(
            [grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
        proj_grad = proj_grads_flatten[j][start_idx:start_idx + flatten_dim]
        proj_grad = tf.reshape(proj_grad, grad_shape)
        if len(proj_grads) < len(var_list):
          proj_grads.append(proj_grad)
        else:
          proj_grads[idx] += proj_grad
        start_idx += flatten_dim
    grads_and_vars = list(zip(proj_grads, var_list))
    return grads_and_vars

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    return self._optimizer.apply_gradients(grads_and_vars, global_step, name)

# pylint: disable=protected-access
  def __getattr__(self, name):
    """Forward all other calls to the base optimizer."""
    return getattr(self._optimizer, name)

  def _create_slots(self, var_list):
    return self._optimizer._create_slots(var_list)

  def _prepare(self):
    return self._optimizer._prepare()

  def _apply_dense(self, grad, var):
    return self._optimizer._apply_dense(grad, var)

  def _resource_apply_dense(self, grad, var):
    return self._optimizer._resource_apply_dense(grad, var)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    return self._optimizer._apply_sparse_shared(grad, var, indices, scatter_add)

  def _apply_sparse(self, grad, var):
    return self._optimizer._apply_sparse(grad, var)

  def _resource_scatter_add(self, x, i, v):
    return self._optimizer._resource_scatter_add(x, i, v)

  def _resource_apply_sparse(self, grad, var, indices):
    return self._optimizer._resource_apply_sparse(grad, var, indices)

  def _finish(self, update_ops, name_scope):
    return self._optimizer._finish(update_ops, name_scope)
# pylint: enable=protected-access

  def _call_if_callable(self, param):
    """Call the function if param is callable."""
    return param() if callable(param) else param
