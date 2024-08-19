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

"""Tensorflow implementation of PCGrad.
"""
import fnmatch
import random
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
               use_locking=False,
               allowlist=None,
               denylist=None,
               use_per_variable_impl=True):
    """Initializes the PCGrad class.

    Args:
      optimizer_to_wrap: The optimizer that is being wrapped with PCGrad.
      use_collection_losses: Whether to query collections pcgrad_losses for
        losses list.
      use_locking: Bool. If True apply use locks to prevent concurrent updates
        to variables.
      allowlist: An Iterable of shell-style wildcard strings which are used to
        choose PCGrad variables. A variable will use PCGrad if its
        fully-qualified name matches one or more of the wildcard expressions in
        this iterable (as evaluated by `fnmatch.fnmatchcase`). Otherwise, it
        will be trained without PCGrad. If None, no variables are filtered.
      denylist: An Iterable of shell-style wildcard strings which are used to
        choose PCGrad variables. A variable will use PCGrad if its
        fully-qualified name does not match one or more of the wildcard
        expressions in this iterable (as evaluated by `fnmatch.fnmatchcase`).
        Otherwise, it will be trained without PCGrad. If None, no variables are
        filtered.
      use_per_variable_impl: If set to True, a more memory efficient
        implementation will be used, it might be slower though.
    """
    super(PCGrad, self).__init__(use_locking, self.__class__.__name__)
    self._optimizer = optimizer_to_wrap
    self._use_collection_losses = use_collection_losses
    self._allowlist = allowlist
    self._denylist = denylist
    self._use_per_variable_impl = use_per_variable_impl

  def _create_pcgrad_var_list(self, variables):
    accepts = []
    rejects = []
    if self._allowlist is None:
      self._allowlist = ["*"]
    if self._denylist is None:
      self._denylist = []
    for v in variables:
      if any(fnmatch.fnmatchcase(v.op.name, w)
             for w in self._allowlist) and not any(
                 fnmatch.fnmatchcase(v.op.name, w) for w in self._denylist):
        accepts.append(v)
      else:
        rejects.append(v)
    return accepts, rejects

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
    random.shuffle(loss)

    # Get vars to do PCGrad on.
    pcgrad_vars, other_vars = self._create_pcgrad_var_list(var_list)

    if self._use_per_variable_impl:
      pcgrad_grads = self._compute_projected_grads_per_variable(
          pcgrad_vars, loss)
    else:
      pcgrad_grads = self._compute_projected_grads(pcgrad_vars, loss)
    pcgrad_grads_and_vars = list(zip(pcgrad_grads, pcgrad_vars))

    other_grads_and_vars = []
    if other_vars:
      other_grads_and_vars = self._optimizer.compute_gradients(
          loss=tf.reduce_sum(loss, axis=0),
          var_list=other_vars,
          gate_gradients=gate_gradients,
          aggregation_method=aggregation_method,
          colocate_gradients_with_ops=colocate_gradients_with_ops,
          grad_loss=grad_loss)
    grads_and_vars = pcgrad_grads_and_vars + other_grads_and_vars
    return grads_and_vars

  def _compute_projected_grads_per_variable(self, pcgrad_vars, loss):
    if not pcgrad_vars:
      return []
    # Compute per tasks grads.
    task_var_grads = []
    for task_loss in loss:
      original_pcgrad_grads_vars = self._optimizer.compute_gradients(
          task_loss, pcgrad_vars)
      original_grads = []
      for grad_var in original_pcgrad_grads_vars:
        if grad_var[0] is None:
          continue
        original_grads.append(grad_var[0])
      task_var_grads.append(original_grads)
    var_task_grads = list(zip(*task_var_grads))

    proj_grads = []
    # Loop over vars.
    for task_grads in var_task_grads:
      var_grad = 0
      for task_grad in task_grads:  # Loop over tasks
        if task_grad is not None:
          grad = task_grad
          for inner_task_grad in task_grads:
            if inner_task_grad is not None:
              inner_product = tf.reduce_sum(grad * inner_task_grad)
              proj_direction = inner_product / (
                  tf.reduce_sum(inner_task_grad * inner_task_grad) +
                  tf.constant(1e-5, dtype=grad.dtype))
              grad -= tf.minimum(proj_direction, 0.) * inner_task_grad
          var_grad += grad
      proj_grads.append(var_grad)
    return proj_grads

  def _compute_projected_grads(self, pcgrad_vars, loss):
    if not pcgrad_vars:
      return []
    num_tasks = len(loss)
    loss = tf.stack(loss)
    # Compute per-task gradients.
    def compute_per_task_grads(task):
      grad_list = []
      original_pcgrad_grads_vars = self._optimizer.compute_gradients(
          task, pcgrad_vars)
      original_grads = [grad_var[0] for grad_var in original_pcgrad_grads_vars]
      for grad in original_grads:
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
      for idx, var in enumerate(pcgrad_vars):
        grad_shape = var.get_shape()
        flatten_dim = np.prod(
            [grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
        proj_grad = proj_grads_flatten[j][start_idx:start_idx + flatten_dim]
        proj_grad = tf.reshape(proj_grad, grad_shape)
        if len(proj_grads) < len(pcgrad_vars):
          proj_grads.append(proj_grad)
        else:
          proj_grads[idx] += proj_grad
        start_idx += flatten_dim
    return proj_grads

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
