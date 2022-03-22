# coding=utf-8
# Copyright 2022 The Tensor2Robot Authors.
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
"""A generatic MAML model which can operate on any TFModel."""

import abc
import copy
from typing import Dict, Optional, Text, TypeVar

from absl import logging
import gin
import six
from six.moves import range
from tensor2robot.meta_learning import maml_inner_loop
from tensor2robot.meta_learning import meta_tfdata
from tensor2robot.meta_learning import preprocessors
from tensor2robot.models import abstract_model
from tensor2robot.preprocessors import abstract_preprocessor
from tensor2robot.utils import tensorspec_utils as utils
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow.compat.v1 as tf  # tf
from tensorflow.contrib import training as contrib_training


# pylint: disable=invalid-name
MAMLPreprocessorType = TypeVar(
    'MAMLPreprocessorType', bound=preprocessors.MAMLPreprocessorV2)
# pylint: enable=invalid-name


def pfor_map_fn(fn, elems):
  """Provides a map_fn-like wrapper around pfor.

  TODO(T2R_CONTRIBUTORS): For now, fn's conv filters should be loop invariant.
  TODO(T2R_CONTRIBUTORS): Using `AssignSubvariableOp` in fn will cause errors.

  Args:
    fn: The callable to be performed. It accepts one argument, which will have
      the same (possibly nested) structure as `elems`. It should output a
      tensor or structure of tensors.
    elems: A tensor or (possibly nested) structure of tensors, each of which
      will be unpacked along their first dimension. The nested sequence of
      resulting slices will be applied to `fn`.

  Returns:
    A tensor or (possibly nested) structure of tensors, resulting from stacking
    the outputs of `fn` on the unpacked `elems`.
  """
  def loop_fn(i):
    """Calls fn on the i'th entry of each tensor in `elems`."""
    gathered_elems = tf.nest.map_structure(lambda x: tf.gather(x, i), elems)
    return fn(gathered_elems)
  batch_size = tf.shape(tf.nest.flatten(elems)[0])[0]
  return tf.vectorized_map(loop_fn, batch_size)


@gin.configurable
@six.add_metaclass(abc.ABCMeta)
class MAMLModel(abstract_model.AbstractT2RModel):
  """Base class for Meta-Learning Models (e.g. MAML)."""

  def __init__(
      self,
      base_model,
      preprocessor_cls = None,
      num_inner_loop_steps = 1,
      var_scope = None,
      use_parallel_for = False,
      **kwargs):
    """Initialize the MAMLModel.

    Args:
      base_model: The base model whose inference/train functions are used in the
        MAML inner loop.
      preprocessor_cls: The preprocessor class.
      num_inner_loop_steps: The number of inner loop adaptation steps.
      var_scope: Optionally specify a scope of variables to train on. Defaults
        to using all trainable variables.
      use_parallel_for: Use parallel_for.pfor instead of map_fn to vectorize
        across tasks. DO NOT USE if the base model contains conv layers or uses
        exponential moving average (e.g. for soft-target networks).
      **kwargs: Parent keyword args.

    Raises:
      ValueError: If the preprocessor_cls is invalid.
    """
    super(MAMLModel, self).__init__(**kwargs)
    self._base_model = base_model
    self._preprocessor_cls = preprocessor_cls
    self._num_inner_loop_steps = num_inner_loop_steps
    self._var_scope = var_scope
    self._use_parallel_for = use_parallel_for
    if self._num_inner_loop_steps < 1:
      logging.warning(
          'num_inner_loop_steps has to be > 0, but is %s therefore has been'
          'set to 1.', num_inner_loop_steps)
      self._num_inner_loop_steps = 1
    if self._preprocessor_cls is not None and not issubclass(
        preprocessor_cls, preprocessors.MAMLPreprocessorV2):
      raise ValueError('Only instances of MAMLPreprocessorV2 are supported.')

  @property
  def default_preprocessor_cls(self):
    return preprocessors.MAMLPreprocessorV2

  @property
  def preprocessor(self):
    preprocessor_cls = self._preprocessor_cls
    if preprocessor_cls is None:
      preprocessor_cls = self.default_preprocessor_cls
    self._preprocessor = preprocessor_cls(self._base_model.preprocessor)
    return self._preprocessor

  def get_feature_specification(
      self, mode):
    """See parent class."""
    return preprocessors.create_maml_feature_spec(
        self._base_model.get_feature_specification(mode),
        self._base_model.get_label_specification(mode))

  def get_label_specification(
      self, mode):
    """See parent class."""
    return preprocessors.create_maml_label_spec(
        self._base_model.get_label_specification(mode))

  def get_feature_specification_for_packing(
      self, mode):
    """See parent class."""
    return self._base_model.preprocessor.get_in_feature_specification(mode)

  def get_label_specification_for_packing(
      self, mode):
    """See parent class."""
    return self._base_model.preprocessor.get_in_label_specification(mode)

  def infer_base_model_output_dtypes(self, mode,
                                     params):
    """Infer the dtypes of the model in a separate graph.

    Args:
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.
      params: An optional dict of hyper parameters that will be passed into
        input_fn and model_fn. Keys are names of parameters, values are basic
        python types. There are reserved keys for TPUEstimator, including
        'batch_size'.

    Returns:
      dtypes: A dict containing all output dtypes {str: dtype}.
    """
    dtype_inference_graph = tf.Graph()
    with dtype_inference_graph.as_default():
      with tf.variable_scope('IGNORE_ONLY_TO_INFER_OUTPUT_DTYPES'):
        # In this graph we can now create placeholders in order to infer the
        # right dtype of the outputs.
        feature_spec = self.get_feature_specification(mode)
        features_dtype = utils.make_placeholders(
            feature_spec.condition.features, batch_size=-1)
        labels_dtype = utils.make_placeholders(
            feature_spec.condition.labels, batch_size=-1)
        # We need to infer the output dtypes.
        features_condition = utils.flatten_spec_structure(features_dtype)
        labels_condition = utils.flatten_spec_structure(labels_dtype)
        infered_outputs = self._base_model.inference_network_fn(
            features=features_condition,
            labels=labels_condition,
            mode=mode,
            params=params)
        dtypes = {}
        for key, value in infered_outputs.items():
          dtypes[key] = value.dtype
        return dtypes

  def _map_task_learn(self, task_learn_fn, elems, mode, params=None):
    """Maps a task learning/adaptation function across a batch of tasks.

    Args:
      task_learn_fn: A callable which will be mapped across `elems` once `elems`
        is unpacked along the first (task) dimension.
      elems: A (possibly) nested structure of tensors. Its tensors will be
        unpacked along the first dimension before passing into task_learn_fn.
      mode: (ModeKeys) Specifies if this is training, evaluation, or prediction.
      params: An optional dict of hyper parameters.

    Returns:
      The stacked outputs of task_learn_fn after being mapped across `elems`.
    """

    if self._use_parallel_for:
      return pfor_map_fn(task_learn_fn, elems)
    else:
      if params is None:
        params = {}
      params['is_inner_loop'] = True
      dtypes = self.infer_base_model_output_dtypes(mode, params)

      # We flatten the features to infer the batch_size for parallel iterations.
      # The flattened TensorSpecStruct enables us to get the
      # first element of condition without knowning the name.
      parallel_iterations = list(utils.flatten_spec_structure(
          elems).values())[0].get_shape().as_list()[0]
      # Use parallel execution per batch, if we don't know the batch_size we
      # use the standard.
      if parallel_iterations is None:
        parallel_iterations = 10

      # The output for val, the inner loop training steps, and training loss.
      dtype = ([dtypes] * 2, [dtypes] * (self._num_inner_loop_steps + 1),
               [tf.float32] * (self._num_inner_loop_steps + 1))

      return tf.map_fn(
          task_learn_fn,
          elems=elems,
          dtype=dtype,
          parallel_iterations=parallel_iterations)

  def inference_network_fn(self,
                           features,
                           labels,
                           mode,
                           config=None,
                           params=None):
    """The inference network implementation.

    Args:
      features: This is the first item returned from the input_fn and parsed
        by tensorspec_utils.validate_and_pack. A spec_structure which fulfills
        the requirements of the self.get_feature_specification.
      labels: This is the second item returned from the input_fn and parsed
        by tensorspec_utils.validate_and_pack. A spec_structure which fulfills
        the requirements of the self.get_feature_specification.
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.
      config: (Optional tf.estimator.RunConfig or contrib_tpu.RunConfig)
        Will receive what is passed to Estimator in config parameter, or the
        default config (tf.estimator.RunConfig). Allows updating things in your
        model_fn based on  configuration such as num_ps_replicas, or model_dir.
      params: An optional dict of hyper parameters that will be passed into
        input_fn and model_fn. Keys are names of parameters, values are basic
        python types. There are reserved keys for TPUEstimator,
        including 'batch_size'.
    Returns:
      predictions: A dict with output tensors.
    """
    del config

    maml_inner_loop_instance = maml_inner_loop.MAMLInnerLoopGradientDescent()

    def task_learn(inputs_list):
      """Meta-learning for an individual task, for use with map_fn.

      Args:
          inputs_list: A list of [(condition_features,  condition_labels), ...,
            (inference_features, inference_labels)] individual tasks.

      Returns:
          condition_output: Output of model on conditioning data, before weight
            update.
          inference_output: Output of model on evaluation data, after weight
            update.
      """
      # Disable a_func's summary creation in the inner loop of MAML, since
      # summaries are not supported inside while_loop.
      inner_loop_params = copy.deepcopy(params)
      if inner_loop_params is None:
        inner_loop_params = {}
      inner_loop_params['use_summaries'] = False
      inner_loop_params['maml_inner_loop'] = True

      inference_output, condition_outputs, inner_loss = (
          maml_inner_loop_instance.inner_loop(
              inputs_list=inputs_list,
              inference_network_fn=self._base_model.inference_network_fn,
              model_train_fn=self._base_model.model_train_fn,
              mode=mode,
              params=inner_loop_params))
      return inference_output, condition_outputs, inner_loss

    # Since we need the same format for the mapping function to be fed in any
    # circumstance we overwrite the unused_inference_labels which as the name
    # suggests are not used during the inner loop.
    unused_inference_labels = labels
    if labels is None:
      unused_inference_labels = features.condition.labels
    elems = ((features.condition.features,
              features.condition.labels),) * self._num_inner_loop_steps + (
                  (features.inference.features, unused_inference_labels),)

    # inference output refers to the output we typically optimize with MAML.
    # condition_output refers to the inner loop outputs which could also be
    # further optimized, but in standard MAML are assumed to be simple
    # gradient descent steps of some form. This does NOT play well with batch
    # norm currently due to use of while_loop.
    inference_output, condition_output, inner_loss = self._map_task_learn(
        task_learn, elems, mode, params)

    if self.use_summaries(params):
      maml_inner_loop_instance.add_parameter_summaries()
      for index, inner_loss_step in enumerate(inner_loss):
        tf.summary.scalar('inner_loss_{}'.format(index),
                          tf.reduce_mean(inner_loss_step))

    # Note, this is the first iteration output and loss, prior to any
    # adaptation. In total we have num_inner_loop_steps + 1 since we do one more
    # forward pass for which we do not compute and apply gradients. This step is
    # to monitor the effect of the inner loop.
    base_condition_output = condition_output[0]
    unconditioned_inference_output = inference_output[0]
    conditioned_inference_output = inference_output[1]

    predictions = utils.TensorSpecStruct()

    # We keep the full outputs such that we can simply call the
    # model_condition_fn of the base model.
    predictions.full_condition_output = (
        utils.TensorSpecStruct(list(base_condition_output.items())))

    for pos, base_condition_output in enumerate(condition_output):
      predictions['full_condition_outputs/output_{}'.format(pos)] = (
          utils.TensorSpecStruct(list(base_condition_output.items())))

    predictions.full_inference_output_unconditioned = (
        utils.TensorSpecStruct(list(unconditioned_inference_output.items())))
    predictions.full_inference_output = (
        utils.TensorSpecStruct(list(conditioned_inference_output.items())))
    if self.use_summaries(params):
      for key, inference in predictions.items():
        tf.summary.histogram(key, inference)
      for key in unconditioned_inference_output.keys():
        delta = (conditioned_inference_output[key] -
                 unconditioned_inference_output[key])
        tf.summary.histogram('delta/{}'.format(key), delta)

    predictions = self._select_inference_output(predictions)
    if 'condition_output' not in predictions:
      raise ValueError(
          'The required condition_output is not in predictions {}.'.format(
              list(predictions.keys())))
    if 'inference_output' not in predictions:
      raise ValueError(
          'The required inference_output is not in predictions {}.'.format(
              list(predictions.keys())))
    return predictions

  @abc.abstractmethod
  def _select_inference_output(self,
                               predictions
                              ):
    """Select and assign the condition_output and inference_output.

    Args:
      predictions: The predictions created by the inference models, containing
        the full outputs of condition and inference under
        full_condition_output/* and full_inference_output/*.

    Returns:
      predictions: The same data structure with two additional {key: Tensors},
        condition_output and inference_output used within the meta policies.
    """
    return predictions

  def create_train_op(self,
                      loss,
                      optimizer,
                      update_ops=None,
                      train_outputs=None):
    """Create meta-training op.

    MAMLModel has a configurable var_scope used to select which variables to
    train on. Note that MAMLInnerLoopGradientDescent also has such a parameter
    to decide which variables to update in the *inner* loop. If you don't want
    to update a set of variables in both the inner and outer loop, you'll need
    to configure var_scope for both MAMLModel *and*
    MAMLInnerLoopGradientDescent.

    Args:
      loss: The loss we compute within model_train_fn.
      optimizer: An instance of `tf.train.Optimizer`.
      update_ops: List of update ops to execute alongside the training op.
      train_outputs: (Optional) A dict with additional tensors the training
        model generates.

    Returns:
      train_op: Op for the training step.
    """
    vars_to_train = tf.trainable_variables()
    if self._var_scope is not None:
      vars_to_train = [
          v for v in vars_to_train if v.op.name.startswith(self._var_scope)]
    summarize_gradients = self._summarize_gradients
    if self.is_device_tpu:
      # TPUs don't support summaries up until now. Hence, we overwrite the user
      # provided summarize_gradients option to False.
      if self._summarize_gradients:
        logging.info('We cannot use summarize_gradients on TPUs.')
      summarize_gradients = False
    return contrib_training.create_train_op(
        loss,
        optimizer,
        variables_to_train=vars_to_train,
        summarize_gradients=summarize_gradients,
        update_ops=update_ops)

  def model_train_fn(self,
                     features,
                     labels,
                     inference_outputs,
                     mode,
                     config=None,
                     params=None):
    """The training model implementation.

    Args:
      features: This is the first item returned from the input_fn and parsed
        by tensorspec_utils.validate_and_pack. A spec_structure which fulfills
        the requirements of the self.get_feature_specification.
      labels: This is the second item returned from the input_fn and parsed
        by tensorspec_utils.validate_and_pack. A spec_structure which fulfills
        the requirements of the self.get_feature_specification.
      inference_outputs: A dict containing the output tensors of
        model_inference_fn.
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.
      config: (Optional tf.estimator.RunConfig or contrib_tpu.RunConfig)
        Will receive what is passed to Estimator in config parameter, or the
        default config (tf.estimator.RunConfig). Allows updating things in your
        model_fn based on  configuration such as num_ps_replicas, or model_dir.
      params: An optional dict of hyper parameters that will be passed into
        input_fn and model_fn. Keys are names of parameters, values are basic
        python types. There are reserved keys for TPUEstimator,
        including 'batch_size'.
    Returns:
      The loss and optionally train_outputs of the base model.
    """
    # Since the base model assumes data in the format [batch_size] + shape
    # and not [num_tasks, num_samples_per_task, shape] we need to flatten
    # the data prior to calling the model_train_fn.
    condition_features_flat_batch = meta_tfdata.flatten_batch_examples(
        features.condition.features)
    condition_labels_flat_batch = meta_tfdata.flatten_batch_examples(
        features.condition.labels)
    for inner_loop_step in range(self._num_inner_loop_steps + 1):
      condition_output_flat_batch = meta_tfdata.flatten_batch_examples(
          inference_outputs['full_condition_outputs/output_{}'.format(
              inner_loop_step)])
      with tf.variable_scope('inner_loop_step_{}'.format(inner_loop_step)):
        self._base_model.add_summaries(
            features=condition_features_flat_batch,
            labels=condition_labels_flat_batch,
            inference_outputs=condition_output_flat_batch,
            train_loss=None,
            train_outputs=None,
            mode=mode,
            params=params)

    # Since the base model assumes data in the format [batch_size] + shape
    # and not [num_tasks, num_samples_per_task, shape] we need to flatten
    # the data prior to calling the model_train_fn.
    inference_output_flat_batch = meta_tfdata.flatten_batch_examples(
        inference_outputs.full_inference_output)
    inference_features_flat_batch = meta_tfdata.flatten_batch_examples(
        features.inference.features)
    labels_flat_batch = meta_tfdata.flatten_batch_examples(labels)

    with tf.variable_scope('unconditioned_inference'):
      uncondition_output_flat_batch = meta_tfdata.flatten_batch_examples(
          inference_outputs.full_inference_output_unconditioned)
      self._base_model.add_summaries(
          features=condition_features_flat_batch,
          labels=condition_labels_flat_batch,
          inference_outputs=uncondition_output_flat_batch,
          train_loss=None,
          train_outputs=None,
          mode=mode,
          params=params)

    if params is None:
      params = {}
    params['is_outer_loss'] = True
    return self._base_model.model_train_fn(
        features=inference_features_flat_batch,
        labels=labels_flat_batch,
        inference_outputs=inference_output_flat_batch,
        config=config,
        mode=mode,
        params=params)

  def model_eval_fn(self,
                    features,
                    labels,
                    inference_outputs,
                    train_loss,
                    train_outputs,
                    mode,
                    config=None,
                    params=None):
    """The eval model implementation.

    Args:
      features: This is the first item returned from the input_fn and parsed
        by tensorspec_utils.validate_and_pack. A spec_structure which fulfills
        the requirements of the self.get_feature_specification.
      labels: This is the second item returned from the input_fn and parsed
        by tensorspec_utils.validate_and_pack. A spec_structure which fulfills
        the requirements of the self.get_feature_specification.
      inference_outputs: A dict containing the output tensors of
        model_inference_fn.
      train_loss: The final loss from model_train_fn.
      train_outputs: A dict containing the output tensors (dict) of
        model_train_fn.
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.
      config: (Optional tf.estimator.RunConfig or contrib_tpu.RunConfig)
        Will receive what is passed to Estimator in config parameter, or the
        default config (tf.estimator.RunConfig). Allows updating things in your
        model_fn based on  configuration such as num_ps_replicas, or model_dir.
      params: An optional dict of hyper parameters that will be passed into
        input_fn and model_fn. Keys are names of parameters, values are basic
        python types. There are reserved keys for TPUEstimator,
        including 'batch_size'.
    Returns:
      The eval_metrics determined by the base_model.
    """
    inference_output_flat_batch = meta_tfdata.flatten_batch_examples(
        inference_outputs.full_inference_output)
    inference_features_flat_batch = meta_tfdata.flatten_batch_examples(
        features.inference.features)
    labels_flat_batch = meta_tfdata.flatten_batch_examples(labels)
    return self._base_model.model_eval_fn(
        features=inference_features_flat_batch,
        labels=labels_flat_batch,
        inference_outputs=inference_output_flat_batch,
        train_loss=train_loss,
        train_outputs=train_outputs,
        mode=mode,
        config=config,
        params=params)

  def maybe_init_from_checkpoint(self):
    self._base_model.maybe_init_from_checkpoint()
