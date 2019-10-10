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
"""t2r_model abstract subclasses."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
from typing import Optional, Text

from absl import flags
import gin
from gin.tf import utils as gin_utils
from tensor2robot.models import abstract_model
from tensor2robot.models import model_interface
from tensor2robot.preprocessors import abstract_preprocessor
from tensor2robot.preprocessors import tpu_preprocessor_wrapper
from tensor2robot.utils import tensorspec_utils
import tensorflow as tf

FLAGS = flags.FLAGS
TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT

RunConfigType = abstract_model.RunConfigType
ParamsType = abstract_model.ParamsType
DictOrSpec = abstract_model.DictOrSpec
ModelTrainOutputType = abstract_model.ModelTrainOutputType
ExportOutputType = abstract_model.ExportOutputType


@gin.configurable(whitelist=['disable_for_cpu_debugging'])
def get_cross_shard_optimizer(optimizer, disable_for_cpu_debugging=False):
  if disable_for_cpu_debugging:
    return optimizer
  return tf.contrib.tpu.CrossShardOptimizer(optimizer)


@gin.configurable
class TPUT2RModelWrapper(model_interface.ModelInterface):
  """A thin t2r_model wrapper optimizing the model specifications for TPUs.

  TPUs operate most efficient using the tf.DType bfloat16. This will both
  require less memory on device and less bandwidth to feed the TPUs with data.
  Preprocessing however is done on CPUs and therefore has to operate on float32
  and not bfloat16. Therefore, this wrapper will convert all float32 inputs
  to the model to bfloat16 and ensures that the saved models will be operational
  on CPU/GPU/TPU.
  """

  def __init__(self, t2r_model):
    if not t2r_model.is_device_tpu:
      raise ValueError('The TPUT2RModelWrapper only works with models '
                       'operating on TPU devices.')
    # Note, the model has to be set prior to calling the super constructor since
    # properties are used within the constructor call.
    self._t2r_model = t2r_model
    super(TPUT2RModelWrapper, self).__init__()

  @property
  def t2r_model(self):
    return self._t2r_model

  def get_run_config(self):
    """Get the RunConfig for Estimator model."""
    return self._t2r_model.get_run_config()

  def get_tpu_run_config(self):
    """Get the TPU RunConfig for Estimator model."""
    return self._t2r_model.get_tpu_run_config()

  def get_session_config(self):
    return self._t2r_model.get_session_config()

  def is_device_tpu(self):
    """Returns True if the device is TPU otherwise False."""
    return self._t2r_model.is_device_tpu

  def is_device_gpu(self):
    """Returns True if the device is GPU otherwise False."""
    return self._t2r_model.is_device_gpu

  def device_type(self):
    """Returns the device type string."""
    return self._t2r_model.device_type

  def get_feature_specification(
      self, mode):
    """Returns the feature specification with bfloat16 replacing float32."""
    return tensorspec_utils.replace_dtype(
        self._t2r_model.get_feature_specification(mode),
        from_dtype=tf.float32,
        to_dtype=tf.bfloat16)

  def get_label_specification(self, mode):
    """Returns the label specification with bfloat16 replacing float32."""
    return tensorspec_utils.replace_dtype(
        self._t2r_model.get_label_specification(mode),
        from_dtype=tf.float32,
        to_dtype=tf.bfloat16)

  @property
  def preprocessor(self):
    return tpu_preprocessor_wrapper.TPUPreprocessorWrapper(
        preprocessor=self._t2r_model.preprocessor)

  def model_fn(self,
               features,
               labels,
               mode,
               config = None,
               params = None):
    """Estimator model_fn.

    Note, this function overwrites the model_fn of the wrapped t2r_model since
    is replaces specifications with their TPU corresponding calls and introduces
    additional casting conversion after the specification has been verified.

    Args:
      features: This is the first item returned from the input_fn and parsed by
        tensorspec_utils.validate_and_pack. A spec_structure which fulfills the
        requirements of the self.get_feature_specification.
      labels: This is the second item returned from the input_fn and parsed by
        tensorspec_utils.validate_and_pack. A spec_structure which fulfills the
        requirements of the self.get_feature_specification.
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.
      config: (Optional tf.estimator.RunConfig or tf.contrib.tpu.RunConfig) Will
        receive what is passed to Estimator in config parameter, or the default
        config (tf.estimator.RunConfig). Allows updating things in your model_fn
        based on  configuration such as num_ps_replicas, or model_dir.
      params: An optional dict of hyper parameters that will be passed into
        input_fn and model_fn. Keys are names of parameters, values are basic
        python types. There are reserved keys for TPUEstimator, including
        'batch_size'.

    Raises:
      ValueError: If the mode key is not supported, not in [PREDICT, TRAIN,
        EVAL].

    Returns:
      A TPUEstimatorSpec.
    """

    features = tensorspec_utils.validate_and_pack(
        expected_spec=self.get_feature_specification(mode),
        actual_tensors_or_spec=features,
        ignore_batch=True)
    if labels:
      labels = tensorspec_utils.validate_and_pack(
          expected_spec=self.get_label_specification(mode),
          actual_tensors_or_spec=labels,
          ignore_batch=True)

    # In order to support both TPU and CPU for inference, tensors
    # with dtype=bfloat16 will be casted to float32.
    # Note, despite casting the benefit of bfloat16 are still maintained
    # for TPUs since this operation is a noop on this platform.
    # See http://shortn/_TTg3ZyATRo for rationale.
    features = tensorspec_utils.cast_bfloat16_to_float32(features)
    if labels is not None:
      labels = tensorspec_utils.cast_bfloat16_to_float32(labels)

    inference_outputs = self._t2r_model.inference_network_fn(
        features, labels, mode, config, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
      model_fn_results = self._t2r_model.create_export_outputs_fn(
          features, inference_outputs, mode, config, params)
      export_outputs = None
      if isinstance(model_fn_results, tuple):
        predictions = model_fn_results[0]
        export_outputs = model_fn_results[1]
      elif isinstance(model_fn_results, dict):
        export_outputs = {}
        if len(model_fn_results) == 1:
          name, output = list(model_fn_results.items())[0]
          export_outputs[name] = tf.estimator.export.RegressionOutput(output)
        export_outputs[tf.saved_model.signature_constants
                       .DEFAULT_SERVING_SIGNATURE_DEF_KEY] = (
                           tf.estimator.export.PredictOutput(model_fn_results))
        predictions = model_fn_results
      else:
        raise ValueError('The create_export_outputs_fn should return a '
                         'tuple(predictions, export_outputs) or predictions.')

      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, export_outputs=export_outputs)

    train_fn_result = self._t2r_model.model_train_fn(features, labels,
                                                     inference_outputs, mode,
                                                     config, params)
    if isinstance(train_fn_result, tf.Tensor):
      train_loss = train_fn_result
      train_outputs = {}
    elif isinstance(train_fn_result, tuple):
      train_loss = train_fn_result[0]
      train_outputs = train_fn_result[1]
    else:
      raise ValueError('The model_train_fn should return a '
                       'tuple(loss, train_outputs) or loss.')

    if mode == tf.estimator.ModeKeys.TRAIN:
      # Create the tf.train.Optimizer.
      optimizer = get_cross_shard_optimizer(self._t2r_model.create_optimizer())

      # Required for batch norm usage.
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = self._t2r_model.create_train_op(train_loss, optimizer)

      self._t2r_model.add_summaries(features, labels, inference_outputs,
                                    train_loss, train_outputs, mode, config,
                                    params)

      # For TPUs the init has to happen in a scaffold function. Since the model
      # already contains one implementation which is internal to the model
      # this call is simply wrapped.
      # No new variables are allowed to be added, otherwise
      # we would not initialize these variables.
      # Note, this feature is only available for train to bootstrap a model
      # (partially) from a different model. As soon as this checkpoint is
      # written all other modes will use the local checkpoint within
      # model_dir.

      def create_scaffold_fn():
        """Creates a scaffold instance."""
        self._t2r_model.maybe_init_from_checkpoint()
        # Return the value of the property first since it might be changed.
        scaffold_fn = self._t2r_model.scaffold_fn
        scaffold = scaffold_fn()
        # In order to export asynchronously the saver has to be registered
        # in the graph collection. The scaffold function might register a
        # saver already which is why it is checked here and a saver only
        # added it has none has been added.
        if not tf.get_collection(tf.GraphKeys.SAVERS):
          # TODO(T2R_CONTRIBUTORS): Switch to using gin config for all saver params.
          keep_checkpoint_every_n_hours = None
          max_to_keep = None
          if config is not None:
            keep_checkpoint_every_n_hours = config.keep_checkpoint_every_n_hours
            max_to_keep = config.keep_checkpoint_max
          saver = abstract_model.gin_configurable_saver(
              keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
              max_to_keep=max_to_keep,
          )
          tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        return scaffold

      training_hooks = []

      # EstimatorSpec has training_chief_hooks, but TPUEstimatorSpec does not,
      # so we have to use training_hooks here and check is_chief.
      if config and config.is_chief:  # pytype: disable=attribute-error
        training_hooks.append(
            gin_utils.GinConfigSaverHook(
                config.model_dir, summarize_config=True))

      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=train_loss,
          train_op=train_op,
          training_hooks=training_hooks,
          scaffold_fn=create_scaffold_fn)

    if mode == tf.estimator.ModeKeys.EVAL:
      self._t2r_model.add_summaries(features, labels, inference_outputs,
                                    train_loss, train_outputs, mode, config,
                                    params)
      eval_metrics = self._t2r_model.model_eval_fn(features, labels,
                                                   inference_outputs,
                                                   train_loss, train_outputs,
                                                   mode, config, params)
      evaluation_hooks = self._t2r_model.get_eval_hooks(config, params)
      if config and config.is_chief:  # pytype: disable=attribute-error
        eval_name = params.get('eval_name', 'eval')  # pytype: disable=attribute-error
        evaluation_hooks.append(
            gin_utils.GinConfigSaverHook(
                os.path.join(config.model_dir, eval_name),
                summarize_config=True))

      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=train_loss,
          eval_metrics=eval_metrics,
          evaluation_hooks=evaluation_hooks)

    raise ValueError('The mode {} is not supported yet.'.format(mode))

  def get_feature_specification_for_packing(self, mode):
    return self._t2r_model.preprocessor.get_in_feature_specification(mode)

  def get_label_specification_for_packing(self, mode):
    return self._t2r_model.preprocessor.get_in_label_specification(mode)
