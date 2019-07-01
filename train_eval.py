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

"""Library for offline training TFModels with Estimator API."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import tempfile

from absl import flags
from absl import logging
import gin
from tensor2robot import t2r_pb2
from tensor2robot.export_generators import abstract_export_generator
from tensor2robot.export_generators import default_export_generator
from tensor2robot.hooks import hook_builder
from tensor2robot.input_generators import abstract_input_generator
from tensor2robot.models import model_interface
from tensor2robot.models import tpu_model_wrapper
from tensor2robot.utils import tensorspec_utils
import tensorflow as tf  # tf

from typing import Any, Callable, Dict, List, Optional, Text

EXPORTER_FN = Callable[[
    model_interface.ModelInterface, abstract_export_generator
    .AbstractExportGenerator
], List[tf.estimator.Exporter]]

FLAGS = flags.FLAGS

try:
  flags.DEFINE_list(
      'gin_configs', None, 'A comma-separated list of paths to Gin '
      'configuration files.')
  flags.DEFINE_multi_string(
      'gin_bindings', [], 'A newline separated list of Gin parameter bindings.')
except flags.DuplicateFlagError:
  pass

gin_configurable_eval_spec = gin.external_configurable(
    tf.estimator.EvalSpec, name='tf.estimator.EvalSpec')


def print_spec(tensor_spec):
  """Iterate over a spec and print its values in sorted order.

  Args:
    tensor_spec: A dict, (named)tuple, list or a hierarchy thereof filled by
      TensorSpecs(subclasses) or Tensors.
  """
  for key, value in sorted(
      tensorspec_utils.flatten_spec_structure(tensor_spec).items()):
    logging.info('%s: %s', key, value)


def print_specification(t2r_model):
  """Print the specification for the model and its preprocessor.

  Args:
    t2r_model: A TFModel from which we obtain the preprocessor used to prepare
      the input generator instance for usage.
  """
  for mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.PREDICT]:
    logging.info('Preprocessor in feature specification for mode %s', mode)
    print_spec(t2r_model.preprocessor.get_in_feature_specification(mode))
    logging.info('Preprocessor in label specification.')
    print_spec(t2r_model.preprocessor.get_in_label_specification(mode))

    logging.info('Preprocessor out feature specification.')
    print_spec(t2r_model.preprocessor.get_out_feature_specification(mode))
    logging.info('Preprocessor out label specification.')
    print_spec(t2r_model.preprocessor.get_out_label_specification(mode))

    logging.info('Model feature specification.')
    print_spec(t2r_model.get_feature_specification(mode))
    logging.info('Model label specification.')
    print_spec(t2r_model.get_label_specification(mode))


def provide_input_generator_with_model_information(
    input_generator_instance,
    t2r_model,
    mode,
):
  """Fill the input generator with information provided by a TFModel instance.

  Args:
    input_generator_instance: The input generator instance which will be filled
      with the feature_spec, label_spec and the preprocess_fn of the model.
    t2r_model: A TFModel from which we obtain the preprocessor used to prepare
      the input generator instance for usage.
    mode: The mode used during training.

  Raises:
    ValueError: We raise this error in case the provided
      input_generator_instance is not derived from AbstractInputGenerator.

  Returns:
    input_generator_instance: The prepared instance which was passed in.
      Note it is not a copy but an in-place operation.
  """
  if not isinstance(input_generator_instance,
                    abstract_input_generator.AbstractInputGenerator):
    raise ValueError('The input generator must be a subclass of '
                     'abstract_input_generator.AbstractInputGenerator.')

  input_generator_instance.set_specification_from_model(t2r_model, mode)
  return input_generator_instance


@gin.configurable
def create_tpu_estimator(t2r_model,
                         model_dir,
                         train_batch_size = 32,
                         eval_batch_size = 1,
                         use_tpu_hardware = True,
                         params = None,
                         export_to_cpu = True,
                         export_to_tpu = True,
                         **kwargs):
  """Wrapper for TPUEstimator to provide a common interface for instantiation.

  Args:
    t2r_model: An instance of the model we will train or evaluate.
    model_dir: An optional location where we want to store or load our model
      from.
    train_batch_size: The batch size for training.
    eval_batch_size: The batch size for evaluation.
    use_tpu_hardware: If False, the TPUEstimator is used but executed on CPU.
      This is valuable for debugging, otherwise this parameter can be ignored.
    params: An optional dict of hyper parameters that will be passed into
      input_fn and model_fn. Keys are names of parameters, values are basic
      python types. There are reserved keys for TPUEstimator, including
      'batch_size'.
    export_to_cpu: If True, export a savedmodel to cpu.
    export_to_tpu: If True, export a savedmodel to tpu.
    **kwargs: Keyword arguments are only used to enable the same interface for
      tpu estimator and estimator.

  Returns:
    An instance of tf.contrib.tpu.TPUEstimator.
  """
  del kwargs
  return tf.contrib.tpu.TPUEstimator(
      model_fn=t2r_model.model_fn,
      model_dir=model_dir,
      config=t2r_model.get_tpu_run_config(),
      use_tpu=t2r_model.is_device_tpu and use_tpu_hardware,
      train_batch_size=train_batch_size,
      eval_batch_size=eval_batch_size,
      export_to_cpu=export_to_cpu,
      export_to_tpu=export_to_tpu,
      params=params)


@gin.configurable
def create_estimator(t2r_model,
                     model_dir,
                     params = None,
                     **kwargs):
  """Wrapper for Estimator to provide a common interface for instantiation.

  Args:
    t2r_model: An instance of the model we will train or evaluate.
    model_dir: An optional location where we want to store or load our model
      from.
    params: An optional dict of hyper parameters that will be passed into
      input_fn and model_fn. Keys are names of parameters, values are basic
      python types. There are reserved keys for TPUEstimator,
      including 'batch_size'.
    **kwargs: Keyword arguments are only used to enable the same interface for
      tpu estimator and estimator.

  Returns:
    An instance of tf.estimator.Estimator.
  """
  del kwargs
  return tf.estimator.Estimator(
      model_fn=t2r_model.model_fn,
      model_dir=model_dir,
      config=t2r_model.get_run_config(),
      params=params)


@gin.configurable
def create_valid_result_smaller(result_key = 'loss'):
  """Creates a compare_fn with the correct function signature.

  Args:
    result_key: The key to identify the result within best_eval_result and
      current_eval_result. Possible keys are `loss` or any other key which is
      specified in the models metrics dict.

  Returns:
    A valid_result_smaller, a compare_fn with the only allowed function
    signature.
  """

  def valid_result_smaller(best_eval_result, current_eval_result):
    """Compares two evaluation results.

    This implementation is a robustified version of the default _result_smaller
    implementation which had the requirement that 'result' is available in all
    results which does not necessarily has to hold true in case there are
    multiple summary writers.

    Args:
      best_eval_result: best eval metrics.
      current_eval_result: current eval metrics.

    Returns:
      True if the result of current_eval_result is smaller or best_eval_result
      was
        invalid so far; otherwise, False.
    """
    if not best_eval_result or result_key not in best_eval_result:
      # This effectively means all entries so far did not have the result_key.
      # Therefore, we can simply use the current_eval_result since the
      # the best_eval_result so far is invalid.
      return True

    if not current_eval_result or result_key not in current_eval_result:
      return False

    return best_eval_result[result_key] > current_eval_result[result_key]

  return valid_result_smaller


@gin.configurable
def create_valid_result_larger(result_key = 'loss'):
  """Creates a compare_fn with the correct function signature.

  Args:
    result_key: The key to identify the result within best_eval_result and
      current_eval_result. Possible keys are `loss` or any other key which is
      specified in the models metrics dict.

  Returns:
    A valid_result_larger, a compare_fn with the only allowed function
      signature.
  """

  def valid_result_larger(best_eval_result, current_eval_result):
    """Compares two evaluation results.

    This implementation is a robustified version of the default _result_larger
    implementation which had the requirement that 'result' is available in all
    results which does not necessarily has to hold true in case there are
    multiple summary writers.

    Args:
      best_eval_result: best eval metrics.
      current_eval_result: current eval metrics.

    Returns:
      True if the result of current_eval_result is larger or best_eval_result
        was invalid so far; otherwise, False.
    """
    if not best_eval_result or result_key not in best_eval_result:
      # This effectively means all entries so far did not have the result_key.
      # Therefore, we can simply use the current_eval_result since the
      # the best_eval_result so far is invalid.
      return True

    if not current_eval_result or result_key not in current_eval_result:
      return False

    return best_eval_result[result_key] < current_eval_result[result_key]

  return valid_result_larger


@gin.configurable
def create_default_exporters(
    t2r_model,
    export_generator,
    compare_fn=create_valid_result_smaller):
  """Creates a list of Exporter to export saved models during evaluation.

  Args:
    t2r_model: The model to be exported.
    export_generator: An export_generator.AbstractExportGenerator.
    compare_fn: The function used to deterimne the best model to export.

  Returns:
    A list containing two exporters, one for numpy and another one for
      tf_example interface.
  """
  if export_generator is None:
    export_generator = default_export_generator.DefaultExportGenerator()
  # Create pkl of the input to save alongside the exported models
  tmpdir = tempfile.mkdtemp()
  in_feature_spec = t2r_model.get_feature_specification_for_packing(
      mode=tf.estimator.ModeKeys.PREDICT)
  in_label_spec = t2r_model.get_label_specification_for_packing(
      mode=tf.estimator.ModeKeys.PREDICT)
  t2r_assets = t2r_pb2.T2RAssets()
  t2r_assets.feature_spec.CopyFrom(in_feature_spec.to_proto())
  t2r_assets.label_spec.CopyFrom(in_label_spec.to_proto())
  t2r_assets_filename = os.path.join(tmpdir,
                                     tensorspec_utils.T2R_ASSETS_FILENAME)
  tensorspec_utils.write_t2r_assets_to_file(t2r_assets, t2r_assets_filename)
  assets = {tensorspec_utils.T2R_ASSETS_FILENAME: t2r_assets_filename}
  export_generator.set_specification_from_model(t2r_model)

  exporters = []
  exporters.append(
      tf.estimator.BestExporter(
          name='best_exporter_numpy',
          compare_fn=compare_fn(),
          serving_input_receiver_fn=export_generator
          .create_serving_input_receiver_numpy_fn(),
          assets_extra=assets))
  exporters.append(
      tf.estimator.BestExporter(
          name='best_exporter_tf_example',
          compare_fn=compare_fn(),
          serving_input_receiver_fn=export_generator
          .create_serving_input_receiver_tf_example_fn(),
          assets_extra=assets))
  exporters.append(
      tf.estimator.LatestExporter(
          name='latest_exporter_numpy',
          serving_input_receiver_fn=export_generator
          .create_serving_input_receiver_numpy_fn(),
          assets_extra=assets))
  exporters.append(
      tf.estimator.LatestExporter(
          name='latest_exporter_tf_example',
          serving_input_receiver_fn=export_generator
          .create_serving_input_receiver_tf_example_fn(),
          assets_extra=assets))
  return exporters


@gin.configurable
def predict_from_model(
    t2r_model,
    input_generator_predict,
    model_dir = '/tmp/estimator_models_log_dir/'):
  """Use a TFModel for predictions.

  Args:
    t2r_model: An instance of the model we will use for predictions
    input_generator_predict: The input generator of data to predict.
    model_dir: A path containing checkpoints of the model to load.

  Returns:
    An iterator which yields predictions in the order of
    input_generator_predict.
  """
  input_generator_predict = provide_input_generator_with_model_information(
      input_generator_predict, t2r_model, mode=tf.estimator.ModeKeys.PREDICT)

  input_fn = input_generator_predict.create_dataset_input_fn(
      mode=tf.estimator.ModeKeys.PREDICT)
  create_estimator_fn = create_estimator
  if t2r_model.is_device_tpu:
    create_estimator_fn = create_tpu_estimator

  estimator = create_estimator_fn(t2r_model=t2r_model, model_dir=model_dir)

  return estimator.predict(input_fn)


@gin.configurable
def train_eval_model(
    t2r_model,
    input_generator_train = None,
    input_generator_eval = None,
    max_train_steps = 1000,
    model_dir = '/tmp/estimator_models_log_dir/',
    eval_steps = 100,
    eval_throttle_secs = 600,
    create_exporters_fn = None,
    export_generator = None,
    use_continuous_eval = True,
    train_hook_builders = None,
    chief_train_hook_builders = None,
    eval_hook_builders = None,
):
  """Train and evaluate a T2RModel.

  We will either train, evaluate or train and evaluate a estimator model
  depending on the provided input generators.

  Args:
    t2r_model: An instance of the model we will train or evaluate.
    input_generator_train: An optional instance of an input generator. If
      provided then we will optimize the model until max_train_steps.
    input_generator_eval: An optional instance of an input generator. If
      provided then we will evaluate the model for at most eval_steps.
    max_train_steps: An optional maximum number of steps. For TPU training, it
      is a mandetory flag.
    model_dir: An optional location where we want to store or load our model
      from.
    eval_steps: An optional maximum number of evaluation steps.
    eval_throttle_secs: An optional number of seconds to wait before evaluating
      the next checkpoint.
    create_exporters_fn: An optional function which creates exporters for saved
      models during eval.
    export_generator: An export_generator.AbstractExportGenerator.
    use_continuous_eval: If True the evaluation job waits for new checkpoints
      and continuously evaluates every checkpoint. If False, only the latest
      checkpoint is evaluated or if None exists a model is initialized,
      evaluated and the job exists. Note, this parameter is only used if no
      input generator for training is provided.
    train_hook_builders: A optional list of HookBuilders to build trainer hooks
      to pass to the estimator.
    chief_train_hook_builders: A optional list of HookBuilders to build trainer
      hooks to pass to the estimator, only on the chief.
    eval_hook_builders: A optional list of HookBuilders to build eval hooks to
      pass to the estimator.

  Raises:
    ValueError: If neither a input_generator for train nor eval is available.
  """

  # TODO(T2R_CONTRIBUTORS): Document behavior in T2R README.
  use_tpu_tf_wrapper = t2r_model.is_device_tpu

  if use_tpu_tf_wrapper:
    t2r_model = tpu_model_wrapper.TPUT2RModelWrapper(t2r_model=t2r_model)

  print_specification(t2r_model)

  params = {}
  # Train Input Generator.
  train_batch_size = None
  train_spec = None
  if input_generator_train is not None:
    input_generator_train = provide_input_generator_with_model_information(
        input_generator_train,
        t2r_model,
        mode=tf.estimator.ModeKeys.TRAIN,
    )
    train_batch_size = input_generator_train.batch_size

  # Eval Input Generator.
  eval_batch_size = None
  eval_spec = None
  if input_generator_eval is not None:
    input_generator_eval = provide_input_generator_with_model_information(
        input_generator_eval, t2r_model, mode=tf.estimator.ModeKeys.EVAL)
    eval_batch_size = input_generator_eval.batch_size

  create_estimator_fn = create_estimator
  if t2r_model.is_device_tpu:
    create_estimator_fn = create_tpu_estimator

  estimator = create_estimator_fn(
      t2r_model=t2r_model,
      model_dir=model_dir,
      train_batch_size=train_batch_size,
      eval_batch_size=eval_batch_size,
      params=params)

  if export_generator is None:
    export_generator = default_export_generator.DefaultExportGenerator()

  # Inline helper function for building hooks.
  def _build_hooks(hook_builders):
    hooks = []
    if hook_builders:
      for builder in hook_builders:
        hooks.extend(
            builder.create_hooks(t2r_model, estimator, export_generator))
    return hooks

  # TrainSpec and Hooks.
  if input_generator_train is not None:
    train_hooks = _build_hooks(train_hook_builders)
    if t2r_model.get_run_config().is_chief:
      train_hooks.extend(_build_hooks(chief_train_hook_builders))
    train_spec = tf.estimator.TrainSpec(
        input_fn=input_generator_train.create_dataset_input_fn(
            mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=max_train_steps,
        hooks=train_hooks)

  # EvalSpec, Exporters, and Hooks.
  if input_generator_eval is not None:
    exporters = None
    if create_exporters_fn is not None:
      exporters = create_exporters_fn(t2r_model, export_generator)
    eval_hooks = _build_hooks(eval_hook_builders)
    eval_spec = gin_configurable_eval_spec(
        input_fn=input_generator_eval.create_dataset_input_fn(
            mode=tf.estimator.ModeKeys.EVAL),
        steps=eval_steps,
        throttle_secs=eval_throttle_secs,
        exporters=exporters,
        hooks=eval_hooks)
    # If the eval spec has a name we create the custom output dir such that
    # the metrics coincide with the summaries. Note, this is useful when
    # launching several separate evaluation processes.
    if eval_spec.name is not None:
      params['eval_name'] = 'eval_{}'.format(eval_spec.name)

  logging.info('gin operative configuration:')
  logging.info(gin.operative_config_str())

  if (train_spec is not None and eval_spec is not None):
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  elif train_spec is not None:
    estimator.train(
        input_fn=train_spec.input_fn,
        hooks=train_spec.hooks,
        max_steps=train_spec.max_steps)
  elif eval_spec is not None:
    if not use_continuous_eval:
      estimator.evaluate(
          input_fn=eval_spec.input_fn, steps=eval_steps, name=eval_spec.name)
      return

    # This will start with the latest checkpoint and wait afterwards for a new
    # checkpoint for the next evaluation.
    for checkpoint_path in tf.contrib.training.checkpoints_iterator(
        estimator.model_dir):
      eval_result = estimator.evaluate(
          input_fn=eval_spec.input_fn,
          checkpoint_path=checkpoint_path,
          steps=eval_steps,
          name=eval_spec.name)
      if eval_spec.exporters:
        for exporter in eval_spec.exporters:
          export_path = os.path.join(estimator.model_dir, exporter.name)
          if eval_spec.name is not None:
            export_path = os.path.join(estimator.model_dir, 'eval_{}'.format(
                eval_spec.name), exporter.name)
          exporter.export(
              estimator=estimator,
              export_path=export_path,
              checkpoint_path=checkpoint_path,
              eval_result=eval_result,
              is_the_final_export=True)
  else:
    raise ValueError('Neither train nor eval was provided.')
