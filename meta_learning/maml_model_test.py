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

# Lint as: python2, python3
"""Tests for robotics.learning.estimator_models.meta_learning.maml_model."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy
import functools
import os
from typing import Optional, Dict, Any, Text
from absl import flags

from absl.testing import parameterized
import gin
from tensor2robot.meta_learning import maml_model
from tensor2robot.models import abstract_model
from tensor2robot.preprocessors import noop_preprocessor
from tensor2robot.utils import mocks
from tensor2robot.utils import tensorspec_utils
from tensor2robot.utils import tfdata
from tensor2robot.utils import train_eval
import tensorflow.compat.v1 as tf
from tensorflow.contrib import predictor as contrib_predictor
FLAGS = flags.FLAGS

_NUM_CONDITION_SAMPLES_PER_TASK = 2
_BATCH_SIZE = 5
_MAX_STEPS = 10


class MockMetaInputGenerator(mocks.MockInputGenerator):
  """A simple mock input generator for meta learning."""

  def __init__(self, num_condition_samples_per_task,
               num_inference_samples_per_task, **parent_kwargs):
    super(MockMetaInputGenerator, self).__init__(**parent_kwargs)
    self._num_condition_samples_per_task = num_condition_samples_per_task
    self._num_inference_samples_per_task = num_inference_samples_per_task

  def set_specification_from_model(self,
                                   t2r_model,
                                   mode):
    """See base class documentation."""
    super(MockMetaInputGenerator,
          self).set_specification_from_model(t2r_model, mode)
    self._meta_map_fn = t2r_model.preprocessor.create_meta_map_fn(
        self._num_condition_samples_per_task,
        self._num_inference_samples_per_task)

  def _create_dataset(self, mode, params=None):
    """See base class documentation."""
    del params

    features, labels = self.create_numpy_data()

    tf_features = {
        'x': tf.constant(features, tf.float32),
    }

    tf_labels = {'y': tf.constant(labels, dtype=tf.float32)}
    dataset = tf.data.Dataset.from_tensor_slices((tf_features, tf_labels))
    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.repeat()
      dataset = dataset.shuffle(buffer_size=features.shape[0])

    batch_size = (
        self._num_condition_samples_per_task +
        self._num_inference_samples_per_task)
    # We batch for the meta_map_fn, this is necessary otherwise we get index
    # errors.
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    # We create the meta specs according to the model.
    dataset = dataset.map(map_func=self._meta_map_fn, num_parallel_calls=1)

    # Note, if drop_remainder = False, the resulting dataset has no static
    # shape which is required for the meta preprocessing.
    dataset = dataset.batch(self._batch_size, drop_remainder=True)
    preprocess_fn = functools.partial(
        self._preprocess_fn, mode=tf.estimator.ModeKeys.TRAIN)
    return dataset.map(map_func=preprocess_fn, num_parallel_calls=1)


class MockMetaExportGenerator(mocks.MockExportGenerator):
  """A simple mock input generator for meta learning."""

  def __init__(self, num_condition_samples_per_task,
               num_inference_samples_per_task, **parent_kwargs):
    super(MockMetaExportGenerator, self).__init__(**parent_kwargs)
    self._num_condition_samples_per_task = num_condition_samples_per_task
    self._num_inference_samples_per_task = num_inference_samples_per_task

  def set_specification_from_model(self,
                                   t2r_model):
    """See base class documentation."""
    super(MockMetaExportGenerator, self).set_specification_from_model(t2r_model)
    self._base_feature_spec = (
        t2r_model.preprocessor.base_preprocessor.get_in_feature_specification(
            tf.estimator.ModeKeys.PREDICT))
    tensorspec_utils.assert_valid_spec_structure(self._base_feature_spec)
    self._base_label_spec = (
        t2r_model.preprocessor.base_preprocessor.get_in_label_specification(
            tf.estimator.ModeKeys.PREDICT))
    tensorspec_utils.assert_valid_spec_structure(self._base_label_spec)

  def create_serving_input_receiver_numpy_fn(self, params=None):
    """Create a serving input receiver for numpy.

    Args:
      params: An optional dict of hyper parameters that will be passed into
        input_fn and model_fn. Keys are names of parameters, values are basic
        python types. There are reserved keys for TPUEstimator, including
        'batch_size'.

    Returns:
      serving_input_receiver_fn: A callable which creates the serving inputs.
    """
    del params

    def serving_input_receiver_fn():
      """Create the ServingInputReceiver to export a saved model.

      Returns:
        An instance of ServingInputReceiver.
      """
      # We have to filter our specs since only required tensors are
      # used for inference time.
      flat_feature_spec = tensorspec_utils.flatten_spec_structure(
          self._feature_spec)
      # We need to freeze the conditioning and inference shapes.
      for key, value in flat_feature_spec.condition.items():
        ref_shape = value.shape.as_list()
        shape = [self._num_condition_samples_per_task] + ref_shape[1:]
        flat_feature_spec.condition[key] = (
            tensorspec_utils.ExtendedTensorSpec.from_spec(value, shape=shape))

      for key, value in flat_feature_spec.inference.items():
        ref_shape = value.shape.as_list()
        shape = [self._num_inference_samples_per_task] + ref_shape[1:]
        flat_feature_spec.inference[key] = (
            tensorspec_utils.ExtendedTensorSpec.from_spec(value, shape=shape))

      required_feature_spec = (
          tensorspec_utils.filter_required_flat_tensor_spec(flat_feature_spec))
      receiver_tensors = tensorspec_utils.make_placeholders(
          required_feature_spec)

      # We want to ensure that our feature processing pipeline operates on a
      # copy of the features and does not alter the receiver_tensors.
      features = tensorspec_utils.flatten_spec_structure(
          copy.copy(receiver_tensors))

      if self._preprocess_fn is not None:
        features, _ = self._preprocess_fn(
            features=features, labels=None, mode=tf.estimator.ModeKeys.PREDICT)

      return tf.estimator.export.ServingInputReceiver(features,
                                                      receiver_tensors)

    return serving_input_receiver_fn

  def create_serving_input_receiver_tf_example_fn(self,
                                                  params = None):
    """Create a serving input receiver for tf_examples.

    Args:
      params: An optional dict of hyper parameters that will be passed into
        input_fn and model_fn. Keys are names of parameters, values are basic
        python types. There are reserved keys for TPUEstimator, including
        'batch_size'.

    Returns:
      serving_input_receiver_fn: A callable which creates the serving inputs.
    """
    del params

    def serving_input_receiver_fn():
      """Create the ServingInputReceiver to export a saved model.

      Returns:
        An instance of ServingInputReceiver.
      """
      # We only assume one input, a string which containes the serialized proto.
      receiver_tensors = {
          'input_example_tensor':
              tf.placeholder(
                  dtype=tf.string, shape=[None], name='input_example_tensor')
      }
      # We have to filter our specs since only required tensors are
      # used for inference time.
      flat_feature_spec = tensorspec_utils.flatten_spec_structure(
          self._feature_spec)

      # We need to freeze the conditioning and inference shapes.
      for key, value in flat_feature_spec.condition.items():
        ref_shape = value.shape.as_list()
        shape = [self._num_condition_samples_per_task] + ref_shape[1:]
        flat_feature_spec.condition[key] = (
            tensorspec_utils.ExtendedTensorSpec.from_spec(value, shape=shape))

      for key, value in flat_feature_spec.inference.items():
        ref_shape = value.shape.as_list()
        shape = [self._num_inference_samples_per_task] + ref_shape[1:]
        flat_feature_spec.inference[key] = (
            tensorspec_utils.ExtendedTensorSpec.from_spec(value, shape=shape))

      required_feature_spec = (
          tensorspec_utils.filter_required_flat_tensor_spec(flat_feature_spec))

      parse_tf_example_fn = tfdata.create_parse_tf_example_fn(
          feature_tspec=required_feature_spec)

      features = parse_tf_example_fn(receiver_tensors['input_example_tensor'])

      if self._preprocess_fn is not None:
        features, _ = self._preprocess_fn(
            features=features, labels=None, mode=tf.estimator.ModeKeys.PREDICT)

      return tf.estimator.export.ServingInputReceiver(features,
                                                      receiver_tensors)

    return serving_input_receiver_fn


class MockMAMLModel(maml_model.MAMLModel):

  def _select_inference_output(self, predictions):
    # We select our output for inference.
    predictions.condition_output = predictions.full_condition_output.logit
    predictions.inference_output = predictions.full_inference_output.logit
    return predictions


class MAMLModelTest(parameterized.TestCase):

  @parameterized.parameters([0, 1, 2, 3])
  def test_maml_model(self, num_inner_loop_steps):
    model_dir = os.path.join(FLAGS.test_tmpdir, str(num_inner_loop_steps))
    gin.bind_parameter('tf.estimator.RunConfig.save_checkpoints_steps',
                       _MAX_STEPS // 2)
    if tf.io.gfile.exists(model_dir):
      tf.io.gfile.rmtree(model_dir)

    mock_base_model = mocks.MockT2RModel(
        preprocessor_cls=noop_preprocessor.NoOpPreprocessor)

    mock_tf_model = MockMAMLModel(
        base_model=mock_base_model, num_inner_loop_steps=num_inner_loop_steps)

    # Note, we by choice use the same amount of conditioning samples for
    # inference as well during train and change the model for eval/inference
    # to only produce one output sample.
    mock_input_generator_train = MockMetaInputGenerator(
        batch_size=_BATCH_SIZE,
        num_condition_samples_per_task=_NUM_CONDITION_SAMPLES_PER_TASK,
        num_inference_samples_per_task=_NUM_CONDITION_SAMPLES_PER_TASK)
    mock_input_generator_train.set_specification_from_model(
        mock_tf_model, mode=tf.estimator.ModeKeys.TRAIN)

    mock_input_generator_eval = MockMetaInputGenerator(
        batch_size=_BATCH_SIZE,
        num_condition_samples_per_task=_NUM_CONDITION_SAMPLES_PER_TASK,
        num_inference_samples_per_task=1)
    mock_input_generator_eval.set_specification_from_model(
        mock_tf_model, mode=tf.estimator.ModeKeys.TRAIN)
    mock_export_generator = MockMetaExportGenerator(
        num_condition_samples_per_task=_NUM_CONDITION_SAMPLES_PER_TASK,
        num_inference_samples_per_task=1)

    train_eval.train_eval_model(
        t2r_model=mock_tf_model,
        input_generator_train=mock_input_generator_train,
        input_generator_eval=mock_input_generator_eval,
        max_train_steps=_MAX_STEPS,
        model_dir=model_dir,
        export_generator=mock_export_generator,
        create_exporters_fn=train_eval.create_default_exporters)
    export_dir = os.path.join(model_dir, 'export')
    # best_exporter_numpy, best_exporter_tf_example.
    self.assertLen(tf.io.gfile.glob(os.path.join(export_dir, '*')), 4)
    numpy_predictor_fn = contrib_predictor.from_saved_model(
        tf.io.gfile.glob(os.path.join(export_dir, 'best_exporter_numpy',
                                      '*'))[-1])

    feed_tensor_keys = sorted(numpy_predictor_fn.feed_tensors.keys())
    self.assertCountEqual(
        ['condition/features/x', 'condition/labels/y', 'inference/features/x'],
        feed_tensor_keys,
    )

    tf_example_predictor_fn = contrib_predictor.from_saved_model(
        tf.io.gfile.glob(
            os.path.join(export_dir, 'best_exporter_tf_example', '*'))[-1])
    self.assertCountEqual(['input_example_tensor'],
                          list(tf_example_predictor_fn.feed_tensors.keys()))


if __name__ == '__main__':
  tf.test.main()
