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
"""Tests for tensor2robot.train_eval."""

import functools
import os
from absl import flags
import gin
import mock
import numpy as np

from six.moves import zip
from tensor2robot.hooks import hook_builder
from tensor2robot.models import abstract_model
from tensor2robot.preprocessors import noop_preprocessor
from tensor2robot.utils import mocks
from tensor2robot.utils import train_eval
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
from tensorflow.contrib import predictor as contrib_predictor

FLAGS = flags.FLAGS

_MAX_TRAIN_STEPS = 400
_EVAL_STEPS = 40
_BATCH_SIZE = 4
_EVAL_THROTTLE_SECS = 0.0


class FakeHook(tf.train.SessionRunHook):

  def __init__(self):
    self._mock = mock.MagicMock()

  def begin(self):
    self._mock.begin()
    return

  @property
  def mock(self):
    return self._mock


class FakeHookBuilder(hook_builder.HookBuilder):

  def __init__(self):
    self._hook = FakeHook()

  def create_hooks(self, *args, **kwargs):
    del args, kwargs
    return [self._hook]

  @property
  def hook_mock(self):
    return self._hook.mock


class TrainEvalTest(tf.test.TestCase):

  def tearDown(self):
    gin.clear_config()
    super(TrainEvalTest, self).tearDown()

  def _compute_total_loss(self, labels, logits):
    """Summation of the categorical hinge loss for labels and logits."""
    error = 0.
    for label, logit in zip(labels, logits):
      # Reference tensorflow implementation can be found in keras.losses.
      positive = (label * logit)
      negative = ((1 - label) * logit)
      error += np.maximum(0., negative - positive + 1.)
    return error

  def test_train_eval_model(self):
    """Tests that a simple model trains and exported models are valid."""
    gin.bind_parameter('tf.estimator.RunConfig.save_checkpoints_steps', 100)
    model_dir = self.create_tempdir().full_path
    mock_t2r_model = mocks.MockT2RModel(
        preprocessor_cls=noop_preprocessor.NoOpPreprocessor)

    mock_input_generator_train = mocks.MockInputGenerator(
        batch_size=_BATCH_SIZE)
    mock_input_generator_eval = mocks.MockInputGenerator(batch_size=1)
    fake_hook_builder = FakeHookBuilder()

    train_eval.train_eval_model(
        t2r_model=mock_t2r_model,
        input_generator_train=mock_input_generator_train,
        input_generator_eval=mock_input_generator_eval,
        max_train_steps=_MAX_TRAIN_STEPS,
        model_dir=model_dir,
        train_hook_builders=[fake_hook_builder],
        eval_hook_builders=[fake_hook_builder],
        eval_steps=_EVAL_STEPS,
        eval_throttle_secs=_EVAL_THROTTLE_SECS,
        create_exporters_fn=train_eval.create_default_exporters)

    self.assertTrue(fake_hook_builder.hook_mock.begin.called)

    # We ensure that both numpy and tf_example inference models are exported.
    best_exporter_numpy_path = os.path.join(model_dir, 'export',
                                            'best_exporter_numpy', '*')
    numpy_model_paths = sorted(tf.io.gfile.glob(best_exporter_numpy_path))
    # There should be at least 1 exported model.
    self.assertGreater(len(numpy_model_paths), 0)
    # This mock network converges nicely which is why we have several best
    # models, by default we keep the best 5 and the latest one is always the
    # best.
    self.assertLessEqual(len(numpy_model_paths), 5)

    best_exporter_tf_example_path = os.path.join(
        model_dir, 'export', 'best_exporter_tf_example', '*')

    tf_example_model_paths = sorted(
        tf.io.gfile.glob(best_exporter_tf_example_path))
    # There should be at least 1 exported model.
    self.assertGreater(len(tf_example_model_paths), 0)
    # This mock network converges nicely which is why we have several best
    # models, by default we keep the best 5 and the latest one is always the
    # best.
    self.assertLessEqual(len(tf_example_model_paths), 5)

    # We test both saved models within one test since the bulk of the time
    # is spent training the model in the firstplace.

    # Verify that the serving estimator does exactly the same as the normal
    # estimator with all the parameters.
    estimator_predict = tf_estimator.Estimator(
        model_fn=mock_t2r_model.model_fn,
        config=tf_estimator.RunConfig(model_dir=model_dir))

    prediction_ref = estimator_predict.predict(
        input_fn=mock_input_generator_eval.create_dataset_input_fn(
            mode=tf_estimator.ModeKeys.EVAL))

    # Now we can load our exported estimator graph with the numpy feed_dict
    # interface, there are no dependencies on the model_fn or preprocessor
    # anymore.
    # We load the latest model since it had the best eval performance.
    numpy_predictor_fn = contrib_predictor.from_saved_model(
        numpy_model_paths[-1])

    features, labels = mock_input_generator_eval.create_numpy_data()

    ref_error = self._compute_total_loss(
        labels, [val['logit'].flatten() for val in prediction_ref])

    numpy_predictions = []
    for feature, label in zip(features, labels):
      predicted = numpy_predictor_fn({'x': feature.reshape(
          1, -1)})['logit'].flatten()
      numpy_predictions.append(predicted)
      # This ensures that we actually achieve near-perfect classification.
      if label > 0:
        self.assertGreater(predicted[0], 0)
      else:
        self.assertLess(predicted[0], 0)
    numpy_error = self._compute_total_loss(labels, numpy_predictions)

    # Now we can load our exported estimator graph with the tf_example feed_dict
    # interface, there are no dependencies on the model_fn or preprocessor
    # anymore.
    # We load the latest model since it had the best eval performance.
    tf_example_predictor_fn = contrib_predictor.from_saved_model(
        tf_example_model_paths[-1])
    tf_example_predictions = []
    for feature, label in zip(features, labels):
      # We have to create our serialized tf.Example proto.
      example = tf.train.Example()
      example.features.feature['measured_position'].float_list.value.extend(
          feature)
      feed_dict = {
          'input_example_tensor':
              np.array(example.SerializeToString()).reshape(1,)
      }
      predicted = tf_example_predictor_fn(feed_dict)['logit'].flatten()
      tf_example_predictions.append(predicted)
      # This ensures that we actually achieve perfect classification.
      if label > 0:
        self.assertGreater(predicted[0], 0)
      else:
        self.assertLess(predicted[0], 0)
    tf_example_error = self._compute_total_loss(labels, tf_example_predictions)

    np.testing.assert_almost_equal(tf_example_error, numpy_error)
    # The exported saved models both have to have the same performance and since
    # we train on eval on the same fixed dataset the latest and greatest
    # model error should also be the best.
    np.testing.assert_almost_equal(ref_error, tf_example_error, decimal=3)

  def test_init_from_checkpoint_global_step(self):
    """Tests that a simple model trains and exported models are valid."""
    gin.bind_parameter('tf.estimator.RunConfig.save_checkpoints_steps', 100)
    gin.bind_parameter('tf.estimator.RunConfig.keep_checkpoint_max', 3)
    model_dir = self.create_tempdir().full_path
    mock_t2r_model = mocks.MockT2RModel(
        preprocessor_cls=noop_preprocessor.NoOpPreprocessor)

    mock_input_generator_train = mocks.MockInputGenerator(
        batch_size=_BATCH_SIZE)

    train_eval.train_eval_model(
        t2r_model=mock_t2r_model,
        input_generator_train=mock_input_generator_train,
        max_train_steps=_MAX_TRAIN_STEPS,
        model_dir=model_dir,
        eval_steps=_EVAL_STEPS,
        eval_throttle_secs=_EVAL_THROTTLE_SECS,
        create_exporters_fn=train_eval.create_default_exporters)
    # The model trains for 200 steps and saves a checkpoint each 100 steps and
    # keeps 3 -> len == 3.
    self.assertLen(tf.io.gfile.glob(os.path.join(model_dir, 'model*.meta')), 3)

    # The continuous training has its own directory.
    continue_model_dir = self.create_tempdir().full_path
    init_from_checkpoint_fn = functools.partial(
        abstract_model.default_init_from_checkpoint_fn, checkpoint=model_dir)
    continue_mock_t2r_model = mocks.MockT2RModel(
        preprocessor_cls=noop_preprocessor.NoOpPreprocessor,
        init_from_checkpoint_fn=init_from_checkpoint_fn)
    continue_mock_input_generator_train = mocks.MockInputGenerator(
        batch_size=_BATCH_SIZE)
    train_eval.train_eval_model(
        t2r_model=continue_mock_t2r_model,
        input_generator_train=continue_mock_input_generator_train,
        model_dir=continue_model_dir,
        max_train_steps=_MAX_TRAIN_STEPS + 100,
        eval_steps=_EVAL_STEPS,
        eval_throttle_secs=_EVAL_THROTTLE_SECS,
        create_exporters_fn=train_eval.create_default_exporters)
    # If the model was successful restored including the global step, only 1
    # additional checkpoint to the init one should be created -> len == 2.
    self.assertLen(
        tf.io.gfile.glob(os.path.join(continue_model_dir, 'model*.meta')), 2)

  def test_init_from_checkpoint_use_avg_model_params_and_weights(self):
    """Tests that a simple model trains and exported models are valid."""
    gin.bind_parameter('tf.estimator.RunConfig.save_checkpoints_steps', 100)
    gin.bind_parameter('tf.estimator.RunConfig.keep_checkpoint_max', 3)
    model_dir = self.create_tempdir().full_path
    mock_t2r_model = mocks.MockT2RModel(
        preprocessor_cls=noop_preprocessor.NoOpPreprocessor,
        use_avg_model_params=True)

    mock_input_generator_train = mocks.MockInputGenerator(
        batch_size=_BATCH_SIZE)

    mock_input_generator = mocks.MockInputGenerator(batch_size=1)
    mock_input_generator.set_specification_from_model(
        mock_t2r_model, tf_estimator.ModeKeys.TRAIN)

    train_eval.train_eval_model(
        t2r_model=mock_t2r_model,
        input_generator_train=mock_input_generator_train,
        max_train_steps=_MAX_TRAIN_STEPS,
        model_dir=model_dir)

    init_checkpoint = tf.train.NewCheckpointReader(
        tf.train.latest_checkpoint(model_dir))

    # Verify that the serving estimator does exactly the same as the normal
    # estimator with all the parameters.
    initial_estimator_predict = tf_estimator.Estimator(
        model_fn=mock_t2r_model.model_fn,
        config=tf_estimator.RunConfig(model_dir=model_dir))

    # pylint: disable=g-complex-comprehension
    initial_predictions = [
        prediction['logit'] for prediction in list(
            initial_estimator_predict.predict(
                input_fn=mock_input_generator.create_dataset_input_fn(
                    mode=tf_estimator.ModeKeys.EVAL)))
    ]

    # The continuous training has its own directory.
    continue_model_dir = self.create_tempdir().full_path
    init_from_checkpoint_fn = functools.partial(
        abstract_model.default_init_from_checkpoint_fn, checkpoint=model_dir)
    continue_mock_t2r_model = mocks.MockT2RModel(
        preprocessor_cls=noop_preprocessor.NoOpPreprocessor,
        init_from_checkpoint_fn=init_from_checkpoint_fn)
    continue_mock_input_generator_train = mocks.MockInputGenerator(
        batch_size=_BATCH_SIZE)
    # Re-initialize the model and train for one step, basically the same
    # performance as the original model.
    train_eval.train_eval_model(
        t2r_model=continue_mock_t2r_model,
        input_generator_train=continue_mock_input_generator_train,
        model_dir=continue_model_dir,
        max_train_steps=_MAX_TRAIN_STEPS)

    continue_checkpoint = tf.train.NewCheckpointReader(
        tf.train.latest_checkpoint(continue_model_dir))

    for tensor_name, _ in tf.train.list_variables(model_dir):
      if 'ExponentialMovingAverage' in tensor_name:
        # These values are replaced by the swapping saver when using the
        # use_avg_model_params.
        continue
      if 'Adam' in tensor_name:
        # The adam optimizer values are not required.
        continue
      if 'global_step' in tensor_name:
        # The global step will be incremented by 1.
        continue
      self.assertAllClose(
          init_checkpoint.get_tensor(tensor_name),
          continue_checkpoint.get_tensor(tensor_name),
          atol=1e-3)

    # Verify that the serving estimator does exactly the same as the normal
    # estimator with all the parameters.
    continue_estimator_predict = tf_estimator.Estimator(
        model_fn=mock_t2r_model.model_fn,
        config=tf_estimator.RunConfig(model_dir=continue_model_dir))

    continue_predictions = [
        prediction['logit'] for prediction in list(
            continue_estimator_predict.predict(
                input_fn=mock_input_generator.create_dataset_input_fn(
                    mode=tf_estimator.ModeKeys.EVAL)))
    ]

    self.assertTrue(
        np.allclose(initial_predictions, continue_predictions, atol=1e-1))

    # A randomly initialized model estimator with all the parameters.
    random_estimator_predict = tf_estimator.Estimator(
        model_fn=mock_t2r_model.model_fn)

    random_predictions = [
        prediction['logit'] for prediction in list(
            random_estimator_predict.predict(
                input_fn=mock_input_generator.create_dataset_input_fn(
                    mode=tf_estimator.ModeKeys.EVAL)))
    ]
    self.assertFalse(
        np.allclose(initial_predictions, random_predictions, atol=1e-2))

  def test_freezing_some_variables(self):
    """Tests we can freeze training for parts of the network."""
    def freeze_biases(var):
      # Update all variables except bias variables.
      return 'bias' not in var.name

    gin.bind_parameter('tf.estimator.RunConfig.save_checkpoints_steps', 100)
    gin.bind_parameter('create_train_op.filter_trainables_fn', freeze_biases)
    model_dir = self.create_tempdir().full_path
    mock_t2r_model = mocks.MockT2RModel(
        preprocessor_cls=noop_preprocessor.NoOpPreprocessor)
    mock_input_generator_train = mocks.MockInputGenerator(
        batch_size=_BATCH_SIZE)

    train_eval.train_eval_model(
        t2r_model=mock_t2r_model,
        input_generator_train=mock_input_generator_train,
        max_train_steps=_MAX_TRAIN_STEPS,
        model_dir=model_dir)

    start_checkpoint = tf.train.NewCheckpointReader(
        os.path.join(model_dir, 'model.ckpt-0'))
    last_checkpoint = tf.train.NewCheckpointReader(
        tf.train.latest_checkpoint(model_dir))
    for var_name, _ in tf.train.list_variables(model_dir):
      # Some of the batch norm moving averages are constant over training on the
      # mock data used.
      if 'batch_norm' in var_name:
        continue
      if 'bias' not in var_name:
        # Should update.
        self.assertNotAllClose(
            start_checkpoint.get_tensor(var_name),
            last_checkpoint.get_tensor(var_name),
            atol=1e-3)
      else:
        # Should not update.
        self.assertAllClose(
            start_checkpoint.get_tensor(var_name),
            last_checkpoint.get_tensor(var_name),
            atol=1e-3)


if __name__ == '__main__':
  tf.test.main()
