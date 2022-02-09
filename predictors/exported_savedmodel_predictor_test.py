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

"""Tests for tensor2robot.predictors.exported_savedmodel_predictor."""

import os

from absl import flags
from absl.testing import parameterized
import gin
import numpy as np
from tensor2robot.hooks import async_export_hook_builder
from tensor2robot.input_generators import default_input_generator
from tensor2robot.predictors import exported_savedmodel_predictor
from tensor2robot.utils import mocks
from tensor2robot.utils import tensorspec_utils
from tensor2robot.utils import train_eval
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

_EXPORT_DIR = 'asyn_export'

_BATCH_SIZE = 2
_MAX_TRAIN_STEPS = 3
_MAX_EVAL_STEPS = 2


class ExportedSavedmodelPredictorTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ExportedSavedmodelPredictorTest, self).setUp()
    gin.clear_config()
    gin.parse_config('tf.estimator.RunConfig.save_checkpoints_steps=1')

  @parameterized.parameters((False), (True))
  def test_predictor_with_default_exporter(self, is_async):
    input_generator = default_input_generator.DefaultRandomInputGenerator(
        batch_size=_BATCH_SIZE)
    model_dir = self.create_tempdir().full_path
    mock_model = mocks.MockT2RModel()
    train_eval.train_eval_model(
        t2r_model=mock_model,
        input_generator_train=input_generator,
        input_generator_eval=input_generator,
        max_train_steps=_MAX_TRAIN_STEPS,
        eval_steps=_MAX_EVAL_STEPS,
        model_dir=model_dir,
        create_exporters_fn=train_eval.create_default_exporters)

    predictor = exported_savedmodel_predictor.ExportedSavedModelPredictor(
        export_dir=os.path.join(model_dir, 'export', 'latest_exporter_numpy'))
    with self.assertRaises(ValueError):
      predictor.get_feature_specification()
    with self.assertRaises(ValueError):
      predictor.predict({'does_not_matter': np.zeros(1)})
    with self.assertRaises(ValueError):
      _ = predictor.model_version
    self.assertEqual(predictor.global_step, -1)
    self.assertTrue(predictor.restore(is_async=is_async))
    self.assertGreater(predictor.model_version, 0)
    self.assertEqual(predictor.global_step, 3)
    ref_feature_spec = mock_model.preprocessor.get_in_feature_specification(
        tf.estimator.ModeKeys.PREDICT)
    tensorspec_utils.assert_equal(predictor.get_feature_specification(),
                                  ref_feature_spec)
    features = tensorspec_utils.make_random_numpy(
        ref_feature_spec, batch_size=_BATCH_SIZE)
    predictions = predictor.predict(features)
    self.assertLen(predictions, 1)
    self.assertCountEqual(predictions.keys(), ['logit'])
    self.assertEqual(predictions['logit'].shape, (2, 1))

  def test_predictor_with_async_hook(self):
    model_dir = self.create_tempdir().full_path
    export_dir = os.path.join(model_dir, _EXPORT_DIR)
    hook_builder = async_export_hook_builder.AsyncExportHookBuilder(
        export_dir=export_dir,
        create_export_fn=async_export_hook_builder.default_create_export_fn)
    input_generator = default_input_generator.DefaultRandomInputGenerator(
        batch_size=_BATCH_SIZE)
    mock_model = mocks.MockT2RModel()
    train_eval.train_eval_model(
        t2r_model=mock_model,
        input_generator_train=input_generator,
        train_hook_builders=[hook_builder],
        max_train_steps=_MAX_TRAIN_STEPS,
        model_dir=model_dir)

    predictor = exported_savedmodel_predictor.ExportedSavedModelPredictor(
        export_dir=os.path.join(model_dir, _EXPORT_DIR))
    with self.assertRaises(ValueError):
      predictor.get_feature_specification()
    with self.assertRaises(ValueError):
      predictor.predict({'does_not_matter': np.zeros(1)})
    with self.assertRaises(ValueError):
      _ = predictor.model_version
    self.assertEqual(predictor.global_step, -1)
    self.assertTrue(predictor.restore())
    self.assertGreater(predictor.model_version, 0)
    # NOTE: The async hook builder will export the global step.
    self.assertEqual(predictor.global_step, 3)
    ref_feature_spec = mock_model.preprocessor.get_in_feature_specification(
        tf.estimator.ModeKeys.PREDICT)
    tensorspec_utils.assert_equal(predictor.get_feature_specification(),
                                  ref_feature_spec)
    features = tensorspec_utils.make_random_numpy(
        ref_feature_spec, batch_size=_BATCH_SIZE)
    predictions = predictor.predict(features)
    self.assertLen(predictions, 1)
    self.assertCountEqual(predictions.keys(), ['logit'])
    self.assertEqual(predictions['logit'].shape, (2, 1))

  def test_predictor_timeout(self):
    predictor = exported_savedmodel_predictor.ExportedSavedModelPredictor(
        export_dir='/random/path/which/does/not/exist', timeout=1)
    self.assertFalse(predictor.restore())

  @parameterized.parameters(
      (exported_savedmodel_predictor.RestoreOptions.RESTORE_SYNCHRONOUSLY),
      (exported_savedmodel_predictor.RestoreOptions.RESTORE_ASYNCHRONOUSLY))
  def test_predictor_init_with_default_exporter(self, restore_model_option):
    input_generator = default_input_generator.DefaultRandomInputGenerator(
        batch_size=_BATCH_SIZE)
    model_dir = self.create_tempdir().full_path
    mock_model = mocks.MockT2RModel()
    train_eval.train_eval_model(
        t2r_model=mock_model,
        input_generator_train=input_generator,
        input_generator_eval=input_generator,
        max_train_steps=_MAX_TRAIN_STEPS,
        eval_steps=_MAX_EVAL_STEPS,
        model_dir=model_dir,
        create_exporters_fn=train_eval.create_default_exporters)

    predictor = exported_savedmodel_predictor.ExportedSavedModelPredictor(
        export_dir=os.path.join(model_dir, 'export', 'latest_exporter_numpy'),
        restore_model_option=restore_model_option)
    if restore_model_option == exported_savedmodel_predictor.RestoreOptions.RESTORE_SYNCHRONOUSLY:
      predictor.restore()
    self.assertGreater(predictor.model_version, 0)
    self.assertEqual(predictor.global_step, 3)
    ref_feature_spec = mock_model.preprocessor.get_in_feature_specification(
        tf.estimator.ModeKeys.PREDICT)
    tensorspec_utils.assert_equal(predictor.get_feature_specification(),
                                  ref_feature_spec)

  def test_predictor_loading_timeout_0(self):
    input_generator = default_input_generator.DefaultRandomInputGenerator(
        batch_size=_BATCH_SIZE)
    model_dir = self.create_tempdir().full_path
    mock_model = mocks.MockT2RModel()
    train_eval.train_eval_model(
        t2r_model=mock_model,
        input_generator_train=input_generator,
        input_generator_eval=input_generator,
        max_train_steps=_MAX_TRAIN_STEPS,
        eval_steps=_MAX_EVAL_STEPS,
        model_dir=model_dir,
        create_exporters_fn=train_eval.create_default_exporters)

    predictor = exported_savedmodel_predictor.ExportedSavedModelPredictor(
        export_dir=os.path.join(model_dir, 'export', 'latest_exporter_numpy'),
        timeout=0)
    predictor.restore()
    self.assertGreater(predictor.model_version, 0)
    self.assertEqual(predictor.global_step, 3)
    ref_feature_spec = mock_model.preprocessor.get_in_feature_specification(
        tf.estimator.ModeKeys.PREDICT)
    tensorspec_utils.assert_equal(predictor.get_feature_specification(),
                                  ref_feature_spec)

  def test_predictor_load_final_model(self):
    input_generator = default_input_generator.DefaultRandomInputGenerator(
        batch_size=_BATCH_SIZE)
    model_dir = self.create_tempdir().full_path
    mock_model = mocks.MockT2RModel()
    train_eval.train_eval_model(
        t2r_model=mock_model,
        input_generator_train=input_generator,
        input_generator_eval=input_generator,
        max_train_steps=_MAX_TRAIN_STEPS,
        eval_steps=_MAX_EVAL_STEPS,
        model_dir=model_dir,
        create_exporters_fn=train_eval.create_default_exporters)
    export_dir = os.path.join(model_dir, 'export', 'latest_exporter_numpy')
    final_export_dir = sorted(
        tf.io.gfile.glob(os.path.join(export_dir, '*')), reverse=True)[0]
    predictor = exported_savedmodel_predictor.ExportedSavedModelPredictor(
        export_dir=final_export_dir)
    predictor.restore()
    self.assertGreater(predictor.model_version, 0)
    self.assertEqual(predictor.global_step, 3)
    ref_feature_spec = mock_model.preprocessor.get_in_feature_specification(
        tf.estimator.ModeKeys.PREDICT)
    tensorspec_utils.assert_equal(predictor.get_feature_specification(),
                                  ref_feature_spec)


if __name__ == '__main__':
  tf.test.main()
