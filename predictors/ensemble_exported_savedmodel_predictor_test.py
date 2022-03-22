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

"""Tests for tensor2robot.predictors.ensemble_exported_savedmodel_predictor."""

import os

from absl import flags
from absl.testing import parameterized
import gin
import numpy as np
from tensor2robot.input_generators import default_input_generator
from tensor2robot.predictors import ensemble_exported_savedmodel_predictor
from tensor2robot.utils import mocks
from tensor2robot.utils import tensorspec_utils
from tensor2robot.utils import train_eval
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

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

  def test_predictor_with_default_exporter(self):
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
    # Create ensemble by duplicating the same directory multiple times.
    export_dirs = ','.join(
        [os.path.join(model_dir, 'export', 'latest_exporter_numpy')] * 2)
    predictor = ensemble_exported_savedmodel_predictor.EnsembleExportedSavedModelPredictor(
        export_dirs=export_dirs, local_export_root=None, ensemble_size=2)
    predictor.resample_ensemble()
    with self.assertRaises(ValueError):
      predictor.get_feature_specification()
    with self.assertRaises(ValueError):
      predictor.predict({'does_not_matter': np.zeros(1)})
    with self.assertRaises(ValueError):
      _ = predictor.model_version
    self.assertEqual(predictor.global_step, -1)
    self.assertTrue(predictor.restore(is_async=False))
    self.assertGreater(predictor.model_version, 0)
    self.assertEqual(predictor.global_step, -1)
    ref_feature_spec = mock_model.preprocessor.get_in_feature_specification(
        tf_estimator.ModeKeys.PREDICT)
    tensorspec_utils.assert_equal(predictor.get_feature_specification(),
                                  ref_feature_spec)
    features = tensorspec_utils.make_random_numpy(
        ref_feature_spec, batch_size=_BATCH_SIZE)
    predictions = predictor.predict(features)
    self.assertLen(predictions, 1)
    self.assertCountEqual(predictions.keys(), ['logit'])
    self.assertEqual(predictions['logit'].shape, (2, 1))


if __name__ == '__main__':
  tf.test.main()
