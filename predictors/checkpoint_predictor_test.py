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

"""Tests for py.tensor2robot.predictors.checkpoint_predictor_test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import gin
import numpy as np
from tensor2robot.input_generators import default_input_generator
from tensor2robot.predictors import checkpoint_predictor
from tensor2robot.utils import mocks
from tensor2robot.utils import tensorspec_utils
from tensor2robot.utils import train_eval
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

_BATCH_SIZE = 2
_MAX_TRAIN_STEPS = 3


class CheckpointPredictorTest(tf.test.TestCase):

  def setUp(self):
    super(CheckpointPredictorTest, self).setUp()
    gin.clear_config()
    gin.parse_config('tf.estimator.RunConfig.save_checkpoints_steps=1')

  def test_predictor(self):
    input_generator = default_input_generator.DefaultRandomInputGenerator(
        batch_size=_BATCH_SIZE)
    model_dir = self.create_tempdir().full_path
    mock_model = mocks.MockT2RModel()
    train_eval.train_eval_model(
        t2r_model=mock_model,
        input_generator_train=input_generator,
        max_train_steps=_MAX_TRAIN_STEPS,
        model_dir=model_dir)

    predictor = checkpoint_predictor.CheckpointPredictor(
        t2r_model=mock_model, checkpoint_dir=model_dir, use_gpu=False)
    with self.assertRaises(ValueError):
      predictor.predict({'does_not_matter': np.zeros(1)})
    self.assertEqual(predictor.model_version, -1)
    self.assertEqual(predictor.global_step, -1)
    self.assertTrue(predictor.restore())
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
    self.assertCountEqual(sorted(predictions.keys()), ['logit'])
    self.assertEqual(predictions['logit'].shape, (2, 1))

  def test_predictor_timeout(self):
    mock_model = mocks.MockT2RModel()
    predictor = checkpoint_predictor.CheckpointPredictor(
        t2r_model=mock_model,
        checkpoint_dir='/random/path/which/does/not/exist',
        timeout=1)
    self.assertFalse(predictor.restore())

  def test_predictor_raises(self):
    mock_model = mocks.MockT2RModel()
    # Raises because no checkpoint_dir and has been set and restore is called.
    predictor = checkpoint_predictor.CheckpointPredictor(t2r_model=mock_model)
    with self.assertRaises(ValueError):
      predictor.restore()


if __name__ == '__main__':
  tf.test.main()
