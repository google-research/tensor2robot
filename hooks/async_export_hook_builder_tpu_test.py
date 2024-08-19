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

"""Tests for TD3 Hooks."""

import os
import gin
from tensor2robot.hooks import async_export_hook_builder
from tensor2robot.predictors import exported_savedmodel_predictor
from tensor2robot.preprocessors import noop_preprocessor
from tensor2robot.utils import mocks
from tensor2robot.utils import train_eval
import tensorflow.compat.v1 as tf  # tf

_EXPORT_DIR = 'export_dir'
_BATCH_SIZES_FOR_EXPORT = [128]
_MAX_STEPS = 4
_BATCH_SIZE = 4


class AsyncExportHookBuilderTest(tf.test.TestCase):

  def test_with_mock_training(self):
    model_dir = self.create_tempdir().full_path
    mock_t2r_model = mocks.MockT2RModel(
        preprocessor_cls=noop_preprocessor.NoOpPreprocessor,
        device_type='tpu',
        use_avg_model_params=True)

    mock_input_generator = mocks.MockInputGenerator(batch_size=_BATCH_SIZE)
    export_dir = os.path.join(model_dir, _EXPORT_DIR)
    hook_builder = async_export_hook_builder.AsyncExportHookBuilder(
        export_dir=export_dir,
        create_export_fn=async_export_hook_builder.default_create_export_fn)

    gin.parse_config('tf.contrib.tpu.TPUConfig.iterations_per_loop=1')
    gin.parse_config('tf.estimator.RunConfig.save_checkpoints_steps=1')

    # We optimize our network.
    train_eval.train_eval_model(
        t2r_model=mock_t2r_model,
        input_generator_train=mock_input_generator,
        train_hook_builders=[hook_builder],
        model_dir=model_dir,
        max_train_steps=_MAX_STEPS)
    self.assertNotEmpty(tf.io.gfile.listdir(model_dir))
    self.assertNotEmpty(tf.io.gfile.listdir(export_dir))
    for exported_model_dir in tf.io.gfile.listdir(export_dir):
      self.assertNotEmpty(
          tf.io.gfile.listdir(os.path.join(export_dir, exported_model_dir)))
    predictor = exported_savedmodel_predictor.ExportedSavedModelPredictor(
        export_dir=export_dir)
    self.assertTrue(predictor.restore())


if __name__ == '__main__':
  tf.test.main()
