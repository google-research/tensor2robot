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

"""Tests for TD3 Hooks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
from tensor2robot.export_generators import abstract_export_generator
from tensor2robot.hooks import checkpoint_hooks
from tensor2robot.hooks import td3
from tensor2robot.utils import mocks
from tensor2robot.utils import tensorspec_utils
import tensorflow.compat.v1 as tf  # tf

_BATCH_SIZES_FOR_EXPORT = [128]
_MODEL_DIR = "model_dir"
_NUMPY_WARMUP_REQUESTS = "warmup_requests"
_EXPORT_DIR = "export_dir"
_LAGGED_EXPORT_DIR = "lagged_export_dir"


class MockEstimator(tf.estimator.Estimator):

  def __init__(self):
    pass

  @property
  def model_dir(self):
    return _MODEL_DIR


class Td3Test(tf.test.TestCase):

  @mock.patch.object(MockEstimator, "export_saved_model")
  @mock.patch.object(checkpoint_hooks.LaggedCheckpointListener, "__init__")
  @mock.patch.object(mocks.MockExportGenerator,
                     "create_serving_input_receiver_numpy_fn")
  @mock.patch.object(abstract_export_generator.AbstractExportGenerator,
                     "create_warmup_requests_numpy")
  def test_hooks(self, mock_create_warmup_requests_numpy,
                 mock_create_serving_input_receiver_numpy_fn,
                 mock_checkpoint_init, mock_export_saved_model):

    def _checkpoint_init(export_fn, export_dir, **kwargs):
      del kwargs
      export_fn(export_dir, global_step=1)
      return None

    mock_checkpoint_init.side_effect = _checkpoint_init

    export_generator = mocks.MockExportGenerator()

    hook_builder = td3.TD3Hooks(
        export_dir=_EXPORT_DIR,
        lagged_export_dir=_LAGGED_EXPORT_DIR,
        batch_sizes_for_export=_BATCH_SIZES_FOR_EXPORT,
        export_generator=export_generator)

    model = mocks.MockT2RModel()
    estimator = MockEstimator()

    mock_create_warmup_requests_numpy.return_value = _NUMPY_WARMUP_REQUESTS

    hooks = hook_builder.create_hooks(t2r_model=model, estimator=estimator)
    self.assertLen(hooks, 1)

    mock_create_warmup_requests_numpy.assert_called_with(
        batch_sizes=_BATCH_SIZES_FOR_EXPORT,
        export_dir=_MODEL_DIR)

    mock_export_saved_model.assert_called_with(
        serving_input_receiver_fn=mock.ANY,
        export_dir_base=_EXPORT_DIR,
        assets_extra={
            "tf_serving_warmup_requests": _NUMPY_WARMUP_REQUESTS,
            tensorspec_utils.T2R_ASSETS_FILENAME: mock.ANY
        })

    mock_create_serving_input_receiver_numpy_fn.assert_called()


if __name__ == "__main__":
  tf.test.main()
