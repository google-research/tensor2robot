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

"""Hook builders for TD3 distributed training with SavedModels."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import tempfile

import gin
from tensor2robot.export_generators import abstract_export_generator
from tensor2robot.hooks import checkpoint_hooks
from tensor2robot.hooks import hook_builder
from tensor2robot.models import model_interface
from tensor2robot.utils import tensorspec_utils
import tensorflow as tf  # tf

from typing import Text, List, Callable

CreateExportFnType = Callable[[
    model_interface.ModelInterface,
    tf.estimator.Estimator,
    abstract_export_generator.AbstractExportGenerator,
], Callable[[Text], Text]]


@gin.configurable(whitelist=['batch_sizes_for_export'])
def default_create_export_fn(
    t2r_model,
    estimator,
    export_generator,
    batch_sizes_for_export):
  """Create an export function for a device type.

  Args:
    t2r_model: A T2RModel instance.
    estimator: The estimator used for training.
    export_generator: An export generator.
    batch_sizes_for_export: A list of batch sizes for warming up serving.

  Returns:
    A callable function which exports a saved model and returns the path.
  """
  warmup_requests_file = export_generator.create_warmup_requests_numpy(
      batch_sizes=batch_sizes_for_export, export_dir=estimator.model_dir)

  # Create pkl of the input to save alongside the exported models
  tmpdir = tempfile.mkdtemp()
  in_feature_spec = t2r_model.get_feature_specification_for_packing(
      mode=tf.estimator.ModeKeys.PREDICT)
  in_label_spec = t2r_model.get_label_specification_for_packing(
      mode=tf.estimator.ModeKeys.PREDICT)
  input_specs_pkl_filename = os.path.join(tmpdir, 'input_specs.pkl')
  tensorspec_utils.write_to_file(in_feature_spec, in_label_spec,
                                 input_specs_pkl_filename)
  assets = {
      'tf_serving_warmup_requests': warmup_requests_file,
      'input_specs.pkl': input_specs_pkl_filename
  }

  def _export_fn(export_dir):
    return estimator.export_saved_model(
        export_dir_base=export_dir,
        serving_input_receiver_fn=export_generator
        .create_serving_input_receiver_numpy_fn(),
        assets_extra=assets)

  return _export_fn


@gin.configurable
class AsyncExportHookBuilder(hook_builder.HookBuilder):
  """Creates hooks for exporting for cpu and tpu for serving.

  Arguments:
    export_dir: Directory to output the latest models.
    save_secs: Interval to save models, and copy the latest model from
      `export_dir` to `lagged_export_dir`.
    num_versions: Number of model versions to save in each directory
  """

  def __init__(
      self,
      export_dir,
      save_secs = 90,
      num_versions = 3,
      create_export_fn = default_create_export_fn,
  ):
    super(AsyncExportHookBuilder, self).__init__()
    self._save_secs = save_secs
    self._num_versions = num_versions
    self._export_dir = export_dir
    self._create_export_fn = create_export_fn

  def create_hooks(
      self, t2r_model, estimator,
      export_generator
  ):
    export_generator.set_specification_from_model(t2r_model)
    return [
        tf.contrib.tpu.AsyncCheckpointSaverHook(
            save_secs=self._save_secs,
            checkpoint_dir=estimator.model_dir,
            listeners=[
                checkpoint_hooks.CheckpointExportListener(
                    export_fn=self._create_export_fn(t2r_model, estimator,
                                                     export_generator),
                    num_versions=self._num_versions,
                    export_dir=self._export_dir)
            ])
    ]
