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

"""Hook builders for TD3 distributed training with SavedModels."""

import os
import tempfile
from typing import Text, List, Callable, Optional

import gin
from tensor2robot.export_generators import abstract_export_generator
from tensor2robot.export_generators import default_export_generator
from tensor2robot.hooks import checkpoint_hooks
from tensor2robot.hooks import hook_builder
from tensor2robot.models import model_interface
from tensor2robot.proto import t2r_pb2
from tensor2robot.utils import tensorspec_utils
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow.compat.v1 as tf  # tf

from tensorflow.contrib import tpu as contrib_tpu

CreateExportFnType = Callable[[
    model_interface.ModelInterface,
    tf_estimator.Estimator,
    abstract_export_generator.AbstractExportGenerator,
], Callable[[Text, int], Text]]


def default_create_export_fn(
    t2r_model,
    estimator,
    export_generator
):
  """Create an export function for a device type.

  Args:
    t2r_model: A T2RModel instance.
    estimator: The estimator used for training.
    export_generator: An export generator.

  Returns:
    A callable function which exports a saved model and returns the path.
  """

  in_feature_spec = t2r_model.get_feature_specification_for_packing(
      mode=tf_estimator.ModeKeys.PREDICT)
  in_label_spec = t2r_model.get_label_specification_for_packing(
      mode=tf_estimator.ModeKeys.PREDICT)
  t2r_assets = t2r_pb2.T2RAssets()
  t2r_assets.feature_spec.CopyFrom(in_feature_spec.to_proto())
  t2r_assets.label_spec.CopyFrom(in_label_spec.to_proto())

  def _export_fn(export_dir, global_step):
    """The actual closure function creating the exported model and assets."""
    # Create additional assets for the exported models
    t2r_assets.global_step = global_step
    tmpdir = tempfile.mkdtemp()
    t2r_assets_filename = os.path.join(tmpdir,
                                       tensorspec_utils.T2R_ASSETS_FILENAME)
    tensorspec_utils.write_t2r_assets_to_file(t2r_assets, t2r_assets_filename)
    assets = {
        tensorspec_utils.T2R_ASSETS_FILENAME: t2r_assets_filename,
    }
    return estimator.export_saved_model(
        export_dir_base=export_dir,
        serving_input_receiver_fn=export_generator
        .create_serving_input_receiver_numpy_fn(),
        assets_extra=assets)

  return _export_fn


@gin.configurable
class AsyncExportHookBuilder(hook_builder.HookBuilder):
  """Creates hooks for exporting for cpu and tpu for serving.

  Attributes:
    export_dir: Directory to output the latest models.
    save_secs: Interval to save models, and copy the latest model from
      `export_dir` to `lagged_export_dir`.
    num_versions: Number of model versions to save in each directory
    export_generator: The export generator used to generate the
      serving_input_receiver_fn.
  """

  def __init__(
      self,
      export_dir,
      save_secs = 90,
      num_versions = 3,
      create_export_fn = default_create_export_fn,
      export_generator = None,
  ):
    super(AsyncExportHookBuilder, self).__init__()
    self._save_secs = save_secs
    self._num_versions = num_versions
    self._export_dir = export_dir
    self._create_export_fn = create_export_fn
    if export_generator is None:
      self._export_generator = default_export_generator.DefaultExportGenerator()
    else:
      self._export_generator = export_generator

  def create_hooks(
      self,
      t2r_model,
      estimator,
  ):
    self._export_generator.set_specification_from_model(t2r_model)
    return [
        contrib_tpu.AsyncCheckpointSaverHook(
            save_secs=self._save_secs,
            checkpoint_dir=estimator.model_dir,
            listeners=[
                checkpoint_hooks.CheckpointExportListener(
                    export_fn=self._create_export_fn(t2r_model, estimator,
                                                     self._export_generator),
                    num_versions=self._num_versions,
                    export_dir=self._export_dir)
            ])
    ]
