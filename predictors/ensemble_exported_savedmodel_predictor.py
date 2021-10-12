# coding=utf-8
# Copyright 2021 The Tensor2Robot Authors.
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
"""An experimental predictor over an ensemble of exported models."""

import os
from typing import Callable, Dict, List, Optional, Text  # pylint: disable=unused-import

import gin
import numpy as np
from tensor2robot.predictors import abstract_predictor
from tensor2robot.predictors import exported_savedmodel_predictor
from tensor2robot.utils import tensorspec_utils
import tensorflow.compat.v1 as tf  # tf

RestoreOptions = exported_savedmodel_predictor.RestoreOptions


@gin.configurable
class EnsembleExportedSavedModelPredictor(
    abstract_predictor.AbstractPredictor):
  """A predictor loading multiple exported saved models."""

  def __init__(
      self,
      export_dirs,
      local_export_root = None,
      ensemble_size = 1,
      timeout = 600,
      tf_config = None,
      restore_model_option = RestoreOptions.DO_NOT_RESTORE):
    """Creates an instance.

    Args:
      export_dirs: A comma-separated list of exported saved model directory
        paths.
      local_export_root: When loading the model, if export dir does not exist,
        looks for the basename in local_export_dir. This is useful for on-robot
        deployments where we want to refer to the export_dir by its original
        path name but resolve to a local path when actually loading it.
      ensemble_size: Integer specifying how many predictors to sample from and
        aggregate predictions over, each time reset() is called. If =1, then
        is identical to random checkpoint selection. If >1, predictions are
        averaged.
      timeout: (defaults to 600 seconds) If no checkpoint has been found after
        timeout seconds restore fails. At least attempt will be made to load
        even with timeout 0 or -1.
      tf_config: The tf.ConfigProto used to configure the TensorFlow session.
      restore_model_option: If set to RestoreOptions.DO_NOT_RESTORE, the model
        is not restored in the constructor. If set to
        RestoreOptions.RESTORE_SYNCHRONOUSLY, the model is restored in this
        thread, if set to RestoreOptions.RESTORE_ASYNCHRONOUSLY, the model is
        restored in a separate thread.
    """
    super(EnsembleExportedSavedModelPredictor, self).__init__()
    self._export_dirs = export_dirs.split(',')
    self._local_export_root = local_export_root
    self._ensemble_size = ensemble_size
    # Resolve export dirs.
    self._resolved_export_dirs = []
    for path in self._export_dirs:
      if local_export_root is not None:
        resolved_path = os.path.join(local_export_root, os.path.basename(path))
        if not os.path.exists(resolved_path):
          raise ValueError(
              'Cannot find local export dir {}'.format(resolved_path))
      else:
        # may need CNS access.
        resolved_path = path
      self._resolved_export_dirs.append(resolved_path)
    # Instantiate each predictor.
    self._predictors = []
    for export_dir in self._resolved_export_dirs:
      self._predictors.append(
          exported_savedmodel_predictor.ExportedSavedModelPredictor(
              export_dir=export_dir, timeout=timeout, tf_config=tf_config,
              restore_model_option=restore_model_option))
    # State variable tracking which predictors are being used.
    self._ensemble_export_dirs = []
    self._predictor_ensemble = []

  def predict(self, features):
    """Featurize once, then pass through predictor ensemble."""
    self.assert_is_loaded()
    # If using an action-tiled model, the action tiling must align with the spec
    # structure. If the supplied inputs align with the batch-tiled action,
    # expand the input to feed the tiled batch elements.
    flattened_feature_spec = tensorspec_utils.flatten_spec_structure(
        self.get_feature_specification())

    def _maybe_expand_dim(path, val):
      model_spec = flattened_feature_spec.get(path)
      if model_spec and model_spec.shape.as_list() == list(val.shape):
        return np.expand_dims(val, 0)
      return val
    features = {k: _maybe_expand_dim(k, val) for k, val in features.items()}
    predictions = [p._predict_fn(features) for p in self._predictor_ensemble]  # pylint: disable=protected-access
    if self._ensemble_size > 1:
      return tf.nest.map_structure(
          lambda *list_of_arrays: np.mean(list_of_arrays, axis=0), *predictions)
    else:
      return predictions[0]

  def resample_ensemble(self):
    indices = np.random.choice(len(self._export_dirs), self._ensemble_size)
    self._predictor_ensemble = [self._predictors[i] for i in indices]
    self._ensemble_export_dirs = [self._export_dirs[i] for i in indices]

  def get_feature_specification(self):
    """Exposes the required input features for evaluation of the model."""
    return self._predictors[0].get_feature_specification()

  def get_label_specification(self
                             ):
    """Exposes the optional labels for evaluation of the model."""
    return self._predictors[0].get_label_specification()

  def restore(self, is_async = False):
    """Restores the model parameters from the latest available data."""
    return np.all([p.restore() for p in self._predictors])

  def assert_is_loaded(self):
    """Raises a ValueError if model parameters haven't been loaded.

    If restore() has been called with is_async=True, the thread will be joined
    before the check is performed.
    """
    if not self._predictor_ensemble:
      raise ValueError('predictor ensemble empty, call reset() first.')
    self._predictor_ensemble[0].assert_is_loaded()

  @property
  def model_version(self):
    """The version of the model currently in use.

    Returns:
      The timestamp since Unix Epoch which allows to uniquely identify the
      exported saved model.
    """
    self.assert_is_loaded()
    return int(os.path.basename(
        self._predictor_ensemble[0].model_path))

  @property
  def global_step(self):
    """The global step of the model currently in use.

    Returns:
      The global step the model was exported at./
    """
    if not self._predictor_ensemble or self._ensemble_size != 1:
      return -1
    else:
      # Predictors are loaded and reset() has been called and ensemble_size = 1.
      return self._predictor_ensemble[0].global_step

  @property
  def model_path(self):
    """The path of the model currently in use."""
    self.assert_is_loaded()
    return ','.join(self._ensemble_export_dirs)

  def close(self):
    """Closes all open handles used throughout model evaluation."""
    # NoOp for this predictor.
