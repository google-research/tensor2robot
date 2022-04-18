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

"""A predictor based on exported saved models."""

import enum
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Text  # pylint: disable=unused-import

from absl import logging
import gin
import numpy as np
from tensor2robot.predictors import abstract_predictor
from tensor2robot.utils import tensorspec_utils
import tensorflow.compat.v1 as tf  # tf


_BUSY_WAITING_SLEEP_TIME_IN_SECS = 10


@gin.configurable
def create_tf_config(per_process_gpu_memory_fraction):
  tf_config = tf.ConfigProto()
  if per_process_gpu_memory_fraction is not None:
    tf_config.gpu_options.CopyFrom(
        tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction))
  return tf_config


@gin.constants_from_enum
class RestoreOptions(enum.Enum):
  DO_NOT_RESTORE = 0
  RESTORE_SYNCHRONOUSLY = 1
  RESTORE_ASYNCHRONOUSLY = 2


@gin.configurable
class ExportedSavedModelPredictor(abstract_predictor.AbstractPredictor):
  """A predictor loading from exported saved models."""

  def __init__(
      self,
      export_dir,
      timeout = 600,
      tf_config = None,
      restore_model_option = RestoreOptions.DO_NOT_RESTORE):
    """Creates an instance.

    Args:
      export_dir: A path to a directory containing exported saved models in
        their respective directories. All directories are assumed to contain a
        saved model and the last (lexicographical sorting) represents the best
        model. Alternatively, if export_dir is points to a specific
        lexicographic subdirectory it will restore from there.
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
    super(ExportedSavedModelPredictor, self).__init__()
    self._export_dir = export_dir
    self._timeout = timeout
    self._latest_export_dir = None
    self._predict_fn = None  # type: Callable
    self._feature_spec = None  # type: tensorspec_utils.TensorSpecStruct
    self._label_spec = None
    self._tf_config = tf_config
    self._global_step = -1
    self._restore_thread = None  # type: threading.Thread
    if restore_model_option is not RestoreOptions.DO_NOT_RESTORE:
      is_async = restore_model_option is RestoreOptions.RESTORE_ASYNCHRONOUSLY
      self.restore(is_async=is_async)

  def predict(self, features):
    """Predicts based on feature input using the loaded model.

    Args:
      features: A dict containing the features used for predictions.

    Returns:
      The result of the queried model predictions.
    """

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
    return self._predict_fn(features)

  def get_feature_specification(self):
    """Exposes the required input features for evaluation of the model."""
    self.assert_is_loaded()
    return self._feature_spec

  def get_label_specification(self
                             ):
    """Exposes the optional labels for evaluation of the model."""
    self.assert_is_loaded()
    return self._label_spec

  def init_randomly(self):
    """Initializes model parameters from with random values."""
    raise ValueError('Random initialization is not supported '
                     'for ExportedSavedModelPredictors.')

  def restore(self, is_async = False):
    """Restores the model parameters from the latest available data.

    Note, if this function was called previously with is_async set to True, this
    function will wait until that finished before restoring model parameters.

    Args:
      is_async: If set, the model parameters are restored in a separate thread.

    Raises:
      ValueError: If no checkpoint can be found or loaded within the user
        defined timeout.

    Returns:
      True if a exported saved model has been loaded and False otherwise.
    """
    if is_async:
      logging.info('Restoring model parameters in a separate thread.')
      self._restore_thread = threading.Thread(
          target=self._restore,
          daemon=True,
          name='ExportedSavedModelPredictor.restore')
      self._restore_thread.start()
      return True

    self._maybe_join_restore_thread()
    return self._restore()

  def _is_final_export_dir(self, export_dir):
    """Checks if the export directory is final one, which will not change.

    Args:
      export_dir: The export directory path. This is typically a directory
        containing many directories encoding seconds since unix time 0 or the
        final directory containing the saved_model.pb.

    Returns:
      True if the export directory ends with saved_model.py.
    """
    saved_model = os.path.join(export_dir, 'saved_model.pb')
    t2r_assets = os.path.join(export_dir, 'assets.extra', 't2r_assets.pbtxt')
    if tf.io.gfile.exists(saved_model) and tf.io.gfile.exists(t2r_assets):
      return True
    return False

  def _restore(self):
    """Restores the model parameters from the latest available data.

    Raises:
      ValueError: If no checkpoint can be found or loaded within the user
        defined timeout.

    Returns:
      True if a exported saved model has been loaded and False otherwise.
    """
    start_time = time.time()
    while True:
      if self._is_final_export_dir(self._export_dir):
        model_dir = self._export_dir
      else:
        logging.info('Looking for a model in subfolders of %s',
                     self._export_dir)
        model_dir = self._latest_valid_model_dirs(
            tf.io.gfile.glob(os.path.join(self._export_dir, '*')))

      if model_dir is not None:
        logging.info('Found latest model at %s. ', model_dir)
        break

      logging.info('Waiting for an exported model to become available at %s.',
                   self._export_dir)
      if time.time() - start_time > self._timeout:
        break
      # Since a checkpoint might not be available and this is a busy waiting
      # loop, we throttle checking for checkpoints.
      time.sleep(_BUSY_WAITING_SLEEP_TIME_IN_SECS)

    if model_dir is None:
      logging.warning('No checkpoint available after %s seconds.',
                      str(self._timeout))
      return False

    if self._latest_export_dir == model_dir:
      # The latest model has already been loaded.
      return True

    logging.info('Loading the latest model at %s. ', model_dir)
    self._latest_export_dir = model_dir
    start_time_loading = time.time()

    # Note, loading from a saved model might require several attempts if
    # the checkpoint gets written asynchronously.
    while True:
      try:
        t2r_assets_file_path = os.path.join(
            model_dir, tensorspec_utils.EXTRA_ASSETS_DIRECTORY,
            tensorspec_utils.T2R_ASSETS_FILENAME)
        t2r_assets = tensorspec_utils.load_t2r_assets_to_file(
            t2r_assets_file_path)
        self._feature_spec = tensorspec_utils.TensorSpecStruct.from_proto(
            t2r_assets.feature_spec)  # pytype: disable=wrong-arg-types
        self._label_spec = tensorspec_utils.TensorSpecStruct.from_proto(
            t2r_assets.label_spec)  # pytype: disable=wrong-arg-types

        if t2r_assets.HasField('global_step'):
          self._global_step = t2r_assets.global_step
        else:
          logging.warning(
              'Error loading the global step, therefore using the previously'
              'set global step %s.', str(self.global_step))

        self._predict_fn = contrib_predictor.from_saved_model(
            model_dir, config=self._tf_config)
        model_global_step = self._predict_fn.session.run(
            self._predict_fn.graph.get_collection(tf.GraphKeys.GLOBAL_STEP))[0]
        if (model_global_step is not None and
            model_global_step != self._global_step):
          logging.warning(
              'Using the global step loaded from the model %s and not the '
              'one from the assets file %s.', str(model_global_step),
              str(self._global_step))
          self._global_step = model_global_step
        return True
      except ValueError as err:
        logging.warning(
            'Error loading model as %s:\n%s\nThe next attempt at loading the '
            'latest model will be in %d seconds', model_dir, err,
            _BUSY_WAITING_SLEEP_TIME_IN_SECS)
      if time.time() - start_time_loading < self._timeout:
        break
      # Since a checkpoint might be written by the tf model concurrently
      # this is a busy waiting loop.
      time.sleep(_BUSY_WAITING_SLEEP_TIME_IN_SECS)
    logging.warning(
        'The checkpoint at %s could not be loaded after '
        '%s seconds.', str(self._latest_export_dir), str(self._timeout))
    return False

  def close(self):
    """Closes all open handles used throughout model evaluation."""
    # NoOp for this predictor.

  def assert_is_loaded(self):
    """Raises a ValueError if model parameters haven't been loaded.

    If restore() has been called with is_async=True, the thread will be joined
    before the check is performed.
    """
    self._maybe_join_restore_thread()
    if self._predict_fn is None:
      raise ValueError('The predictor has not yet been successfully restored.')

  @property
  def model_version(self):
    """The version of the model currently in use.

    Returns:
      The timestamp since Unix Epoch which allows to uniquely identify the
      exported saved model.
    """
    self.assert_is_loaded()
    return int(os.path.basename(self._latest_export_dir))

  @property
  def global_step(self):
    """The global step of the model currently in use.

    Returns:
      The global step the model was exported at.
    """
    return self._global_step

  @property
  def model_path(self):
    """The path of the model currently in use."""
    self.assert_is_loaded()
    return self._latest_export_dir

  def _latest_valid_model_dirs(self, model_dirs):
    """Returns the latest valid exported saved model directories.

    The exported saved models directory names are numbers (timestamp) which
    monotonically increase, meaning the largest directory name will contain
    the latest exported model. Lexicographical sorting will maintain this
    order. However, while exporting models, a temporary directory is generated
    prefixed with temp- which needs to be removed. Note, in case a writer
    crashes this directory might persist which is why we strip all invalid
    directory names starting from the back.

    Args:
      model_dirs: A list of all discovered, exported model dirs with the
        timestamp as the directory name.

    Returns:
      model_dirs: All discovered model dirs which are infact numbers and are
        populated with the model.
    """

    def _isvalid(model_dir):
      model_dir_is_numeric = os.path.basename(model_dir).isdigit()
      model_exists = tf.io.gfile.exists(
          os.path.join(model_dir,
                       tf.saved_model.constants.SAVED_MODEL_FILENAME_PB))

      assets_exists = tf.io.gfile.exists(
          os.path.join(model_dir, tensorspec_utils.EXTRA_ASSETS_DIRECTORY))
      return model_dir_is_numeric and model_exists and assets_exists

    # The exported saved models directory names are numbers (timestamp) which
    # monotonically increase, meaning the largest directory name will contain
    # the latest exported model. Lexicographical sorting will maintain this
    # order.
    for model_dir in sorted(model_dirs, key=os.path.basename, reverse=True):
      logging.info('Checking %s for valid model.', model_dir)
      if _isvalid(model_dir):
        return model_dir
    else:
      logging.info('No valid model found in %d directories.', len(model_dirs))

  def _maybe_join_restore_thread(self):
    if self._restore_thread:
      logging.info('Joining thread that restores model parameters.')
      self._restore_thread.join()
      self._restore_thread = None
