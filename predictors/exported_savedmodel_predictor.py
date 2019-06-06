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

"""A predictor based on exported saved models."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import time

from absl import logging
import gin
import numpy as np
from tensor2robot.predictors import abstract_predictor
from tensor2robot.utils import tensorspec_utils
import tensorflow as tf  # tf
from typing import Dict, Callable, List, Text, Optional



_BUSY_WAITING_SLEEP_TIME_IN_SECS = 10


@gin.configurable
class ExportedSavedModelPredictor(abstract_predictor.AbstractPredictor):
  """A predictor loading from exported saved models."""

  def __init__(self,
               export_dir,
               timeout = 600,
               tf_config = None):
    """Creates an instance.

    Args:
      export_dir: A path to a directory containing exported saved models in
        their respective directories. All directories are assumed to contain a
        saved model and the last (lexicographical sorting) represents the best
        model.
      timeout: (defaults to 600 seconds) If no checkpoint has been found after
        timeout seconds restore fails.
      tf_config: The tf.ConfigProto used to configure the TensorFlow session.
    """
    super(ExportedSavedModelPredictor, self).__init__()
    self._export_dir = export_dir
    self._timeout = timeout
    self._latest_export_dir = None
    self._predict_fn = None  # type: Callable
    self._feature_spec = None  # type: tensorspec_utils.TensorSpecStruct
    self._label_spec = None
    self._tf_config = tf_config

  def predict(self, features):
    """Predicts based on feature input using the loaded model.

    Args:
      features: A dict containing the features used for predictions.
    Returns:
      The result of the queried model predictions.
    """
    self.assert_is_loaded()
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

  def restore(self):
    """Restores the model parameters from the latest available data.

    Raises:
      ValueError: If no checkpoint can be found or loaded within the user
        defined timeout.

    Returns:
      True if a exported saved model has been loaded and False otherwise.
    """
    start_time = time.time()

    while time.time() - start_time < self._timeout:
      # The exported saved models directory names are numbers (timestamp) which
      # monotonically increase, meaning the largest directory name will contain
      # the latest exported model. Lexicographical sorting will maintain this
      # order.
      model_dirs = sorted(tf.io.gfile.glob(os.path.join(self._export_dir, '*')))
      model_dirs = self._remove_invalid_model_dirnames(model_dirs)

      if len(model_dirs) >= 1:
        logging.info('Found latest model at %s. ', model_dirs[-1])
        break

      logging.info('Waiting for an exported model to become available at %s.',
                   self._export_dir)
      # Since a checkpoint might not be available and this is a busy waiting
      # loop, we throttle checking for checkpoints.
      time.sleep(_BUSY_WAITING_SLEEP_TIME_IN_SECS)

    if model_dirs is None or not model_dirs:
      logging.warning('No checkpoint available after %s seconds.',
                      str(self._timeout))
      return False

    if self._latest_export_dir == model_dirs[-1]:
      # The latest model has already been loaded.
      return True

    logging.info('Loading the latest model at %s. ', model_dirs[-1])
    self._latest_export_dir = model_dirs[-1]
    start_time_loading = time.time()

    # Note, loading from a saved model might require several attempts if
    # the checkpoint gets written asynchronously.
    while time.time() - start_time_loading < self._timeout:
      try:
        self._predict_fn = tf.contrib.predictor.from_saved_model(
            model_dirs[-1], config=self._tf_config)

        # Load input specs from file.
        input_spec_filename = os.path.join(model_dirs[-1], 'assets.extra',
                                           'input_specs.pkl')
        self._feature_spec, self._label_spec = (
            tensorspec_utils.load_from_file(input_spec_filename))
        return True
      except ValueError as err:
        logging.warning(
            'Error loading model as %s:\n%s\nThe next attempt at loading the '
            'latest model will be in %d seconds', model_dirs[-1], err,
            _BUSY_WAITING_SLEEP_TIME_IN_SECS)
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
  def model_path(self):
    """The path of the model currently in use."""
    self.assert_is_loaded()
    return self._latest_export_dir

  def _remove_invalid_model_dirnames(self,
                                     model_dirs):
    """Removes invalid exported saved model directory names.

    The exported saved models directory names are numbers (timestamp) which
    monotonically increase, meaning the largest directory name will contain
    the latest exported model. Lexicographical sorting will maintain this
    order. However, while exporting models, a temporary directory is generated
    prefixed with temp- which needs to be removed. Note, in case a writer
    crashes this directory might persist which is why we strip all invalid
    directory names starting from the back.

    Args:
      model_dirs: A list of all discovered, lexicographical sorted exported
        model dirs.

    Returns:
      model_dirs: All discovered model dirs which are infact numbers.
    """

    def _isvalid(model_dir):
      return os.path.basename(model_dir).isdigit()

    return filter(_isvalid, model_dirs)
