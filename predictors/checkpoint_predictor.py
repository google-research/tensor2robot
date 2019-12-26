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

"""Predictor which instantiates a model from a checkpoint."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import time

from absl import logging
import gin
import numpy as np
from tensor2robot.models import abstract_model
from tensor2robot.predictors import abstract_predictor
from tensor2robot.utils import tensorspec_utils
import tensorflow.compat.v1 as tf  # tf
from typing import Dict, Optional, Text


_BUSY_WAITING_SLEEP_TIME_IN_SECS = 1


@gin.configurable
class CheckpointPredictor(abstract_predictor.AbstractPredictor):
  """A predictor using model checkpoints."""

  def __init__(
      self,
      t2r_model,
      checkpoint_dir = None,
      use_gpu = True,
      timeout = 600,
      tf_intra_op_parallelism_threads = 4,
      tf_inter_op_parallelism_threads = 4):
    """Load the model from registry and build the model in a new tf graph.

    Args:
      t2r_model: A T2RModel instance.
      checkpoint_dir: The directory to find the checkpoint. If set to `None`, no
        checkpoint will be loaded and if init_with_random_variables is set to
        True a random model is initialized. Note, either checkpoint_dir or
        init_with_random_variable has to be set but not both.
      use_gpu: If True, will attempt to use GPU for inference.
      timeout: (defaults to 600 seconds) If no checkpoint has been found after
        timeout seconds restore fails.
      tf_intra_op_parallelism_threads: see `tf.ConfigProto`
      tf_inter_op_parallelism_threads: see `tf.ConfigProto`
    """
    self._checkpoint_dir = checkpoint_dir
    self._timeout = timeout

    # As done in model_inference.py, a separate graph is used to build the
    # target network.
    g = tf.Graph()
    mode = tf.estimator.ModeKeys.PREDICT
    with g.as_default():
      preprocessor = t2r_model.preprocessor
      feature_tspec = preprocessor.get_in_feature_specification(mode)
      # We perform inference, hence we only want the required tensors.
      self._feature_tspec = tensorspec_utils.filter_required_flat_tensor_spec(
          feature_tspec)
      label_tspec = preprocessor.get_in_feature_specification(mode)
      self._label_tspec = tensorspec_utils.filter_required_flat_tensor_spec(
          label_tspec)

      self._features = tensorspec_utils.make_placeholders(
          self._feature_tspec, batch_size=None)

      preprocessed_features, _ = preprocessor.preprocess(
          features=self._features, labels=None, mode=mode)
      estimator_spec = t2r_model.model_fn(preprocessed_features, None, mode)
      self._predictions = estimator_spec.predictions
      config = tf.ConfigProto(
          device_count={'GPU': 1 if use_gpu else 0},
          intra_op_parallelism_threads=tf_intra_op_parallelism_threads,
          inter_op_parallelism_threads=tf_inter_op_parallelism_threads)
      self._sess = tf.Session(graph=g, config=config)
      self._t2r_model = t2r_model
      # The location of the last checkpoint loaded.
      self._current_checkpoint_path = None
      self._tf_global_step = tf.train.get_or_create_global_step()
      # The PREDICT graph is generated which contains only the model specific
      # variables and not training specific variables, e.g. Adam, Momentum.
      var_list = contrib_framework.get_variables()
      self._saver = tf.train.Saver(var_list=var_list)
      # Default init op in case init_randomly is called.
      self._global_init_op = tf.global_variables_initializer()

    self._model_was_restored = False

  def predict(self, features):
    """Predicts based on feature input using the loaded model.

    Args:
      features: A dict containing the features used for predictions.
    Returns:
      The result of the queried model predictions.
    """
    self.assert_is_loaded()
    return self._sess.run(
        self._predictions,
        tensorspec_utils.map_feed_dict(self._features, features))

  def get_feature_specification(self):
    """Exposes the required input features for evaluation of the model."""
    return self._feature_tspec

  def get_label_specification(self
                             ):
    """Exposes the optional labels for evaluation of the model."""
    return self._label_tspec

  def init_randomly(self):
    """Initializes model parameters from with random values."""
    self._model_was_restored = True
    logging.info('Initializing model with random weights')
    self._sess.run(self._global_init_op)

  def restore(self):
    """Restores the model parameters from the latest available data.

    Raises:
      ValueError: If no checkpoint_dir has been provided.

    Returns:
      True if the model is loaded from a new checkpoint or when already
      loaded the model has not been updated and False otherwise.
    """
    if self._checkpoint_dir is None:
      raise ValueError(
          'The predictor cannot be restored since no checkpoint_dir has been'
          'passed.')
    # If we don't have any checkpoint and have not initialized with random
    # weights, wait for checkpoint indefinitely, other just try to load the new
    # checkpoint if it's currently available.
    logging.info(
        'About to wait_for_new_checkpoint with checkpoint_dir: %s '
        'current checkpoint_path: %s and timeout: %s', self._checkpoint_dir,
        self._current_checkpoint_path, self._timeout)

    start_time = time.time()
    latest_checkpoint = None
    while (time.time() - start_time < self._timeout and
           latest_checkpoint is None):
      latest_checkpoint = tf.train.latest_checkpoint(self._checkpoint_dir)
      if latest_checkpoint is None:
        logging.warning(
            'No checkpoint found at %s:\nThe next attempt to check for '
            'latest model will be in %d seconds', self._checkpoint_dir,
            _BUSY_WAITING_SLEEP_TIME_IN_SECS)
        time.sleep(_BUSY_WAITING_SLEEP_TIME_IN_SECS)

    if latest_checkpoint is None:
      return False

    if latest_checkpoint == self._current_checkpoint_path:
      logging.info('Checkpoint \'%s\' wasn\'t updated.', latest_checkpoint)
      return True

    self._saver.restore(self._sess, latest_checkpoint)
    self._current_checkpoint_path = latest_checkpoint

    self._model_was_restored = True
    return True

  def close(self):
    """Closes all open handles used throughout model evaluation."""
    self._sess.close()
    tf.reset_default_graph()
    self._model_was_restored = False

  def assert_is_loaded(self):
    """Raises a ValueError if the predictor has not been restored yet."""
    if not self._model_was_restored:
      raise ValueError('The predictor has not yet been successfully restored.')

  @property
  def model_version(self):
    """The version of the model currently in use."""
    return self.global_step

  @property
  def global_step(self):
    """The global step of the model currently in use."""
    try:
      # If a model has not been loaded the global step of the model
      # is not valid which is why we return -1.
      self.assert_is_loaded()
    except ValueError:
      return -1
    return self._sess.run(self._tf_global_step)

  @property
  def model_path(self):
    """The path of the model currently in use."""
    self.assert_is_loaded()
    return self._current_checkpoint_path
