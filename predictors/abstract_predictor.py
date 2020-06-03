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

# Lint as python3
"""An abstract predictor to load tf models and expose a predict function."""

import abc
from typing import Dict, Optional, Text

import numpy as np
import six
from tensor2robot.utils import tensorspec_utils


class AbstractPredictor(six.with_metaclass(abc.ABCMeta, object)):
  """A predictor responsible to load a T2RModel and expose a predict function.

  The purpose of the predictor is to abstract model loading and running, e.g.
  using a raw session interface, a tensorflow predictor created from saved
  models or tensorflow 2.0 models.
  """

  @abc.abstractmethod
  def predict(self, features):
    """Predicts based on feature input using the loaded model.

    Args:
      features: A dict containing the features used for predictions.
    Returns:
      The result of the queried model predictions.
    """

  @abc.abstractmethod
  def get_feature_specification(self):
    """Exposes the required input features for evaluation of the model."""

  def get_label_specification(self
                             ):
    """Exposes the optional labels for evaluation of the model."""
    return None

  @abc.abstractmethod
  def restore(self):
    """Restores the model parameters from the latest available data."""

  def init_randomly(self):
    """Initializes model parameters from with random values."""

  @abc.abstractmethod
  def close(self):
    """Closes all open handles used throughout model evaluation."""

  @abc.abstractmethod
  def assert_is_loaded(self):
    """Raises a ValueError if the predictor has not been restored yet."""

  @property
  def model_version(self):
    """The version of the model currently in use."""
    return 0

  @property
  def global_step(self):
    """The global step of the model currently in use."""
    return 0

  @property
  def model_path(self):
    """The path of the model currently in use."""
    return ''
