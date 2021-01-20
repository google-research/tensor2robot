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

# Lint as: python3
"""A hook to log all variables."""

from typing import Optional

from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import framework as contrib_framework


class VariableLoggerHook(tf.train.SessionRunHook):
  """A hook to log variables via a session run hook."""

  def __init__(self, max_num_variable_values = None):
    """Initializes a VariableLoggerHook.

    Args:
      max_num_variable_values: If not None, at most max_num_variable_values will
        be logged per variable.
    """
    super(VariableLoggerHook, self).__init__()
    self._max_num_variable_values = max_num_variable_values

  def begin(self):
    """Captures all variables to be read out during the session run."""
    self._variables_to_log = contrib_framework.get_variables()

  def before_run(self, run_context):
    """Adds the variables to the run args."""
    return tf.train.SessionRunArgs(self._variables_to_log)

  def after_run(self, run_context, run_values):
    del run_context
    original = np.get_printoptions()
    np.set_printoptions(suppress=True)
    for variable, variable_value in zip(self._variables_to_log,
                                        run_values.results):
      if not isinstance(variable_value, np.ndarray):
        continue
      variable_value = variable_value.ravel()
      logging.info('%s.mean = %s', variable.op.name, np.mean(variable_value))
      logging.info('%s.std = %s', variable.op.name, np.std(variable_value))
      if self._max_num_variable_values:
        variable_value = variable_value[:self._max_num_variable_values]
      logging.info('%s = %s', variable.op.name, variable_value)
    np.set_printoptions(**original)
