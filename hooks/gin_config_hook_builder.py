# coding=utf-8
# Copyright 2023 The Tensor2Robot Authors.
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

"""Builds hooks that write out the operative gin configuration.
"""

from typing import List

from absl import logging
import gin
from tensor2robot.hooks import hook_builder
from tensor2robot.models import model_interface
from tensorflow import estimator as tf_estimator


@gin.configurable
class GinConfigLoggerHook(tf_estimator.SessionRunHook):
  """A SessionRunHook that logs the operative config to stdout."""

  def __init__(self, only_once=True):
    self._only_once = only_once
    self._written_at_least_once = False

  def after_create_session(self, session=None, coord=None):
    """Logs Gin's operative config."""
    if self._only_once and self._written_at_least_once:
      return

    logging.info('Gin operative configuration:')
    for gin_config_line in gin.operative_config_str().splitlines():
      logging.info(gin_config_line)
    self._written_at_least_once = True


@gin.configurable
class OperativeGinConfigLoggerHookBuilder(hook_builder.HookBuilder):

  def create_hooks(
      self,
      t2r_model,
      estimator,
  ):
    return [GinConfigLoggerHook()]
