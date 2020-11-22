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
"""Hook that logs golden values to be used in unit tests.

In the Data -> Checkpoint -> Inference -> Eval flow, this verifies no regression
occurred in Data -> Checkpoint.
"""

import os
from typing import List
from absl import logging
import gin
import numpy as np
from tensor2robot.hooks import hook_builder
from tensor2robot.models import model_interface
import tensorflow.compat.v1 as tf

ModeKeys = tf.estimator.ModeKeys
COLLECTION = 'golden'
PREFIX = 'golden_'


def add_golden_tensor(tensor, name):
  """Adds tensor to be tracked."""
  tf.add_to_collection(COLLECTION, tf.identity(tensor, name=PREFIX + name))


class GoldenValuesHook(tf.train.SessionRunHook):
  """SessionRunHook that saves loss metrics to file."""

  def __init__(self,
               log_directory):
    self._log_directory = log_directory

  def begin(self):
    self._measurements = []

  def end(self, session):
    # Record measurements.
    del session
    np.save(os.path.join(self._log_directory, 'golden_values.npy'),
            self._measurements)

  def before_run(self, run_context):
    return tf.train.SessionRunArgs(
        fetches=tf.get_collection_ref(COLLECTION))

  def after_run(self, run_context, run_values):
    # Strip the 'golden_' prefix before saving the data.
    golden_values = {t.name.split(PREFIX)[1]: v for t, v in
                     zip(tf.get_collection_ref(COLLECTION), run_values.results)}
    logging.info('Recorded golden values for %s', golden_values.keys())
    self._measurements.append(golden_values)


@gin.configurable
class GoldenValuesHookBuilder(hook_builder.HookBuilder):
  """Hook builder for generating golden values."""

  def create_hooks(
      self,
      t2r_model,
      estimator,
  ):
    return [GoldenValuesHook(estimator.model_dir)]
