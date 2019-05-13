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

"""Interface to manage building hooks."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc

from tensor2robot.export_generators import abstract_export_generator
from tensor2robot.models import model_interface
import tensorflow as tf  # tf
from typing import List


class HookBuilder(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def create_hooks(
      self, t2r_model,
      estimator,
      export_generator
  ):
    """Create hooks for the trainer.

    Subclasses can add arguments here.

    Arguments:
      t2r_model: Provided model
      estimator: Provided estimator instance
      export_generator: An instance of an export_generator.
    Returns:
      A list of tf.train.SessionRunHooks to add to the trainer.
    """
