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

"""Tests for estimator_models.input_generators.abstract_input_generator."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import functools
from absl import flags
from tensor2robot.input_generators import abstract_input_generator
from tensor2robot.preprocessors import noop_preprocessor
from tensor2robot.utils import mocks
import tensorflow as tf


FLAGS = flags.FLAGS

BATCH_SIZE = 32


class AbstractInputGeneratorTest(tf.test.TestCase):

  def test_init_abstract(self):
    with self.assertRaises(TypeError):
      abstract_input_generator.AbstractInputGenerator()

  def test_set_preprocess_fn(self):
    mock_input_generator = mocks.MockInputGenerator(batch_size=BATCH_SIZE)
    preprocessor = noop_preprocessor.NoOpPreprocessor()
    with self.assertRaises(ValueError):
      # This should raise since we pass a function with `mode` not already
      # filled in either by a closure or functools.partial.
      mock_input_generator.set_preprocess_fn(preprocessor.preprocess)

    preprocess_fn = functools.partial(preprocessor.preprocess, labels=None)
    with self.assertRaises(ValueError):
      # This should raise since we pass a partial function but `mode`
      # is not abstracted away.
      mock_input_generator.set_preprocess_fn(preprocess_fn)


if __name__ == '__main__':
  tf.test.main()
