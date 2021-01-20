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
"""Binary for training TFModels with Estimator API."""

from absl import app
from absl import flags
import gin
from tensor2robot.utils import train_eval
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS


def main(unused_argv):
  gin.parse_config_files_and_bindings(
      FLAGS.gin_configs, FLAGS.gin_bindings, print_includes_and_imports=True)
  train_eval.train_eval_model()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
