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

"""Runs data collection and policy evaluation for RL experiments.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import gin
from tensor2robot.utils import continuous_collect_eval
# Contains the gin_configs and gin_bindings flag definitions.
FLAGS = flags.FLAGS

try:
  flags.DEFINE_list(
      'gin_configs', None,
      'A comma-separated list of paths to Gin configuration files.')
  flags.DEFINE_multi_string(
      'gin_bindings', [], 'A newline separated list of Gin parameter bindings.')
except flags.DuplicateFlagError:
  pass

flags.DEFINE_string('root_dir', '',
                    'Root directory of experiment.')
flags.mark_flag_as_required('gin_configs')


def main(unused_argv):
  del unused_argv
  gin.parse_config_files_and_bindings(FLAGS.gin_configs, FLAGS.gin_bindings)
  continuous_collect_eval.collect_eval_loop(root_dir=FLAGS.root_dir)


if __name__ == '__main__':
  app.run(main)
