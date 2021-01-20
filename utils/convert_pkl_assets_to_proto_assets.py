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

"""Convert existing pickle based assets to t2r_pb2 based assets."""

import os
from typing import Text

from absl import app
from absl import flags

from tensor2robot.proto import t2r_pb2
from tensor2robot.utils import tensorspec_utils
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS

flags.DEFINE_string('assets_filepath', None,
                    'The path to the exported savedmodel assets directory.')


def convert(assets_filepath):
  """Converts existing asset pickle based files to t2r proto based assets."""

  t2r_assets = t2r_pb2.T2RAssets()
  input_spec_filepath = os.path.join(assets_filepath, 'input_specs.pkl')
  if not tf.io.gfile.exists(input_spec_filepath):
    raise ValueError('No file exists for {}.'.format(input_spec_filepath))
  feature_spec, label_spec = tensorspec_utils.load_input_spec_from_file(
      input_spec_filepath)

  t2r_assets.feature_spec.CopyFrom(feature_spec.to_proto())
  t2r_assets.label_spec.CopyFrom(label_spec.to_proto())

  global_step_filepath = os.path.join(assets_filepath, 'global_step.pkl')
  if tf.io.gfile.exists(global_step_filepath):
    global_step = tensorspec_utils.load_input_spec_from_file(
        global_step_filepath)
    t2r_assets.global_step = global_step

  t2r_assets_filepath = os.path.join(assets_filepath,
                                     tensorspec_utils.T2R_ASSETS_FILENAME)
  tensorspec_utils.write_t2r_assets_to_file(t2r_assets, t2r_assets_filepath)


def main(unused_argv):
  flags.mark_flag_as_required('assets_filepath')
  convert(FLAGS.assets_filepath)


if __name__ == '__main__':
  app.run(main)
