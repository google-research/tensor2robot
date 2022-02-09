# coding=utf-8
# Copyright 2022 The Tensor2Robot Authors.
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
"""Tests for tensor2robot.export_generator.sabstract_export_generator."""

from six.moves import zip
from tensor2robot.export_generators import abstract_export_generator
from tensor2robot.preprocessors import noop_preprocessor
from tensor2robot.utils import mocks
import tensorflow.compat.v1 as tf
from tensorflow_serving.apis import prediction_log_pb2


class AbstractExportGeneratorTest(tf.test.TestCase):

  def test_init_abstract(self):
    with self.assertRaises(TypeError):
      abstract_export_generator.AbstractExportGenerator()

  def test_create_warmup_requests_numpy(self):
    mock_t2r_model = mocks.MockT2RModel(
        preprocessor_cls=noop_preprocessor.NoOpPreprocessor)
    exporter = mocks.MockExportGenerator()
    exporter.set_specification_from_model(mock_t2r_model)

    export_dir = self.create_tempdir()
    batch_sizes = [2, 4]
    request_filename = exporter.create_warmup_requests_numpy(
        batch_sizes=batch_sizes, export_dir=export_dir.full_path)

    for expected_batch_size, record in zip(
        batch_sizes, tf.compat.v1.io.tf_record_iterator(request_filename)):
      record_proto = prediction_log_pb2.PredictionLog()
      record_proto.ParseFromString(record)
      request = record_proto.predict_log.request
      self.assertEqual(request.model_spec.name, 'MockT2RModel')
      for _, in_tensor in request.inputs.items():
        self.assertEqual(in_tensor.tensor_shape.dim[0].size,
                         expected_batch_size)


if __name__ == '__main__':
  tf.test.main()
