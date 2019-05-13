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

"""Tests for learning.estimator_models.meta_learning.meta_tf_models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2robot.meta_learning import meta_tf_models
from tensor2robot.preprocessors import abstract_preprocessor
from tensor2robot.utils import tensorspec_utils
import tensorflow as tf


class MockBasePreprocessor(abstract_preprocessor.AbstractPreprocessor):

  def _get_feature_specification(self):
    spec = tensorspec_utils.TensorSpecStruct()
    spec.action = tensorspec_utils.ExtendedTensorSpec(
        name='action', shape=(1,), dtype=tf.float32)
    spec.velocity = tensorspec_utils.ExtendedTensorSpec(
        name='velocity', shape=(1,), dtype=tf.float32, is_optional=True)
    return spec

  def get_in_feature_specification(self):
    return self._get_feature_specification()

  def get_out_feature_specification(self):
    return self._get_feature_specification()

  def _get_label_specification(self):
    spec = tensorspec_utils.TensorSpecStruct()
    spec.target = tensorspec_utils.ExtendedTensorSpec(
        name='target', shape=(1,), dtype=tf.float32)
    spec.proxy = tensorspec_utils.ExtendedTensorSpec(
        name='proxy', shape=(1,), dtype=tf.float32, is_optional=True)
    return spec

  def get_in_label_specification(self):
    return self._get_label_specification()

  def get_out_label_specification(self):
    return self._get_label_specification()

  def _preprocess_fn(self, features, labels, unused_mode):
    return features, labels


class MetaTfModelsTest(tf.test.TestCase):

  def test_meta_preprocessor_required_specs(self):
    meta_preprocessor = meta_tf_models.MetaPreprocessor(
        base_preprocessor=MockBasePreprocessor(),
        num_train_samples_per_task=1,
        num_val_samples_per_task=1)
    ref_feature_spec = meta_preprocessor.get_in_feature_specification()
    filtered_feature_spec = tensorspec_utils.filter_required_flat_tensor_spec(
        meta_preprocessor.get_in_feature_specification())
    self.assertDictEqual(ref_feature_spec, filtered_feature_spec)

    ref_label_spec = meta_preprocessor.get_in_label_specification()
    filtered_label_spec = tensorspec_utils.filter_required_flat_tensor_spec(
        meta_preprocessor.get_in_label_specification())
    self.assertDictEqual(ref_label_spec, filtered_label_spec)


if __name__ == '__main__':
  tf.test.main()
