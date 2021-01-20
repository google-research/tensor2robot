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

"""Tests for learning.estimator_models.preprocessor.tpu_preprocessor_wrapper."""

from absl import flags
from tensor2robot.preprocessors import noop_preprocessor
from tensor2robot.preprocessors import tpu_preprocessor_wrapper
from tensor2robot.utils import tensorspec_utils
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

_FEATURE_SPEC_NO_CAST = tensorspec_utils.TensorSpecStruct()
_FEATURE_SPEC_NO_CAST.data_float_32 = tensorspec_utils.ExtendedTensorSpec(
    name='data_float32', dtype=tf.float32, shape=(1,))
_FEATURE_SPEC_NO_CAST.optional_value = tensorspec_utils.ExtendedTensorSpec(
    name='optional_value', dtype=tf.int32, shape=(1,), is_optional=True)
_LABEL_SPEC_NO_CAST = tensorspec_utils.TensorSpecStruct()
_LABEL_SPEC_NO_CAST.some_value = tensorspec_utils.ExtendedTensorSpec(
    name='data_string', dtype=tf.string, shape=(1,))
_LABEL_SPEC_NO_CAST.optional_value = tensorspec_utils.ExtendedTensorSpec(
    name='optional_value', dtype=tf.float32, shape=(1,), is_optional=True)

_FEATURE_SPEC_CAST = tensorspec_utils.TensorSpecStruct()
_FEATURE_SPEC_CAST.data_bfloat16 = tensorspec_utils.ExtendedTensorSpec(
    name='data_float32', dtype=tf.bfloat16, shape=(1,))
_FEATURE_SPEC_CAST.optional_value = tensorspec_utils.ExtendedTensorSpec(
    name='optional_value', dtype=tf.int32, shape=(1,), is_optional=True)
_LABEL_SPEC_CAST = tensorspec_utils.TensorSpecStruct()
_LABEL_SPEC_CAST.some_value = tensorspec_utils.ExtendedTensorSpec(
    name='data_string', dtype=tf.string, shape=(1,))
_LABEL_SPEC_CAST.optional_value = tensorspec_utils.ExtendedTensorSpec(
    name='optional_value', dtype=tf.bfloat16, shape=(1,), is_optional=True)

_MODE_TRAIN = tf.estimator.ModeKeys.TRAIN


class MockPreprocessor(noop_preprocessor.NoOpPreprocessor):

  def get_in_feature_specification(self, mode):
    feature_spec = tensorspec_utils.flatten_spec_structure(
        self._model_feature_specification_fn(mode))
    # This will raise since we only cast from tf.float32 to tf.bfloat16.
    feature_spec.data_bfloat16 = tensorspec_utils.ExtendedTensorSpec.from_spec(
        spec=feature_spec.data_bfloat16, dtype=tf.int32)
    return feature_spec


class TPUPreprocessorWrapperTest(tf.test.TestCase):

  def test_remove_optional(self):
    preprocessor = noop_preprocessor.NoOpPreprocessor(
        model_feature_specification_fn=lambda mode: _FEATURE_SPEC_NO_CAST,
        model_label_specification_fn=lambda mode: _LABEL_SPEC_NO_CAST)
    tpu_preprocessor = tpu_preprocessor_wrapper.TPUPreprocessorWrapper(
        preprocessor=preprocessor)
    self.assertDictEqual(
        tpu_preprocessor.get_in_feature_specification(_MODE_TRAIN),
        preprocessor.get_in_feature_specification(_MODE_TRAIN))
    self.assertDictEqual(
        tpu_preprocessor.get_in_label_specification(_MODE_TRAIN),
        preprocessor.get_in_label_specification(_MODE_TRAIN))
    out_feature_spec = tensorspec_utils.replace_dtype(
        preprocessor.get_out_feature_specification(_MODE_TRAIN),
        from_dtype=tf.float32,
        to_dtype=tf.bfloat16)
    del out_feature_spec['optional_value']
    self.assertDictEqual(
        tpu_preprocessor.get_out_feature_specification(_MODE_TRAIN),
        out_feature_spec)
    out_label_spec = tensorspec_utils.replace_dtype(
        preprocessor.get_out_label_specification(_MODE_TRAIN),
        from_dtype=tf.float32,
        to_dtype=tf.bfloat16)
    del out_label_spec['optional_value']
    self.assertDictEqual(
        tpu_preprocessor.get_out_label_specification(_MODE_TRAIN),
        out_label_spec)

  def test_cast_bfloat16_success(self):
    preprocessor = noop_preprocessor.NoOpPreprocessor(
        model_feature_specification_fn=lambda mode: _FEATURE_SPEC_CAST,
        model_label_specification_fn=lambda mode: _LABEL_SPEC_CAST)
    tpu_preprocessor = tpu_preprocessor_wrapper.TPUPreprocessorWrapper(
        preprocessor=preprocessor)

    # The spec structure elements with bfloat16 are converted to float32 within
    # the TPUPreprocessorWrapper such that we can create proper parser and
    # do CPU preprocessing.
    feature_spec = preprocessor.get_in_feature_specification(_MODE_TRAIN)
    feature_spec.data_bfloat16 = tensorspec_utils.ExtendedTensorSpec.from_spec(
        spec=feature_spec.data_bfloat16, dtype=tf.float32)
    label_spec = preprocessor.get_in_label_specification(_MODE_TRAIN)
    label_spec.optional_value = tensorspec_utils.ExtendedTensorSpec.from_spec(
        spec=label_spec.optional_value, dtype=tf.float32)
    self.assertDictEqual(
        tpu_preprocessor.get_in_feature_specification(_MODE_TRAIN),
        feature_spec)
    self.assertDictEqual(
        tpu_preprocessor.get_in_label_specification(_MODE_TRAIN), label_spec)

    out_feature_spec = preprocessor.get_out_feature_specification(_MODE_TRAIN)
    del out_feature_spec['optional_value']
    out_label_spec = preprocessor.get_out_label_specification(_MODE_TRAIN)
    del out_label_spec['optional_value']
    self.assertDictEqual(
        tpu_preprocessor.get_out_feature_specification(_MODE_TRAIN),
        out_feature_spec)
    self.assertDictEqual(
        tpu_preprocessor.get_out_label_specification(_MODE_TRAIN),
        out_label_spec)

    features = tensorspec_utils.make_placeholders(
        tpu_preprocessor.get_in_feature_specification(_MODE_TRAIN),
        batch_size=2)
    labels = tensorspec_utils.make_placeholders(
        tpu_preprocessor.get_in_label_specification(_MODE_TRAIN), batch_size=2)

    # Make sure features and labels are transformed correctly. Basically
    # float32 is replaced with bfloat16 for the specs which ask for bfloat16.
    out_features, out_labels = tpu_preprocessor.preprocess(
        features=features, labels=labels, mode=_MODE_TRAIN)
    for ref_key, ref_value in out_features.items():
      self.assertEqual(out_features[ref_key].dtype, ref_value.dtype)
    for ref_key, ref_value in out_labels.items():
      self.assertEqual(out_labels[ref_key].dtype, ref_value.dtype)

    # Make sure features without labels are transformed correctly. Basically
    # float32 is replaced with bfloat16 for the specs which ask for bfloat16.
    out_features, out_labels = tpu_preprocessor.preprocess(
        features=features, labels=None, mode=_MODE_TRAIN)
    self.assertIsNone(out_labels)
    for ref_key, ref_value in out_features.items():
      self.assertEqual(out_features[ref_key].dtype, ref_value.dtype)

  def test_cast_bfloat16_raises(self):
    # Note, this preprocessor will alter the input data type from float32 to
    # int32 which will trigger ValueError when attempting to cast.
    preprocessor = MockPreprocessor(
        model_feature_specification_fn=lambda mode: _FEATURE_SPEC_CAST,
        model_label_specification_fn=lambda mode: _LABEL_SPEC_CAST)
    tpu_preprocessor = tpu_preprocessor_wrapper.TPUPreprocessorWrapper(
        preprocessor=preprocessor)
    out_feature_spec = preprocessor.get_out_feature_specification(_MODE_TRAIN)
    del out_feature_spec['optional_value']
    out_label_spec = preprocessor.get_out_label_specification(_MODE_TRAIN)
    del out_label_spec['optional_value']
    self.assertDictEqual(
        tpu_preprocessor.get_out_feature_specification(_MODE_TRAIN),
        out_feature_spec)
    self.assertDictEqual(
        tpu_preprocessor.get_out_label_specification(_MODE_TRAIN),
        out_label_spec)

    features = tensorspec_utils.make_placeholders(
        tpu_preprocessor.get_in_feature_specification(_MODE_TRAIN),
        batch_size=2)
    labels = tensorspec_utils.make_placeholders(
        tpu_preprocessor.get_in_label_specification(_MODE_TRAIN), batch_size=2)

    # Make sure the exception gets triggered if features and labels are passed.
    with self.assertRaises(ValueError):
      tpu_preprocessor.preprocess(
          features=features, labels=labels, mode=_MODE_TRAIN)

    # Make sure the exception gets triggered if only features are passed.
    with self.assertRaises(ValueError):
      tpu_preprocessor.preprocess(
          features=features, labels=None, mode=_MODE_TRAIN)


if __name__ == '__main__':
  tf.test.main()
