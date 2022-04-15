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

"""Tests for robotics.learning.estimator_models.meta_learning.preprocessors."""

import functools

from absl.testing import parameterized
import numpy as np
import six
from six.moves import range
from tensor2robot.meta_learning import preprocessors
from tensor2robot.preprocessors import abstract_preprocessor
from tensor2robot.utils import tensorspec_utils as utils
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
from tensorflow.contrib import framework as contrib_framework

nest = contrib_framework.nest

TSpec = utils.TensorSpecStruct

_RANDOM_SEED = 1234
_DEFAULT_IN_IMAGE_SHAPE = (640, 512, 3)
_DEFAULT_OUT_IMAGE_SHAPE = (256, 320, 3)
_DEFAULT_ACTION_SHAPE = (5,)
_DEFAULT_REWARD_SHAPE = (1,)


class MockBasePreprocessor(abstract_preprocessor.AbstractPreprocessor):

  def get_in_feature_specification(self, mode):
    del mode
    feature_spec = TSpec()
    feature_spec.image = utils.ExtendedTensorSpec(
        shape=_DEFAULT_IN_IMAGE_SHAPE,
        dtype=tf.uint8,
        is_optional=False,
        data_format='jpeg',
        name='state/image')
    feature_spec.action = utils.ExtendedTensorSpec(
        shape=_DEFAULT_ACTION_SHAPE,
        dtype=tf.float32,
        is_optional=False,
        name='state/action')
    return feature_spec

  def get_out_feature_specification(self, mode):
    del mode
    feature_spec = TSpec()
    feature_spec.image = utils.ExtendedTensorSpec(
        shape=_DEFAULT_OUT_IMAGE_SHAPE,
        dtype=tf.float32,
        is_optional=False,
        name='state/image')
    feature_spec.original_image = utils.ExtendedTensorSpec(
        shape=_DEFAULT_IN_IMAGE_SHAPE, dtype=tf.float32, is_optional=True)
    feature_spec.action = utils.ExtendedTensorSpec(
        shape=_DEFAULT_ACTION_SHAPE,
        dtype=tf.float32,
        is_optional=False,
        name='state/action')
    return feature_spec

  def get_in_label_specification(self, mode):
    del mode
    label_spec = TSpec()
    label_spec.reward = utils.ExtendedTensorSpec(
        shape=_DEFAULT_REWARD_SHAPE,
        dtype=tf.float32,
        is_optional=False,
        name='reward')
    return label_spec

  def get_out_label_specification(self, mode):
    del mode
    label_spec = TSpec()
    label_spec.reward = utils.ExtendedTensorSpec(
        shape=_DEFAULT_REWARD_SHAPE,
        dtype=tf.float32,
        is_optional=False,
        name='reward')
    return label_spec

  def _preprocess_fn(self, features, labels, mode):
    features.original_image = tf.image.convert_image_dtype(
        features.image, tf.float32)
    features.image = tf.image.resize_bilinear(
        features.original_image,
        size=self.get_out_feature_specification(mode).image.shape[:2])
    return features, labels


class PreprocessorsTest(tf.test.TestCase, parameterized.TestCase):

  def _create_mock_tensors(self,
                           base_preprocessor,
                           batch_size,
                           mode=tf_estimator.ModeKeys.TRAIN):
    np.random.seed(_RANDOM_SEED)
    features = utils.make_random_numpy(
        base_preprocessor.get_in_feature_specification(mode),
        batch_size=batch_size)
    labels = utils.make_random_numpy(
        base_preprocessor.get_in_label_specification(mode),
        batch_size=batch_size)
    return (features, labels)

  def _init_mock(self, batch_size, mode=tf_estimator.ModeKeys.TRAIN):
    base_preprocessor = MockBasePreprocessor()
    maml_preprocessor = preprocessors.MAMLPreprocessorV2(
        base_preprocessor=MockBasePreprocessor())
    mock_tensors = self._create_mock_tensors(base_preprocessor, batch_size,
                                             mode)
    return maml_preprocessor, mock_tensors

  @parameterized.parameters((1, 1), (1, 2), (2, 1), (2, 2))
  def test_maml_preprocessor_v2_meta_map_fn_raises(
      self, num_condition_samples_per_task, num_inference_samples_per_task):
    batch_size = (
        num_condition_samples_per_task + num_inference_samples_per_task)
    init_mock = self._init_mock(2 * batch_size)
    maml_preprocessor, mock_tensors = init_mock

    # Create a failure case for not enough data in the batch.
    dataset = tf.data.Dataset.from_tensor_slices(mock_tensors)
    # Note, if drop_remainder = False, the resulting dataset has no static
    # shape which is required for the meta preprocessing.
    dataset = dataset.batch(batch_size - 1, drop_remainder=True)

    # Trigger raise conditions for create_meta_map_fn due to
    # num_*_samples_per_task being None or not > 0.
    with self.assertRaises(ValueError):
      map_fn = maml_preprocessor.create_meta_map_fn(
          None, num_inference_samples_per_task)
    with self.assertRaises(ValueError):
      map_fn = maml_preprocessor.create_meta_map_fn(
          num_condition_samples_per_task, None)
    with self.assertRaises(ValueError):
      map_fn = maml_preprocessor.create_meta_map_fn(
          -num_condition_samples_per_task, num_inference_samples_per_task)
    with self.assertRaises(ValueError):
      map_fn = maml_preprocessor.create_meta_map_fn(
          num_condition_samples_per_task, -num_inference_samples_per_task)

    map_fn = maml_preprocessor.create_meta_map_fn(
        num_condition_samples_per_task, num_inference_samples_per_task)
    with self.assertRaises(ValueError):
      dataset.map(map_func=map_fn, num_parallel_calls=1)

    # Create a failure case for too many examples in a batch.
    dataset = tf.data.Dataset.from_tensor_slices(mock_tensors)
    # Note, if drop_remainder = False, the resulting dataset has no static
    # shape which is required for the meta preprocessing.
    dataset = dataset.batch(batch_size + 1, drop_remainder=True)
    map_fn = maml_preprocessor.create_meta_map_fn(
        num_condition_samples_per_task, num_inference_samples_per_task)
    with self.assertRaises(ValueError):
      dataset.map(map_func=map_fn, num_parallel_calls=1)

    # Create a failure case because the batch_size is not known at graph
    # construction time.
    dataset = tf.data.Dataset.from_tensor_slices(mock_tensors)
    # Note, if drop_remainder = False, the resulting dataset has no static
    # shape which is required for the meta preprocessing.
    dataset = dataset.batch(batch_size + 1, drop_remainder=False)
    map_fn = maml_preprocessor.create_meta_map_fn(
        num_condition_samples_per_task, num_inference_samples_per_task)
    with self.assertRaises(ValueError):
      dataset.map(map_func=map_fn, num_parallel_calls=1)

  @parameterized.parameters((1, 1), (1, 2), (2, 1), (2, 2))
  def test_maml_preprocessor_v2_meta_map_fn(
      self, num_condition_samples_per_task, num_inference_samples_per_task):
    batch_size = (
        num_condition_samples_per_task + num_inference_samples_per_task)
    init_mock = self._init_mock(2 * batch_size)
    maml_preprocessor, mock_tensors = init_mock

    with self.session() as sess:
      dataset = tf.data.Dataset.from_tensor_slices(mock_tensors)
      # Note, if drop_remainder = False, the resulting dataset has no static
      # shape which is required for the meta preprocessing.
      dataset = dataset.batch(batch_size, drop_remainder=True)

      map_fn = maml_preprocessor.create_meta_map_fn(
          num_condition_samples_per_task, num_inference_samples_per_task)
      dataset = dataset.map(map_func=map_fn, num_parallel_calls=1)
      raw_meta_features, raw_meta_labels = dataset.make_one_shot_iterator(
      ).get_next()

      np_raw_meta_features, np_raw_meta_labels = sess.run(
          [raw_meta_features, raw_meta_labels])
      ref_features, ref_labels = mock_tensors

      self.assertEqual(
          list(np_raw_meta_features.condition.features.keys()),
          list(np_raw_meta_features.inference.features.keys()))
      for feature_name in np_raw_meta_features.condition.features.keys():
        np.testing.assert_array_almost_equal(
            np_raw_meta_features.condition.features[feature_name],
            ref_features[feature_name][:num_condition_samples_per_task])
        np.testing.assert_array_almost_equal(
            np_raw_meta_features.inference.features[feature_name],
            ref_features[feature_name]
            [num_condition_samples_per_task:batch_size])

      # The labels and the condition labels have to have the same keys.
      self.assertEqual(
          list(np_raw_meta_features.condition.labels.keys()),
          list(np_raw_meta_labels.keys()))
      for label_name in np_raw_meta_features.condition.labels.keys():
        np.testing.assert_array_almost_equal(
            np_raw_meta_features.condition.labels[label_name],
            ref_labels[label_name][:num_condition_samples_per_task])
        np.testing.assert_array_almost_equal(
            np_raw_meta_labels[label_name],
            ref_labels[label_name][num_condition_samples_per_task:batch_size])

  @parameterized.parameters((1, 1, 1), (1, 2, 2), (2, 1, 2), (1, 2, 3),
                            (2, 1, 3), (2, 2, 3))
  def test_maml_preprocessor_v2_preprocess(self, num_condition_samples_per_task,
                                           num_inference_samples_per_task,
                                           outer_batch_size):
    inner_batch_size = (
        num_condition_samples_per_task + num_inference_samples_per_task)
    init_mock = self._init_mock(outer_batch_size * inner_batch_size)
    maml_preprocessor, mock_tensors = init_mock

    with self.session() as sess:
      dataset = tf.data.Dataset.from_tensor_slices(mock_tensors)
      # Note, if drop_remainder = False, the resulting dataset has no static
      # shape which is required for the meta preprocessing.
      dataset = dataset.batch(inner_batch_size, drop_remainder=True)

      map_fn = maml_preprocessor.create_meta_map_fn(
          num_condition_samples_per_task, num_inference_samples_per_task)
      dataset = dataset.map(map_func=map_fn, num_parallel_calls=1)

      # Note, if drop_remainder = False, the resulting dataset has no static
      # shape which is required for the meta preprocessing.
      dataset = dataset.batch(outer_batch_size, drop_remainder=True)

      preprocess_fn = functools.partial(
          maml_preprocessor.preprocess, mode=tf_estimator.ModeKeys.TRAIN)
      dataset = dataset.map(map_func=preprocess_fn, num_parallel_calls=1)

      raw_meta_features, raw_meta_labels = dataset.make_one_shot_iterator(
      ).get_next()

      np_raw_meta_features, np_raw_meta_labels = sess.run(
          [raw_meta_features, raw_meta_labels])
      ref_features, ref_labels = mock_tensors

      self.assertEqual(
          list(np_raw_meta_features.condition.features.keys()),
          list(np_raw_meta_features.inference.features.keys()))

      # The image has been resized. Therefore, we ensure that its shape is
      # correct. Note, we have to strip the outer and inner batch dimensions.
      self.assertEqual(np_raw_meta_features.condition.features.image.shape[2:],
                       _DEFAULT_OUT_IMAGE_SHAPE)
      self.assertEqual(np_raw_meta_features.inference.features.image.shape[2:],
                       _DEFAULT_OUT_IMAGE_SHAPE)

      # The following tests are important to ensure that our reshaping,
      # flattening and unflattening actually preserves all information.

      # We can only test those two since the image has been resized.
      # Since the featurename has been altered during preprocessing we have
      # to index the reference data differently.
      # Further, we only test the first batch, since everything afterwards
      # would require more index slicing :).
      # For the image we have to convert the original data into float32 since
      # that is the required conversion for our preprocessor.
      np.testing.assert_array_almost_equal(
          np_raw_meta_features.condition.features['original_image'][0],
          ref_features['image'][:num_condition_samples_per_task].astype(
              np.float32) / 255)
      np.testing.assert_array_almost_equal(
          np_raw_meta_features.inference.features['original_image'][0],
          ref_features['image'][num_condition_samples_per_task:inner_batch_size]
          .astype(np.float32) / 255)

      np.testing.assert_array_almost_equal(
          np_raw_meta_features.condition.features['action'][0],
          ref_features['action'][:num_condition_samples_per_task])
      np.testing.assert_array_almost_equal(
          np_raw_meta_features.inference.features['action'][0],
          ref_features['action']
          [num_condition_samples_per_task:inner_batch_size])

      # The labels and the condition labels have to have the same keys.
      self.assertEqual(
          list(np_raw_meta_features.condition.labels.keys()),
          list(np_raw_meta_labels.keys()))
      for label_name in np_raw_meta_features.condition.labels.keys():
        np.testing.assert_array_almost_equal(
            np_raw_meta_features.condition.labels[label_name][0],
            ref_labels[label_name][:num_condition_samples_per_task])
        np.testing.assert_array_almost_equal(
            np_raw_meta_labels[label_name][0], ref_labels[label_name]
            [num_condition_samples_per_task:inner_batch_size])

  def test_create_metaexample_spec(self):
    feature_spec = TSpec()
    feature_spec.image = utils.ExtendedTensorSpec(
        shape=_DEFAULT_IN_IMAGE_SHAPE,
        dtype=tf.uint8,
        is_optional=False,
        data_format='jpeg',
        name='state/image')
    feature_spec.action = utils.ExtendedTensorSpec(
        shape=_DEFAULT_ACTION_SHAPE,
        dtype=tf.float32,
        is_optional=False,
        name='state/action')

    num_samples_in_task = 3
    metaexample_spec = preprocessors.create_metaexample_spec(
        feature_spec, num_samples_in_task, 'condition')

    flat_feature_spec = utils.flatten_spec_structure(feature_spec)
    self.assertLen(
        list(metaexample_spec.keys()),
        num_samples_in_task * len(list(flat_feature_spec.keys())))

    for key in flat_feature_spec:
      for i in range(num_samples_in_task):
        meta_example_key = six.ensure_str(key) + '/{:d}'.format(i)
        self.assertIn(meta_example_key, list(metaexample_spec.keys()))
        self.assertTrue(
            six.ensure_str(metaexample_spec[meta_example_key].name).startswith(
                'condition_ep'))

  def test_stack_intratask_episodes(self):
    feature_spec = TSpec()
    feature_spec.image = utils.ExtendedTensorSpec(
        shape=_DEFAULT_IN_IMAGE_SHAPE,
        dtype=tf.uint8,
        is_optional=False,
        data_format='jpeg',
        name='state/image')
    feature_spec.action = utils.ExtendedTensorSpec(
        shape=_DEFAULT_ACTION_SHAPE,
        dtype=tf.float32,
        is_optional=False,
        name='state/action')

    batch_size = 2
    num_samples_in_task = 3
    metaexample_spec = preprocessors.create_metaexample_spec(
        feature_spec, num_samples_in_task, 'condition')
    tensors = utils.make_random_numpy(metaexample_spec, batch_size)
    out_tensors = preprocessors.stack_intra_task_episodes(
        tensors, num_samples_in_task)

    self.assertEqual(
        out_tensors.image.shape,
        (batch_size, num_samples_in_task) + _DEFAULT_IN_IMAGE_SHAPE)
    self.assertEqual(
        out_tensors.action.shape,
        (batch_size, num_samples_in_task) + _DEFAULT_ACTION_SHAPE)

  @parameterized.parameters((1, 1, 1), (1, 2, 2), (2, 1, 2), (1, 2, 3),
                            (2, 1, 3), (2, 3, 1))
  def test_meta_example_preprocess(
      self,
      num_condition_samples_per_task,
      num_inference_samples_per_task,
      outer_batch_size):
    base_preprocessor = MockBasePreprocessor()
    meta_example_preprocessor = preprocessors.FixedLenMetaExamplePreprocessor(
        base_preprocessor=base_preprocessor,
        num_condition_samples_per_task=num_condition_samples_per_task,
        num_inference_samples_per_task=num_inference_samples_per_task)
    mock_tensors = self._create_mock_tensors(
        meta_example_preprocessor, outer_batch_size)

    with self.session() as sess:
      dataset = tf.data.Dataset.from_tensor_slices(mock_tensors)

      dataset = dataset.batch(outer_batch_size, drop_remainder=True)

      preprocess_fn = functools.partial(
          meta_example_preprocessor.preprocess,
          mode=tf_estimator.ModeKeys.TRAIN)
      dataset = dataset.map(map_func=preprocess_fn, num_parallel_calls=1)

      raw_meta_features, raw_meta_labels = (
          dataset.make_one_shot_iterator().get_next())
      np_raw_meta_features, np_raw_meta_labels = sess.run(
          [raw_meta_features, raw_meta_labels])
      ref_features, ref_labels = mock_tensors

      self.assertEqual(
          list(np_raw_meta_features.condition.features.keys()),
          list(np_raw_meta_features.inference.features.keys()))

      # The labels and the condition labels have to have the same keys.
      self.assertEqual(
          list(np_raw_meta_features.condition.labels.keys()),
          list(np_raw_meta_labels.keys()))

      # The image has been resized. Therefore, we ensure that its shape is
      # correct. Note, we have to strip the outer and inner batch dimensions.
      self.assertEqual(
          np_raw_meta_features.condition.features.image.shape[2:],
          _DEFAULT_OUT_IMAGE_SHAPE)
      self.assertEqual(
          np_raw_meta_features.inference.features.image.shape[2:],
          _DEFAULT_OUT_IMAGE_SHAPE)

      for i in range(num_condition_samples_per_task):
        np.testing.assert_array_almost_equal(
            np_raw_meta_features.condition.features['action'][:, i, Ellipsis],
            ref_features['condition/features/action/{:d}'.format(i)])
        for label_name in np_raw_meta_features.condition.labels.keys():
          np.testing.assert_array_almost_equal(
              np_raw_meta_features.condition.labels[label_name][:, i, Ellipsis],
              ref_features['condition/labels/{:s}/{:d}'.format(
                  label_name, i)])

      for i in range(num_inference_samples_per_task):
        np.testing.assert_array_almost_equal(
            np_raw_meta_features.inference.features['action'][:, i, Ellipsis],
            ref_features['inference/features/action/{:d}'.format(i)])
        for label_name in np_raw_meta_features.condition.labels.keys():
          np.testing.assert_array_almost_equal(
              np_raw_meta_labels[label_name][:, i, Ellipsis],
              ref_labels[six.ensure_str(label_name) + '/{:d}'.format(i)])


if __name__ == '__main__':
  tf.test.main()
