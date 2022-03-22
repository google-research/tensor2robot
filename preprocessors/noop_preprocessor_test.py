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

"""Tests for learning.estimator_models.preprocessors.noop_preprocessor."""

import collections

from absl.testing import parameterized

import numpy as np
from tensor2robot.preprocessors import noop_preprocessor
from tensor2robot.utils import tensorspec_utils
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

MockFeatures = collections.namedtuple(
    'MockFeatures', ['images', 'actions', 'optional_hierarchy'])
MockHierachy = collections.namedtuple('MockHierachy', ['debug_images'])

MockFeaturesRequired = collections.namedtuple('MockFeaturesRequired',
                                              ['images', 'actions'])

MockFeaturesBroken = collections.namedtuple('MockFeaturesBroken',
                                            ['images', 'actions', 'broken'])

MockLabels = collections.namedtuple('MockLabels', ['score'])

mock_features_required = MockFeaturesRequired(
    images=tensorspec_utils.ExtendedTensorSpec(
        shape=(224, 224, 3), dtype=tf.float32),
    actions=tensorspec_utils.ExtendedTensorSpec(shape=(6,), dtype=tf.float32),
)

mock_features_broken = MockFeaturesBroken(
    images=tensorspec_utils.ExtendedTensorSpec(
        shape=(224, 224, 3), dtype=tf.float32),
    actions=tensorspec_utils.ExtendedTensorSpec(shape=(6,), dtype=tf.float32),
    broken=tensorspec_utils.ExtendedTensorSpec(shape=(1,), dtype=tf.float32),
)


def mock_features_broken_fn(mode):
  del mode
  return mock_features_broken


mock_features = MockFeatures(
    images=mock_features_required.images,
    actions=mock_features_required.actions,
    optional_hierarchy=MockHierachy(
        debug_images=tensorspec_utils.ExtendedTensorSpec(
            shape=(224, 224, 3), dtype=tf.float32, is_optional=True)))


def mock_features_fn(mode):
  del mode
  return mock_features


mock_labels = MockLabels(
    score=tensorspec_utils.ExtendedTensorSpec(shape=(1,), dtype=tf.float32),)


def mock_labels_fn(mode):
  del mode
  return mock_labels


INVALID_SPEC_STRUCTURES = (('non_tensorspec_tensor_values_dict', {
    'test': 1,
}), ('non_tensorspec_tensor_values_named_tuple',
     MockFeaturesRequired(images=np.random.random_sample(10),
                          actions='action')))


class NoOpPreprocessorTest(parameterized.TestCase, tf.test.TestCase):

  def test_init_noop_preprocessor(self):
    noop_preprocessor.NoOpPreprocessor(mock_features_fn, mock_labels_fn)

  @parameterized.named_parameters(*INVALID_SPEC_STRUCTURES)
  def test_init_noop_preprocessor_raises(self, spec_or_tensors):
    spec_or_tensors_fn = lambda _: spec_or_tensors
    with self.assertRaises(ValueError):
      noop_preprocessor.NoOpPreprocessor(spec_or_tensors_fn, mock_labels_fn)
    with self.assertRaises(ValueError):
      noop_preprocessor.NoOpPreprocessor(mock_features_fn, spec_or_tensors_fn)

  def test_noop_preprocessor_preprocess_fn(self):

    def preprocess(preprocessor, feature_spec, label_spec, flatten=False):
      with tf.Session() as sess:
        feature_placeholders = tensorspec_utils.make_placeholders(
            feature_spec, batch_size=1)
        label_placeholders = None
        if label_spec is not None:
          label_placeholders = tensorspec_utils.make_placeholders(
              label_spec, batch_size=1)

        # Normally we want our features and labels to be flattened.
        # However we support not flattened hierarchies as well.
        if flatten:
          feature_placeholders = tensorspec_utils.flatten_spec_structure(
              feature_placeholders)
          if label_spec is not None:
            label_placeholders = tensorspec_utils.flatten_spec_structure(
                label_placeholders)

        (features_preprocessed, labels_preprocessed) = preprocessor.preprocess(
            features=feature_placeholders,
            labels=label_placeholders,
            mode=tf_estimator.ModeKeys.TRAIN)

        # We create a mapping of {key: np.array} or a namedtuple spec structure.
        np_feature_spec = tensorspec_utils.make_random_numpy(
            feature_spec, batch_size=1)
        if label_placeholders is not None:
          np_label_spec = tensorspec_utils.make_random_numpy(
              label_spec, batch_size=1)

        # We create our feed dict which basically consists of
        # {placeholders: np.array}.
        feed_dict = tensorspec_utils.map_feed_dict(feature_placeholders,
                                                   np_feature_spec,
                                                   ignore_batch=True)
        if label_placeholders is not None:
          feed_dict = tensorspec_utils.map_feed_dict(label_placeholders,
                                                     np_label_spec,
                                                     feed_dict,
                                                     ignore_batch=True)

        fetch_results = [features_preprocessed]
        if label_placeholders is not None:
          fetch_results.append(labels_preprocessed)

        np_preprocessed = sess.run(
            fetch_results, feed_dict=feed_dict)

        np_features_preprocessed = np_preprocessed[0]
        if label_placeholders is not None:
          np_labels_preprocessed = np_preprocessed[1]

        np_feature_spec = tensorspec_utils.flatten_spec_structure(
            np_feature_spec)
        if label_placeholders is not None:
          np_label_spec = tensorspec_utils.flatten_spec_structure(np_label_spec)

        for key, value in np_feature_spec.items():
          np.testing.assert_allclose(value, np_features_preprocessed[key])

        if label_placeholders is not None:
          for key, value in np_label_spec.items():
            np.testing.assert_allclose(value, np_labels_preprocessed[key])

    preprocessor = noop_preprocessor.NoOpPreprocessor(mock_features_fn,
                                                      mock_labels_fn)

    # Here we test that we can pass through our features, flattened and
    # unflattend.
    preprocess(preprocessor, mock_features, mock_labels, flatten=True)
    preprocess(preprocessor, mock_features, mock_labels, flatten=False)

    # Now we test that we can pass through the required subset.
    # Note, this really means the optional values are not provided and our
    # preprocessor does not complain.
    preprocess(
        preprocessor, mock_features_required, mock_labels, flatten=False)
    preprocess(
        preprocessor, mock_features_required, mock_labels, flatten=True)

    # Labels are not required.
    preprocess(
        preprocessor, mock_features_required, None, flatten=True)

    # Now we will make a new preprocessor with additional requirements which
    # should be broken since we have no features which fulfill the requirements.
    preprocessor = noop_preprocessor.NoOpPreprocessor(mock_features_broken_fn,
                                                      mock_labels_fn)
    with self.assertRaises(ValueError):
      preprocess(
          preprocessor, mock_features_required, mock_labels, flatten=False)
    with self.assertRaises(ValueError):
      preprocess(
          preprocessor, mock_features_required, mock_labels, flatten=True)


if __name__ == '__main__':
  tf.test.main()
