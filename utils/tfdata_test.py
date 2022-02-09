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
"""Tests for tensor2robot.utils.tfdata."""

import collections
import os
from absl.testing import parameterized
import numpy as np

from six.moves import range
from tensor2robot.utils import image
from tensor2robot.utils import tensorspec_utils
from tensor2robot.utils import tfdata
import tensorflow.compat.v1 as tf

FLAGS = tf.app.flags.FLAGS

PoseEnvFeature = collections.namedtuple(
    'PoseEnvFeature', ['state', 'action'])
PoseEnvLabel = collections.namedtuple(
    'PoseEnvLabel', ['reward'])

TSPEC = tensorspec_utils.ExtendedTensorSpec

NUM_FAKE_FILES = 3
TEST_IMAGE_SHAPE = [28, 28, 3]
TEST_IMAGE = np.ones(TEST_IMAGE_SHAPE, dtype=np.int32)


class TFDataTest(parameterized.TestCase, tf.test.TestCase):

  def _create_fake_file_pattern(self,
                                fake_filename_prefix,
                                file_id='*',
                                fake_file_type='*'):
    return os.path.join(
        FLAGS.test_tmpdir, '{}_{}.{}'.format(fake_filename_prefix, file_id,
                                             fake_file_type))

  def _create_fake_files(self, fake_filename_prefix, fake_file_type):
    ref_file_paths = []
    for file_id in range(NUM_FAKE_FILES):
      file_path = self._create_fake_file_pattern(fake_filename_prefix, file_id,
                                                 fake_file_type)
      # We basically touch a file.
      with open(file_path, 'a'):
        os.utime(file_path, None)
      ref_file_paths.append(file_path)
    return self._create_fake_file_pattern(
        fake_filename_prefix, fake_file_type=fake_file_type), ref_file_paths

  def test_parsing(self):
    base_dir = 'tensor2robot'
    file_pattern = os.path.join(
        FLAGS.test_srcdir, base_dir, 'test_data/pose_env_test_data.tfrecord')
    dataset = tfdata.parallel_read(file_patterns=file_pattern)
    state_spec = TSPEC(shape=(64, 64, 3), dtype=tf.uint8, name='state/image',
                       data_format='jpeg')
    action_spec = TSPEC(shape=(2), dtype=tf.bfloat16, name='pose')
    reward_spec = TSPEC(shape=(), dtype=tf.float32, name='reward')
    feature_tspec = PoseEnvFeature(state=state_spec, action=action_spec)
    label_tspec = PoseEnvLabel(reward=reward_spec)

    batched_dataset = dataset.batch(batch_size=1)
    dataset = tfdata.serialized_to_parsed(batched_dataset, feature_tspec,
                                          label_tspec)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    tensorspec_utils.assert_equal(feature_tspec, features, ignore_batch=True)
    tensorspec_utils.assert_equal(label_tspec, labels, ignore_batch=True)
    with self.session() as session:
      features_, labels_ = session.run([features, labels])
      self.assertAllEqual([1, 64, 64, 3], features_.state.shape)
      self.assertAllEqual([1, 2], features_.action.shape)
      self.assertAllEqual((1,), labels_.reward.shape)

  def _write_test_images_examples(self, batch_of_list_of_encoded_images,
                                  file_path):
    with tf.python_io.TFRecordWriter(file_path) as writer:
      for list_of_encoded_images in batch_of_list_of_encoded_images:
        example = tf.train.Example()
        for encoded_image in list_of_encoded_images:
          example.features.feature['image/encoded'].bytes_list.value.append(
              encoded_image)
        writer.write(example.SerializeToString())

  def test_compress_decompress_fn(self):
    batch_size = 5
    base_dir = 'tensor2robot'
    file_pattern = os.path.join(FLAGS.test_srcdir, base_dir,
                                'test_data/pose_env_test_data.tfrecord')
    dataset = tfdata.parallel_read(file_patterns=file_pattern)
    state_spec = TSPEC(
        shape=(64, 64, 3),
        dtype=tf.uint8,
        name='state/image',
        data_format='jpeg')
    action_spec = TSPEC(shape=(2), dtype=tf.bfloat16, name='pose')
    reward_spec = TSPEC(shape=(), dtype=tf.float32, name='reward')
    feature_spec = tensorspec_utils.TensorSpecStruct(
        state=state_spec, action=action_spec)
    label_spec = tensorspec_utils.TensorSpecStruct(reward=reward_spec)

    dataset = tfdata.parallel_read(file_patterns=file_pattern)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = tfdata.serialized_to_parsed(dataset, feature_spec, label_spec)
    features, _ = dataset.make_one_shot_iterator().get_next()
    # Check tensor shapes.
    self.assertAllEqual((batch_size,) + feature_spec.state.shape,
                        features.state.get_shape().as_list())
    with self.session() as session:
      original_features = session.run(features)

    dataset = tfdata.parallel_read(file_patterns=file_pattern)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = tfdata.serialized_to_parsed(dataset, feature_spec, label_spec)
    dataset = dataset.map(
        tfdata.create_compress_fn(feature_spec, label_spec, quality=100))
    dataset = dataset.map(tfdata.create_decompress_fn(feature_spec, label_spec))
    features, _ = dataset.make_one_shot_iterator().get_next()
    # Check tensor shapes.
    self.assertAllEqual((batch_size,) + feature_spec.state.shape,
                        features.state.get_shape().as_list())
    with self.session() as session:
      compressed_decompressed_features = session.run(features)
    ref_state = original_features.state.astype(np.float32) / 255
    state = compressed_decompressed_features.state.astype(np.float32) / 255
    np.testing.assert_almost_equal(ref_state, state, decimal=1)

  @parameterized.named_parameters(
      ('uint8', np.uint8, tf.uint8),
      ('uint16', np.uint16, tf.uint16),
  )
  def test_images_decoding(self, np_data_type, tf_data_type):
    file_pattern = os.path.join(self.create_tempdir().full_path,
                                'test.tfrecord')
    image_width = 640
    image_height = 512
    maxval = np.iinfo(np_data_type).max  # Maximum value for byte-encoded image.
    image_np = np.random.uniform(
        size=(image_height, image_width), high=maxval).astype(np.int32)
    png_encoded_image = image.numpy_to_image_string(image_np, 'png',
                                                    np_data_type)
    test_data = [[png_encoded_image]]
    self._write_test_images_examples(test_data, file_pattern)
    feature_spec = tensorspec_utils.TensorSpecStruct()
    feature_spec.images = tensorspec_utils.ExtendedTensorSpec(
        shape=(image_height, image_width, 1),
        dtype=tf_data_type,
        name='image/encoded',
        data_format='png')
    dataset = tfdata.parallel_read(file_patterns=file_pattern)
    dataset = dataset.batch(1, drop_remainder=True)
    if np_data_type == np.uint32:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        dataset = tfdata.serialized_to_parsed(dataset, feature_spec, None)
    else:
      dataset = tfdata.serialized_to_parsed(dataset, feature_spec, None)
    features = dataset.make_one_shot_iterator().get_next()
    # Check tensor shapes.
    self.assertAllEqual(
        [1, image_height, image_width, 1],
        features.images.get_shape().as_list())
    with self.session() as session:
      np_features = session.run(features)
      self.assertEqual(np_features['images'].dtype, np_data_type)

  def test_images_decoding_raises(self):
    file_pattern = os.path.join(self.create_tempdir().full_path,
                                'test.tfrecord')
    image_width = 640
    image_height = 512
    maxval = np.iinfo(np.uint32).max  # Maximum value for byte-encoded image.
    image_np = np.random.uniform(
        size=(image_height, image_width), high=maxval).astype(np.int32)
    png_encoded_image = image.numpy_to_image_string(image_np, 'png',
                                                    np.uint32)
    test_data = [[png_encoded_image]]
    self._write_test_images_examples(test_data, file_pattern)
    feature_spec = tensorspec_utils.TensorSpecStruct()
    feature_spec.images = tensorspec_utils.ExtendedTensorSpec(
        shape=(image_height, image_width, 1),
        dtype=tf.uint32,
        name='image/encoded',
        data_format='png')
    dataset = tfdata.parallel_read(file_patterns=file_pattern)
    dataset = dataset.batch(1, drop_remainder=True)
    with self.assertRaises(ValueError):
      tfdata.serialized_to_parsed(dataset, feature_spec, None)

  def _write_test_sequence_examples(self, sequence_length, tfrecord_path):
    example = tf.train.SequenceExample()
    context_key = 'context_feature'
    seq_key = 'sequence_feature'
    image_seq_key = 'image_sequence_feature'
    example.context.feature[context_key].int64_list.value.append(10)
    for i in range(sequence_length):
      f = example.feature_lists.feature_list[seq_key].feature.add()
      f.float_list.value.extend([3, 1])
      f = example.feature_lists.feature_list[image_seq_key].feature.add()
      img = TEST_IMAGE * i
      f.bytes_list.value.append(image.numpy_to_image_string(img, 'jpeg'))
    with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
      writer.write(example.SerializeToString())

  def _write_test_varlen_examples(self, data_of_lists, file_path):
    with tf.python_io.TFRecordWriter(file_path) as writer:
      for data in data_of_lists:
        example = tf.train.Example()
        example.features.feature['varlen'].int64_list.value.extend(data)
        writer.write(example.SerializeToString())

  @parameterized.named_parameters(
      ('batch_size=1', 1),
      ('batch_size=2', 2),
  )
  def test_varlen_feature_spec(self, batch_size):
    file_pattern = os.path.join(self.create_tempdir().full_path,
                                'test.tfrecord')
    test_data = [[1], [1, 2]]
    self._write_test_varlen_examples(test_data, file_pattern)
    feature_spec = tensorspec_utils.TensorSpecStruct()
    feature_spec.varlen = tensorspec_utils.ExtendedTensorSpec(
        shape=(3,), dtype=tf.int64, name='varlen', varlen_default_value=3.0)
    dataset = tfdata.parallel_read(file_patterns=file_pattern)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = tfdata.serialized_to_parsed(dataset, feature_spec, None)
    features = dataset.make_one_shot_iterator().get_next()
    # Check tensor shapes.
    self.assertAllEqual([None, 3], features.varlen.get_shape().as_list())
    with self.session() as session:
      np_features = session.run(features)
      self.assertAllEqual(np_features.varlen,
                          np.array([[1, 3, 3], [1, 2, 3]][:batch_size]))
      self.assertAllEqual([batch_size, 3], np_features.varlen.shape)
      # Check that images are equal.

  def _write_test_varlen_images_examples(self, batch_of_list_of_encoded_images,
                                         file_path):
    with tf.python_io.TFRecordWriter(file_path) as writer:
      for list_of_encoded_images in batch_of_list_of_encoded_images:
        example = tf.train.Example()
        for encoded_image in list_of_encoded_images:
          example.features.feature['varlen_images'].bytes_list.value.append(
              encoded_image)
        writer.write(example.SerializeToString())

  @parameterized.named_parameters(
      ('batch_size=1', 1),
      ('batch_size=2', 2),
  )
  def test_varlen_images_feature_spec(self, batch_size):
    file_pattern = os.path.join(self.create_tempdir().full_path,
                                'test.tfrecord')
    image_width = 640
    image_height = 512
    padded_varlen_size = 3
    maxval = 255  # Maximum value for byte-encoded image.
    image_np = np.random.uniform(
        size=(image_height, image_width), high=maxval).astype(np.int32)
    png_encoded_image = image.numpy_to_image_string(image_np, 'png')
    test_data = [[png_encoded_image], [png_encoded_image, png_encoded_image]]
    self._write_test_varlen_images_examples(test_data, file_pattern)
    feature_spec = tensorspec_utils.TensorSpecStruct()
    feature_spec.varlen_images = tensorspec_utils.ExtendedTensorSpec(
        shape=(padded_varlen_size, image_height, image_width, 1),
        dtype=tf.uint8,
        name='varlen_images',
        data_format='png',
        varlen_default_value=0)
    dataset = tfdata.parallel_read(file_patterns=file_pattern)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = tfdata.serialized_to_parsed(dataset, feature_spec, None)
    features = dataset.make_one_shot_iterator().get_next()
    # Check tensor shapes.
    self.assertAllEqual(
        [None, padded_varlen_size, image_height, image_width, 1],
        features.varlen_images.get_shape().as_list())
    with self.session() as session:
      np_features = session.run(features)
      black_image = np.zeros((image_height, image_width))
      self.assertAllEqual(
          np_features.varlen_images,
          np.expand_dims(
              np.stack([
                  np.stack([image_np, black_image, black_image]),
                  np.stack([image_np, image_np, black_image])
              ])[:batch_size, :, :, :],
              axis=-1))
      self.assertAllEqual(
          [batch_size, padded_varlen_size, image_height, image_width, 1],
          np_features.varlen_images.shape)

  @parameterized.named_parameters(
      ('batch_size=1', 1),
      ('batch_size=2', 2),
  )
  def test_varlen_images_feature_spec_raises(self, batch_size):
    file_pattern = os.path.join(self.create_tempdir().full_path,
                                'test.tfrecord')
    image_width = 640
    image_height = 512
    padded_varlen_size = 3
    maxval = 255  # Maximum value for byte-encoded image.
    image_np = np.random.uniform(
        size=(image_height, image_width), high=maxval).astype(np.int32)
    image_with_invalid_size = np.ones((1024, 1280)) * 255
    png_encoded_image = image.numpy_to_image_string(image_np, 'png')
    png_encoded_image_with_invalid_size = image.numpy_to_image_string(
        image_with_invalid_size, 'png')
    test_data = [[png_encoded_image_with_invalid_size],
                 [png_encoded_image, png_encoded_image_with_invalid_size]]
    self._write_test_varlen_images_examples(test_data, file_pattern)
    feature_spec = tensorspec_utils.TensorSpecStruct()
    feature_spec.varlen_images = tensorspec_utils.ExtendedTensorSpec(
        shape=(padded_varlen_size, image_height, image_width, 1),
        dtype=tf.uint8,
        name='varlen_images',
        data_format='png',
        varlen_default_value=0)
    dataset = tfdata.parallel_read(file_patterns=file_pattern)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = tfdata.serialized_to_parsed(dataset, feature_spec, None)
    features = dataset.make_one_shot_iterator().get_next()
    # Check tensor shapes.
    self.assertAllEqual(
        [None, padded_varlen_size, image_height, image_width, 1],
        features.varlen_images.get_shape().as_list())
    with self.session() as session:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        session.run(features)

  @parameterized.named_parameters(
      ('batch_size=1', 1),
      ('batch_size=2', 2),
  )
  def test_sequence_parsing(self, batch_size):
    file_pattern = os.path.join(self.create_tempdir().full_path,
                                'test.tfrecord')
    sequence_length = 3
    self._write_test_sequence_examples(sequence_length, file_pattern)
    dataset = tfdata.parallel_read(file_patterns=file_pattern)
    # Features
    state_spec_1 = tensorspec_utils.ExtendedTensorSpec(
        shape=(TEST_IMAGE_SHAPE), dtype=tf.uint8, is_sequence=True,
        name='image_sequence_feature', data_format='JPEG')
    state_spec_2 = tensorspec_utils.ExtendedTensorSpec(
        shape=(2), dtype=tf.float32, is_sequence=True, name='sequence_feature')
    feature_tspec = PoseEnvFeature(state=state_spec_1, action=state_spec_2)
    feature_tspec = tensorspec_utils.add_sequence_length_specs(feature_tspec)
    # Labels
    reward_spec = tensorspec_utils.ExtendedTensorSpec(
        shape=(), dtype=tf.int64, is_sequence=False, name='context_feature')
    label_tspec = PoseEnvLabel(reward=reward_spec)
    label_tspec = tensorspec_utils.add_sequence_length_specs(label_tspec)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = tfdata.serialized_to_parsed(dataset, feature_tspec, label_tspec)
    features, labels = dataset.make_one_shot_iterator().get_next()
    # Check tensor shapes.
    self.assertAllEqual(
        [batch_size, None] + TEST_IMAGE_SHAPE, features.state.shape.as_list())
    self.assertAllEqual(
        [batch_size, None, 2], features.action.shape.as_list())
    self.assertAllEqual([batch_size], features.state_length.shape.as_list())
    self.assertAllEqual([batch_size], features.action_length.shape.as_list())
    self.assertAllEqual([batch_size], labels.reward.shape.as_list())
    with self.session() as session:
      features_, labels_ = session.run([features, labels])
      # Check that images are equal.
      for i in range(3):
        img = TEST_IMAGE * i
        self.assertAllEqual(img, features_.state[0, i])
      # Check that numpy shapes are equal.
      self.assertAllEqual(
          [batch_size, sequence_length] + TEST_IMAGE_SHAPE,
          features_.state.shape)
      self.assertAllEqual([sequence_length]*batch_size, features_.state_length)
      self.assertAllEqual(
          [batch_size, sequence_length, 2], features_.action.shape)
      self.assertAllEqual([batch_size], labels_.reward.shape)

  @parameterized.named_parameters(
      ('tfrecord', 'random_prefix_c', 'tfrecord'),
  )
  def test_get_data_format_and_filenames(self, fake_filename_prefix,
                                         ref_data_format):
    file_pattern_sstables, ref_file_paths = self._create_fake_files(
        fake_filename_prefix=fake_filename_prefix,
        fake_file_type=ref_data_format)
    data_format, filenames = tfdata.get_data_format_and_filenames(
        file_pattern_sstables)
    self.assertAllEqual(sorted(filenames), sorted(ref_file_paths))
    self.assertEqual(data_format, ref_data_format)

    # Test the ability to strip the filetype prefix.
    data_format, filenames = tfdata.get_data_format_and_filenames(
        '{}:{}'.format(ref_data_format, file_pattern_sstables))
    self.assertAllEqual(sorted(filenames), sorted(ref_file_paths))
    self.assertEqual(data_format, ref_data_format)

    # Test the ability to use a comma-separated string
    data_format, filenames = tfdata.get_data_format_and_filenames(
        ','.join(ref_file_paths))
    self.assertAllEqual(sorted(filenames), sorted(ref_file_paths))
    self.assertEqual(data_format, ref_data_format)

  def test_get_data_format_and_filenames_raises(self):
    # This should raise since the data format cannot be inferred from the
    # file pattern.
    with self.assertRaises(ValueError):
      tfdata.get_data_format_and_filenames('a wrong file pattern')


  @parameterized.named_parameters(
      ('tfrecord_suffix', '/abc/stuff.tfrecord', 'tfrecord'),
      ('tfrecord_prefix', 'tfrecord:/abc/stuff', 'tfrecord'),
      ('tfrecord_suffix_and_prefix', 'tfrecord:/abc/stuff', 'tfrecord'),
  )
  def test_infer_data_format(self, file_patterns, ref_data_format):
    self.assertEqual(tfdata.infer_data_format(file_patterns), ref_data_format)


  def test_get_batch_size(self):
    self.assertEqual(
        tfdata.get_batch_size({'batch_size': 64}, batch_size=128), 64)
    self.assertEqual(
        tfdata.get_batch_size({'batch_size': 64}, batch_size=64), 64)
    self.assertEqual(
        tfdata.get_batch_size({'batch_size': 64}, batch_size=None), 64)
    self.assertEqual(
        tfdata.get_batch_size({'batch_size': None}, batch_size=64), 64)


if __name__ == '__main__':
  tf.test.main()
