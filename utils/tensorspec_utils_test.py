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

# Lint as: python2, python3
"""Tests for tensor2robot.utils.tensorspec_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import os
import pickle

from absl.testing import parameterized

import numpy as np
from tensor2robot.utils import tensorspec_utils as utils
import tensorflow as tf
nest = tf.contrib.framework.nest

TSPEC = utils.ExtendedTensorSpec

MockFoo = collections.namedtuple('Foo', ['sim', 'real'])
MockBar = collections.namedtuple('Bar', ['images', 'actions'])
T1 = TSPEC((224, 224, 3), tf.float32, 'images', data_format='jpeg')
T2 = TSPEC((6), tf.float32, 'actions')
T3 = TSPEC((), tf.float32, 'reward')
O4 = TSPEC((224, 224, 3), tf.float32, 'debug_images', is_optional=True)
T5 = TSPEC((224, 224, 3), tf.float32, 'jpeg_images', data_format='jpeg')
O6 = TSPEC((6), tf.float32, 'debug_actions', is_optional=True)
S7 = TSPEC((6), tf.float32, 'sequence_actions', is_sequence=True)
D1 = TSPEC((224, 224, 3), tf.float32, 'debug_images', dataset_key='d1')
D2 = TSPEC((224, 224, 3), tf.float32, 'debug_images', dataset_key='d2')
PARAMS = ['dict', 'namedtuple', 'list', 'tuple']

# Mocks for testing optional spec structures.
MockNested = collections.namedtuple('Nested', ['train', 'test'])
mock_nested_spec = MockNested(
    train=MockBar(images=T1, actions=T2), test=MockBar(images=T1, actions=T2))
MockNestedOptional = collections.namedtuple('NestedOptional',
                                            ['train', 'test', 'optional'])
mock_nested_optional_spec = MockNestedOptional(
    train=MockBar(images=T1, actions=T2),
    test=MockBar(images=T1, actions=T2),
    optional=MockBar(images=O4, actions=O6))
MockNestedSubset = collections.namedtuple('NestedSubset', ['train'])
mock_nested_subset_spec = MockNestedSubset(train=MockBar(images=T1, actions=T2))

INVALID_SPEC_STRUCTURES = (('non_tensorspec_tensor_values_dict', {
    'test': 10
}), ('non_tensorspec_tensor_values_named_tuple',
     MockBar(images=np.random.random_sample(10), actions='action')),
                           ('duplicated_names_in_spec_structure',
                            MockBar(
                                images=TSPEC(
                                    shape=(3, 2, 3),
                                    dtype=tf.float32,
                                    name='images'),
                                actions=TSPEC(
                                    shape=(3, 2),
                                    dtype=tf.float32,
                                    name='images'))))

REFERENCE_FLAT_ORDERED_DICT = collections.OrderedDict([
    ('train/images', T1),
    ('train/actions', T2),
    ('test/images', T1),
    ('test/actions', T2),
    ('optional/images', O4),
    ('optional/actions', O6),
])


class TensorspecUtilsTest(parameterized.TestCase, tf.test.TestCase):
  """Test of tensorspec utils.

  Most of the tests regarding ExtendedTensorSpec are a copy of
  tensorflow.python.framework.tensor_spec_test. We cannot re-use this test due
  to visibility, hence we replicate the test.
  """

  def test_assert_equal(self):
    mock_nested_spec_copy = copy.deepcopy(mock_nested_spec)
    utils.assert_equal(mock_nested_spec, mock_nested_spec_copy)

  def test_assert_not_equal(self):
    with self.assertRaises(ValueError):
      utils.assert_equal(mock_nested_spec, mock_nested_subset_spec)

  def test_assert_required(self):
    # A subset of the required spec is asked for, that should work.
    utils.assert_required(mock_nested_subset_spec, mock_nested_spec)
    # We would like to have more, but all additional specs are optional.
    utils.assert_required(mock_nested_optional_spec, mock_nested_spec)
    # We ask for more required tensor_specs than we actually have.
    with self.assertRaises(ValueError):
      utils.assert_required(mock_nested_spec, mock_nested_subset_spec)

  def test_init_with_attributes(self):
    train = utils.TensorSpecStruct(images=T1, actions=T2)
    flat_nested_optional_spec = utils.flatten_spec_structure(
        mock_nested_optional_spec)
    utils.assert_equal(train, flat_nested_optional_spec.train)
    alternative_dict = {'o6': O6, 'o4': O4}
    hierarchy = utils.TensorSpecStruct(
        nested_optional_spec=mock_nested_optional_spec,
        alternative=alternative_dict)
    utils.assert_equal(hierarchy.nested_optional_spec,
                       flat_nested_optional_spec)
    self.assertDictEqual(hierarchy.alternative.to_dict(), alternative_dict)
    self.assertCountEqual(list(hierarchy.alternative.keys()), ['o4', 'o6'])
    self.assertCountEqual(
        list(hierarchy.keys()), [
            'nested_optional_spec/train/images',
            'nested_optional_spec/train/actions',
            'nested_optional_spec/test/images',
            'nested_optional_spec/test/actions',
            'nested_optional_spec/optional/images',
            'nested_optional_spec/optional/actions', 'alternative/o6',
            'alternative/o4'
        ])

  def test_extended_spec_proto(self):
    extended_tensorspec_proto = T1.to_proto()
    t1_from_proto = utils.ExtendedTensorSpec.from_serialized_proto(
        extended_tensorspec_proto.SerializeToString())
    utils.assert_equal_spec_or_tensor(T1, t1_from_proto)

  def test_tensorspec_struct_proto(self):
    tensor_spec_struct = utils.TensorSpecStruct(REFERENCE_FLAT_ORDERED_DICT)
    tss_from_proto = utils.TensorSpecStruct.from_serialized_proto(
        tensor_spec_struct.to_proto().SerializeToString())
    self.assertDictEqual(tensor_spec_struct.to_dict(), tss_from_proto.to_dict())

  def test_flatten_spec_structure(self):
    flatten_spec_structure = utils.flatten_spec_structure(
        mock_nested_subset_spec)
    self.assertDictEqual(flatten_spec_structure, {
        'train/images': T1,
        'train/actions': T2
    })

  def test_tensor_spec_struct_init(self):
    flat_ordered_dict_with_attributes = utils.TensorSpecStruct(
        REFERENCE_FLAT_ORDERED_DICT)
    self.assertDictEqual(flat_ordered_dict_with_attributes,
                         REFERENCE_FLAT_ORDERED_DICT)

    # Ensure we see the right subset of the data.
    self.assertEqual(
        list(flat_ordered_dict_with_attributes.train.keys()),
        ['images', 'actions'])

  def test_tensor_spec_struct_to_dict(self):
    ref = utils.TensorSpecStruct(REFERENCE_FLAT_ORDERED_DICT)
    ref_dict = ref.to_dict()
    from_dict = utils.TensorSpecStruct(ref_dict)
    self.assertDictEqual(ref_dict, from_dict.to_dict())

  def test_tensor_spec_struct_assignment(self):
    test_flat_ordered_dict = utils.TensorSpecStruct()

    # We cannot assign an empty ordered dict.
    with self.assertRaises(ValueError):
      test_flat_ordered_dict.should_raise = (
          utils.TensorSpecStruct())

    # Invalid data types for assignment
    # TODO(T2R_CONTRIBUTORS): Deterimine which type is not supported by pytype.
    # for should_raise in ['1', 1, 1.0, {}]:
    #   with self.assertRaises(ValueError):
    #     test_flat_ordered_dict.should_raise = should_raise

    sub_data = utils.TensorSpecStruct()
    sub_data.data = np.ones(1)
    test_flat_ordered_dict.sub_data = sub_data
    # Now we can also extend.
    test_flat_ordered_dict.sub_data.additional = np.zeros(1)

  def test_tensor_spec_struct_adding_attribute(self):
    flat_ordered_dict_with_attributes = utils.TensorSpecStruct(
        REFERENCE_FLAT_ORDERED_DICT)
    # Now we check that we can change an attribute and it affects the parent.
    flat_ordered_dict_with_attributes.train.addition = O6
    self.assertEqual(
        list(flat_ordered_dict_with_attributes.train.keys()),
        ['images', 'actions', 'addition'])

    self.assertEqual(flat_ordered_dict_with_attributes.train.addition,
                     flat_ordered_dict_with_attributes['train/addition'])

    # It propagates to the parent.
    self.assertEqual(
        list(flat_ordered_dict_with_attributes.keys()),
        list(REFERENCE_FLAT_ORDERED_DICT.keys()) + ['train/addition'])
    self.assertEqual(flat_ordered_dict_with_attributes['train/addition'],
                     flat_ordered_dict_with_attributes.train.addition)

  def test_tensor_spec_struct_attribut_errors(self):
    flat_ordered_dict_with_attributes = utils.TensorSpecStruct(
        REFERENCE_FLAT_ORDERED_DICT)
    # These attributes do not exist.
    with self.assertRaises(AttributeError):
      _ = flat_ordered_dict_with_attributes['optional_typo']
    with self.assertRaises(AttributeError):
      _ = flat_ordered_dict_with_attributes.optional_typo
    self.assertDictEqual(
        flat_ordered_dict_with_attributes['optional'].to_dict(),
        flat_ordered_dict_with_attributes.optional.to_dict())

  def test_tensor_spec_struct_deleting_element(self):
    flat_ordered_dict_with_attributes = utils.TensorSpecStruct(
        REFERENCE_FLAT_ORDERED_DICT)
    # Now we show that we can delete items and it will propagate down.
    self.assertEqual(flat_ordered_dict_with_attributes.optional.images, O4)

    del flat_ordered_dict_with_attributes['optional/images']
    with self.assertRaises(AttributeError):
      _ = flat_ordered_dict_with_attributes.optional.images

    # Now we show that we can delete items and it will propagate up.
    self.assertIn('test/actions', flat_ordered_dict_with_attributes)
    test = flat_ordered_dict_with_attributes.test
    del test['actions']
    self.assertNotIn('test/actions', flat_ordered_dict_with_attributes)

  def test_tensor_spec_struct_composing_with_a_dict(self):
    # We reset our base instance.
    flat_ordered_dict_with_attributes = utils.TensorSpecStruct(
        REFERENCE_FLAT_ORDERED_DICT)
    new_field = utils.TensorSpecStruct(
        REFERENCE_FLAT_ORDERED_DICT)
    flat_ordered_dict_with_attributes.new_field = new_field
    self.assertEqual(
        list(flat_ordered_dict_with_attributes.new_field.keys()),
        list(new_field.keys()))
    self.assertEqual(
        list(flat_ordered_dict_with_attributes.keys()),
        list(REFERENCE_FLAT_ORDERED_DICT.keys()) +
        ['new_field/' + key for key in new_field.keys()])

  def test_tensor_spec_struct_composing_with_namedtuple(self):
    # Test that we can combine namedtuple and TensorSpecStruct.
    # This is important since non top level dicts use the dict_view but
    # nest.flatten uses the cpython interface which is why we need to maintain
    # a local copy of the data in subviews.
    test_spec = MockNested(
        train={
            'a': np.ones(1),
            'b': 2 * np.ones(1)
        },
        test=MockBar(images=T1, actions=T2))
    ref_flat_spec = utils.flatten_spec_structure(test_spec)
    new_test_spec = MockNested(
        train=ref_flat_spec.train, test=MockBar(images=T1, actions=T2))
    new_flat_spec = utils.flatten_spec_structure(new_test_spec)
    for key in ref_flat_spec:
      self.assertIn(key, new_flat_spec)
      self.assertEqual(new_flat_spec[key], ref_flat_spec[key])

  def test_tensor_spec_struct_correct_hierarchy(self):
    # Test that our prefix works in the correct way.
    # This test used to break due to an prefix indexing error.
    test_flat_ordered_dict = utils.TensorSpecStruct()
    test_flat_ordered_dict.val_mode = np.ones(1)
    val = utils.TensorSpecStruct()
    val.mode = np.zeros(1)
    val.data = np.ones(1)
    test_flat_ordered_dict.val = val
    self.assertEqual(list(test_flat_ordered_dict.val.keys()), ['mode', 'data'])
    self.assertEqual(
        list(test_flat_ordered_dict['val'].keys()), ['mode', 'data'])

  def test_filter_required_tensor_spec_struct(self):
    tensor_spec_struct = utils.flatten_spec_structure(mock_nested_optional_spec)
    self.assertDictEqual(
        tensor_spec_struct, {
            'train/images': T1,
            'train/actions': T2,
            'test/images': T1,
            'test/actions': T2,
            'optional/images': O4,
            'optional/actions': O6,
        })
    required_tensor_spec_struct = utils.filter_required_flat_tensor_spec(
        tensor_spec_struct)
    self.assertDictEqual(
        required_tensor_spec_struct, {
            'train/images': T1,
            'train/actions': T2,
            'test/images': T1,
            'test/actions': T2,
        })

    # In case we pass a hierarchical spec this function should raise.
    with self.assertRaises(ValueError):
      required_tensor_spec_struct = utils.filter_required_flat_tensor_spec(
          mock_nested_optional_spec)

  def test_tensorspec_to_feature_dict(self):
    features, tensor_spec_dict = utils.tensorspec_to_feature_dict(
        mock_nested_subset_spec, decode_images=True)
    self.assertDictEqual(tensor_spec_dict, {
        'images': T1,
        'actions': T2,
    })
    self.assertDictEqual(
        features, {
            'images': tf.FixedLenFeature((), tf.string),
            'actions': tf.FixedLenFeature(T2.shape, T2.dtype),
        })
    features, tensor_spec_dict = utils.tensorspec_to_feature_dict(
        mock_nested_subset_spec, decode_images=False)
    self.assertDictEqual(tensor_spec_dict, {
        'images': T1,
        'actions': T2,
    })
    self.assertDictEqual(
        features, {
            'images': tf.FixedLenFeature(T1.shape, T1.dtype),
            'actions': tf.FixedLenFeature(T2.shape, T2.dtype),
        })

  def test_assert_equal_spec_or_tensor(self):
    self.assertIsNone(utils.assert_equal_spec_or_tensor(T1, T1))
    self.assertIsNone(
        utils.assert_equal_spec_or_tensor(
            T1, TSPEC(shape=(224, 224, 3), dtype=tf.float32)))
    self.assertIsNone(
        utils.assert_equal_spec_or_tensor(
            T1, TSPEC(shape=(224, 224, 3), dtype=tf.float32, name='random')))
    self.assertIsNone(
        utils.assert_equal_spec_or_tensor(
            T1, tf.constant(value=np.zeros((224, 224, 3), dtype=np.float32))))

    # Should raise because of wrong shape.
    with self.assertRaises(ValueError):
      utils.assert_equal_spec_or_tensor(T1, T2)

    # Should raise because of wrong shape.
    with self.assertRaises(ValueError):
      utils.assert_equal_spec_or_tensor(
          T1, TSPEC(shape=(224, 223, 3), dtype=tf.float32))

    # Should raise because of wrong dtype.
    with self.assertRaises(ValueError):
      utils.assert_equal_spec_or_tensor(
          T1, TSPEC(shape=(224, 224, 3), dtype=tf.uint8))

    # Should raise because of wrong dtype.
    with self.assertRaises(ValueError):
      utils.assert_equal_spec_or_tensor(
          T1, tf.constant(value=np.zeros((224, 224, 3), dtype=np.uint8)))

  def test_is_flat_spec_or_tensors_structure(self):
    flatten_spec_structure = utils.flatten_spec_structure(
        mock_nested_subset_spec)
    self.assertFalse(
        utils.is_flat_spec_or_tensors_structure(mock_nested_subset_spec))
    self.assertTrue(
        utils.is_flat_spec_or_tensors_structure(flatten_spec_structure))

    # Flat lists are not tensor_spec_structs since they are not proper
    # {path: spec_or_tensor_or_numpy} structures. They would not be invariant
    # under flatten_spec_structure calls.
    self.assertFalse(
        utils.is_flat_spec_or_tensors_structure([T1, T2])
    )

    # Flat dictionaries are proper tensor_spec_struct_or_tensors, they would
    # be invariant under flatten_spec_structure.
    self.assertTrue(
        utils.is_flat_spec_or_tensors_structure({
            't1': T1,
            't2': T2
        }))

  def test_pack_flat_sequence_to_spec_structure(self):
    subset_placeholders = utils.make_placeholders(mock_nested_subset_spec)
    flattened_subset_placeholders = utils.flatten_spec_structure(
        subset_placeholders)
    packed_subset_placeholders = utils.pack_flat_sequence_to_spec_structure(
        mock_nested_subset_spec, flattened_subset_placeholders)
    utils.assert_equal(subset_placeholders, packed_subset_placeholders)
    utils.assert_equal(
        mock_nested_subset_spec, packed_subset_placeholders, ignore_batch=True)

    placeholders = utils.make_placeholders(mock_nested_spec)
    flattened_placeholders = utils.flatten_spec_structure(placeholders)
    packed_placeholders = utils.pack_flat_sequence_to_spec_structure(
        mock_nested_subset_spec, flattened_placeholders)
    # We only subselect what we need in pack_flat_sequence_to_spec_structure,
    # hence, we should recover what we wanted.
    utils.assert_equal(
        mock_nested_subset_spec, packed_placeholders, ignore_batch=True)
    utils.assert_equal(subset_placeholders, packed_placeholders)

    packed_optional_placeholders = utils.pack_flat_sequence_to_spec_structure(
        mock_nested_optional_spec, flattened_placeholders)
    # Although mock_nested_optional_spec would like more tensors
    # flattened_placeholders cannot provide them, fortunately they are optional.
    utils.assert_required(packed_optional_placeholders, placeholders)
    utils.assert_required(
        mock_nested_spec, packed_optional_placeholders, ignore_batch=True)

  def test_pack_flat_sequence_to_spec_structure_ensure_order(self):
    test_spec = utils.TensorSpecStruct()
    test_spec.b = utils.ExtendedTensorSpec(
        shape=(1,), dtype=tf.float32, name='b')
    test_spec.a = utils.ExtendedTensorSpec(
        shape=(1,), dtype=tf.float32, name='a')
    test_spec.c = utils.ExtendedTensorSpec(
        shape=(1,), dtype=tf.float32, name='c')
    placeholders = utils.make_placeholders(test_spec)
    packed_placeholders = utils.pack_flat_sequence_to_spec_structure(
        test_spec, placeholders)
    for pos, order_name in enumerate(['a', 'b', 'c']):
      self.assertEqual(list(packed_placeholders.keys())[pos], order_name)
      self.assertEqual(
          list(packed_placeholders.values())[pos].op.name, order_name)

  def test_validate_flatten_and_pack(self):
    # An example data pipeline.
    # Some input generator creates input features according to some spec.
    input_features = utils.make_placeholders(mock_nested_spec)
    # Assume a preprocessor has altered these input_features and we want
    # to pass the data on to the next stage, then we simply assure that
    # our output is according to our spec and flatten.
    flat_input_features = utils.validate_and_flatten(
        mock_nested_optional_spec, input_features, ignore_batch=True)
    utils.assert_required(
        mock_nested_optional_spec, input_features, ignore_batch=True)

    # Then e.g. the model_fn receives the flat_input_spec and validates
    # that it is according to it's requirements and packs it back into the
    # spec structure.
    output_features = utils.validate_and_pack(
        mock_nested_subset_spec, flat_input_features, ignore_batch=True)
    utils.assert_required(
        mock_nested_subset_spec, output_features, ignore_batch=True)

  def test_add_sequence_length_specs(self):
    input_spec = utils.TensorSpecStruct(image1=D1, actions=S7)
    modified_spec = utils.add_sequence_length_specs(input_spec)
    expected_length_spec = utils.ExtendedTensorSpec(
        shape=(), dtype=tf.int64, name='sequence_actions_length')
    self.assertEqual(modified_spec.actions_length, expected_length_spec)

  def test_filter_spec_structure_by_dataset(self):
    test_spec = utils.TensorSpecStruct(image1=D1, image2=D2)
    for dataset_key, name, spec in zip(
        ['d1', 'd2'], ['image1', 'image2'], [D1, D2]):
      filtered_spec = utils.filter_spec_structure_by_dataset(
          test_spec, dataset_key)
      self.assertDictEqual(filtered_spec, {name: spec})

  @parameterized.named_parameters(*INVALID_SPEC_STRUCTURES)
  def test_flatten_spec_structure_raises(self, spec):
    self.skipTest(
        'b/134442538: Deterimine which type is not supported by pytype.')
    # with self.assertRaises(ValueError):
    #   utils.flatten_spec_structure(spec)

  @parameterized.parameters(*PARAMS)
  def test_assert_valid_spec_structure_is_valid(self, collection_type):
    spec = self._make_tensorspec_collection(collection_type)
    utils.assert_valid_spec_structure(spec)

  @parameterized.named_parameters(*INVALID_SPEC_STRUCTURES)
  def test_assert_valid_spec_structure_invalid(self, spec_or_tensors):
    with self.assertRaises(ValueError):
      utils.assert_valid_spec_structure(spec_or_tensors)

  def test_accepts_numpy_dtype(self):
    desc = utils.ExtendedTensorSpec([1], np.float32)
    self.assertEqual(desc.dtype, tf.float32)

  def test_accepts_tensor_shape(self):
    desc = utils.ExtendedTensorSpec(tf.TensorShape([1]), tf.float32)
    self.assertEqual(desc.shape, tf.TensorShape([1]))

  def test_unknown_shape(self):
    desc = utils.ExtendedTensorSpec(shape=None, dtype=tf.float32)
    self.assertEqual(desc.shape, tf.TensorShape(None))

  def test_sequence_shape(self):
    desc = utils.ExtendedTensorSpec(
        shape=(3, 2), dtype=tf.float32, is_sequence=True)
    self.assertEqual(desc.shape, tf.TensorShape((3, 2)))

  def test_shape_compatibility(self):
    unknown = tf.placeholder(tf.int64)
    partial = tf.placeholder(tf.int64, shape=[None, 1])
    full = tf.placeholder(tf.int64, shape=[2, 3])
    rank3 = tf.placeholder(tf.int64, shape=[4, 5, 6])

    desc_unknown = utils.ExtendedTensorSpec(None, tf.int64)
    self.assertTrue(desc_unknown.is_compatible_with(unknown))
    self.assertTrue(desc_unknown.is_compatible_with(partial))
    self.assertTrue(desc_unknown.is_compatible_with(full))
    self.assertTrue(desc_unknown.is_compatible_with(rank3))

    desc_partial = utils.ExtendedTensorSpec([2, None], tf.int64)
    self.assertTrue(desc_partial.is_compatible_with(unknown))
    self.assertTrue(desc_partial.is_compatible_with(partial))
    self.assertTrue(desc_partial.is_compatible_with(full))
    self.assertFalse(desc_partial.is_compatible_with(rank3))

    desc_full = utils.ExtendedTensorSpec([2, 3], tf.int64)
    self.assertTrue(desc_full.is_compatible_with(unknown))
    self.assertFalse(desc_full.is_compatible_with(partial))
    self.assertTrue(desc_full.is_compatible_with(full))
    self.assertFalse(desc_full.is_compatible_with(rank3))

    desc_rank3 = utils.ExtendedTensorSpec([4, 5, 6], tf.int64)
    self.assertTrue(desc_rank3.is_compatible_with(unknown))
    self.assertFalse(desc_rank3.is_compatible_with(partial))
    self.assertFalse(desc_rank3.is_compatible_with(full))
    self.assertTrue(desc_rank3.is_compatible_with(rank3))

  def test_type_compatibility(self):
    floats = tf.placeholder(tf.float32, shape=[10, 10])
    ints = tf.placeholder(tf.int32, shape=[10, 10])
    desc = utils.ExtendedTensorSpec(shape=(10, 10), dtype=tf.float32)
    self.assertTrue(desc.is_compatible_with(floats))
    self.assertFalse(desc.is_compatible_with(ints))

  def test_name(self):
    desc = utils.ExtendedTensorSpec([1], tf.float32, name='beep')
    self.assertEqual(desc.name, 'beep')

  def test_repr(self):
    desc1 = utils.ExtendedTensorSpec([1],
                                     tf.float32,
                                     name='beep',
                                     is_optional=True,
                                     data_format='jpeg',
                                     varlen_default_value=1)
    self.assertEqual(
        repr(desc1),
        "ExtendedTensorSpec(shape=(1,), dtype=tf.float32, name='beep', "
        "is_optional=True, is_sequence=False, is_extracted=False, "
        "data_format='jpeg', dataset_key='', varlen_default_value=1)")
    desc2 = utils.ExtendedTensorSpec([1, None], tf.int32, is_sequence=True)
    self.assertEqual(
        repr(desc2),
        "ExtendedTensorSpec(shape=(1, ?), dtype=tf.int32, name=None, "
        "is_optional=False, is_sequence=True, is_extracted=False, "
        "data_format=None, dataset_key='', varlen_default_value=None)")

  def test_from_spec(self):
    spec_1 = utils.ExtendedTensorSpec((1, 2), tf.int32)
    spec_2 = utils.ExtendedTensorSpec.from_spec(spec_1)
    self.assertEqual(spec_1, spec_2)

    # We make sure that we can actually overwrite the name.
    spec_1 = utils.ExtendedTensorSpec((1, 2), tf.int32, name='spec_1')
    spec_2 = utils.ExtendedTensorSpec.from_spec(spec_1, name='spec_2')

    # The name is not checked when we check for equality so it should still
    # pass. That is the default behavior of TensorSpec, therefore, we want to
    # maintain this behavior.
    self.assertEqual(spec_1, spec_2)
    self.assertEqual(spec_1.name, 'spec_1')
    self.assertEqual(spec_2.name, 'spec_2')

    # Add batch dimension.
    spec_2 = utils.ExtendedTensorSpec.from_spec(spec_1, batch_size=16)
    self.assertNotEqual(spec_1, spec_2)
    self.assertEqual(spec_1.shape, spec_2.shape[1:])
    self.assertEqual(spec_2.shape[0].value, 16)

    # Add batch dimension.
    spec_2 = utils.ExtendedTensorSpec.from_spec(spec_1, batch_size=-1)
    self.assertEqual(spec_2.shape[1:], spec_1.shape)
    self.assertIsNone(spec_2.shape[0].value)

    # Sequential.
    spec_1 = utils.ExtendedTensorSpec((1, 2), tf.int32, is_sequence=True)
    spec_2 = utils.ExtendedTensorSpec.from_spec(spec_1, batch_size=-1)
    self.assertEqual(spec_2.shape[1:], spec_1.shape)
    self.assertTrue(spec_2.is_sequence)

  def test_from_tensor(self):
    zero = tf.constant(0)
    spec = utils.ExtendedTensorSpec.from_tensor(zero)
    self.assertEqual(spec.dtype, tf.int32)
    self.assertEqual(spec.shape, [])
    self.assertEqual(spec.name, 'Const')

  def test_from_placeholder(self):
    unknown = tf.placeholder(tf.int64, name='unknown')
    partial = tf.placeholder(tf.float32, shape=[None, 1], name='partial')
    spec_1 = utils.ExtendedTensorSpec.from_tensor(unknown)
    self.assertEqual(spec_1.dtype, tf.int64)
    self.assertEqual(spec_1.shape, None)
    self.assertEqual(spec_1.name, 'unknown')
    spec_2 = utils.ExtendedTensorSpec.from_tensor(partial)
    self.assertEqual(spec_2.dtype, tf.float32)
    self.assertEqual(spec_2.shape.as_list(), [None, 1])
    self.assertEqual(spec_2.name, 'partial')

  def test_serialization(self):
    desc = utils.ExtendedTensorSpec([1, 5], tf.float32, 'test')
    self.assertEqual(pickle.loads(pickle.dumps(desc)), desc)

  @parameterized.parameters([True, False])
  def test_is_optional(self, is_optional):
    desc = utils.ExtendedTensorSpec(
        shape=[1], dtype=np.float32, is_optional=is_optional)
    self.assertEqual(desc.is_optional, is_optional)
    desc_copy = utils.ExtendedTensorSpec.from_spec(desc)
    self.assertEqual(desc_copy.is_optional, is_optional)
    desc_overwrite = utils.ExtendedTensorSpec.from_spec(
        desc, is_optional=not is_optional)
    self.assertEqual(desc_overwrite.is_optional, not is_optional)

  def test_extended_from_spec(self):
    desc = tf.contrib.framework.TensorSpec(
        shape=[1], dtype=np.float32)
    extended_desc = utils.ExtendedTensorSpec.from_spec(desc)
    self.assertEqual(desc, extended_desc)

  @parameterized.parameters([True, False])
  def test_optional_tensor_spec(self, is_optional):
    desc = utils.ExtendedTensorSpec(
        shape=[1], dtype=np.float32, is_optional=is_optional)
    self.assertEqual(desc.is_optional, is_optional)

  @parameterized.parameters(['jpeg', 'png'])
  def test_data_format(self, data_format):
    desc = utils.ExtendedTensorSpec(
        shape=[1], dtype=np.float32, data_format=data_format)
    self.assertEqual(desc.data_format, data_format)
    desc_copy = utils.ExtendedTensorSpec.from_spec(desc)
    self.assertEqual(desc_copy.data_format, data_format)
    desc_overwrite = utils.ExtendedTensorSpec.from_spec(
        desc, data_format='NO_FORMAT')
    self.assertEqual(desc_overwrite.data_format, 'NO_FORMAT')

  @parameterized.parameters(*PARAMS)
  def test_copy(self, collection_type):
    spec = self._make_tensorspec_collection(collection_type)
    spec_copy = utils.copy_tensorspec(spec)
    utils.assert_equal(spec, spec_copy)

  def test_copy_none_name(self):
    spec = utils.TensorSpecStruct()
    spec.none_name = utils.ExtendedTensorSpec(shape=(1,), dtype=tf.float32)
    spec.with_name = utils.ExtendedTensorSpec(
        shape=(2,), dtype=tf.float32, name='with_name')
    spec_copy = utils.copy_tensorspec(spec, prefix='test')
    # Spec equality does not check the name
    utils.assert_equal(spec, spec_copy)
    self.assertEqual(spec_copy.none_name.name, 'test/')
    self.assertEqual(spec_copy.with_name.name, 'test/with_name')

  @parameterized.parameters(*PARAMS)
  def test_make_placeholders(self, collection_type):
    spec = self._make_tensorspec_collection(collection_type)
    placeholders = utils.make_placeholders(spec)
    placeholder_spec = utils.tensorspec_from_tensors(placeholders)
    utils.assert_equal(
        spec, placeholder_spec, ignore_batch=True)
    with self.assertRaises(ValueError):
      utils.assert_equal(spec, placeholder_spec, ignore_batch=False)

  def _make_tensorspec_collection(self, collection_type):
    if collection_type == 'list':
      return [T1, T2, T3, O4, T5, S7]
    elif collection_type == 'tuple':
      return (T1, T2, T3, O4, T5, S7)
    elif collection_type == 'dict':
      return {'t1': T1, 't2': T2, 't3': T3, 't4': O4, 't5': T5, 's7': S7}
    elif collection_type == 'namedtuple':
      return MockFoo(MockBar(T1, T2), MockBar(T1, T2))

  def _write_test_examples(self, data_of_lists, file_path):
    writer = tf.python_io.TFRecordWriter(file_path)
    for data in data_of_lists:
      example = tf.train.Example()
      example.features.feature['varlen'].int64_list.value.extend(data)
      writer.write(example.SerializeToString())
    writer.close()

  def test_pad_sparse_tensor_to_spec_shape(self):
    varlen_spec = utils.ExtendedTensorSpec(
        shape=(3,), dtype=tf.int64, name='varlen', varlen_default_value=3.0)
    tmp_dir = self.create_tempdir().full_path
    file_path_padded_to_size_two = os.path.join(tmp_dir, 'size_two.tfrecord')
    test_data = [[1], [1, 2]]
    self._write_test_examples(test_data, file_path_padded_to_size_two)
    dataset = tf.data.TFRecordDataset(
        filenames=tf.constant([file_path_padded_to_size_two]))
    dataset = dataset.batch(len(test_data), drop_remainder=True)

    def parse_fn(example):
      return tf.parse_example(example, {'varlen': tf.VarLenFeature(tf.int64)})

    dataset = dataset.map(parse_fn)
    sparse_tensors = dataset.make_one_shot_iterator().get_next()['varlen']
    tensor = utils.pad_sparse_tensor_to_spec_shape(sparse_tensors, varlen_spec)
    with self.session() as sess:
      np_tensor = sess.run(tensor)
      self.assertAllEqual(np_tensor, np.array([[1, 3, 3], [1, 2, 3]]))

  def test_pad_sparse_tensor_to_spec_shape_raises(self):
    varlen_spec = utils.ExtendedTensorSpec(
        shape=(3,), dtype=tf.int64, name='varlen', varlen_default_value=3.0)
    tmp_dir = self.create_tempdir().full_path
    file_path_padded_to_size_two = os.path.join(tmp_dir, 'size_two.tfrecord')
    # This will raise because the desired max shape is 3 but we create an
    # example with shape 4.
    test_data = [[1, 2, 3, 4]]
    self._write_test_examples(test_data, file_path_padded_to_size_two)
    dataset = tf.data.TFRecordDataset(
        filenames=tf.constant([file_path_padded_to_size_two]))
    dataset = dataset.batch(len(test_data), drop_remainder=True)

    def parse_fn(example):
      return tf.parse_example(example, {'varlen': tf.VarLenFeature(tf.int64)})

    dataset = dataset.map(parse_fn)
    sparse_tensors = dataset.make_one_shot_iterator().get_next()['varlen']
    tensor = utils.pad_sparse_tensor_to_spec_shape(sparse_tensors, varlen_spec)
    with self.session() as sess:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(tensor)

  def test_varlen_default_value_raise(self):
    with self.assertRaises(ValueError):
      # This raises since only rank 1 tensors are supported for varlen.
      utils.ExtendedTensorSpec(
          shape=(3, 2), dtype=tf.int64, name='varlen', varlen_default_value=3.0)


if __name__ == '__main__':
  tf.test.main()
