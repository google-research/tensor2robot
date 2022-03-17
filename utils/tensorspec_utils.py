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
"""Manipulating (nested) collections of TensorSpec objects."""

import collections
import pprint
from typing import Any, Dict, List, Optional, Text, Union

from absl import logging
import gin
import numpy as np
from six.moves import cPickle
from six.moves import zip
from tensor2robot.proto import t2r_pb2
import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from tensorflow.contrib import framework as contrib_framework

nest = contrib_framework.nest
TSPEC = contrib_framework.TensorSpec

EXTRA_ASSETS_DIRECTORY = 'assets.extra'
T2R_ASSETS_FILENAME = 't2r_assets.pbtxt'


class ExtendedTensorSpec(TSPEC, object):
  """Extension to TensorSpec to suport optional tensors and data formats.

  An ExtendedTensorSpec allows an API to describe the Tensors that it accepts or
  returns, before that Tensor exists. This allows dynamic and flexible graph
  construction and configuration. Compared to a TensorSpec an ExtendedTensorSpec
  adds the additional fields is_optional and data_format.
  """

  __slots__ = ('_is_optional', '_is_sequence', '_is_extracted', '_data_format',
               '_dataset_key', '_varlen_default_value')

  def __init__(self,
               shape,
               dtype,
               name = None,
               is_optional = None,
               is_sequence = False,
               is_extracted = False,
               data_format = None,
               dataset_key = None,
               varlen_default_value = None):
    """Creates a TensorSpec.

    Args:
      shape: Value convertible to `tf.TensorShape`. The shape of the tensor.
      dtype: Value convertible to `tf.DType`. The type of the tensor values.
      name: Optional name for the Tensor.
      is_optional: If True, the tensor is optional, required otherwise.
      is_sequence: If True, interpret as tf.FixedLenSequenceFeature instead of
        tf.FixedLenFeature.
      is_extracted: If True, implies this spec was inferred from a Tensor or
        np.array.
      data_format: Optional name of the data_format, e.g. jpeg, png.
      dataset_key: Optional key name of which dataset to pull this tensor from.
      varlen_default_value: Optional if a value other than None is provided
        the spec is assumed to be a VarLenFeature with the default value in the
        corrensponding data type. When using a VarLenFeature, the 0th index in
        the shape corresponds to the length that the feature will be padded or
        clipped to. When padded, the varlen_default_value will be used for
        padding. When clipped, some data might be ignored.

    Raises:
      TypeError: If shape is not convertible to a `tf.TensorShape`, or dtype is
        not convertible to a `tf.DType`.
    """
    super(ExtendedTensorSpec, self).__init__(
        shape=shape, dtype=dtype, name=name)
    if is_optional is None:
      is_optional = False
    self._is_optional = is_optional
    self._is_sequence = is_sequence
    self._is_extracted = is_extracted
    self._data_format = data_format
    if dataset_key is None:
      dataset_key = ''
    self._dataset_key = dataset_key
    self._varlen_default_value = varlen_default_value
    if self._varlen_default_value is not None:
      if data_format is None and len(self.shape) != 1:
        raise ValueError(
            ('VarLenFeatures are only supported for shapes of rank 1 ({}) when '
             'not using an image spec.').format(shape))
      if data_format is not None and len(self.shape) != 4:
        raise ValueError(
            ('VarLenFeatures are only supported for shapes of rank 4 ({}) when '
             'using an image spec.').format(shape))

  @classmethod
  def from_spec(cls,
                spec,
                shape=None,
                dtype=None,
                name = None,
                is_optional = None,
                is_sequence = None,
                is_extracted = None,
                data_format = None,
                dataset_key = None,
                batch_size = None,
                varlen_default_value = None):
    if not (isinstance(spec, TSPEC) or isinstance(spec, ExtendedTensorSpec)):
      raise ValueError('from_spec requires TensorSpec or ExtendedTensorSpec.')

    if is_optional is None:
      is_optional = getattr(spec, 'is_optional', False)

    if is_sequence is None:
      is_sequence = getattr(spec, 'is_sequence', False)

    if is_extracted is None:
      is_extracted = getattr(spec, 'is_extracted', False)

    if data_format is None:
      # To support the usage of the base TensorSpec.
      if hasattr(spec, 'data_format'):
        data_format = spec.data_format

    if dataset_key is None:
      dataset_key = getattr(spec, 'dataset_key', '')

    if shape is None:
      shape = spec.shape

    if batch_size:
      if not isinstance(batch_size, int):
        raise ValueError('batch_size must be an integer.')
      if batch_size < 0:
        # In order to set an optional batch_size we pass a negative batch_size
        # and convert it to None. We cannot pass None directly since we need
        # the default in order to not alter a batch_size when we copy a spec.
        shape = tf.TensorShape([None] + shape.as_list())
      else:
        shape = tf.TensorShape([batch_size] + shape.as_list())

    if varlen_default_value is None:
      varlen_default_value = getattr(spec, 'varlen_default_value', None)
    return cls(shape, dtype or spec.dtype, name or spec.name, is_optional,
               is_sequence, is_extracted, data_format, dataset_key,
               varlen_default_value)

  @classmethod
  def from_tensor(cls,
                  tensor,
                  name = None):
    # Check if the tensor is a Eager tensor for which the type is tf internal
    # and therefore can only be checked indirectly.
    if hasattr(tensor, 'numpy'):
      return ExtendedTensorSpec(
          tensor.shape, tensor.dtype, name, is_extracted=True)
    elif isinstance(tensor, tf.Tensor):
      return ExtendedTensorSpec(
          tensor.shape, tensor.dtype, name or tensor.op.name, is_extracted=True)
    else:
      raise ValueError('`tensor` should be a tf.Tensor')

  @classmethod
  def from_proto(cls, extended_tensor_spec_proto):
    # Fill the required fields.
    kwargs = {
        'shape': extended_tensor_spec_proto.shape,
        'dtype': tf.DType(extended_tensor_spec_proto.dtype),
    }
    # Fill the optional fields.
    for optional_field in [
        'name', 'is_optional', 'is_extracted', 'data_format', 'dataset_key',
        'varlen_default_value'
    ]:
      if extended_tensor_spec_proto.HasField(optional_field):
        kwargs[optional_field] = getattr(extended_tensor_spec_proto,
                                         optional_field)
    return cls(**kwargs)

  def to_proto(self):
    extended_tensor_spec_proto = t2r_pb2.ExtendedTensorSpec()
    extended_tensor_spec_proto.shape.extend(self.shape)
    extended_tensor_spec_proto.dtype = self.dtype.as_datatype_enum
    if self.name is not None:
      extended_tensor_spec_proto.name = self.name
    if self.is_optional is not None:
      extended_tensor_spec_proto.is_optional = self.is_optional
    if self.is_extracted is not None:
      extended_tensor_spec_proto.is_extracted = self.is_extracted
    if self.data_format is not None:
      extended_tensor_spec_proto.data_format = self.data_format
    if self.varlen_default_value is not None:
      extended_tensor_spec_proto.varlen_default_value = (
          self.varlen_default_value)
    return extended_tensor_spec_proto

  @classmethod
  def from_serialized_proto(cls, serialized_extended_tensor_spec_proto):
    extended_tensor_spec_proto = t2r_pb2.ExtendedTensorSpec()
    extended_tensor_spec_proto.ParseFromString(
        serialized_extended_tensor_spec_proto)  # pytype: disable=wrong-arg-types
    return cls.from_proto(extended_tensor_spec_proto)

  @classmethod
  def to_spec(cls, instance):
    if isinstance(instance, tf.Tensor):
      return ExtendedTensorSpec.from_tensor(instance)
    elif isinstance(instance, TSPEC):
      return ExtendedTensorSpec.from_spec(instance)
    elif isinstance(instance, np.ndarray):
      return ExtendedTensorSpec(
          shape=instance.shape, dtype=tf.as_dtype(instance.dtype),
          is_extracted=True)
    raise ValueError(
        'We cannot convert {} with type {} to ExtendedTensorSpec'.format(
            instance, type(instance)))

  @property
  def is_optional(self):
    """Returns if the tensor is optional or required."""
    return self._is_optional

  @property
  def is_sequence(self):
    """Returns if the tensor is a variable-length sequence."""
    return self._is_sequence

  @property
  def is_extracted(self):
    """Returns if the TensorSpec was extracted from a Tensor or np.array."""
    return self._is_extracted

  @property
  def data_format(self):
    """Returns the `data_format` of the tensor."""
    return self._data_format

  @property
  def dataset_key(self):
    """Returns the `dataset_key` of the tensor."""
    return self._dataset_key

  @property
  def varlen_default_value(self):
    """Returns the `varlen_default_value` of the tensor."""
    return self._varlen_default_value

  def __eq__(self, other):
    return (self._shape == other._shape  # pylint: disable=protected-access
            and self.dtype == other.dtype)

  def __repr__(self):
    return ('ExtendedTensorSpec(shape={}, dtype={}, name={}, is_optional={}, '
            'is_sequence={}, is_extracted={}, data_format={}, '
            'dataset_key={}, varlen_default_value={})').format(
                self.shape, repr(self.dtype), repr(self.name),
                repr(self.is_optional), repr(self.is_sequence),
                repr(self.is_extracted), repr(self.data_format),
                repr(self.dataset_key), repr(self.varlen_default_value))

  def __reduce__(self):
    return ExtendedTensorSpec, (self._shape, self._dtype, self._name,
                                self._is_optional, self._is_sequence,
                                self._is_extracted, self._data_format,
                                self._dataset_key, self._varlen_default_value)


class _OrderedDictKeysView(collections.abc.KeysView):

  def __reversed__(self):
    for key in reversed(self._mapping):  # pytype: disable=attribute-error
      yield key


class _OrderedDictItemsView(collections.abc.ItemsView):

  def __reversed__(self):
    for key in reversed(self._mapping):  # pytype: disable=attribute-error
      yield (key, self._mapping[key])  # pytype: disable=attribute-error


class _OrderedDictValuesView(collections.abc.ValuesView):

  def __reversed__(self):
    for key in reversed(self._mapping):  # pytype: disable=attribute-error
      yield self._mapping[key]  # pytype: disable=attribute-error


class TensorSpecStruct(collections.OrderedDict):
  """Extension to OrderedDict to allow mutable attribute access.

  Tensorflow supports, dicts, namedtuples, lists and nested versions
  thereof as inputs to models and usage within tf.data.
  TPUEstimator only support flattened data structures. Estimator models
  make extensive use of defining specs using (nested) namedtuples. This
  new datastructure, used in flattened spec structures maintains (nested)
  attribute access and allows mutable access to members. It internally operates
  on the flattened OrderedDict and is therefore fully compatible with the
  whole tensorflow ecosystem and the estimator models infrastructure.

  In the following we illustrate the behavior of this data structure by example.

  existing_ordered_dict = collections.OrderedDict([
    ('train/images', data),
    ('train/actions', data),
    ('test/images', data),
    ('test/actions', data),
    ('magic', data),
  ])
  Here, data should be any of TensorSpec, numpy array or Tensor.

  ordered_dict_with_attr = TensorSpecStruct(existing_ordered_dict)
  train = ordered_dict_with_attr.train
  # train is a new TensorSpecStruct which operates on the original
  # ordered_dict_with_attr data and exposes 'images' and 'actions' from train.

  # We can now add new specs to train.
  train.additional = ExtendedTensorSpec(...)
  # This results in the additional key for
  # train.keys() = ['images', 'actions', 'additional']
  # and also for
  # ordered_dict_with_attr.keys() = ['train/images', 'train/actions',
  # 'test/images', 'test/actions', 'train/additional']

  # We can also delete elements by removing them from the train view
  del train['images']
  # This results in
  # train.keys() = ['actions', 'additional']
  # and also for
  # ordered_dict_with_attr.keys() = ['train/actions',
  # 'test/images', 'test/actions', 'train/additional']
  # Now the following access will raise an AttributeError
  train.images

  # or from the toplevel instance
  del ordered_dict_with_attr['train/actions']
  # This results in
  # train.keys() = ['additional']
  # and also for
  # ordered_dict_with_attr.keys() = [
  # 'test/images', 'test/actions', 'train/additional']
  # Now the following access will raise an AttributeError
  train.actions

  # In the same way we can overwrite existing values with TensorSpec, Tensor,
  # or np.ndarray
  train.additional = np.random.random_sample()
  # Note both views always maintain the same data.
  assertEqual(train.additional, ordered_dict_with_attr['train/additional'])
  """

  def __init__(self, *args, **kwargs):
    # The path prefix is used to provide the different "views" during attribute
    # access. We do not expose this parameter as an explicit named parameter
    # since we do not want to break the OrderedDict interface and it is only
    # for internal use.
    self._path_prefix = ''
    if '__internal_path_prefix' in kwargs:
      self._path_prefix = kwargs.pop('__internal_path_prefix')
    # We use this member to ensure that we always operate on the instance
    # which we have instated, this is not exposed to the user and should not
    # be altered. It is only for internals.
    self._dict_view = None
    if '__internal_dict_view' in kwargs:
      self._dict_view = kwargs.pop('__internal_dict_view')

    # We call the super constructor last since we overwrite __getattr__ which
    # is called by the constructor. If we already introduce our internal
    # data members it does not trigger __getattr__ and we do not have to create
    # extra cases.
    super(TensorSpecStruct, self).__init__(*args, **kwargs)

    if self._dict_view is not None:
      # We keep a local copy of the spec in order to allow for fast c-level
      # access required for nest.flatten.
      # Note, we never operate on it other then providing it to the c interface.
      # It is completely in sync with the dict_view though.
      for key in self._dict_view_keys():
        super(TensorSpecStruct, self).__setitem__(key, self._dict_view[key])

    # Dicts can be initialized with key_value pairs. In order to support this
    # functionality we assign these pairs to the internal representation.
    # Note, the assignment operator will perform additional checks ensuring
    # that only valid structures are passed.
    for key, value in kwargs.items():
      if not key.startswith('_'):
        self[key] = value

  def to_dict(self):
    """Create a new `shallow` dict instance.

    Returns:
      A `shallow` dict copy of the current instance.
    """
    return dict(list(self.items()))

  @classmethod
  def from_proto(cls, tensor_spec_struct_proto):
    return cls({
        k: ExtendedTensorSpec.from_proto(v)
        for k, v in tensor_spec_struct_proto.key_value.items()
    })

  @classmethod
  def from_serialized_proto(cls, serialized_tensor_spec_struct_proto):
    tensor_spec_struct_proto = t2r_pb2.TensorSpecStruct()
    tensor_spec_struct_proto.ParseFromString(
        serialized_tensor_spec_struct_proto)  # pytype: disable=wrong-arg-types
    return cls.from_proto(tensor_spec_struct_proto)

  def to_proto(self):
    """Converts the TensorSpecStruct to a proto."""
    t2r_tensor_spec_struct = t2r_pb2.TensorSpecStruct()
    for key, value in self.items():
      if not hasattr(value, 'to_proto'):
        raise ValueError('Only data structures which support to_proto, e.g.'
                         'ExtendedTensorSpec are allowed within a '
                         'TensorSpecStruct when convertig to a proto. The type '
                         'for key {} is however {} with the value {}.'.format(
                             key, type(value), value))
      t2r_tensor_spec_struct.key_value[key].CopyFrom(value.to_proto())
    return t2r_tensor_spec_struct

  def __getitem__(self, key):
    if key.startswith('_'):
      return super(TensorSpecStruct, self).__getitem__(key)

    # If we want to access an element of the flat internal ordered dict we
    # need to add the path prefix.
    key_with_path_prefix = self._add_path_prefix(key)

    # After checking if this is the main instance or a proxy, we check if
    # the dict contains the key. If this is the case we can return.
    if self._dict_view is None:
      if key_with_path_prefix in self:
        # We are the main instance, hence we can use the normal implementation.
        return super(TensorSpecStruct,
                     self).__getitem__(key_with_path_prefix)
    elif key_with_path_prefix in self._dict_view:
      # All non top level instances, basically attributes accesses operate on
      # the view.
      return self._dict_view[key_with_path_prefix]

    # In the absence of the key, we are NOT done. We want to have a hierarchical
    # access which is important for attributes but we also want to expose this
    # feature for item access.

    # In order to allow attribute like access we construct a hierarchy
    # of dicts from our flat version.
    hierarchy = self._create_hierarchy(list(self.items()))

    # In case a hierarchical key is passed, e.g. 'train/actions' we iterate
    # through the hierarchy.
    key_split_into_hierarchy = key.split('/')

    while key_split_into_hierarchy:
      key_current = key_split_into_hierarchy.pop(0)
      if key_current not in hierarchy:
        raise AttributeError(
            'No attribute with the name {} exists for {}'.format(
                key_current, self))

      hierarchy = hierarchy[key_current]

    if isinstance(hierarchy, dict):
      # Note, the top level instance represents our data and has no internal
      # dict view thus we provide self.
      return TensorSpecStruct(
          __internal_path_prefix=self._add_path_prefix(key),
          __internal_dict_view=self._dict_view or self)

    # We are a leaf, hence, we can return the data.
    return hierarchy

  def __setitem__(self, key, value):
    if key.startswith('_'):
      super(TensorSpecStruct, self).__setitem__(key, value)

    value = self._check_valid_types_for_assignment(value)
    # We keep a local copy of the spec in order to allow for fast c-level access
    # required for nest.flatten.
    if isinstance(value, (TensorSpecStruct, dict)):
      for sub_key, sub_value in value.items():
        # Note, do not add the path prefix here since it is a recursion.
        self[key + '/' + sub_key] = sub_value
    else:
      # Note, add the path prefix here since it is the end of the recursion.
      super(TensorSpecStruct,
            self).__setitem__(self._add_path_prefix(key), value)
    if self._dict_view is not None:
      # All non top level instances, basically attributes accesses operate on
      # the view.
      self._add_to_dict(self._dict_view, key, value)

  def _add_to_dict(self, dict_instance, name, item):
    item = self._check_valid_types_for_assignment(item)
    if isinstance(item, (TensorSpecStruct, dict)):
      for key, value in item.items():
        # Note, do not add the path prefix here since it is a recursion.
        self._add_to_dict(dict_instance,
                          name + '/' + key, value)
    else:
      # Note, add the path prefix here since it is the end of the recursion.
      dict_instance[self._add_path_prefix(name)] = item

  def __len__(self):
    if self._dict_view is None:
      return super(TensorSpecStruct, self).__len__()
    # Note, dict_view_keys already filters with the path_prefix resulting
    # in the proper length for our dict properties.
    return len(self._dict_view_keys())

  def __iter__(self, *args, **kwargs):
    if self._dict_view is None:
      # We simply yield through all keys, the default implementation.
      for value in super(TensorSpecStruct, self).__iter__(
          *args, **kwargs):
        yield value
    else:
      # We are a view, hence we want to only show our part of the path and
      # strip the prefix away.
      for key in self._dict_view_keys():
        yield self._strip_path_prefix(key)

  def _dict_view_keys(self):
    """A simple filter for the dict_view to only get our keys.

    Returns:
      A list of valid keys.
    """
    return [
        key for key in self._dict_view.keys()
        if key.startswith(self._path_prefix + '/')
    ]

  def _strip_path_prefix(self, path):
    """Strip away the prefix from our path keys.

    Args:
      path: A path like str.

    Returns:
      The prefix properly stripped away.
    """
    return path[len(self._path_prefix):].lstrip('/')

  def _add_path_prefix(self, path):
    """Add the path prefix to our path.

    Args:
      path: A path like str.

    Returns:
      The prefix properly stripped away.
    """
    if self._path_prefix:
      return self._path_prefix + '/' + path
    return path

  def _check_valid_types_for_assignment(self, item):
    """Checks that we only have valid types for assignment.

    Args:
      item: An item which will be assigned to the
        TensorSpecStruct.

    Raises:
      ValueError: If the data type is not a tf.Tensor, a TensorSpec, an
        np.ndarray, a not empty TensorSpecStruct or a not empty
        dict.

    Returns:
      The same item in case it meets all requirements for all but named_tuples,
      this data structure has to be converted to a dictionnary for further
      processing.
    """
    if item is None:
      return item
    if isinstance(item, tuple) and hasattr(item, '_asdict'):
      # namedtuple is not a type so we check that the tuple implements the
      # minimal namedtuple interface we need, namely the _asdict() function.
      item = item._asdict()

    # Note, the types
    # tf.data.experimental.TensorStructure, tf.DType, tf.TensorShape, type
    # are used within tf.data and have therefore to be supported.
    # This is not mentioned in user facing error messages since it is not
    # encouraged.

    # if not isinstance(item, (tf.Tensor, TSPEC, np.ndarray, TensorSpecStruct,
    #                          dict, tf.data.experimental.TensorStructure,
    #                          tf.DType, tf.TensorShape, type, List)):
    #   raise ValueError(
    #       'Only tensors, TensorSpecs, numpy arrays, NamedTuples '
    #       'TensorSpecStruct, and dicts can be assigned. but item '
    #       'is {} with type {}.'.format(item, type(item)))

    if isinstance(item, TensorSpecStruct):
      if not item:
        raise ValueError(
            'We cannot assign an empty TensorSpecStruct. '
            'Please, first fill another TensorSpecStruct and '
            'add it to this instance.')
    if isinstance(item, dict):
      if not item:
        raise ValueError('We cannot assign an empty dict.')
    return item

  def __delitem__(self, key):
    path_key = self._add_path_prefix(key)
    if self._dict_view is not None:
      if path_key in self._dict_view:
        del self._dict_view[path_key]
      if path_key in self:
        super(TensorSpecStruct, self).__delitem__(path_key)
    else:
      super(TensorSpecStruct, self).__delitem__(path_key)

  def __setattr__(self, name, item):
    if name.startswith('_'):
      return super(TensorSpecStruct, self).__setattr__(name, item)
    self[name] = item

  def __repr__(self):
    return 'TensorSpecStruct(\n' + pprint.pformat(self.to_dict()) + ')'

  def __getattr__(self, item):
    if item.startswith('_'):
      # Maintain the normal behavior of the dict required for proper
      # initializing the OrderedDict.
      raise AttributeError('The attribute {} does not exist.'.format(item))
    return self[item]

  def _create_hierarchy(self, items):
    """Create a hierarchy of ordered dicts from the provided items.

    Args:
      items: Items returned from a dict-like structure.

    Returns:
      hierarchy: A hierarchy of OrderedDict inferred from the keys,
        split by '/'.
    """
    hierarchy = collections.OrderedDict()
    for key, value in items:
      current_hierarchy = hierarchy
      key_hierarchy = key.split('/')
      while len(key_hierarchy) > 1:
        # We create a hierarchy of OrderedDict if they do not yet exist.
        current_key = key_hierarchy.pop(0)
        current_hierarchy[current_key] = current_hierarchy.get(
            current_key, collections.OrderedDict())
        current_hierarchy = current_hierarchy[current_key]
      current_hierarchy[key_hierarchy[0]] = value
    return hierarchy

  def keys(self):
    """D.keys() -> a set-like object providing a view on D's keys."""
    # Note: returns list to support Python2-style indexing compatibility.
    return list(_OrderedDictKeysView(self))  # pytype: disable=wrong-arg-count

  def items(self):
    """D.items() -> a set-like object providing a view on D's items."""
    return list(_OrderedDictItemsView(self))  # pytype: disable=wrong-arg-count

  def values(self):
    """D.values() -> an object providing a view on D's values."""
    return list(_OrderedDictValuesView(self))  # pytype: disable=wrong-arg-count


@gin.configurable
def convert_to_tensorspecstruct(inputs):
  return TensorSpecStruct(inputs)


def replace_dtype(tensor_spec_struct, from_dtype,
                  to_dtype):
  """Replaces all elements of type from_dtype with to_dtype.

  This functionality is useful for TPU training since it is most efficient with
  bfloat16 whereas preprocessing on CPU only operates on float32.

  Args:
    tensor_spec_struct: The instance of TensorSpecStruct which will be updated
      in-place.
    from_dtype: The dtype which will be replaced.
    to_dtype: The target dtype.

  Returns:
    The in-place updated TensorSpecStruct.
  """
  for key, value in tensor_spec_struct.items():
    if value.dtype == from_dtype:
      tensor_spec_struct[key] = ExtendedTensorSpec.from_spec(
          spec=value, dtype=to_dtype)
  return tensor_spec_struct


def cast_float32_to_bfloat16(tensor_spec_struct,
                             output_spec):
  """Casts tensors with dtype float32 to bfloat16 depending on the out spec.

  Args:
    tensor_spec_struct: The instance of TensorSpecStruct which will be updated
      in-place.
    output_spec: The reference TensorSpecStruct which allows to infer which
      tensors should be cast to bfloat16.

  Returns:
    The in-place updated TensorSpecStruct.
  """
  for key, value in output_spec.items():
    if value is not None and value.dtype == tf.bfloat16:
      if tensor_spec_struct[key].dtype != tf.float32:
        raise ValueError(
            'Attempting to convert non tf.float32 type {} to tf.bfloat16 '
            'for the element {} with the name {}.'.format(
                tensor_spec_struct[key].dtype, tensor_spec_struct[key], key))
      tensor_spec_struct[key] = tf.cast(
          tensor_spec_struct[key], dtype=tf.bfloat16)
  return tensor_spec_struct


def cast_bfloat16_to_float32(
    tensor_spec_struct):
  """Casts tensors with dtype bfloat16 to float32.

  Args:
    tensor_spec_struct: The instance of TensorSpecStruct which will be updated
      in-place.

  Returns:
    The in-place updated TensorSpecStruct.
  """
  for key, value in tensor_spec_struct.items():
    if value is not None and value.dtype == tf.bfloat16:
      tensor_spec_struct[key] = tf.cast(value, dtype=tf.float32)
  return tensor_spec_struct


def copy_tensorspec(spec_structure,
                    prefix = '',
                    batch_size = None):
  """Returns a copy of the namedtuple with tensor names having a new prefix.

  Args:
    spec_structure: A dict, (named)tuple, list or a hierarchy thereof filled by
      TensorSpecs(subclasses) or Tensors.
    prefix: A string scope to prepend to all copied TensorSpec names.
    batch_size: Optional (int) batch size to prepend to tensor specs.

  Returns:
    A copy of spec_structure, with names possibly renamed.
  """
  assert_valid_spec_structure(spec_structure)
  if prefix:
    prefix += '/'

  def map_spec(spec):
    # String concatenate does not work with None.
    name = spec.name
    if name is None:
      name = ''
    return spec.from_spec(spec, name=prefix + name, batch_size=batch_size)

  return nest.map_structure(map_spec, spec_structure)


def make_placeholders(spec_structure, batch_size = None):
  """Create placeholder equivalents of spec_structure.

  Args:
    spec_structure: A dict, (named)tuple, list or a hierarchy thereof filled by
      TensorSpecs(subclasses).
    batch_size: If None, we will have a flexible shape (None,) + shape. If <= 0
      we will omit an explicit batch dimension and otherwise have a fixed
      (batch_size,) + shape.

  Returns:
    Equivalent structure as spec_structure, with TensorSpecs converted to
    placeholders with variable batch size.
  """

  # TODO(kappler, b/115347601): Create a function which maps placeholders
  #   to tensorspecs. This is one of the two place where this functionality
  #   needs to be added.

  assert_valid_spec_structure(spec_structure)

  def make_placeholder(t):
    t = ExtendedTensorSpec.from_spec(t)
    shape = tuple(t.shape.as_list())
    if t.is_sequence:
      shape = (None,) + shape
    if batch_size is None:
      shape = (None,) + shape
    elif batch_size > 0:
      shape = (batch_size,) + shape
    return tf.placeholder(t.dtype, shape, name=t.name)
  return nest.map_structure(make_placeholder, spec_structure)


def make_random_tensors(spec_structure, batch_size = 2):
  """Create random inputs for tensor_spec (for unit testing).

  Args:
    spec_structure: A dict, (named)tuple, list or a hierarchy thereof filled by
      TensorSpecs(subclasses).
    batch_size: If None, we will have a flexible shape (None,) + shape. If <= 0
      we will omit an explicit batch dimension and otherwise have a fixed
      (batch_size,) + shape.

  Returns:
    Equivalent structure as spec_structure, with TensorSpecs converted to
    placeholders with variable batch size.
  """
  assert_valid_spec_structure(spec_structure)

  def make_random(t):
    maxval = 255 if t.dtype in [tf.uint8, tf.int32, tf.int64] else 1.0
    dtype = tf.int32 if t.dtype == tf.uint8 else t.dtype
    shape = tuple(t.shape.as_list())
    if batch_size is None:
      shape = (None,) + shape
    elif batch_size > 0:
      shape = (batch_size,) + shape
    r = tf.random_uniform(shape, maxval=maxval, dtype=dtype)
    return tf.cast(r, t.dtype)

  return nest.map_structure(make_random, spec_structure)


def make_constant_numpy(spec_structure, constant_value,
                        batch_size = 2,
                        sequence_length = 3):
  """Create constant numpy inputs for tensor_spec (for unit testing).

  We pass 1s for all the input data.

  Args:
    spec_structure: A dict, (named)tuple, list or a hierarchy thereof filled by
      TensorSpecs(subclasses).
    constant_value: The constant value that is used to initialize the data.
    batch_size: If None, we will have a flexible shape (None,) + shape. If <= 0
      we will omit an explicit batch dimension and otherwise have a fixed
      (batch_size,) + shape.
    sequence_length: Appends sequence length batch dimension for specs with
      is_sequence=True. Set None to NOT prepend this sequence dimension (this
      is a workaround to allow MetaExample to consume data for specs that set
      is_sequence=True).

  Returns:
    Equivalent structure as spec_structure, with TensorSpecs converted to
    placeholders with variable batch size.
  """
  assert_valid_spec_structure(spec_structure)

  def make_fixed(t):
    shape = tuple(t.shape.as_list())
    if isinstance(t, ExtendedTensorSpec) and t.is_sequence:
      shape = (sequence_length,) + shape
    if batch_size is None:
      shape = (None,) + shape
    elif batch_size > 0:
      shape = (batch_size,) + shape
    r = np.full(shape, constant_value)
    return r.astype(t.dtype.as_numpy_dtype)

  return nest.map_structure(make_fixed, spec_structure)


def make_random_numpy(spec_structure,
                      batch_size = 2,
                      sequence_length = 3):
  """Create random numpy inputs for tensor_spec (for unit testing).

  Args:
    spec_structure: A dict, (named)tuple, list or a hierarchy thereof filled by
      TensorSpecs(subclasses).
    batch_size: If None, we will have a flexible shape (None,) + shape. If <= 0
      we will omit an explicit batch dimension and otherwise have a fixed
      (batch_size,) + shape.
    sequence_length: Appends sequence length batch dimension for specs with
      is_sequence=True. Set None to NOT prepend this sequence dimension (this
      is a workaround to allow MetaExample to consume data for specs that set
      is_sequence=True).

  Returns:
    Equivalent structure as spec_structure, with TensorSpecs converted to
    placeholders with variable batch size.
  """
  assert_valid_spec_structure(spec_structure)
  def make_random(t):
    """Make a random numpy array from a TensorSpec."""
    maxval = 255 if t.dtype in [tf.uint8, tf.int32, tf.int64] else 1.0
    shape = tuple(t.shape.as_list())
    if isinstance(t, ExtendedTensorSpec) and t.is_sequence:
      shape = (sequence_length,) + shape
    if batch_size is None:
      shape = (None,) + shape
    elif batch_size > 0:
      shape = (batch_size,) + shape
    r = np.random.uniform(size=shape, high=maxval)
    return r.astype(t.dtype.as_numpy_dtype)

  return nest.map_structure(make_random, spec_structure)


def map_feed_dict(spec_placeholders,
                  spec_numpy,
                  feed_dict = None,
                  ignore_batch = False):
  """Map a spec with placeholders to a spec of numpy arrays.

  Note, we will verify that the numpy arrays actually match the required shapes
  of the spec_placeholders.

  Args:
    spec_placeholders: A spec structure of placeholders.
    spec_numpy: A spec structure of numpy arrays.
    feed_dict: (Optional) An existing feed_dict instance which we append
      otherwise we create one.
    ignore_batch: If True, we ignore the batch dimensions for shape comparison.

  Raises:
    ValueError: If we would overwrite an existing placeholder in the feed_dict.
      This might happen if we provide labels and features which use the same
      placeholders, they never should though.

  Returns:
    A feed_dict with a verified {placeholder: np.array} mapping.
  """
  if not is_flat_spec_or_tensors_structure(spec_placeholders):
    spec_placeholders = flatten_spec_structure(spec_placeholders)
  if not is_flat_spec_or_tensors_structure(spec_numpy):
    spec_numpy = flatten_spec_structure(spec_numpy)

  if feed_dict is None:
    feed_dict = {}

  assert_required(
      maybe_ignore_batch(spec_placeholders, ignore_batch),
      maybe_ignore_batch(spec_numpy, ignore_batch))

  for key, value in spec_numpy.items():
    placeholder = spec_placeholders[key]
    if placeholder in feed_dict:
      raise ValueError(
          'We would overwrite existing placeholder mapping {}.'.format(key))
    feed_dict[spec_placeholders[key]] = value
  return feed_dict


def map_predict_fn_dict(spec_structure,
                        spec_numpy,
                        feed_dict = None,
                        ignore_batch = False):
  """Map a spec with placeholders to a spec of numpy arrays.

  Note, we will verify that the numpy arrays actually match the required shapes
  of the spec_structure.

  Args:
    spec_structure: A spec structure of inputs.
    spec_numpy: A {key: value} pair of feature names and numpy arrays.
    feed_dict: (Optional) An existing feed_dict instance which we append
      otherwise we create one.
    ignore_batch: If True, we ignore the batch dimensions for spec_numpy will
      be ignored. This assumes that the spec_structure has no batch_size which
      is true for TFModels.

  Raises:
    ValueError: If we would overwrite an existing placeholder in the feed_dict.
      This might happen if we provide labels and features which use the same
      placeholders, they never should though.

  Returns:
    A feed_dict with a verified {feed_tensor_name: np.array} mapping.
  """
  if not is_flat_spec_or_tensors_structure(spec_numpy):
    spec_numpy = flatten_spec_structure(spec_numpy)
  if feed_dict is None:
    feed_dict = {}

  assert_required(spec_structure, maybe_ignore_batch(spec_numpy, ignore_batch))

  for key, value in spec_numpy.items():
    if key not in spec_structure:
      continue

    if key in feed_dict:
      raise ValueError(
          'We would overwrite existing placeholder mapping {}.'.format(key))
    feed_dict[key] = value
  return feed_dict


def map_feed_dict_unsafe(feature_placeholders_spec, np_inputs_spec):
  """Deprecated function to create a feed_dict to be passed to session.run.

  tensorspec_utils.map_feed_dict should be used instead.  map_feed_dict_unsafe
  does not check that there is actually any agreement between
  feature_placeholders_spec or np_inputs spec in terms of dtype, shape
  or additional unused attributes within np_inputs_spec.

  Args:
    feature_placeholders_spec: An TensorSpecStruct containing
      {str: tf.placeholder}.
    np_inputs_spec: The numpy input according to the same spec.

  Returns:
    A mapping {placeholder: np.ndarray} which can be fed to a tensorflow
      session.run.
  """
  logging.warning('map_feed_dict_unsafe is deprecated. '
                  'Please update to map_feed_dict.')
  flat_spec = flatten_spec_structure(feature_placeholders_spec)
  flat_np_inputs = flatten_spec_structure(np_inputs_spec)
  for key, value in flat_np_inputs.items():
    if key not in flat_spec:
      logging.warn(
          'np_inputs has an input: %s, not found in the tensorspec.', key)
  feed_dict = {}
  for key, value in flat_spec.items():
    feed_dict[value] = flat_np_inputs[key]
  return feed_dict


def tensorspec_from_tensors(tensors):
  """Converts a collection of tensors to a collection of TensorSpec.

  A collection can only be a dict, namedtuple or a hierarchy thereof containing
  tensors or placeholders.

  Args:
    tensors: A dict, (named)tuple, list or a hierarchy thereof filled by
      tensors.

  Returns:
    Equivalent structure of tensors with Tensors replaced with TensorSpec.
  """
  assert_valid_spec_structure(tensors)

  # Every tensor needs to have a unique name. This is a requirement for the
  # spec structure. We use the closure to pass the integer into the map_fn.
  # Note we cannot simply use unique_index = 0 since integers cannot be changed
  # without changing the reference.
  unique_index = [0]

  def map_fn(tensor):
    unique_name = '{}/{}'.format(tensor.op.name, unique_index[0])
    unique_index[0] += 1
    return ExtendedTensorSpec.from_tensor(tensor, unique_name)

  return nest.map_structure(map_fn, tensors)


def maybe_ignore_batch(spec_or_tensors, ignore_batch = False):
  """Optionally strips the batch dimension and returns new spec.

  Args:
    spec_or_tensors: A dict, (named)tuple, list or a hierarchy thereof filled by
      TensorSpecs(subclasses) or Tensors.
    ignore_batch: If True, the spec_or_batch's batch dimensions are ignored for
      shape comparison.

  Returns:
    spec_or_tensors: If ignore_batch=True we return a spec structure with the
      stripped batch_dimension otherwise we return spec_or_tensors.
  """
  if ignore_batch:
    def map_fn(spec):
      if isinstance(spec, np.ndarray):
        spec = tf.convert_to_tensor(spec)
      if isinstance(spec, tf.Tensor):
        return ExtendedTensorSpec.from_tensor(spec[0])
      else:
        return ExtendedTensorSpec.from_spec(spec, shape=spec.shape[1:])
    return nest.map_structure(
        map_fn,
        spec_or_tensors)
  return spec_or_tensors


def assert_equal_spec_or_tensor(expected_spec_or_tensor, actual_spec_or_tensor):
  """Check that our expected and actual specs or tensors are equal.

  We check by using the hash function since this allows to use tensors
  as well as specs since TensorSpec will use self._shape internally.
  Any other access to shape would require additional checks.

  Args:
    expected_spec_or_tensor: A TensorSpec(subclass) or Tensor.
    actual_spec_or_tensor: A TensorSpec(subclass) or Tensor.

  Raises:
    ValueError: If the expected_spec_or_tensor != actual_spec_or_tensor.
  """
  expected_spec = ExtendedTensorSpec.to_spec(expected_spec_or_tensor)
  actual_spec = ExtendedTensorSpec.to_spec(actual_spec_or_tensor)
  # If actual_spec_or_tensor is a tensor/ndarray, actual_spec will have
  # is_sequence=False and the sequence dimension will appear in the shape.
  # To correct this, we drop the leading dimension of actual_spec.
  if (isinstance(expected_spec, ExtendedTensorSpec) and
      expected_spec.is_sequence and
      actual_spec.is_extracted):
    actual_spec = maybe_ignore_batch(actual_spec, ignore_batch=True)

  if expected_spec.dtype != actual_spec.dtype:
    raise ValueError(
        'TensorSpec.dtype {} does not match TensorSpec.dtype {} in specs'
        '\n expected: {}\n actual: {}'.format(
            expected_spec.dtype, actual_spec.dtype, expected_spec, actual_spec))
  if len(expected_spec.shape) != len(actual_spec.shape):
    raise ValueError(
        'TensorSpec.shape {} does not match TensorSpec.shape {} in specs'
        '\n expected: {}\n actual: {}'.format(
            expected_spec.shape, actual_spec.shape, expected_spec, actual_spec))
  for expected_dim, actual_dim in zip(expected_spec.shape, actual_spec.shape):
    if expected_dim is None:
      continue
    if expected_dim != actual_dim:
      raise ValueError(
          'TensorSpec.shape {} does not match TensorSpec.shape {}.'.format(
              expected_spec.shape, actual_spec.shape))


def assert_equal(expected_tensors_or_spec,
                 actual_tensors_or_spec,
                 ignore_batch = False):
  """Asserts two TensorSpecs have the same structure, shapes, dtypes.

  To handle ExtendedTensorSpec that sets is_sequence=True, we ignore sequence
  dimensions in maybe_ignore_batch.

  Args:
    expected_tensors_or_spec: A dict, (named)tuple, list or a hierarchy thereof
      filled by TensorSpecs(subclasses) or Tensors.
    actual_tensors_or_spec: A dict, (named)tuple, list or a hierarchy thereof
      filled by TensorSpecs(subclasses) or Tensors.
    ignore_batch: If True, actual_output_spec's batch dimensions are ignored for
      shape comparison.
  """
  actual_tensors_or_spec = maybe_ignore_batch(actual_tensors_or_spec,
                                              ignore_batch)
  flattened_expected_tensors_or_spec = flatten_spec_structure(
      expected_tensors_or_spec)
  flattened_actual_tensors_or_spec = flatten_spec_structure(
      actual_tensors_or_spec)
  nest.map_structure(assert_equal_spec_or_tensor,
                     flattened_expected_tensors_or_spec,
                     flattened_actual_tensors_or_spec)


def assert_required(expected_spec,
                    actual_tensors_or_spec,
                    ignore_batch = False):
  """Asserts two TensorSpecs have the same structure for required TensorSpecs.

  Args:
    expected_spec: A dict, (named)tuple, list or a hierarchy thereof containing
      TensorSpecs(subclasses)
    actual_tensors_or_spec: A dict, (named)tuple, list or a hierarchy thereof
      filled by TensorSpecs(subclasses) or Tensors.
    ignore_batch: If True, actual_output_spec's batch dimensions are ignored for
      shape comparison.
  """
  # We first make sure that the expected_spec can be mapped to the
  # actual_tensors_or_spec. This way we handle all optional cases.
  flat_actual_spec = flatten_spec_structure(actual_tensors_or_spec)

  # The following packing will raise if we cannot create a structure with
  # all required tensor(specs).
  actual_tensors_or_spec = pack_flat_sequence_to_spec_structure(
      expected_spec, flat_actual_spec)
  # We flatten the result to purge all optional tensors.
  flat_actual_spec = flatten_spec_structure(actual_tensors_or_spec)

  # Since the pack_flat_sequence_to_spec_structure ensures that all required
  # tensors(spec) are available and it drops all optional elements if no
  # data is provided, we filter flat_expected_spec by the available keys in
  # flat_actual_spec.
  flat_expected_spec = flatten_spec_structure(expected_spec)
  flat_expected_spec = {
      key: value
      for key, value in flat_expected_spec.items()
      if key in flat_actual_spec
  }

  # Now we are sure that we have the same structure, thus we can check for
  # equality.
  assert_equal(
      flat_expected_spec, flat_actual_spec, ignore_batch)


def validate_and_flatten(expected_spec,
                         actual_tensors_or_spec,
                         ignore_batch = False):
  """Validate that TensorSpecs (required) are fulfilled and flatten the result.

  Args:
    expected_spec: A dict, (named)tuple, list or a hierarchy thereof filled by
      TensorSpecs(subclasses)
    actual_tensors_or_spec: A dict, (named)tuple, list or a hierarchy thereof
      filled by TensorSpecs(subclasses) or Tensors.
    ignore_batch: If True, actual_output_spec's batch dimensions are ignored for
      shape comparison.

  Returns:
    A flattened sequence with the joined string paths.
  """
  assert_valid_spec_structure(expected_spec)
  assert_valid_spec_structure(actual_tensors_or_spec)
  try:
    assert_required(
        expected_spec, actual_tensors_or_spec, ignore_batch)
  except ValueError as e:
    logging.error(
        'The actual_spec_or_tensor does not fulfill the expected_spec:')
    for key, value in sorted(flatten_spec_structure(expected_spec).items()):
      logging.error('expected_spec: %s: %s', key, value)
    for key, value in sorted(
        flatten_spec_structure(actual_tensors_or_spec).items()):
      logging.error('actual_spec:   %s: %s', key, value)
    raise e

  return flatten_spec_structure(actual_tensors_or_spec)


def validate_and_pack(expected_spec,
                      actual_tensors_or_spec,
                      ignore_batch = False):
  """Validate that TensorSpecs (required) are fulfilled and pack the result.

  Args:
    expected_spec: A dict, (named)tuple, list or a hierarchy thereof filled by
      TensorSpecs(subclasses)
    actual_tensors_or_spec: The actual tensors_or_spec, we support flattened and
      packed tensors or specs.
    ignore_batch: If True, actual_output_spec's batch dimensions are ignored for
      shape comparison.

  Returns:
    A flattened sequence with the joined string paths.
  """
  assert_valid_spec_structure(expected_spec)
  assert_valid_spec_structure(actual_tensors_or_spec)
  if not is_flat_spec_or_tensors_structure(actual_tensors_or_spec):
    actual_tensors_or_spec = flatten_spec_structure(actual_tensors_or_spec)
  try:
    assert_required(
        expected_spec, actual_tensors_or_spec, ignore_batch)
  except ValueError as e:
    logging.error(
        'The actual_spec_or_tensor does not fulfill the expected_spec:')
    for key, value in sorted(flatten_spec_structure(expected_spec).items()):
      logging.error('expected_spec: %s: %s', key, value)
    for key, value in sorted(
        flatten_spec_structure(actual_tensors_or_spec).items()):
      logging.error('actual_spec:   %s: %s', key, value)
    raise e
  return pack_flat_sequence_to_spec_structure(expected_spec,
                                              actual_tensors_or_spec)


def add_sequence_length_specs(
    spec_structure):
  """Augments a TensorSpecStruct with key + '_length' specs."""
  flat_spec_structure = flatten_spec_structure(spec_structure)
  for key, value in flat_spec_structure.items():
    if value.is_sequence:
      flat_spec_structure[key + '_length'] = ExtendedTensorSpec(
          shape=(), dtype=tf.int64, name=value.name + '_length')
  return flat_spec_structure


def filter_spec_structure_by_dataset(
    spec_structure,
    dataset_key,
    filter_none = True):
  """Subset of flattened spec structure whose dataset matches dataset_key."""
  flattened_spec_structure = flatten_spec_structure(spec_structure, filter_none)
  return TensorSpecStruct([
      key_value for key_value in flattened_spec_structure.items()
      if (key_value[1].dataset_key == dataset_key or not dataset_key)
  ])


def flatten_spec_structure(
    spec_structure, filter_none = True):
  """Returns flattened sequence from a given spec_structure.

  This function exists that we always flatten our spec_structures
  joined_string_paths. Note, we only support (hierarchical) dicts or
  namedtuples for spec_structures. This will ensure that we don't flatten
  our spec_structure several times by accident and every TensorSpec has a
  meaningful path.

  Args:
    spec_structure: A dict, (named)tuple, list or a hierarchy thereof filled by
      TensorSpecs(subclasses) or Tensors.
    filter_none: In case of None as a leaf value we will ignore this value when
      flattening. This is important for e.g. tf.data and in general the desired
      behavior.

  Raises:
    ValueError: If the spec_structure is not valid according to
      assert_valid_spec_structure.

  Returns:
    A flattened sequence with the joined string paths as OrderedDict. Since
    we use OrderedDicts we can safely call flatten_spec_structure multiple
    times.
  """
  assert_valid_spec_structure(spec_structure)

  flattened_spec_structure = spec_structure
  if not is_flat_spec_or_tensors_structure(spec_structure):
    flattened_spec_structure = TensorSpecStruct(
        nest.flatten_with_joined_string_paths(spec_structure))

  if not filter_none:
    return flattened_spec_structure

  # We filter None tensors in the flattening. These tensors were optional
  # and we don't want other apis such as tf.data to worry about non existing
  # optional data.
  return TensorSpecStruct([
      value for value in flattened_spec_structure.items()
      if value[1] is not None
  ])


def pack_flat_sequence_to_spec_structure(
    spec_structure, flat_sequence_with_joined_string_paths):
  """Returns a given flattened sequence packed into a given spec_structure.

  We assume the spec_structure is any of a dict, (named)tuple, list or
  potentially hierarchical of instances of TensorSpecs or subclasses like
  ExtendedTensorSpec. For the latter two cases the packed
  return structure might not be filled if the flat_sequence does not provide
  tensors for optional specs. Note, we only support (hierarchical) dicts or
  namedtuples for spec_structures. This will ensure that we don't flatten
  our spec_structure several times by accident and every TensorSpec has a
  meaningful path.

  Args:
    spec_structure: A dict, (named)tuple, list or a hierarchy thereof filled by
      TensorSpecs(subclasses) or Tensors.
    flat_sequence_with_joined_string_paths: A flat sequence with string paths
      which will be mapped to our spec_structure.

  Raises:
    ValueError: If the spec_structure is not valid according to
      assert_valid_spec_structure. We also raise if a required TensorSpec is not
      provided.

  Returns:
    The flat sequence packed to the desired spec_structure. Note, optional
    TensorSpecs might not be filled if the flat_sequence does not provide
    inputs.
  """
  assert_valid_spec_structure(spec_structure)
  # We first flatten our spec in order to allow to pack all tensors specified
  # if available and not optional. This is one of the few cases for which we
  # don't want to filter None since we need to fill the optional tensor values
  # when we pack the spec.
  tensor_spec_struct = flatten_spec_structure(spec_structure, filter_none=False)

  if isinstance(spec_structure, TensorSpecStruct):
    # If a dict-like structure is used nest.pack_sequence_as uses sorted
    # elements which is different for OrderedDict since they respect order,
    # so we have to ensure we have the same mapping.
    tensor_spec_struct = TensorSpecStruct(sorted(tensor_spec_struct.items()))
    # We also have to make sure that the OrderedDictWithAttribute access is
    # indeed sorted.
    spec_structure = tensor_spec_struct

  if not is_flat_spec_or_tensors_structure(
      flat_sequence_with_joined_string_paths):
    raise ValueError('The provided flat_sequence_with_joined_string_paths is '
                     'not a flat sequence {}.'.format(
                         flat_sequence_with_joined_string_paths))

  # We want to query our flat sequence by keys, hence, we convert it to a dict.
  flat_sequence = dict(list(flat_sequence_with_joined_string_paths.items()))

  # Note, we will only need the actual tensors or tensor_specs not the keys.
  filtered_flat_sequence = []
  for key, tensor_spec in tensor_spec_struct.items():
    if key in flat_sequence:
      filtered_flat_sequence.append(flat_sequence[key])
      continue
    # In order to support normal specs we first check if is_optional is defined.
    if (hasattr(tensor_spec, 'is_optional') and tensor_spec.is_optional):
      # We are all good, the tensor was fortunately optional.
      # However, in order to support the pack_sequence_as, we add the tensor
      # with None.
      filtered_flat_sequence.append(None)
      logging.info('The optional TensorSpec %s is not present at %s.',
                   tensor_spec, key)
      continue
    if tensor_spec is None:
      # Optional tensor_specs might be replaced by None, hence we add them in.
      filtered_flat_sequence.append(None)
      continue

    raise ValueError('The required {} spec {} is not available.'.format(
        key, tensor_spec))

  # Now that we have made sure that we only have the required and optional
  # specs in the filtered_flat_sequence
  return nest.pack_sequence_as(spec_structure, filtered_flat_sequence)


def is_flat_spec_or_tensors_structure(spec_or_tensors):
  """Check that the spec_structure or tensor_structure is flattend.

  Args:
    spec_or_tensors: A dict, (named)tuple, list or a hierarchy thereof filled by
      TensorSpecs(subclasses) or Tensors or np.ndarray.

  Returns:
    True in case of a flat structure, meaning (key: Tensor)  or [tensors],
    False otherwise.
  """

  # We only have a tensor_spec_struct_or_tensors if we have a dict or
  # ordered dict with TensorSpec, Tensor or numpy array.
  if isinstance(spec_or_tensors, dict) or isinstance(spec_or_tensors,
                                                     collections.OrderedDict):
    # We have to check any element of our dict or OrderedDict.
    for value in spec_or_tensors.values():
      if isinstance(value, contrib_framework.TensorSpec):
        continue
      if isinstance(value, tf.Tensor):
        continue
      if isinstance(value, np.ndarray):
        continue

      # An unsupported type in our dict, hence, it's not a proper flat
      # structure.
      return False
    return True
  # Not a dict or OrderedDict, hence, not a proper flat structure.
  return False


def assert_valid_spec_structure(
    spec_structure, used_tensorspec_names = None):
  """Check that the spec_structure is a hierarchical dict or namedtuple.

  Note, we only support (hierarchical) dicts or namedtuples for
  spec_structures. This will ensure that we don't flatten our spec_structure
  several times by accident and every TensorSpec has a meaningful path.

  Args:
    spec_structure: A dict, (named)tuple, list or a hierarchy thereof filled by
      TensorSpecs(subclasses) or Tensors.
    used_tensorspec_names: This parameter is used for recursion, therefore, it
      should not be set by a user. A dict of all
      {tensorspec_name: tensorspec} used so far.

  Raises:
    ValueError: If the spec_structure is neither a dict nor a namedtuple or
      a hierarchy thereof, filled with TensorSpecs or subclasses.
  """
  # If we don't have a set used_tensorspec_names that means we are at the
  # top level and need to be initialized.
  if used_tensorspec_names is None:
    used_tensorspec_names = dict()
  if isinstance(spec_structure, tuple) and hasattr(spec_structure, '_asdict'):
    # namedtuple is not a type so we check that the tuple implements the
    # minimal namedtuple interface we need, namely the _asdict() function.
    spec_structure = spec_structure._asdict()

  if isinstance(spec_structure, dict) or isinstance(spec_structure,
                                                    collections.OrderedDict):
    # We have to check any element of our dict or OrderedDict. Since we also
    # support lists, we can simply introspect the values.
    spec_structure = list(spec_structure.values())

  if isinstance(spec_structure, list) or isinstance(spec_structure, tuple):
    for value in spec_structure:
      if isinstance(value, contrib_framework.TensorSpec):
        # We only add non None TensorSpec names. These names have to be unique
        # within all specified TensorSpec names used so far in this
        # spec_structure.
        if value.name is not None:
          if value.name in used_tensorspec_names:
            try:
              assert_equal_spec_or_tensor(used_tensorspec_names[value.name],
                                          value)
            except ValueError:
              raise ValueError(
                  'All TensorSpecs with a name defined have to be unique '
                  'or non unique specs have to define the same shape and dtype'
                  'within our spec_structure. Yet, the name {} is defined '
                  'twice and describes different specs {} vs {}.'.format(
                      value.name, value, used_tensorspec_names[value.name]))
          used_tensorspec_names[value.name] = value
        continue
      if isinstance(value, tf.Tensor):
        continue
      if isinstance(value, np.ndarray):
        continue
      # We support None which is filled automatically for optional tensors,
      # if not available.
      if value is None:
        continue
      assert_valid_spec_structure(value, used_tensorspec_names)
    return

  raise ValueError('We only support spec_structures of (hierarchical) '
                   'dicts or namedtuples, not {}.'.format(type(spec_structure)))


def filter_required_flat_tensor_spec(flat_tensor_spec):
  """Process a flat tensor spec structure and return only the required subset.

  Args:
    flat_tensor_spec: A flattened sequence (result of flatten_spec_structure)
      with the joined string paths as OrderedDict. Since we use OrderedDicts we
      can safely call flatten_spec_structure multiple times.

  Raises:
    ValueError: If the passed flat_tensor_spec is not a valid flat tensor_spec
      structure.

  Returns:
    filtered_flat_required_tensor_spec: The same flattened sequence but only
      the {key: tensor_spec} pairs for the non optional tensor_spec.
  """
  if not is_flat_spec_or_tensors_structure(flat_tensor_spec):
    raise ValueError('Only flat tensor_spec structures are allowed.')
  filtered_flat_required_tensor_spec = TensorSpecStruct()
  for key, value in flat_tensor_spec.items():
    if hasattr(value, 'is_optional') and value.is_optional:
      continue
    filtered_flat_required_tensor_spec[key] = value
  return filtered_flat_required_tensor_spec


def is_encoded_image_spec(tensor_spec):
  """Determines whether the passed tensor_spec speficies an encoded image."""
  if hasattr(tensor_spec, 'data_format'):
    # If tensor_spec is an ExtendedTensorSpec, use the data_format to check.
    return (tensor_spec.data_format is not None) and (
        tensor_spec.data_format.upper() in ['JPEG', 'PNG'])
  else:
    # Otherwise default to the old "name contains 'image'" logic.
    logging.warn('Using a deprecated tensor specification. '
                 'Use ExtendedTensorSpec.')
    return 'image' in tensor_spec.name


def _get_feature(tensor_spec,
                 decode_images = True):
  """Get FixedLenfeature or FixedLenSequenceFeature for a tensor spec."""
  varlen_default_value = getattr(tensor_spec, 'varlen_default_value', None)
  if getattr(tensor_spec, 'is_sequence', False):
    cls = tf.FixedLenSequenceFeature
  elif varlen_default_value is not None:
    cls = tf.VarLenFeature
  else:
    cls = tf.FixedLenFeature
  if decode_images and is_encoded_image_spec(tensor_spec):
    if varlen_default_value is not None:
      # Contains a variable length list of images.
      return cls(tf.string)
    elif len(tensor_spec.shape) > 3:
      # Contains a fixed length list of images.
      return cls((tensor_spec.shape[0]), tf.string)
    else:
      return cls((), tf.string)
  elif varlen_default_value is not None:
    return cls(tensor_spec.dtype)
  else:
    return cls(tensor_spec.shape, tensor_spec.dtype)


def tensorspec_to_feature_dict(tensor_spec_struct, decode_images = True):
  """Converts collection of tensorspecs to a dict of FixedLenFeatures specs.

  Args:
    tensor_spec_struct: A (possibly nested) collection of TensorSpec.
    decode_images: If True, TensorSpec with data_format 'JPEG' or 'PNG' are
      interpreted as encoded image strings.

  Returns:
    features: A dict mapping feature keys to FixedLenFeature and
      FixedLenSequenceFeature values.

  Raises:
    ValueError: If duplicate keys are found in the TensorSpecs.
  """
  assert_valid_spec_structure(tensor_spec_struct)
  features = {}
  tensor_spec_dict = {}

  # Note it is valid to iterate over all tensors since
  # assert_valid_spec_structure will ensure that non unique tensor_spec names
  # have the identical properties.
  flat_tensor_spec_struct = flatten_spec_structure(tensor_spec_struct)
  for key, tensor_spec in flat_tensor_spec_struct.items():
    if tensor_spec.name is None:
      # Do not attempt to parse TensorSpecs whose name attribute is not set.
      logging.info(
          'TensorSpec name attribute for %s is not set; will not parse this '
          'Tensor from TFExamples.', key)
      continue
    features[tensor_spec.name] = _get_feature(tensor_spec, decode_images)
    tensor_spec_dict[tensor_spec.name] = tensor_spec
  return features, tensor_spec_dict


def pad_or_clip_tensor_to_spec_shape(
    tensor, tensor_spec):
  """Pads or clips a dense tensor to the desired `tensor_spec` shape.

  Target range T for the 2nd index is given by tensor_spec.shape[0].

  A [B, N, ...]-`tensor` is clipped to [B, 0..T-1, ...] for N > T and
  right-padded to [B, T, ...]-shape for N <= T.

  For example, these are the expected outputs when tensor_spec.shape = (3,),
  tensor_spec.dtype=tf.float, and tensor_spec.varlen_default_value = 3.0:

  Padding example:
    tensor: tf.ragged.constant([[1], [1, 2]])
    padded_or_clipped_tensor: tf.ragged.constant([[1, 3, 3], [1, 2, 3]])

  Clipping example:
    tensor: tf.ragged.constant([[1, 2, 3, 4]])
    padded_or_clipped_tensor: tf.ragged.constant([[1, 2, 3]])

  Args:
    tensor: A tensor, typically the result of dataset parsing.
    tensor_spec: The corresponding tensor_spec for the tensor.

  Returns:
    A tensor of the shape defined in the tensor_spec.
  """
  # The cast is necessary so that the default_value dtype of the padded tensor
  # matches that of the expected input/output tensor defined in the tensor_spec.
  default_value = tf.cast(
      tf.constant(tensor_spec.varlen_default_value), dtype=tensor_spec.dtype)
  batch_dim, varlen_dim = tf.unstack(tf.shape(tensor))[:2]

  def _pad_or_clip_fn(tensor):
    """Pads or clips the 0th index of single example in the batch."""
    pad_length = tensor_spec.shape[0] - varlen_dim

    def _pad_fn():
      return tf.pad(
          tensor, [(0, pad_length)] + [(0, 0)] * (len(tensor_spec.shape) - 1),
          constant_values=default_value)

    def _clip_fn():
      assert tensor.shape.as_list()[1:] == tensor_spec.shape[1:]
      return tf.slice(tensor, [0] * len(tensor_spec.shape), tensor_spec.shape)

    return tf.cond(pad_length > 0, _pad_fn, _clip_fn)

  padded_or_clipped_tensor = tf.map_fn(_pad_or_clip_fn, tensor)
  return tf.reshape(padded_or_clipped_tensor,
                    [batch_dim, tensor_spec.shape[0]] +
                    tensor_spec.shape.as_list()[1:])


def write_t2r_assets_to_file(t2r_assets, filename):
  """Writes feature and label specifications to file."""
  with tf.io.gfile.GFile(filename, 'w') as f:
    f.write(text_format.MessageToString(t2r_assets))


def load_t2r_assets_to_file(filename):
  """Reads feature and label specifications from file."""
  try:
    with tf.io.gfile.GFile(filename, 'r') as f:
      t2r_assets = t2r_pb2.T2RAssets()
      text_format.Parse(f.read(), t2r_assets)
      return t2r_assets
  except tf.errors.DeadlineExceededError:
    raise ValueError(
        'The file could not be loaded within the parsing deadline.')


def write_input_spec_to_file(in_feature_spec, in_label_spec, filename):
  """Writes feature and label specifications to file."""
  with tf.gfile.GFile(filename, 'w') as f:
    cPickle.dump({
        'in_feature_spec': in_feature_spec, 'in_label_spec': in_label_spec}, f)  # pytype: disable=wrong-arg-types


def load_input_spec_from_file(filename):
  """Reads feature and label specifications from file."""
  if not tf.io.gfile.exists(filename):
    raise ValueError('The file {} does not exist.'.format(filename))
  with tf.io.gfile.GFile(filename, 'r') as f:
    spec_data = cPickle.load(f)
  feature_spec = spec_data['in_feature_spec']
  label_spec = spec_data['in_label_spec']
  return feature_spec, label_spec


def write_global_step_to_file(global_step, filename):
  """Writes feature and label specifications to file."""
  with tf.gfile.GFile(filename, 'w') as f:
    cPickle.dump({'global_step': global_step}, f)  # pytype: disable=wrong-arg-types


def load_global_step_from_file(filename):
  """Reads feature and label specifications from file."""
  if not tf.io.gfile.exists(filename):
    raise ValueError('The file {} does not exist.'.format(filename))
  with tf.io.gfile.GFile(filename, 'r') as f:
    spec_data = cPickle.load(f)
  return spec_data['global_step']
