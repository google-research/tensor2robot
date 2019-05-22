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

"""Grasp2Vec T2R model."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
from tensor2robot.models import abstract_model
from tensor2robot.preprocessors import spec_transformation_preprocessor
from tensor2robot.research.grasp2vec import losses
from tensor2robot.research.grasp2vec import networks
from tensor2robot.research.grasp2vec import visualization
from tensor2robot.utils import tensorspec_utils

import tensorflow as tf

from typing import Optional, Tuple

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT

RunConfigType = abstract_model.RunConfigType
ParamsType = abstract_model.ParamsType
DictOrSpec = abstract_model.DictOrSpec

ModelTrainOutputType = abstract_model.ModelTrainOutputType
ExportOutputType = abstract_model.ExportOutputType

TensorSpec = tensorspec_utils.ExtendedTensorSpec


def maybe_crop_images(images,
                      params,
                      mode):
  """Helper function to crop a list of image tensors randomly.

  Args:
    images: List of tensors that the same random crop will be applied to.
    params: 6-Tuple of parameters (min_offset_height, max_offset_height,
      target_height, min_offset_width, max_offset_width, target_width) for
      sampling random cropping parameters.
    mode: Whether this is TRAIN, EVAL, or PREDICT.
  Returns:
    List of cropped image tensors.
  """
  (min_offset_height, max_offset_height, target_height,
   min_offset_width, max_offset_width, target_width) = params
  if mode == TRAIN:
    offset_height = tf.random_uniform(
        (), minval=min_offset_height, maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        (), minval=min_offset_width, maxval=max_offset_width, dtype=tf.int32)
  else:
    offset_height = (min_offset_height + max_offset_height)//2
    offset_width = (min_offset_width + max_offset_width)//2
  return [
      tf.image.crop_to_bounding_box(
          img, offset_height, offset_width, target_height, target_width)
      for img in images]


@gin.configurable
class Grasp2VecPreprocessor(
    spec_transformation_preprocessor.SpecTransformationPreprocessor):
  """Image conversions."""

  def __init__(
      self,
      scene_crop=(0, 40, 472, 0, 168, 472),
      goal_crop=(0, 40, 472, 0, 168, 472),
      **kwargs):
    """Initialize the preprocessor.

    Args:
      scene_crop:  A tuple (min_offset_height, max_offset_height, target_height,
        min_offset_width, max_offset_width, target_width) to specify the crop
        details.
      goal_crop:  A tuple (min_offset_height, max_offset_height, target_height,
        min_offset_width, max_offset_width, target_width) to specify the crop
        details.
      **kwargs: Args to be passed to parent preprocessor.
    """
    self._scene_crop = scene_crop
    self._goal_crop = goal_crop
    super(Grasp2VecPreprocessor, self).__init__(**kwargs)

  def _transform_in_feature_specification(
      self, flat_spec_structure
  ):
    """Declare actual shape of serialized data."""
    for name in ['pregrasp_image', 'postgrasp_image', 'goal_image']:
      self.update_spec(
          flat_spec_structure,
          name,
          shape=(512, 640, 3),
          dtype=tf.uint8,
          data_format='jpeg')
    return flat_spec_structure

  def _preprocess_fn(
      self, features,
      labels,
      mode
  ):
    """Crop, distort, random flip, and resize images."""
    scene_images = maybe_crop_images(
        [features['pregrasp_image'], features['postgrasp_image']],
        self._scene_crop, mode)
    features['pregrasp_image'] = scene_images[0]
    features['postgrasp_image'] = scene_images[1]
    features['goal_image'] = maybe_crop_images(
        [features['goal_image']], self._goal_crop, mode)[0]
    for name, image in features.items():
      image = tf.image.convert_image_dtype(image, tf.float32)
      if mode == TRAIN:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
      features[name] = image
    return features, labels


@gin.configurable
class Grasp2VecModel(abstract_model.AbstractT2RModel):
  """Basic Grasp2Vec Model implementation."""

  def __init__(self,
               scene_size,
               goal_size,
               **kwargs):
    """Initialize the model.

    Args:
      scene_size: 2-Tuple of ints specifying height, width of image input.
      goal_size: 2-Tuple of ints specifying height, widt of goal image input.
      **kwargs: Passed to parent (AbstractT2RModel) constructor.
    """
    self._scene_size = scene_size
    self._goal_size = goal_size
    super(Grasp2VecModel, self).__init__(**kwargs)

  def get_feature_specification(
      self, mode):
    tspec = tensorspec_utils.TensorSpecStruct()
    tspec.pregrasp_image = TensorSpec(
        shape=self._scene_size + (3,), dtype=tf.float32, name='image',
        data_format='jpeg')
    tspec.postgrasp_image = TensorSpec(
        shape=self._scene_size + (3,), dtype=tf.float32, name='postgrasp_image',
        data_format='jpeg')
    tspec.goal_image = TensorSpec(
        shape=self._goal_size + (3,), dtype=tf.float32, name='present_image',
        data_format='jpeg')
    return tspec

  def get_label_specification(self, mode
                             ):
    # Grasp2Vec is unsupervised; requires no labels.
    return tensorspec_utils.TensorSpecStruct()

  @property
  def default_preprocessor_cls(self):
    return Grasp2VecPreprocessor

  def inference_network_fn(self,
                           features,
                           labels,
                           mode,
                           config = None,
                           params = None):
    """Forward model."""
    # Merge scene images into a single batch to take advantage of vectorization.
    scene_images = tf.concat(
        [features.pregrasp_image, features.postgrasp_image], axis=0)
    v, s = networks.Embedding(scene_images, mode, params, scope='scene')
    pre_v, post_v = tf.split(v, 2, axis=0)
    pre_s, post_s = tf.split(s, 2, axis=0)
    goal_v, goal_s = networks.Embedding(
        features.goal_image, mode, params, scope='goal')
    outputs = {
        'pre_vector': pre_v,
        'post_vector': post_v,
        'pre_spatial': pre_s,
        'post_spatial': post_s,
        'goal_vector': goal_v,
        'goal_spatial': goal_s
    }
    return outputs

  def model_train_fn(self,
                     features,
                     labels,
                     inference_outputs,
                     mode,
                     config = None,
                     params = None):
    npairs_loss = losses.NPairsLoss(
        inference_outputs['pre_vector'],
        inference_outputs['goal_vector'],
        inference_outputs['post_vector'], params)
    train_outputs = {'npairs_loss': npairs_loss}
    return npairs_loss, train_outputs

  def add_summaries(self,
                    features,
                    labels,
                    inference_outputs,
                    train_loss,
                    train_outputs,
                    mode,
                    config = None,
                    params = None):
    del labels, train_loss, train_outputs, mode
    del config
    if not self.use_summaries(params):
      return
    for key in ['pregrasp', 'postgrasp', 'goal']:
      feature_name = key + '_image'
      if feature_name in features.keys():
        tf.summary.image('image/%s' % key, features[feature_name])
    heatmaps = visualization.add_heatmap_summary(
        inference_outputs['goal_vector'], inference_outputs['pre_spatial'],
        'goal_pregrasp_map')
    visualization.add_spatial_softmax(
        heatmaps, features.pregrasp_image)
