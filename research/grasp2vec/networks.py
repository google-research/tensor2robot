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

"""Implements forward pass for embeddings.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2robot.research.grasp2vec import resnet
import tensorflow as tf

slim = tf.contrib.slim


def Embedding(image, mode, params, reuse=tf.AUTO_REUSE, scope='scene'):
  """Implements scene or goal embedding.

  Args:
    image: Batch of images corresponding to scene or goal.
    mode: Mode is tf.estimator.ModeKeys.EVAL, TRAIN, or PREDICT (unused).
    params: Hyperparameters for the network.
    reuse: Reuse parameter for variable scope.
    scope: The variable_scope to use for the variables.
  Returns:
    A tuple (batch of summed embeddings, batch of embedding maps).
  """
  del params
  is_training = mode == tf.estimator.ModeKeys.TRAIN
  with tf.variable_scope(scope, reuse=reuse):
    scene = resnet.get_resnet50_spatial(image, is_training)
    scene = tf.nn.relu(scene)
    summed_scene = tf.reduce_mean(scene, axis=[1, 2])
  return summed_scene, scene
