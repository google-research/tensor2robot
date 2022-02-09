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

"""Tests for tensor2robot.layers.tec."""

from tensor2robot.layers import tec
import tensorflow.compat.v1 as tf


class TECTest(tf.test.TestCase):

  def test_embed_condition_images(self):
    images = tf.random.normal((4, 100, 100, 3))
    embedding = tec.embed_condition_images(
        images, 'test_embed', fc_layers=(100, 20))
    self.assertEqual(embedding.shape.as_list(), [4, 20])

  def test_doubly_batched_embed_condition_images(self):
    doubly_batched_images = tf.random.normal((3, 4, 10, 12, 3))
    with self.assertRaises(ValueError):
      tec.embed_condition_images(doubly_batched_images, 'test_embed')

  def test_reduce_temporal_embeddings(self):
    temporal_embeddings = tf.random.normal((4, 20, 16))
    embedding = tec.reduce_temporal_embeddings(
        temporal_embeddings, 10, 'test_reduce')
    self.assertEqual(embedding.shape.as_list(), [4, 10])

  def test_doubly_batched_reduce_temporal_embeddings(self):
    temporal_embeddings = tf.random.normal((2, 4, 20, 16))
    with self.assertRaises(ValueError):
      tec.reduce_temporal_embeddings(temporal_embeddings, 10, 'test_reduce')

  def test_contrastive_loss(self):
    inf_embeddings = tf.nn.l2_normalize(
        tf.ones((5, 1, 10), dtype=tf.float32), axis=-1)
    con_embeddings = tf.nn.l2_normalize(
        tf.ones((5, 1, 10), dtype=tf.float32), axis=-1)
    tec.compute_embedding_contrastive_loss(inf_embeddings, con_embeddings)


if __name__ == '__main__':
  tf.test.main()
