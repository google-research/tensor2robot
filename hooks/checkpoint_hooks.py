# coding=utf-8
# Copyright 2023 The Tensor2Robot Authors.
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

"""CheckpointSaverListener hooks for exporting SavedModels."""

import collections
import distutils.dir_util
import functools
import os
from typing import Text, Callable, Optional, List

from absl import logging
import six
import tensorflow.compat.v1 as tf  # tf

copy_fn = distutils.dir_util.copy_tree


class _DirectoryVersionGC(object):
  """Observes a stream of incoming directories, and removes the oldest item."""

  def __init__(self, num_versions):
    self._queue = collections.deque()
    self._num_versions = num_versions

  def observe(self, directory):
    self._queue.append(directory)
    self._remove_if_necessary()

  def observe_multiple(self, directory_list):
    self._queue.extend(directory_list)
    self._remove_if_necessary()

  def _remove_if_necessary(self):
    while len(self._queue) > self._num_versions:
      tf.io.gfile.rmtree(self._queue.popleft())


class CheckpointExportListener(tf.train.CheckpointSaverListener):
  """Listener that exports the model after creating a checkpoint.

  This is used to support Tensorflow serving for remote inference during
  training and evaluation of Tensorflow models.
  """

  def __init__(self,
               export_fn,
               export_dir,
               num_versions = None):
    """Initializes a `CheckpointExportListener`.

    Args:
      export_fn: function which exports the model.
      export_dir: directory to export models
      num_versions: number of exports to keep. If unset, keep all exports.
    """
    self._export_fn = export_fn
    self._export_dir = six.ensure_text(export_dir)
    tf.io.gfile.makedirs(self._export_dir)
    self._gc = None
    if num_versions:
      self._gc = _DirectoryVersionGC(num_versions)
      export_dir_contents = sorted(tf.gfile.ListDirectory(self._export_dir))
      self._gc.observe_multiple([
          os.path.join(self._export_dir, filename)
          for filename in export_dir_contents
      ])

  def after_save(self, session, global_step):
    logging.info('Exporting SavedModel at global_step %d', global_step)
    exported_path = six.ensure_text(
        self._export_fn(self._export_dir, global_step))
    logging.info('Saved model to %s', exported_path)
    if self._gc:
      self._gc.observe(exported_path)
    return exported_path


class LaggedCheckpointListener(CheckpointExportListener):
  """Listener that exports the model after creating a checkpoint.

  This is used to support Tensorflow serving for remote inference during
  training and evaluation of Tensorflow models. This listener also exports the
  **second oldest** model to a separate export directory to support TD3
  training.
  """

  def __init__(self, export_fn, export_dir,
               lagged_export_dir, num_versions):
    """Initializes a LaggedCheckpointListener.

    Args:
      export_fn: function which exports the model.
      export_dir: directory to the latest models
      lagged_export_dir: directory to export the second-oldest model
      num_versions: number of saved versions to keep
    """
    CheckpointExportListener.__init__(self, export_fn, export_dir, num_versions)

    self._lagged_export_dir = lagged_export_dir
    self._second_oldest_model_dir = None
    self._current_model_dir = None
    self._lagged_model_dir = None
    if self._gc:
      self._lagged_gc = _DirectoryVersionGC(num_versions)
    tf.io.gfile.makedirs(self._lagged_export_dir)
    export_dir_contents = sorted(tf.gfile.ListDirectory(self._export_dir))
    lagged_export_dir_contents = sorted(
        tf.gfile.ListDirectory(self._lagged_export_dir))

    if self._gc:
      self._lagged_gc.observe_multiple([
          os.path.join(self._lagged_export_dir, filename)
          for filename in lagged_export_dir_contents
      ])
    # This copies models if the lagged export dir is out of sync.
    if len(export_dir_contents) == 1:
      self._current_model_dir = os.path.join(self._export_dir,
                                             export_dir_contents[0])
      if export_dir_contents == lagged_export_dir_contents:
        self._lagged_model_dir = os.path.join(self._lagged_export_dir,
                                              lagged_export_dir_contents[0])
      else:
        self._lagged_model_dir = self._copy_savedmodel(self._current_model_dir,
                                                       lagged_export_dir)
    elif len(export_dir_contents) > 1:
      second_last_exported_model = export_dir_contents[-2]
      self._current_model_dir = os.path.join(self._export_dir,
                                             export_dir_contents[-1])
      lagged_dir_empty = not lagged_export_dir_contents
      if (lagged_dir_empty or
          second_last_exported_model != lagged_export_dir_contents[-1]):
        self._lagged_model_dir = self._copy_savedmodel(
            os.path.join(self._export_dir, second_last_exported_model),
            lagged_export_dir)
      else:
        self._lagged_model_dir = os.path.join(self._lagged_export_dir,
                                              lagged_export_dir_contents[-1])

  def _copy_savedmodel(self, source_dir, destination):
    """Copy source_dir to destination.

    This recursively copies all of the files in `source_dir` to destination.
    `source_dir` is assumed to have the SavedModel format.

    Args:
      source_dir: Source directory, should be a path to a SavedModel directory.
      destination: Base directory to copy these.

    Returns:
      Destination path of the copied model.
    """
    source_dir = six.ensure_text(source_dir)
    destination = six.ensure_text(destination)
    basename = os.path.basename(source_dir)
    dest_base_dir = os.path.join(destination, basename)
    copy_fn(source_dir, dest_base_dir)
    return dest_base_dir

  def _copy_lagged_model(self, source_dir, destination):
    destination_path = self._copy_savedmodel(source_dir, destination)
    if self._lagged_gc:
      self._lagged_gc.observe(destination_path)
    return destination_path

  def after_save(self, session, global_step):
    """Exports SavedModel to lagged and current directory after checkpoint save.

    This ensures that the lagged_model_dir is one 'version' behind
    the current_model_dir.

    Args:
      session: tensorflow session
      global_step: current global step
    """

    export_dir = CheckpointExportListener.after_save(self, session, global_step)
    if not self._current_model_dir:
      self._lagged_model_dir = self._copy_lagged_model(export_dir,
                                                       self._lagged_export_dir)
    elif os.path.basename(self._current_model_dir) == os.path.basename(
        self._lagged_model_dir):
      # If the lagged model is up to date with the current model directory.
      pass
    else:
      self._lagged_model_dir = self._copy_lagged_model(self._current_model_dir,
                                                       self._lagged_export_dir)

    self._current_model_dir = export_dir
