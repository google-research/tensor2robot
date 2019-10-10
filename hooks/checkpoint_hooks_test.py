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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from absl import logging
from six.moves import range
from tensor2robot.hooks import checkpoint_hooks
import tensorflow as tf  # tf

FLAGS = flags.FLAGS


def _CheckpointDir(export_dir, checkpoint_id):
  return os.path.join(export_dir, str(checkpoint_id))


def _MakeSavedModel(export_dir, checkpoint_id):
  """Make a fake SavedModel directory.

  Builds the following files to make a 'synthetic SavedModel checkpoint'
    export_dir/checkpoint_id/savedModel.txt
    export_dir/checkpoint_id/variables/variables.txt
  Args:
    export_dir: base directory to export to
    checkpoint_id: integer id of the checkpoint to write.

  Returns:
    string of saved model (export_dir/checkpoint_id)
  """

  def _TouchFile(filename):
    with tf.gfile.GFile(filename, 'w') as f:
      f.write('abc')

  checkpoint_dir = _CheckpointDir(export_dir, checkpoint_id)
  tf.gfile.MkDir(checkpoint_dir)
  tf.gfile.MkDir(os.path.join(checkpoint_dir, 'variables'))
  _TouchFile(os.path.join(checkpoint_dir, 'savedModel.txt'))
  _TouchFile(os.path.join(checkpoint_dir, 'variables', 'variables.txt'))
  logging.info(checkpoint_dir)
  return checkpoint_dir


class CheckpointExportListener(tf.test.TestCase):

  def setUp(self):
    self._test_dir = FLAGS.test_tmpdir
    self._export_dir = os.path.join(self._test_dir, 'export')
    tf.gfile.MakeDirs(self._export_dir)
    self._export_id = 0
    super(CheckpointExportListener, self).setUp()

  def tearDown(self):
    tf.gfile.DeleteRecursively(self._test_dir)
    super(CheckpointExportListener, self).tearDown()

  def _ExportFn(self, export_dir, global_step):
    del global_step
    self._export_id += 1
    return _MakeSavedModel(export_dir, self._export_id)

  def testCheckpointExportListener(self):
    listener = checkpoint_hooks.CheckpointExportListener(
        self._ExportFn, self._export_dir)
    listener.after_save(None, 10)
    self.assertTrue(tf.gfile.Exists(_CheckpointDir(self._export_dir, 1)))

  def testCheckpointExportListenerGC(self):
    listener = checkpoint_hooks.CheckpointExportListener(
        self._ExportFn, self._export_dir, num_versions=3)
    for step in range(5):
      listener.after_save(None, step)
    self.assertTrue(tf.gfile.Exists(_CheckpointDir(self._export_dir, 5)))
    self.assertFalse(tf.gfile.Exists(_CheckpointDir(self._export_dir, 2)))

  def testCheckpointExportListenerGCRestore(self):
    for step in range(6):
      _MakeSavedModel(self._export_dir, step)
    # Initializer does GC on old checkpoints.
    checkpoint_hooks.CheckpointExportListener(
        self._ExportFn, self._export_dir, num_versions=3)
    self.assertTrue(tf.gfile.Exists(_CheckpointDir(self._export_dir, 5)))
    self.assertFalse(tf.gfile.Exists(_CheckpointDir(self._export_dir, 2)))


class LaggedCheckpointListenerTest(tf.test.TestCase):

  def setUp(self):
    self._test_dir = FLAGS.test_tmpdir

    self._export_dir = os.path.join(self._test_dir, 'export')
    self._lagged_export_dir = os.path.join(self._test_dir, 'lagged_export')
    tf.gfile.MakeDirs(self._export_dir)
    tf.gfile.MakeDirs(self._lagged_export_dir)
    self._export_id = 0
    super(LaggedCheckpointListenerTest, self).setUp()

  def tearDown(self):
    tf.gfile.DeleteRecursively(self._test_dir)
    super(LaggedCheckpointListenerTest, self).tearDown()

  def _ExportFn(self, export_dir, global_step):
    del global_step
    self._export_id += 1
    return _MakeSavedModel(export_dir, self._export_id)

  def DefaultLaggedCheckpointListener(self):
    return checkpoint_hooks.LaggedCheckpointListener(
        export_fn=self._ExportFn,
        export_dir=self._export_dir,
        lagged_export_dir=self._lagged_export_dir,
        num_versions=3)

  def testEmptyDir(self):
    listener = self.DefaultLaggedCheckpointListener()
    listener.after_save(None, 10)
    self.assertTrue(
        tf.gfile.Exists(_CheckpointDir(self._lagged_export_dir, 1)))
    self.assertTrue(tf.gfile.Exists(_CheckpointDir(self._export_dir, 1)))

    listener.after_save(None, 11)
    self.assertTrue(tf.gfile.Exists(_CheckpointDir(self._export_dir, 2)))
    # Lagged policy has not updated.
    self.assertFalse(
        tf.gfile.Exists(_CheckpointDir(self._lagged_export_dir, 2)))

    listener.after_save(None, 12)
    # Lagged policy updates the lagged_dir.
    self.assertTrue(tf.gfile.Exists(_CheckpointDir(self._export_dir, 3)))
    self.assertTrue(
        tf.gfile.Exists(_CheckpointDir(self._lagged_export_dir, 2)))

  def testInitOneSavedModel(self):
    _MakeSavedModel(self._export_dir, 1)
    self.DefaultLaggedCheckpointListener()
    # Constructor copies over SavedModel.
    self.assertTrue(tf.gfile.Exists(_CheckpointDir(self._export_dir, 1)))

  def testInitSavedModelUptoDate(self):
    _MakeSavedModel(self._export_dir, 1)
    _MakeSavedModel(self._lagged_export_dir, 1)
    listener = self.DefaultLaggedCheckpointListener()
    self._export_id = 1  # This only matters for mocking new exports.
    listener.after_save(None, 11)

    self.assertTrue(tf.gfile.Exists(_CheckpointDir(self._export_dir, 2)))
    # Lagged policy has not updated.
    self.assertFalse(
        tf.gfile.Exists(_CheckpointDir(self._lagged_export_dir, 2)))

  def testInitSavedModelFromLaggedPosition(self):
    _MakeSavedModel(self._export_dir, 1)
    _MakeSavedModel(self._export_dir, 2)

    _MakeSavedModel(self._lagged_export_dir, 1)
    self._export_id = 2  # This only matters for mocking new exports.
    listener = self.DefaultLaggedCheckpointListener()
    listener.after_save(None, 11)

    self.assertTrue(tf.gfile.Exists(_CheckpointDir(self._export_dir, 3)))
    self.assertTrue(
        tf.gfile.Exists(_CheckpointDir(self._lagged_export_dir, 2)))


if __name__ == '__main__':
  tf.test.main()
