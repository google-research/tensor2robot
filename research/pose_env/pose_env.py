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
"""Duck pose prediction task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gin
import gym
import numpy as np
import pybullet


def get_pybullet_urdf_root():
  """Returns base path to URDFs. Differs between open-source and internal."""
  open_source = True
  if open_source:
    import pybullet_data  # pylint: disable=g-import-not-at-top
    urdf_root = pybullet_data.getDataPath()
  return urdf_root


@gin.configurable
class PoseEnvRandomPolicy(object):
  """A random policy for the PoseEnv, used for dataset generation."""

  def reset(self):
    pass

  @property
  def global_step(self):
    return 0

  def sample_action(self, obs, explore_prob):
    del obs, explore_prob
    return np.random.uniform(low=-1., high=1., size=2), None


@gin.configurable
class PoseToyEnv(gym.Env):
  """Predict object pose given current image.

  Observation spec:
    image: Randomly chosen camera angle and object pose. Camera angle is kept
      the same between episodes in a single trial.

  Action spec:
    pose: Predicted pose of the object in `image`.

  Reward is -|| target_pose - pose ||_2
  """

  def __init__(self,
               render_mode='DIRECT',
               hidden_drift=False,
               urdf_root=''):
    """Construct a Duck pose prediction task.

    Args:
      render_mode: Whether to render headless or with a GUI.
      hidden_drift: If True, each task will assign a hidden random drift where
        the rendered pose differs from the true pose. Requires meta-learning
        adaptation to solve.
      urdf_root: Path to URDF files.
    """
    if render_mode == 'GUI':
      self.cid = pybullet.connect(pybullet.GUI)
    elif render_mode == 'DIRECT':
      self.cid = pybullet.connect(pybullet.DIRECT)
    self._width, self._height = 64, 64
    self._urdf_root = urdf_root
    self._hidden_drift = hidden_drift
    self._hidden_drift_xyz = None
    self._setup()
    self.reset_task()

  def _setup(self):
    """Sets up the robot + tray + objects.
    """
    pybullet.resetSimulation(physicsClientId=self.cid)
    pybullet.setPhysicsEngineParameter(numSolverIterations=150,
                                       physicsClientId=self.cid)
    # pybullet.setTimeStep(self._time_step, physicsClientId=self.cid)
    pybullet.setGravity(0, 0, -10, physicsClientId=self.cid)
    plane_path = os.path.join(self._urdf_root, 'plane.urdf')
    pybullet.loadURDF(plane_path, [0, 0, -1],
                      physicsClientId=self.cid)
    table_path = os.path.join(self._urdf_root, 'table/table.urdf')
    pybullet.loadURDF(table_path, [0.0, 0.0, -.65],
                      [0., 0., 0., 1.], physicsClientId=self.cid)
    # Load the target object
    duck_path = os.path.join(self._urdf_root, 'duck_vhacd.urdf')
    pos = [0]*3
    orn = [0., 0., 0., 1.]
    self._target_uid = pybullet.loadURDF(
        duck_path, pos, orn, physicsClientId=self.cid)

  def reset_task(self):
    self._reset_camera()
    if self._hidden_drift:
      self._hidden_drift_xyz = np.random.uniform(low=-.3, high=.3, size=3)
      self._hidden_drift_xyz[2] = 0
    self.set_new_pose()

  def set_new_pose(self):
    self._target_pose = self._sample_pose()
    self._move_duck(self._target_pose)
    if self._hidden_drift:
      self._target_pose += self._hidden_drift_xyz

  def reset(self):
    # Move the duck somewhere, take an image.
    # Assumes reset_task has been called.
    # self.set_new_pose()
    return self.get_observation()

  def _move_duck(self, pose):
    x, y, angle = pose
    orn = pybullet.getQuaternionFromEuler([0, 0, angle])
    pybullet.resetBasePositionAndOrientation(
        self._target_uid,
        [x, y, 0.],
        [orn[0], orn[1], orn[2], orn[3]],
        physicsClientId=self.cid)

  def _sample_pose(self):
    x = np.random.uniform(low=-.7, high=.7)
    y = np.random.uniform(low=-.4, high=.4)
    angle = np.random.uniform(low=-180, high=180)
    return np.array([x, y, angle])

  def _reset_camera(self):
    look = [0., 0., 0.]
    distance = 3.
    pitch = -30 + np.random.uniform(-10, 10)
    yaw = np.random.uniform(-180, 180)
    roll = 0
    self._view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
        look, distance, yaw, pitch, roll, 2)
    fov = 30
    aspect = self._width / self._height
    near = 0.1
    far = 10
    self._proj_matrix = pybullet.computeProjectionMatrixFOV(
        fov, aspect, near, far)

  def _get_image(self):
    img_arr = pybullet.getCameraImage(width=self._width,
                                      height=self._height,
                                      viewMatrix=self._view_matrix,
                                      projectionMatrix=self._proj_matrix,
                                      physicsClientId=self.cid)
    rgb = img_arr[2]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
    return np_img_arr[:, :, :3]

  def get_observation(self):
    return self._get_image()

  def step(self, action):
    reward = -np.linalg.norm(action - self._target_pose[:2]).astype(np.float32)
    done = True
    debug = {'target_pose': self._target_pose[:2].astype(np.float32)}
    observation = self.get_observation()
    return observation, reward, done, debug
