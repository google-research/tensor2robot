# coding=utf-8
# Copyright 2024 The Tensor2Robot Authors.
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

"""Collect/Eval a policy on the live environment."""

import os
import time
from typing import Text

import gin
import gym
import tensorflow.compat.v1 as tf


@gin.configurable
def collect_eval_loop(
    collect_env,
    eval_env,
    policy_class,
    num_collect = 2000,
    num_eval = 100,
    run_agent_fn=None,
    root_dir = '',
    continuous = False,
    min_collect_eval_step = 0,
    max_steps = 1,
    pre_collect_eval_fn=None,
    record_eval_env_video = False,
    init_with_random_variables = False):
  """Like dql_grasping.collect_eval, but can run continuously.

  Args:
    collect_env: (gym.Env) Gym environment to collect data from (and train the
      policy on).
    eval_env: (gym.Env) Gym environment to evaluate the policy on. Can be
      another instance of collect_env, or a different environment if one
      wishes to evaluate generalization capability. The only constraint is
      that the action and observation spaces have to be equivalent. If None,
      eval_env is not evaluated.
    policy_class: Policy class that we want to train.
    num_collect: (int) Number of episodes to collect from collect_env.
    num_eval: (int) Number of episodes to evaluate from eval_env.
    run_agent_fn: (Optional) Python function that executes the interaction of
      the policy with the environment. Defaults to run_env.run_env.
    root_dir: Base directory where collect data and eval data are
      stored.
    continuous: If True, loop and wait for new ckpt to load a policy from
      (up until the ckpt number exceeds max_steps).
    min_collect_eval_step: An integer which specifies the lowest ckpt step
      number that we will collect/evaluate.
    max_steps: (Ignored unless continuous=True). An integer controlling when
      to stop looping: once we see a policy with global_step > max_steps, we
      stop.
    pre_collect_eval_fn: This callable will be run prior to the start of this
      collect/eval loop. Example use: pushing a record dataset into a replay
      buffer at the start of training.
    record_eval_env_video: Whether to enable video recording in our eval env.
    init_with_random_variables: If True, initializes policy model with random
      variables instead (useful for unit testing).
  """
  if pre_collect_eval_fn:
    pre_collect_eval_fn()

  collect_dir = os.path.join(root_dir, 'policy_collect')
  eval_dir = os.path.join(root_dir, 'eval')

  policy = policy_class()
  prev_global_step = -1
  while True:
    global_step = None
    if hasattr(policy, 'restore'):
      if init_with_random_variables:
        policy.init_randomly()
      else:
        policy.restore()
    global_step = policy.global_step

    if global_step is None or global_step < min_collect_eval_step \
        or global_step <= prev_global_step:
      time.sleep(10)
      continue

    if collect_env:
      run_agent_fn(collect_env, policy=policy, num_episodes=num_collect,
                   root_dir=collect_dir, global_step=global_step, tag='collect')
    if eval_env:
      if record_eval_env_video and hasattr(eval_env, 'set_video_output_dir'):
        eval_env.set_video_output_dir(
            os.path.join(root_dir, 'videos', str(global_step)))
      run_agent_fn(eval_env, policy=policy, num_episodes=num_eval,
                   root_dir=eval_dir, global_step=global_step, tag='eval')
    if not continuous or global_step >= max_steps:
      tf.logging.info('Completed collect/eval on final ckpt.')
      break

    prev_global_step = global_step
