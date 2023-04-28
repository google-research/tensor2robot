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

"""Contains model definitions of grasping models."""

from absl import logging
import gin

from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf
import tf_slim as slim

from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import seq2seq as contrib_seq2seq

# Global constant. The number of layers that are part of the PNN.
NUM_LAYERS = 19
BATCH_SIZE = 64
# The number of actions to sample when estimating max_a Q(s, a) for DQN models.
# A lower NUM_SAMPLES is faster to evaluate, but is less accurate. In
# small-scale tests we found 100 was a reasonable trade-off between speed and
# accuracy.
NUM_SAMPLES = 100


class GraspingModel(object):
  """Base class for a grasping model.
  """

  def __init__(self,
               batch_norm_decay=0.9997,
               batch_norm_epsilon=0.001,
               l2_regularization=0.00007):
    """Default parameters are training defaults for 10-replica GPU training."""
    self._batch_norm_decay = batch_norm_decay
    self._batch_norm_epsilon = batch_norm_epsilon
    self._l2_regularization = l2_regularization

  @property
  def grasp_model_input_keys(self):
    """Returns the keys of model_input to use in create_grasp_params_input.

    Subclassed models using different features should override with the features
    to use.
    """
    return ['world_vector', 'vertical_rotation']

  def create_grasp_params_input(self, model_input, concat_axis=1):
    """Creates grasp params input from translation and rotation parameters.

    Args:
      model_input: A `dict` containing input parameters into the model.
      This function uses the elements of self.model_inputs to pack the values.
      concat_axis: The axis to concatenate observations along.
      Useful for dealing with action-minibatches.
    Returns:
      A tensor encapsulated the grasp params input.
    """
    return tf.concat([
        model_input[grasp_input] for grasp_input in self.grasp_model_input_keys
    ], concat_axis)

  @classmethod
  def create_default_input_specifications(cls):
    """Creates the default input_specifications for preprocessing.

    The default setting uses only main(RGB) images.

    Returns:
      input_specifications: A dictionary to specify the input modalities.
    """
    grasp_param_sizes = {
        'projected_vector': 2,
        'tip_vectors_first_finger': 2,
        'tip_vectors_second_finger': 2,
        'vertical_rotation': 2,
        'camera_vector': 3,
        'world_vector': 3,
        'wrist_vector': 3,
    }
    return {
        'include_initial_image': True,
        'include_main_images': True,
        'include_wrist_images': False,
        'include_depth_images': False,
        'include_segmask_images': False,
        'include_present_images': False,
        'include_goal_images': False,
        'include_target_object_id': True,
        'include_next_image': False,
        'include_placing_parameters': False,
        'use_displacement_pantilt': False,
        'end_to_end_grasping': False,
        'end_to_end_grasping_gripper_status': False,
        'end_to_end_grasping_height_to_bottom': False,
        'end_to_end_grasping_workspace_deltas': False,
        'end_to_end_grasping_async': False,
        'grasp_param_sizes': grasp_param_sizes,
        'include_hand_eye_calibration': False,
        'nav_to_grasp': False,
    }

  @classmethod
  def create_input_specifications(cls):
    """Creates the input_specifications for preprocessing for each model.

    Returns:
      input_specifications: A dictionary to specify the input modalities.
    """
    input_specifications = cls.create_default_input_specifications()
    return input_specifications

  def add_losses(self,
                 config,
                 logits,
                 end_points,
                 label,
                 loss_type,
                 use_tpu=False):
    """Add the losses to train the model.

    Args:
      config: The slim config deployment used.
      logits: The logits that the model generates.
      end_points: The end points that the model generates.
      label: The labels of the current batch.
      loss_type: The type of loss to use.
      use_tpu: Whether to run on TPU.
    """
    logits = tf.check_numerics(logits, 'Logits is not a number.')
    label = tf.check_numerics(label, 'Label is not a number.')
    if loss_type == 'cross_entropy':
      slim.losses.softmax_cross_entropy(logits, label)
    elif loss_type == 'log':
      slim.losses.log_loss(end_points['predictions'], label)
    elif loss_type == 'huber':
      tf.losses.huber_loss(label, end_points['predictions'])
    else:
      slim.losses.sum_of_squares(end_points['predictions'], label)

    logging.debug('end points predictions %s', str(end_points['predictions']))
    logging.debug('label %s', str(label))
    if not use_tpu:
      with tf.device(config.inputs_device()):
        slim.summaries.add_histogram_summaries(
            list(end_points.values()), 'Predictions')
        slim.summaries.add_zero_fraction_summaries(list(end_points.values()))
        slim.summaries.add_histogram_summary(label, 'Labels')
        slim.summaries.add_histogram_summaries(
            slim.variables.get_model_variables())

  def create_image_input(self,
                         init_image,
                         grasp_image,
                         wrist_image,
                         init_depth=None,
                         grasp_depth=None,
                         init_segmask=None,
                         grasp_segmask=None,
                         goal_image=None,
                         input_specifications=None):
    """Converts individual input images to the array of image input.

    Args:
      init_image: Init image from the main camera.
      grasp_image: Grasp image from the main camera.
      wrist_image: Wrist image.
      init_depth: Init image from the main depth camera.
      grasp_depth: Grasp image from the main depth camera.
      init_segmask: Init segmentation mask image from the main camera.
      grasp_segmask: Grasp segmentation mask image from the main camera.
      goal_image: Wrist image encoding grasp goal.
      input_specifications: A dictionary to specify the input modalities. Set it
        None to use the default input specifications predefined for each model.

    Returns:
      images: An array of images which this model can deal with. This array
      can safely be passed to the model function.
    Raises:
          ValueError: If the required image inputs are not provided.
    """
    if input_specifications is None:
      input_specifications = self.create_input_specifications()

    image_input = []

    if input_specifications['include_initial_image']:
      if init_image is None:
        raise ValueError('init_image is needed for image_input.')
      image_input.append(init_image)
    else:
      image_input.append(None)
      del init_image

    if input_specifications['include_main_images']:
      if grasp_image is None:
        raise ValueError('grasp_image is needed for image_input.')
      image_input.append(grasp_image)
    else:
      del grasp_image

    if input_specifications['include_wrist_images']:
      if wrist_image is None:
        raise ValueError('wrist_image is needed for image_input.')
      image_input += [wrist_image]
    else:
      del wrist_image

    if input_specifications['include_depth_images']:
      if init_depth is None:
        raise ValueError('init_depth is needed for image_input.')
      if grasp_depth is None:
        raise ValueError('grasp_depth is needed for image_input.')
      image_input += [init_depth, grasp_depth]
    else:
      del init_depth
      del grasp_depth

    if input_specifications['include_segmask_images']:
      if init_segmask is None:
        raise ValueError('init_segmask is needed for image_input.')
      if grasp_segmask is None:
        raise ValueError('grasp_segmask is needed for image_input.')
      image_input += [init_segmask, grasp_segmask]
    else:
      del init_segmask
      del grasp_segmask

    if input_specifications['include_goal_images']:
      if goal_image is None:
        raise ValueError('goal_image is needed for image_input.')
      image_input += [goal_image]

    return image_input

  def reduce_output(self, model_outputs):
    """Reduces the model outputs to a batch of scalars.

    In the default case the model_outputs are simply returned without any
    change.
    Args:
      model_outputs: A `np.array` of shape [batch_size, N_MODEL_OUTPUTS].

    Returns:
      Returns `model_outputs`.
    """
    return model_outputs

  def model_from_batch(self, batch, **model_kwargs):
    """Retrieves data from the batch and builds the Tensorflow model.

    For legacy reasons, we don't want to update the function signature for
    model(), because it's called all over the code-base. Most subclasses should
    overwrite model() instead of this function.

    Args:
      batch: A batch of data, either from grasping_data or the replay buffer.
      **model_kwargs: Keyword arguments. Forwarded to the model() function.

    Returns:
      (logits, end_points) where logits is a Tensor for the linear output of the
      model, and end_points is a dictionary of Tensors at various points of the
      network. The return value may be (None, None) if the model has no output.
    """
    image = batch.get('distorted_images')
    image_1 = batch.get('distorted_images1')
    image_wrist = batch.get('distorted_wrist_images')
    depth = batch.get('distorted_depths')
    depth_1 = batch.get('distorted_depths1')
    segmask = batch.get('distorted_segmasks')
    segmask_1 = batch.get('distorted_segmasks1')
    goal_image = batch.get('decoded_goal_images')
    # Some of image, image_1, image_wrist might be None, but the
    # model.create_image_input is responsible for checking the input
    # sanity, and picking those modalities, which is is capable of handling.
    input_image_tensors = self.create_image_input(image, image_1, image_wrist,
                                                  depth, depth_1, segmask,
                                                  segmask_1, goal_image)
    grasp_params = self.create_grasp_params_input(batch)
    logits, end_points = self.model(input_image_tensors, grasp_params,
                                    **model_kwargs)
    return logits, end_points


@gin.configurable
class Grasping44FlexibleGraspParams(GraspingModel):
  """The Grasping44 model which ignores the pre-grasp image.

  This model is equivalent to the original Grasping44 model, except it
  generalizes how grasp_params are handled. Here we construct the network input
  directly from a tensor grasp_params with shape (batch_size, grasp_params_size)
  without splitting it into specific subblocks. We leave the option of passing
  names for subblocks of grasp_params to be used for graph construction.
  """

  def __init__(self,
               action_batch_size=None,
               also_tile_batch_in_training=False,
               create_var_scope=True,
               **kwargs):
    """Creates a Grasping44FlexibleGraspParams.

    Can handle more than a single sample of grasp param per image. To save on
    computation and memory, instead of replicating raw image, we replicate the
    image embedding after the conv layers before the grasp params are added to
    the graph.

    Args:
      action_batch_size: The number of samples of grasp params to be used with
        same state image. This is typically used for CEM sampling where we don't
        want to replicate the raw image for each param sample and a the same
        time do the calculation as one block. grasp params are expected to be
        of size [batch, action_batch_size, d] if action_batch_size is not None.
        If action_batch_size is None, then expect one sample of grasp_params
        (2D tensor).
      also_tile_batch_in_training: if action_batch_size is not None, whether to
        tile the batch in training. In prediction, it is always tiled if
        action_batch_size is not None.
      create_var_scope: If True, creates a new variable scope.
      **kwargs: Model-specific keyword arguments.
    """
    super(Grasping44FlexibleGraspParams, self).__init__(**kwargs)
    self._action_batch_size = action_batch_size
    self._also_tile_batch_in_training = also_tile_batch_in_training
    self._create_var_scope = create_var_scope
    self.activation_layers = []
    self.num_convs = [6, 6, 3]
    self.hid_layers = 2

  def model(self,
            images,
            grasp_params,
            num_classes=1,
            is_training=False,
            softmax=False,
            restore=True,
            grasp_param_names=None,
            goal_spatial_fn=None,
            goal_vector_fn=None,
            scope=None,
            reuse=None,
            **kwargs):
    """Creates a tensorflow graph for this model.

    Args:
      images: A list of 4D tensors containing image data
      grasp_params: A 3D tensor of batch_size x action_batch_size x PARAMS_SIZE
        containing grasp params or a 2D tensor of batch_size x PARAMS_SIZE if
        action_batch_size is not None.
      num_classes: Number of classes to predict in the final layer
      is_training: If the model is in training or not
      softmax: If true the final layer is a softmax, logistic otherwise
      restore: To restore logit weights or not when initializing from a
        checkpoint
      grasp_param_names: A dictionary that maps sub-blocks of `grasp_params`to
        names (string). If not None, the naming is used in graph construction.
        A key `block_name` and value (`offset`, `size`,) assign a name to
        a block `grasp_params[:, offset:(offset + size)]`.
      goal_spatial_fn: Optional function, returns a 3-D tensor to merge into the
        features, for instance conditioning the Q function on some goal feature
        map.
      goal_vector_fn: Optional function, returns a 1-D vector to merge into
        features, conditioning Q function on some goal embedding.
      scope: The top-level scope of the tensorflow graph.
      reuse: True, None, or tf.AUTO_REUSE; if True, we go into reuse mode for
        this scope as well as all sub-scopes; if tf.AUTO_REUSE, we create
        variables if they do not exist, and return them otherwise; if None, we
        inherit the parent scope's reuse flag.
      **kwargs: Model-specific arguments.
    Returns:
      graph: A tensorflow graph for the model

    Raises:
      ValueError: if restore=False as it is currently not supported
    """
    del kwargs
    if not restore:
      raise ValueError("This model doesn't yet support restore=False")
    batch_norm_var_collection = 'moving_vars'
    batch_norm = {
        # Decay for the moving averages.
        'decay': self._batch_norm_decay,
        # epsilon to prevent 0s in variance.
        'epsilon': self._batch_norm_epsilon,
        # collection containing the moving mean and moving variance.
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        },
        # Whether to scale after normalization.
        'scale': True,
    }
    end_points = {}

    tile_batch = (len(grasp_params.shape) == 3)

    if tile_batch:

      def expand_to_megabatch(feature):
        # Collapse second dimension of megabatch.
        dim = tf.shape(feature)[2]
        return tf.reshape(feature, [-1, dim])

      grasp_params = contrib_framework.nest.map_structure(
          expand_to_megabatch, grasp_params)

    # Note that we need to do this before calling the tf.variable_scope
    # since there seems to be a bug in TF that reuse=True does not work with
    # scope=None even if the default_name is passed.
    # TODO(T2R_CONTRIBUTORS): Remove this None check and pass in the class name as
    # the default_name in the tf.variable_scope initialization.
    def _run():
      """Forward pass through the network."""
      with slim.arg_scope([slim.dropout], is_training=is_training):
        with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            weights_regularizer=slim.l2_regularizer(self._l2_regularization),
            activation_fn=tf.nn.relu,
            trainable=is_training):
          with slim.arg_scope(
              [slim.conv2d, slim.max_pool2d], stride=1, padding='SAME'):
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm):
              _, grasp_image = images
              net = slim.conv2d(
                  grasp_image,
                  64, [6, 6],
                  stride=2,
                  scope='conv1_1',
                  activation_fn=None,
                  normalizer_fn=None,
                  normalizer_params=None)
              # Old checkpoints (such as those used for tests) did not have
              # scaling on the separate batch norm operations (those not
              # associated with a conv operation), so only setting the scale
              # parameter in arg_scope would break the tests. We set scale=
              # False for these separate batch norm operations temporarily.
              # However, future users are encouraged to not set scale=False so
              # that barch_norm parameters are consistent through the whole
              # network.
              net = tf.nn.relu(slim.batch_norm(net, scale=False))
              net = slim.max_pool2d(net, [3, 3], stride=3, scope='pool1')
              self.activation_layers.append(net)
              for l in range(2, 2 + self.num_convs[0]):
                net = slim.conv2d(net, 64, [5, 5], scope='conv%d' % l)
                self.activation_layers.append(net)
              net = slim.max_pool2d(net, [3, 3], stride=3, scope='pool2')
              end_points['pool2'] = net
              self.activation_layers.append(net)
              logging.debug('pool2')
              logging.debug(net.get_shape())

              if grasp_param_names is None:
                grasp_param_blocks = [grasp_params]
                grasp_param_block_names = ['fcgrasp']
              else:
                grasp_param_blocks = []
                grasp_param_block_names = []
                # Note: Creating variables must happen in a deterministic
                # order, otherwise some workers will look for variables on the
                # wrong parameter servers, so we sort the grasp_param_names
                # here.
                for block_name in sorted(grasp_param_names):
                  offset, size = grasp_param_names[block_name]
                  grasp_param_blocks += [
                      tf.slice(grasp_params, [0, offset], [-1, size])
                  ]
                  grasp_param_block_names += [block_name]

              grasp_param_tensors = []
              for block, name in zip(grasp_param_blocks,
                                     grasp_param_block_names):
                grasp_param_tensors += [
                    slim.fully_connected(
                        block,
                        256,
                        scope=name,
                        activation_fn=None,
                        normalizer_fn=None,
                        normalizer_params=None)
                ]

              fcgrasp = tf.add_n(grasp_param_tensors)

              # Old checkpoints (such as those used for tests) did not have
              # scaling on the separate batch norm operations (those not
              # associated with a conv operation), so only setting the scale
              # parameter in arg_scope would break the tests. We set scale=
              # False for these separate batch norm operations temporarily.
              # However, future users are encouraged to not set scale=False so
              # that barch_norm parameters are consistent through the whole
              # network.
              fcgrasp = tf.nn.relu(slim.batch_norm(fcgrasp, scale=False))
              fcgrasp = slim.fully_connected(fcgrasp, 64, scope='fcgrasp2')
              context = tf.reshape(fcgrasp, [-1, 1, 1, 64])
              end_points['fcgrasp'] = fcgrasp
              # Tile the image embedding action_batch_size times to align
              # with the expanded action dimension of action_batch_size.
              # Same image is used with all the actions in a action_batch.
              # net pre expansion should be [batch, *, *, *]
              # net post expansion should be [batch x action_batch, *, *, *]
              if tile_batch:
                net = contrib_seq2seq.tile_batch(net, self._action_batch_size)
              net = tf.add(net, context)
              logging.debug('net post add %s', net)
              end_points['vsum'] = net
              self.activation_layers.append(net)
              logging.debug('vsum')
              logging.debug(net.get_shape())
              for l in range(2 + sum(self.num_convs[:1]),
                             2 + sum(self.num_convs[:2])):
                net = slim.conv2d(net, 64, [3, 3], scope='conv%d' % l)
                logging.debug('conv%d', l)
                self.activation_layers.append(net)
              logging.debug(net.get_shape())
              net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')
              logging.debug('pool3')
              logging.debug(net.get_shape())
              self.activation_layers.append(net)
              for l in range(2 + sum(self.num_convs[:2]),
                             2 + sum(self.num_convs[:3])):
                net = slim.conv2d(
                    net, 64, [3, 3], scope='conv%d' % l, padding='VALID')
                self.activation_layers.append(net)
              logging.debug('final conv')
              logging.debug(net.get_shape())
              end_points['final_conv'] = net

              batch_size = tf.shape(net)[0]
              if goal_spatial_fn is not None:
                goal_spatial = goal_spatial_fn()
                # Tile goal to match net batch size (e.g. CEM).
                goal_batch_size = tf.shape(goal_spatial)[0]
                goal_spatial = tf.tile(
                    goal_spatial, [batch_size//goal_batch_size, 1, 1, 1])
                # Merging features in style of Fang 2017.
                net = tf.concat([net, goal_spatial], axis=3)
              net = slim.flatten(net, scope='flatten')

              if goal_vector_fn is not None:
                goal_vector = goal_vector_fn()
                goal_batch_size = tf.shape(goal_vector)[0]
                goal_vector = tf.tile(
                    goal_vector, [batch_size//goal_batch_size, 1])
                net = tf.concat([net, goal_vector], axis=1)

              for l in range(self.hid_layers):
                net = slim.fully_connected(net, 64, scope='fc%d' % l)

              name = 'logit'
              if num_classes > 1:
                name = 'logit_%d' % num_classes
              logits = slim.fully_connected(
                  net,
                  num_classes,
                  activation_fn=None,
                  scope=name,
                  normalizer_fn=None,
                  normalizer_params=None)
              end_points['logits'] = logits
              if softmax:
                predictions = tf.nn.softmax(logits)
              else:
                predictions = tf.nn.sigmoid(logits)
              if tile_batch:

                if num_classes > 1:
                  predictions = tf.reshape(
                      predictions, [-1, self._action_batch_size, num_classes])
                else:
                  predictions = tf.reshape(predictions,
                                           [-1, self._action_batch_size])
              end_points['predictions'] = predictions
              return logits, end_points

    if self._create_var_scope:
      if scope is None:
        scope = self.__class__.__name__
      with tf.variable_scope(scope,
                             values=[images],
                             reuse=reuse):
        with slim.arg_scope([slim.batch_norm],
                            is_training=is_training,
                            decay=batch_norm['decay'],
                            epsilon=batch_norm['epsilon'],
                            scale=batch_norm['scale']):
          logits, end_points = _run()
    else:
      with slim.arg_scope([slim.batch_norm],
                          is_training=is_training,
                          decay=batch_norm['decay'],
                          epsilon=batch_norm['epsilon'],
                          scale=batch_norm['scale'],
                          updates_collections=None):
        logits, end_points = _run()

    return logits, end_points


class Grasping44E2EOpenCloseTerminateGripperStatusHeightToBottom(
    Grasping44FlexibleGraspParams):
  """The Grasping44 model which controls gripper actions and terminate action.

  Also this model maintains current gripper status (binary open/close) and
  height from the bottom of the bin to the gripper as part of the state.
  """

  def __init__(self,
               action_batch_size=None,
               also_tile_batch_in_training=False,
               **kwargs):
    super(Grasping44E2EOpenCloseTerminateGripperStatusHeightToBottom,
          self).__init__(
              action_batch_size=action_batch_size,
              also_tile_batch_in_training=also_tile_batch_in_training,
              **kwargs)
    self.activation_layers = []
    self.num_convs = [6, 6, 3]
    self.hid_layers = 2

  @property
  def grasp_model_input_keys(self):
    return [
        'world_vector', 'vertical_rotation', 'close_gripper', 'open_gripper',
        'terminate_episode', 'gripper_closed', 'height_to_bottom'
    ]

  @classmethod
  def create_input_specifications(cls):
    """Creates the default input_specifications for preprocessing.

    The default setting uses only main(RGB) images.

    Returns:
      input_specifications: A dictionary to specify the input modalities.
    """
    grasp_param_sizes = {
        'vertical_rotation': 2,
        'world_vector': 3,
        'close_gripper': 1,
        'open_gripper': 1,
        'terminate_episode': 1,
        'gripper_closed': 1,
        'height_to_bottom': 1,
    }
    input_specifications = cls.create_default_input_specifications()
    input_specifications.update(
        end_to_end_grasping=True,
        include_initial_image=False,
        end_to_end_grasping_gripper_status=True,
        end_to_end_grasping_height_to_bottom=True,
        grasp_param_sizes=grasp_param_sizes)
    return input_specifications

  def __call__(self,
               images,
               grasp_params,
               num_classes=1,
               is_training=False,
               softmax=False,
               restore=True,
               scope=None,
               reuse=None,
               **kwargs):
    return self.model(images, grasp_params, num_classes, is_training, softmax,
                      restore, scope, reuse, **kwargs)

  def model(self,
            images,
            grasp_params,
            num_classes=1,
            is_training=False,
            softmax=False,
            restore=True,
            scope=None,
            reuse=None,
            **kwargs):
    """Creates a tensorflow graph for this model.

    Args:
      images: A list of 4D tensors containing image data
      grasp_params: A 2D tensor of batch_size x PARAMS_SIZE containing grasp
                    params
      num_classes: Number of classes to predict in the final layer
      is_training: If the model is in training or not
      softmax: If true the final layer is a softmax, logistic otherwise
      restore: To restore logit weights or not when initializing from a
               checkpoint
      scope: The top-level scope of the tensorflow graph
      reuse: True, None, or tf.AUTO_REUSE; if True, we go into reuse mode for
        this scope as well as all sub-scopes; if tf.AUTO_REUSE, we create
        variables if they do not exist, and return them otherwise; if None, we
        inherit the parent scope's reuse flag.

      **kwargs: Model-specific arguments.
    Returns:
      graph: A tensorflow graph for the model

    Raises:
      ValueError: if restore=False as it is currently not supported
    """
    return super(Grasping44E2EOpenCloseTerminateGripperStatusHeightToBottom,
                 self).model(
                     images,
                     grasp_params,
                     num_classes=num_classes,
                     is_training=is_training,
                     softmax=softmax,
                     restore=restore,
                     scope=scope,
                     reuse=reuse,
                     grasp_param_names={
                         'fcgrasp_wv': (0, 3),
                         'fcgrasp_vr': (3, 2),
                         'fcgrasp_gripper_close': (5, 1),
                         'fcgrasp_gripper_open': (6, 1),
                         'fcgrasp_terminate_episode': (7, 1),
                         'fcgrasp_gripper_closed': (8, 1),
                         'fcgrasp_height_to_bottom': (9, 1)
                     },
                     **kwargs)

