# Tensor2Robot

This repository contains distributed machine learning and reinforcement learning
infrastructure.

It is used internally at Alphabet, and open-sourced with the intention of making
research at Robotics @ Google more reproducible for the broader robotics and
computer vision communities.


## Projects and Publications Using Tensor2Robot

-   [QT-Opt](research/qtopt)
-   [Grasp2Vec](research/grasp2vec)
-   [Watch, Try, Learn](research/vrgripper)
-   [BC-Z](research/bcz)

## Features

Tensor2Robot (T2R) is a library for training, evaluation, and inference of
large-scale deep neural networks, tailored specifically for neural networks
relating to robotic perception and control. It is based on the
[TensorFlow](tensorflow.org) deep learning framework.

A common task in robotics research involves adding a new sensor modality or new
label tensor to a neural network graph. This involves 1) changing what data is
saved, 2) changing data pipeline code to read in new modalities at training
time 3) adding a new `tf.placeholder` to handle the new input modality at test
time. The main feature of Tensor2Robot is the automatic generation of TensorFlow
code for steps 2 and 3. Tensor2Robot can automatically generate placeholders for
a model to match its inputs, or alternatively exports a `SavedModel` that can be
used with a `TFExportedSavedModelPolicy` so that the original graph does not
have to be re-constructed.

Another common task encountered in ML involves cropping / transforming input
modalities, such as jpeg-decoding and applying random image distortions at
training time. The `Preprocessor` class declares its own input features and
labels, and is expected to output shapes compatible with the input features and
labels of the model. Example preprocessors can be found in
[preprocessors](preprocessors/).

## Design Decisions

- Scalability: This codebase is designed for training large-scale, real-world
robotic perception models with algorithms that do not require a tight
perception-action-learning loop (supervised learning, off-policy reinforcement
learning of large computer vision models). An example setup might involve
multiple GPUs pulling data asynchronously from a replay buffer and training
with an off-policy RL algorithm, while data collection agents periodically
update their checkpoints and push experiences to the same replay buffer.
This can also be run on a single workstation for smaller-scale experiments.

- T2R is *NOT* a general-purpose reinforcement learning library. Due to the
sizes of models we work with (e.g. grasping from vision) Inference is assumed
to be within 1-10Hz and without real-time guarantees, and training is assumed
to be distributed by default. If you are doing reinforcement learning with small
networks (e.g. two-layer perceptron) with on-policy RL (e.g. PPO), or require
hard real-time guarantees, this is probably not the right codebase to use. We
recommend using
[TF-Agents](https://github.com/tensorflow/agents) or
[Dopamine](https://github.com/google/dopamine) for those use cases.

- Minimize boilerplate: Tensor2Robot Models auto-generate their own data input
pipelines and provide sensible defaults for optimizers, common architectures
(actors, critics), and train/eval scaffolding. Models automatically work with
both GPUs and TPUs (via `TPUEstimator`), parsing bmp/gif/jpeg/png-encoded
images.

- gin-configurable: [Gin-Config](https://github.com/google/gin-config) is used
to configure models, policies, and other experiment hyperparameters.

## Quickstart

Requirements: Python 3.

```
git clone https://github.com/google/tensor2robot
# Optional: Create a virtualenv
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -r tensor2robot/requirements.txt
python -m tensor2robot.research.pose_env.pose_env_test

# Install protoc and compile the protobufs.
pip install protobuf
cd tensor2robot/proto
protoc -I=./ --python_out=`pwd` tensor2robot/t2r.proto
python -m tensor2robot.research.pose_env.pose_env_models_test
```

## T2RModel

To use Tensor2Robot, a user defines a `T2RModel` object that define their input
requirements by specifications - one for their features (feature_spec) and one
for their labels (label_spec):

These specifications define all required and optional tensors in order to call
the model_fn. An input pipeline parameterized with the model's input pipeline
will ensure that all required specifications are fulfilled. **Note**: we always
omit the batch dimension and only specify the shape of a single element.

At training time, the T2RModel provides `model_train_fn` or `model_eval_fn` as
the `model_fn` argument `tf.estimator.Estimator` class. Both `model_train_fn`
and `model_eval_fn` are defined with respect to the features, labels, and
outputs of `inference_network_fn`, which presumably implements the shared
portions of the train/eval graphs.


```bash
class MyModel(T2RModel):
  def get_feature_specification(self, mode):
    spec = tensorspec_utils.TensorSpecStruct()
    spec['state'] = ExtendedTensorSpec(
      shape=(8,128), dtype=tf.float32, name='s')
  def get_label_specification(self, mode):
    spec = tensorspec_utils.TensorSpecStruct()
    spec['action'] = ExtendedTensorSpec(shape=(8), dtype=tf.float32, name='a')
  def inference_network_fn(self,
                           features: tensorspec_utils.TensorSpecStruct,
                           labels: Optional[tensorspec_utils.TensorSpecStruct],
                           mode: tf.estimator.ModeKeys,
                           config: RunConfigType = None,
                           params: ParamsType = None) -> DictOrSpec:
    inference_outputs = {}
    inference_outputs['predictions'] = layers.fully_connected(features.state, 8)
    return inference_outputs
  def model_train_fn(self,
                     features: tensorspec_utils.TensorSpecStruct,
                     labels: tensorspec_utils.TensorSpecStruct,
                     inference_outputs: DictOrSpec,
                     mode: tf.estimator.ModeKeys,
                     config: RunConfigType = None,
                     params: ParamsType = None) -> ModelTrainOutputType:
    """See base class."""
    del features, config
    loss = tf.losses.mean_squared_error(
      labels.action, inference_outputs['predictions'])
    return loss
```


Note how the **key** on the **left hand side** has a value for **name** that is
different from the one on the **right hand side** within the ExtendedTensorSpec.
The key on the left is used within the model_fn to access the loaded tensor
whereas the name is used when creating the `parse_tf_example_fn` or
numpy_feed_dict. We ensure that the name is unique within the whole spec, unless
the specs match, otherwise we cannot guarantee the mapping functionality.


### Benefits of Inheriting a T2RModel

- Self-contained input specifications for features and labels.
- Auto-generated `tf.data.Dataset` pipelines for
`tf.train.Examples` and `tf.train.SequenceExamples`.
- For policy inference, T2RModels can generate placeholders or export
`SavedModel`s that are hermetic and can be used with `ExportSavedModelPolicy`.
- Automatic construction of `model_fn` for Estimator for training and evaluation
graphs that share a single `inference_network_fn`.
- It is possible to compose multiple models' `inference_network_fn` and
`model_train_fn` together under a single model. This abstraction allows us to
implement generic Meta-Learning models (e.g. MAML) that call their sub-model's
`model_train_fn`.
- Automatic support for distributed training on GPUs and TPUs.


### Policies and Placeholders

For performance reasons, policy inference is done by a vanilla `session.run()`
or a `predict_fn` call on the output of a model, instead of Estimator.predict.
`tensorspec_utils.make_placeholders` automatically creates placeholders from a
spec structure which can be used in combination with a matching hierarchy of
numpy  inputs to create a feed_dict.

```bash
# batch_size = -1 -> No batch size will be prepended to the spec.
# batch_size = None -> We will prepend None, have a variable batch_size.
# batch_size > 0 -> We will have a fixed batch_size.
placeholders = tensorspec_utils.make_placeholders(hierarchical_spec, batch_size=None)

feed_dict = inference_model.MakeFeedDict(placeholders, numpy_inputs)
# This can be passed to a sess.run function to evaluate the model.
```

If you use `TFExportedSavedModelPolicy`, note that your T2RModel should not
query the static batch shape (`x.shape[0]`) in the graph. This is because
placeholder generation creates inputs with unknown batch shape `None`, causing
static shape retrieval to fail. Instead, use `tf.shape(x)[0]` to access batch
shapes dynamically.

## Working with Tensor Specifications

**Specifications** can be a hierarchical data structure of either

*   dictionaries (dict),
*   tuples (tuple),
*   lists (list), or
*   TensorSpecStruct (preferred).

The leaf elements have to be of type TensorSpec or ExtendedTensorSpec
(preferred). In the following, we will present some examples using
ExtendedTensorSec and TensorSpecStruct to illustrate the different
usecases.

We use tensorspec_utils.TensorSpecStruct() for specifying specs
since this data structure is mutuable, provides attribute (dot) access and item
iteration. Further, we can use pytype to ensure compile time type checking. This
data structure is both hierarchical and flat: The dictionary interface using
.items() is flat representing hierarchical data with paths, as shown later on.
However, we maintain a hierarchical interface using attribute access, also shown
later on. Therefore, we can use this data structure in order to create and alter
hierarchical specifications which work with both TPUEstimator and Estimator
since both apis operate on the dictionary view.


### Hierarchical example

Creating a hierarchical spec from spec using tensorspec_utils.copy_tensorspec.

```bash
simple_spec = tensorspec_utils.TensorSpecStruct()
simple_spec['state'] = ExtendedTensorSpec(
  shape=(8,128), dtype=tf.float32, name='s')
simple_spec['action'] = ExtendedTensorSpec(shape=(8), dtype=tf.float32, name='a')

hierarchical_spec = tensorspec_utils.TensorSpecStruct()
hierarchical_spec.train = tensorspec_utils.copy_tensorspec(simple_spec, prefix=’train’)
```

Note, we use attribute access to define the train hierarchy. This will copy all
our specs from simple_spec internally to

```bash
# 'train/{}' -> 'train/state' == simple_spec.state and
# 'train/action' == simple_spec.action
```

We forbid the following pattern:

```bash
hierarchical_spec.train = tensorspec_utils.TensorSpecStruct()
```

and encourage the user to use this pattern instead.

```bash
train = tensorspec_utils.TensorSpecStruct()
train.stuff = ExtendedTensorSpec(...)
hierarchical_spec.train = train
# or
hierarchical_spec['train/stuff'] = ExtendedTensorSpec(...)
```

```bash
# All of the following statements are True.
hierarchical_spec.train.state == simple_spec.state
hierarchical_spec.train.action == simple_spec.action
hierarchical_spec.keys() == ['train/state', 'train/action']
hierarchical_spec.train.keys() == ['state', 'action']
```

Now we want to use the same spec another time for our input.

```bash
hierarchical_spec.val = tensorspec_utils.copy_tensorspec( simple_spec,
  prefix='val')

# All of the following statements are True.
hierarchical_spec.keys() == ['train/state', 'train/action', 'val/state',
   'val/action']
hierarchical_spec.train.keys() == ['state', 'action']
hierarchical_spec.train.state.name == 'train/s'
hierarchical_spec.val.keys() == ['state', 'action']
hierarchical_spec.val.state.name == 'val/s'
```

Manually extending/creating a hierarchical spec from an existing simple spec is
also possible. TensorSpec is an immutable data structure therefore the recommend
way to alter a spec is:

```bash
hierarchical.train.state = ExtendedTensorSpec.from_spec(
  hierarchical.train.state, ...PARAMETERS TO OVERWRITE)
```

A different way of changing a hierarchical spec would be:

```bash
for key, value in simple_spec.items():
  hierarchical[‘test/’ + key] = ExtendedTensorSpec.from_spec(
    value, name=’something_random/’+value.name)
# hierarchical_spec.keys() == ['train/state', 'train/action', ‘val/state’,
#   ‘val/action’, ‘test/state’, ‘test/action’]
# hierarchical_spec.train.keys() == ['state', 'action']
# hierarchical_spec.val.keys() == ['state', 'action']
# hierarchical_spec.test.keys() == ['state', 'action']
# hierarchical_spec.test.state.name == ‘something_random/s’
```

### Sequential Inputs

Tensor2Robot can parse both `tf.train.Example` and `tf.train.SequenceExample`
protos (useful for training recurrent models like LSTMs). To declare a model
whose data is parsed from SequenceExamples, set `is_sequence=True`.

```bash
spec['state'] = ExtendedTensorSpec(
  shape=(8,128), dtype=tf.float32, name='s', is_sequence=True)
```

This will result in a parsed tensor of shape `(b, ?, 8, 128)` where b is the
batch size and the second dimension is the unknown sequence length (only known
at run-time). Note that if `is_sequence=True` for any ExtendedTensorSpec in the
TensorSpecStruct, the proto will be assumed to be a SequenceExample (and
non-sequential Tensors will be assumed to reside in example.context).


### Flattening hierarchical specification structures

Any valid spec structure can be flattened into a
`tensorspec_utils.TensorSpecStruct`. In the following we show
different hierarchical data structures and the effect of
`flatten_spec_structure`.

```bash
flat_hierarchy = tensorspec_utils.flatten_spec_structure(hierarchical)
```

Note, `tensorspec_utils.TensorSpecStruct` will have flat
dictionary access. We can print/access/change all elements of our spec by
iterating over the items.

```bash
for key, value in flat_hierarchy.items():
  print('path: {}, spec: {}'.format(key, value))
# This will print:
# path: train/state, spec: ExtendedTensorSpec(
#    shape=(8, 128), dtype=tf.float32, name='train/s')
# path: train/action, spec: ExtendedTensorSpec(
#    shape=(8), dtype=tf.float32, name='train/a')
# path: val/state, spec: ExtendedTensorSpec(
#    shape=(8, 128), dtype=tf.float32, name='val/s')
# path: val/action, spec: ExtendedTensorSpec(
#    shape=(8), dtype=tf.float32, name='val/a')
# path: test/state, spec: ExtendedTensorSpec(
#    shape=(8, 128), dtype=tf.float32, name='something_random/s')
# path: test/action, spec: ExtendedTensorSpec(
#    shape=(8), dtype=tf.float32, name='something_random/a')
# This data structure still maintains the hierarchical user interface.
train = flat_hierarchy.train
for key, value in flat_hierarchy.items():
  print('path: {}, spec: {}'.format(key, value))
# This will print:
# path: state, spec: ExtendedTensorSpec(
#    shape=(8, 128), dtype=tf.float32, name='train/s')
# path: action, spec: ExtendedTensorSpec(
#    shape=(8), dtype=tf.float32, name='train/a')
```

**Note, the path has changed, but the name is still from the hierarchy.** This
is an important distinction. The model could access the data in a different
manner but the same "name" is used to access `tf.Examples` of feed_dicts in
order to feed the tensors.

An alternative hierarchical spec using namedtuples:

```bash
Hierarchy = namedtuple('Hierarchy', ['train', 'val'])
Sample = namedtuple('Sample', ['state', 'action'])
hierarchy = Hierarchy(
  train=Sample(
    state=ExtendedTensorSpec(shape=(8, 128), dtype=tf.float32, name='train/s'),
    action=ExtendedTensorSpec(shape=(8), dtype=tf.float32, name='train/a'),
  ),
  eval=Sample(
    state=ExtendedTensorSpec(shape=(8, 128), dtype=tf.float32, name='val/s'),
    action=ExtendedTensorSpec(shape=(8), dtype=tf.float32, name='val/a'),
  )
)
flat_hierarchy = tensorspec_utils.flatten_spec_structure(hierarchy)

for key, value in flat_hierarchy.items():
  print('path: {}, spec: {}'.format(key, value))
# This will print:
# path: train/state, spec: ExtendedTensorSpec(
#    shape=(8, 128), dtype=tf.float32, name='train/s')
# path: train/action, spec: ExtendedTensorSpec(
#    shape=(8), dtype=tf.float32, name='train/a')
# path: val/state, spec: ExtendedTensorSpec(
#    shape=(8, 128), dtype=tf.float32, name='val/s')
# path: val/action, spec: ExtendedTensorSpec(
#    shape=(8), dtype=tf.float32, name='val/a')
```

**Note, hierarchy (namedtuple) is immutable whereas flat_hierarchy is a mutable
instance of TensorSpecStruct.**

### Validate and flatten or pack

`tensorspec_utils.validate_and_flatten` and `tensorspec_utils.validate_and_pack`
allow to verify that an existing, e.g. loaded spec data structure filled with
tensors fulfills our expected spec structure and is flattened or packed into a
hierarchical structure.


## Disclaimer

This is not an official Google product. External support not guaranteed. The API
may change subject to Alphabet needs. File a GitHub issue if you have questions.
