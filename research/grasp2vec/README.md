# Grasp2Vec

Source codes for reproducing "Grasp2Vec: Learning Object Representations from
Self-Supervised Grasping".

Links

-   [Project Website](https://sites.google.com/site/grasp2vec/)
-   [Paper](https://arxiv.org/abs/1811.06964)
-   [Google AI Blog Post](https://ai.googleblog.com/2018/12/grasp2vec-learning-object.html)

## Authors

Eric Jang<sup>*1</sup>, Coline Devin<sup>*2</sup>, Vincent
Vanhoucke<sup>1</sup>, Sergey Levine<sup>12</sup>

<sup>*</sup>Equal Contribution, <sup>1</sup> Google Brain, <sup>2</sup>UC
Berkeley

## Training the Model

Data is not included in this repository, so you will have to provide your own
training/eval datasets of TFRecords. The Grasp2Vec T2R model attempts to parse
the following Feature spec from the data, before cropping and resizing the
parsed images:

```
tspec.pregrasp_image = TensorSpec(shape=(512, 640, 3),
    dtype=tf.uint8, name='image', data_format='jpeg')
tspec.postgrasp_image = TensorSpec(
    shape=(512, 640, 3), dtype=tf.uint8, name='postgrasp_image',
    data_format='jpeg')
tspec.goal_image = TensorSpec(
    shape=(512, 640, 3), dtype=tf.uint8, name='present_image',
    data_format='jpeg')
```

Note that `image`, `postgrasp_image`, `present_image` are the names of features
stored in the TFExample feature map.

```
python3 -m tensor2robot.bin.run_t2r_trainer --logtostderr \
  --gin_configs="tensor2robot/research/grasp2vec/configs/train_grasp2vec.gin" \
  --gin_bindings="train_eval_model.model_dir='/tmp/grasp2vec/'" \
  --gin_bindings="TRAIN_DATA='/path/to/your/data/train*' \
  --gin_bindings="EVAL_DATA='/path/to/your/data/val*'"
```

Tensorboard will show heatmap localization visualization summaries as shown in
the paper.
